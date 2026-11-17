import os
import sys

DEFAULT_DEVICE = 1  # physical GPU used when --device is not given


def _pin_single_gpu():
    """Expose ONLY the requested physical GPU to this process.

    Must run before CUDA initializes. CUDA's "current device" is per-thread, so
    set_device() in the main thread would not stop background threads (e.g. the
    pin_memory() calls in the prefetcher) from creating a context on GPU 0;
    hiding the other GPUs does. Returns True if we set the variable ourselves.
    If CUDA_VISIBLE_DEVICES is already set, we respect it and --device indexes into it.
    """
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        return False
    dev = DEFAULT_DEVICE
    for i, a in enumerate(sys.argv):
        if a == "--device" and i + 1 < len(sys.argv):
            dev = int(sys.argv[i + 1])
        elif a.startswith("--device="):
            dev = int(a.split("=", 1)[1])
    os.environ["CUDA_VISIBLE_DEVICES"] = str(dev)
    return True


PINNED_GPU = _pin_single_gpu()

import argparse
import hashlib
import itertools
import json
import logging
import math
import random
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO

import requests
import torch
from requests.adapters import HTTPAdapter
from datasets import load_dataset
from PIL import Image
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoTokenizer,
    CLIPVisionConfig,
    CLIPVisionModel,
    GPTNeoXForCausalLM,
)

from visual_gpt_neox import VisualGPTNeoXConfig, VisualGPTNeoXForCausalLM

logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)
log = logging.getLogger("train")

session = requests.Session()
IMAGE_CACHE = {"dir": None}  # optional on-disk cache of raw image bytes (set in train())
_END = object()


def fetch_image(url, timeout=10):
    try:
        path, data = None, None
        if IMAGE_CACHE["dir"]:
            path = os.path.join(IMAGE_CACHE["dir"], hashlib.sha1(url.encode()).hexdigest())
            if os.path.exists(path):
                with open(path, "rb") as f:
                    data = f.read()
                path = None  # already cached
        if data is None:
            r = session.get(url, timeout=timeout)
            r.raise_for_status()
            data = r.content

        with Image.open(BytesIO(data)) as im:
            if im.mode in ("P", "PA", "LA", "RGBA") or "transparency" in im.info:
                im = im.convert("RGBA")
                bg = Image.new("RGB", im.size, (255, 255, 255))
                bg.paste(im, mask=im.getchannel("A"))
                img = bg
            else:
                img = im.convert("RGB")

        if path:  # cache only bytes that decoded successfully; atomic rename
            tmp = f"{path}.{os.getpid()}.{threading.get_ident()}.tmp"
            with open(tmp, "wb") as f:
                f.write(data)
            os.replace(tmp, path)
        return img
    except Exception:
        return None


def load_batch(batch, pool):
    images = list(pool.map(fetch_image, batch["image_url"]))
    keep = [i for i, im in enumerate(images) if im is not None]
    return [images[i] for i in keep], [batch["caption"][i] for i in keep]


def prefetch(idxs, fn, executor, depth):
    """Yield fn(idx) for idx in idxs, in order, with `depth` calls running ahead in background threads."""
    it = iter(idxs)
    futs = deque(executor.submit(fn, i) for i in itertools.islice(it, depth))
    while futs:
        fut = futs.popleft()
        nxt = next(it, _END)
        if nxt is not _END:
            futs.append(executor.submit(fn, nxt))
        yield fut.result()


def set_trainable(model, train_vision, train_lm):
    """Projector is always trained; vision tower / LM are optional."""
    for n, p in model.named_parameters():
        if n.startswith("visual_gpt_neox.vision_encoder."):
            p.requires_grad = train_vision
        elif n.startswith("visual_gpt_neox.mlp."):
            p.requires_grad = True
        else:
            p.requires_grad = train_lm


def group_of(name):
    if name.startswith("visual_gpt_neox.vision_encoder."):
        return "vision"
    if name.startswith("visual_gpt_neox.mlp."):
        return "mlp"
    return "lm"


def is_ckpt_step(step, every):
    """Pythia-style: step 0, powers of two up to 512, then every `every` steps."""
    return step % every == 0 or (0 < step <= 512 and step & (step - 1) == 0)


def train(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.accum < 1:
        raise ValueError("--accum must be >= 1")
    if PINNED_GPU:
        # only the requested GPU is visible, and it is cuda:0 inside this process
        assert os.environ["CUDA_VISIBLE_DEVICES"] == str(args.device), "--device parsing mismatch"
        device = torch.device("cuda:0")
    else:
        # CUDA_VISIBLE_DEVICES was set externally; --device indexes into the visible set
        device = torch.device(f"cuda:{args.device}")
    if device.index >= torch.cuda.device_count():
        raise RuntimeError(
            f"--device {args.device} not available: {torch.cuda.device_count()} visible GPU(s) "
            f"(CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')})"
        )
    torch.cuda.set_device(device)
    if args.tf32:  # faster fp32 matmuls (10-bit mantissa); only matters with --no-bf16
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if args.bf16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError("--bf16 requested but this GPU does not support bfloat16")
    log.info(
        f"using {torch.cuda.get_device_name(device)} (physical GPU {os.environ.get('CUDA_VISIBLE_DEVICES')}); "
        f"effective batch = {args.bs} x {args.accum} = {args.bs * args.accum}; bf16={args.bf16}"
    )

    # default requests pool holds 10 connections; match it to the download thread count
    adapter = HTTPAdapter(pool_connections=args.workers, pool_maxsize=args.workers)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    if args.image_cache:
        os.makedirs(args.image_cache, exist_ok=True)
        IMAGE_CACHE["dir"] = args.image_cache

    exp_dir, log_file = None, None
    if args.resume:
        if args.debug:
            raise ValueError("--resume cannot be combined with --debug")
        exp_dir = args.resume
        if not os.path.exists(os.path.join(exp_dir, "train_state.pt")):
            raise FileNotFoundError(f"no train_state.pt in {exp_dir}")
    elif not args.debug:
        os.makedirs(args.ckpts, exist_ok=True)
        exps = [int(x) for x in os.listdir(args.ckpts) if x.isdigit()]
        exp_dir = os.path.join(args.ckpts, str(max(exps, default=0) + 1))
        os.makedirs(exp_dir)

    tok = AutoTokenizer.from_pretrained(args.lm)
    tok.pad_token = tok.eos_token  # Pythia has no pad token
    tok.padding_side = "right"

    # ---- build components (fresh construction == native random init) ----
    lm_cfg = AutoConfig.from_pretrained(args.lm)
    clip_hidden, clip_inter = args.clip_hidden_size, args.clip_intermediate_size
    if args.no_mlp and args.reset_vision and clip_hidden != lm_cfg.hidden_size:
        # no projector: the vision tower's output dim must equal the LM hidden size
        clip_inter = clip_inter * lm_cfg.hidden_size // clip_hidden  # keep the MLP expansion ratio
        clip_hidden = lm_cfg.hidden_size
        log.info(f"--no_mlp: vision hidden size set to LM hidden size ({clip_hidden}), intermediate {clip_inter}")
    if args.no_mlp and args.reset_vision and clip_hidden % args.clip_num_attention_heads:
        raise ValueError(
            f"vision hidden size {clip_hidden} is not divisible by --clip_num_attention_heads "
            f"{args.clip_num_attention_heads}; pick a head count that divides it"
        )
    if args.reset_vision:
        vision_cfg = CLIPVisionConfig(  # architecture only, no pretrained weights
            hidden_size=clip_hidden,
            intermediate_size=clip_inter,
            num_hidden_layers=args.clip_num_hidden_layers,
            num_attention_heads=args.clip_num_attention_heads,
            image_size=args.clip_image_size,
            patch_size=args.clip_patch_size,
        )
        vision = CLIPVisionModel(vision_cfg)
    else:
        # architecture must match the checkpoint, so --clip_* args are ignored here
        vision = CLIPVisionModel.from_pretrained(args.vis)
        vision_cfg = vision.config

    if args.reset_lm:
        lm = GPTNeoXForCausalLM(lm_cfg)
    else:
        lm = GPTNeoXForCausalLM.from_pretrained(args.lm)

    if args.no_mlp and vision_cfg.hidden_size != lm_cfg.hidden_size:
        raise ValueError(
            f"--no_mlp needs vision hidden size ({vision_cfg.hidden_size}) == LM hidden size "
            f"({lm_cfg.hidden_size}); use --reset_vision (it adapts automatically) or drop --no_mlp"
        )

    S, P = vision_cfg.image_size, vision_cfg.patch_size
    tokens_per_image = (S // P) ** 2 + 1  # patches + CLS (we use last_hidden_state as-is)
    proc = AutoImageProcessor.from_pretrained(
        args.vis, size={"shortest_edge": S}, crop_size={"height": S, "width": S}
    )

    cfg = VisualGPTNeoXConfig.from_llm_and_vision(
        lm.config, vision_cfg, use_projector=not args.no_mlp
    )
    model = VisualGPTNeoXForCausalLM(cfg)
    model.load_vision_encoder(vision)  # after construction: post_init re-inits everything
    model.load_language_model(lm)
    del vision, lm

    set_trainable(
        model,
        train_vision=args.train_vision or args.reset_vision,
        train_lm=args.train_lm or args.reset_lm,
    )
    # Per-module LRs: freshly initialized modules (incl. the projector) use --lr,
    # pretrained ones use --pretrained_lr; --vision_lr / --lm_lr override either.
    lrs = {
        "mlp": args.lr,
        "vision": args.vision_lr
        if args.vision_lr is not None
        else (args.lr if args.reset_vision else args.pretrained_lr),
        "lm": args.lm_lr
        if args.lm_lr is not None
        else (args.lr if args.reset_lm else args.pretrained_lr),
    }
    buckets = {"vision": [], "mlp": [], "lm": []}
    for n, p in model.named_parameters():
        if p.requires_grad:
            buckets[group_of(n)].append(p)
    param_groups = [
        {"params": ps, "lr": lrs[k], "name": k} for k, ps in buckets.items() if ps
    ]
    params = [p for g in param_groups for p in g["params"]]
    for g in param_groups:
        log.info(f"{g['name']}: {sum(p.numel() for p in g['params']):,} params, lr={g['lr']}")

    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(
        param_groups, weight_decay=args.wd, betas=(args.beta1, args.beta2), fused=True
    )

    pixmo_full = load_dataset("allenai/pixmo-cap")["train"]
    # The last --eval_size examples are held out (never trained on) for the image-use eval.
    n_train = len(pixmo_full) - args.eval_size
    if n_train <= 0:
        raise ValueError("--eval_size leaves no training data")

    # ---- LR schedule: linear warmup, then cosine decay to min_lr_ratio (or constant) ----
    steps_per_epoch = max(1, ((n_train + args.bs - 1) // args.bs) // args.accum)  # optimizer steps
    n_epochs = args.max_epochs
    planned_steps = n_epochs * steps_per_epoch
    if args.max_steps:  # overrides --max_epochs
        n_epochs = math.ceil(args.max_steps / steps_per_epoch)
        planned_steps = args.max_steps

    def lr_mult(step):
        if args.warmup_steps > 0 and step < args.warmup_steps:
            return (step + 1) / args.warmup_steps
        if args.schedule == "constant":
            return 1.0
        progress = min(1.0, (step - args.warmup_steps) / max(1, planned_steps - args.warmup_steps))
        return args.min_lr_ratio + (1 - args.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_mult)
    log.info(
        f"{n_train:,} train examples, {steps_per_epoch:,} optimizer steps/epoch, "
        f"{planned_steps:,} planned steps, schedule={args.schedule} warmup={args.warmup_steps}"
    )

    hparams = dict(vars(args))
    hparams["vision_param_count"] = sum(
        p.numel() for p in model.visual_gpt_neox.vision_encoder.parameters()
    )
    hparams.update(n_train=n_train, steps_per_epoch=steps_per_epoch, planned_steps=planned_steps)
    eval_file = None
    if exp_dir:
        mode = "a" if args.resume else "w"
        if not args.resume:
            with open(os.path.join(exp_dir, "hparams.json"), "w") as f:
                json.dump(hparams, f, indent=2)
        log_file = open(os.path.join(exp_dir, "log.tsv"), mode)
        eval_file = open(os.path.join(exp_dir, "eval.tsv"), mode)
        if not args.resume:
            log_file.write("step\tloss\tvision_tokens\ttext_tokens\tdata_s\tstep_s\tlr_mult\tgrad_norm\n")
            eval_file.write(
                "step\tloss_correct\tloss_shuffled\tloss_blank\tgap_shuffled_minus_correct"
                "\tloss_text_only\tgain_vs_text_only\n"
            )

    total_steps = 0
    state = {"saved": None, "evaled": None, "epoch": 0, "steps_in_epoch": 0}
    resume_epoch, resume_skip = 0, 0
    if args.resume:
        ts = torch.load(os.path.join(exp_dir, "train_state.pt"), map_location="cpu", weights_only=False)
        res = model.load_state_dict(ts["trainable"], strict=False)  # frozen weights come from the pretrained init
        assert not res.unexpected_keys, res.unexpected_keys
        optimizer.load_state_dict(ts["optimizer"])
        scheduler.load_state_dict(ts["scheduler"])
        total_steps, resume_epoch, resume_skip = ts["total_steps"], ts["epoch"], ts["steps_in_epoch"]
        state.update(saved=total_steps, evaled=total_steps, epoch=resume_epoch, steps_in_epoch=resume_skip)
        log.info(f"resumed from {exp_dir}: step {total_steps}, epoch {resume_epoch}, +{resume_skip} steps in epoch")

    def save_train_state():
        """Everything needed to resume: trainable weights, optimizer, scheduler, position in the data."""
        tmp = os.path.join(exp_dir, "train_state.pt.tmp")
        torch.save(
            {
                "trainable": {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad},
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "total_steps": total_steps,
                "epoch": state["epoch"],
                "steps_in_epoch": state["steps_in_epoch"],
            },
            tmp,
        )
        os.replace(tmp, os.path.join(exp_dir, "train_state.pt"))  # atomic: never a half-written file

    def maybe_save(force=False):
        if exp_dir and state["saved"] != total_steps and (force or is_ckpt_step(total_steps, args.ckpt_every)):
            model.save_pretrained(os.path.join(exp_dir, f"step_{total_steps}"))
            save_train_state()
            state["saved"] = total_steps

    try:
        maybe_save()  # step 0
        with ThreadPoolExecutor(max_workers=args.workers) as pool, ThreadPoolExecutor(
            max_workers=args.prefetch_workers
        ) as prep_pool:

            def prepare(idx):
                """Runs in a background thread: download + decode + image preprocessing."""
                images, captions = load_batch(pixmo_full[idx : min(idx + args.bs, n_train)], pool)
                if not images:
                    return None
                pv = proc(images=images, return_tensors="pt")["pixel_values"]
                return pv.pin_memory(), captions  # pinned -> async host-to-device copy

            # ---- held-out eval: does the model actually use the image? ----
            eval_pv, eval_caps = None, []
            if args.eval_size > 0 and args.eval_every > 0:
                log.info(f"building held-out eval set ({args.eval_size} examples)...")
                pvs = []
                for i in range(n_train, len(pixmo_full), args.bs):
                    images, caps = load_batch(pixmo_full[i : min(i + args.bs, len(pixmo_full))], pool)
                    if images:
                        pvs.append(proc(images=images, return_tensors="pt")["pixel_values"])
                        eval_caps += caps
                if len(eval_caps) >= 2:
                    eval_pv = torch.cat(pvs)
                    log.info(f"eval set: {len(eval_caps)} usable examples")
                else:
                    log.warning("could not download enough eval images; eval disabled")

            @torch.no_grad()
            def evaluate():
                """Per-token loss with correct / shuffled (another caption's image) / blank images,
                plus a text-only baseline: the same LM with no image tokens (EOS as the first token)."""
                model.eval()
                shuffled = eval_pv.roll(1, dims=0)
                tot = {k: 0.0 for k in ("correct", "shuffled", "blank", "text_only")}
                n_tok = 0
                for i in range(0, len(eval_caps), args.bs):
                    sl = slice(i, i + args.bs)
                    enc = tok(
                        [c + tok.eos_token for c in eval_caps[sl]],
                        return_tensors="pt", padding=True, truncation=True,
                        max_length=args.max_text_len, pad_to_multiple_of=8,
                    ).to(device)
                    labels = enc["input_ids"].masked_fill(enc["attention_mask"] == 0, -100)
                    n = int((labels != -100).sum())
                    n_tok += n
                    for name, pv in (
                        ("correct", eval_pv[sl]),
                        ("shuffled", shuffled[sl]),
                        ("blank", torch.zeros_like(eval_pv[sl])),
                    ):
                        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.bf16):
                            out = model(
                                input_ids=enc["input_ids"],
                                attention_mask=enc["attention_mask"],
                                labels=labels,
                                vision_inputs={"pixel_values": pv.to(device)},
                                use_cache=False,
                            )
                        tot[name] = tot[name] + out.loss.detach().float() * n  # token-weighted
                    # Text-only baseline: [EOS] + caption, loss on the caption tokens only, so it
                    # predicts exactly the same tokens as the image conditions above.
                    b = labels.shape[0]
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.bf16):
                        out = model(
                            input_ids=torch.cat(
                                [enc["input_ids"].new_full((b, 1), tok.eos_token_id), enc["input_ids"]], dim=1
                            ),
                            attention_mask=torch.cat(
                                [enc["attention_mask"].new_ones(b, 1), enc["attention_mask"]], dim=1
                            ),
                            labels=torch.cat([labels.new_full((b, 1), -100), labels], dim=1),
                            use_cache=False,
                        )
                    tot["text_only"] = tot["text_only"] + out.loss.detach().float() * n
                model.train()
                return {k: float(v) / n_tok for k, v in tot.items()}

            def run_eval():
                r = evaluate()
                gap = r["shuffled"] - r["correct"]
                gain = r["text_only"] - r["correct"]  # > 0: the image prefix beats the text-only LM
                log.info(
                    f"EVAL step {total_steps}: correct {r['correct']:.4f} shuffled {r['shuffled']:.4f} "
                    f"blank {r['blank']:.4f} text_only {r['text_only']:.4f} | "
                    f"gap(shuffled-correct) {gap:+.4f} gain_vs_text_only {gain:+.4f}"
                )
                if eval_file:
                    eval_file.write(
                        f"{total_steps}\t{r['correct']}\t{r['shuffled']}\t{r['blank']}\t{gap}"
                        f"\t{r['text_only']}\t{gain}\n"
                    )
                    eval_file.flush()
                state["evaled"] = total_steps

            if eval_pv is not None and state["evaled"] != total_steps:
                run_eval()  # step-0 baseline

            done = False
            for epoch in range(resume_epoch, n_epochs):
                train_idxs = list(range(0, n_train, args.bs))
                random.Random(args.seed * 1_000_003 + epoch).shuffle(train_idxs)  # reproducible, so resume works
                n_epoch_batches = len(train_idxs)
                skip_steps = resume_skip if epoch == resume_epoch else 0  # optimizer steps already done
                skip = skip_steps * args.accum
                train_idxs = train_idxs[skip:]
                state["epoch"], state["steps_in_epoch"] = epoch, skip_steps
                batches = prefetch(train_idxs, prepare, prep_pool, args.prefetch)

                # a partial accumulation window at the end of an epoch is dropped
                optimizer.zero_grad(set_to_none=True)
                micro, tw = 0, None
                win_loss, win_data, win_vis, win_txt = 0.0, 0.0, 0, 0

                for step in range(len(train_idxs)):
                    t0 = time.perf_counter()
                    item = next(batches)  # blocks only if the background threads fell behind
                    if item is None:
                        continue
                    if tw is None:
                        tw = t0  # start of this accumulation window
                    pixel_values, captions = item
                    assert pixel_values.shape[-1] == S, "processor/vision size mismatch"

                    vision_inputs = {"pixel_values": pixel_values.to(device, non_blocking=True)}
                    enc = tok(
                        [c + tok.eos_token for c in captions],  # teach the model to stop
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=args.max_text_len,
                        pad_to_multiple_of=8,  # tensor-core friendly shapes; padding is masked
                    )
                    win_txt += int(enc["attention_mask"].sum())  # counted on CPU: no GPU sync
                    win_vis += pixel_values.shape[0] * tokens_per_image
                    texts = enc.to(device)
                    # ignore padding in the loss (image positions are masked inside the model)
                    labels = texts["input_ids"].masked_fill(texts["attention_mask"] == 0, -100)

                    win_data += time.perf_counter() - t0  # time spent waiting on / preparing data
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.bf16):
                        out = model(
                            input_ids=texts["input_ids"],
                            attention_mask=texts["attention_mask"],
                            labels=labels,
                            vision_inputs=vision_inputs,
                            use_cache=False,  # no KV cache needed in training
                        )
                    (out.loss / args.accum).backward()
                    win_loss = win_loss + out.loss.detach()  # stays on GPU until the window ends
                    micro += 1
                    if micro < args.accum:
                        continue

                    cur_mult = lr_mult(total_steps)  # multiplier applied to this update
                    gnorm = torch.nn.utils.clip_grad_norm_(params, args.clip)  # returns the pre-clip norm
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                    total_steps += 1
                    state["steps_in_epoch"] += 1
                    loss_item = (win_loss / micro).item()  # forces a GPU sync, so timing is accurate
                    grad_norm = float(gnorm)
                    step_s = (time.perf_counter() - tw) - win_data
                    data_s = win_data

                    if log_file:
                        log_file.write(
                            f"{total_steps}\t{loss_item}\t{win_vis}\t{win_txt}"
                            f"\t{data_s:.4f}\t{step_s:.4f}\t{cur_mult:.5f}\t{grad_norm:.4f}\n"
                        )

                    if total_steps % args.log_freq == 0:
                        if log_file:
                            log_file.flush()
                        log.info(
                            f"epoch {epoch} batch {skip + step}/{n_epoch_batches} step {total_steps} "
                            f"loss {loss_item:.4f} gnorm {grad_norm:.2f} lr_mult {cur_mult:.3f} "
                            f"data {data_s:.2f}s compute {step_s:.2f}s"
                        )

                    maybe_save()
                    if eval_pv is not None and total_steps % args.eval_every == 0:
                        run_eval()
                    micro, tw = 0, None
                    win_loss, win_data, win_vis, win_txt = 0.0, 0.0, 0, 0

                    if total_steps >= planned_steps or (args.stop_after and total_steps >= args.stop_after):
                        done = True
                        break
                if done:
                    break

            if eval_pv is not None and state["evaled"] != total_steps:
                run_eval()  # final eval
            maybe_save(force=True)  # final weights, even if not on the checkpoint schedule
    finally:
        if log_file:
            log_file.close()
        if eval_file:
            eval_file.close()


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)  # abbreviations would bypass the early --device parse

    parser.add_argument("--device", type=int, default=DEFAULT_DEVICE, help="physical GPU index; only this GPU is made visible")
    parser.add_argument("--ckpts", type=str, default="/cs-data-01/drd92/pythia-vlms")
    parser.add_argument("--debug", action="store_true", help="no logging / checkpoints")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--lm", type=str, default="EleutherAI/pythia-160m")
    parser.add_argument("--vis", type=str, default="openai/clip-vit-base-patch32")

    parser.add_argument("--bs", type=int, default=32, help="micro-batch size (per forward/backward)")
    parser.add_argument("--accum", type=int, default=10, help="gradient accumulation steps; effective batch = bs * accum")
    parser.add_argument(
        "--bf16", action=argparse.BooleanOptionalAction, default=False,
        help="bfloat16 autocast; OFF by default (it badly hurt the frozen Pythia LM's loss in our tests)",
    )
    parser.add_argument(
        "--no_mlp", action="store_true",
        help="no projector MLP: vision output feeds the LM directly (with --reset_vision the "
             "vision hidden size is set to the LM hidden size)",
    )
    parser.add_argument("--max_epochs", type=int, default=20)
    parser.add_argument("--max_text_len", type=int, default=512)
    parser.add_argument("--workers", type=int, default=32, help="image download threads")
    parser.add_argument("--prefetch", type=int, default=4, help="batches prepared ahead of the GPU")
    parser.add_argument("--prefetch_workers", type=int, default=2, help="threads preparing batches")
    parser.add_argument(
        "--image_cache", type=str, default="/cs-data-01/drd92/pixmo-image-cache",
        help="dir for caching raw downloaded images across epochs, on by default "
             "(can be 100+ GB for the full dataset); pass '' to disable",
    )

    parser.add_argument("--lr", type=float, default=1e-3, help="LR for from-scratch modules and the projector")
    parser.add_argument("--pretrained_lr", type=float, default=1e-5, help="LR for pretrained modules being fine-tuned")
    parser.add_argument("--vision_lr", type=float, default=None, help="override vision tower LR")
    parser.add_argument("--lm_lr", type=float, default=None, help="override LM LR")
    parser.add_argument("--wd", type=float, default=1e-5)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.995)
    parser.add_argument("--clip", type=float, default=1.0, help="grad-norm clipping")

    # only used with --reset_vision
    parser.add_argument("--clip_hidden_size", type=int, default=768)  # overridden by LM hidden size with --no_mlp
    parser.add_argument("--clip_intermediate_size", type=int, default=3072)  # 4 * hidden
    parser.add_argument("--clip_num_hidden_layers", type=int, default=12)
    parser.add_argument("--clip_num_attention_heads", type=int, default=12)
    parser.add_argument("--clip_image_size", type=int, default=224)
    parser.add_argument("--clip_patch_size", type=int, default=32)

    parser.add_argument("--reset_vision", action="store_true", help="random-init vision tower (and train it)")
    parser.add_argument("--reset_lm", action="store_true", help="random-init LM (and train it)")
    parser.add_argument("--train_vision", action="store_true", help="fine-tune a pretrained vision tower")
    parser.add_argument("--train_lm", action="store_true", help="fine-tune a pretrained LM")

    parser.add_argument("--tf32", action="store_true", help="allow TF32 matmuls (faster fp32 on Ampere+; default is full fp32)")
    parser.add_argument("--max_steps", type=int, default=None, help="total training length in optimizer steps (overrides --max_epochs); the LR schedule is stretched/compressed to fit it, e.g. 143000 to match Pythia")
    parser.add_argument("--resume", type=str, default=None, help="path of an experiment dir (containing train_state.pt) to continue; use the same other flags")
    parser.add_argument("--stop_after", type=int, default=None, help="stop early after this many optimizer steps WITHOUT changing the LR schedule (for debugging: the run is an exact prefix of the full run)")
    parser.add_argument("--schedule", choices=["cosine", "constant"], default="cosine")
    parser.add_argument("--warmup_steps", type=int, default=100, help="linear LR warmup, in optimizer steps")
    parser.add_argument("--min_lr_ratio", type=float, default=0.1, help="cosine decays to this fraction of the peak LR")
    parser.add_argument("--eval_every", type=int, default=1000, help="held-out eval every N optimizer steps (0 = off)")
    parser.add_argument("--eval_size", type=int, default=1024, help="examples held out of training for eval (0 = train on everything)")
    parser.add_argument("--log_freq", type=int, default=10)
    parser.add_argument("--ckpt_every", type=int, default=1000)

    train(parser.parse_args())


if __name__ == "__main__":
    main()