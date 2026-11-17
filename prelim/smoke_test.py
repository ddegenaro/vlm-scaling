"""
Smoke test: forward + backward + optimizer step timing and peak GPU memory
across a sweep of batch sizes, for a Qwen3-VL-style architecture.

Example:
    python smoke_test.py --batch_sizes 1 2 4 8 16 32
    python smoke_test.py --batch_sizes 4 8 16 --grad_checkpointing
    python smoke_test.py --random_init   # same architecture, fresh weights
"""

import argparse
import gc
import json
import statistics
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from time import perf_counter

import requests
import torch
from datasets import load_dataset
from PIL import Image
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from transformers import AutoConfig, AutoModelForMultimodalLM, AutoProcessor

GiB = 1024**3


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def _fetch_image(url, size):
    """Download one image and resize to a fixed square. Returns None on failure."""
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        img = Image.open(BytesIO(r.content)).convert("RGB")
        return img.resize((size, size))
    except Exception:
        return None


def build_pool(dataset, n_samples, image_size, workers):
    """
    Download images up front so network latency never shows up in step timing.
    A fixed image size gives every sample the same number of vision tokens, which
    keeps memory/timing comparable across batch sizes.
    """
    n_candidates = int(n_samples * 1.5) + 16  # some pixmo-cap URLs are dead
    rows = dataset.select(range(n_candidates))
    urls, caps = rows["image_url"], rows["caption"]

    print(f"Downloading up to {n_candidates} images ({workers} threads)...")
    with ThreadPoolExecutor(workers) as ex:
        imgs = list(ex.map(lambda u: _fetch_image(u, image_size), urls))

    pool = [(im, c) for im, c in zip(imgs, caps) if im is not None][:n_samples]
    if len(pool) < n_samples:
        raise RuntimeError(f"Only got {len(pool)}/{n_samples} images; check network.")
    print(f"Image pool ready: {len(pool)} samples")
    return pool


def make_inputs(processor, samples, device):
    """Chat-template the (image, caption) pairs and build LM labels."""
    messages = [
        [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Describe this image."},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": cap}]},
        ]
        for _, cap in samples
    ]
    texts = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    inputs = processor(
        text=texts,
        images=[img for img, _ in samples],
        padding=True,
        return_tensors="pt",
    )

    labels = inputs["input_ids"].clone()
    labels[inputs["attention_mask"] == 0] = -100  # padding
    image_token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
    labels[labels == image_token_id] = -100  # don't train on image placeholders

    inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}
    inputs["labels"] = labels.to(device)
    return inputs


# --------------------------------------------------------------------------- #
# Benchmark
# --------------------------------------------------------------------------- #
def free_memory():
    gc.collect()
    torch.cuda.empty_cache()


def bench_batch_size(model, optimizer, scheduler, processor, pool, bs, args):
    """Run warmup + measured steps at one batch size. Returns a result dict."""
    device = next(model.parameters()).device
    torch.cuda.set_device(device)  # events/empty_cache act on the current device
    model.train()
    free_memory()

    total_steps = args.warmup_steps + args.num_batches
    fwd_ms, bwd_ms, opt_ms, prep_ms = [], [], [], []
    seq_lens, mem_after_fwd = [], []

    try:
        for step in range(total_steps):
            if step == args.warmup_steps:
                # Peak stats cover only measured steps, not warmup/allocator ramp-up.
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)

            # cycle through the pool so we don't need bs * num_batches unique images
            start = (step * bs) % (len(pool) - bs + 1)
            samples = pool[start : start + bs]

            t0 = perf_counter()
            inputs = make_inputs(processor, samples, device)
            t_prep = (perf_counter() - t0) * 1e3

            ev = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
            ev[0].record()
            loss = model(**inputs).loss
            ev[1].record()
            mem_fwd = torch.cuda.memory_allocated(device)  # activations held for backward
            loss.backward()
            ev[2].record()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            ev[3].record()
            torch.cuda.synchronize(device)

            if step >= args.warmup_steps:
                fwd_ms.append(ev[0].elapsed_time(ev[1]))
                bwd_ms.append(ev[1].elapsed_time(ev[2]))
                opt_ms.append(ev[2].elapsed_time(ev[3]))
                prep_ms.append(t_prep)
                seq_lens.append(inputs["input_ids"].shape[1])
                mem_after_fwd.append(mem_fwd)

            if not torch.isfinite(loss):
                print(f"  [bs={bs}] WARNING: non-finite loss at step {step}")

    except torch.OutOfMemoryError:
        optimizer.zero_grad(set_to_none=True)
        loss = inputs = None
        free_memory()
        return {"batch_size": bs, "oom": True}

    step_ms = [f + b + o for f, b, o in zip(fwd_ms, bwd_ms, opt_ms)]
    mean_seq = statistics.mean(seq_lens)
    return {
        "batch_size": bs,
        "oom": False,
        "fwd_ms": statistics.mean(fwd_ms),
        "bwd_ms": statistics.mean(bwd_ms),
        "opt_ms": statistics.mean(opt_ms),
        "step_ms": statistics.mean(step_ms),
        "step_ms_std": statistics.pstdev(step_ms),
        "prep_ms_cpu": statistics.mean(prep_ms),
        "samples_per_s": bs / (statistics.mean(step_ms) / 1e3),
        "tokens_per_s": bs * mean_seq / (statistics.mean(step_ms) / 1e3),
        "mean_seq_len": mean_seq,
        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / GiB,
        "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / GiB,
        "after_fwd_gib": statistics.mean(mem_after_fwd) / GiB,
    }


def print_table(results):
    cols = [
        ("bs", "batch_size", "{:>4}"),
        ("seq", "mean_seq_len", "{:>6.0f}"),
        ("fwd ms", "fwd_ms", "{:>8.1f}"),
        ("bwd ms", "bwd_ms", "{:>8.1f}"),
        ("opt ms", "opt_ms", "{:>7.1f}"),
        ("step ms", "step_ms", "{:>8.1f}"),
        ("±", "step_ms_std", "{:>6.1f}"),
        ("samp/s", "samples_per_s", "{:>7.2f}"),
        ("tok/s", "tokens_per_s", "{:>9.0f}"),
        ("fwd GiB", "after_fwd_gib", "{:>8.2f}"),
        ("peak alloc", "peak_allocated_gib", "{:>10.2f}"),
        ("peak rsvd", "peak_reserved_gib", "{:>9.2f}"),
    ]
    print("\n" + " ".join(f"{h:>{len(f.format(0))}}" for h, _, f in cols))
    for r in results:
        if r["oom"]:
            print(f"{r['batch_size']:>4}  OOM")
        else:
            print(" ".join(f.format(r[k]) for _, k, f in cols))


# --------------------------------------------------------------------------- #
# Entry points
# --------------------------------------------------------------------------- #
def run_smoke_test(args):
    assert torch.cuda.is_available(), "CUDA GPU required"
    device = torch.device(args.device)
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    torch.cuda.set_device(device)
    dtype = getattr(torch, args.dtype)
    props = torch.cuda.get_device_properties(device)
    print(f"GPU: {device} = {props.name}, {props.total_memory / GiB:.1f} GiB")

    # ---- data (downloaded before anything is timed) ----
    dataset = load_dataset(args.dataset, split="train")
    batch_sizes = sorted(args.batch_sizes)
    pool_size = max(batch_sizes) * args.pool_batches
    pool = build_pool(dataset, pool_size, args.image_size, args.download_workers)

    # ---- model ----
    processor = AutoProcessor.from_pretrained(args.model_name)
    if args.random_init:
        config = AutoConfig.from_pretrained(args.model_name)
        model = AutoModelForMultimodalLM.from_config(config, dtype=dtype)
        model.to(device)
    else:
        model = AutoModelForMultimodalLM.from_pretrained(
            args.model_name, dtype=dtype, device_map={"": device.index}
        )
    if args.grad_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    # Sanity check: everything must live on the requested GPU, and be visible to the
    # memory counters we report from.
    bad = {str(p.device) for p in model.parameters() if p.device != device}
    assert not bad, f"parameters found on {bad}, expected {device}"
    assert torch.cuda.memory_allocated(device) > 0, f"no memory allocated on {device}"
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.model_name} | {n_params / 1e9:.2f}B params | {args.dtype}")
    print(
        f"Weights on {device}: {torch.cuda.memory_allocated(device) / GiB:.2f} GiB | "
        f"grad ckpt: {args.grad_checkpointing}"
    )

    # ---- optimizer / scheduler ----
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        betas=tuple(args.betas),
        eps=args.eps,
        weight_decay=args.wd,
        fused=True,
    )
    scheduler = CosineAnnealingWarmRestarts(
        optimizer=optimizer,
        T_0=args.T_0,
        T_mult=args.T_mult,
        eta_min=args.eta_min,
        last_epoch=args.last_epoch,
    )

    # ---- sweep (ascending; stop at first OOM since larger sizes will also fail) ----
    results = []
    for bs in batch_sizes:
        print(f"\n=== batch size {bs} ===")
        r = bench_batch_size(model, optimizer, scheduler, processor, pool, bs, args)
        results.append(r)
        if r["oom"]:
            print("  OOM -> stopping sweep")
            break
        print(
            f"  step {r['step_ms']:.0f} ms | peak alloc {r['peak_allocated_gib']:.2f} GiB"
        )

    print_table(results)
    with open(args.output, "w") as f:
        json.dump({"args": vars(args), "results": results}, f, indent=2)
    print(f"\nSaved results to {args.output}")


def parse_args():
    p = argparse.ArgumentParser()

    # scaling of smoke test
    p.add_argument("--device", default="cuda:0", help="e.g. cuda:2")
    p.add_argument("--model_name", default="Qwen/Qwen3-VL-2B-Instruct")
    p.add_argument("--dataset", default="allenai/pixmo-cap")
    p.add_argument("--batch_sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    p.add_argument("--num_batches", type=int, default=10, help="measured steps per bs")
    p.add_argument("--warmup_steps", type=int, default=3)
    p.add_argument("--pool_batches", type=int, default=4,
                   help="image pool = max(batch_sizes) * this; steps cycle through it")
    p.add_argument("--image_size", type=int, default=448,
                   help="fixed square resize; 448 -> 196 vision tokens for Qwen3-VL")
    p.add_argument("--download_workers", type=int, default=32)
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32", "float16"])
    p.add_argument("--grad_checkpointing", action="store_true")
    p.add_argument("--random_init", action="store_true",
                   help="same architecture, randomly initialised weights")
    p.add_argument("--output", default="smoke_test_results.json")

    # optimizer
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--betas", type=float, nargs=2, default=[0.9, 0.95])
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--wd", type=float, default=0.1)

    # scheduler (units are optimizer steps)
    p.add_argument("--T_0", type=int, default=10)
    p.add_argument("--T_mult", type=int, default=2)
    p.add_argument("--eta_min", type=float, default=1e-6)
    p.add_argument("--last_epoch", type=int, default=-1)

    return p.parse_args()


def main():
    run_smoke_test(parse_args())


if __name__ == "__main__":
    main()
