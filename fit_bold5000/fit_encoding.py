"""
Fit ridge encoding models: features from EVERY vision-transformer layer of the
VisualGPTNeoX vision encoder -> BOLD5000 ROI responses, for every checkpoint
matched by MODEL_DIR.

MODEL_DIR may contain a wildcard (".../step_*"): each matching checkpoint
folder gets its own report  <REPORT_DIR>/<subject>_<checkpoint>.tsv  and its own
per-voxel arrays under  <ARRAY_DIR>/<subject>/<checkpoint>/.
Checkpoints whose .tsv already exists are skipped, so an interrupted sweep can
simply be re-run.
"""
import csv
import glob
import os
import re
import sys
import traceback
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ----------------------------------------------------------------- CONFIG
ROI_DIR = "/cs-data-01/drd92/doi_10_5061_dryad_wpzgmsbtr__v20240621"
NAMES_DIR = "/home/drd92/vlm-scaling/fit_bold5000/trial_lists"
IMAGE_DIR = "/cs-data-01/drd92/BOLD5000_Stimuli/Scene_Stimuli"
MODEL_DIR = "/cs-data-01/drd92/pythia-vlms/7/step_*"      # a single checkpoint folder, or a glob
MODEL_CODE_DIR = "/home/drd92/vlm-scaling"
ARRAY_DIR = "/cs-data-01/drd92/fit_bold5000/arrays"
REPORT_DIR = "/home/drd92/vlm-scaling/fit_bold5000/reports"
DEVICE = "cuda:3"
BATCH_SIZE = 512        # images per forward pass; lower it if the GPU runs out of memory
POOLING = "cls+mean"   # how each layer's tokens become one vector: "cls", "mean", "cls+mean", or "grid2"
ROIS = ["EarlyVis", "LOC", "OPA", "PPA", "RSC"]           # each uses LH<roi> + RH<roi>
TEST_FRACTION = 0.2
SEED = 0
SKIP_EXISTING = True   # skip checkpoints whose .tsv report already exists
# -----------------------------------------------------------------------


def load_roi(f, roi, n_trials):
    blocks = []
    for key in (f"LH{roi}", f"RH{roi}"):
        arr = np.asarray(f[key][()], dtype=np.float32)
        assert arr.shape[0] == n_trials, f"{key}: {arr.shape} vs {n_trials} trials"
        blocks.append(arr)
    return np.concatenate(blocks, axis=1)


def average_repeats(names, Y):
    """Images shown several times: average their responses."""
    uniq = sorted(set(names))
    idx = {n: i for i, n in enumerate(uniq)}
    sums = np.zeros((len(uniq), Y.shape[1]))
    counts = np.zeros(len(uniq))
    for n, y in zip(names, Y):
        sums[idx[n]] += y
        counts[idx[n]] += 1
    return uniq, (sums / counts[:, None]).astype(np.float32)


def find_image_paths(names):
    index = {}
    for p in Path(IMAGE_DIR).rglob("*"):
        if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
            index[p.name] = p
            index.setdefault(p.stem, p)
    missing = [n for n in names if n not in index and Path(n).stem not in index]
    if missing:
        raise FileNotFoundError(f"{len(missing)} images not found under {IMAGE_DIR}, e.g. {missing[:3]}")
    return [index.get(n) or index[Path(n).stem] for n in names]


# ------------------------------------------------------------ CHECKPOINTS
def find_checkpoints():
    """Expand MODEL_DIR (may contain a wildcard) into checkpoint folders, ordered by step number."""
    found, skipped = [], []
    for p in map(Path, glob.glob(MODEL_DIR)):
        (found if p.is_dir() and (p / "config.json").exists() else skipped).append(p)
    for p in skipped:
        print(f"Ignoring {p} (not a folder with config.json)")
    if not found:
        sys.exit(f"No checkpoints found matching {MODEL_DIR}")

    def step_key(p):
        m = re.findall(r"\d+", p.name)
        return (int(m[-1]) if m else -1, p.name)

    return sorted(found, key=step_key)


# ------------------------------------------------------------- YOUR MODEL
def build_model(ckpt_dir):
    """Load one VisualGPTNeoX checkpoint and return just its vision encoder (a CLIPVisionModel)
    plus the image preprocessor. The MLP projector and the language model are not used."""
    sys.path.insert(0, str(Path(MODEL_CODE_DIR).resolve()))
    from visual_gpt_neox import VisualGPTNeoXForCausalLM  # also registers the custom config
    from transformers import CLIPImageProcessor

    full = VisualGPTNeoXForCausalLM.from_pretrained(str(ckpt_dir))
    vision = full.visual_gpt_neox.vision_encoder.eval()
    # Default CLIPImageProcessor = OpenAI CLIP preprocessing (resize 224, center crop, CLIP mean/std).
    return vision, CLIPImageProcessor()


def pool_tokens(h, mode):
    """h: (batch, 1 + n_patches, dim); token 0 is CLS. Returns (batch, features)."""
    cls, patches = h[:, 0], h[:, 1:]
    if mode == "cls":
        return cls
    if mode == "mean":
        return patches.mean(dim=1)
    if mode == "cls+mean":
        return torch.cat([cls, patches.mean(dim=1)], dim=1)
    if mode == "grid2":  # keep coarse spatial layout: patches on a 7x7 grid -> 2x2 average
        b, n, d = patches.shape
        s = int(round(n ** 0.5))
        grid = patches.reshape(b, s, s, d).permute(0, 3, 1, 2)
        return torch.nn.functional.adaptive_avg_pool2d(grid, 2).flatten(1)
    raise ValueError(f"unknown POOLING '{mode}'")


@torch.no_grad()
def extract_features(paths, ckpt_dir, batch_size=None):
    """Returns (n_images, n_layers, n_features): one pooled vector per vision-transformer layer."""
    batch_size = batch_size or BATCH_SIZE
    vision, preprocess = build_model(ckpt_dir)
    vision.to(DEVICE)

    feats = []
    for i in range(0, len(paths), batch_size):
        imgs = [Image.open(p).convert("RGB") for p in paths[i:i + batch_size]]
        pixel_values = preprocess(images=imgs, return_tensors="pt")["pixel_values"].to(DEVICE)
        hidden = vision(pixel_values=pixel_values, output_hidden_states=True).hidden_states
        # hidden[0] = patch embeddings (before any transformer layer); hidden[1:] = output of each layer.
        # hidden[-1] is exactly what the language model's projector receives.
        layers = hidden[1:]
        feats.append(torch.stack([pool_tokens(h, POOLING) for h in layers], dim=1).float().cpu().numpy())
        print(f"  features {min(i + batch_size, len(paths))}/{len(paths)}", end="\r")
    print()

    del vision
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return np.concatenate(feats)


# ------------------------------------------------------------------- FIT
def pearson_per_column(pred, true):
    pred = pred - pred.mean(0)
    true = true - true.mean(0)
    return (pred * true).sum(0) / (np.sqrt((pred ** 2).sum(0) * (true ** 2).sum(0)) + 1e-12)


TSV_COLUMNS = ["subject", "checkpoint", "pooling", "layer", "roi", "n_voxels",
               "median_r", "mean_r", "max_r", "frac_r_gt_0.1"]


def fit_checkpoint(X, tr, te, roi_data, subject, ckpt_name, array_dir):
    """Fit every (layer x ROI) for one checkpoint; returns one report row per fit."""
    n_layers = X.shape[1]
    Xtr, Xte = [], []
    for l in range(n_layers):
        sc = StandardScaler().fit(X[tr, l])
        Xtr.append(sc.transform(X[tr, l]))
        Xte.append(sc.transform(X[te, l]))

    rows = []
    for roi in ROIS:
        Y, ys, Ytr = roi_data[roi]
        for l in range(n_layers):
            ridge = RidgeCV(alphas=np.logspace(1, 7, 13), alpha_per_target=True)
            ridge.fit(Xtr[l], Ytr)
            r = pearson_per_column(ys.inverse_transform(ridge.predict(Xte[l])), Y[te])
            np.save(array_dir / f"r_{roi}_layer{l + 1:02d}.npy", r)
            rows.append({
                "subject": subject, "checkpoint": ckpt_name, "pooling": POOLING,
                "layer": l + 1, "roi": roi, "n_voxels": Y.shape[1],
                "median_r": f"{np.median(r):.5f}", "mean_r": f"{r.mean():.5f}",
                "max_r": f"{r.max():.5f}", "frac_r_gt_0.1": f"{(r > 0.1).mean():.4f}",
            })
    return rows


def write_tsv(path, rows):
    tmp = path.with_suffix(".tsv.tmp")  # write-then-rename so a killed run never leaves a partial report
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=TSV_COLUMNS, delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


def main(subject):
    names = [l.strip() for l in open(Path(NAMES_DIR) / f"{subject}_trial_names.txt") if l.strip()]
    # Repeated images are listed as "rep_<name>" but the image file is just "<name>"
    names = [n[4:] if n.startswith("rep_") else n for n in names]
    h5_path = Path(ROI_DIR) / f"{subject}_BOLD5000_ROIs.h5"
    print(f"{subject}: {len(names)} trials")

    # Responses for every ROI (trials averaged over repeats)
    responses = {}
    with h5py.File(h5_path, "r") as f:
        for roi in ROIS:
            uniq, responses[roi] = average_repeats(names, load_roi(f, roi, len(names)))
    print(f"{len(uniq)} unique images")
    image_paths = find_image_paths(uniq)

    # One image-level split and one response standardization, shared by all checkpoints, layers and ROIs
    tr, te = train_test_split(np.arange(len(uniq)), test_size=TEST_FRACTION, random_state=SEED)
    roi_data = {}
    for roi, Y in responses.items():
        ys = StandardScaler().fit(Y[tr])
        roi_data[roi] = (Y, ys, ys.transform(Y[tr]))

    report_dir, subj_array_dir = Path(REPORT_DIR), Path(ARRAY_DIR) / subject
    report_dir.mkdir(parents=True, exist_ok=True)
    subj_array_dir.mkdir(parents=True, exist_ok=True)
    (subj_array_dir / "test_images.txt").write_text("\n".join(uniq[i] for i in te) + "\n")

    ckpts = find_checkpoints()
    print(f"{len(ckpts)} checkpoints: {ckpts[0].name} ... {ckpts[-1].name}\n")

    failed = []
    for ci, ckpt in enumerate(ckpts, 1):
        tsv_path = report_dir / f"{subject}_{ckpt.name}.tsv"
        if SKIP_EXISTING and tsv_path.exists():
            print(f"[{ci}/{len(ckpts)}] {ckpt.name}: report exists, skipping")
            continue
        print(f"[{ci}/{len(ckpts)}] {ckpt.name}")
        try:
            X = extract_features(image_paths, ckpt)
            array_dir = subj_array_dir / ckpt.name
            array_dir.mkdir(exist_ok=True)
            rows = fit_checkpoint(X, tr, te, roi_data, subject, ckpt.name, array_dir)
        except Exception:
            traceback.print_exc()
            failed.append(ckpt.name)
            continue
        write_tsv(tsv_path, rows)
        best = {roi: max((r for r in rows if r["roi"] == roi), key=lambda r: float(r["median_r"])) for roi in ROIS}
        print("  best layer (median r): " + ", ".join(f"{roi} L{b['layer']} ({float(b['median_r']):.3f})" for roi, b in best.items()))
        print(f"  wrote {tsv_path}")

    if failed:
        print(f"\nFAILED checkpoints (re-run to retry): {failed}")


if __name__ == "__main__":
    for subject in ["CSI1", "CSI2", "CSI3", "CSI4"]:
        main(subject)
