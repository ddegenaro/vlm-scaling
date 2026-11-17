#!/usr/bin/env bash
# Sweep of systems-relevant settings for smoke_test.py on a single L40S.
#
#   dtype            : bfloat16 | float32
#   grad checkpoint  : off | on
#   image_size       : 336 | 448 | 672   (-> 144 | 196 | 441 vision tokens)
#
# 2 x 2 x 3 = 12 runs. Each run is its own process, so GPU memory and the
# allocator start clean every time. Finished runs are skipped, so you can
# re-run the script after an interruption.
#
# Usage:
#   ./sweep.sh
#   GPU=2 ./sweep.sh        # run on physical GPU 2 only
#   OUT_DIR=results_v2 EXTRA_ARGS="--num_batches 20" ./sweep.sh

set -uo pipefail

OUT_DIR="${OUT_DIR:-sweep_results}"
EXTRA_ARGS="${EXTRA_ARGS:---random_init}"   # random_init = same arch, fresh weights
PYTHON="${PYTHON:-python}"
GPU="${GPU:-2}"   # PHYSICAL GPU index, e.g. GPU=2 ./sweep.sh
# Hide every other GPU from the benchmark processes. This guarantees nothing in the
# stack (torch, accelerate, kernels, ...) can create a context on another GPU.
# Inside the process the chosen GPU is then simply cuda:0.
export CUDA_VISIBLE_DEVICES="$GPU"

DTYPES=(bfloat16 float32)
IMAGE_SIZES=(336 448 672)

# Checkpointing frees activation memory, so it gets a higher batch-size ladder.
BS_NO_CKPT="1 2 4 8 16 32"
BS_CKPT="4 8 16 32 64"

mkdir -p "$OUT_DIR/logs"

for dtype in "${DTYPES[@]}"; do
  for ckpt in off on; do
    for size in "${IMAGE_SIZES[@]}"; do
      tag="${dtype}_ckpt-${ckpt}_img${size}"
      out="$OUT_DIR/${tag}.json"
      log="$OUT_DIR/logs/${tag}.log"

      if [[ -f "$out" ]]; then
        echo "[skip] $tag (already done)"
        continue
      fi

      if [[ "$ckpt" == "on" ]]; then
        bs="$BS_CKPT"; ckpt_flag="--grad_checkpointing"
      else
        bs="$BS_NO_CKPT"; ckpt_flag=""
      fi

      echo "[run ] $tag  (physical GPU $GPU, batch sizes: $bs)"
      # shellcheck disable=SC2086  # intentional word splitting for flag lists
      if ! $PYTHON smoke_test.py \
            --device cuda:0 \
            --dtype "$dtype" \
            --image_size "$size" \
            --batch_sizes $bs \
            $ckpt_flag \
            $EXTRA_ARGS \
            --output "$out" >"$log" 2>&1; then
        echo "[FAIL] $tag -> see $log"
      fi
    done
  done
done

# ---- summary across all runs ------------------------------------------------
$PYTHON - "$OUT_DIR" <<'EOF'
import glob, json, os, sys

out_dir = sys.argv[1]
rows = []
for path in sorted(glob.glob(os.path.join(out_dir, "*.json"))):
    tag = os.path.basename(path)[:-5]
    res = json.load(open(path))["results"]
    ok = [r for r in res if not r["oom"]]
    oom = [r["batch_size"] for r in res if r["oom"]]
    if not ok:
        rows.append((tag, "-", "-", "-", "-", f"OOM at bs={oom[0]}" if oom else "no data"))
        continue
    best = max(ok, key=lambda r: r["tokens_per_s"])
    last = ok[-1]
    rows.append((
        tag,
        str(last["batch_size"]),
        f"{last['peak_allocated_gib']:.1f}",
        str(best["batch_size"]),
        f"{best['tokens_per_s']:.0f}",
        f"OOM at bs={oom[0]}" if oom else "no OOM",
    ))

hdr = ("config", "max bs ok", "peak GiB", "best bs", "best tok/s", "note")
w = [max(len(str(r[i])) for r in rows + [hdr]) for i in range(len(hdr))]
fmt = "  ".join(f"{{:<{x}}}" for x in w)
print("\n" + fmt.format(*hdr))
print(fmt.format(*("-" * x for x in w)))
for r in rows:
    print(fmt.format(*r))
EOF
