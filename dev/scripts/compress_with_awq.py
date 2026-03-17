#!/usr/bin/env python3
"""Calibrate AWQ scales then compress Qwen2.5-1.5B with INT4+MSE+AWQ."""
import subprocess
import sys
import tempfile
from pathlib import Path

MODEL_DIR    = Path.home() / "models" / "Qwen2.5-1.5B-Instruct-bf16"
OUTPUT_DIR   = Path.home() / "models" / "Qwen2.5-1.5B-Instruct-squished-int4-awq"
N_SAMPLES    = 20
# alpha controls AWQ scale aggressiveness: s = mean_act^alpha
# alpha=0.5 (paper default) limits max amplification to s_max = mean_act_max^0.5.
# For Qwen2.5-1.5B the max mean_act ≈ 20, giving max scale ≈ 4.5 (too high for INT4).
# alpha=0.1 caps max amplification at ~1.38× which is friendlier to per-group INT4.
AWQ_ALPHA    = 0.1
# min_scale=0.0 means no floor on scales (default). Setting to 1.0 would
# only amplify channels with mean_act > 1.0 while leaving below-1.0 channels
# unchanged, but benchmarking shows alpha=0.1 with no floor gives the best
# average accuracy (especially arc_easy +2.6% over INT4 MSE).
AWQ_MIN_SCALE = 0.0

# ── Step 1: AWQ calibration ──────────────────────────────────────────────────
print("Step 1: AWQ calibration...")
import mlx_lm
from squish.quant.awq import collect_activation_scales, save_awq_scales
import mlx.core as mx

print(f"  Loading {MODEL_DIR.name} ...")
model, tokenizer = mlx_lm.load(str(MODEL_DIR))
print(f"  Collecting activation scales (n={N_SAMPLES}, alpha={AWQ_ALPHA}, min_scale={AWQ_MIN_SCALE})...")
scales = collect_activation_scales(model, tokenizer, n_samples=N_SAMPLES, alpha=AWQ_ALPHA, min_scale=AWQ_MIN_SCALE, verbose=True)
awq_dir = tempfile.mkdtemp(prefix="squish_awq_")
save_awq_scales(scales, awq_dir, verbose=False)
print(f"  ✓  AWQ scales → {awq_dir}  ({len(scales)} layers)")
del model
mx.clear_cache()

# ── Step 2: Compress ─────────────────────────────────────────────────────────
import shutil
if OUTPUT_DIR.exists():
    shutil.rmtree(OUTPUT_DIR)
print(f"\nStep 2: Compressing to {OUTPUT_DIR} ...")
result = subprocess.run([
    sys.executable, "-m", "squish.convert",
    "--model-dir",   str(MODEL_DIR),
    "--output",      str(OUTPUT_DIR),
    "--format",      "npy-dir",
    "--int4",
    "--super-weight",
    "--awq-scales",  awq_dir,
    "--verbose",
])
if result.returncode != 0:
    print(f"\n✗ Compression failed (exit {result.returncode})")
    sys.exit(result.returncode)

# Size summary
total = sum(f.stat().st_size for f in OUTPUT_DIR.rglob("*") if f.is_file())
print(f"\n✓ Done: {total/1e9:.3f} GB  →  {OUTPUT_DIR}")
