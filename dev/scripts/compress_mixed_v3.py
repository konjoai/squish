#!/usr/bin/env python3
"""Compress: FP16 attn + INT4 g=16 AWQ alpha=0.15.

v1 (alpha=0.10) gave arc_easy=0.746 (+0.012 vs no-AWQ baseline 0.734).
Hypothesis: scaling alpha 0.10 → 0.15 pushes arc_easy the remaining 0.004
needed to clear the BF16 reference of 0.750, while hellaswag stays ≥0.606.

All other settings identical to v1: g=16, n=20 calibration, passthrough=self_attn.
"""
import subprocess
import sys
import tempfile
from pathlib import Path

MODEL_DIR       = Path.home() / "models" / "Qwen2.5-1.5B-Instruct-bf16"
OUTPUT_DIR      = Path.home() / "models" / "Qwen2.5-1.5B-Instruct-squished-mixed-v3"
N_SAMPLES       = 20   # same as v1
AWQ_ALPHA       = 0.15 # was 0.10
AWQ_MIN_SCALE   = 0.0
INT4_GROUP_SIZE = 16   # same as v1

_MLP_LEAVES = frozenset({"gate_proj", "up_proj", "fc1", "dense_h_to_4h"})

print("Step 1: AWQ calibration (MLP gate/up only, alpha=0.15)...")
import mlx_lm
from squish.quant.awq import collect_activation_scales, save_awq_scales
import mlx.core as mx

print(f"  Loading {MODEL_DIR.name} ...")
model, tokenizer = mlx_lm.load(str(MODEL_DIR))
print(f"  Collecting activation scales (n={N_SAMPLES}, alpha={AWQ_ALPHA})...")
scales = collect_activation_scales(
    model, tokenizer, n_samples=N_SAMPLES, alpha=AWQ_ALPHA,
    min_scale=AWQ_MIN_SCALE, verbose=True,
)
before = len(scales)
scales = {k: v for k, v in scales.items() if k.split(".")[-1] in _MLP_LEAVES}
print(f"  MLP-only filter: {before} → {len(scales)} layers")
awq_dir = tempfile.mkdtemp(prefix="squish_awq_v3_")
save_awq_scales(scales, awq_dir, verbose=False)
print(f"  ✓  AWQ scales → {awq_dir}  ({len(scales)} layers)")
del model
mx.clear_cache()

import shutil
if OUTPUT_DIR.exists():
    shutil.rmtree(OUTPUT_DIR)
print(f"\nStep 2: Compressing to {OUTPUT_DIR} ...")
result = subprocess.run([
    sys.executable, "-m", "squish.convert",
    "--model-dir",       str(MODEL_DIR),
    "--output",          str(OUTPUT_DIR),
    "--format",          "npy-dir",
    "--int4",
    "--int4-group-size", str(INT4_GROUP_SIZE),
    "--super-weight",
    "--awq-scales",      awq_dir,
    "--passthrough",     "self_attn",
    "--verbose",
])
if result.returncode != 0:
    sys.exit(result.returncode)

total = sum(f.stat().st_size for f in OUTPUT_DIR.rglob("*") if f.is_file())
print(f"\n✓ Done: {total/1e9:.3f} GB  →  {OUTPUT_DIR}")
