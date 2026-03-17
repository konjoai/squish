#!/usr/bin/env python3
"""Compress Qwen2.5-1.5B: FP16 attn + INT4 g=32 MLP + lighter AWQ.

Compared to mixed v1 (g=16, alpha=0.1, n=20):
  - Group size 16 → 32: each group has 2× more elements → lower per-group
    quantization error → should recover arc_easy toward 0.750+
  - alpha 0.1 → 0.05: lighter activation smoothing → less LayerNorm distortion
    → should recover hellaswag toward 0.612 (the no-AWQ ceiling)
  - Calibration 20 → 64 samples: better AWQ scale estimates → less noise

Target: all four tasks ≥ squish-path BF16 reference
  arc_easy > 0.750, hellaswag > 0.612, piqa > 0.772, winogrande > 0.630
"""
import subprocess
import sys
import tempfile
from pathlib import Path

MODEL_DIR       = Path.home() / "models" / "Qwen2.5-1.5B-Instruct-bf16"
OUTPUT_DIR      = Path.home() / "models" / "Qwen2.5-1.5B-Instruct-squished-mixed-v2"
N_SAMPLES       = 64    # was 20
AWQ_ALPHA       = 0.05  # was 0.1
AWQ_MIN_SCALE   = 0.0
INT4_GROUP_SIZE = 32    # was 16

_MLP_LEAVES = frozenset({"gate_proj", "up_proj", "fc1", "dense_h_to_4h"})

# ── Step 1: AWQ calibration (MLP gate/up only) ────────────────────────────────
print("Step 1: AWQ calibration (MLP gate/up only)...")
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
# Filter to MLP gate/up only — attn is FP16 so no AWQ compensation needed
before = len(scales)
scales = {k: v for k, v in scales.items() if k.split(".")[-1] in _MLP_LEAVES}
print(f"  MLP-only filter: {before} → {len(scales)} layers")
awq_dir = tempfile.mkdtemp(prefix="squish_awq_v2_")
save_awq_scales(scales, awq_dir, verbose=False)
print(f"  ✓  AWQ scales → {awq_dir}  ({len(scales)} layers)")
del model
mx.clear_cache()

# ── Step 2: Compress ──────────────────────────────────────────────────────────
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
    "--passthrough",     "self_attn",   # keep all attention FP16
    "--verbose",
])
if result.returncode != 0:
    print(f"\n✗ Compression failed (exit {result.returncode})")
    sys.exit(result.returncode)

total = sum(f.stat().st_size for f in OUTPUT_DIR.rglob("*") if f.is_file())
print(f"\n✓ Done: {total/1e9:.3f} GB  →  {OUTPUT_DIR}")
