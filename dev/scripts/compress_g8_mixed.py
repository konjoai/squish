#!/usr/bin/env python3
"""Compress Qwen2.5-1.5B as mixed-precision with finer MLP groups.

FP16 attention + INT4 g=8 MLP + AWQ (MLP-only, alpha=0.1).

Hypothesis: g=8 (vs g=16) halves MLP quantization error and should recover
the remaining hellaswag gap (-0.029) from the g=16 mixed model.
Expected size: ~3.05 GB (+210 MB scale overhead vs g=16).
"""
import subprocess
import sys
import tempfile
from pathlib import Path

MODEL_DIR    = Path.home() / "models" / "Qwen2.5-1.5B-Instruct-bf16"
OUTPUT_DIR   = Path.home() / "models" / "Qwen2.5-1.5B-Instruct-squished-g8-mixed"
N_SAMPLES    = 20
AWQ_ALPHA    = 0.1
AWQ_MIN_SCALE = 0.0
INT4_GROUP_SIZE = 8   # finer groups for MLP

_MLP_LEAVES = frozenset({"gate_proj", "up_proj", "fc1", "dense_h_to_4h"})

# ── Step 1: AWQ calibration (MLP-only) ───────────────────────────────────────
print("Step 1: AWQ calibration (MLP-only, alpha=0.1)...")
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

awq_dir = tempfile.mkdtemp(prefix="squish_awq_g8_")
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
    "--passthrough",     "self_attn",
    "--verbose",
])
if result.returncode != 0:
    print(f"\n✗ Compression failed (exit {result.returncode})")
    sys.exit(result.returncode)

total = sum(f.stat().st_size for f in OUTPUT_DIR.rglob("*") if f.is_file())
print(f"\n✓ Done: {total/1e9:.3f} GB  →  {OUTPUT_DIR}")
