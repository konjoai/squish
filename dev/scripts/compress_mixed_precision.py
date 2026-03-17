#!/usr/bin/env python3
"""Compress Qwen2.5-1.5B as mixed-precision: FP16 attention + INT4 MLP + AWQ.

Hypothesis: attention projections (q/k/v/o) drive context-reasoning tasks
(hellaswag, piqa, winogrande). Keeping them at FP16 should recover those
losses while INT4+AWQ on MLP preserves the arc_easy gain.

Architecture:
  - self_attn.*: FP16 passthrough (~308 MB)
  - mlp.gate_proj / mlp.up_proj: INT4 g=16 + AWQ (MLP-only)
  - mlp.down_proj: INT4 g=16 (no AWQ — down_proj has no shared LN)
  - embed_tokens / lm_head / layernorm: FP16 passthrough (existing behavior)
"""
import subprocess
import sys
import tempfile
from pathlib import Path

MODEL_DIR    = Path.home() / "models" / "Qwen2.5-1.5B-Instruct-bf16"
OUTPUT_DIR   = Path.home() / "models" / "Qwen2.5-1.5B-Instruct-squished-mixed"
N_SAMPLES    = 20
AWQ_ALPHA    = 0.1
AWQ_MIN_SCALE = 0.0
INT4_GROUP_SIZE = 16

_MLP_LEAVES = frozenset({"gate_proj", "up_proj", "fc1", "dense_h_to_4h"})

# ── Step 1: AWQ calibration (MLP-only) ──────────────────────────────────────
print("Step 1: AWQ calibration (MLP-only)...")
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
# Filter to MLP-only: attention is FP16 so no AWQ compensation needed there
before = len(scales)
scales = {k: v for k, v in scales.items() if k.split(".")[-1] in _MLP_LEAVES}
print(f"  MLP-only filter: {before} → {len(scales)} layers (attn is FP16 passthrough)")

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
    "--model-dir",       str(MODEL_DIR),
    "--output",          str(OUTPUT_DIR),
    "--format",          "npy-dir",
    "--int4",
    "--int4-group-size", str(INT4_GROUP_SIZE),
    "--super-weight",
    "--awq-scales",      awq_dir,
    # Keep all attention projections at FP16 — substring match on "self_attn"
    "--passthrough",     "self_attn",
    "--verbose",
])
if result.returncode != 0:
    print(f"\n✗ Compression failed (exit {result.returncode})")
    sys.exit(result.returncode)

# Size summary
total = sum(f.stat().st_size for f in OUTPUT_DIR.rglob("*") if f.is_file())
print(f"\n✓ Done: {total/1e9:.3f} GB  →  {OUTPUT_DIR}")
