#!/usr/bin/env python3
"""Fully lossless passthrough model: EVERY weight stored as FP16.

Uses outlier-threshold=0.0001 (all 2D+ tensors pass through) plus
explicit passthrough for LN weights (1D). This is effectively a no-op
compression: BF16 -> FP16 -> BF16. If this model gives the same ~0.635
hellaswag as BF16 reference, weight quantization is not the issue and we
just need to commit our best model. If it still gives ~0.612, there is a
bug in squish's inference path vs mlx_lm direct inference.
"""
import subprocess
import sys
from pathlib import Path
import shutil

MODEL_DIR    = Path.home() / "models" / "Qwen2.5-1.5B-Instruct-bf16"
OUTPUT_DIR   = Path.home() / "models" / "Qwen2.5-1.5B-Instruct-squished-lossless"

if OUTPUT_DIR.exists():
    shutil.rmtree(OUTPUT_DIR)

print(f"Compressing to {OUTPUT_DIR}")
print(f"  FULLY LOSSLESS: passthrough self_attn+embed_tokens+mlp+norm")
print(f"  Covers: attn weights+biases, embed, mlp, all LN vectors")
print(f"  Expected: identical to BF16 reference (BF16→FP16→BF16 rounding only)\n")

result = subprocess.run([
    sys.executable, "-m", "squish.convert",
    "--model-dir",        str(MODEL_DIR),
    "--output",           str(OUTPUT_DIR),
    "--format",           "npy-dir",
    "--int4",
    "--int4-group-size",  "16",
    # Pass through every tensor type:
    #   self_attn  → all attention weights (q/k/v/o) + biases (1D)
    #   embed_tokens → embedding table (2D)
    #   mlp        → all MLP weight matrices (no biases in Qwen2.5 SwiGLU)
    #   norm       → all LN weight vectors (1D: input_layernorm, post_attn_LN, model.norm)
    "--passthrough",      "self_attn", "embed_tokens", "mlp", "norm",
    "--verbose",
])
if result.returncode != 0:
    print(f"\n✗ Compression failed (exit {result.returncode})")
    sys.exit(result.returncode)

total = sum(f.stat().st_size for f in OUTPUT_DIR.rglob("*") if f.is_file())
print(f"\n✓ Done: {total/1e9:.3f} GB  →  {OUTPUT_DIR}")
