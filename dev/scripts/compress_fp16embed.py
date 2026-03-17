#!/usr/bin/env python3
"""Compress Qwen2.5-1.5B: FP16 attn + FP16 embed_tokens + no AWQ.

Key insight: embed_tokens has outlier ratio 19.2, just under the default
threshold=20, so it gets INT4-quantized.  At 151936×1536 rows, every token
lookup accumulates INT4 error.  Combined with FP16 passthrough on self_attn,
this variant keeps both attn and embedding at FP16 while letting the outlier
threshold handle MLP (82/84 already FP16 at threshold=20).

No AWQ: AWQ modifies FP16 MLP weights (W *= diag(s)) and LN gamma (/= s),
adding floating-point distortion that measured -0.006 hellaswag vs clean FP16.

Expected remaining INT4: 2 MLP tensors (ratio ≤ 20) + 56 LayerNorm weights
(1D, not checked for outliers).  Model size: ~3.2 GB.
"""
import subprocess
import sys
from pathlib import Path
import shutil

MODEL_DIR    = Path.home() / "models" / "Qwen2.5-1.5B-Instruct-bf16"
OUTPUT_DIR   = Path.home() / "models" / "Qwen2.5-1.5B-Instruct-squished-fp16embed"
INT4_GROUP_SIZE = 16

if OUTPUT_DIR.exists():
    shutil.rmtree(OUTPUT_DIR)

print(f"Compressing to {OUTPUT_DIR}")
print(f"  FP16: self_attn (all q/k/v/o) + embed_tokens")
print(f"  INT4 g={INT4_GROUP_SIZE}: 2 MLP tensors + LayerNorm weights (outlier threshold=20)")
print(f"  No AWQ\n")

result = subprocess.run([
    sys.executable, "-m", "squish.convert",
    "--model-dir",       str(MODEL_DIR),
    "--output",          str(OUTPUT_DIR),
    "--format",          "npy-dir",
    "--int4",
    "--int4-group-size", str(INT4_GROUP_SIZE),
    "--super-weight",
    # Two explicit passthrough patterns:
    #   "self_attn"    → all q/k/v/o projection weights + biases
    #   "embed_tokens" → the vocabulary embedding table (ratio 19.2, just under threshold)
    "--passthrough",     "self_attn", "embed_tokens",
    # No --awq-scales: clean FP16 weights without scale distortion
    "--verbose",
])
if result.returncode != 0:
    print(f"\n✗ Compression failed (exit {result.returncode})")
    sys.exit(result.returncode)

total = sum(f.stat().st_size for f in OUTPUT_DIR.rglob("*") if f.is_file())
print(f"\n✓ Done: {total/1e9:.3f} GB  →  {OUTPUT_DIR}")
