#!/usr/bin/env python3
"""Compress Qwen2.5-1.5B: FP16 everything except LN weights.

Passes through: self_attn + embed_tokens + ALL MLP tensors.
The only INT4 remaining: 57 LayerNorm weight vectors (each 1536-dim, ~3KB).

This validates whether those last 2 INT4 MLP tensors or any INT4 LN weights
are causing the persistent hellaswag gap of 0.023 from BF16.
"""
import subprocess
import sys
from pathlib import Path
import shutil

MODEL_DIR    = Path.home() / "models" / "Qwen2.5-1.5B-Instruct-bf16"
OUTPUT_DIR   = Path.home() / "models" / "Qwen2.5-1.5B-Instruct-squished-fp16mlp"
INT4_GROUP_SIZE = 16

if OUTPUT_DIR.exists():
    shutil.rmtree(OUTPUT_DIR)

print(f"Compressing to {OUTPUT_DIR}")
print(f"  FP16 passthrough: self_attn + embed_tokens + mlp")
print(f"  INT4 g={INT4_GROUP_SIZE}: LayerNorm weights only (57 × 1536-dim vectors)")
print(f"  No AWQ\n")

result = subprocess.run([
    sys.executable, "-m", "squish.convert",
    "--model-dir",       str(MODEL_DIR),
    "--output",          str(OUTPUT_DIR),
    "--format",          "npy-dir",
    "--int4",
    "--int4-group-size", str(INT4_GROUP_SIZE),
    "--super-weight",
    "--passthrough",     "self_attn", "embed_tokens", "mlp",
    "--verbose",
])
if result.returncode != 0:
    print(f"\n✗ Compression failed (exit {result.returncode})")
    sys.exit(result.returncode)

total = sum(f.stat().st_size for f in OUTPUT_DIR.rglob("*") if f.is_file())
print(f"\n✓ Done: {total/1e9:.3f} GB  →  {OUTPUT_DIR}")
