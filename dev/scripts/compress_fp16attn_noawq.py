#!/usr/bin/env python3
"""Compress Qwen2.5-1.5B: FP16 attention, no AWQ, default outlier threshold.

Hypothesis: with outlier threshold=20, almost all MLP tensors become FP16
passthrough anyway.  AWQ modifies those FP16 weights (W *= diag(s)),
introducing errors vs BF16 that are hurting hellaswag.  Removing AWQ should
restore hellaswag close to BF16 while keeping attn at FP16.

The tradeoff is arc_easy: without AWQ, arc_easy may drop ~0.020-0.030.
This test isolates whether AWQ modification is the hellaswag culprit.
"""
import subprocess
import sys
from pathlib import Path

MODEL_DIR    = Path.home() / "models" / "Qwen2.5-1.5B-Instruct-bf16"
OUTPUT_DIR   = Path.home() / "models" / "Qwen2.5-1.5B-Instruct-squished-fp16attn-noawq"
INT4_GROUP_SIZE = 16

import shutil
if OUTPUT_DIR.exists():
    shutil.rmtree(OUTPUT_DIR)
print(f"Compressing to {OUTPUT_DIR} (FP16 attn, no AWQ, g=16)...")
result = subprocess.run([
    sys.executable, "-m", "squish.convert",
    "--model-dir",       str(MODEL_DIR),
    "--output",          str(OUTPUT_DIR),
    "--format",          "npy-dir",
    "--int4",
    "--int4-group-size", str(INT4_GROUP_SIZE),
    "--super-weight",
    # No --awq-scales: pure quantization without scale distortion
    "--passthrough",     "self_attn",
    "--verbose",
])
if result.returncode != 0:
    print(f"\n✗ Compression failed (exit {result.returncode})")
    sys.exit(result.returncode)

total = sum(f.stat().st_size for f in OUTPUT_DIR.rglob("*") if f.is_file())
print(f"\n✓ Done: {total/1e9:.3f} GB  →  {OUTPUT_DIR}")
