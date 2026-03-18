#!/usr/bin/env python3
"""Verify that the class-swizzling AWQ hook fix captures activation statistics."""
import sys

import mlx.core as mx
import mlx_lm

from squish.quant.awq import collect_activation_scales

MODEL_DIR = "/Users/wscholl/models/Qwen2.5-1.5B-Instruct-bf16"

print("Loading model...")
model, tok = mlx_lm.load(MODEL_DIR)

print("Running AWQ calibration (n=4 — quick smoke test)...")
scales = collect_activation_scales(model, tok, n_samples=4, verbose=True)

print(f"\nResult: {len(scales)} layer scales captured")
if len(scales) == 0:
    print("FAIL: 0 layers — hook fix did not work")
    sys.exit(1)

k = next(iter(scales))
s = scales[k]
print(f"  Example: '{k}'  shape={s.shape}  mean={s.mean():.4f}  max={s.max():.4f}")
print("PASS: AWQ class-swizzling hook fix is working")
