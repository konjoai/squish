#!/usr/bin/env python3
"""Test that the n=1 fix in quantize_int4_asymmetric_mse works correctly."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np

# Simulate model.norm.weight: predominantly small values with one outlier
weight_1d = np.ones(1536, dtype=np.float32)
weight_1d[42] = 8.875  # the critical outlier

flat = weight_1d.reshape(1, 1536)  # simulate how convert.py reshapes it
print(f"Input: shape={flat.shape}  max={flat.max()}")

from squish.quant.quantizer import quantize_int4_asymmetric_mse, dequantize_int4_asymmetric

packed, scales, zps = quantize_int4_asymmetric_mse(flat, group_size=32)
recon = dequantize_int4_asymmetric(packed, scales, zps, group_size=32)
print(f"Recon: shape={recon.shape}  max={recon.max():.4f}")
print(f"  outlier preserved: {recon.max() >= 8.0}")
print(f"  recon[0, 42] = {recon[0, 42]:.4f} (original = {flat[0, 42]})")
