#!/usr/bin/env python3
"""Quick check of the NEW model's norm.weight tensor."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import numpy as np
from squish.quant.compressed_loader import _dequantize_npy_dir

tensor_dir = Path.home() / "models/Qwen2.5-1.5B-Instruct-squished-int4-mse/tensors"
key = "model__norm__weight"
arr = _dequantize_npy_dir(tensor_dir, key)
print(f"model.norm.weight: shape={arr.shape}  range=[{arr.min():.4f}, {arr.max():.4f}]  std={arr.std():.4f}")
print(f"Max value (should be ~8.875): {arr.max():.4f}")
print(f"PASS" if arr.max() >= 8.0 else "FAIL - outlier still clipped!")
