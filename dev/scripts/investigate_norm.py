#!/usr/bin/env python3
"""Investigate the model.norm.weight discrepancy."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import torch
from safetensors.torch import load_file as st_load
from squish.quant.compressed_loader import _dequantize_npy_dir

OLD_DIR    = Path.home() / "models/Qwen2.5-1.5B-Instruct-bf16-compressed"
NEW_DIR    = Path.home() / "models/Qwen2.5-1.5B-Instruct-squished-int4-mse"
BF16_MODEL = Path.home() / "models/Qwen2.5-1.5B-Instruct-bf16/model.safetensors"

orig_st = {k: v.float().numpy() for k, v in st_load(str(BF16_MODEL)).items()}

# ── model.norm.weight ─────────────────────────────────────────────────────────
print("=== model.norm.weight ===")
key = "model__norm__weight"
safek = "model.norm.weight"

orig = orig_st[safek]
print(f"BF16 original: shape={orig.shape}  range=[{orig.min():.4f}, {orig.max():.4f}]  std={orig.std():.4f}")

old_files = sorted(f.name for f in (OLD_DIR / "tensors").glob(key + "*"))
new_files = sorted(f.name for f in (NEW_DIR / "tensors").glob(key + "*"))
print(f"Old tensor files: {old_files}")
print(f"New tensor files: {new_files}")

old_arr = _dequantize_npy_dir(OLD_DIR / "tensors", key)
new_arr = _dequantize_npy_dir(NEW_DIR / "tensors", key)
print(f"Old dequant: shape={old_arr.shape}  range=[{old_arr.min():.4f}, {old_arr.max():.4f}]  std={old_arr.std():.4f}")
print(f"New dequant: shape={new_arr.shape}  range=[{new_arr.min():.4f}, {new_arr.max():.4f}]  std={new_arr.std():.4f}")
print(f"orig vs old diff: max={np.abs(orig.ravel()-old_arr.ravel()).max():.4f}")
print(f"orig vs new diff: max={np.abs(orig.ravel()-new_arr.ravel()).max():.4f}")

# ── Finalized versions ─────────────────────────────────────────────────────────
old_fn = np.load(str(OLD_DIR / "finalized" / f"{key}.npy")).astype(np.float32)
new_fn = np.load(str(NEW_DIR / "finalized" / f"{key}.npy")).astype(np.float32)
print(f"\nOld finalized: shape={old_fn.shape}  range=[{old_fn.min():.4f}, {old_fn.max():.4f}]  std={old_fn.std():.4f}")
print(f"New finalized: shape={new_fn.shape}  range=[{new_fn.min():.4f}, {new_fn.max():.4f}]  std={new_fn.std():.4f}")

# ── k_proj.bias layer 0 ──────────────────────────────────────────────────────
print("\n=== model.layers.0.self_attn.k_proj.bias ===")
key2   = "model__layers__0__self_attn__k_proj__bias"
safek2 = "model.layers.0.self_attn.k_proj.bias"
orig2  = orig_st[safek2]
old_files2 = sorted(f.name for f in (OLD_DIR / "tensors").glob(key2 + "*"))
new_files2 = sorted(f.name for f in (NEW_DIR / "tensors").glob(key2 + "*"))
print(f"BF16 original: shape={orig2.shape}  range=[{orig2.min():.4f}, {orig2.max():.4f}]")
print(f"Old tensor files: {old_files2}")
print(f"New tensor files: {new_files2}")

old_arr2 = _dequantize_npy_dir(OLD_DIR / "tensors", key2)
new_arr2 = _dequantize_npy_dir(NEW_DIR / "tensors", key2)
print(f"Old dequant: range=[{old_arr2.min():.4f}, {old_arr2.max():.4f}]  std={old_arr2.std():.4f}")
print(f"New dequant: range=[{new_arr2.min():.4f}, {new_arr2.max():.4f}]  std={new_arr2.std():.4f}")
print(f"orig vs old diff: max={np.abs(orig2.ravel()-old_arr2.ravel()).max():.6f}")
print(f"orig vs new diff: max={np.abs(orig2.ravel()-new_arr2.ravel()).max():.6f}")
