#!/usr/bin/env python3
"""Check group sizes and loading details for old vs new models."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np

old_d = Path.home() / "models/Qwen2.5-1.5B-Instruct-bf16-compressed/tensors"
new_d = Path.home() / "models/Qwen2.5-1.5B-Instruct-squished-int4-mse/tensors"

# Count passthrough vs INT4 tensors
old_pt  = len(list(old_d.glob("*__pt.npy")))
old_q4  = len(list(old_d.glob("*__q4.npy")))    # symmetric
old_q4a = len(list(old_d.glob("*__q4a.npy")))   # asymmetric
new_pt  = len(list(new_d.glob("*__pt.npy")))
new_q4  = len(list(new_d.glob("*__q4.npy")))
new_q4a = len(list(new_d.glob("*__q4a.npy")))

print("=== Tensor counts ===")
print(f"Old model:  pt={old_pt}  q4(sym)={old_q4}  q4a(asym)={old_q4a}")
print(f"New model:  pt={new_pt}  q4(sym)={new_q4}  q4a(asym)={new_q4a}")

# Check a quantized layer's key tensor: scales shape → determines group size
key_q4  = "model__layers__0__mlp__gate_proj__weight"  # has __q4a in new, __q4 in old
old_q  = np.load(str(old_d / (key_q4 + "__q4.npy")))
old_s  = np.load(str(old_d / (key_q4 + "__s4.npy")))
new_qa = np.load(str(new_d / (key_q4 + "__q4a.npy")))
new_sa = np.load(str(new_d / (key_q4 + "__s4a.npy")))
new_za = np.load(str(new_d / (key_q4 + "__z4a.npy")))

n_rows = 8960
n_cols = 1536
old_gs = n_cols * 2 // old_q.shape[1] if old_q.ndim == 2 else "?"
new_gs = n_cols * 2 // new_qa.shape[1] if new_qa.ndim == 2 else "?"
old_scales_per_row = old_s.shape[1] if old_s.ndim == 2 else old_s.size // old_s.shape[0]
new_scales_per_row = new_sa.shape[1] if new_sa.ndim == 2 else new_sa.size // new_sa.shape[0]

print(f"\n=== gate_proj.weight ({n_rows}x{n_cols}) ===")
print(f"Old  q4  shape: {old_q.shape}  dtype={old_q.dtype}  => group_size={old_gs}")
print(f"Old  s4  shape: {old_s.shape}  dtype={old_s.dtype}")
print(f"New  q4a shape: {new_qa.shape} dtype={new_qa.dtype} => group_size={new_gs}")
print(f"New  s4a shape: {new_sa.shape} dtype={new_sa.dtype}")
print(f"New  z4a shape: {new_za.shape} dtype={new_za.dtype}")

# Check a passthrough tensor to verify its shape
key_pt = "model__layers__0__self_attn__q_proj__weight"
old_pt_arr = np.load(str(old_d / (key_pt + "__pt.npy")))
new_pt_arr = np.load(str(new_d / (key_pt + "__pt.npy")))
print(f"\n=== q_proj (passthrough) ===")
print(f"Old pt shape: {old_pt_arr.shape}  dtype={old_pt_arr.dtype}")
print(f"New pt shape: {new_pt_arr.shape}  dtype={new_pt_arr.dtype}")

# Quick dequant sanity check
from squish.io.loader_utils import _dequantize_npy
arr_old = _dequantize_npy(old_d, key_q4)
arr_new = _dequantize_npy(new_d, key_q4)
print(f"\n=== Dequantized gate_proj ===")
print(f"Old reconstructed shape: {arr_old.shape}  range: [{arr_old.min():.4f}, {arr_old.max():.4f}]")
print(f"New reconstructed shape: {arr_new.shape}  range: [{arr_new.min():.4f}, {arr_new.max():.4f}]")
