#!/usr/bin/env python3
"""Diagnose regression: compare SNR of old symmetric vs new asymmetric+MSE INT4."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import torch
from safetensors.torch import load_file as st_load
from squish.io.loader_utils import _dequantize_npy

old_d = Path.home() / "models/Qwen2.5-1.5B-Instruct-bf16-compressed/tensors"
new_d = Path.home() / "models/Qwen2.5-1.5B-Instruct-squished-int4-mse/tensors"
bf16_model = Path.home() / "models/Qwen2.5-1.5B-Instruct-bf16/model.safetensors"

# Load BF16 reference weights (use torch to handle bfloat16)
orig_st_torch = st_load(str(bf16_model))
orig_st = {k: v.float().numpy() for k, v in orig_st_torch.items()}

test_keys = [
    ("model.layers.0.self_attn.q_proj.weight",    "model__layers__0__self_attn__q_proj__weight"),
    ("model.layers.0.mlp.gate_proj.weight",        "model__layers__0__mlp__gate_proj__weight"),
    ("model.layers.14.self_attn.v_proj.weight",    "model__layers__14__self_attn__v_proj__weight"),
]


def snr(orig, approx):
    orig = orig.reshape(approx.shape).astype(np.float32)
    approx = approx.astype(np.float32)
    sig = np.mean(orig ** 2)
    err = np.mean((orig - approx) ** 2)
    return 10 * np.log10(sig / max(err, 1e-30))


print(f"{'Tensor':55s}  {'Old (sym)':10s}  {'New (asym+MSE)':15s}  {'Delta':8s}")
print("-" * 100)

for safetensor_key, npy_key in test_keys:
    orig = orig_st[safetensor_key].astype(np.float32)

    # Check what files exist in each dir
    old_files = sorted(f.name for f in old_d.glob(npy_key + "*"))
    new_files = sorted(f.name for f in new_d.glob(npy_key + "*"))

    old_arr = _dequantize_npy(old_d, npy_key)
    new_arr = _dequantize_npy(new_d, npy_key)

    snr_old = snr(orig, old_arr)
    snr_new = snr(orig, new_arr)
    delta   = snr_new - snr_old

    print(f"  {safetensor_key:53s}  {snr_old:8.2f} dB  {snr_new:13.2f} dB  {delta:+.2f} dB")
    print(f"    old files: {old_files}")
    print(f"    new files: {new_files}")
    print(f"    old range: [{old_arr.min():.4f}, {old_arr.max():.4f}]  shape={old_arr.shape}")
    print(f"    new range: [{new_arr.min():.4f}, {new_arr.max():.4f}]  shape={new_arr.shape}")
    print()
