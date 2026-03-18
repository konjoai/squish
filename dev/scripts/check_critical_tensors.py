#!/usr/bin/env python3
"""Check embed_tokens and lm_head tensors - these are critical for inference."""
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

def snr(orig, approx):
    orig = orig.reshape(approx.shape).astype(np.float32)
    approx = approx.astype(np.float32)
    sig = np.mean(orig ** 2)
    err = np.mean((orig - approx) ** 2)
    return 10 * np.log10(sig / max(err, 1e-30))

def check_tensor(safetensor_key, npy_key):
    orig = orig_st[safetensor_key]
    
    old_files = sorted(f.name for f in (OLD_DIR / "tensors").glob(npy_key + "*"))
    new_files = sorted(f.name for f in (NEW_DIR / "tensors").glob(npy_key + "*"))
    
    old_arr = _dequantize_npy_dir(OLD_DIR / "tensors", npy_key)
    new_arr = _dequantize_npy_dir(NEW_DIR / "tensors", npy_key)
    
    snr_old = snr(orig, old_arr)
    snr_new = snr(orig, new_arr)
    
    # Also check finalized cached versions
    old_fkey = safetensor_key.replace(".", "__")
    new_fkey = safetensor_key.replace(".", "__")
    old_final_path = OLD_DIR / "finalized" / f"{old_fkey}.npy"
    new_final_path = NEW_DIR / "finalized" / f"{new_fkey}.npy"
    
    old_final = np.load(str(old_final_path)) if old_final_path.exists() else None
    new_final = np.load(str(new_final_path)) if new_final_path.exists() else None
    
    snr_old_f = snr(orig, old_final) if old_final is not None else float('nan')
    snr_new_f = snr(orig, new_final) if new_final is not None else float('nan')
    
    print(f"\n  {safetensor_key}")
    print(f"    orig shape:   {orig.shape}")
    print(f"    old files:    {old_files}")
    print(f"    new files:    {new_files}")
    print(f"    old SNR (raw):      {snr_old:8.2f} dB")
    print(f"    new SNR (raw):      {snr_new:8.2f} dB  ({snr_new-snr_old:+.2f} dB)")
    print(f"    old SNR (final16):  {snr_old_f:8.2f} dB")
    print(f"    new SNR (final16):  {snr_new_f:8.2f} dB  ({snr_new_f-snr_old_f:+.2f} dB)")
    
    # Check value ranges
    print(f"    old range: [{old_arr.min():.4f}, {old_arr.max():.4f}]")
    print(f"    new range: [{new_arr.min():.4f}, {new_arr.max():.4f}]")
    if new_final is not None:
        print(f"    final range: [{new_final.min():.4f}, {new_final.max():.4f}]")

print("=== Critical tensors check ===")
check_tensor("model.embed_tokens.weight",       "model__embed_tokens__weight")
check_tensor("lm_head.weight",                  "lm_head__weight")
check_tensor("model.layers.0.mlp.gate_proj.weight", "model__layers__0__mlp__gate_proj__weight")
