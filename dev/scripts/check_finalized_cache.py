#!/usr/bin/env python3
"""Check if finalized cache matches fresh dequantization for the new model."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
from squish.quant.compressed_loader import _dequantize_npy_dir

NEW_DIR     = Path.home() / "models/Qwen2.5-1.5B-Instruct-squished-int4-mse"
tensor_dir  = NEW_DIR / "tensors"
final_dir   = NEW_DIR / "finalized"

# Pick a quantized tensor (not passthrough)
sk  = "model__layers__0__mlp__gate_proj__weight"
fname = "model.layers.0.mlp.gate_proj.weight"
fkey  = fname.replace(".", "__")  # same as what finalized uses

print(f"Comparing: {fname}")

# Fresh dequantize from INT4 tensors
fresh = _dequantize_npy_dir(tensor_dir, sk)
print(f"  Fresh:     shape={fresh.shape}  dtype={fresh.dtype}  range=[{fresh.min():.4f}, {fresh.max():.4f}]")

# Load from finalized cache
final_path = final_dir / f"{fkey}.npy"
if not final_path.exists():
    print(f"  Finalized not found: {final_path}")
else:
    cached = np.load(str(final_path))
    print(f"  Cached:    shape={cached.shape}  dtype={cached.dtype}  range=[{cached.min():.4f}, {cached.max():.4f}]")
    
    # Compare
    diff   = np.abs(fresh.astype(np.float32) - cached.astype(np.float32))
    max_diff = diff.max()
    mean_diff = diff.mean()
    print(f"  Max diff: {max_diff:.6f}  Mean diff: {mean_diff:.6f}")
    if max_diff > 0.01:
        print("  *** MISMATCH DETECTED — finalized cache may be stale/wrong ***")
    else:
        print("  OK — finalized matches fresh dequantization (within float16 precision)")

# Also check a passthrough tensor
sk_pt  = "model__layers__0__self_attn__q_proj__weight"
fname_pt = "model.layers.0.self_attn.q_proj.weight"
fkey_pt  = fname_pt.replace(".", "__")
fresh_pt = _dequantize_npy_dir(tensor_dir, sk_pt)
final_pt = np.load(str(final_dir / f"{fkey_pt}.npy"))
diff_pt  = np.abs(fresh_pt.astype(np.float32) - final_pt.astype(np.float32))
print(f"\n  Passthrough q_proj: max_diff={diff_pt.max():.6f} (should be <0.01)")
