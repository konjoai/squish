#!/usr/bin/env python3
"""Direct comparison of finalized embed_tokens tensors between models."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np

OLD_FINAL = Path.home() / "models/Qwen2.5-1.5B-Instruct-bf16-compressed/finalized"
NEW_FINAL = Path.home() / "models/Qwen2.5-1.5B-Instruct-squished-int4-mse/finalized"

# Look at what files exist in each finalized dir
old_names = sorted(p.stem for p in OLD_FINAL.glob("*.npy"))
new_names = sorted(p.stem for p in NEW_FINAL.glob("*.npy"))

print(f"Old finalized: {len(old_names)} files")
print(f"New finalized: {len(new_names)} files")

in_old_not_new = set(old_names) - set(new_names)
in_new_not_old = set(new_names) - set(old_names)
print(f"Only in old: {in_old_not_new}")
print(f"Only in new: {in_new_not_old}")

# Check the specific weights used for inference
# embed_tokens: "model__embed_tokens__weight"
# First few tokens (token IDs 785, 3974, 13876, 38835 for "The quick brown fox")
key = "model__embed_tokens__weight"
old_emb = np.load(str(OLD_FINAL / f"{key}.npy"), mmap_mode='r').astype(np.float32)
new_emb = np.load(str(NEW_FINAL / f"{key}.npy"), mmap_mode='r').astype(np.float32)

print(f"\n=== embed_tokens ===")
print(f"Old shape={old_emb.shape}  dtype=f32  range=[{old_emb.min():.4f}, {old_emb.max():.4f}]  std={old_emb.std():.4f}")
print(f"New shape={new_emb.shape}  dtype=f32  range=[{new_emb.min():.4f}, {new_emb.max():.4f}]  std={new_emb.std():.4f}")

# Check embeddings for our test tokens
for tok_id, tok_str in [(785, "The"), (3974, "quick"), (38835, "fox")]:
    old_v = old_emb[tok_id]
    new_v = new_emb[tok_id]
    diff  = np.abs(old_v - new_v)
    cos   = float(np.dot(old_v, new_v) / (np.linalg.norm(old_v) * np.linalg.norm(new_v)))
    print(f"  token {tok_id} ({tok_str!r}): cos_sim={cos:.4f}  max_diff={diff.max():.4f}  norm_old={np.linalg.norm(old_v):.3f}  norm_new={np.linalg.norm(new_v):.3f}")

# Check lm_head if it exists
lm_key = "lm_head__weight"
if (OLD_FINAL / f"{lm_key}.npy").exists():
    old_lm = np.load(str(OLD_FINAL / f"{lm_key}.npy"), mmap_mode='r').astype(np.float32)
    new_lm = np.load(str(NEW_FINAL / f"{lm_key}.npy"), mmap_mode='r').astype(np.float32)
    print(f"\n=== lm_head ===")
    print(f"Old shape={old_lm.shape}  range=[{old_lm.min():.4f}, {old_lm.max():.4f}]")
    print(f"New shape={new_lm.shape}  range=[{new_lm.min():.4f}, {new_lm.max():.4f}]")
else:
    print(f"\n  lm_head not in finalized (likely tied to embed_tokens)")

# Compare ALL tensors for max magnitude difference
print("\n=== Checking all matching tensors for max value change ===")
big_diffs = []
for name in sorted(set(old_names) & set(new_names)):
    o = np.load(str(OLD_FINAL / f"{name}.npy"), mmap_mode='r').astype(np.float32)
    n = np.load(str(NEW_FINAL / f"{name}.npy"), mmap_mode='r').astype(np.float32)
    if o.shape != n.shape:
        print(f"  SHAPE MISMATCH: {name} old={o.shape} new={n.shape}")
        continue
    max_d = np.abs(o - n).max()
    if max_d > 0.1:  # only report meaningful differences
        big_diffs.append((name, float(max_d), float(o.std()), float(n.std())))

big_diffs.sort(key=lambda x: -x[1])
print(f"  Tensors with max diff > 0.1: {len(big_diffs)}")
for name, md, so, sn in big_diffs[:10]:
    print(f"    {name}: max_diff={md:.4f}  old_std={so:.4f}  new_std={sn:.4f}")
