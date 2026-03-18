#!/usr/bin/env python3
"""Load only the new model and test a forward pass."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import mlx.core as mx
import numpy as np
from squish.quant.compressed_loader import load_compressed_model

MODEL_DIR    = str(Path.home() / "models/Qwen2.5-1.5B-Instruct-bf16")
NEW_COMP_DIR = str(Path.home() / "models/Qwen2.5-1.5B-Instruct-squished-int4-mse")

print("Loading NEW asymmetric+MSE INT4 model (standalone)...")
model, tok, stats = load_compressed_model(
    model_dir=MODEL_DIR, npz_path=NEW_COMP_DIR, verbose=True, return_stats=True)
print(f"  loader={stats.get('loader')}")

# Also load BF16 reference for comparison
BF16_DIR = str(Path.home() / "models/Qwen2.5-1.5B-Instruct-bf16")

prompt = "The quick brown fox"
tokens = tok.encode(prompt, return_tensors="np")
input_ids = mx.array(tokens, dtype=mx.int32)

print(f"\nPrompt: '{prompt}'")
print(f"Token ids: {input_ids.tolist()}")

out = model(input_ids)
mx.eval(out)

# Handle tuple/raw logits
def get_logits(x):
    if isinstance(x, tuple): x = x[0]
    if x.ndim == 3: return np.array(x[0, -1].astype(mx.float32))
    return np.array(x[-1].astype(mx.float32))

logits = get_logits(out)
print(f"\nLogit stats: min={logits.min():.3f} max={logits.max():.3f} mean={logits.mean():.3f}")
print(f"Any NaN? {np.isnan(logits).any()}  Any Inf? {np.isinf(logits).any()}")

top5 = np.argsort(logits)[-5:][::-1]
print(f"\nTop-5 next tokens:")
for tid in top5:
    print(f"  {tid:6d}: {tok.decode([tid])!r:20s}  logit={logits[tid]:.3f}")
