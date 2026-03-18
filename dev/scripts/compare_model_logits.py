#!/usr/bin/env python3
"""Quick sanity check: load both models and run a forward pass, compare logits."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import mlx.core as mx
import numpy as np
from squish.quant.compressed_loader import load_compressed_model

MODEL_DIR    = str(Path.home() / "models/Qwen2.5-1.5B-Instruct-bf16")
OLD_COMP_DIR = str(Path.home() / "models/Qwen2.5-1.5B-Instruct-bf16-compressed")
NEW_COMP_DIR = str(Path.home() / "models/Qwen2.5-1.5B-Instruct-squished-int4-mse")

print("Loading OLD symmetric INT4 model...")
old_model, old_tok, old_stats = load_compressed_model(
    model_dir=MODEL_DIR, npz_path=OLD_COMP_DIR, verbose=False, return_stats=True)
print(f"  loader={old_stats.get('loader')}  decompression={old_stats.get('decompression_time_s', 0):.1f}s")

print("\nLoading NEW asymmetric+MSE INT4 model...")
new_model, new_tok, new_stats = load_compressed_model(
    model_dir=MODEL_DIR, npz_path=NEW_COMP_DIR, verbose=False, return_stats=True)
print(f"  loader={new_stats.get('loader')}  decompression={new_stats.get('decompression_time_s', 0):.1f}s")

# Run a forward pass with a simple prompt
prompt = "The quick brown fox"
tokens = old_tok.encode(prompt, return_tensors="np")
input_ids = mx.array(tokens, dtype=mx.int32)

print(f"\nTest prompt: '{prompt}'")
print(f"Token ids: {input_ids.tolist()}")

old_out = old_model(input_ids)
mx.eval(old_out)
new_out = new_model(input_ids)
mx.eval(new_out)

# Handle tuple output (logits, ...) or raw logits
def get_logits(out):
    if isinstance(out, tuple):
        out = out[0]
    # out shape: (batch, seq, vocab) or (seq, vocab)
    if out.ndim == 3:
        return np.array(out[0, -1].astype(mx.float32))
    return np.array(out[-1].astype(mx.float32))

old_logits = get_logits(old_out)
new_logits = get_logits(new_out)

# Top-5 tokens from each model
old_top5 = np.argsort(old_logits)[-5:][::-1]
new_top5 = np.argsort(new_logits)[-5:][::-1]

print(f"\nOld model top-5 next tokens:")
for tok_id in old_top5:
    print(f"  {tok_id:6d}: {old_tok.decode([tok_id])!r:20s}  logit={old_logits[tok_id]:.3f}")

print(f"\nNew model top-5 next tokens:")
for tok_id in new_top5:
    print(f"  {tok_id:6d}: {new_tok.decode([tok_id])!r:20s}  logit={new_logits[tok_id]:.3f}")

# Compat
cos_sim = float(np.dot(old_logits, new_logits) / 
                (np.linalg.norm(old_logits) * np.linalg.norm(new_logits)))
print(f"\nLogit cosine similarity: {cos_sim:.4f}  (1.0 = identical, <0.99 = concerning)")
