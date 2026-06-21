#!/usr/bin/env python3
"""Does PyramidKV actually beat uniform SnapKV? — measured on real attention.

PyramidKV's premise: lower transformer layers attend broadly (importance spread
over many tokens) while upper layers concentrate on a few. If true, moving KV
budget from upper to lower layers — at the SAME total memory — retains more of
the attention-importance mass that SnapKV eviction would otherwise drop.

This bench tests that premise directly on a real model: it captures each layer's
K from a long-prompt forward pass, scores token importance with squish's own
SnapKV formula, then compares the fraction of total importance retained by
`uniform` vs `pyramid` budgets (identical total). No synthetic data.

Run:
  ~/squish/.venv/bin/python benchmarks/perf/pyramidkv_retention_bench.py \
      [~/models/Qwen2.5-1.5B-Instruct-int4]
"""
from __future__ import annotations

import os
import sys

import mlx.core as mx
import numpy as np

from squish.kv.kv_cache import _compute_layer_budgets

_MODEL = os.path.expanduser("~/models/Qwen2.5-1.5B-Instruct-int4")
SNAP_WINDOW = 32
PER_LAYER_BUDGET = 512   # uniform budget; pyramid redistributes the same total
PYRAMID_BETA = 0.5
PROMPT_TOKENS = 2048


def _importance(k_f32: np.ndarray, snap_window: int) -> np.ndarray:
    """SnapKV importance per token position — mirrors `_snap_evict` exactly.

    k_f32: (n_heads, n_tokens, head_dim) → importance: (n_tokens,)
    """
    head_dim = k_f32.shape[-1]
    q = k_f32[:, -snap_window:, :]
    logits = np.einsum("nhd, nTd -> nhT", q, k_f32) * (1.0 / (head_dim ** 0.5))
    exp_l = np.exp(logits - logits.max(axis=-1, keepdims=True))
    attn = exp_l / exp_l.sum(axis=-1, keepdims=True)
    return attn.sum(axis=(0, 1))


def _retained_mass(importance: np.ndarray, budget: int) -> float:
    """Fraction of total importance kept by retaining the top-`budget` tokens."""
    n = importance.shape[0]
    if budget >= n:
        return 1.0
    keep = np.argsort(-importance)[:budget]
    return float(importance[keep].sum() / (importance.sum() + 1e-9))


def _varied_prompt(tok, n_tokens: int) -> list[int]:
    """A token-diverse long prompt so attention has real structure to evict."""
    import random
    clauses = [
        "the distributed scheduler reconciles partitions across availability zones",
        "a lock-free ring buffer absorbs bursts without backpressure stalls",
        "the migration replays the write-ahead log under a fenced epoch token",
        "telemetry spans propagate through the gateway with sampled baggage",
        "the planner rewrites the predicate before the index is consulted",
        "garbage collection pauses are amortized across the eviction window",
        "the consensus group elects a leader once the quorum lease expires",
        "vectorized kernels fuse the projection into a single memory pass",
        "the rate limiter sheds load when the token bucket drains to zero",
        "checkpoint snapshots are deduplicated against the content address store",
    ]
    rng = random.Random(17)
    parts: list[str] = ["Engineering postmortem. "]
    ids = tok.encode("".join(parts))
    while len(ids) < n_tokens:
        sentence = ", ".join(rng.choice(clauses) for _ in range(rng.randint(3, 6)))
        parts.append(sentence.capitalize() + ". ")
        ids = tok.encode("".join(parts))
    return ids[:n_tokens]


def _total_retained(per_layer_imp, budgets) -> float:
    kept = total = 0.0
    for imp, b in zip(per_layer_imp, budgets):
        tot = float(imp.sum())
        kept += _retained_mass(imp, b) * tot
        total += tot
    return kept / (total + 1e-9)


def main() -> int:
    model_path = sys.argv[1] if len(sys.argv) > 1 else _MODEL
    n_tokens = int(sys.argv[2]) if len(sys.argv) > 2 else PROMPT_TOKENS
    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache

    model, tok = load(model_path)
    ids = _varied_prompt(tok, n_tokens)
    caches = make_prompt_cache(model)
    logits = model(mx.array([ids]), cache=caches)
    mx.eval(logits)

    n_layers = len(caches)
    per_layer_imp = []
    for c in caches:
        keys, offset = getattr(c, "keys", None), getattr(c, "offset", None)
        if keys is None or offset is None:
            continue
        k = np.array(keys[0, :, :offset, :].astype(mx.float32))
        per_layer_imp.append(_importance(k, SNAP_WINDOW))

    print(f"model={os.path.basename(model_path)}  layers={n_layers}  "
          f"prompt={len(ids)} varied tok  β={PYRAMID_BETA}\n")
    print(f"{'budget/layer':>13}{'uniform':>10}{'pyramid':>10}{'rev-pyramid':>13}"
          f"{'Δpyr(pp)':>10}")
    any_win = False
    for budget in (512, 256, 128, 64, 32):
        uni = _compute_layer_budgets(n_layers, budget, "uniform")
        pyr = _compute_layer_budgets(n_layers, budget, "pyramid", PYRAMID_BETA,
                                     floor=SNAP_WINDOW)
        rev = pyr[::-1]  # more budget to UPPER layers (opposite hypothesis)
        u = _total_retained(per_layer_imp, uni)
        p = _total_retained(per_layer_imp, pyr)
        r = _total_retained(per_layer_imp, rev)
        d = (p - u) * 100
        any_win = any_win or d > 0.05
        print(f"{budget:>13}{u * 100:>9.2f}%{p * 100:>9.2f}%{r * 100:>12.2f}%"
              f"{d:>+10.2f}")
    verdict = ("helps in some regime" if any_win
               else "does NOT beat uniform on this model/workload")
    print(f"\nverdict: pyramid {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
