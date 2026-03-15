#!/usr/bin/env python3
"""bench_moe_lookahead.py — Phase 14 MoE Lookahead Router benchmark.

Simulates the MoELookaheadRouter on synthetic hidden-state sequences that
mimic the token-by-token decode pattern of DeepSeek-Coder-V2-Lite
(16B total, 2.4B active, 64 experts, top-2 routing).

Metrics collected
─────────────────
- Prefetch hit rate (%) across rolling token windows
- Per-step router latency (µs) with and without look-ahead EMA
- Hit rate at three synthetic regimes:
    * flat:     all tokens identical      → near-100% hit rate
    * random:   i.i.d. Gaussian           → ~25% (chance level for top-2/64)
    * drifting: slow EMA-like distribution shift → mid-range hit rate

Results are printed as a table and saved to
``dev/results/moe_lookahead_bench.json``.

Usage::

    python dev/benchmarks/bench_moe_lookahead.py
    python dev/benchmarks/bench_moe_lookahead.py --n-tokens 500 --n-experts 64 --top-k 2

"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from squish.moe.moe_lookahead import MoELookaheadConfig, MoELookaheadRouter


# ---------------------------------------------------------------------------
# Synthetic regimes
# ---------------------------------------------------------------------------

def _hidden_flat(n_tokens: int, hidden_dim: int, rng) -> "list[Any]":
    """All tokens produce nearly identical hidden states."""
    base = rng.standard_normal(hidden_dim).astype("float32")
    return [base + rng.standard_normal(hidden_dim).astype("float32") * 0.01
            for _ in range(n_tokens)]


def _hidden_random(n_tokens: int, hidden_dim: int, rng) -> "list[Any]":
    """Fully i.i.d. Gaussian — no temporal structure."""
    return [rng.standard_normal(hidden_dim).astype("float32") for _ in range(n_tokens)]


def _hidden_drifting(n_tokens: int, hidden_dim: int, rng, drift: float = 0.05) -> "list[Any]":
    """Slowly drifting centroid — represents realistic long-context generation."""
    import numpy as np
    state = rng.standard_normal(hidden_dim).astype("float32")
    result = []
    for _ in range(n_tokens):
        state = (1 - drift) * state + drift * rng.standard_normal(hidden_dim).astype("float32")
        result.append(state.copy())
    return result


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

@dataclass
class BenchResult:
    regime:        str
    n_tokens:      int
    hit_rate:      float           # fraction 0–1
    latency_us:    float           # per-token µs (with lookahead)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "regime":     self.regime,
            "n_tokens":   self.n_tokens,
            "hit_rate_%": round(self.hit_rate * 100, 2),
            "latency_us": round(self.latency_us, 3),
        }


def _run_regime(
    name: str,
    hidden_states: "list[Any]",
    config: MoELookaheadConfig,
) -> BenchResult:
    """Feed a synthetic hidden-state sequence through the router and collect metrics."""
    import numpy as np

    router = MoELookaheadRouter(config)
    latencies = []

    for h in hidden_states:
        h2d = h.reshape(1, -1)

        t0 = time.perf_counter()
        # Prefetch step (call before route to set up pending prefetch)
        router.prefetch_set(h2d)
        # Route step (evaluates hit rate against previous prefetch, updates EMA)
        router.route(h2d)
        t1 = time.perf_counter()

        latencies.append((t1 - t0) * 1e6)

    stats = router.stats()
    hit_rate   = stats["hit_rate"] if stats["hit_rate"] >= 0 else 0.0
    latency_us = sum(latencies) / len(latencies) if latencies else 0.0

    return BenchResult(
        regime=name,
        n_tokens=len(hidden_states),
        hit_rate=hit_rate,
        latency_us=latency_us,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import numpy as np

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-tokens",   type=int, default=200, metavar="N",
                    help="Tokens per regime (default 200)")
    ap.add_argument("--n-experts",  type=int, default=64,  metavar="E",
                    help="Total MoE experts (default 64, matches DeepSeek-Coder-V2-Lite)")
    ap.add_argument("--top-k",      type=int, default=2,   metavar="K",
                    help="Experts per token (default 2)")
    ap.add_argument("--hidden-dim", type=int, default=512, metavar="D",
                    help="Hidden state dimension (default 512)")
    ap.add_argument("--lookahead",  type=int, default=3,   metavar="L",
                    help="Lookahead horizon (default 3)")
    ap.add_argument("--output",     default="dev/results/moe_lookahead_bench.json",
                    help="Output JSON path")
    args = ap.parse_args()

    cfg = MoELookaheadConfig(
        n_experts=args.n_experts,
        top_k=args.top_k,
        hidden_dim=args.hidden_dim,
        lookahead_steps=args.lookahead,
    )

    rng = np.random.default_rng(42)

    regimes = [
        ("flat",     _hidden_flat(args.n_tokens,     args.hidden_dim, rng)),
        ("random",   _hidden_random(args.n_tokens,   args.hidden_dim, rng)),
        ("drifting", _hidden_drifting(args.n_tokens, args.hidden_dim, rng)),
    ]

    results: List[BenchResult] = []
    print()
    print(f"  MoE Lookahead Benchmark  "
          f"(experts={args.n_experts}, top_k={args.top_k}, "
          f"hidden={args.hidden_dim}, lookahead={args.lookahead}, "
          f"n_tokens={args.n_tokens})")
    print(f"  {'Regime':<12} {'Hit rate':>10}  {'Latency µs/tok':>16}")
    print(f"  {'─'*12} {'─'*10}  {'─'*16}")

    for name, hidden in regimes:
        r = _run_regime(name, hidden, cfg)
        results.append(r)
        print(f"  {r.regime:<12} {r.hit_rate*100:>9.1f}%  {r.latency_us:>14.2f}")

    print()

    # Save results
    out_path = os.path.join(_REPO_ROOT, args.output)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    payload = {
        "config": {
            "n_experts":      args.n_experts,
            "top_k":          args.top_k,
            "hidden_dim":     args.hidden_dim,
            "lookahead_steps": args.lookahead,
            "n_tokens":       args.n_tokens,
        },
        "results": [r.as_dict() for r in results],
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  Saved → {args.output}")
    print()


if __name__ == "__main__":
    main()
