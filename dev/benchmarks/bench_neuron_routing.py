#!/usr/bin/env python3
"""bench_neuron_routing.py — Phase 10A memory-bandwidth microbenchmark.

Measures tokens/sec and relative DRAM traffic reduction with and without
hot/cold neuron routing on a synthetic Qwen-scale MLP.

Usage::

    python dev/benchmarks/bench_neuron_routing.py [--n-layers 32] [--hidden 4096]
        [--ffn-dim 11008] [--seq-len 512] [--n-tokens 2048]
        [--hot-fraction 0.20] [--out dev/results/neuron_routing_bench.json]

The benchmark synthesises random weight matrices of the same shape as a
7 B-class model FFN block and runs N forward passes through both the dense
(unrouted) and hot/cold-split (routed) paths.  Because there is no real DRAM
pressure in a Python benchmark, "bandwidth reduction" is inferred analytically
from the hot_fraction and cold_fraction: only ``hot_fraction`` of the neurons
require GPU-resident weights; the rest can in principle stay on CPU.

Results are saved to *--out* in JSON format and also printed in a compact
table.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List

import numpy as np

# ---------------------------------------------------------------------------
# Make sure squish package is importable from repo root
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _dense_swiglu(x: np.ndarray, gate_w: np.ndarray,
                  up_w: np.ndarray, down_w: np.ndarray) -> np.ndarray:
    """Dense SwiGLU forward pass: silu(x @ gate_w.T) * (x @ up_w.T) @ down_w.T."""
    gate = x @ gate_w.T
    silu_gate = gate * (1.0 / (1.0 + np.exp(-gate)))
    up_out = x @ up_w.T
    hidden = silu_gate * up_out
    return hidden @ down_w.T


def _routed_swiglu(x: np.ndarray, gate_w: np.ndarray, up_w: np.ndarray,
                   down_w: np.ndarray, hot_idx: np.ndarray,
                   cold_idx: np.ndarray) -> np.ndarray:
    """Neuron-routed SwiGLU: compute hot and cold tiers independently."""
    g_hot, u_hot, d_hot = gate_w[hot_idx], up_w[hot_idx], down_w[:, hot_idx]
    g_cold, u_cold, d_cold = gate_w[cold_idx], up_w[cold_idx], down_w[:, cold_idx]

    def _tier(g_w, u_w, d_w):
        g = x @ g_w.T
        g = g * (1.0 / (1.0 + np.exp(-g)))
        u = x @ u_w.T
        return (g * u) @ d_w.T

    return _tier(g_hot, u_hot, d_hot) + _tier(g_cold, u_cold, d_cold)


def _bench_layer(gate_w, up_w, down_w, hot_idx, cold_idx,
                 n_tokens: int, seq_len: int) -> Dict[str, Any]:
    """Time one FFN layer (dense vs routed) over n_tokens // seq_len steps."""
    steps = max(1, n_tokens // seq_len)
    x = np.random.randn(seq_len, gate_w.shape[1]).astype(np.float32)

    # Dense
    t0 = time.perf_counter()
    for _ in range(steps):
        _dense_swiglu(x, gate_w, up_w, down_w)
    dense_s = time.perf_counter() - t0

    # Routed
    t0 = time.perf_counter()
    for _ in range(steps):
        _routed_swiglu(x, gate_w, up_w, down_w, hot_idx, cold_idx)
    routed_s = time.perf_counter() - t0

    total_tokens = steps * seq_len
    return {
        "dense_tps":  round(total_tokens / max(dense_s, 1e-9)),
        "routed_tps": round(total_tokens / max(routed_s, 1e-9)),
        "dense_s":    round(dense_s, 4),
        "routed_s":   round(routed_s, 4),
        "speedup":    round(dense_s / max(routed_s, 1e-9), 3),
    }


def run_bench(
    n_layers: int = 4,
    hidden: int = 4096,
    ffn_dim: int = 11008,
    seq_len: int = 64,
    n_tokens: int = 1024,
    hot_fraction: float = 0.20,
) -> Dict[str, Any]:
    """Run the neuron-routing microbenchmark and return structured results."""
    rng = np.random.RandomState(42)

    hot_n = max(1, int(ffn_dim * hot_fraction))
    cold_n = ffn_dim - hot_n
    # Simulate a fixed profile where neurons 0..hot_n-1 are "hot"
    hot_idx = np.arange(hot_n, dtype=np.int64)
    cold_idx = np.arange(hot_n, ffn_dim, dtype=np.int64)

    layer_results: List[Dict[str, Any]] = []
    for layer_id in range(n_layers):
        gate_w = rng.randn(ffn_dim, hidden).astype(np.float32) * 0.01
        up_w   = rng.randn(ffn_dim, hidden).astype(np.float32) * 0.01
        down_w = rng.randn(hidden, ffn_dim).astype(np.float32) * 0.01
        res = _bench_layer(gate_w, up_w, down_w, hot_idx, cold_idx, n_tokens, seq_len)
        res["layer"] = layer_id
        layer_results.append(res)

    avg_speedup = float(np.mean([r["speedup"] for r in layer_results]))
    # Analytical DRAM reduction: routed loads hot+cold weights but avoids
    # loading cold weights into the GPU compute tier.
    hot_ratio = hot_n / ffn_dim
    cold_ratio = cold_n / ffn_dim
    inferred_bw_reduction = 1.0 - hot_ratio  # cold tier stays on CPU

    return {
        "benchmark": "neuron_routing",
        "config": {
            "n_layers": n_layers,
            "hidden": hidden,
            "ffn_dim": ffn_dim,
            "seq_len": seq_len,
            "n_tokens": n_tokens,
            "hot_fraction": hot_fraction,
            "hot_neurons": int(hot_n),
            "cold_neurons": int(cold_n),
        },
        "results": layer_results,
        "summary": {
            "avg_speedup": round(avg_speedup, 3),
            "inferred_dram_reduction_pct": round(inferred_bw_reduction * 100, 1),
            "note": "speedup vs dense on CPU-numpy; real M-series gains depend on MLX Metal dispatch",
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 10A neuron-routing microbenchmark")
    ap.add_argument("--n-layers",      type=int,   default=4,     help="number of FFN layers to bench")
    ap.add_argument("--hidden",        type=int,   default=512,   help="model hidden dim (use 4096 for 7B)")
    ap.add_argument("--ffn-dim",       type=int,   default=1024,  help="FFN intermediate dim (use 11008 for 7B)")
    ap.add_argument("--seq-len",       type=int,   default=32,    help="token sequence length per step")
    ap.add_argument("--n-tokens",      type=int,   default=256,   help="total tokens to process")
    ap.add_argument("--hot-fraction",  type=float, default=0.20,  help="hot neuron fraction")
    ap.add_argument("--out", default="dev/results/neuron_routing_bench.json",
                    help="output JSON path")
    args = ap.parse_args()

    print(f"[bench_neuron_routing] layers={args.n_layers} hidden={args.hidden} "
          f"ffn={args.ffn_dim} hot={args.hot_fraction:.0%}")
    result = run_bench(
        n_layers=args.n_layers,
        hidden=args.hidden,
        ffn_dim=args.ffn_dim,
        seq_len=args.seq_len,
        n_tokens=args.n_tokens,
        hot_fraction=args.hot_fraction,
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[bench_neuron_routing] saved → {args.out}")

    s = result["summary"]
    print(f"  avg speedup  : {s['avg_speedup']:.3f}×")
    print(f"  DRAM reduc.  : {s['inferred_dram_reduction_pct']:.1f}%  (analytical)")
    print(f"  note         : {s['note']}")


if __name__ == "__main__":
    main()
