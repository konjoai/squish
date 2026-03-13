#!/usr/bin/env python3
"""bench_wave10.py — Wave 10 phase-gate benchmark suite.

Orchestrates all Wave 10 (Apple Silicon Memory Bandwidth) microbenchmarks:

  Phase 10A — Neuron Routing   (bench_neuron_routing.py)
  Phase 10B — Metal Fusion     (bench_metal_fusion.py)

Saves combined results to ``dev/results/wave10_bench.json`` and prints a
summary table.  Pass ``--dry-run`` to skip writing and just print results.

Usage::

    python dev/benchmarks/bench_wave10.py [--dry-run]
        [--out dev/results/wave10_bench.json]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

_BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
if _BENCH_DIR not in sys.path:
    sys.path.insert(0, _BENCH_DIR)

from bench_metal_fusion import run_bench as _bench_metal_fusion   # noqa: E402
from bench_neuron_routing import run_bench as _bench_neuron_routing  # noqa: E402


def run_wave10(
    # neuron routing params
    nr_n_layers: int = 4,
    nr_hidden: int = 512,
    nr_ffn_dim: int = 1024,
    nr_seq_len: int = 32,
    nr_n_tokens: int = 256,
    nr_hot_fraction: float = 0.20,
    # metal fusion params
    mf_head_dim: int = 128,
    mf_n_heads: int = 8,
    mf_ffn_dim: int = 512,
    mf_seq_lens=None,
    mf_warmup: int = 2,
    mf_reps: int = 5,
) -> Dict[str, Any]:
    """Run all Wave 10 benchmarks and return combined result dict."""
    if mf_seq_lens is None:
        mf_seq_lens = [128, 1024]

    print("[wave10] Running Phase 10A — neuron routing …")
    nr_result = _bench_neuron_routing(
        n_layers=nr_n_layers,
        hidden=nr_hidden,
        ffn_dim=nr_ffn_dim,
        seq_len=nr_seq_len,
        n_tokens=nr_n_tokens,
        hot_fraction=nr_hot_fraction,
    )

    print("[wave10] Running Phase 10B — metal fusion …")
    mf_result = _bench_metal_fusion(
        head_dim=mf_head_dim,
        n_heads=mf_n_heads,
        ffn_dim=mf_ffn_dim,
        seq_lens=mf_seq_lens,
        warmup=mf_warmup,
        reps=mf_reps,
    )

    return {
        "wave": 10,
        "phases": {
            "10A_neuron_routing": nr_result,
            "10B_metal_fusion": mf_result,
        },
        "summary": {
            "neuron_routing_avg_speedup": nr_result["summary"]["avg_speedup"],
            "neuron_routing_dram_reduction_pct": nr_result["summary"]["inferred_dram_reduction_pct"],
            "metal_fusion_avg_speedup": mf_result["summary"]["avg_speedup"],
            "metal_available": mf_result["metal_available"],
        },
    }


def _print_table(result: Dict[str, Any]) -> None:
    s = result["summary"]
    print()
    print("┌─────────────────────────────────────────────────────────────────┐")
    print("│                    Wave 10 Benchmark Summary                    │")
    print("├──────────────────────────────┬──────────────────────────────────┤")
    print(f"│ Neuron routing avg speedup   │ {s['neuron_routing_avg_speedup']:.3f}×"
          + " " * max(0, 26 - len(f"{s['neuron_routing_avg_speedup']:.3f}×")) + "│")
    print(f"│ Neuron routing DRAM reduc.   │ {s['neuron_routing_dram_reduction_pct']:.1f}% (analytical)"
          + " " * max(0, 14 - len(f"{s['neuron_routing_dram_reduction_pct']:.1f}%")) + "│")
    print(f"│ Metal fusion avg speedup     │ {s['metal_fusion_avg_speedup']:.3f}×"
          + " " * max(0, 26 - len(f"{s['metal_fusion_avg_speedup']:.3f}×")) + "│")
    metal_str = "yes (M-series Metal)" if s["metal_available"] else "no (numpy fallback)"
    print(f"│ Metal hardware               │ {metal_str}"
          + " " * max(0, 34 - len(metal_str)) + "│")
    print("└──────────────────────────────┴──────────────────────────────────┘")
    print()


def main() -> None:
    ap = argparse.ArgumentParser(description="Wave 10 phase-gate benchmark suite")
    ap.add_argument("--dry-run", action="store_true",
                    help="print results but do not write JSON file")
    ap.add_argument("--out", default="dev/results/wave10_bench.json",
                    help="output JSON file path")
    # Expose a subset of tuning knobs for quick experimentation
    ap.add_argument("--nr-hidden",       type=int,   default=512)
    ap.add_argument("--nr-ffn-dim",      type=int,   default=1024)
    ap.add_argument("--nr-hot-fraction", type=float, default=0.20)
    ap.add_argument("--mf-head-dim",     type=int,   default=128)
    ap.add_argument("--mf-ffn-dim",      type=int,   default=512)
    ap.add_argument("--mf-seq-lens",     type=int,   nargs="+", default=[128, 1024])
    args = ap.parse_args()

    result = run_wave10(
        nr_hidden=args.nr_hidden,
        nr_ffn_dim=args.nr_ffn_dim,
        nr_hot_fraction=args.nr_hot_fraction,
        mf_head_dim=args.mf_head_dim,
        mf_ffn_dim=args.mf_ffn_dim,
        mf_seq_lens=args.mf_seq_lens,
    )

    _print_table(result)

    if not args.dry_run:
        out_path = os.path.abspath(args.out)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"[wave10] results saved → {out_path}")
    else:
        print("[wave10] --dry-run: results not written to disk")


if __name__ == "__main__":
    main()
