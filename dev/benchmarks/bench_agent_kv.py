#!/usr/bin/env python3
"""bench_agent_kv.py — Phase 13A AgentKV memory benchmark.

Analytically measures peak RAM reduction from using AgentKVCache (INT2 history tier)
versus standard FP16 KV cache at context lengths 4K / 8K / 16K / 32K for a
Qwen2.5-14B-scale model.

Because real DRAM pressure is not measurable in a CPU-only Python benchmark,
memory is *analytically estimated* from model configuration:

  FP16 KV size (bytes) = 2 × n_layers × n_kv_heads × head_dim × context_len × 2 (K+V)
  AgentKV size (bytes) = sink_FP16 + history_INT2 + window_FP16

Results are saved to dev/results/agent_kv_bench.json and printed as a table.

Usage::

    python dev/benchmarks/bench_agent_kv.py
    python dev/benchmarks/bench_agent_kv.py --model 7b --context-lengths 4096 8192 16384

"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


# ---------------------------------------------------------------------------
# Model configurations
# ---------------------------------------------------------------------------

@dataclass
class ModelSpec:
    name: str
    n_layers: int
    n_kv_heads: int
    head_dim: int


_MODEL_SPECS: Dict[str, ModelSpec] = {
    "7b":  ModelSpec("Qwen2.5-7B",   28, 8,  128),
    "14b": ModelSpec("Qwen2.5-14B",  40, 8,  128),
    "32b": ModelSpec("Qwen2.5-32B",  64, 8,  128),
}

_DEFAULT_CONTEXT_LENGTHS = [4096, 8192, 16384, 32768]


# ---------------------------------------------------------------------------
# Memory estimation helpers
# ---------------------------------------------------------------------------

def _fp16_kv_gb(spec: ModelSpec, context_len: int) -> float:
    """Estimate full FP16 KV cache size in GB."""
    # K + V, each is n_layers × n_kv_heads × head_dim × context_len × 2 bytes
    bytes_per_token = 2 * spec.n_layers * spec.n_kv_heads * spec.head_dim * 2  # 2B for FP16
    return bytes_per_token * context_len / 1e9


def _agent_kv_gb(
    spec: ModelSpec,
    context_len: int,
    sink_tokens: int = 4,
    local_window: int = 128,
    history_bits: int = 2,
    group_size: int = 16,
) -> Dict[str, float]:
    """Estimate AgentKV tiered KV cache size in GB."""
    # Each tier stores K + V tensors
    # FP16 = 2 bytes; INT2 = 0.25 bytes (2 bits); INT4 = 0.5 bytes
    bits_to_bytes = {2: 0.25, 4: 0.5, 8: 1.0}
    hist_bytes_per_element = bits_to_bytes.get(history_bits, 0.25)

    history_len = max(0, context_len - sink_tokens - local_window)
    elements_per_token = spec.n_layers * spec.n_kv_heads * spec.head_dim * 2  # K+V

    sink_gb   = elements_per_token * sink_tokens   * 2                  / 1e9
    window_gb = elements_per_token * local_window  * 2                  / 1e9
    # History: group-quantized to history_bits
    # Scale includes slight overhead for group centroids (~1/group_size extra)
    hist_overhead = 1.0 + (1.0 / group_size)
    history_gb = elements_per_token * history_len * hist_bytes_per_element * hist_overhead / 1e9

    total_gb = sink_gb + window_gb + history_gb
    return {
        "sink_gb":    round(sink_gb, 4),
        "window_gb":  round(window_gb, 4),
        "history_gb": round(history_gb, 4),
        "total_gb":   round(total_gb, 4),
    }


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def run_bench(
    model_key: str = "14b",
    context_lengths: List[int] = None,
    sink_tokens: int = 4,
    local_window: int = 128,
    history_bits: int = 2,
) -> Dict[str, Any]:
    if context_lengths is None:
        context_lengths = _DEFAULT_CONTEXT_LENGTHS

    spec = _MODEL_SPECS[model_key]
    results = []

    for ctx_len in context_lengths:
        fp16_gb = _fp16_kv_gb(spec, ctx_len)
        agent = _agent_kv_gb(spec, ctx_len, sink_tokens, local_window, history_bits)
        reduction_pct = 100.0 * (fp16_gb - agent["total_gb"]) / fp16_gb if fp16_gb > 0 else 0.0
        compression_ratio = fp16_gb / agent["total_gb"] if agent["total_gb"] > 0 else 0.0

        results.append({
            "context_len":       ctx_len,
            "fp16_kv_gb":        round(fp16_gb, 3),
            "agent_kv_total_gb": agent["total_gb"],
            "agent_kv_sink_gb":  agent["sink_gb"],
            "agent_kv_window_gb":agent["window_gb"],
            "agent_kv_hist_gb":  agent["history_gb"],
            "reduction_pct":     round(reduction_pct, 1),
            "compression_ratio": round(compression_ratio, 2),
        })

    avg_reduction = sum(r["reduction_pct"] for r in results) / len(results)
    avg_ratio = sum(r["compression_ratio"] for r in results) / len(results)

    return {
        "benchmark": "agent_kv_memory",
        "config": {
            "model":           spec.name,
            "n_layers":        spec.n_layers,
            "n_kv_heads":      spec.n_kv_heads,
            "head_dim":        spec.head_dim,
            "sink_tokens":     sink_tokens,
            "local_window":    local_window,
            "history_bits":    history_bits,
        },
        "results": results,
        "summary": {
            "avg_kv_reduction_pct":  round(avg_reduction, 1),
            "avg_compression_ratio": round(avg_ratio, 2),
            "note": (
                f"Analytical estimate; actual savings on hardware depend on "
                f"MLX Metal dispatch and quantization overhead. "
                f"INT{history_bits} history tier vs FP16 baseline."
            ),
        },
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 13A AgentKV memory benchmark")
    ap.add_argument("--model", choices=list(_MODEL_SPECS), default="14b",
                    help="Model scale (default: 14b)")
    ap.add_argument("--context-lengths", type=int, nargs="+",
                    default=_DEFAULT_CONTEXT_LENGTHS,
                    metavar="N",
                    help="Context lengths to benchmark (tokens)")
    ap.add_argument("--sink-tokens",  type=int, default=4)
    ap.add_argument("--local-window", type=int, default=128)
    ap.add_argument("--history-bits", type=int, choices=[2, 4, 8], default=2)
    ap.add_argument("--out", default="dev/results/agent_kv_bench.json",
                    help="Output JSON path")
    args = ap.parse_args()

    print(f"[bench_agent_kv] model={args.model}  "
          f"context_lengths={args.context_lengths}  "
          f"history_bits=INT{args.history_bits}")

    result = run_bench(
        model_key=args.model,
        context_lengths=args.context_lengths,
        sink_tokens=args.sink_tokens,
        local_window=args.local_window,
        history_bits=args.history_bits,
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[bench_agent_kv] saved → {args.out}")

    # Print table
    cfg = result["config"]
    print(f"\n  Model: {cfg['model']}  "
          f"({cfg['n_layers']} layers × {cfg['n_kv_heads']} KV-heads × {cfg['head_dim']} head-dim)")
    print(f"  AgentKV config: sink={cfg['sink_tokens']} tokens  "
          f"window={cfg['local_window']} tokens  "
          f"history=INT{cfg['history_bits']}")
    print()
    print(f"  {'Context':>10}  {'FP16 KV':>10}  {'AgentKV':>10}  "
          f"{'Reduction':>10}  {'Ratio':>8}")
    print(f"  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*8}")
    for r in result["results"]:
        print(f"  {r['context_len']:>10,}  "
              f"{r['fp16_kv_gb']:>9.3f}G  "
              f"{r['agent_kv_total_gb']:>9.3f}G  "
              f"{r['reduction_pct']:>9.1f}%  "
              f"{r['compression_ratio']:>7.2f}×")
    s = result["summary"]
    print()
    print(f"  Avg reduction : {s['avg_kv_reduction_pct']:.1f}%")
    print(f"  Avg ratio     : {s['avg_compression_ratio']:.2f}×")
    print(f"  Note          : {s['note']}")


if __name__ == "__main__":
    main()
