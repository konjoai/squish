#!/usr/bin/env python3
"""bench_metal_fusion.py — Phase 10B Metal-fusion kernel microbenchmark.

Compares fused vs. unfused operator latency for RoPE-QK, SwiGLU, and INT8
KV-attention across seq_len in {128, 1024, 8192}.

Because ``mx.metal.kernel`` is not available in CI / non-Apple environments,
both paths run through the numpy fallbacks in :mod:`squish.metal_fusion` when
Metal is absent.  On an M-series Mac with MLX 0.18+ the fused paths will
dispatch compiled Metal kernels instead.

Usage::

    python dev/benchmarks/bench_metal_fusion.py [--head-dim 128] [--n-heads 8]
        [--warmup 3] [--reps 10] [--out dev/results/metal_fusion_bench.json]

Results are written in JSON format and summarised to stdout.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List

import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from squish.metal_fusion import (  # noqa: E402
    MetalFusionConfig,
    MetalFusionKernels,
    _METAL_FUSION_AVAILABLE,
    fused_int8_kv_attn,
    fused_rope_qk,
    fused_swiglu,
)


# ---------------------------------------------------------------------------
# Unfused reference implementations (numpy-only)
# ---------------------------------------------------------------------------

def _ref_rope_qk(Q: np.ndarray, K: np.ndarray,
                 cos: np.ndarray, sin: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Reference RoPE rotation without any fusion."""
    half = Q.shape[-1] // 2
    cos_h, sin_h = cos[..., :half], sin[..., :half]
    def _rot(x):
        x1, x2 = x[..., :half], x[..., half:]
        return np.concatenate([x1 * cos_h - x2 * sin_h,
                                x1 * sin_h + x2 * cos_h], axis=-1)
    return _rot(Q), _rot(K)


def _ref_swiglu(gate: np.ndarray, up: np.ndarray) -> np.ndarray:
    return gate * (1.0 / (1.0 + np.exp(-gate))) * up


def _ref_int8_kv_attn(q: np.ndarray, k_int8: np.ndarray, v_int8: np.ndarray,
                      k_scales: np.ndarray, v_scales: np.ndarray) -> np.ndarray:
    k_f = k_int8.astype(np.float32) * k_scales  # scales shape (..., 1) broadcasts
    v_f = v_int8.astype(np.float32) * v_scales
    scale = 1.0 / np.sqrt(q.shape[-1])
    scores = q @ k_f.T * scale
    scores -= scores.max(axis=-1, keepdims=True)
    w = np.exp(scores)
    w /= w.sum(axis=-1, keepdims=True)
    return w @ v_f


# ---------------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------------

def _time_fn(fn, warmup: int, reps: int) -> float:
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t0) / reps


# ---------------------------------------------------------------------------
# Per-operator benchmarks
# ---------------------------------------------------------------------------

def bench_rope(seq_len: int, head_dim: int, n_heads: int,
               kernels: MetalFusionKernels, warmup: int, reps: int) -> Dict[str, Any]:
    rng = np.random.RandomState(0)
    Q = rng.randn(seq_len, n_heads, head_dim).astype(np.float32)
    K = rng.randn(seq_len, n_heads, head_dim).astype(np.float32)
    theta = 10000.0 ** (-np.arange(0, head_dim, 2, dtype=np.float32) / head_dim)
    t = np.arange(seq_len, dtype=np.float32)
    freqs = np.outer(t, theta)
    # shape: (seq_len, 1, head_dim) → broadcasts over n_heads
    cos = np.repeat(np.cos(freqs), 2, axis=-1)[:, np.newaxis, :]
    sin = np.repeat(np.sin(freqs), 2, axis=-1)[:, np.newaxis, :]

    ref_ms  = _time_fn(lambda: _ref_rope_qk(Q, K, cos, sin), warmup, reps) * 1000
    fuse_ms = _time_fn(lambda: fused_rope_qk(Q, K, cos, sin, kernels=kernels), warmup, reps) * 1000
    return {
        "op": "rope_qk", "seq_len": seq_len,
        "ref_ms": round(ref_ms, 4), "fused_ms": round(fuse_ms, 4),
        "speedup": round(ref_ms / max(fuse_ms, 1e-9), 3),
    }


def bench_swiglu(seq_len: int, ffn_dim: int,
                 kernels: MetalFusionKernels, warmup: int, reps: int) -> Dict[str, Any]:
    rng = np.random.RandomState(1)
    gate = rng.randn(seq_len, ffn_dim).astype(np.float32)
    up   = rng.randn(seq_len, ffn_dim).astype(np.float32)

    ref_ms  = _time_fn(lambda: _ref_swiglu(gate, up), warmup, reps) * 1000
    fuse_ms = _time_fn(lambda: fused_swiglu(gate, up, kernels=kernels), warmup, reps) * 1000
    return {
        "op": "swiglu", "seq_len": seq_len,
        "ref_ms": round(ref_ms, 4), "fused_ms": round(fuse_ms, 4),
        "speedup": round(ref_ms / max(fuse_ms, 1e-9), 3),
    }


def bench_int8_attn(seq_len: int, head_dim: int,
                    kernels: MetalFusionKernels, warmup: int, reps: int) -> Dict[str, Any]:
    rng = np.random.RandomState(2)
    q      = rng.randn(seq_len, head_dim).astype(np.float32)
    k_int8 = (rng.randn(seq_len, head_dim) * 127).astype(np.int8)
    v_int8 = (rng.randn(seq_len, head_dim) * 127).astype(np.int8)
    # shape (seq_len, 1) so it broadcasts over head_dim
    k_sc   = rng.rand(seq_len, 1).astype(np.float32) * 0.01
    v_sc   = rng.rand(seq_len, 1).astype(np.float32) * 0.01

    ref_ms  = _time_fn(lambda: _ref_int8_kv_attn(q, k_int8, v_int8, k_sc, v_sc), warmup, reps) * 1000
    fuse_ms = _time_fn(lambda: fused_int8_kv_attn(q, k_int8, v_int8, k_sc, v_sc, kernels=kernels), warmup, reps) * 1000
    return {
        "op": "int8_kv_attn", "seq_len": seq_len,
        "ref_ms": round(ref_ms, 4), "fused_ms": round(fuse_ms, 4),
        "speedup": round(ref_ms / max(fuse_ms, 1e-9), 3),
    }


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def run_bench(
    head_dim: int = 128,
    n_heads: int = 8,
    ffn_dim: int = 512,
    seq_lens: List[int] = None,
    warmup: int = 2,
    reps: int = 5,
) -> Dict[str, Any]:
    if seq_lens is None:
        seq_lens = [128, 1024, 8192]

    kernels = MetalFusionKernels(MetalFusionConfig(require_metal=False))
    results: List[Dict[str, Any]] = []
    for sl in seq_lens:
        results.append(bench_rope(sl, head_dim, n_heads, kernels, warmup, reps))
        results.append(bench_swiglu(sl, ffn_dim, kernels, warmup, reps))
        results.append(bench_int8_attn(sl, head_dim, kernels, warmup, reps))

    avg_speedup = float(np.mean([r["speedup"] for r in results]))
    return {
        "benchmark": "metal_fusion",
        "metal_available": _METAL_FUSION_AVAILABLE,
        "kernels": {
            "rope_enabled":      kernels.rope_enabled,
            "swiglu_enabled":    kernels.swiglu_enabled,
            "int8_attn_enabled": kernels.int8_attn_enabled,
        },
        "config": {
            "head_dim": head_dim,
            "n_heads": n_heads,
            "ffn_dim": ffn_dim,
            "seq_lens": seq_lens,
            "warmup": warmup,
            "reps": reps,
        },
        "results": results,
        "summary": {
            "avg_speedup": round(avg_speedup, 3),
            "note": (
                "Metal paths active" if _METAL_FUSION_AVAILABLE
                else "numpy fallback — no Metal hardware detected"
            ),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 10B Metal-fusion microbenchmark")
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--n-heads",  type=int, default=8)
    ap.add_argument("--ffn-dim",  type=int, default=512)
    ap.add_argument("--seq-lens", type=int, nargs="+", default=[128, 1024, 8192])
    ap.add_argument("--warmup",   type=int, default=2)
    ap.add_argument("--reps",     type=int, default=5)
    ap.add_argument("--out", default="dev/results/metal_fusion_bench.json")
    args = ap.parse_args()

    print(f"[bench_metal_fusion] Metal={'yes' if _METAL_FUSION_AVAILABLE else 'no (numpy fallback)'}  "
          f"head_dim={args.head_dim}  ffn_dim={args.ffn_dim}")
    result = run_bench(
        head_dim=args.head_dim,
        n_heads=args.n_heads,
        ffn_dim=args.ffn_dim,
        seq_lens=args.seq_lens,
        warmup=args.warmup,
        reps=args.reps,
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[bench_metal_fusion] saved → {args.out}")

    print(f"  avg speedup  : {result['summary']['avg_speedup']:.3f}×")
    print(f"  note         : {result['summary']['note']}")
    print()
    print(f"  {'op':<16} {'seq_len':>8}  {'ref_ms':>10}  {'fused_ms':>10}  {'speedup':>8}")
    print("  " + "-" * 58)
    for r in result["results"]:
        print(f"  {r['op']:<16} {r['seq_len']:>8}  {r['ref_ms']:>10.4f}  "
              f"{r['fused_ms']:>10.4f}  {r['speedup']:>8.3f}×")


if __name__ == "__main__":
    main()
