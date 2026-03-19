"""
dev/benchmarks/bench_wave27_28.py

Micro-benchmark suite for Wave 27 (Phase 1 wiring) and Wave 28 (Phase 2 modules).

Measures latency/throughput for each new optimisation in isolation on
Apple Silicon (MLX) using tiny synthetic models.  Results are written to
dev/results/wave27_28_bench.json.

Usage::

    python dev/benchmarks/bench_wave27_28.py
    python dev/benchmarks/bench_wave27_28.py --output /tmp/results.json
    python dev/benchmarks/bench_wave27_28.py --runs 100 --vocab 1024
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class BenchResult:
    name: str
    runs: int
    mean_ms: float
    p50_ms: float
    p95_ms: float
    std_ms: float
    notes: str = ""

    def __str__(self) -> str:
        return (
            f"{self.name:<45}  "
            f"mean={self.mean_ms:7.3f} ms  "
            f"p50={self.p50_ms:7.3f} ms  "
            f"p95={self.p95_ms:7.3f} ms  "
            f"{self.notes}"
        )


def _bench(name: str, fn, runs: int, warmup: int = 3) -> BenchResult:
    """Time *fn()* for *runs* repetitions and return a BenchResult."""
    for _ in range(warmup):
        fn()
    times_ms = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        times_ms.append((time.perf_counter() - t0) * 1e3)
    arr = np.array(times_ms)
    return BenchResult(
        name    = name,
        runs    = runs,
        mean_ms = float(arr.mean()),
        p50_ms  = float(np.median(arr)),
        p95_ms  = float(np.percentile(arr, 95)),
        std_ms  = float(arr.std()),
    )


# ---------------------------------------------------------------------------
# Wave 27 — Phase 1 benchmarks
# ---------------------------------------------------------------------------

def bench_fused_sampler(vocab: int, runs: int) -> List[BenchResult]:
    """Compare FusedSampler vs manual temperature+top-p sampling."""
    results = []
    rng    = np.random.default_rng(42)
    logits = rng.standard_normal(vocab).astype(np.float32)

    # --- Baseline: manual top-p ---
    def _manual_sample():
        lv = logits / 0.8
        lv -= lv.max()
        p = np.exp(lv)
        p /= p.sum()
        # top-p nucleus
        sorted_idx = np.argsort(p)[::-1]
        sorted_p   = p[sorted_idx]
        cum        = np.cumsum(sorted_p)
        mask       = (cum - sorted_p) > 0.9
        sorted_p[mask] = 0.0
        p[sorted_idx] = sorted_p
        p /= p.sum()
        np.random.choice(vocab, p=p)

    results.append(_bench("wave27/fused-sampler/baseline-manual", _manual_sample, runs))

    try:
        from squish.hardware.fused_sampler import FusedSampler, SamplerConfig
        sampler = FusedSampler(SamplerConfig(temperature=0.8, top_p=0.9))
        results.append(_bench(
            "wave27/fused-sampler/FusedSampler.sample",
            lambda: sampler.sample(logits),
            runs,
        ))
    except ImportError as e:
        results.append(BenchResult("wave27/fused-sampler/FusedSampler.sample", 0, 0, 0, 0, 0, f"SKIP: {e}"))

    return results


def bench_cache_warmup(runs: int) -> List[BenchResult]:
    """Measure CacheWarmupPredictor.record_access() overhead."""
    results: List[BenchResult] = []
    try:
        import time as _t
        from squish.kv.cache_warmup import CacheWarmupPredictor, WarmupConfig
        predictor = CacheWarmupPredictor(WarmupConfig(top_k=32))
        prefix    = list(range(256))
        t0        = _t.monotonic()
        results.append(_bench(
            "wave27/cache-warmup/record_access",
            lambda: predictor.record_access(prefix, _t.monotonic()),
            runs,
        ))
        results.append(_bench(
            "wave27/cache-warmup/get_warmup_candidates",
            lambda: predictor.get_warmup_candidates(),
            runs,
        ))
    except ImportError as e:
        results.append(BenchResult("wave27/cache-warmup", 0, 0, 0, 0, 0, f"SKIP: {e}"))
    return results


def bench_confidence_estimator(vocab: int, runs: int) -> List[BenchResult]:
    """Measure ConfidenceEstimator overhead (Layer-skip, Step 1E)."""
    results: List[BenchResult] = []
    try:
        from squish.token.layer_skip import ConfidenceEstimator
        rng    = np.random.default_rng(7)
        logits = rng.standard_normal(vocab).astype(np.float32)
        for metric in ("max_prob", "margin", "neg_entropy"):
            est = ConfidenceEstimator(metric)
            results.append(_bench(
                f"wave27/layer-skip/ConfidenceEstimator({metric})",
                lambda m=metric, e=est: e.estimate(logits),
                runs,
            ))
    except ImportError as e:
        results.append(BenchResult("wave27/layer-skip", 0, 0, 0, 0, 0, f"SKIP: {e}"))
    return results


# ---------------------------------------------------------------------------
# Wave 28 — Phase 2 benchmarks
# ---------------------------------------------------------------------------

def bench_adaptive_prefill_fusion(runs: int) -> List[BenchResult]:
    """Measure PrefillFusionController.plan() for various prompt lengths."""
    results: List[BenchResult] = []
    try:
        from squish.streaming.adaptive_prefill_fusion import (
            PrefillFusionConfig,
            PrefillFusionController,
        )
        ctrl = PrefillFusionController(PrefillFusionConfig())
        rng  = np.random.default_rng(0)
        for n in (64, 512, 2048):
            ids = rng.integers(0, 50000, n).tolist()
            results.append(_bench(
                f"wave28/adaptive-prefill/plan(n={n})",
                lambda token_ids=ids: ctrl.plan(token_ids),
                runs,
            ))
    except ImportError as e:
        results.append(BenchResult("wave28/adaptive-prefill", 0, 0, 0, 0, 0, f"SKIP: {e}"))
    return results


def bench_draft_multiplexer(runs: int) -> List[BenchResult]:
    """Measure DraftMultiplexer.select() and update() overhead."""
    results: List[BenchResult] = []
    try:
        from squish.speculative.draft_multiplexer import (
            DraftMultiplexer,
            DraftMultiplexerConfig,
            DraftStrategy,
        )
        mux = DraftMultiplexer(DraftMultiplexerConfig(min_samples=1))
        # Seed enough samples so we exercise EMA path
        for _ in range(10):
            mux.update(DraftStrategy.EAGLE3, "conversation", 0.7, 50.0)
            mux.update(DraftStrategy.NGRAM,  "conversation", 0.4, 90.0)
        results.append(_bench(
            "wave28/draft-mux/select(prompt)",
            lambda: mux.select(prompt="What is the best approach here?"),
            runs,
        ))
        results.append(_bench(
            "wave28/draft-mux/update",
            lambda: mux.update(DraftStrategy.EAGLE3, "coding", 0.72, 48.0),
            runs,
        ))
    except ImportError as e:
        results.append(BenchResult("wave28/draft-mux", 0, 0, 0, 0, 0, f"SKIP: {e}"))
    return results


def bench_per_layer_sparse_attn(runs: int) -> List[BenchResult]:
    """Measure PerLayerSparseAttn profiling and mask query overhead."""
    results: List[BenchResult] = []
    try:
        from squish.attention.per_layer_sparse_attn import (
            PerLayerSparseConfig,
            PerLayerSparseAttn,
        )
        cfg  = PerLayerSparseConfig(n_layers=32, n_heads=32, min_seq_len=64, warmup_steps=0)
        ctrl = PerLayerSparseAttn(cfg)
        rng  = np.random.default_rng(5)
        for seq_len in (128, 512, 2048):
            attn_w = rng.random((32, 32, seq_len, seq_len)).astype(np.float32)
            attn_w /= attn_w.sum(axis=-1, keepdims=True)
            results.append(_bench(
                f"wave28/per-layer-sparse/profile_prefill(seq={seq_len})",
                lambda w=attn_w: ctrl.profile_prefill(w),
                max(runs // 5, 3),  # profiling is more expensive; fewer runs
            ))
        ctrl.profile_prefill(attn_w)  # ensure profiles are ready
        results.append(_bench(
            "wave28/per-layer-sparse/sparse_mask(layer=0)",
            lambda: ctrl.sparse_mask(0),
            runs,
        ))
    except ImportError as e:
        results.append(BenchResult("wave28/per-layer-sparse", 0, 0, 0, 0, 0, f"SKIP: {e}"))
    return results


def bench_speculative_prefill(runs: int) -> List[BenchResult]:
    """Measure SpeculativePrefiller overhead (pure-NumPy forward stubs)."""
    results: List[BenchResult] = []
    try:
        from squish.speculative.speculative_prefill import (
            SpecPrefillConfig,
            SpeculativePrefiller,
        )
        n_layers = 32
        n_heads  = 8
        head_dim = 64
        cfg = SpecPrefillConfig(n_layers=n_layers, min_prompt_len=16)
        rng = np.random.default_rng(55)

        def _draft(ids):
            return [rng.standard_normal((n_heads, len(ids), head_dim)).astype(np.float32)
                    for _ in range(n_layers)]

        def _target(ids, mask):
            return [rng.standard_normal((n_heads, len(ids), head_dim)).astype(np.float32)
                    for _ in range(n_layers)]

        p = SpeculativePrefiller(_draft, _target, cfg)
        for n in (64, 256, 512):
            ids = list(range(n))
            results.append(_bench(
                f"wave28/spec-prefill/prefill(n={n})",
                lambda token_ids=ids: p.prefill(token_ids),
                max(runs // 10, 3),
            ))
    except ImportError as e:
        results.append(BenchResult("wave28/spec-prefill", 0, 0, 0, 0, 0, f"SKIP: {e}"))
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Wave 27+28 micro-benchmark suite for Squish inference optimisations"
    )
    ap.add_argument("--runs",   type=int, default=50, help="Repetitions per benchmark")
    ap.add_argument("--vocab",  type=int, default=32000, help="Vocabulary size for sampler benchmarks")
    ap.add_argument(
        "--output",
        type=str,
        default=str(Path(__file__).parent.parent / "results" / "wave27_28_bench.json"),
        help="JSON output path",
    )
    return ap.parse_args()


def main() -> None:
    args   = _parse_args()
    runs   = args.runs
    vocab  = args.vocab
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*75}")
    print(f"  Squish Wave 27+28 Benchmark  —  runs={runs}  vocab={vocab}")
    print(f"{'='*75}\n")

    all_results: List[BenchResult] = []

    sections = [
        ("Wave 27 — FusedSampler",          lambda: bench_fused_sampler(vocab, runs)),
        ("Wave 27 — CacheWarmup",            lambda: bench_cache_warmup(runs)),
        ("Wave 27 — ConfidenceEstimator",    lambda: bench_confidence_estimator(vocab, runs)),
        ("Wave 28 — AdaptivePrefillFusion",  lambda: bench_adaptive_prefill_fusion(runs)),
        ("Wave 28 — DraftMultiplexer",       lambda: bench_draft_multiplexer(runs)),
        ("Wave 28 — PerLayerSparseAttn",     lambda: bench_per_layer_sparse_attn(runs)),
        ("Wave 28 — SpeculativePrefill",     lambda: bench_speculative_prefill(runs)),
    ]

    for title, fn in sections:
        print(f"  {title}")
        print(f"  {'-' * 60}")
        try:
            results = fn()
        except Exception as e:
            print(f"  ERROR: {e}")
            results = []
        for r in results:
            print(f"  {r}")
            all_results.append(r)
        print()

    # Write JSON
    data = [asdict(r) for r in all_results]
    output.write_text(json.dumps(data, indent=2))
    print(f"\nResults written to: {output}\n")


if __name__ == "__main__":
    main()
