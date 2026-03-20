"""
BenchmarkHarness — Statistically Rigorous 30-Trial Benchmark Suite.

Purpose: produce the benchmark table required for the Squish technical report.
Prior benchmarks used single-run numbers; this harness:
  • Runs n_trials (default 30) per (model, baseline) combination.
  • Discards warmup trials.
  • Reports mean ± σ, P50, P99 for TTFT and tokens/sec.
  • Outputs a reproducible markdown table for the paper.

Usage::

    harness = BenchmarkHarness(BenchmarkConfig(n_trials=30))
    result = harness.run_model("qwen2.5-1.5b", inference_fn)
    print(harness.to_markdown_table([result]))
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkConfig:
    """Configuration for the benchmark harness."""
    n_trials: int = 30
    warmup_trials: int = 3
    max_tokens: int = 100
    models: List[str] = field(default_factory=list)
    baselines: List[str] = field(default_factory=list)
    timeout_seconds: float = 60.0   # max per-trial wall time

    def __post_init__(self) -> None:
        if self.n_trials < 1:
            raise ValueError(f"n_trials must be >= 1, got {self.n_trials}")
        if self.warmup_trials < 0:
            raise ValueError(f"warmup_trials must be >= 0, got {self.warmup_trials}")
        if self.timeout_seconds < 0:
            raise ValueError(f"timeout_seconds must be >= 0, got {self.timeout_seconds}")


# ---------------------------------------------------------------------------
# Trial result
# ---------------------------------------------------------------------------

@dataclass
class TrialResult:
    """Measurements from a single benchmark trial."""
    trial_idx: int
    ttft_ms: float              # time to first token (ms)
    total_ms: float             # total wall time (ms)
    tokens_generated: int
    tokens_per_sec: float
    peak_memory_gb: float       # peak memory usage (GB) — 0 if unavailable
    warmup: bool = False        # True → excluded from aggregate stats

    @classmethod
    def from_timing(
        cls,
        trial_idx: int,
        ttft_ms: float,
        total_ms: float,
        tokens_generated: int,
        peak_memory_gb: float = 0.0,
        warmup: bool = False,
    ) -> "TrialResult":
        tps = tokens_generated / (total_ms / 1000.0) if total_ms > 0 else 0.0
        return cls(
            trial_idx=trial_idx,
            ttft_ms=ttft_ms,
            total_ms=total_ms,
            tokens_generated=tokens_generated,
            tokens_per_sec=tps,
            peak_memory_gb=peak_memory_gb,
            warmup=warmup,
        )


# ---------------------------------------------------------------------------
# Aggregate statistics
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkStats:
    """Aggregate statistics for one (model, n_trials) run."""
    model: str
    n_trials: int
    # TTFT
    ttft_mean_ms: float
    ttft_std_ms: float
    ttft_p50_ms: float
    ttft_p99_ms: float
    # Tokens per second
    tps_mean: float
    tps_std: float
    tps_p50: float
    tps_p99: float
    # Memory
    peak_memory_mean_gb: float
    # Raw trials (for post-hoc analysis)
    trials: List[TrialResult] = field(default_factory=list, repr=False)

    @classmethod
    def from_trials(cls, model: str, trials: List[TrialResult]) -> "BenchmarkStats":
        active = [t for t in trials if not t.warmup]
        if not active:
            raise ValueError("No active (non-warmup) trials to aggregate")

        ttft = np.array([t.ttft_ms for t in active])
        tps = np.array([t.tokens_per_sec for t in active])
        mem = np.array([t.peak_memory_gb for t in active])

        return cls(
            model=model,
            n_trials=len(active),
            ttft_mean_ms=float(ttft.mean()),
            ttft_std_ms=float(ttft.std()),
            ttft_p50_ms=float(np.percentile(ttft, 50)),
            ttft_p99_ms=float(np.percentile(ttft, 99)),
            tps_mean=float(tps.mean()),
            tps_std=float(tps.std()),
            tps_p50=float(np.percentile(tps, 50)),
            tps_p99=float(np.percentile(tps, 99)),
            peak_memory_mean_gb=float(mem.mean()),
            trials=trials,
        )

    def to_markdown_row(self) -> str:
        """Format as a markdown table row."""
        return (
            f"| {self.model} "
            f"| {self.ttft_mean_ms:.1f} ± {self.ttft_std_ms:.1f} "
            f"| {self.ttft_p50_ms:.1f} "
            f"| {self.ttft_p99_ms:.1f} "
            f"| {self.tps_mean:.1f} ± {self.tps_std:.1f} "
            f"| {self.tps_p50:.1f} "
            f"| {self.peak_memory_mean_gb:.2f} "
            f"| {self.n_trials} |"
        )

    def __repr__(self) -> str:
        return (
            f"BenchmarkStats({self.model!r}: "
            f"TTFT={self.ttft_mean_ms:.1f}±{self.ttft_std_ms:.1f}ms, "
            f"TPS={self.tps_mean:.1f}±{self.tps_std:.1f}, "
            f"n={self.n_trials})"
        )


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------

class BenchmarkHarness:
    """Run repeated benchmark trials and aggregate results.

    The inference_fn protocol::

        def inference_fn(prompt: str, max_tokens: int) -> BenchmarkCallResult:
            ...

    Where BenchmarkCallResult has:
        .ttft_ms: float
        .total_ms: float
        .tokens_generated: int
        .peak_memory_gb: float (optional, 0 if unavailable)
    """

    def __init__(self, config: BenchmarkConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------
    # Run one model
    # ------------------------------------------------------------------

    def run_model(
        self,
        model_name: str,
        inference_fn: Callable[..., Any],
        prompt: str = "Explain the concept of speculative decoding in detail.",
    ) -> BenchmarkStats:
        """Run n_trials + warmup_trials and return aggregate stats.

        Args:
            model_name:   label for the model.
            inference_fn: callable with signature
                          (prompt, max_tokens) → object with .ttft_ms,
                          .total_ms, .tokens_generated, .peak_memory_gb
            prompt:       benchmark prompt.

        Returns:
            BenchmarkStats over the (non-warmup) trials.
        """
        cfg = self.config
        total_runs = cfg.warmup_trials + cfg.n_trials
        trials: List[TrialResult] = []

        for i in range(total_runs):
            is_warmup = i < cfg.warmup_trials
            result = inference_fn(prompt, cfg.max_tokens)

            trial = TrialResult.from_timing(
                trial_idx=i,
                ttft_ms=float(getattr(result, "ttft_ms", 0.0)),
                total_ms=float(getattr(result, "total_ms", 0.0)),
                tokens_generated=int(getattr(result, "tokens_generated", 0)),
                peak_memory_gb=float(getattr(result, "peak_memory_gb", 0.0)),
                warmup=is_warmup,
            )
            trials.append(trial)

        return BenchmarkStats.from_trials(model_name, trials)

    # ------------------------------------------------------------------
    # Run all models
    # ------------------------------------------------------------------

    def run_all(
        self,
        inference_fns: Dict[str, Callable[..., Any]],
        prompt: str = "Explain the concept of speculative decoding in detail.",
    ) -> List[BenchmarkStats]:
        """Run all model/baseline pairs.

        Args:
            inference_fns: dict mapping model_name → inference_fn.
            prompt:        shared benchmark prompt.

        Returns:
            List of BenchmarkStats, one per model/baseline.
        """
        results = []
        for name, fn in inference_fns.items():
            stats = self.run_model(name, fn, prompt)
            results.append(stats)
        return results

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------

    @staticmethod
    def to_markdown_table(results: List[BenchmarkStats]) -> str:
        """Format a list of results as a markdown table.

        Returns:
            Full markdown table string ready for docs/paper/.
        """
        header = (
            "| Model "
            "| TTFT mean±σ (ms) "
            "| TTFT P50 (ms) "
            "| TTFT P99 (ms) "
            "| TPS mean±σ "
            "| TPS P50 "
            "| Peak RAM (GB) "
            "| Trials |"
        )
        separator = "|-------|---------|--------|--------|-----|-----|--------|--------|"
        rows = [r.to_markdown_row() for r in results]
        return "\n".join([header, separator] + rows)

    @staticmethod
    def speedup_table(
        results: List[BenchmarkStats],
        baseline_name: str,
    ) -> str:
        """Format a speedup-over-baseline table.

        Args:
            results:        list of BenchmarkStats.
            baseline_name:  model_name of the baseline row.

        Returns:
            Markdown table with speedup multipliers.
        """
        baseline = next((r for r in results if r.model == baseline_name), None)
        if baseline is None:
            return f"Baseline '{baseline_name}' not found in results."

        header = "| Model | TTFT speedup | TPS speedup |"
        separator = "|-------|------------|-----------|"
        rows = []
        for r in results:
            ttft_speedup = (
                baseline.ttft_mean_ms / r.ttft_mean_ms
                if r.ttft_mean_ms > 0
                else float("inf")
            )
            tps_speedup = (
                r.tps_mean / baseline.tps_mean if baseline.tps_mean > 0 else float("inf")
            )
            rows.append(
                f"| {r.model} | {ttft_speedup:.2f}× | {tps_speedup:.2f}× |"
            )
        return "\n".join([header, separator] + rows)

    def __repr__(self) -> str:
        return (
            f"BenchmarkHarness(n_trials={self.config.n_trials}, "
            f"warmup={self.config.warmup_trials})"
        )
