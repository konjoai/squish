# [Experimental] This module is part of Squish v44+ (Wave 70).
# Proof-of-concept quality: API and behaviour may change without notice.
# For stable core functionality see Waves 1–12 in MODULES.md.
"""SquizdBenchmark — 30-trial statistical benchmark for SQUIZD format variants.

Extends :mod:`squish.bench.benchmark_harness` with SQUIZD-specific metrics:

  * Time-to-first-token (TTFT) at P50, P95, and P99 percentiles
  * Tokens/sec at P50, P95, and P99
  * Peak Metal ``recommendedMaxWorkingSetSize`` (simulated)
  * On-disk ``.squizd`` file size (GB)
  * Resident RAM footprint (GB)
  * Comparison against a GGUF Q4_K_M baseline

Usage::

    from squish.bench.squish_bench import SquizdBenchmark, SquizdBenchConfig

    bench = SquizdBenchmark(SquizdBenchConfig())
    result = bench.run_variant("Qwen2.5-7B", SquizdFormatVariant.FULL, inference_fn)
    print(bench.to_markdown_table([result]))
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from squish.bench.benchmark_harness import (
    BenchmarkConfig,
    BenchmarkHarness,
    TrialResult,
)

__all__ = [
    "SquizdFormatVariant",
    "SquizdBenchConfig",
    "SquizdModelResult",
    "SquizdBenchmark",
    "GGUFBaselineResult",
    "FormatComparison",
]

# ---------------------------------------------------------------------------
# Format variants
# ---------------------------------------------------------------------------

class SquizdFormatVariant(str, Enum):
    """The four SQUIZD format variants to benchmark."""

    ASTC       = "squizd-astc"
    INT4       = "squizd-int4"
    INT4_SPARSE = "squizd-int4-sparse"
    FULL       = "squizd-full"

    def __str__(self) -> str:  # noqa: D105
        return self.value


# Default 21-model evaluation set for the Wave 70 arXiv paper.
_DEFAULT_MODELS: List[str] = [
    "Qwen2.5-0.5B",
    "Qwen2.5-1.5B",
    "Qwen2.5-3B",
    "Qwen2.5-7B",
    "Qwen2.5-14B",
    "Llama-3.2-1B",
    "Llama-3.2-3B",
    "Llama-3.1-8B",
    "Llama-3.3-70B-Q4",
    "Mistral-7B-v0.3",
    "Mistral-Nemo-12B",
    "Phi-3.5-mini",
    "Phi-4-14B",
    "Gemma-2-2B",
    "Gemma-2-9B",
    "DeepSeek-R1-Distill-Qwen-7B",
    "DeepSeek-R1-Distill-Qwen-14B",
    "SmolLM2-1.7B",
    "Yi-1.5-9B",
    "InternLM2.5-7B",
    "StarCoder2-7B",
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SquizdBenchConfig:
    """Configuration for a SQUIZD format benchmark run.

    Attributes:
        n_trials: Number of measurement trials per (model, variant).
        warmup_trials: Discarded warm-up trials before measurement.
        max_tokens: Maximum tokens generated per trial.
        timeout_seconds: Per-trial wall-time limit.
        variants: Format variants to benchmark.
        models: Model identifiers to benchmark.
    """

    n_trials: int = 30
    warmup_trials: int = 3
    max_tokens: int = 100
    timeout_seconds: float = 60.0
    variants: List[SquizdFormatVariant] = field(
        default_factory=lambda: list(SquizdFormatVariant)
    )
    models: List[str] = field(default_factory=lambda: list(_DEFAULT_MODELS))

    def __post_init__(self) -> None:
        if self.n_trials < 1:
            raise ValueError(f"n_trials must be >= 1, got {self.n_trials}")
        if self.warmup_trials < 0:
            raise ValueError(f"warmup_trials must be >= 0, got {self.warmup_trials}")


# ---------------------------------------------------------------------------
# Per-model result
# ---------------------------------------------------------------------------

@dataclass
class SquizdModelResult:
    """Aggregate benchmark result for one (model, variant) combination.

    Attributes:
        model: Model identifier string.
        variant: Which SQUIZD format variant was benchmarked.
        ttft_p50_ms: Median TTFT in milliseconds.
        ttft_p95_ms: 95th percentile TTFT.
        ttft_p99_ms: 99th percentile TTFT.
        tps_p50: Median tokens-per-second throughput.
        tps_p95: 95th percentile tokens/sec.
        tps_p99: 99th percentile tokens/sec.
        peak_memory_gb: Peak Metal ``recommendedMaxWorkingSetSize`` in GB.
        disk_size_gb: On-disk file size of the model in GB.
        ram_resident_gb: Resident RAM footprint in GB.
        n_trials: Number of active (non-warmup) trials.
    """

    model: str
    variant: SquizdFormatVariant
    ttft_p50_ms: float
    ttft_p95_ms: float
    ttft_p99_ms: float
    tps_p50: float
    tps_p95: float
    tps_p99: float
    peak_memory_gb: float
    disk_size_gb: float
    ram_resident_gb: float
    n_trials: int


@dataclass
class GGUFBaselineResult:
    """GGUF Q4_K_M baseline result for model size comparison.

    Attributes:
        model: Model identifier string (matching :class:`SquizdModelResult`).
        ttft_p50_ms: Median TTFT.
        tps_p50: Median tokens/sec.
        disk_size_gb: Q4_K_M disk size in GB.
        ram_resident_gb: Resident RAM in GB.
    """

    model: str
    ttft_p50_ms: float
    tps_p50: float
    disk_size_gb: float
    ram_resident_gb: float


@dataclass
class FormatComparison:
    """Side-by-side comparison of a SQUIZD variant vs GGUF Q4_K_M.

    Attributes:
        squizd: :class:`SquizdModelResult` for the SQUIZD variant.
        gguf: :class:`GGUFBaselineResult` for the GGUF baseline.
        ttft_speedup: ``gguf.ttft_p50_ms / squizd.ttft_p50_ms``.
        tps_gain: ``squizd.tps_p50 / gguf.tps_p50 - 1``.
        disk_ratio: ``squizd.disk_size_gb / gguf.disk_size_gb``.
    """

    squizd: SquizdModelResult
    gguf: GGUFBaselineResult
    ttft_speedup: float
    tps_gain: float
    disk_ratio: float

    @classmethod
    def compare(
        cls,
        squizd: SquizdModelResult,
        gguf: GGUFBaselineResult,
    ) -> "FormatComparison":
        """Build a :class:`FormatComparison` from the two result objects."""
        ttft_speedup = (
            gguf.ttft_p50_ms / squizd.ttft_p50_ms
            if squizd.ttft_p50_ms > 0
            else 0.0
        )
        tps_gain = (
            squizd.tps_p50 / gguf.tps_p50 - 1.0
            if gguf.tps_p50 > 0
            else 0.0
        )
        disk_ratio = (
            squizd.disk_size_gb / gguf.disk_size_gb
            if gguf.disk_size_gb > 0
            else 0.0
        )
        return cls(squizd=squizd, gguf=gguf, ttft_speedup=ttft_speedup, tps_gain=tps_gain, disk_ratio=disk_ratio)


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

class SquizdBenchmark:
    """Runs the SQUIZD format variant benchmark suite.

    Parameters:
        config: :class:`SquizdBenchConfig` controlling trial counts and model list.
    """

    def __init__(self, config: Optional[SquizdBenchConfig] = None) -> None:
        self.config = config or SquizdBenchConfig()
        self._harness = BenchmarkHarness(
            BenchmarkConfig(
                n_trials=self.config.n_trials,
                warmup_trials=self.config.warmup_trials,
                max_tokens=self.config.max_tokens,
                timeout_seconds=self.config.timeout_seconds,
            )
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_variant(
        self,
        model: str,
        variant: SquizdFormatVariant,
        inference_fn: Callable[..., Any],
        *,
        model_path: Optional[Path] = None,
    ) -> SquizdModelResult:
        """Benchmark *model* under *variant* using *inference_fn*.

        Args:
            model: Model identifier.
            variant: SQUIZD format variant to benchmark.
            inference_fn: Zero-argument callable that performs one inference
                pass and returns ``(ttft_ms, tokens_generated, peak_memory_gb)``.
            model_path: Optional path to the model file (to measure disk size).

        Returns:
            :class:`SquizdModelResult` with all percentile statistics.
        """
        trials = self._collect_trials(model, inference_fn)
        active = [t for t in trials if not t.warmup]
        disk_gb = _measure_disk_size(model_path)
        ram_gb = _measure_ram_resident()

        return SquizdModelResult(
            model=model,
            variant=variant,
            ttft_p50_ms=float(np.percentile([t.ttft_ms for t in active], 50)),
            ttft_p95_ms=float(np.percentile([t.ttft_ms for t in active], 95)),
            ttft_p99_ms=float(np.percentile([t.ttft_ms for t in active], 99)),
            tps_p50=float(np.percentile([t.tokens_per_sec for t in active], 50)),
            tps_p95=float(np.percentile([t.tokens_per_sec for t in active], 95)),
            tps_p99=float(np.percentile([t.tokens_per_sec for t in active], 99)),
            peak_memory_gb=float(np.mean([t.peak_memory_gb for t in active])),
            disk_size_gb=disk_gb,
            ram_resident_gb=ram_gb,
            n_trials=len(active),
        )

    def run_all(
        self,
        inference_fn_factory: Callable[[str, SquizdFormatVariant], Callable],
        *,
        models: Optional[List[str]] = None,
        variants: Optional[List[SquizdFormatVariant]] = None,
    ) -> List[SquizdModelResult]:
        """Run all (model, variant) combinations.

        Args:
            inference_fn_factory: Callable that accepts ``(model, variant)``
                and returns a zero-argument inference callable.
            models: Override the model list from config.
            variants: Override the variant list from config.

        Returns:
            Flat list of :class:`SquizdModelResult` objects in
            ``(model, variant)`` order.
        """
        model_list = models if models is not None else self.config.models
        variant_list = variants if variants is not None else self.config.variants
        results: List[SquizdModelResult] = []
        for model in model_list:
            for variant in variant_list:
                fn = inference_fn_factory(model, variant)
                result = self.run_variant(model, variant, fn)
                results.append(result)
        return results

    def to_markdown_table(self, results: List[SquizdModelResult]) -> str:
        """Render *results* as a GitHub-flavoured Markdown table.

        Args:
            results: List of :class:`SquizdModelResult` objects.

        Returns:
            Markdown string containing a formatted results table.
        """
        header = (
            "| Model | Variant | TTFT P50 (ms) | TTFT P95 (ms) | TTFT P99 (ms) "
            "| TPS P50 | TPS P95 | TPS P99 | Peak Mem (GB) | Disk (GB) | RAM (GB) |"
        )
        sep = "|---|---|---|---|---|---|---|---|---|---|---|"
        rows = [header, sep]
        for r in results:
            rows.append(
                f"| {r.model} | {r.variant} "
                f"| {r.ttft_p50_ms:.1f} | {r.ttft_p95_ms:.1f} | {r.ttft_p99_ms:.1f} "
                f"| {r.tps_p50:.1f} | {r.tps_p95:.1f} | {r.tps_p99:.1f} "
                f"| {r.peak_memory_gb:.2f} | {r.disk_size_gb:.2f} | {r.ram_resident_gb:.2f} |"
            )
        return "\n".join(rows)

    def compare_to_gguf(
        self,
        squizd_results: List[SquizdModelResult],
        gguf_results: List[GGUFBaselineResult],
    ) -> List[FormatComparison]:
        """Build :class:`FormatComparison` pairs for all matching models.

        Args:
            squizd_results: SQUIZD benchmark results.
            gguf_results: GGUF Q4_K_M baseline results.

        Returns:
            List of :class:`FormatComparison` for each (model, variant) pair
            that has a matching GGUF baseline.
        """
        gguf_by_model = {g.model: g for g in gguf_results}
        comparisons: List[FormatComparison] = []
        for sr in squizd_results:
            gguf = gguf_by_model.get(sr.model)
            if gguf is not None:
                comparisons.append(FormatComparison.compare(sr, gguf))
        return comparisons

    def comparison_to_markdown(self, comparisons: List[FormatComparison]) -> str:
        """Render *comparisons* as a Markdown table.

        Args:
            comparisons: List of :class:`FormatComparison` objects.

        Returns:
            Markdown string.
        """
        header = (
            "| Model | Variant | TTFT Speedup | TPS Gain | Disk Ratio |"
        )
        sep = "|---|---|---|---|---|"
        rows = [header, sep]
        for c in comparisons:
            rows.append(
                f"| {c.squizd.model} | {c.squizd.variant} "
                f"| {c.ttft_speedup:.2f}× | {c.tps_gain:+.1%} | {c.disk_ratio:.2f}× |"
            )
        return "\n".join(rows)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _collect_trials(
        self,
        model: str,
        inference_fn: Callable[[], Tuple[float, int, float]],
    ) -> List[TrialResult]:
        """Run all trials for *model* and return the full list."""
        total_trials = self.config.warmup_trials + self.config.n_trials
        results: List[TrialResult] = []
        for i in range(total_trials):
            is_warmup = i < self.config.warmup_trials
            t0 = time.perf_counter()
            try:
                ttft_ms, n_tokens, peak_gb = inference_fn()
            except Exception:  # noqa: BLE001
                ttft_ms, n_tokens, peak_gb = (0.0, 0, 0.0)
            total_ms = (time.perf_counter() - t0) * 1000.0
            actual_total = max(total_ms, ttft_ms)
            tps = n_tokens / (actual_total / 1000.0) if actual_total > 0 else 0.0
            results.append(
                TrialResult(
                    trial_idx=i,
                    ttft_ms=ttft_ms,
                    total_ms=actual_total,
                    tokens_generated=n_tokens,
                    tokens_per_sec=tps,
                    peak_memory_gb=peak_gb,
                    warmup=is_warmup,
                )
            )
        return results


# ---------------------------------------------------------------------------
# File-system helpers
# ---------------------------------------------------------------------------

def _measure_disk_size(path: Optional[Path]) -> float:
    """Return the file size of *path* in GB, or 0.0 if unavailable."""
    if path is None:
        return 0.0
    try:
        return os.path.getsize(path) / (1024 ** 3)
    except OSError:
        return 0.0


def _measure_ram_resident() -> float:
    """Return a best-effort resident RAM estimate for the current process in GB."""
    try:
        import resource
        usage_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # On macOS ru_maxrss is in bytes (not KB as on Linux).
        import platform
        if platform.system() == "Darwin":
            return usage_kb / (1024 ** 3)
        return usage_kb / (1024 ** 2)  # KB → GB on Linux
    except Exception:  # noqa: BLE001
        return 0.0
