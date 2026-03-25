"""squish/serving/obs_report.py

Observability report generator for Squish inference server.

Combines span-trace data from squish.telemetry with latency-percentile data
from squish.hardware.production_profiler to produce a structured bottleneck
report with actionable remediation hints.

Public API
----------
detect_bottlenecks(profiler, threshold_ms=200)
    Return list of slow operations whose p99 exceeds *threshold_ms*.

generate_report(profiler, tracer)
    Return a dict suitable for JSON serialisation as the /v1/obs-report body.
"""
from __future__ import annotations

__all__ = ["detect_bottlenecks", "generate_report"]

from typing import Any

# ---------------------------------------------------------------------------
# Remediation hints keyed on operation / span-name prefix
# ---------------------------------------------------------------------------

_REMEDIATION_HINTS: dict[str, str] = {
    "gen.prefill":          "Try `--chunk-prefill-size 64` to split large prefills",
    "gen.tokenize":         "Tokenizer cold — improves after first request",
    "gen.decode_loop":      "Enable `--blazing` for sub-3s TTFT on M3/M4/M5",
    "gen.compress":         "Compression overhead — consider `--no-compress`",
    "gen.speculative":      "Speculative decode draft mismatch; try a larger draft model",
    "gen.prefix_cache":     "Prefix cache miss — warming up; latency improves with reuse",
    "gen.semantic_cache":   "Semantic cache latency; disable with `--no-semantic-cache`",
    "startup.kv_cache_init":"Use `--no-kv-cache` if KV cache not needed",
    "server.model_load":    "Slow model load — try INT4 quantization or `--fast-warmup`",
    "http.chat_completions":"High HTTP handler latency — check request queue depth",
    "ttft_ms":              "High TTFT — try `--chunk-prefill-size 64` or `--blazing`",
    "model_load_ms":        "Slow model load — consider smaller quantization level",
    "decode_step_ms":       "Slow decode — enable `--blazing` or reduce `--max-tokens`",
}


def _hint_for(operation: str) -> str:
    """Return the best-matching remediation hint for *operation*.

    First tries an exact match, then falls back to prefix matching.
    Returns an empty string when no hint is available.
    """
    if operation in _REMEDIATION_HINTS:
        return _REMEDIATION_HINTS[operation]
    for prefix, hint in _REMEDIATION_HINTS.items():
        if operation.startswith(prefix):
            return hint
    return ""


# ---------------------------------------------------------------------------
# detect_bottlenecks
# ---------------------------------------------------------------------------

def detect_bottlenecks(
    profiler: Any,
    threshold_ms: float = 200.0,
) -> list[dict[str, Any]]:
    """Return operations whose p99 latency exceeds *threshold_ms*.

    Parameters
    ----------
    profiler:
        A ``ProductionProfiler`` instance (or ``None`` — returns empty list).
    threshold_ms:
        p99 latency threshold in milliseconds.  Operations at or above this
        value are included in the bottleneck list.

    Returns
    -------
    list[dict]
        Sorted descending by p99_ms.  Each item has keys:
        ``op``, ``p99_ms``, ``mean_ms``, ``n_samples``, ``hint``.
    """
    if profiler is None:
        return []

    bottlenecks: list[dict[str, Any]] = []
    for op, stats in profiler.report().items():
        if stats.p99_ms >= threshold_ms:
            bottlenecks.append({
                "op":        op,
                "p99_ms":    round(stats.p99_ms, 3),
                "mean_ms":   round(stats.mean_ms, 3),
                "n_samples": stats.n_samples,
                "hint":      _hint_for(op),
            })

    bottlenecks.sort(key=lambda b: b["p99_ms"], reverse=True)
    return bottlenecks


# ---------------------------------------------------------------------------
# generate_report
# ---------------------------------------------------------------------------

def generate_report(
    profiler: Any,
    tracer: Any,
    bottleneck_threshold_ms: float = 200.0,
) -> dict[str, Any]:
    """Generate a full observability report as a JSON-serialisable dict.

    Parameters
    ----------
    profiler:
        A ``ProductionProfiler`` instance, or ``None``.
    tracer:
        A ``Tracer`` instance from ``squish.telemetry``, or ``None``.
    bottleneck_threshold_ms:
        Passed through to :func:`detect_bottlenecks`.

    Returns
    -------
    dict with keys:
        ``status``         — ``"ok"`` or ``"degraded"``
        ``bottlenecks``    — list from :func:`detect_bottlenecks`
        ``profile``        — per-operation stats dict (from profiler.to_json_dict())
        ``recent_spans``   — list of the 10 slowest recent spans (from tracer)
        ``profiler_ops``   — list of tracked operation names
    """
    bottlenecks = detect_bottlenecks(profiler, threshold_ms=bottleneck_threshold_ms)
    status = "degraded" if bottlenecks else "ok"

    profile: dict[str, Any] = {}
    profiler_ops: list[str] = []
    if profiler is not None:
        try:
            profile = profiler.to_json_dict()
            profiler_ops = profiler.operations
        except Exception:
            profile = {}
            profiler_ops = []

    recent_spans: list[dict[str, Any]] = []
    if tracer is not None:
        try:
            slowest = tracer.slowest_spans(n=10)
            recent_spans = [s.to_dict() for s in slowest]
        except Exception:
            recent_spans = []

    return {
        "status":       status,
        "bottlenecks":  bottlenecks,
        "profile":      profile,
        "profiler_ops": profiler_ops,
        "recent_spans": recent_spans,
    }
