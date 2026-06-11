"""Inference Quality Monitor — rolling-window metrics for squish requests.

Records per-request latency, tokens-per-second, TTFT, and error rates.
Exposes P50/P95/P99 percentile stats via :func:`get_quality_monitor`.

Public API
──────────
    RequestMetric   — frozen dataclass, one record per completed request
    QualityStats    — frozen dataclass, per-model aggregated statistics
    QualityReport   — frozen dataclass, all-models report for a window
    QualityMonitor  — thread-safe rolling-window tracker
    get_quality_monitor() → QualityMonitor singleton
"""
from __future__ import annotations

import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

__all__ = [
    "RequestMetric",
    "QualityStats",
    "QualityReport",
    "QualityMonitor",
    "get_quality_monitor",
    "record_completion_metric",
    "quality_response_dict",
]


# ── Percentile helper ─────────────────────────────────────────────────────────


def _percentile(values: list[float], p: float) -> float:
    """Return the p-th percentile of *values* using linear interpolation.

    Math: i = (p / 100) * (n - 1). The result is interpolated between
    floor(i) and ceil(i) so that the output matches numpy.percentile(values, p).

    Parameters
    ----------
    values:
        Pre-sorted list of floats. Must be non-empty.
    p:
        Percentile in [0, 100].
    """
    if not values:
        return 0.0
    n = len(values)
    if n == 1:
        return values[0]
    # i = (p / 100) * (n - 1)
    i = (p / 100.0) * (n - 1)
    lo = int(math.floor(i))
    hi = int(math.ceil(i))
    frac = i - lo
    return values[lo] * (1.0 - frac) + values[hi] * frac


# ── Data classes ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class RequestMetric:
    """Immutable record of a single completed inference request.

    Attributes
    ----------
    timestamp:
        ``time.monotonic()`` at request completion.
    model_id:
        Which model served this request.
    latency_ms:
        Wall-clock time from first token requested to last (milliseconds).
    ttft_ms:
        Time-to-first-token in milliseconds; ``None`` if unavailable.
    tokens_generated:
        Total tokens produced.
    tokens_per_sec:
        ``tokens_generated / (latency_ms / 1000)``.
    success:
        ``False`` if an exception was raised during generation.
    error_type:
        Exception class name, or ``None`` on success.
    """

    timestamp: float
    model_id: str
    latency_ms: float
    ttft_ms: float | None
    tokens_generated: int
    tokens_per_sec: float
    success: bool
    error_type: str | None


@dataclass(frozen=True)
class QualityStats:
    """Aggregated per-model quality statistics over a rolling window.

    Attributes
    ----------
    model_id:
        The model these stats describe.
    window_seconds:
        Duration of the rolling window used for aggregation.
    n_requests:
        Total requests in window.
    n_errors:
        Requests where ``success=False``.
    error_rate:
        ``n_errors / max(1, n_requests)``.
    latency_p50, latency_p95, latency_p99, latency_mean:
        Latency percentiles and mean in milliseconds; ``0.0`` if no requests.
    tokens_per_sec_p50, tokens_per_sec_mean:
        Throughput stats in tokens/second.
    ttft_p50, ttft_p95:
        Time-to-first-token percentiles; ``None`` if no TTFT data available.
    generated_at:
        ISO-8601 UTC timestamp of report generation.
    """

    model_id: str
    window_seconds: int
    n_requests: int
    n_errors: int
    error_rate: float

    latency_p50: float
    latency_p95: float
    latency_p99: float
    latency_mean: float

    tokens_per_sec_p50: float
    tokens_per_sec_mean: float

    ttft_p50: float | None
    ttft_p95: float | None

    generated_at: str

    def to_dict(self) -> dict:
        """Return a JSON-serialisable plain-dict representation."""
        return {
            "model_id":            self.model_id,
            "window_seconds":      self.window_seconds,
            "n_requests":          self.n_requests,
            "n_errors":            self.n_errors,
            "error_rate":          self.error_rate,
            "latency_p50_ms":      self.latency_p50,
            "latency_p95_ms":      self.latency_p95,
            "latency_p99_ms":      self.latency_p99,
            "latency_mean_ms":     self.latency_mean,
            "tokens_per_sec_p50":  self.tokens_per_sec_p50,
            "tokens_per_sec_mean": self.tokens_per_sec_mean,
            "ttft_p50_ms":         self.ttft_p50,
            "ttft_p95_ms":         self.ttft_p95,
            "generated_at":        self.generated_at,
        }


@dataclass(frozen=True)
class QualityReport:
    """Full quality report for all models in the rolling window.

    Attributes
    ----------
    window_seconds:
        Duration of the rolling window.
    total_requests:
        Sum of ``n_requests`` across all models.
    models:
        One :class:`QualityStats` per distinct ``model_id`` seen in window.
    generated_at:
        ISO-8601 UTC timestamp.
    """

    window_seconds: int
    total_requests: int
    models: list  # list[QualityStats]
    generated_at: str

    def to_dict(self) -> dict:
        """Return a JSON-serialisable plain-dict representation."""
        return {
            "window_seconds":  self.window_seconds,
            "total_requests":  self.total_requests,
            "models":          [m.to_dict() for m in self.models],
            "generated_at":    self.generated_at,
        }


# ── Monitor ───────────────────────────────────────────────────────────────────


class QualityMonitor:
    """Thread-safe rolling-window inference quality tracker.

    Uses a :class:`~collections.deque` bounded by *max_events* (not by time —
    trimming is lazy, on every :meth:`record` call).  Thread safety is provided
    by :class:`threading.Lock`.

    Parameters
    ----------
    window_seconds:
        Default window for :meth:`report` and :meth:`stats_for`.
    max_events:
        Hard upper bound on stored events (FIFO eviction via deque maxlen).
    """

    def __init__(
        self,
        window_seconds: int = 3600,
        max_events: int = 10_000,
    ) -> None:
        """Initialise the monitor with the given window and capacity."""
        self._window_seconds = window_seconds
        self._events: deque = deque(maxlen=max_events)
        self._lock = threading.Lock()

    # ── Public API ────────────────────────────────────────────────────────

    def record(self, metric: RequestMetric) -> None:
        """Add *metric* and trim events older than the window.

        Thread-safe. Never raises — errors are swallowed with a warning so
        that monitoring never interrupts generation.
        """
        try:
            cutoff = metric.timestamp - self._window_seconds
            with self._lock:
                self._events.append(metric)
                _trim_deque(self._events, cutoff)
        except Exception:  # noqa: BLE001
            import logging as _logging
            _logging.getLogger(__name__).warning(
                "QualityMonitor.record() failed — metric dropped",
                exc_info=True,
            )

    def report(self, window_seconds: int | None = None) -> QualityReport:
        """Compute per-model stats for *window_seconds* (default: self.window_seconds).

        Always returns a valid :class:`QualityReport`; empty stats are valid.
        """
        win = window_seconds if window_seconds is not None else self._window_seconds
        now = time.monotonic()
        cutoff = now - win
        with self._lock:
            events = [e for e in self._events if e.timestamp >= cutoff]

        generated_at = _utc_now()
        model_ids = dict.fromkeys(e.model_id for e in events)  # ordered, unique
        model_stats = [
            _compute_stats(
                [e for e in events if e.model_id == mid],
                mid,
                win,
                generated_at,
            )
            for mid in model_ids
        ]
        return QualityReport(
            window_seconds=win,
            total_requests=sum(s.n_requests for s in model_stats),
            models=model_stats,
            generated_at=generated_at,
        )

    def stats_for(
        self,
        model_id: str,
        window_seconds: int | None = None,
    ) -> QualityStats | None:
        """Return per-model stats, or ``None`` if no events for *model_id* in window."""
        win = window_seconds if window_seconds is not None else self._window_seconds
        now = time.monotonic()
        cutoff = now - win
        with self._lock:
            events = [
                e for e in self._events
                if e.model_id == model_id and e.timestamp >= cutoff
            ]
        if not events:
            return None
        return _compute_stats(events, model_id, win, _utc_now())

    def clear(self) -> None:
        """Reset all stored metrics."""
        with self._lock:
            self._events.clear()


# ── Private helpers ───────────────────────────────────────────────────────────


def _trim_deque(dq: deque, cutoff: float) -> None:
    """Remove events from the left of *dq* whose timestamp < *cutoff*."""
    while dq and dq[0].timestamp < cutoff:
        dq.popleft()


def _utc_now() -> str:
    """Return current UTC time as ISO-8601 string."""
    return datetime.now(tz=timezone.utc).isoformat()


def _compute_stats(
    events: list[RequestMetric],
    model_id: str,
    window_seconds: int,
    generated_at: str,
) -> QualityStats:
    """Compute :class:`QualityStats` from a list of events for one model."""
    n = len(events)
    n_errors = sum(1 for e in events if not e.success)

    if n == 0:
        return QualityStats(
            model_id=model_id,
            window_seconds=window_seconds,
            n_requests=0,
            n_errors=0,
            error_rate=0.0,
            latency_p50=0.0,
            latency_p95=0.0,
            latency_p99=0.0,
            latency_mean=0.0,
            tokens_per_sec_p50=0.0,
            tokens_per_sec_mean=0.0,
            ttft_p50=None,
            ttft_p95=None,
            generated_at=generated_at,
        )

    latencies = sorted(e.latency_ms for e in events)
    tps_vals  = sorted(e.tokens_per_sec for e in events)
    ttft_vals = sorted(e.ttft_ms for e in events if e.ttft_ms is not None)

    return QualityStats(
        model_id=model_id,
        window_seconds=window_seconds,
        n_requests=n,
        n_errors=n_errors,
        error_rate=n_errors / max(1, n),
        latency_p50=_percentile(latencies, 50),
        latency_p95=_percentile(latencies, 95),
        latency_p99=_percentile(latencies, 99),
        latency_mean=sum(latencies) / n,
        tokens_per_sec_p50=_percentile(tps_vals, 50),
        tokens_per_sec_mean=sum(tps_vals) / n,
        ttft_p50=_percentile(ttft_vals, 50) if ttft_vals else None,
        ttft_p95=_percentile(ttft_vals, 95) if ttft_vals else None,
        generated_at=generated_at,
    )


# ── Module-level singleton ────────────────────────────────────────────────────

_monitor_singleton: QualityMonitor | None = None
_singleton_lock = threading.Lock()


def get_quality_monitor() -> QualityMonitor:
    """Return the module-level :class:`QualityMonitor` singleton.

    Created on first call and reused thereafter.  Thread-safe.
    """
    global _monitor_singleton
    if _monitor_singleton is None:
        with _singleton_lock:
            if _monitor_singleton is None:
                _monitor_singleton = QualityMonitor()
    return _monitor_singleton


def record_completion_metric(
    model_id: str,
    duration_s: float,
    ttft_s: float,
    n_tokens: int,
    tps: float,
) -> None:
    """Record a successful completion in the singleton :class:`QualityMonitor`.

    Never raises — swallows and logs any exception so monitoring can never
    interrupt a generation path.

    Parameters
    ----------
    model_id:
        Identifier for the model that served the request.
    duration_s:
        Total request wall-clock duration in seconds.
    ttft_s:
        Time-to-first-token in seconds; ``0`` means unavailable.
    n_tokens:
        Number of tokens generated.
    tps:
        Tokens per second (pre-computed by the caller).
    """
    try:
        get_quality_monitor().record(RequestMetric(
            timestamp=time.monotonic(),
            model_id=model_id or "unknown",
            latency_ms=duration_s * 1000.0,
            ttft_ms=ttft_s * 1000.0 if ttft_s > 0 else None,
            tokens_generated=n_tokens,
            tokens_per_sec=tps,
            success=True,
            error_type=None,
        ))
    except Exception:  # noqa: BLE001
        import logging as _log  # noqa: PLC0415
        _log.getLogger(__name__).warning("Quality monitor metric dropped", exc_info=True)


def quality_response_dict(window: int, model_filter: str = "") -> dict:
    """Build a JSON-serialisable quality report dict for the /v1/quality endpoint.

    Parameters
    ----------
    window:
        Rolling window in seconds; clamped to [60, 86400].
    model_filter:
        When non-empty, restrict results to this model_id.
    """
    win    = max(60, min(86400, window))
    report = get_quality_monitor().report(window_seconds=win)
    if model_filter:
        ms = [m for m in report.models if m.model_id == model_filter]
        return {"window_seconds": win, "total_requests": sum(m.n_requests for m in ms),
                "models": [m.to_dict() for m in ms], "generated_at": report.generated_at}
    result = report.to_dict()
    result["window_seconds"] = win
    return result
