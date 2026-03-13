# [Experimental] This module is part of Squish v6+ (Waves 19–26).
# Proof-of-concept quality: API and behaviour may change without notice.
# For stable core functionality see Waves 1–12 in MODULES.md.
#!/usr/bin/env python3
"""
squish/health_check.py

HealthCheck — Inference server degradation detection and alerting.

Monitors rolling latency percentiles and error rates, classifying the
inference server's health into OK, DEGRADED, or CRITICAL states.  A
:class:`HealthMetric` encapsulates a single measurement alongside its
warn/critical thresholds, while :class:`InferenceHealthMonitor` aggregates
multiple metrics from live request telemetry.

The rolling window covers the most recent 1 000 requests so that old
behaviour does not mask current problems.  Health transitions occur the
moment a threshold is crossed — there is no hysteresis by default.

Example usage::

    from squish.serving.health_check import InferenceHealthMonitor, HealthState

    monitor = InferenceHealthMonitor(
        warn_latency_ms=500.0,
        crit_latency_ms=2000.0,
        warn_error_rate=0.05,
        crit_error_rate=0.20,
    )
    for _ in range(100):
        monitor.record_request(latency_ms=300.0, success=True)
    monitor.record_request(latency_ms=2500.0, success=False)

    health = monitor.overall_health()
    print(f"health={health!r}, p99={monitor.stats.p99_latency_ms:.1f} ms")
    for metric in monitor.get_metrics():
        print(f"  {metric.name}={metric.value:.4f}  state={metric.state}")
"""

from __future__ import annotations

__all__ = [
    "HealthState",
    "HealthMetric",
    "InferenceHealthMonitor",
    "HealthStats",
]

from collections import deque
from dataclasses import dataclass
from typing import List

import numpy as np


# ---------------------------------------------------------------------------
# Internal constants
# ---------------------------------------------------------------------------

_WINDOW_SIZE: int = 1_000  # rolling request window for percentile computation


# ---------------------------------------------------------------------------
# Health state
# ---------------------------------------------------------------------------

class HealthState:
    """Enumeration of inference server health states.

    Attributes:
        OK:       No threshold breached; server operating normally.
        DEGRADED: Warn threshold breached; performance is impaired.
        CRITICAL: Critical threshold breached; service is at risk.
    """

    OK: str = "ok"
    DEGRADED: str = "degraded"
    CRITICAL: str = "critical"


# ---------------------------------------------------------------------------
# Health metric
# ---------------------------------------------------------------------------

@dataclass
class HealthMetric:
    """A single named metric with warn and critical thresholds.

    Attributes:
        name:            Human-readable metric identifier.
        value:           Current measured value.
        threshold_warn:  Value at or above which state becomes DEGRADED.
        threshold_crit:  Value at or above which state becomes CRITICAL.
    """

    name: str
    value: float
    threshold_warn: float
    threshold_crit: float

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("name must be a non-empty string")
        if self.threshold_warn < 0.0:
            raise ValueError(
                f"threshold_warn must be >= 0, got {self.threshold_warn}"
            )
        if self.threshold_crit < self.threshold_warn:
            raise ValueError(
                f"threshold_crit ({self.threshold_crit}) must be >= "
                f"threshold_warn ({self.threshold_warn})"
            )

    @property
    def state(self) -> str:
        """Compute the health state for this metric.

        Compares ``value`` against ``threshold_crit`` first (most severe),
        then ``threshold_warn``, and defaults to OK.

        Returns:
            :attr:`HealthState.CRITICAL` if ``value >= threshold_crit``,
            :attr:`HealthState.DEGRADED` if ``value >= threshold_warn``,
            :attr:`HealthState.OK` otherwise.
        """
        if self.value >= self.threshold_crit:
            return HealthState.CRITICAL
        if self.value >= self.threshold_warn:
            return HealthState.DEGRADED
        return HealthState.OK


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

@dataclass
class HealthStats:
    """Cumulative statistics maintained by :class:`InferenceHealthMonitor`.

    Attributes:
        total_requests:  Total requests recorded (successful + failed).
        total_errors:    Total failed requests recorded.
        p50_latency_ms:  50th-percentile latency over the rolling window (ms).
        p99_latency_ms:  99th-percentile latency over the rolling window (ms).
    """

    total_requests: int = 0
    total_errors: int = 0
    p50_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0

    @property
    def error_rate(self) -> float:
        """Fraction of all recorded requests that failed (0.0–1.0).

        Returns 0.0 when no requests have been recorded yet.
        """
        if self.total_requests == 0:
            return 0.0
        return self.total_errors / self.total_requests


# ---------------------------------------------------------------------------
# Monitor
# ---------------------------------------------------------------------------

class InferenceHealthMonitor:
    """Rolling-window inference server health monitor.

    Tracks latency percentiles (p50, p99) and error rate over a fixed-size
    rolling window of recent requests.  Health is classified as OK, DEGRADED,
    or CRITICAL by comparing live metric values against per-metric thresholds.

    The rolling window for health evaluation covers at most the last
    ``_WINDOW_SIZE`` (1 000) requests.  Cumulative totals are maintained
    separately in :attr:`stats` and are not bounded.

    Args:
        warn_latency_ms:  p99 latency (ms) at which health becomes DEGRADED.
        crit_latency_ms:  p99 latency (ms) at which health becomes CRITICAL.
        warn_error_rate:  Error rate fraction at which health becomes DEGRADED.
        crit_error_rate:  Error rate fraction at which health becomes CRITICAL.

    Raises:
        ValueError: if any threshold is out of range or ordering is violated.
    """

    def __init__(
        self,
        warn_latency_ms: float = 500.0,
        crit_latency_ms: float = 2_000.0,
        warn_error_rate: float = 0.05,
        crit_error_rate: float = 0.20,
    ) -> None:
        if warn_latency_ms <= 0.0:
            raise ValueError(
                f"warn_latency_ms must be > 0, got {warn_latency_ms}"
            )
        if crit_latency_ms < warn_latency_ms:
            raise ValueError(
                f"crit_latency_ms ({crit_latency_ms}) must be >= "
                f"warn_latency_ms ({warn_latency_ms})"
            )
        if not (0.0 <= warn_error_rate <= 1.0):
            raise ValueError(
                f"warn_error_rate must be in [0, 1], got {warn_error_rate}"
            )
        if not (warn_error_rate <= crit_error_rate <= 1.0):
            raise ValueError(
                f"crit_error_rate ({crit_error_rate}) must be in "
                f"[warn_error_rate, 1.0]"
            )

        self._warn_latency_ms = warn_latency_ms
        self._crit_latency_ms = crit_latency_ms
        self._warn_error_rate = warn_error_rate
        self._crit_error_rate = crit_error_rate

        # Rolling windows (bounded to _WINDOW_SIZE most recent requests).
        self._latency_window: deque[float] = deque(maxlen=_WINDOW_SIZE)
        self._error_window: deque[bool] = deque(maxlen=_WINDOW_SIZE)

        self._stats = HealthStats()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_request(self, latency_ms: float, success: bool) -> None:
        """Record the outcome of a completed inference request.

        Appends the latency and success flag to the rolling windows and
        recomputes rolling p50/p99 percentiles in :attr:`stats`.

        Args:
            latency_ms: End-to-end request latency in milliseconds (>= 0).
            success:    ``True`` if the request completed without error.

        Raises:
            ValueError: if ``latency_ms`` is negative.
        """
        if latency_ms < 0.0:
            raise ValueError(
                f"latency_ms must be >= 0, got {latency_ms}"
            )

        self._latency_window.append(latency_ms)
        self._error_window.append(not success)

        self._stats.total_requests += 1
        if not success:
            self._stats.total_errors += 1

        # Recompute rolling percentiles from the bounded window.
        arr = np.array(self._latency_window, dtype=np.float64)
        self._stats.p50_latency_ms = float(np.percentile(arr, 50))
        self._stats.p99_latency_ms = float(np.percentile(arr, 99))

    def overall_health(self) -> str:
        """Return the worst health state across all tracked metrics.

        Evaluates each metric returned by :meth:`get_metrics` and returns
        the most severe state found.

        Returns:
            :attr:`HealthState.CRITICAL` if any metric is critical,
            :attr:`HealthState.DEGRADED` if any metric is degraded,
            :attr:`HealthState.OK` otherwise.
        """
        metrics = self.get_metrics()
        states = {m.state for m in metrics}
        if HealthState.CRITICAL in states:
            return HealthState.CRITICAL
        if HealthState.DEGRADED in states:
            return HealthState.DEGRADED
        return HealthState.OK

    def get_metrics(self) -> List[HealthMetric]:
        """Return the current set of tracked health metrics.

        Metrics are computed from the rolling request window so that they
        reflect recent behaviour rather than lifetime averages.

        Returns:
            A list containing:
            - ``p99_latency_ms``: rolling p99 latency vs latency thresholds.
            - ``error_rate``: rolling error rate vs error-rate thresholds.
        """
        # Rolling error rate from the bounded window.
        rolling_error_rate = (
            sum(self._error_window) / len(self._error_window)
            if self._error_window
            else 0.0
        )
        return [
            HealthMetric(
                name="p99_latency_ms",
                value=self._stats.p99_latency_ms,
                threshold_warn=self._warn_latency_ms,
                threshold_crit=self._crit_latency_ms,
            ),
            HealthMetric(
                name="error_rate",
                value=rolling_error_rate,
                threshold_warn=self._warn_error_rate,
                threshold_crit=self._crit_error_rate,
            ),
        ]

    @property
    def stats(self) -> HealthStats:
        """Cumulative and rolling statistics updated on each request."""
        return self._stats


# ---------------------------------------------------------------------------
# Internal utilities
# ---------------------------------------------------------------------------

# (No private helpers required beyond numpy percentile calls.)
