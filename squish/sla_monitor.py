#!/usr/bin/env python3
"""
squish/sla_monitor.py

SLAMonitor — Real-time SLA violation detection and automated remediation.

Tracks latency and error-rate SLOs over a sliding window of recent requests.
When a violation is detected on consecutive checks it escalates from
"warning" to "critical" severity after ``escalation_threshold`` consecutive
violation events.

The module deliberately avoids triggering remediation itself; callers inspect
the returned :class:`SLAViolation` list from :meth:`SLAMonitor.check` and
initiate corrective actions (e.g. shed load, disable draft model, alert) in
their own infrastructure.

Example usage::

    from squish.sla_monitor import SLAMonitor, ViolationPolicy

    policy = ViolationPolicy(
        max_latency_ms=2000.0,
        max_error_rate=0.05,
        violation_window=100,
        escalation_threshold=3,
    )
    monitor = SLAMonitor(policy)

    for _ in range(50):
        monitor.record(latency_ms=500.0, success=True)
    monitor.record(latency_ms=3000.0, success=False)

    violations = monitor.check()
    for v in violations:
        print(f"{v.violation_type} {v.severity}: {v.value:.1f} > {v.threshold:.1f}")
    print(f"healthy={monitor.is_healthy()}")
"""

from __future__ import annotations

__all__ = [
    "ViolationPolicy",
    "ViolationType",
    "SLAViolation",
    "SLAMonitor",
    "SLAStats",
]

import time
from collections import deque
from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------

@dataclass
class ViolationPolicy:
    """SLA thresholds and escalation parameters.

    Attributes:
        max_latency_ms:        p99 latency threshold in milliseconds.
        max_error_rate:        Maximum tolerable error-rate fraction [0, 1].
        violation_window:      Rolling-window size in number of requests.
        escalation_threshold:  Consecutive violation events before severity
                               escalates from "warning" to "critical".
    """

    max_latency_ms: float = 2_000.0
    max_error_rate: float = 0.05
    violation_window: int = 100
    escalation_threshold: int = 3

    def __post_init__(self) -> None:
        if self.max_latency_ms <= 0.0:
            raise ValueError(
                f"max_latency_ms must be > 0, got {self.max_latency_ms}"
            )
        if not (0.0 < self.max_error_rate <= 1.0):
            raise ValueError(
                f"max_error_rate must be in (0, 1], got {self.max_error_rate}"
            )
        if self.violation_window < 1:
            raise ValueError(
                f"violation_window must be >= 1, got {self.violation_window}"
            )
        if self.escalation_threshold < 1:
            raise ValueError(
                f"escalation_threshold must be >= 1, "
                f"got {self.escalation_threshold}"
            )


# ---------------------------------------------------------------------------
# Violation type
# ---------------------------------------------------------------------------

class ViolationType:
    """Enumeration of SLA violation categories.

    Attributes:
        LATENCY:    p99 latency exceeded ``max_latency_ms``.
        ERROR_RATE: Rolling error rate exceeded ``max_error_rate``.
    """

    LATENCY: str = "latency"
    ERROR_RATE: str = "error_rate"


# ---------------------------------------------------------------------------
# Violation record
# ---------------------------------------------------------------------------

@dataclass
class SLAViolation:
    """A single detected SLA violation.

    Attributes:
        violation_type: One of :attr:`ViolationType.LATENCY` or
                        :attr:`ViolationType.ERROR_RATE`.
        severity:       ``"warning"`` or ``"critical"`` depending on
                        how many consecutive violation events have occurred.
        value:          Measured value that exceeded the threshold.
        threshold:      The threshold that was breached.
        timestamp:      Unix timestamp when this violation was detected.
    """

    violation_type: str
    severity: str
    value: float
    threshold: float
    timestamp: float


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

@dataclass
class SLAStats:
    """Cumulative statistics maintained by :class:`SLAMonitor`.

    Attributes:
        total_records:         Total calls to :meth:`SLAMonitor.record`.
        total_violations:      Cumulative violations detected by
                               :meth:`SLAMonitor.check`.
        consecutive_violations: Number of consecutive :meth:`SLAMonitor.check`
                                calls that returned at least one violation.
    """

    total_records: int = 0
    total_violations: int = 0
    consecutive_violations: int = 0


# ---------------------------------------------------------------------------
# Monitor
# ---------------------------------------------------------------------------

class SLAMonitor:
    """Real-time SLA violation detector with escalation.

    Maintains a rolling window of ``violation_window`` requests for both
    p99 latency and error-rate computation.  :meth:`check` examines the
    current window and returns any active violations.

    Severity escalates from ``"warning"`` to ``"critical"`` after
    ``escalation_threshold`` consecutive :meth:`check` calls each return
    at least one violation.

    Args:
        policy: A :class:`ViolationPolicy` instance with SLA thresholds.
    """

    def __init__(self, policy: ViolationPolicy) -> None:
        self._policy = policy
        self._latency_window: deque[float] = deque(
            maxlen=policy.violation_window
        )
        self._error_window: deque[bool] = deque(
            maxlen=policy.violation_window
        )
        self._stats = SLAStats()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record(self, latency_ms: float, success: bool) -> None:
        """Record the outcome of a completed request.

        Appends the latency and success flag to the rolling windows.

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
        self._stats.total_records += 1

    def check(self) -> list[SLAViolation]:
        """Evaluate the current rolling window for SLA violations.

        Computes the p99 latency and error rate over the window and returns
        a :class:`SLAViolation` for each threshold breached.  Severity is
        ``"critical"`` once ``escalation_threshold`` consecutive check calls
        have each returned at least one violation; otherwise ``"warning"``.

        Returns:
            A list of :class:`SLAViolation` instances (may be empty).
        """
        if not self._latency_window:
            # No data yet — cannot determine violations.
            return []

        policy = self._policy
        now = time.time()
        violations: list[SLAViolation] = []

        # Compute rolling metrics.
        arr = np.array(self._latency_window, dtype=np.float64)
        p99_latency = float(np.percentile(arr, 99))
        error_rate = (
            sum(self._error_window) / len(self._error_window)
            if self._error_window
            else 0.0
        )

        severity = (
            "critical"
            if self._stats.consecutive_violations >= policy.escalation_threshold
            else "warning"
        )

        if p99_latency > policy.max_latency_ms:
            violations.append(
                SLAViolation(
                    violation_type=ViolationType.LATENCY,
                    severity=severity,
                    value=p99_latency,
                    threshold=policy.max_latency_ms,
                    timestamp=now,
                )
            )

        if error_rate > policy.max_error_rate:
            violations.append(
                SLAViolation(
                    violation_type=ViolationType.ERROR_RATE,
                    severity=severity,
                    value=error_rate,
                    threshold=policy.max_error_rate,
                    timestamp=now,
                )
            )

        # Update consecutive-violation counter.
        if violations:
            self._stats.consecutive_violations += 1
            self._stats.total_violations += len(violations)
        else:
            self._stats.consecutive_violations = 0

        return violations

    def is_healthy(self) -> bool:
        """Return ``True`` if the most recent :meth:`check` found no violations.

        Calls :meth:`check` internally; callers that need the violation
        details should call :meth:`check` directly rather than using this
        helper to avoid double-counting consecutive-violation state.

        Returns:
            ``True`` iff ``check()`` returns an empty list.
        """
        return len(self.check()) == 0

    @property
    def stats(self) -> SLAStats:
        """Cumulative SLA monitoring statistics (updated in place)."""
        return self._stats
