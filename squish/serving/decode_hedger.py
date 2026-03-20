"""
squish/serving/decode_hedger.py

Hedged decode execution for latency-SLO compliance on Apple Silicon.

Inspired by "The Tail at Scale" (Dean & Barroso, Google, CACM 2013): for
requests with tight p99 latency requirements, launch a parallel redundant
decode at a higher speculation depth.  Return whichever path completes
first; cancel the other.

On M-series chips (especially M3-Ultra / M4-Max) with asymmetric
performance/efficiency core clusters, a second decode stream can run on
idle P-cores while the primary uses the GPU/ANE — hedging against unlucky
rejection cascades in the primary speculative path at near-zero marginal
cost when the second context occupies otherwise-idle compute.

Three scheduling policies
-------------------------
``ALWAYS``
    Always launch a hedge — maximises responsiveness but doubles compute.
``THRESHOLD``
    Hedge when the estimated remaining token count exceeds
    ``config.token_threshold`` — good for long-form generation SLOs.
``ADAPTIVE``
    Hedge when the rolling p99 latency exceeds ``config.p99_target_ms``.
    Self-calibrates from observed request latency history.

Reference
---------
Dean, J., & Barroso, L. A. (2013). The tail at scale. CACM 56(2), 74–80.
Adapted for LLM speculative decode context.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum, auto
from typing import Deque


class HedgePolicy(Enum):
    """Scheduling policy for the decode hedger."""

    ALWAYS = auto()
    """Always launch a parallel hedge."""
    THRESHOLD = auto()
    """Hedge when estimated remaining tokens >= token_threshold."""
    ADAPTIVE = auto()
    """Hedge when rolling p99 latency exceeds p99_target_ms."""


@dataclass
class DecodeHedgerConfig:
    """Configuration for the hedged decode controller.

    Parameters
    ----------
    policy:
        Override schedule policy.  Defaults to ``THRESHOLD``.
    token_threshold:
        Minimum remaining tokens to activate a hedge under ``THRESHOLD``
        policy.  Longer outputs benefit most from hedging.
    p99_window_size:
        Rolling window depth for p99 latency measurement used by
        ``ADAPTIVE`` policy.
    p99_target_ms:
        Target p99 latency in milliseconds for ``ADAPTIVE`` policy.
        Hedges are activated when the measured p99 exceeds this value.
    hedge_timeout_ms:
        Maximum milliseconds to wait for the hedged path before declaring
        the primary path the winner and cancelling the hedge.
    max_parallel_hedges:
        Maximum number of concurrently active hedged requests.
    """

    policy: HedgePolicy = HedgePolicy.THRESHOLD
    token_threshold: int = 64
    p99_window_size: int = 100
    p99_target_ms: float = 500.0
    hedge_timeout_ms: float = 2_000.0
    max_parallel_hedges: int = 2

    def __post_init__(self) -> None:
        if self.token_threshold < 1:
            raise ValueError("token_threshold must be >= 1")
        if self.p99_window_size < 10:
            raise ValueError("p99_window_size must be >= 10")
        if self.p99_target_ms <= 0:
            raise ValueError("p99_target_ms must be positive")
        if self.hedge_timeout_ms <= 0:
            raise ValueError("hedge_timeout_ms must be positive")
        if self.max_parallel_hedges < 1:
            raise ValueError("max_parallel_hedges must be >= 1")


class DecodeHedger:
    """Latency hedger for speculative decode paths.

    Tracks per-request latency history and decides whether to activate a
    parallel redundant decode with higher speculation depth when the primary
    path is at risk of missing a latency SLO.

    This class manages the *policy* and *statistics* of hedging only — it
    does not itself spawn tasks.  Callers check ``should_hedge()``, call
    ``begin_hedge()`` when launching the second path, and call
    ``end_hedge()`` when a path completes.

    Usage
    -----
    ::

        hedger = DecodeHedger(DecodeHedgerConfig(policy=HedgePolicy.ADAPTIVE))
        if hedger.should_hedge(estimated_tokens_left):
            hid = hedger.begin_hedge()
            # ... launch parallel decode ...
            hedger.end_hedge(hid, hedge_won=True, latency_ms=312.5)
        else:
            hedger.record_latency(latency_ms)
    """

    def __init__(self, config: DecodeHedgerConfig | None = None) -> None:
        self.config = config or DecodeHedgerConfig()
        self._latency_history: Deque[float] = deque(
            maxlen=self.config.p99_window_size
        )
        self._active_hedges: int = 0
        self._total_hedges_launched: int = 0
        self._total_hedges_won: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def should_hedge(self, estimated_remaining_tokens: int) -> bool:
        """Decide whether to launch a parallel hedge for an in-flight request.

        Parameters
        ----------
        estimated_remaining_tokens:
            Best-effort estimate of tokens left to generate.

        Returns
        -------
        bool
            True → caller should launch a parallel decode path.
        """
        if self._active_hedges >= self.config.max_parallel_hedges:
            return False

        policy = self.config.policy
        if policy == HedgePolicy.ALWAYS:
            return True
        if policy == HedgePolicy.THRESHOLD:
            return estimated_remaining_tokens >= self.config.token_threshold
        if policy == HedgePolicy.ADAPTIVE:
            return self._adaptive_should_hedge()
        return False

    def begin_hedge(self) -> int:
        """Register that a hedged decode path has been launched.

        Returns
        -------
        int
            A hedge ID to pass back to ``end_hedge``.
        """
        self._active_hedges += 1
        self._total_hedges_launched += 1
        return self._total_hedges_launched

    def end_hedge(
        self, hedge_id: int, hedge_won: bool, latency_ms: float
    ) -> None:
        """Record the outcome of a hedged decode.

        Parameters
        ----------
        hedge_id:
            The ID returned by ``begin_hedge``.
        hedge_won:
            True when the hedged path finished before the primary path.
        latency_ms:
            End-to-end request latency in milliseconds.
        """
        self._active_hedges = max(0, self._active_hedges - 1)
        self._latency_history.append(latency_ms)
        if hedge_won:
            self._total_hedges_won += 1

    def record_latency(self, latency_ms: float) -> None:
        """Record a non-hedged request latency for adaptive-policy calibration.

        Parameters
        ----------
        latency_ms:
            End-to-end latency of a completed (non-hedged) request.
        """
        self._latency_history.append(latency_ms)

    def reset_stats(self) -> None:
        """Reset statistics without affecting currently active hedges."""
        self._latency_history.clear()
        self._total_hedges_launched = 0
        self._total_hedges_won = 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def p99_latency_ms(self) -> float:
        """Estimated p99 latency from the rolling window (0.0 if < 2 samples)."""
        n = len(self._latency_history)
        if n < 2:
            return 0.0
        sorted_lat = sorted(self._latency_history)
        idx = min(int(0.99 * n), n - 1)
        return float(sorted_lat[idx])

    @property
    def p50_latency_ms(self) -> float:
        """Estimated median latency from the rolling window."""
        n = len(self._latency_history)
        if n == 0:
            return 0.0
        sorted_lat = sorted(self._latency_history)
        return float(sorted_lat[n // 2])

    @property
    def hedge_win_rate(self) -> float:
        """Fraction of hedges that completed before the primary path."""
        if self._total_hedges_launched == 0:
            return 0.0
        return self._total_hedges_won / self._total_hedges_launched

    @property
    def active_hedges(self) -> int:
        """Count of currently active (in-flight) hedged paths."""
        return self._active_hedges

    @property
    def total_hedges_launched(self) -> int:
        """Cumulative count of hedged paths launched since creation / last reset."""
        return self._total_hedges_launched

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _adaptive_should_hedge(self) -> bool:
        if not self._latency_history:
            return False
        return self.p99_latency_ms > self.config.p99_target_ms
