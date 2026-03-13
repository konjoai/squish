#!/usr/bin/env python3
"""
squish/token_budget_gate.py

TokenBudgetGate — Hard per-request token budget with graceful truncation.

Every inference request is assigned a maximum number of output tokens.  The
:class:`TokenBudgetGate` enforces that limit token-by-token: callers invoke
:meth:`tick` once per generated token and check the boolean return value to
decide whether to continue or stop.

Two enforcement modes are supported:

* ``"hard"`` (default) — :meth:`tick` returns ``False`` the moment the budget
  is exhausted, providing a hard stop.
* ``"soft"`` — :meth:`tick` returns ``False`` when the budget is exhausted,
  but the caller may choose to treat this as advisory; the gate will not raise
  an exception on further ``tick`` calls after exhaustion.

An optional ``warn_at_fraction`` threshold (default 0.9) emits a warning into
the stats when the fraction of tokens used first crosses that threshold,
allowing upstream systems to prepare for imminent termination.

Example usage::

    from squish.token.token_budget_gate import BudgetPolicy, TokenBudgetGate

    policy = BudgetPolicy(mode="hard", warn_at_fraction=0.8)
    gate = TokenBudgetGate(max_tokens=10, policy=policy)

    for step in range(15):
        ok = gate.tick()
        if not ok:
            print(f"budget exhausted at step {step}")
            break

    print(f"fraction used: {gate.fraction_used():.0%}")
    print(gate.stats)
"""

from __future__ import annotations

__all__ = [
    "BudgetPolicy",
    "TokenBudgetGate",
    "BudgetGateStats",
]

from dataclasses import dataclass

import numpy as np  # noqa: F401  — imported for dtype compatibility in future extensions

_VALID_MODES = frozenset({"hard", "soft"})


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------


@dataclass
class BudgetPolicy:
    """Enforcement policy for a :class:`TokenBudgetGate`.

    Attributes:
        mode:             ``"hard"`` or ``"soft"``.  In hard mode, generation
                          is expected to stop immediately when the budget is
                          exhausted.  In soft mode, the gate signals exhaustion
                          but does not mandate termination.
        soft_penalty:     Penalty weight applied per over-budget token in soft
                          mode (unused in hard mode).  Reserved for future
                          implementations that adjust log-probabilities.
                          Must be in [0, 1].
        warn_at_fraction: Fraction of budget consumed at which a warning is
                          first recorded in stats.  Must be in (0, 1].
    """

    mode: str = "hard"
    soft_penalty: float = 0.1
    warn_at_fraction: float = 0.9

    def __post_init__(self) -> None:
        if self.mode not in _VALID_MODES:
            raise ValueError(
                f"mode must be one of {sorted(_VALID_MODES)}, got '{self.mode}'"
            )
        if not (0.0 <= self.soft_penalty <= 1.0):
            raise ValueError(
                f"soft_penalty must be in [0, 1], got {self.soft_penalty}"
            )
        if not (0.0 < self.warn_at_fraction <= 1.0):
            raise ValueError(
                f"warn_at_fraction must be in (0, 1], got {self.warn_at_fraction}"
            )


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


@dataclass
class BudgetGateStats:
    """Cumulative statistics across all requests processed by a gate.

    Attributes:
        total_requests:      Total number of :meth:`reset` calls (i.e. request
                             boundaries).
        total_tokens_gated:  Total tokens processed across all requests.
        hard_stops:          Number of requests that hit the hard limit (budget
                             fully exhausted in ``"hard"`` mode).
        warnings_issued:     Number of requests that triggered the
                             ``warn_at_fraction`` threshold.
    """

    total_requests: int = 0
    total_tokens_gated: int = 0
    hard_stops: int = 0
    warnings_issued: int = 0


# ---------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------


class TokenBudgetGate:
    """Per-request token budget enforcer.

    Maintains a running count of tokens consumed and signals the caller when
    the budget ceiling is reached.  The gate is designed to be reused across
    multiple requests via :meth:`reset`.

    Args:
        max_tokens: Maximum number of tokens allowed per request.  Must be >= 1.
        policy:     A :class:`BudgetPolicy` instance describing the enforcement
                    behaviour.
    """

    def __init__(self, max_tokens: int, policy: BudgetPolicy) -> None:
        if max_tokens < 1:
            raise ValueError(f"max_tokens must be >= 1, got {max_tokens}")
        self._max_tokens = max_tokens
        self._policy = policy
        self._used: int = 0
        self._warned: bool = False
        self._stats = BudgetGateStats()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def tick(self, n_tokens: int = 1) -> bool:
        """Consume *n_tokens* from the budget and report availability.

        In ``"hard"`` mode, returns ``False`` as soon as the cumulative token
        count reaches or exceeds ``max_tokens``.  In ``"soft"`` mode, returns
        ``False`` at the same threshold but does not prevent the caller from
        continuing.

        Emits a warning (increments ``stats.warnings_issued``) the first time
        the consumed fraction exceeds ``policy.warn_at_fraction``.

        Args:
            n_tokens: Number of tokens to consume in this call.  Must be >= 1.

        Returns:
            ``True`` if the budget is still available after this tick,
            ``False`` if the budget has been reached or exceeded.

        Raises:
            ValueError: if *n_tokens* is < 1.
        """
        if n_tokens < 1:
            raise ValueError(f"n_tokens must be >= 1, got {n_tokens}")

        if self._used >= self._max_tokens:
            # Already exhausted — do not consume further but still signal.
            return False

        self._used += n_tokens
        self._stats.total_tokens_gated += n_tokens

        # Issue warning when warn threshold first crossed.
        if not self._warned and self.fraction_used() >= self._policy.warn_at_fraction:
            self._warned = True
            self._stats.warnings_issued += 1

        available = self._used < self._max_tokens
        if not available and self._policy.mode == "hard":
            self._stats.hard_stops += 1

        return available

    def remaining(self) -> int:
        """Number of tokens remaining in the current request's budget.

        Returns 0 when the budget is fully consumed.
        """
        return max(0, self._max_tokens - self._used)

    def fraction_used(self) -> float:
        """Fraction of the budget consumed so far (0.0–1.0+).

        May exceed 1.0 if :meth:`tick` was called with ``n_tokens > 1`` and
        the batch crossed the boundary in a single step.
        """
        return self._used / self._max_tokens

    def reset(self) -> None:
        """Reset the gate for a new request.

        Clears the token counter and the warning flag.  Increments
        ``stats.total_requests``.
        """
        self._used = 0
        self._warned = False
        self._stats.total_requests += 1

    @property
    def is_exhausted(self) -> bool:
        """``True`` if the current request's budget has been fully consumed."""
        return self._used >= self._max_tokens

    @property
    def stats(self) -> BudgetGateStats:
        """Cumulative statistics across all requests processed by this gate."""
        return self._stats

    @property
    def max_tokens(self) -> int:
        """The maximum token budget this gate was constructed with."""
        return self._max_tokens

    @property
    def tokens_used(self) -> int:
        """Tokens consumed so far in the current request."""
        return self._used
