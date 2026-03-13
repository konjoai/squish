# [Experimental] This module is part of Squish v6+ (Waves 19–26).
# Proof-of-concept quality: API and behaviour may change without notice.
# For stable core functionality see Waves 1–12 in MODULES.md.
"""BudgetSpec — Token-budget-aware speculative decoding.

Under a hard token budget (e.g., max 512 output tokens), standard speculative
decoding wastes verify passes when only a few tokens remain before the budget
is exhausted.  :class:`BudgetSpecDecoder` tracks how many tokens have been
generated and reduces the draft length linearly once generation crosses
``ramp_down_at × total_budget`` tokens, gracefully degrading to single-token
drafts near the limit and preventing overshoot.

This approach is complementary to any draft model: the only change is the
number of draft tokens requested per speculative round.

Usage::

    import numpy as np
    from squish.serving.budget_spec import BudgetSpecDecoder, BudgetConfig

    cfg     = BudgetConfig(total_budget=512, n_draft=5, ramp_down_at=0.9)
    decoder = BudgetSpecDecoder(cfg)

    while not decoder.is_exhausted():
        n = decoder.effective_draft_len()
        # … run draft model for n steps …
        n_accepted = 3  # from verifier
        decoder.step(n_accepted)

    print(decoder.stats)
    decoder.reset()  # ready for the next request
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["BudgetConfig", "BudgetState", "BudgetSpecDecoder", "BudgetStats"]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class BudgetConfig:
    """Configuration for token-budget-aware speculative decoding.

    Attributes:
        total_budget: Maximum number of output tokens allowed per request.
        n_draft: Maximum draft length when the budget is plentiful.
        ramp_down_at: Fraction of ``total_budget`` at which the draft length
            begins to decrease linearly toward 1.  Must be in ``(0, 1]``.
    """

    total_budget: int = 512
    n_draft: int = 5
    ramp_down_at: float = 0.9

    def __post_init__(self) -> None:
        if self.total_budget < 1:
            raise ValueError(f"total_budget must be >= 1; got {self.total_budget}")
        if self.n_draft < 1:
            raise ValueError(f"n_draft must be >= 1; got {self.n_draft}")
        if not (0.0 < self.ramp_down_at <= 1.0):
            raise ValueError(
                f"ramp_down_at must be in (0, 1]; got {self.ramp_down_at}"
            )


# ---------------------------------------------------------------------------
# State / Stats
# ---------------------------------------------------------------------------


@dataclass
class BudgetState:
    """Snapshot of the current decode state.

    Attributes:
        tokens_generated: Number of tokens emitted so far in this request.
        remaining: Tokens remaining before the budget is exhausted.
        current_draft_len: Draft length that would be used on the next call
            to :meth:`BudgetSpecDecoder.effective_draft_len`.
    """

    tokens_generated: int = 0
    remaining: int = 0
    current_draft_len: int = 0


@dataclass
class BudgetStats:
    """Accumulated statistics across all requests.

    Attributes:
        total_requests: Number of requests that have been reset via
            :meth:`BudgetSpecDecoder.reset`.
        total_tokens: Total tokens generated across all requests.
        total_draft_calls: Total number of :meth:`BudgetSpecDecoder.effective_draft_len`
            calls across all requests.
    """

    total_requests: int = 0
    total_tokens: int = 0
    total_draft_calls: int = 0

    @property
    def avg_draft_len(self) -> float:
        """Average tokens generated per draft call across all requests."""
        if self.total_draft_calls == 0:
            return 0.0
        return self.total_tokens / self.total_draft_calls


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------


class BudgetSpecDecoder:
    """Token-budget-aware speculative decode controller.

    Tracks the number of generated tokens and adjusts the draft length so
    that the speculative loop never overshoots the token budget.

    Draft-length policy:

    * ``tokens_generated < ramp_down_at * total_budget``:
      use full ``n_draft``.
    * ``ramp_down_at * total_budget <= tokens_generated < total_budget``:
      linearly ramp down from ``n_draft`` to 1.
    * Always clamp to ``min(computed_len, remaining)`` to prevent overshoot.

    Args:
        config: :class:`BudgetConfig` instance.
    """

    def __init__(self, config: BudgetConfig) -> None:
        self.config = config
        self._tokens_generated: int = 0
        self._stats = BudgetStats()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_draft_len(self) -> int:
        """Compute the effective draft length without updating statistics.

        Returns:
            Integer draft length in ``[0, n_draft]``.  Returns 0 when the
            budget is exhausted.
        """
        cfg = self.config
        remaining = cfg.total_budget - self._tokens_generated
        if remaining <= 0:
            return 0

        ramp_start = cfg.ramp_down_at * cfg.total_budget

        if self._tokens_generated < ramp_start:
            draft_len = cfg.n_draft
        else:
            ramp_window = cfg.total_budget - ramp_start
            if ramp_window <= 0.0:
                draft_len = 1
            else:
                progress = (self._tokens_generated - ramp_start) / ramp_window
                # Linear ramp from n_draft at 0 to 1 at 1
                draft_len = max(1, round(cfg.n_draft * (1.0 - progress) + 1.0 * progress))

        return max(1, min(draft_len, remaining))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def effective_draft_len(self) -> int:
        """Return the current effective draft length and record the call.

        Decreases linearly once token generation crosses ``ramp_down_at``
        of the total budget, and is always clamped to the remaining budget.

        Returns:
            Integer draft length in ``[1, n_draft]``, or 0 if exhausted.
        """
        result = self._compute_draft_len()
        self._stats.total_draft_calls += 1
        return result

    def step(self, n_accepted: int) -> None:
        """Record accepted tokens from one speculative decode round.

        Clamps ``n_accepted`` to the remaining budget to prevent overshoot.

        Args:
            n_accepted: Number of tokens accepted by the verifier.
                Must be >= 0.

        Raises:
            ValueError: If ``n_accepted`` is negative.
        """
        if n_accepted < 0:
            raise ValueError(f"n_accepted must be >= 0; got {n_accepted}")
        remaining = self.config.total_budget - self._tokens_generated
        actual = min(n_accepted, max(0, remaining))
        self._tokens_generated += actual
        self._stats.total_tokens += actual

    def is_exhausted(self) -> bool:
        """Return ``True`` when the token budget has been fully consumed."""
        return self._tokens_generated >= self.config.total_budget

    def reset(self) -> None:
        """Reset internal state for a new request.

        Increments :attr:`BudgetStats.total_requests`.
        """
        self._stats.total_requests += 1
        self._tokens_generated = 0

    @property
    def state(self) -> BudgetState:
        """Read-only snapshot of the current decode state.

        Does *not* update any statistics counters.
        """
        cfg = self.config
        remaining = max(0, cfg.total_budget - self._tokens_generated)
        return BudgetState(
            tokens_generated=self._tokens_generated,
            remaining=remaining,
            current_draft_len=self._compute_draft_len(),
        )

    @property
    def stats(self) -> BudgetStats:
        """Current accumulated :class:`BudgetStats`."""
        return self._stats
