"""BudgetForcingDecoder: s1-style per-request thinking budget control (arXiv 2501.12599).

Muennighoff et al. (Stanford, 2025).  Append a continuation token ("Wait") to extend
reasoning when the budget has not yet been reached; inject a commitment trigger
("Final Answer:") to cap the thinking chain at a configurable token limit.  Zero retraining
required — purely prompt-engineering at the serving layer.

Reference: Muennighoff et al., "s1: Simple Test-Time Scaling", arXiv 2501.12599, 2025.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

__all__ = [
    "BudgetForcingConfig",
    "BudgetForcingState",
    "BudgetForcingDecoder",
]

# ---------------------------------------------------------------------------
# Default token strings (model-agnostic placeholders; replace per model)
# ---------------------------------------------------------------------------
_DEFAULT_WAIT_TOKEN = "Wait"
_DEFAULT_COMMIT_TOKEN = "Final Answer:"
_DEFAULT_THINK_OPEN = "<think>"
_DEFAULT_THINK_CLOSE = "</think>"


@dataclass
class BudgetForcingConfig:
    """Configuration for :class:`BudgetForcingDecoder`.

    Attributes:
        max_thinking_tokens: Hard token budget for the thinking segment.
        wait_token: String injected to nudge the model to continue reasoning.
        commit_token: String injected to trigger commitment at the budget limit.
        soft_ramp_start: Fraction of budget at which temperature begins rising.
        soft_ramp_max_temp: Maximum temperature multiplier at budget exhaustion.
        think_open_token: Marker that begins a thinking segment.
        think_close_token: Marker that ends a thinking segment.
        seed: Unused; retained for API consistency.
    """

    max_thinking_tokens: int = 512
    wait_token: str = _DEFAULT_WAIT_TOKEN
    commit_token: str = _DEFAULT_COMMIT_TOKEN
    soft_ramp_start: float = 0.8
    soft_ramp_max_temp: float = 2.0
    think_open_token: str = _DEFAULT_THINK_OPEN
    think_close_token: str = _DEFAULT_THINK_CLOSE
    seed: int = 0

    def __post_init__(self) -> None:
        if self.max_thinking_tokens < 1:
            raise ValueError(
                f"max_thinking_tokens must be ≥ 1, got {self.max_thinking_tokens}"
            )
        if not 0.0 < self.soft_ramp_start < 1.0:
            raise ValueError(
                f"soft_ramp_start must be in (0, 1), got {self.soft_ramp_start}"
            )
        if self.soft_ramp_max_temp <= 1.0:
            raise ValueError(
                f"soft_ramp_max_temp must be > 1.0, got {self.soft_ramp_max_temp}"
            )


@dataclass
class BudgetForcingState:
    """Mutable per-request state tracked by :class:`BudgetForcingDecoder`.

    Attributes:
        thinking_tokens_used: Count of tokens generated inside the thinking segment.
        committed: True once the commitment trigger has been injected.
        in_thinking_segment: True if the token stream is currently inside ``<think>``.
        injections: List of injected strings in order (Wait tokens + final commit).
    """

    thinking_tokens_used: int = 0
    committed: bool = False
    in_thinking_segment: bool = False
    injections: List[str] = field(default_factory=list)

    @property
    def budget_exhausted(self) -> bool:
        return self.committed


class BudgetForcingDecoder:
    """Per-request thinking-budget controller.

    Tracks the thinking-segment token count for a single generation request
    and decides what string (if any) to inject after each decoded token.

    Usage::

        cfg = BudgetForcingConfig(max_thinking_tokens=256)
        decoder = BudgetForcingDecoder(cfg)
        state = decoder.new_state()
        for token in model_stream:
            injection, temp_mult = decoder.step(token, state)
            if injection:
                feed_to_model(injection)

    """

    def __init__(self, config: BudgetForcingConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def new_state(self) -> BudgetForcingState:
        """Create fresh per-request state."""
        return BudgetForcingState()

    def step(
        self,
        token: str,
        state: BudgetForcingState,
    ) -> Tuple[Optional[str], float]:
        """Process one decoded *token* and return (injection, temperature_multiplier).

        Parameters
        ----------
        token:
            The decoded token string.
        state:
            Per-request mutable state.

        Returns
        -------
        injection:
            String to inject into the prompt/context before the next decode step,
            or ``None`` if nothing should be injected.
        temperature_multiplier:
            Scale factor to apply to the base temperature for the *next* token.
        """
        cfg = self.config

        # Track thinking segment entry/exit
        if cfg.think_open_token in token:
            state.in_thinking_segment = True
        if cfg.think_close_token in token:
            state.in_thinking_segment = False

        if state.committed:
            return None, 1.0

        if state.in_thinking_segment:
            state.thinking_tokens_used += 1

        # Compute soft-ramp temperature multiplier
        temp_mult = self._temperature_multiplier(state)

        # Hard budget: inject commit trigger
        if state.thinking_tokens_used >= cfg.max_thinking_tokens:
            state.committed = True
            state.injections.append(cfg.commit_token)
            return cfg.commit_token, temp_mult

        return None, temp_mult

    def should_extend(self, state: BudgetForcingState) -> bool:
        """Return True if a Wait token should be injected to extend reasoning."""
        if state.committed:
            return False
        return state.in_thinking_segment and (
            state.thinking_tokens_used < self.config.max_thinking_tokens
        )

    def inject_wait(self, state: BudgetForcingState) -> str:
        """Inject a Wait token and record it in *state*.  Returns token string."""
        token = self.config.wait_token
        state.injections.append(token)
        return token

    def budget_fraction(self, state: BudgetForcingState) -> float:
        """Fraction of the thinking budget consumed (0.0 – 1.0)."""
        return min(
            state.thinking_tokens_used / self.config.max_thinking_tokens, 1.0
        )

    def reset(self, state: BudgetForcingState) -> None:
        """Reset *state* for reuse in a new request."""
        state.thinking_tokens_used = 0
        state.committed = False
        state.in_thinking_segment = False
        state.injections.clear()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _temperature_multiplier(self, state: BudgetForcingState) -> float:
        """Linearly ramp temperature from 1.0 to soft_ramp_max_temp."""
        cfg = self.config
        frac = self.budget_fraction(state)
        ramp_start = cfg.soft_ramp_start
        if frac <= ramp_start:
            return 1.0
        ramp_frac = (frac - ramp_start) / (1.0 - ramp_start)
        return 1.0 + ramp_frac * (cfg.soft_ramp_max_temp - 1.0)
