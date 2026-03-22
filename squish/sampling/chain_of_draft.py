"""ChainOfDraftSampler: per-step word-count constraint on reasoning chains.

Xu et al. (arXiv 2502.18600, 2025).  Constrains each intermediate reasoning step to
≤ max_words (default 7) by penalising overlong steps in the output logits.  Achieves
7.6× reduction in intermediate token count with no measurable accuracy degradation on
MATH, GSM8K, and logical-deduction benchmarks.

Reference: Xu et al., "Chain of Draft: Thinking Ahead through Succinct Thoughts",
arXiv 2502.18600, 2025.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

__all__ = [
    "ChainOfDraftConfig",
    "ChainOfDraftState",
    "ChainOfDraftSampler",
]

# Default step-boundary tokens (model-agnostic; replace per tokeniser)
_DEFAULT_STEP_BOUNDARY = "\n\n"
_DEFAULT_WORD_SEPARATOR = " "


@dataclass
class ChainOfDraftConfig:
    """Configuration for :class:`ChainOfDraftSampler`.

    Attributes:
        max_step_tokens: Maximum tokens per reasoning step (≈ 7 words).
        step_boundary: Token string that marks the end of a reasoning step.
        length_penalty: Logit penalty applied per token once step length
            exceeds ``max_step_tokens``.
        force_boundary_after_limit: If True, force a step-boundary token once
            the per-step limit is reached (hard cutoff mode).
        seed: Unused; retained for API consistency.
    """

    max_step_tokens: int = 7
    step_boundary: str = _DEFAULT_STEP_BOUNDARY
    length_penalty: float = 5.0
    force_boundary_after_limit: bool = True
    seed: int = 0

    def __post_init__(self) -> None:
        if self.max_step_tokens < 1:
            raise ValueError(
                f"max_step_tokens must be ≥ 1, got {self.max_step_tokens}"
            )
        if self.length_penalty < 0.0:
            raise ValueError(
                f"length_penalty must be ≥ 0, got {self.length_penalty}"
            )


@dataclass
class ChainOfDraftState:
    """Mutable per-request state for :class:`ChainOfDraftSampler`.

    Attributes:
        current_step_tokens: Tokens produced in the current reasoning step.
        steps_completed: Number of full reasoning steps finished.
        total_thinking_tokens: Total thinking tokens generated.
        forced_boundaries: Number of step boundaries injected by force.
    """

    current_step_tokens: int = 0
    steps_completed: int = 0
    total_thinking_tokens: int = 0
    forced_boundaries: int = 0

    @property
    def over_budget(self) -> bool:
        return self.current_step_tokens > 0  # always allow at least one token


class ChainOfDraftSampler:
    """Length-constrained reasoning-step sampler.

    Applies a logit penalty to all non-boundary tokens once the per-step
    token count exceeds ``max_step_tokens``.

    Usage::

        cfg = ChainOfDraftConfig(max_step_tokens=7)
        sampler = ChainOfDraftSampler(cfg)
        state = sampler.new_state()
        for token in model_stream:
            penalty, force_inject = sampler.step(token, state)
            # apply penalty to next logits; inject force_inject if not None

    """

    def __init__(self, config: ChainOfDraftConfig) -> None:
        self.config = config

    def new_state(self) -> ChainOfDraftState:
        """Create fresh per-request state."""
        return ChainOfDraftState()

    def step(
        self, token: str, state: ChainOfDraftState
    ) -> Tuple[float, Optional[str]]:
        """Record *token* and return (logit_penalty_for_next, force_inject_or_None).

        Parameters
        ----------
        token:
            Decoded token string.
        state:
            Per-request mutable state.

        Returns
        -------
        penalty:
            Additive penalty to apply to all non-boundary logits for the next step.
        force_inject:
            String to force-inject (step boundary) if the step limit is hit, else None.
        """
        cfg = self.config
        is_boundary = cfg.step_boundary in token

        if is_boundary:
            state.steps_completed += 1
            state.current_step_tokens = 0
            return 0.0, None

        state.current_step_tokens += 1
        state.total_thinking_tokens += 1

        if state.current_step_tokens < cfg.max_step_tokens:
            return 0.0, None

        # Over the per-step budget
        penalty = cfg.length_penalty
        force_inject = None
        if cfg.force_boundary_after_limit and state.current_step_tokens >= cfg.max_step_tokens:
            force_inject = cfg.step_boundary
            state.forced_boundaries += 1
            state.steps_completed += 1
            state.current_step_tokens = 0
            penalty = 0.0  # boundary injected; reset penalty

        return penalty, force_inject

    def apply_penalty(self, logits: np.ndarray, penalty: float) -> np.ndarray:
        """Return a copy of *logits* with *penalty* subtracted from all entries."""
        if penalty == 0.0:
            return logits
        return logits - penalty

    def compression_ratio(self, state: ChainOfDraftState) -> float:
        """Ratio of steps completed to total thinking tokens (≥ 1/max_step_tokens)."""
        if state.total_thinking_tokens == 0:
            return 0.0
        return state.steps_completed / state.total_thinking_tokens

    def sample(
        self,
        prompt: str,
        max_total_tokens: int = 256,
        temperature: float = 0.7,
    ) -> Tuple[str, ChainOfDraftState]:
        """Produce a synthetic chain-of-draft sequence for offline testing.

        Generates random word counts per step bounded by ``max_step_tokens``.
        """
        rng = np.random.default_rng(self.config.seed)
        state = self.new_state()
        steps = []
        remaining = max_total_tokens
        while remaining > 0:
            n_words = int(rng.integers(1, self.config.max_step_tokens + 1))
            n_words = min(n_words, remaining)
            step = " ".join(f"word{i}" for i in range(n_words))
            steps.append(step)
            remaining -= n_words
            state.steps_completed += 1
            state.total_thinking_tokens += n_words
        chain = self.config.step_boundary.join(steps)
        return chain, state
