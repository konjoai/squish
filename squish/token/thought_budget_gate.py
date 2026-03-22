"""ThoughtBudgetGate: per-token gating to enforce thinking-token budgets.

Inspired by s1 (arXiv 2501.12599) and OpenAI o1-style thinking-budget control.
This module operates at the token-stream level: each token is observed, the
gate tracks which segment (thinking vs answer) is active, and injects a
commit signal when the thinking budget is exhausted or a natural boundary
token is seen.

Coordinates with :class:`~squish.serving.budget_forcing.BudgetForcingDecoder`
but is self-contained and usable independently.

Reference: Muennighoff et al., "s1: Simple Test-Time Scaling", arXiv
2501.12599, 2025.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

__all__ = [
    "ThoughtBudgetConfig",
    "ThoughtBudgetState",
    "ThoughtBudgetGate",
]

_SEGMENT_THINKING = "thinking"
_SEGMENT_ANSWER = "answer"


@dataclass
class ThoughtBudgetConfig:
    """Configuration for :class:`ThoughtBudgetGate`.

    Attributes:
        max_thinking_tokens: Hard cap on thinking-segment tokens.
        boundary_tokens: Tokens that trigger a segment transition when seen.
        commit_trigger: String injected just before the answer segment begins
            when the budget is exhausted.
        soft_budget_fraction: Fraction of budget at which a *soft* warning is
            emitted (does not force commit; informational only).
        seed: RNG seed (reserved).
    """

    max_thinking_tokens: int = 1024
    boundary_tokens: List[str] = field(default_factory=lambda: ["</think>"])
    commit_trigger: str = "Final Answer:"
    soft_budget_fraction: float = 0.8
    seed: int = 0

    def __post_init__(self) -> None:
        if self.max_thinking_tokens < 1:
            raise ValueError(
                f"max_thinking_tokens must be ≥ 1, got {self.max_thinking_tokens}"
            )
        if not (0.0 < self.soft_budget_fraction < 1.0):
            raise ValueError(
                f"soft_budget_fraction must be in (0, 1), got {self.soft_budget_fraction}"
            )


@dataclass
class ThoughtBudgetState:
    """Per-generation mutable state for :class:`ThoughtBudgetGate`.

    Attributes:
        segment: Current segment; one of ``"thinking"`` or ``"answer"``.
        tokens_in_segment: Tokens seen in the current segment.
        total_tokens: Total tokens seen since reset.
        n_boundaries: Number of explicit boundary tokens observed.
        forced_commits: How many times the gate force-injected a commit.
    """

    segment: str = _SEGMENT_THINKING
    tokens_in_segment: int = 0
    total_tokens: int = 0
    n_boundaries: int = 0
    forced_commits: int = 0

    @property
    def in_thinking(self) -> bool:
        return self.segment == _SEGMENT_THINKING

    @property
    def in_answer(self) -> bool:
        return self.segment == _SEGMENT_ANSWER


class ThoughtBudgetGate:
    """At each decoding step decide whether to cross a segment boundary.

    Returns ``(at_boundary, inject_commit)`` tuple:

    * ``at_boundary`` — True when the current token is a known boundary token.
    * ``inject_commit`` — True when the gate is forcing end-of-thinking (the
      caller should prepend ``config.commit_trigger`` before the next token).

    Usage::

        cfg = ThoughtBudgetConfig(max_thinking_tokens=512)
        gate = ThoughtBudgetGate(cfg)
        state = gate.new_state()
        for token_str in token_stream:
            at_boundary, inject_commit = gate.step(token_str, state)
            if inject_commit:
                # flush commit_trigger into the stream
                ...
    """

    def __init__(self, config: ThoughtBudgetConfig) -> None:
        self.config = config
        self._boundary_set = set(config.boundary_tokens)

    # ------------------------------------------------------------------
    # Primary API
    # ------------------------------------------------------------------

    def new_state(self) -> ThoughtBudgetState:
        """Create a fresh gate state."""
        return ThoughtBudgetState()

    def step(self, token: str, state: ThoughtBudgetState) -> Tuple[bool, bool]:
        """Process one decoded token.

        Parameters
        ----------
        token:
            The decoded token string.
        state:
            Mutable state (modified in place).

        Returns
        -------
        (at_boundary, inject_commit):
            ``at_boundary`` is True if *token* is in ``config.boundary_tokens``.
            ``inject_commit`` is True if the thinking budget is just exhausted
            and the gate is requesting a segment commit.
        """
        state.total_tokens += 1
        at_boundary = token in self._boundary_set
        inject_commit = False

        if state.in_thinking:
            state.tokens_in_segment += 1
            if at_boundary:
                self._transition_to_answer(state)
            elif state.tokens_in_segment >= self.config.max_thinking_tokens:
                # Hard budget exhausted → force commit
                inject_commit = True
                state.forced_commits += 1
                self._transition_to_answer(state)
        else:
            # In answer segment; just count
            state.tokens_in_segment += 1
            if at_boundary:
                state.n_boundaries += 1

        return at_boundary, inject_commit

    def budget_fraction(self, state: ThoughtBudgetState) -> float:
        """Fraction of thinking budget consumed (0–1, clamped)."""
        if state.in_answer:
            return 1.0
        return min(
            state.tokens_in_segment / self.config.max_thinking_tokens, 1.0
        )

    def near_soft_budget(self, state: ThoughtBudgetState) -> bool:
        """True when thinking budget has passed the soft-warning threshold."""
        return self.budget_fraction(state) >= self.config.soft_budget_fraction

    def segment_of(self, state: ThoughtBudgetState) -> str:
        """Return the current segment name."""
        return state.segment

    def reset(self, state: ThoughtBudgetState) -> None:
        """Reset *state* in place for reuse."""
        state.segment = _SEGMENT_THINKING
        state.tokens_in_segment = 0
        state.total_tokens = 0
        state.n_boundaries = 0
        state.forced_commits = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _transition_to_answer(self, state: ThoughtBudgetState) -> None:
        state.segment = _SEGMENT_ANSWER
        state.n_boundaries += 1
        state.tokens_in_segment = 0
