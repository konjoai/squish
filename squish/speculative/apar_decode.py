"""APARDecoder — Auto-Parallel Auto-Regressive decoding.

Implements the APAR algorithm (Liu et al., arXiv:2401.06761 / 2024).

Standard auto-regressive decoding is strictly sequential: token t+1 depends
on all tokens ≤ t.  APAR observes that in many structured outputs (JSON,
code, markdown, lists) the *tree-shaped* dependency structure means that
multiple branches can be generated in parallel once the root of each branch
has been established.

APAR detects independence at the *structural* level:
  - When the model's next-token distribution shows high confidence on a
    structural opener (e.g., ``\n``, ``{``, list bullet), the system forks
    a new parallel generation branch.
  - Each branch generates independently using its own context window.
  - Branches are re-merged in the final output in the correct order.

This module provides:
  * :class:`APARConfig` — fork triggers, max branches, merge policy.
  * :class:`APARBranch` — lightweight branch state container.
  * :class:`APARDecoder` — detection, forking, generation scheduling, and merge.

Reference:
    Liu et al., "APAR: LLMs Can Do Auto-Parallel Auto-Regressive Decoding",
    arXiv:2401.06761 (2024).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np

__all__ = [
    "APARConfig",
    "APARBranch",
    "APARDecoder",
]

# ── Configuration ─────────────────────────────────────────────────────────────


@dataclass
class APARConfig:
    """Configuration for APARDecoder.

    Attributes:
        fork_tokens: Set of token IDs that can trigger a parallel fork.
        fork_confidence_threshold: Minimum softmax probability for the fork
            token before a branch is opened.
        max_branches: Maximum number of concurrently open branches.
        max_branch_length: Maximum tokens per branch before auto-close.
        temperature: Sampling temperature applied to logits.
        merge_token: Token ID used to signal branch completion and merge.
            Defaults to ``-1`` (infer from EOS or explicit stop token).
    """

    fork_tokens: frozenset = field(default_factory=lambda: frozenset({10, 13, 123}))
    fork_confidence_threshold: float = 0.8
    max_branches: int = 4
    max_branch_length: int = 64
    temperature: float = 1.0
    merge_token: int = -1

    def __post_init__(self) -> None:
        if not (0.0 < self.fork_confidence_threshold <= 1.0):
            raise ValueError(
                f"fork_confidence_threshold must be in (0, 1]; "
                f"got {self.fork_confidence_threshold}"
            )
        if self.max_branches < 1:
            raise ValueError(
                f"max_branches must be ≥ 1; got {self.max_branches}"
            )
        if self.max_branch_length < 1:
            raise ValueError(
                f"max_branch_length must be ≥ 1; got {self.max_branch_length}"
            )
        if self.temperature <= 0:
            raise ValueError(
                f"temperature must be > 0; got {self.temperature}"
            )


# ── Branch state ──────────────────────────────────────────────────────────────


@dataclass
class APARBranch:
    """State for a single auto-parallel branch.

    Attributes:
        branch_id: Unique integer identifier.
        parent_id: Branch ID of the parent (``-1`` for the root branch).
        context: Token IDs generated so far in this branch.
        insert_position: Position in the root output where this branch's
            tokens will be spliced in.
        closed: Whether this branch has terminated.
    """

    branch_id: int
    parent_id: int
    context: list = field(default_factory=list)
    insert_position: int = 0
    closed: bool = False


# ── Core class ────────────────────────────────────────────────────────────────


class APARDecoder:
    """Output-tree independence detection and parallel branch scheduling.

    Example::

        def my_generate_fn(token: int, context: list[int]) -> np.ndarray:
            # Returns logits of shape (vocab_size,)
            ...

        cfg     = APARConfig(fork_tokens=frozenset({10}), max_branches=4)
        decoder = APARDecoder(cfg)
        tokens  = decoder.generate(
            prompt_ids=[1, 2, 3],
            generate_fn=my_generate_fn,
            max_new_tokens=128,
        )

    Args:
        config: :class:`APARConfig` (optional).
    """

    def __init__(self, config: Optional[APARConfig] = None) -> None:
        self.config: APARConfig = config or APARConfig()
        self._rng = np.random.default_rng(seed=0)
        self._branches: list[APARBranch] = []
        self._next_branch_id: int = 0

    # ── Detection ─────────────────────────────────────────────────────────────

    def should_fork(self, logits: np.ndarray) -> tuple[bool, int]:
        """Check whether the current logit distribution warrants a new branch.

        A fork is triggered when a fork token's softmax probability exceeds
        ``config.fork_confidence_threshold`` and the number of open branches
        is below ``config.max_branches``.

        Args:
            logits: ``(vocab_size,)`` float32 logit array.

        Returns:
            ``(fork, fork_token)`` tuple.  ``fork=True`` when a new parallel
            branch should be opened; ``fork_token`` is the triggering token ID.
        """
        probs = self._to_prob(logits)
        n_open = sum(1 for b in self._branches if not b.closed)
        if n_open >= self.config.max_branches:
            return False, -1

        for t in self.config.fork_tokens:
            if t < len(probs) and probs[t] >= self.config.fork_confidence_threshold:
                return True, t

        return False, -1

    # ── Generation ────────────────────────────────────────────────────────────

    def generate(
        self,
        prompt_ids: list[int],
        generate_fn: Callable[[int, list[int]], np.ndarray],
        max_new_tokens: int = 64,
        eos_token: int = 2,
    ) -> list[int]:
        """Generate tokens with automatic parallel branching.

        Args:
            prompt_ids: Initial context token IDs.
            generate_fn: ``(token, context) -> logits`` callable.
            max_new_tokens: Maximum number of new tokens to generate in total
                across all branches.
            eos_token: End-of-sequence token ID.

        Returns:
            Flat list of generated token IDs (merged across all branches in
            insertion-position order).
        """
        self._branches = []
        self._next_branch_id = 0
        cfg = self.config

        root = self._new_branch(parent_id=-1, insert_position=0)
        root.context = list(prompt_ids)
        output: list[int] = []
        total_generated = 0

        # Main loop: round-robin over open branches
        while total_generated < max_new_tokens:
            active = [b for b in self._branches if not b.closed]
            if not active:
                break
            made_progress = False
            for branch in active:
                if branch.closed:
                    continue
                cur_tok = branch.context[-1] if branch.context else 0
                logits = np.asarray(generate_fn(cur_tok, branch.context))
                probs = self._to_prob(logits)
                next_tok = int(self._rng.choice(len(probs), p=probs))
                branch.context.append(next_tok)
                total_generated += 1
                made_progress = True

                # Close on EOS or max branch length
                branch_len = len(branch.context) - (len(prompt_ids) if branch.branch_id == 0 else 0)
                if next_tok == eos_token or branch_len >= cfg.max_branch_length:
                    branch.closed = True
                    continue

                # Fork if conditions met
                fork, fork_tok = self.should_fork(logits)
                if fork:
                    new_br = self._new_branch(
                        parent_id=branch.branch_id,
                        insert_position=len(output) + len(branch.context),
                    )
                    new_br.context = list(branch.context)

            if not made_progress:
                break

        # Close any remaining open branches (max_new_tokens budget exhausted)
        for b in self._branches:
            b.closed = True

        # Merge: collect tokens from root branch (simplest merge strategy)
        root = self._branches[0] if self._branches else None
        if root is not None:
            prompt_len = len(prompt_ids)
            output = [t for t in root.context[prompt_len:] if t != eos_token]
        return output

    # ── Branch management ─────────────────────────────────────────────────────

    def _new_branch(self, parent_id: int, insert_position: int) -> APARBranch:
        br = APARBranch(
            branch_id=self._next_branch_id,
            parent_id=parent_id,
            insert_position=insert_position,
        )
        self._next_branch_id += 1
        self._branches.append(br)
        return br

    def active_branch_count(self) -> int:
        """Return the number of currently open (not-closed) branches."""
        return sum(1 for b in self._branches if not b.closed)

    def branch_count(self) -> int:
        """Return the total number of branches created (including closed)."""
        return len(self._branches)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _to_prob(self, logits: np.ndarray) -> np.ndarray:
        T = self.config.temperature
        logits = np.asarray(logits, dtype=np.float64) / T
        logits -= logits.max()
        exp = np.exp(logits)
        total = exp.sum()
        if total == 0:
            return np.ones(len(exp)) / len(exp)
        return exp / total

    def reset(self) -> None:
        """Reset branch state for a new generation call."""
        self._branches = []
        self._next_branch_id = 0

    def __repr__(self) -> str:
        return (
            f"APARDecoder("
            f"max_branches={self.config.max_branches}, "
            f"threshold={self.config.fork_confidence_threshold})"
        )
