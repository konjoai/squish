"""
squish/speculative/ouroboros_draft.py

OuroborosDrafter: Lookahead Speculative Drafting with Verified-Token Feedback.

Reference
---------
Zhao et al. "Ouroboros: Speculative Decoding with Large Model Enhanced
Drafting." NeurIPS 2024.

Algorithm
---------
Ouroboros is a speculative decoding draft strategy that enriches the draft
model's context with tokens that have *already been verified* by the target
model in previous rounds.  The "ouroboros" metaphor: the output feeds back
to the drafter.

At each drafting step:
  1. The verified history (``verified_context``) is appended to the raw prompt
     context to form the drafter's input.
  2. The drafter generates ``depth`` candidate tokens via greedy or top-p
     sampling from a token vocabulary.
  3. The target model verifies and accepts a prefix of the draft; accepted
     tokens are appended to ``verified_context`` for the next round.

This module implements the drafter side.  The verifier is the external target
model (not included).  Without a real model, the drafter simulates token
generation via an n-gram table or uniform sampling.

Key properties
--------------
* ``depth`` — number of draft tokens per step (default 5).
* ``feedback_window`` — number of recent verified tokens fed back to the
  drafter (default 16).  Larger window improves acceptance at the cost of
  longer drafter input.
* ``use_ngram`` — use n-gram fallback statistics (default True).
* NumPy-only; no external dependencies.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class OuroborosConfig:
    """Configuration for OuroborosDrafter."""

    depth: int = 5
    """Number of draft tokens generated per step."""

    feedback_window: int = 16
    """Recent verified tokens fed back to the drafter for enriched context."""

    use_ngram: bool = True
    """Seed draft distribution from n-gram statistics when available."""

    ngram_order: int = 2
    """N-gram order for the fallback drafter (default bigram)."""

    vocab_size: int = 32000
    """Token vocabulary size (for simulation purposes)."""

    temperature: float = 0.0
    """Sampling temperature; 0.0 = greedy."""

    def __post_init__(self) -> None:
        if self.depth < 1:
            raise ValueError("depth must be >= 1")
        if self.feedback_window < 0:
            raise ValueError("feedback_window must be >= 0")
        if self.ngram_order < 1:
            raise ValueError("ngram_order must be >= 1")
        if self.vocab_size < 1:
            raise ValueError("vocab_size must be >= 1")
        if self.temperature < 0.0:
            raise ValueError("temperature must be >= 0")


@dataclass
class OuroborosStats:
    """Runtime counters for OuroborosDrafter."""

    draft_steps: int = 0
    total_drafted_tokens: int = 0
    total_accepted_tokens: int = 0
    feedback_uses: int = 0

    @property
    def mean_acceptance_rate(self) -> float:
        if self.total_drafted_tokens == 0:
            return 0.0
        return self.total_accepted_tokens / self.total_drafted_tokens


class OuroborosDrafter:
    """Lookahead speculative drafter with verified-token feedback.

    Usage
    -----
    ::

        drafter = OuroborosDrafter()
        drafts = drafter.draft(input_ids)           # list of candidate token ids
        drafter.accept_feedback(verified_tokens)    # update context
    """

    def __init__(self, config: Optional[OuroborosConfig] = None) -> None:
        self.config = config or OuroborosConfig()
        self.stats = OuroborosStats()
        self._verified_context: List[int] = []
        # N-gram table: context_tuple → {next_token: count}
        self._ngram_counts: Dict[Tuple[int, ...], Dict[int, int]] = {}
        self._rng = np.random.default_rng(seed=42)

    # ------------------------------------------------------------------
    # N-gram management
    # ------------------------------------------------------------------

    def _update_ngram(self, token_ids: Sequence[int]) -> None:
        """Update n-gram statistics from a verified token sequence."""
        order = self.config.ngram_order
        for i in range(len(token_ids) - order):
            ctx = tuple(token_ids[i : i + order])
            next_tok = token_ids[i + order]
            if ctx not in self._ngram_counts:
                self._ngram_counts[ctx] = {}
            self._ngram_counts[ctx][next_tok] = (
                self._ngram_counts[ctx].get(next_tok, 0) + 1
            )

    def _ngram_next(self, context: Tuple[int, ...]) -> Optional[int]:
        """Return the most likely next token from n-gram table, or None."""
        if context not in self._ngram_counts:
            return None
        counts = self._ngram_counts[context]
        return max(counts, key=lambda t: counts[t])

    # ------------------------------------------------------------------
    # Drafting
    # ------------------------------------------------------------------

    def draft(self, input_ids: Sequence[int]) -> List[int]:
        """Generate ``config.depth`` draft token IDs.

        Parameters
        ----------
        input_ids:
            The current sequence of token IDs seen so far.

        Returns
        -------
        draft_tokens:
            List of ``depth`` candidate token IDs.
        """
        self.stats.draft_steps += 1
        cfg = self.config

        # Build enriched context: raw input + recent verified tokens
        feedback = self._verified_context[-cfg.feedback_window :]
        if feedback:
            self.stats.feedback_uses += 1
        enriched: List[int] = list(input_ids) + feedback

        drafts: List[int] = []
        ctx = list(enriched)

        for _ in range(cfg.depth):
            order = cfg.ngram_order
            ngram_ctx = tuple(ctx[-order:]) if len(ctx) >= order else ()

            if cfg.use_ngram:
                tok = self._ngram_next(ngram_ctx)
                if tok is not None:
                    drafts.append(tok)
                    ctx.append(tok)
                    continue

            # Fallback: uniform / temperature sampling
            if cfg.temperature == 0.0:
                # deterministic pseudo-random based on context hash
                seed_val = hash(tuple(ctx[-4:])) % (2**31)
                tok = int(seed_val % cfg.vocab_size)
            else:
                logits = self._rng.standard_normal(cfg.vocab_size)
                logits /= cfg.temperature
                logits -= logits.max()
                probs = np.exp(logits)
                probs /= probs.sum()
                tok = int(self._rng.choice(cfg.vocab_size, p=probs))

            drafts.append(tok)
            ctx.append(tok)

        self.stats.total_drafted_tokens += len(drafts)
        return drafts

    def accept_feedback(self, verified_tokens: Sequence[int]) -> None:
        """Update the verified context after a verification round.

        Parameters
        ----------
        verified_tokens:
            Tokens that were accepted by the target model this round.
        """
        if verified_tokens:
            self._verified_context.extend(verified_tokens)
            self.stats.total_accepted_tokens += len(verified_tokens)
            if self.config.use_ngram:
                self._update_ngram(verified_tokens)

    def reset(self) -> None:
        """Clear state between requests."""
        self._verified_context.clear()
        self.stats = OuroborosStats()
