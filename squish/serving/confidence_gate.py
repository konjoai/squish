"""
ConfidenceGate — Confidence-Threshold Token Commit Gate.

Key idea (from Fast-dLLM / dLLM-Var research):
  Commit only draft tokens whose predicted probability exceeds a threshold;
  re-draft the remainder.  This is complementary to speculative decoding:
  instead of binary accept/reject at verification time, we pre-filter the
  draft sequence before sending it to the main model at all.

Result from Fast-dLLM: up to 2.4× speedup over autoregressive baseline
by avoiding costly full-model verification passes for high-confidence
draft prefixes.

Integration point:  call filter_draft() after the draft model produces
tokens, before invoking the (expensive) main model.  High-confidence
tokens can be committed immediately; the remainder is passed to the
main model for standard speculative verification.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ConfidenceGateConfig:
    """Configuration for the confidence-threshold commit gate."""
    threshold: float = 0.85          # min confidence to commit a token
    min_commit: int = 1              # always commit at least this many tokens
    max_commit: int = 8              # never commit more than this many tokens
    temperature_scaling: float = 1.0 # scale logits before confidence check

    def __post_init__(self) -> None:
        if not (0.0 < self.threshold <= 1.0):
            raise ValueError(f"threshold must be in (0, 1], got {self.threshold}")
        if self.min_commit < 0:
            raise ValueError(f"min_commit must be >= 0, got {self.min_commit}")
        if self.max_commit < self.min_commit:
            raise ValueError(
                f"max_commit ({self.max_commit}) must be >= min_commit ({self.min_commit})"
            )


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

@dataclass
class ConfidenceGateStats:
    """Runtime statistics for ConfidenceGate."""
    filter_calls: int = 0
    tokens_committed: int = 0
    tokens_redrafted: int = 0

    @property
    def commit_rate(self) -> float:
        total = self.tokens_committed + self.tokens_redrafted
        return self.tokens_committed / total if total > 0 else 0.0

    def __repr__(self) -> str:
        return (
            f"ConfidenceGateStats(calls={self.filter_calls}, "
            f"committed={self.tokens_committed}, "
            f"redrafted={self.tokens_redrafted}, "
            f"commit_rate={self.commit_rate:.1%})"
        )


# ---------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------

class ConfidenceGate:
    """Gate that commits high-confidence draft tokens, re-drafts low-confidence ones.

    Usage::

        gate = ConfidenceGate(ConfidenceGateConfig(threshold=0.9))
        commit_tokens, redraft_tokens = gate.filter_draft(draft_tokens, draft_logits)
        # commit_tokens can be appended to the context directly
        # redraft_tokens go to standard speculative verification
    """

    def __init__(self, config: ConfidenceGateConfig) -> None:
        self.config = config
        self.stats = ConfidenceGateStats()

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def confidence(self, logits: np.ndarray) -> float:
        """Max softmax probability (confidence) of the given logits.

        Args:
            logits: (vocab_size,) — un-normalised logits for one position.

        Returns:
            Scalar confidence in [0, 1].
        """
        if logits.ndim != 1:
            raise ValueError(f"logits must be 1-D, got shape {logits.shape}")
        scaled = logits / max(self.config.temperature_scaling, 1e-8)
        scaled = scaled - scaled.max()
        probs = np.exp(scaled)
        probs /= probs.sum()
        return float(probs.max())

    def should_commit(self, logits: np.ndarray) -> bool:
        """Return True if this draft token's confidence is above threshold."""
        return self.confidence(logits) >= self.config.threshold

    def commit_span(self, draft_logits: np.ndarray) -> int:
        """Return the number of consecutive high-confidence tokens from the start.

        Args:
            draft_logits: (draft_len, vocab_size) — one row per draft position.

        Returns:
            Integer in [min_commit, max_commit] — number of tokens to commit.
        """
        cfg = self.config
        draft_len = draft_logits.shape[0]
        n = 0
        for i in range(min(draft_len, cfg.max_commit)):
            if self.should_commit(draft_logits[i]):
                n += 1
            else:
                break
        return max(cfg.min_commit, min(n, cfg.max_commit))

    def filter_draft(
        self,
        draft_tokens: List[int],
        draft_logits: np.ndarray,
    ) -> Tuple[List[int], List[int]]:
        """Split draft tokens into (commit, re-draft) based on confidence.

        Committed tokens come from the high-confidence prefix.
        Remaining tokens are returned for standard speculative verification.

        Args:
            draft_tokens: list of draft token ids.
            draft_logits: (len(draft_tokens), vocab_size) — per-token logits.

        Returns:
            (commit_tokens, redraft_tokens) — both lists of token ids.
        """
        if len(draft_tokens) != draft_logits.shape[0]:
            raise ValueError(
                f"draft_tokens length {len(draft_tokens)} != "
                f"draft_logits rows {draft_logits.shape[0]}"
            )

        n_commit = self.commit_span(draft_logits)
        commit_tokens = draft_tokens[:n_commit]
        redraft_tokens = draft_tokens[n_commit:]

        self.stats.filter_calls += 1
        self.stats.tokens_committed += len(commit_tokens)
        self.stats.tokens_redrafted += len(redraft_tokens)

        return commit_tokens, redraft_tokens

    # ------------------------------------------------------------------
    # Batch variant
    # ------------------------------------------------------------------

    def filter_batch(
        self,
        batch_tokens: List[List[int]],
        batch_logits: List[np.ndarray],
    ) -> List[Tuple[List[int], List[int]]]:
        """Apply filter_draft to each item in a batch.

        Args:
            batch_tokens: list of draft token lists.
            batch_logits: list of (draft_len, vocab) logit arrays.

        Returns:
            List of (commit_tokens, redraft_tokens) pairs.
        """
        return [
            self.filter_draft(toks, logits)
            for toks, logits in zip(batch_tokens, batch_logits)
        ]

    def __repr__(self) -> str:
        return (
            f"ConfidenceGate(threshold={self.config.threshold:.2f}, "
            f"max_commit={self.config.max_commit}, {self.stats})"
        )
