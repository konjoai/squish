"""squish/speculative/spec_rejection.py

SpecRejection — Parallel Draft Candidate Pool with Early Rejection.

Reference
---------
Yang et al. "Speculative Rejection: Accelerating Speculative Decoding
with Diverse Draft Candidates."
NeurIPS 2024 (arXiv:2410.20290).

Algorithm
---------
Standard speculative decoding maintains a single draft candidate per
position.  SpecRejection instead maintains a *pool* of K draft candidates
per token position:

1. Generate K draft candidates (e.g., top-K logits from draft model).
2. Sort by draft probability and reject the lowest-P candidates early
   before sending to the verifier — saving verifier forward-pass budget.
3. Run the target model only over the surviving candidates.
4. Accept/reject each candidate via standard rejection sampling.

Key properties
--------------
* NumPy-only.
* ``pool_size`` — initial number of candidates per position.
* ``early_reject_fraction`` — fraction of candidates rejected before
  calling the target model (default 0.5 → only top-50% go to verifier).
* ``max_draft_len`` — maximum draft length (positions).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

__all__ = [
    "SpecRejectionConfig",
    "SpecRejectionResult",
    "SpecRejection",
]


@dataclass
class SpecRejectionConfig:
    """Configuration for :class:`SpecRejection`.

    Attributes:
        pool_size: Number of draft candidates per token position.
        early_reject_fraction: Fraction eliminated before target model call.
        max_draft_len: Maximum speculative draft length.
        vocab_size: Vocabulary size.
        temperature: Sampling temperature for draft distribution.
    """

    pool_size: int = 8
    early_reject_fraction: float = 0.5
    max_draft_len: int = 4
    vocab_size: int = 32000
    temperature: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.early_reject_fraction < 1.0:
            raise ValueError("early_reject_fraction must be in [0, 1)")


@dataclass
class SpecRejectionResult:
    """Result of one speculative rejection step.

    Attributes:
        accepted_tokens: Token IDs accepted, shape ``(n_accepted,)``.
        n_draft_candidates: Total draft candidates generated.
        n_early_rejected: Candidates eliminated before target model.
        n_target_rejected: Candidates eliminated by rejection sampling.
    """

    accepted_tokens: np.ndarray
    n_draft_candidates: int
    n_early_rejected: int
    n_target_rejected: int

    @property
    def n_accepted(self) -> int:
        """Number of accepted tokens."""
        return int(len(self.accepted_tokens))

    @property
    def acceptance_rate(self) -> float:
        total = self.n_draft_candidates - self.n_early_rejected
        if total == 0:
            return 0.0
        return len(self.accepted_tokens) / total


class SpecRejection:
    """Parallel draft candidate pool with early rejection.

    Parameters
    ----------
    config:
        SpecRejection configuration.
    seed:
        RNG seed for stochastic rejection.
    """

    def __init__(self, config: Optional[SpecRejectionConfig] = None, seed: int = 0) -> None:
        self._cfg = config or SpecRejectionConfig()
        self._rng = np.random.default_rng(seed)
        self._total_generated: int = 0
        self._total_accepted: int = 0

    @property
    def config(self) -> SpecRejectionConfig:
        return self._cfg

    @property
    def acceptance_rate_running(self) -> float:
        if self._total_generated == 0:
            return 0.0
        return self._total_accepted / self._total_generated

    def generate_candidates(self, draft_logits: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Sample one candidate token per sequence position.

        Parameters
        ----------
        draft_logits:
            Logits from draft model.
            Shape ``(vocab_size,)`` for single position or
            ``(draft_len, vocab_size)`` for multiple positions.

        Returns
        -------
        Tuple of (token_ids, draft_log_probs) each shape ``(draft_len,)``.
        """
        logits = np.asarray(draft_logits, dtype=np.float64)
        if logits.ndim == 1:
            logits = logits[None, :]  # (1, vocab_size)
        draft_len, vocab_size = logits.shape
        if self._cfg.temperature != 1.0 and self._cfg.temperature > 0:
            logits = logits / self._cfg.temperature
        logits -= logits.max(axis=-1, keepdims=True)
        probs = np.exp(logits)
        probs /= probs.sum(axis=-1, keepdims=True)
        tokens = np.array([
            self._rng.choice(vocab_size, p=probs[i])
            for i in range(draft_len)
        ], dtype=np.int64)
        log_probs = np.log(probs[np.arange(draft_len), tokens].clip(min=1e-10))
        return tokens, log_probs

    def early_reject(
        self, tokens: np.ndarray, draft_log_probs: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Reject the lowest-P fraction of candidates before target model.

        Returns surviving (tokens, log_probs).
        """
        n_reject = int(len(tokens) * self._cfg.early_reject_fraction)
        n_keep = len(tokens) - n_reject
        order = np.argsort(draft_log_probs)[::-1]  # descending p
        keep = order[:n_keep]
        return tokens[keep], draft_log_probs[keep]

    def rejection_sample(
        self,
        tokens: np.ndarray,
        draft_log_probs: np.ndarray,
        target_log_probs: np.ndarray,
    ) -> np.ndarray:
        """Accept/reject each surviving candidate via rejection sampling.

        Parameters
        ----------
        tokens: Surviving token IDs, shape ``(n_surviving,)``.
        draft_log_probs: Log-probs from draft model, shape ``(n_surviving,)``.
        target_log_probs: Log-probs from target model.
            Shape ``(n_surviving,)`` per-token, or ``(n_surviving, vocab_size)``
            per-position (in which case the token's probability is extracted).

        Returns
        -------
        np.ndarray
            Accepted token IDs (may be empty).
        """
        tl = np.asarray(target_log_probs, dtype=np.float64)
        if tl.ndim == 2:
            # Extract per-token log-prob from target 2D matrix
            tl = tl[np.arange(len(tokens)), tokens]
        log_ratio = tl - draft_log_probs
        u = self._rng.random(len(tokens))
        accepted_mask = (log_ratio >= 0) | (np.log(u) < log_ratio)
        return tokens[accepted_mask]

    def step(
        self,
        draft_logits: np.ndarray,
        target_log_probs: np.ndarray,
    ) -> SpecRejectionResult:
        """Run a full speculative rejection step.

        Parameters
        ----------
        draft_logits:
            Draft model logits, shape ``(vocab_size,)``.
        target_log_probs:
            Target model log-probs for the pool_size tokens, shape
            ``(pool_size,)``.  In practice obtained by running the target
            model in batch.

        Returns
        -------
        SpecRejectionResult
        """
        tokens, draft_lp = self.generate_candidates(draft_logits)
        n_original = len(tokens)
        surviving_tokens, surviving_lp = self.early_reject(tokens, draft_lp)
        n_early_rejected = n_original - len(surviving_tokens)

        # Slice target log-probs to match surviving candidates
        tlp = np.asarray(target_log_probs, dtype=np.float64)
        if tlp.ndim == 2:
            tlp_for_survivors = tlp[:len(surviving_tokens)]
        else:
            tlp_for_survivors = tlp[:len(surviving_tokens)]
        accepted = self.rejection_sample(surviving_tokens, surviving_lp, tlp_for_survivors)
        n_target_rejected = len(surviving_tokens) - len(accepted)

        self._total_generated += n_original
        self._total_accepted += len(accepted)

        return SpecRejectionResult(
            accepted_tokens=accepted,
            n_draft_candidates=n_original,
            n_early_rejected=n_early_rejected,
            n_target_rejected=n_target_rejected,
        )

    def reset_stats(self) -> None:
        self._total_generated = 0
        self._total_accepted = 0
