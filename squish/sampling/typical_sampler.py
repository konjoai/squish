"""squish/sampling/typical_sampler.py

TypicalSampler — Locally Typical Sampling.

Reference
---------
Meister et al. "Locally Typical Sampling." Transactions of the Association
for Computational Linguistics (TACL) 11 (2023) / ACL 2023.

Algorithm
---------
Let ``H(p)`` be the conditional entropy of the next-token distribution.
A token is *typical* if its surprisal ``-log p(w)`` lies within ``tau``
of ``H(p)``.  Specifically, a candidate set ``T`` is constructed by
sorting tokens by ``|I(w) - H(p)|`` (ascending) and keeping the smallest
prefix whose cumulative probability is ≥ ``tau``.  Only tokens in ``T``
are sampled.

Compared to nucleus (top-p) sampling this filter naturally excludes very
high-probability tokens that dominate factual recall tasks, leading to
more natural text.

This module provides:

1. ``TypicalSampler.sample(logits)`` → ``TypicalResult``.
2. ``TypicalSampler.sample_batch(logits)`` → ``(batch,)`` token ids.
3. ``TypicalSampler.filter_logits(logits)`` → log-probs with ``-inf`` outside ``T``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

__all__ = [
    "TypicalConfig",
    "TypicalResult",
    "TypicalSampler",
]


@dataclass
class TypicalConfig:
    """Configuration for :class:`TypicalSampler`.

    Attributes:
        tau: Cumulative probability mass to include in the typical set.
            Values close to 1.0 keep almost all tokens; lower values are
            more restrictive.
        temperature: Softmax temperature applied before typical filtering.
        seed: RNG seed for reproducibility.
    """

    tau: float = 0.9
    temperature: float = 1.0
    seed: int = 0

    def __post_init__(self) -> None:
        if not (0.0 < self.tau <= 1.0):
            raise ValueError(f"tau must be in (0, 1]; got {self.tau}")
        if self.temperature <= 0.0:
            raise ValueError(f"temperature must be > 0; got {self.temperature}")


@dataclass
class TypicalResult:
    """Result of a single :meth:`TypicalSampler.sample` call.

    Attributes:
        token_id: Sampled token index.
        probability: Probability of the sampled token under the filtered distribution.
        n_candidates: Number of tokens in the typical set.
        entropy: Conditional entropy ``H(p)`` of the original distribution.
    """

    token_id: int
    probability: float
    n_candidates: int
    entropy: float


class TypicalSampler:
    """Locally typical sampler.

    Example::

        cfg = TypicalConfig(tau=0.9, temperature=1.0)
        sampler = TypicalSampler(cfg)

        logits = np.random.randn(32000).astype(np.float32)
        result = sampler.sample(logits)
        print(result.token_id, result.n_candidates)
    """

    def __init__(self, config: Optional[TypicalConfig] = None) -> None:
        self._cfg = config or TypicalConfig()
        self._rng = np.random.default_rng(self._cfg.seed)

    @property
    def config(self) -> TypicalConfig:
        return self._cfg

    def _get_log_probs(self, logits: np.ndarray) -> np.ndarray:
        """Apply temperature and return log-softmax probabilities."""
        logits = np.asarray(logits, dtype=np.float64) / self._cfg.temperature
        logits -= logits.max()
        log_probs = logits - np.log(np.sum(np.exp(logits)))
        return log_probs.astype(np.float32)

    def filter_logits(self, logits: np.ndarray) -> np.ndarray:
        """Return log-probs with ``-inf`` for tokens outside the typical set.

        Args:
            logits: ``(vocab_size,)`` raw logits.

        Returns:
            ``(vocab_size,)`` float32 where non-typical tokens are ``-inf``.
        """
        log_probs = self._get_log_probs(logits)
        probs = np.exp(log_probs.astype(np.float64))
        entropy = -float(np.sum(probs * log_probs.astype(np.float64)))  # H(p)

        # Surprisal deviation: |(-log p) - H(p)|
        surprisal = -log_probs.astype(np.float64)
        deviation = np.abs(surprisal - entropy)

        # Sort by deviation ascending, keep smallest prefix summing to >= tau
        sort_order = np.argsort(deviation)
        sorted_probs = probs[sort_order]
        cumulative = np.cumsum(sorted_probs)
        tau = self._cfg.tau
        cutoff = int(np.searchsorted(cumulative, tau, side="right")) + 1
        typical_indices = sort_order[:cutoff]

        filtered = np.full_like(log_probs, -np.inf)
        filtered[typical_indices] = log_probs[typical_indices]
        return filtered

    def sample(self, logits: np.ndarray) -> TypicalResult:
        """Sample one token from the locally typical set.

        Args:
            logits: ``(vocab_size,)`` raw logits.

        Returns:
            :class:`TypicalResult` with sampled token and metadata.
        """
        log_probs_base = self._get_log_probs(logits)
        probs_all = np.exp(log_probs_base.astype(np.float64))
        entropy = -float(np.sum(probs_all * log_probs_base.astype(np.float64)))

        filtered = self.filter_logits(logits)
        valid_mask = np.isfinite(filtered)
        n_candidates = int(valid_mask.sum())

        valid_log = filtered[valid_mask]
        valid_log -= valid_log.max()
        valid_probs = np.exp(valid_log.astype(np.float64))
        valid_probs /= valid_probs.sum()

        valid_indices = np.where(valid_mask)[0]
        chosen_pos = self._rng.choice(len(valid_indices), p=valid_probs)
        token_id = int(valid_indices[chosen_pos])

        return TypicalResult(
            token_id=token_id,
            probability=float(valid_probs[chosen_pos]),
            n_candidates=n_candidates,
            entropy=entropy,
        )

    def sample_batch(self, logits: np.ndarray) -> np.ndarray:
        """Sample one token per row in a batch of logits.

        Args:
            logits: ``(batch, vocab_size)`` raw logits.

        Returns:
            ``(batch,)`` int32 token ids.
        """
        logits = np.asarray(logits, dtype=np.float32)
        return np.array(
            [self.sample(row).token_id for row in logits], dtype=np.int32
        )
