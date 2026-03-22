"""squish/serving/kgw_watermark.py

KGWWatermark — Green/Red List LLM Output Watermarking.

Reference
---------
Kirchenbauer et al. "A Watermark for Large Language Models."
ICML 2023 (arXiv 2301.10226).

Algorithm
---------
At each generation step, the vocabulary is partitioned into a *green list*
(size ``gamma * vocab_size``) and a *red list*, where the partition is
seeded by a hash of the preceding context token.  A bias ``delta`` is added
to the logits of every green-list token before sampling.  Detection runs a
one-sided z-test on the fraction of green tokens in any text: under the
null hypothesis a random token is green with probability ``gamma``.

This module provides:

1. ``KGWWatermark.apply(logits, context_tokens)`` — bias green-list tokens.
2. ``KGWWatermark.detect(token_ids, z_threshold)`` — z-test detection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Set

import numpy as np

__all__ = [
    "KGWConfig",
    "WatermarkResult",
    "KGWWatermark",
]


@dataclass
class KGWConfig:
    """Configuration for :class:`KGWWatermark`.

    Attributes:
        vocab_size: Vocabulary size.
        gamma: Fraction of the vocabulary in the green list (0 < gamma < 1).
        delta: Additive logit bias for green tokens.
        hash_key: Large prime used for context-hash seeding.
        seed: RNG seed for tie-breaking only.
    """

    vocab_size: int = 32000
    gamma: float = 0.25
    delta: float = 2.0
    hash_key: int = 15485863
    seed: int = 0

    def __post_init__(self) -> None:
        if self.vocab_size < 2:
            raise ValueError(f"vocab_size must be >= 2; got {self.vocab_size}")
        if not (0.0 < self.gamma < 1.0):
            raise ValueError(f"gamma must be in (0, 1); got {self.gamma}")
        if self.delta < 0.0:
            raise ValueError(f"delta must be >= 0; got {self.delta}")


@dataclass
class WatermarkResult:
    """Detection result from :meth:`KGWWatermark.detect`.

    Attributes:
        z_score: One-sided z statistic.  Values >> 4 indicate watermarked text.
        is_watermarked: Whether z_score exceeds the detection threshold.
        green_count: Number of tokens in the green list.
        total_tokens: Total tokens analysed.
    """

    z_score: float
    is_watermarked: bool
    green_count: int
    total_tokens: int


class KGWWatermark:
    """Context-seeded green/red list watermark.

    Example::

        cfg = KGWConfig(vocab_size=128, gamma=0.25, delta=2.0)
        wm = KGWWatermark(cfg)

        logits = np.zeros(128)
        biased = wm.apply(logits, context_tokens=[42])
        result = wm.detect(token_ids=[3, 7, 12, 5], z_threshold=4.0)
    """

    def __init__(self, config: Optional[KGWConfig] = None) -> None:
        self._cfg = config or KGWConfig()

    @property
    def config(self) -> KGWConfig:
        return self._cfg

    def _get_green_list(self, context_token: int) -> Set[int]:
        """Return the green-list token set for the given context token.

        The green list is deterministic given ``context_token`` and the
        hash key, enabling stateless detection.

        Args:
            context_token: The last generated token (context seed).

        Returns:
            Set of green-list token indices.
        """
        cfg = self._cfg
        # Use a seeded RNG derived from the context token and hash key.
        seed = int(context_token) * cfg.hash_key % (2**31 - 1)
        rng = np.random.default_rng(seed)
        n_green = max(1, int(cfg.gamma * cfg.vocab_size))
        green_ids = rng.choice(cfg.vocab_size, size=n_green, replace=False)
        return set(int(g) for g in green_ids)

    def apply(self, logits: np.ndarray, context_tokens: List[int]) -> np.ndarray:
        """Add delta bias to green-list logits.

        Args:
            logits: ``(vocab_size,)`` float logit array.
            context_tokens: Preceding token sequence.  The *last* token is
                used as the context seed.  If empty, a fixed seed of 0 is used.

        Returns:
            Biased ``(vocab_size,)`` logit array.
        """
        logits = np.asarray(logits, dtype=np.float32).copy()
        ctx = int(context_tokens[-1]) if context_tokens else 0
        green = self._get_green_list(ctx)
        for tok in green:
            logits[tok] += self._cfg.delta
        return logits

    def detect(
        self, token_ids: List[int], z_threshold: float = 4.0
    ) -> WatermarkResult:
        """Run the z-test to detect watermarking in a token sequence.

        The null hypothesis is that each token is green independently with
        probability ``gamma``.  The test statistic is:

        .. math::

            z = \\frac{|G| - \\gamma T}{\\sqrt{\\gamma(1-\\gamma)T}}

        where ``|G|`` is the number of green tokens and ``T`` is the total.

        Args:
            token_ids: The token sequence to test (must be length ≥ 2 so
                each token has a preceding context token).
            z_threshold: Detection threshold; default 4.0 (p ≈ 3×10⁻⁵).

        Returns:
            :class:`WatermarkResult` with z-score and detection flag.
        """
        if len(token_ids) < 2:
            return WatermarkResult(
                z_score=0.0,
                is_watermarked=False,
                green_count=0,
                total_tokens=len(token_ids),
            )
        green_count = 0
        for i in range(1, len(token_ids)):
            green = self._get_green_list(token_ids[i - 1])
            if token_ids[i] in green:
                green_count += 1
        T = len(token_ids) - 1  # number of tested positions
        gamma = self._cfg.gamma
        expected = gamma * T
        std = np.sqrt(gamma * (1.0 - gamma) * T) if T > 0 else 1.0
        z_score = float((green_count - expected) / max(std, 1e-8))
        return WatermarkResult(
            z_score=z_score,
            is_watermarked=z_score >= z_threshold,
            green_count=green_count,
            total_tokens=T,
        )
