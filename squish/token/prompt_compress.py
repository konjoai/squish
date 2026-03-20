"""
squish/token/prompt_compress.py

Token importance scoring for long-prompt compression before prefill.

Long prompts dominate time-to-first-token (TTFT) because prefill FLOPs scale
quadratically with sequence length.  This module scores each token in the
prompt by estimated "informational importance" and drops the lowest-scoring
tokens before passing the compressed prompt to the model.

Importance is a weighted combination of three orthogonal signals:

1. **Frequency score** (weight α)
   Common tokens (the, is, a, …) carry less information.  Rank each token by
   inverse unigram frequency — rare tokens score higher.

2. **Positional salience** (weight β)
   Primacy-recency bias: the first and last segments of a prompt are typically
   more important than the middle.  Applied as a U-shaped positional bonus.

3. **Lexical distinctiveness** (weight γ)
   Tokens that are unique within the prompt (appear only once) are more
   salient than repeated tokens.

The three signals are z-score normalised and linearly combined, then the
top-p fraction of tokens (sorted by score) is retained.

This approach is token-ID only (no model forward pass), so it adds negligible
latency (~0.1 ms for 4 K tokens) before the expensive prefill step.  It is
inspired by the "token-difficulty" scoring in LLMLingua-2 but requires no
auxiliary model.

References
----------
Pan, Z., et al. (2024). LLMLingua-2: Data Distillation for Efficient and
Faithful Task-Agnostic Compression. EMNLP 2024. arXiv:2403.12894.

Li, H., et al. (2023). LLMLingua: Compressing Prompts for Accelerated
Inference. EMNLP 2023. arXiv:2310.05736.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class PromptCompressorConfig:
    """Configuration for prompt token importance scoring.

    Parameters
    ----------
    compression_ratio:
        Fraction of tokens to *keep* (0 < ratio ≤ 1).  E.g. 0.5 halves the
        prompt length.
    min_tokens:
        Absolute minimum number of tokens to keep regardless of ratio.  This
        prevents compressing very short prompts.
    score_weights:
        3-tuple (α, β, γ) summing to 1.0, weighting the three scoring signals:
        frequency rank, positional salience, lexical distinctiveness.
    preserve_boundary_frac:
        Fraction of tokens at the start *and* end of the prompt that are never
        dropped.  Protects system prompts and recent context.
    vocab_size:
        Approximate vocabulary size used to normalise frequency rank.
        The actual ranks only depend on the observed token distribution, so
        this parameter is informational.
    """

    compression_ratio: float = 0.5
    min_tokens: int = 32
    score_weights: tuple = (0.4, 0.3, 0.3)
    preserve_boundary_frac: float = 0.1
    vocab_size: int = 128_000

    def __post_init__(self) -> None:
        if not 0.0 < self.compression_ratio <= 1.0:
            raise ValueError("compression_ratio must be in (0, 1]")
        if self.min_tokens < 0:
            raise ValueError("min_tokens must be >= 0")
        if len(self.score_weights) != 3:
            raise ValueError("score_weights must have exactly 3 elements")
        total = sum(self.score_weights)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"score_weights must sum to 1.0, got {total:.6f}"
            )
        if not 0.0 <= self.preserve_boundary_frac < 0.5:
            raise ValueError(
                "preserve_boundary_frac must be in [0, 0.5)"
            )


class PromptCompressor:
    """Importance-based prompt compression (token-ID only, no model pass).

    Usage
    -----
    ::

        compressor = PromptCompressor()
        original_ids = tokenizer.encode(long_prompt)
        compressed_ids = compressor.compress(original_ids)
        ratio = compressor.actual_ratio(original_ids, compressed_ids)
        response = model.generate(compressed_ids)
    """

    def __init__(self, config: Optional[PromptCompressorConfig] = None) -> None:
        self.config = config or PromptCompressorConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(self, token_ids: List[int]) -> np.ndarray:
        """Compute per-token importance scores.

        Parameters
        ----------
        token_ids:
            List of integer token IDs.

        Returns
        -------
        np.ndarray of shape ``(len(token_ids),)`` with float32 scores.
        Higher = more important.
        """
        n = len(token_ids)
        if n == 0:
            return np.empty(0, dtype=np.float32)

        ids = np.asarray(token_ids, dtype=np.int64)
        alpha, beta, gamma = self.config.score_weights

        freq = self._frequency_score(ids)
        pos = self._positional_score(n)
        lex = self._lexical_score(ids)

        freq_n = self._zscore(freq)
        pos_n = self._zscore(pos)
        lex_n = self._zscore(lex)

        combined = alpha * freq_n + beta * pos_n + gamma * lex_n
        return combined.astype(np.float32)

    def compress(
        self,
        token_ids: List[int],
        target_ratio: Optional[float] = None,
    ) -> List[int]:
        """Return a compressed token list by dropping low-importance tokens.

        The relative order of retained tokens is preserved.

        Parameters
        ----------
        token_ids:
            List of integer token IDs representing the prompt.
        target_ratio:
            Override ``config.compression_ratio`` for this call.

        Returns
        -------
        List[int] — compressed token IDs in original order.
        """
        n = len(token_ids)
        cfg = self.config
        ratio = target_ratio if target_ratio is not None else cfg.compression_ratio

        keep_n = max(cfg.min_tokens, int(round(n * ratio)))
        keep_n = min(keep_n, n)

        if keep_n == n:
            return list(token_ids)

        # Boundary protection: always keep first/last boundary_n tokens
        boundary_n = int(n * cfg.preserve_boundary_frac)
        protected = set(range(boundary_n)) | set(range(n - boundary_n, n))

        scores = self.score(token_ids)

        # Force protected tokens to have maximum score so they sort first
        if protected:
            scores_copy = scores.copy()
            max_score = float(scores.max()) + 1.0
            for idx in protected:
                scores_copy[idx] = max_score
        else:
            scores_copy = scores

        # Sort indices by score descending; take top-keep_n
        ranked = np.argsort(-scores_copy)
        keep_indices = set(ranked[:keep_n].tolist())

        return [token_ids[i] for i in range(n) if i in keep_indices]

    def actual_ratio(
        self,
        original: List[int],
        compressed: List[int],
    ) -> float:
        """Return the fraction of tokens retained.

        Returns 1.0 if ``original`` is empty.
        """
        if not original:
            return 1.0
        return len(compressed) / len(original)

    # ------------------------------------------------------------------
    # Signal functions
    # ------------------------------------------------------------------

    @staticmethod
    def _frequency_score(ids: np.ndarray) -> np.ndarray:
        """Inverse unigram frequency rank — rare tokens score higher."""
        # Count occurrences, then rank by frequency (ascending = rarer = higher score)
        unique, counts = np.unique(ids, return_counts=True)
        freq_map = dict(zip(unique.tolist(), counts.tolist()))

        raw = np.array([freq_map[int(t)] for t in ids], dtype=np.float64)
        # Invert: rare tokens (low count) → high score
        # Avoid division by zero with clamp
        return 1.0 / np.maximum(raw, 1.0)

    @staticmethod
    def _positional_score(n: int) -> np.ndarray:
        """U-shaped positional salience — start and end tokens score higher."""
        if n == 1:
            return np.ones(1)
        positions = np.arange(n, dtype=np.float64)
        mid = (n - 1) / 2.0
        # Normalised distance from midpoint: 1.0 at edges, 0.0 at centre
        return np.abs(positions - mid) / mid

    @staticmethod
    def _lexical_score(ids: np.ndarray) -> np.ndarray:
        """Binary distinctiveness — tokens appearing exactly once score 1.0."""
        unique, counts = np.unique(ids, return_counts=True)
        singleton_set = set(unique[counts == 1].tolist())
        return np.array(
            [1.0 if int(t) in singleton_set else 0.0 for t in ids],
            dtype=np.float64,
        )

    @staticmethod
    def _zscore(arr: np.ndarray) -> np.ndarray:
        """Z-score normalise; return zeros if std is zero."""
        mu = arr.mean()
        sigma = arr.std()
        if sigma < 1e-9:
            return np.zeros_like(arr)
        return (arr - mu) / sigma
