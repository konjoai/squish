# [Experimental] This module is part of Squish v6+ (Waves 19–26).
# Proof-of-concept quality: API and behaviour may change without notice.
# For stable core functionality see Waves 1–12 in MODULES.md.
"""squish/cot_compress.py

CoTCompress — Saliency-driven pruning of chain-of-thought reasoning traces.

Large language models performing multi-step reasoning emit extended
chain-of-thought (CoT) traces before producing a final answer.  These traces
can span hundreds or thousands of tokens, most of which carry redundant
information — filler phrases, repetitive connective words, and common
high-frequency tokens that appear throughout but contribute little unique
signal.  Compressing such traces reduces KV cache pressure, decreases input
lengths for subsequent reasoning steps, and lowers latency.

CoTCompress uses a TF-IDF-inspired inverse-frequency saliency metric.  For
each token position, the term frequency TF is defined as the fraction of
positions occupied by the same token ID: ``TF(i) = count(token_id[i]) / N``.
Saliency is then ``1 / TF(i) = N / count(token_id[i])``, which assigns high
scores to rare tokens (unusual reasoning words, domain-specific terms,
numerical values) and low scores to common tokens (articles, punctuation,
filler verbs).  Positions are ranked by saliency and the top
``(1 - compress_ratio)`` fraction are retained, subject to a ``min_tokens``
floor to prevent over-compression on short traces.

Retained positions are returned in their original sequential order, preserving
syntactic and logical coherence of the compressed trace.  The compressor
accumulates throughput statistics across calls to support offline analysis of
average compression ratios over a session.

Example usage::

    import numpy as np
    from squish.context.cot_compress import CoTConfig, CoTCompressor

    cfg        = CoTConfig(compress_ratio=0.4, min_tokens=16)
    compressor = CoTCompressor(cfg)

    # Simulate a 120-token CoT trace.
    rng    = np.random.default_rng(0)
    tokens = rng.integers(0, 50_000, size=120, dtype=np.int64)
    out    = compressor.compress(tokens)
    print(f"in={len(tokens)}, out={len(out)}")
    print(f"avg_compression_ratio={compressor.stats.avg_compression_ratio:.3f}")
"""

from __future__ import annotations

__all__ = ["CoTConfig", "CoTCompressor", "CoTStats"]

from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class CoTConfig:
    """Configuration for chain-of-thought trace compression.

    Attributes:
        compress_ratio: Fraction of tokens to remove.  Must be in ``[0, 1)``.
        min_tokens:     Minimum number of tokens to keep regardless of
                        *compress_ratio*.  Must be >= 1.
        smoothing:      Small constant used to guard against potential
                        zero-division when computing saliency.  Must be > 0.
    """

    compress_ratio: float = 0.4
    min_tokens: int = 16
    smoothing: float = 1e-9

    def __post_init__(self) -> None:
        if not (0.0 <= self.compress_ratio < 1.0):
            raise ValueError(
                f"compress_ratio must be in [0, 1), got {self.compress_ratio}"
            )
        if self.min_tokens < 1:
            raise ValueError(
                f"min_tokens must be >= 1, got {self.min_tokens}"
            )
        if self.smoothing <= 0.0:
            raise ValueError(
                f"smoothing must be > 0, got {self.smoothing}"
            )


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


@dataclass
class CoTStats:
    """Aggregate statistics for a :class:`CoTCompressor`.

    Attributes:
        total_compress_calls: Total number of :meth:`~CoTCompressor.compress`
                              invocations.
        total_tokens_in:      Cumulative input token count across all calls.
        total_tokens_out:     Cumulative output token count across all calls.
    """

    total_compress_calls: int = 0
    total_tokens_in: int = 0
    total_tokens_out: int = 0

    @property
    def avg_compression_ratio(self) -> float:
        """Fraction of tokens removed on average: ``1 - out / in``.

        Returns ``0.0`` when no tokens have been processed.
        """
        return 1.0 - self.total_tokens_out / (self.total_tokens_in + 1e-9)


# ---------------------------------------------------------------------------
# Compressor
# ---------------------------------------------------------------------------


class CoTCompressor:
    """Prunes chain-of-thought traces by inverse-frequency token saliency.

    Rare tokens receive high saliency scores and are retained; common
    high-frequency tokens receive low scores and are pruned.  The compressed
    output preserves the original token order.

    Args:
        config: A :class:`CoTConfig` instance.
    """

    def __init__(self, config: CoTConfig) -> None:
        self._cfg = config
        self._stats = CoTStats()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compress(self, token_ids: np.ndarray) -> np.ndarray:
        """Remove low-saliency tokens from *token_ids*.

        Args:
            token_ids: 1-D integer array of token IDs representing the CoT
                       trace.  Accepted dtypes: ``int32``, ``int64``.

        Returns:
            A 1-D integer array of token IDs with low-saliency positions
            removed, in the same order as the input.

        Raises:
            ValueError: If *token_ids* is not a 1-D array.
        """
        token_ids = np.asarray(token_ids)
        if token_ids.ndim != 1:
            raise ValueError(
                f"token_ids must be 1-D, got shape {token_ids.shape}."
            )

        n = len(token_ids)

        # Determine how many tokens to keep.
        n_keep = max(self._cfg.min_tokens, round((1.0 - self._cfg.compress_ratio) * n))
        n_keep = min(n_keep, n)

        if n_keep >= n:
            # Nothing to prune.
            self._stats.total_compress_calls += 1
            self._stats.total_tokens_in      += n
            self._stats.total_tokens_out     += n
            return token_ids.copy()

        # Compute per-position saliency = N / count(token_id[i]).
        unique_ids, counts = np.unique(token_ids, return_counts=True)
        # Build a mapping from token id to its count using a direct lookup.
        count_lookup: dict[int, int] = dict(
            zip(unique_ids.tolist(), counts.tolist())
        )
        # Saliency: rare tokens score high; common tokens score low.
        saliency = np.array(
            [n / (count_lookup[int(t)] + self._cfg.smoothing) for t in token_ids],
            dtype=np.float64,
        )

        # Select the top n_keep positions by saliency.
        top_positions = np.argpartition(-saliency, n_keep - 1)[:n_keep]
        kept_positions = np.sort(top_positions)

        result = token_ids[kept_positions]

        self._stats.total_compress_calls += 1
        self._stats.total_tokens_in      += n
        self._stats.total_tokens_out     += len(result)

        return result

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def stats(self) -> CoTStats:
        """Return a snapshot of cumulative compression statistics."""
        return CoTStats(
            total_compress_calls=self._stats.total_compress_calls,
            total_tokens_in=self._stats.total_tokens_in,
            total_tokens_out=self._stats.total_tokens_out,
        )
