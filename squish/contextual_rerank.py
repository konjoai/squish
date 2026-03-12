"""squish/contextual_rerank.py

ContextualRerank — Context-aware KV token importance re-ranking to guide
cache eviction in long-context LLM inference.

Standard KV cache eviction policies such as sliding-window or recency-based
eviction discard the oldest tokens regardless of their semantic importance.
This leads to degraded quality on tasks that require attending back to key
facts mentioned early in the context (e.g., system instructions, document
titles, entity definitions).  A better eviction policy preserves the tokens
that carry the most directional information while still accounting for the
natural bias toward recent tokens introduced by RoPE and causal masking.

ContextualRerank scores each cached key position using a weighted combination
of two signals: (1) a *norm score* — the mean L2 norm of the key vector across
all attention heads, normalised to ``[0, 1]`` — which identifies positions
whose keys have high magnitude and thus strong directional content; and (2) a
*recency score* — a scalar linearly interpolated from 0.0 (oldest) to 1.0
(newest) over the sequence length — which captures the empirical observation
that recently generated tokens are more likely to be attended in the near
future.  The ``recency_weight`` hyperparameter in ``[0, 1]`` controls the
blend: 0 is purely norm-driven; 1 is purely recency-driven.

When a *query* vector is provided the norm score is replaced by the mean
dot-product of each key with the query, normalised to ``[0, 1]``.  This
query-conditional mode allows callers to identify which cached positions are
most relevant to the current decoding step, enabling speculative prefetch or
targeted retention.

Example usage::

    import numpy as np
    from squish.contextual_rerank import RerankConfig, ContextualReranker

    cfg      = RerankConfig(n_heads=8, head_dim=64, recency_weight=0.5, top_k=32)
    reranker = ContextualReranker(cfg)

    keys  = np.random.randn(8, 128, 64).astype(np.float32)
    top_k = reranker.rerank(keys)
    print(f"top-{len(top_k)} positions: {top_k[:5]}")
    print(reranker.stats)
"""

from __future__ import annotations

__all__ = ["RerankConfig", "ContextualReranker", "RerankStats"]

from dataclasses import dataclass
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class RerankConfig:
    """Configuration for context-aware KV position re-ranking.

    Attributes:
        n_heads:        Number of attention heads.
        head_dim:       Dimension of each attention head.
        recency_weight: Weight given to the recency signal in ``[0, 1]``.
                        The norm/dot-product signal receives weight
                        ``1 - recency_weight``.
        top_k:          Number of top-scoring positions to return.
    """

    n_heads: int = 8
    head_dim: int = 64
    recency_weight: float = 0.5
    top_k: int = 64

    def __post_init__(self) -> None:
        if self.n_heads < 1:
            raise ValueError(f"n_heads must be >= 1, got {self.n_heads}")
        if self.head_dim < 1:
            raise ValueError(f"head_dim must be >= 1, got {self.head_dim}")
        if not (0.0 <= self.recency_weight <= 1.0):
            raise ValueError(
                f"recency_weight must be in [0, 1], got {self.recency_weight}"
            )
        if self.top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {self.top_k}")


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


@dataclass
class RerankStats:
    """Aggregate statistics for a :class:`ContextualReranker`.

    Attributes:
        total_rerank_calls:    Total number of :meth:`~ContextualReranker.rerank`
                               invocations.
        total_positions_ranked: Cumulative number of key positions scored
                                across all calls.
    """

    total_rerank_calls: int = 0
    total_positions_ranked: int = 0


# ---------------------------------------------------------------------------
# Reranker
# ---------------------------------------------------------------------------


class ContextualReranker:
    """Re-ranks KV cache positions by a blend of norm-saliency and recency.

    Positions are scored and the top ``top_k`` indices (sorted by score
    descending) are returned for use by a downstream eviction policy that
    retains exactly those positions.

    Args:
        config: A :class:`RerankConfig` instance.
    """

    def __init__(self, config: RerankConfig) -> None:
        self._cfg = config
        self._total_rerank_calls:    int = 0
        self._total_positions_ranked: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def rerank(
        self,
        keys: np.ndarray,
        query: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Score and rank all key positions, returning the top-k indices.

        Args:
            keys:  Shape ``(n_heads, seq_len, head_dim)``.  The cached key
                   tensor for a single layer.
            query: Optional shape ``(n_heads, head_dim)``.  When provided,
                   the norm score is replaced by the mean dot-product of each
                   key position with the query vector.

        Returns:
            A 1-D int64 array of up to ``min(top_k, seq_len)`` position
            indices sorted by final score in *descending* order (most salient
            first).

        Raises:
            ValueError: If *keys* does not have shape
                        ``(n_heads, seq_len, head_dim)`` or if *query* does
                        not have shape ``(n_heads, head_dim)``.
        """
        keys = np.asarray(keys, dtype=np.float32)
        if keys.ndim != 3:
            raise ValueError(
                f"keys must be 3-D (n_heads, seq_len, head_dim), "
                f"got shape {keys.shape}."
            )
        kh, seq_len, kdim = keys.shape
        if kh != self._cfg.n_heads:
            raise ValueError(
                f"keys n_heads={kh} does not match config.n_heads={self._cfg.n_heads}."
            )
        if kdim != self._cfg.head_dim:
            raise ValueError(
                f"keys head_dim={kdim} does not match "
                f"config.head_dim={self._cfg.head_dim}."
            )

        # Recency score: linearly spaced [0, 1] over seq_len.
        recency = np.linspace(0.0, 1.0, seq_len, dtype=np.float64)

        # Compute the content-based score (norm or dot-product).
        if query is None:
            # L2 norm of each key position, averaged over heads.
            norms = np.linalg.norm(keys, axis=2)  # (n_heads, seq_len)
            raw_score = np.mean(norms, axis=0).astype(np.float64)  # (seq_len,)
        else:
            query = np.asarray(query, dtype=np.float32)
            if query.shape != (self._cfg.n_heads, self._cfg.head_dim):
                raise ValueError(
                    f"query must have shape "
                    f"({self._cfg.n_heads}, {self._cfg.head_dim}), "
                    f"got {query.shape}."
                )
            # Dot-product of each key position with the query, per head.
            # keys: (n_heads, seq_len, head_dim), query: (n_heads, head_dim)
            # dots[h, s] = keys[h, s, :] · query[h, :]
            dots = np.einsum("hsd,hd->hs", keys, query)  # (n_heads, seq_len)
            raw_score = np.mean(dots, axis=0).astype(np.float64)  # (seq_len,)

        # Normalise the content score to [0, 1].
        score_min = raw_score.min()
        score_max = raw_score.max()
        score_range = score_max - score_min
        if score_range > 0.0:
            norm_score = (raw_score - score_min) / score_range
        else:
            norm_score = np.zeros(seq_len, dtype=np.float64)

        # Final blended score.
        final_score = (
            (1.0 - self._cfg.recency_weight) * norm_score
            + self._cfg.recency_weight * recency
        )

        # Select top-k positions by score (descending).
        k = min(self._cfg.top_k, seq_len)
        if k >= seq_len:
            top_idx = np.argsort(-final_score)
        else:
            # argpartition for O(n) partial selection, then sort the partition.
            part = np.argpartition(-final_score, k - 1)[:k]
            top_idx = part[np.argsort(-final_score[part])]

        self._total_rerank_calls      += 1
        self._total_positions_ranked  += seq_len

        return top_idx.astype(np.int64)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def stats(self) -> RerankStats:
        """Return a snapshot of cumulative re-ranking statistics."""
        return RerankStats(
            total_rerank_calls=self._total_rerank_calls,
            total_positions_ranked=self._total_positions_ranked,
        )
