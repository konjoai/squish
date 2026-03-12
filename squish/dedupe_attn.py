"""DedupeAttn — Attention deduplication for repetitive context.

In repetitive context (e.g., repeated system prompts, long documents with
recurring passages), many query vectors produce near-identical attention
outputs.  :class:`AttentionDeduplicator` detects near-duplicate queries via
cosine similarity and returns the cached output instead of recomputing
attention, cutting FLOPs proportionally to the cache hit rate.

A per-head FIFO buffer of up to ``max_cache`` (query, output) pairs is
maintained.  Queries are stored in L2-normalised form so that the inner
product during lookup is equivalent to cosine similarity.

Usage::

    import numpy as np
    from squish.dedupe_attn import AttentionDeduplicator, DedupConfig

    cfg   = DedupConfig(sim_threshold=0.99, max_cache=512, n_heads=8, head_dim=64)
    dedup = AttentionDeduplicator(cfg)

    rng = np.random.default_rng(0)
    q   = rng.standard_normal((8, 64)).astype(np.float32)
    out = rng.standard_normal((8, 64)).astype(np.float32)

    for h in range(8):
        cached = dedup.lookup(q[h], h)
        if cached is not None:
            out[h] = cached
        else:
            # compute_attention(...)
            dedup.store(q[h], out[h], h)

    print(f"Hit rate: {dedup.stats.hit_rate:.3f}")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

__all__ = ["DedupConfig", "AttentionDeduplicator", "DedupStats"]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class DedupConfig:
    """Configuration for :class:`AttentionDeduplicator`.

    Attributes:
        sim_threshold: Cosine-similarity threshold for a cache hit.
            Must be in ``(0, 1]``; typical values are ``0.95`` – ``0.999``.
        max_cache: Maximum number of ``(query, output)`` pairs to retain per
            head.  Oldest entries are evicted (FIFO) when the limit is reached.
        n_heads: Number of attention heads.
        head_dim: Dimension of each head's query/output vector.
    """

    sim_threshold: float = 0.99
    max_cache: int = 512
    n_heads: int = 8
    head_dim: int = 64

    def __post_init__(self) -> None:
        if not (0.0 < self.sim_threshold <= 1.0):
            raise ValueError(
                f"sim_threshold must be in (0, 1]; got {self.sim_threshold}"
            )
        if self.max_cache < 1:
            raise ValueError(f"max_cache must be >= 1; got {self.max_cache}")
        if self.n_heads < 1:
            raise ValueError(f"n_heads must be >= 1; got {self.n_heads}")
        if self.head_dim < 1:
            raise ValueError(f"head_dim must be >= 1; got {self.head_dim}")


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


@dataclass
class DedupStats:
    """Accumulated statistics for :class:`AttentionDeduplicator`.

    Attributes:
        n_lookups: Total number of lookup calls.
        n_hits: Number of lookups that returned a cached result.
        n_stores: Total number of store calls.
    """

    n_lookups: int = 0
    n_hits: int = 0
    n_stores: int = 0

    @property
    def hit_rate(self) -> float:
        """Fraction of lookups that returned a cache hit."""
        if self.n_lookups == 0:
            return 0.0
        return self.n_hits / self.n_lookups


# ---------------------------------------------------------------------------
# Deduplicator
# ---------------------------------------------------------------------------


class AttentionDeduplicator:
    """Near-duplicate attention output cache keyed on query cosine similarity.

    For each attention head a FIFO buffer of up to ``config.max_cache``
    ``(normalised_query, output)`` pairs is maintained.  On lookup the cosine
    similarity between the incoming query and all cached queries is computed
    via a vectorised dot product (queries are stored pre-normalised); if the
    best similarity exceeds ``config.sim_threshold``, the corresponding cached
    output is returned without recomputing attention.

    Queries with near-zero L2 norm are passed through without caching to avoid
    numerical issues with normalisation.

    Args:
        config: :class:`DedupConfig` instance.
    """

    def __init__(self, config: DedupConfig) -> None:
        self.config = config
        # Per-head storage: list of (normed_query, output) tuples
        self._cache: list[list[tuple[np.ndarray, np.ndarray]]] = [
            [] for _ in range(config.n_heads)
        ]
        self._stats = DedupStats()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def lookup(self, q: np.ndarray, head_idx: int) -> Optional[np.ndarray]:
        """Return a cached attention output for a near-duplicate query.

        Args:
            q: Query vector, shape ``(head_dim,)`` float32.
            head_idx: Zero-based attention head index.

        Returns:
            Cached attention output of shape ``(head_dim,)`` float32 if a
            near-duplicate query is found, otherwise ``None``.

        Raises:
            ValueError: If ``q`` has an unexpected shape or ``head_idx`` is
                out of range.
        """
        cfg = self.config
        if q.shape != (cfg.head_dim,):
            raise ValueError(
                f"Expected query shape ({cfg.head_dim},); got {q.shape}"
            )
        if head_idx < 0 or head_idx >= cfg.n_heads:
            raise ValueError(
                f"head_idx must be in [0, {cfg.n_heads}); got {head_idx}"
            )

        self._stats.n_lookups += 1
        bucket = self._cache[head_idx]
        if not bucket:
            return None

        q_f = q.astype(np.float32)
        norm_q = float(np.linalg.norm(q_f))
        if norm_q < 1e-9:
            # Zero-norm query cannot be meaningfully compared
            return None

        q_normed = q_f / norm_q

        # Vectorised cosine similarity: cached queries are already normalised
        cached_queries = np.stack(
            [entry[0] for entry in bucket], axis=0
        )  # (m, head_dim)
        sims = cached_queries @ q_normed  # (m,)

        best_idx = int(np.argmax(sims))
        if sims[best_idx] >= cfg.sim_threshold:
            self._stats.n_hits += 1
            return bucket[best_idx][1].copy()

        return None

    def store(self, q: np.ndarray, output: np.ndarray, head_idx: int) -> None:
        """Cache a ``(query, output)`` pair for future deduplication.

        Queries with near-zero L2 norm are silently discarded.  When the
        per-head buffer is full, the oldest entry is evicted (FIFO).

        Args:
            q: Query vector, shape ``(head_dim,)`` float32.
            output: Attention output to cache, shape ``(head_dim,)`` float32.
            head_idx: Zero-based attention head index.

        Raises:
            ValueError: If shapes are incorrect or ``head_idx`` is out of
                range.
        """
        cfg = self.config
        if q.shape != (cfg.head_dim,):
            raise ValueError(
                f"Expected query shape ({cfg.head_dim},); got {q.shape}"
            )
        if output.shape != (cfg.head_dim,):
            raise ValueError(
                f"Expected output shape ({cfg.head_dim},); got {output.shape}"
            )
        if head_idx < 0 or head_idx >= cfg.n_heads:
            raise ValueError(
                f"head_idx must be in [0, {cfg.n_heads}); got {head_idx}"
            )

        q_f = q.astype(np.float32)
        norm_q = float(np.linalg.norm(q_f))
        if norm_q < 1e-9:
            # Discard degenerate queries
            return

        q_normed = q_f / norm_q
        bucket = self._cache[head_idx]

        if len(bucket) >= cfg.max_cache:
            bucket.pop(0)  # Evict oldest entry (FIFO)

        bucket.append((q_normed, output.astype(np.float32).copy()))
        self._stats.n_stores += 1

    def clear(self) -> None:
        """Empty all per-head caches without resetting statistics."""
        for bucket in self._cache:
            bucket.clear()

    def reset_stats(self) -> None:
        """Reset accumulated statistics to zero."""
        self._stats = DedupStats()

    @property
    def stats(self) -> DedupStats:
        """Current accumulated :class:`DedupStats`."""
        return self._stats
