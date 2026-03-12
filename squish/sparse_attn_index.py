"""SparseAttnIndex — ANN KV retrieval index for very long context attention.

For very long sequences, full attention over all KV entries is prohibitively
expensive.  This module approximates attention by retrieving only the top-k
most relevant KV entries per query head via cosine-similarity search, reducing
the effective attention window from O(seq_len) to O(top_k).

The numpy fallback uses exact cosine similarity (O(seq_len × head_dim)),
which is correct and predictable.  The ``n_probe`` parameter is preserved
for interface compatibility with production ANN backends (e.g. FAISS IVF).

References:
    Kitaev et al., "Reformer: The Efficient Transformer", ICLR 2020.
    https://arxiv.org/abs/2001.04451

    Zaheer et al., "Big Bird: Transformers for Longer Sequences", NeurIPS 2020.
    https://arxiv.org/abs/2007.14062

Usage::

    from squish.sparse_attn_index import SparseAttnIndex, IndexConfig
    import numpy as np

    cfg   = IndexConfig(top_k=64, head_dim=64, n_heads=8)
    index = SparseAttnIndex(cfg)

    keys  = np.random.randn(8, 1024, 64).astype(np.float32)
    index.build(keys)

    q   = np.random.randn(8, 64).astype(np.float32)
    res = index.query(q)
    print(f"top_k indices shape: {res.indices.shape}")   # (8, 64)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "IndexConfig",
    "ANCandidates",
    "SparseAttnIndex",
    "IndexStats",
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class IndexConfig:
    """Configuration for the sparse attention ANN retrieval index.

    Attributes:
        top_k:    Number of KV candidates to retrieve per head per query.
        head_dim: Dimension of each key / query vector.
        n_heads:  Number of attention heads.
        n_probe:  Number of IVF clusters to probe (preserved for interface
                  compatibility with ANN backends; unused in numpy fallback).
    """

    top_k: int = 64
    head_dim: int = 64
    n_heads: int = 8
    n_probe: int = 8

    def __post_init__(self) -> None:
        if self.top_k < 1:
            raise ValueError(f"top_k must be >= 1; got {self.top_k}")
        if self.head_dim < 1:
            raise ValueError(f"head_dim must be >= 1; got {self.head_dim}")
        if self.n_heads < 1:
            raise ValueError(f"n_heads must be >= 1; got {self.n_heads}")
        if self.n_probe < 1:
            raise ValueError(f"n_probe must be >= 1; got {self.n_probe}")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ANCandidates:
    """Top-k ANN retrieval results for all heads.

    Attributes:
        indices: Top-k key position indices per head, shape
                 ``(n_heads, top_k)``, int32.  Sorted in descending
                 similarity order.
        scores:  Cosine similarity scores corresponding to ``indices``,
                 shape ``(n_heads, top_k)``, float32.
    """

    indices: np.ndarray
    scores: np.ndarray


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


@dataclass
class IndexStats:
    """Cumulative statistics for :class:`SparseAttnIndex`.

    Attributes:
        n_builds:           Number of :meth:`~SparseAttnIndex.build` calls.
        n_queries:          Number of :meth:`~SparseAttnIndex.query` calls.
        total_keys_indexed: Total key vectors indexed across all builds.
    """

    n_builds: int = 0
    n_queries: int = 0
    total_keys_indexed: int = 0


# ---------------------------------------------------------------------------
# SparseAttnIndex
# ---------------------------------------------------------------------------


class SparseAttnIndex:
    """In-memory cosine-similarity ANN index for sparse attention.

    Builds a per-head L2-normalised key store and retrieves the top-k most
    similar key positions for each query head using exact cosine similarity
    (numpy fallback).

    Args:
        config: :class:`IndexConfig` instance.
    """

    def __init__(self, config: IndexConfig) -> None:
        self._config = config
        self._keys_norm: np.ndarray | None = None  # (n_heads, seq_len, head_dim)
        self._seq_len: int = 0
        self._stats = IndexStats()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self, keys: np.ndarray) -> None:
        """Build the retrieval index from a full key cache.

        Keys are L2-normalised per-vector before storage so that dot-product
        similarity equals cosine similarity at query time.

        Args:
            keys: Key tensor, shape ``(n_heads, seq_len, head_dim)``,
                  float32.

        Raises:
            ValueError: if ``keys`` shape is inconsistent with the config.
        """
        cfg = self._config
        if keys.ndim != 3:
            raise ValueError(
                f"keys must be 3-D (n_heads, seq_len, head_dim); "
                f"got shape {keys.shape}"
            )
        if keys.shape[0] != cfg.n_heads:
            raise ValueError(
                f"keys.shape[0]={keys.shape[0]} does not match "
                f"config n_heads={cfg.n_heads}"
            )
        if keys.shape[2] != cfg.head_dim:
            raise ValueError(
                f"keys.shape[2]={keys.shape[2]} does not match "
                f"config head_dim={cfg.head_dim}"
            )

        seq_len = keys.shape[1]
        norms = np.linalg.norm(keys, axis=-1, keepdims=True)  # (n_heads, seq_len, 1)
        # Avoid division by zero for zero vectors.
        norms = np.where(norms < 1e-30, 1.0, norms)
        self._keys_norm = (keys / norms).astype(np.float32)
        self._seq_len = seq_len

        self._stats.n_builds += 1
        self._stats.total_keys_indexed += cfg.n_heads * seq_len

    def query(self, q: np.ndarray) -> ANCandidates:
        """Retrieve the top-k most similar key positions for each query head.

        Args:
            q: Query tensor, shape ``(n_heads, head_dim)``, float32.

        Returns:
            :class:`ANCandidates` with per-head top-k indices and cosine
            similarity scores.

        Raises:
            RuntimeError: if :meth:`build` has not been called yet.
            ValueError: if ``q`` shape is inconsistent with the config.
        """
        if self._keys_norm is None:
            raise RuntimeError(
                "build() must be called before query()"
            )

        cfg = self._config
        if q.ndim != 2:
            raise ValueError(
                f"q must be 2-D (n_heads, head_dim); got shape {q.shape}"
            )
        if q.shape[0] != cfg.n_heads:
            raise ValueError(
                f"q.shape[0]={q.shape[0]} does not match "
                f"config n_heads={cfg.n_heads}"
            )
        if q.shape[1] != cfg.head_dim:
            raise ValueError(
                f"q.shape[1]={q.shape[1]} does not match "
                f"config head_dim={cfg.head_dim}"
            )

        n_heads = cfg.n_heads
        effective_k = min(cfg.top_k, self._seq_len)

        # L2-normalise query vectors for cosine similarity.
        q_norms = np.linalg.norm(q, axis=-1, keepdims=True)
        q_norms = np.where(q_norms < 1e-30, 1.0, q_norms)
        q_norm = (q / q_norms).astype(np.float32)

        # Cosine similarity: (n_heads, seq_len) via batched dot product.
        # keys_norm: (n_heads, seq_len, head_dim)
        # q_norm:    (n_heads, head_dim) -> (n_heads, head_dim, 1)
        scores = np.einsum("hsd,hd->hs", self._keys_norm, q_norm)  # (n_heads, seq_len)

        # Retrieve top-k per head using argpartition for efficiency.
        indices_out = np.empty((n_heads, effective_k), dtype=np.int32)
        scores_out = np.empty((n_heads, effective_k), dtype=np.float32)

        for h in range(n_heads):
            s = scores[h]
            if effective_k >= self._seq_len:
                top_idx = np.argsort(s)[::-1]
            else:
                part = np.argpartition(s, -effective_k)[-effective_k:]
                top_idx = part[np.argsort(s[part])[::-1]]
            indices_out[h] = top_idx[:effective_k].astype(np.int32)
            scores_out[h] = s[top_idx[:effective_k]].astype(np.float32)

        # If effective_k < top_k (seq shorter than top_k), pad with -1 / 0.
        if effective_k < cfg.top_k:
            pad_idx = np.full(
                (n_heads, cfg.top_k - effective_k), -1, dtype=np.int32
            )
            pad_scores = np.zeros(
                (n_heads, cfg.top_k - effective_k), dtype=np.float32
            )
            indices_out = np.concatenate([indices_out, pad_idx], axis=1)
            scores_out = np.concatenate([scores_out, pad_scores], axis=1)

        self._stats.n_queries += 1
        return ANCandidates(indices=indices_out, scores=scores_out)

    @property
    def n_indexed(self) -> int:
        """Number of key positions currently stored in the index."""
        return self._seq_len

    @property
    def stats(self) -> IndexStats:
        """Cumulative build and query statistics."""
        return self._stats

    def reset(self) -> None:
        """Clear the index without resetting cumulative statistics."""
        self._keys_norm = None
        self._seq_len = 0
