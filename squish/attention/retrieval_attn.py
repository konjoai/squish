"""
squish/attention/retrieval_attn.py

RetrievalAttention: HNSW-Indexed Approximate KV Retrieval.

Reference
---------
Chen et al. "RetrievalAttention: Accelerating Long-Context LLM Inference via
Vector Retrieval." arXiv 2409.10516, 2024.

Algorithm
---------
For long contexts (>= ``min_length``), each key vector is indexed in a
Hierarchical Navigable Small World (HNSW) graph.  At decode time, the current
query is used to approximate top-K nearest-neighbour keys via HNSW graph
traversal in O(log N), rather than the O(N) exhaustive scan.

Implementation note
-------------------
A full C++ HNSW (e.g. ``hnswlib``) is the production path.  This module
implements a *pure NumPy flat-index fallback* that exposes the same API and
passes the same tests.  If ``hnswlib`` is installed it will be used instead.
The ``_backend`` attribute reflects which path is active.

Key properties
--------------
* ``n_neighbors`` — number of approximate top-K neighbours to retrieve.
* ``ef_construction`` — HNSW build parameter (quality vs build time).
* ``ef_search`` — HNSW search parameter (quality vs query time).
* ``min_length`` — contexts shorter than this fall back to exact attention.
* NumPy fallback always available; no hard dependency.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class RetrievalAttnConfig:
    """Configuration for RetrievalAttention."""

    n_neighbors: int = 64
    """Number of approximate nearest-neighbour KV positions to retrieve."""

    ef_construction: int = 200
    """HNSW build parameter (ignored in NumPy fallback)."""

    ef_search: int = 50
    """HNSW search parameter (ignored in NumPy fallback)."""

    head_dim: int = 64
    """Attention head dimension."""

    min_length: int = 256
    """Contexts shorter than this use exact attention regardless of backend."""

    def __post_init__(self) -> None:
        if self.n_neighbors < 1:
            raise ValueError("n_neighbors must be >= 1")
        if self.ef_construction < 1:
            raise ValueError("ef_construction must be >= 1")
        if self.ef_search < 1:
            raise ValueError("ef_search must be >= 1")


@dataclass
class RetrievalAttnStats:
    """Runtime counters for RetrievalAttention."""

    attn_calls: int = 0
    approx_calls: int = 0
    exact_calls: int = 0
    index_builds: int = 0


class RetrievalAttention:
    """HNSW-indexed approximate attention (NumPy fallback always available).

    Usage
    -----
    ::

        ra = RetrievalAttention()
        # Build or rebuild the index from the current KV cache
        ra.build_index(keys)
        # Attend
        output = ra.attend(query, keys, values)
    """

    def __init__(self, config: Optional[RetrievalAttnConfig] = None) -> None:
        self.config = config or RetrievalAttnConfig()
        self.stats = RetrievalAttnStats()
        self._index_keys: Optional[np.ndarray] = None  # flat NumPy index
        self._backend: str = "numpy"

        # Try to load hnswlib
        try:
            import hnswlib  # type: ignore
            self._hnswlib = hnswlib
            self._hnsw_index = None
            self._backend = "hnswlib"
        except ImportError:
            self._hnswlib = None

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def backend(self) -> str:
        """Return the active index backend ("numpy" or "hnswlib")."""
        return self._backend

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------

    def build_index(self, keys: np.ndarray) -> None:
        """Build or rebuild the KV index.

        Parameters
        ----------
        keys:
            Shape ``(seq_len, head_dim)``.
        """
        self.stats.index_builds += 1
        cfg = self.config
        if self._hnswlib is not None:
            seq_len, hd = keys.shape
            idx = self._hnswlib.Index(space="ip", dim=hd)
            idx.init_index(
                max_elements=seq_len + 1,
                ef_construction=cfg.ef_construction,
                M=16,
            )
            idx.set_ef(cfg.ef_search)
            idx.add_items(keys.astype(np.float32), np.arange(seq_len))
            self._hnsw_index = idx
            self._index_keys = None
        else:
            self._index_keys = keys.astype(np.float32)
            self._hnsw_index = None

    def _approx_top_k(self, query: np.ndarray, k: int) -> np.ndarray:
        """Return top-k position indices (approximate)."""
        cfg = self.config
        if self._hnsw_index is not None:
            labels, _ = self._hnsw_index.knn_query(
                query.astype(np.float32).reshape(1, -1), k=k
            )
            return labels[0]
        # NumPy flat-index fallback: exact inner product search
        assert self._index_keys is not None
        scores = self._index_keys @ query  # (seq_len,)
        actual_k = min(k, len(scores))
        top_idx = np.argpartition(scores, -actual_k)[-actual_k:]
        return top_idx

    # ------------------------------------------------------------------
    # Attention
    # ------------------------------------------------------------------

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        x = x - x.max()
        e = np.exp(x)
        return e / (e.sum() + 1e-9)

    def attend(
        self,
        query: np.ndarray,
        keys: np.ndarray,
        values: np.ndarray,
    ) -> np.ndarray:
        """Approximate (or exact) attention.

        Parameters
        ----------
        query:
            Shape ``(head_dim,)``.
        keys:
            Shape ``(seq_len, head_dim)``.
        values:
            Shape ``(seq_len, head_dim)``.

        Returns
        -------
        output:
            Shape ``(head_dim,)``.
        """
        self.stats.attn_calls += 1
        seq_len = keys.shape[0]
        scale = 1.0 / math.sqrt(self.config.head_dim)

        if seq_len <= self.config.min_length:
            self.stats.exact_calls += 1
            weights = self._softmax((keys @ query) * scale)
            return weights @ values

        self.stats.approx_calls += 1
        # Rebuild the index if the key matrix changed
        if self._index_keys is None or self._index_keys.shape[0] != seq_len:
            self.build_index(keys)

        k = min(self.config.n_neighbors, seq_len)
        top_idx = self._approx_top_k(query, k)
        k_sel = keys[top_idx]
        v_sel = values[top_idx]
        weights = self._softmax((k_sel @ query) * scale)
        return weights @ v_sel

    def reset_stats(self) -> None:
        """Reset runtime counters."""
        self.stats = RetrievalAttnStats()
