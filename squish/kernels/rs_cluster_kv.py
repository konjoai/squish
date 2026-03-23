"""squish/kernels/rs_cluster_kv.py — Rust-backed ClusterKV attention-score aggregation.

Wraps ``squish_quant_rs.cluster_kv_score_f32`` with a NumPy fallback.

ClusterKV groups KV-cache tokens into clusters by attention-pattern similarity
and evicts whole clusters rather than individual tokens.  This module
accelerates the per-eviction-step scoring: for each cluster, sum the softmax
attention weights of all its member tokens — parallelised over clusters via
Rayon ``into_par_iter``.

Reference: Zhang et al., "ClusterKV: Manipulating LLM KV Cache in Semantic
Space for Recallable Compression," arXiv 2412.03213, 2024.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

__all__ = [
    "ClusterKVConfig",
    "RustClusterKV",
]

try:
    import squish_quant as _sq
    _HAS_RUST = hasattr(_sq, "cluster_kv_score_f32")
except ImportError:
    _sq = None  # type: ignore[assignment]
    _HAS_RUST = False


# ── NumPy fallback ────────────────────────────────────────────────────────────


def _numpy_cluster_score(
    assignments: np.ndarray,
    attn_weights: np.ndarray,
    n_clusters: int,
) -> np.ndarray:
    """Sum attention weights per cluster.

    Args:
        assignments:  ``(seq_len,)`` int32 cluster index per token.
        attn_weights: ``(seq_len,)`` float32 softmax attention weight.
        n_clusters:   Total number of clusters.

    Returns:
        ``(n_clusters,)`` float32 total attention weight per cluster.
    """
    scores = np.zeros(n_clusters, dtype=np.float32)
    for c in range(n_clusters):
        mask = assignments == c
        scores[c] = attn_weights[mask].sum()
    return scores


# ── Config / wrapper ──────────────────────────────────────────────────────────


@dataclass
class ClusterKVConfig:
    """Configuration for :class:`RustClusterKV`.

    Attributes:
        evict_ratio: Fraction of clusters to evict when budget is exceeded.
    """

    evict_ratio: float = 0.5


class RustClusterKV:
    """Rust-accelerated ClusterKV per-cluster attention-weight aggregation.

    For each cluster, computes the total softmax attention weight of all
    member tokens.  Low-scoring clusters are candidates for eviction.
    Parallelised over clusters via Rayon ``into_par_iter``.
    Falls back to NumPy when ``squish_quant_rs`` is unavailable.
    """

    def __init__(self, config: Optional[ClusterKVConfig] = None) -> None:
        self._cfg = config or ClusterKVConfig()

    def score_clusters(
        self,
        assignments: np.ndarray,
        attn_weights: np.ndarray,
        n_clusters: int,
    ) -> np.ndarray:
        """Compute total attention weight for each cluster.

        Args:
            assignments:  Token→cluster assignments ``(seq_len,)`` int32.
            attn_weights: Per-token softmax weights ``(seq_len,)`` float32.
            n_clusters:   Total number of clusters.

        Returns:
            ``(n_clusters,)`` float32 total weight per cluster.

        Raises:
            ValueError: If ``assignments`` and ``attn_weights`` lengths differ.
        """
        asgn = np.ascontiguousarray(assignments, dtype=np.int32)
        attn = np.ascontiguousarray(attn_weights, dtype=np.float32)
        if asgn.shape[0] != attn.shape[0]:
            raise ValueError(
                f"assignments len {asgn.shape[0]} != attn_weights len {attn.shape[0]}"
            )
        if _HAS_RUST:
            return np.asarray(
                _sq.cluster_kv_score_f32(asgn, attn, int(n_clusters)),
                dtype=np.float32,
            )
        return _numpy_cluster_score(asgn, attn, int(n_clusters))

    def evict_mask(
        self,
        assignments: np.ndarray,
        attn_weights: np.ndarray,
        n_clusters: int,
        evict_ratio: Optional[float] = None,
    ) -> np.ndarray:
        """Return a boolean token-level eviction mask.

        Tokens belonging to the lowest-scoring ``evict_ratio`` fraction of
        clusters are marked for eviction (True = evict).

        Args:
            assignments:  ``(S,)`` int32 cluster index per token.
            attn_weights: ``(S,)`` float32 attention weights.
            n_clusters:   Total number of clusters.
            evict_ratio:  Fraction of clusters to evict (overrides config).

        Returns:
            ``(S,)`` bool mask — True means evict this slot.
        """
        scores = self.score_clusters(assignments, attn_weights, n_clusters)
        ratio = float(evict_ratio) if evict_ratio is not None else self._cfg.evict_ratio
        n_evict = max(0, int(n_clusters * ratio))
        evict_ids = set(np.argsort(scores)[:n_evict].tolist())
        asgn = np.ascontiguousarray(assignments, dtype=np.int32).ravel()
        return np.array([int(a) in evict_ids for a in asgn], dtype=bool)

    def backend(self) -> str:
        return "rust" if _HAS_RUST else "numpy"
