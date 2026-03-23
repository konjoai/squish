"""squish/kernels/mojo/cluster_kv_mojo.py — Mojo-backed ClusterKV scoring.

Wraps ``cluster_kv_score`` Mojo kernel via MojoBridge with a NumPy fallback.
ClusterKV aggregates token-level attention weights per cluster to score which
clusters deserve to stay in the KV cache.

Reference: Wu et al., "ClusterKV: Manipulating LLM KV Cache in Semantic Space
for Recallable Compression," arXiv 2412.03213, 2024.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from squish.kernels.mojo.mojo_bridge import MojoBridge

__all__ = [
    "ClusterKVMojoConfig",
    "MojoClusterKV",
]

_bridge = MojoBridge()
_score_kernel = _bridge.load_kernel("cluster_kv_score")


# ── NumPy fallbacks ───────────────────────────────────────────────────────────


def _numpy_score(
    assignments: np.ndarray,
    attn_weights: np.ndarray,
    n_clusters: int,
) -> np.ndarray:
    """Sum attention weights per cluster.

    Args:
        assignments:  ``(S,)`` int32 cluster index per token.
        attn_weights: ``(S,)`` float32 query-to-token attention.
        n_clusters:   Number of clusters.

    Returns:
        ``(n_clusters,)`` float32 cluster scores.
    """
    scores = np.zeros(n_clusters, dtype=np.float32)
    np.add.at(scores, assignments, attn_weights)
    return scores


# ── Config / wrapper ──────────────────────────────────────────────────────────


@dataclass
class ClusterKVMojoConfig:
    """Configuration for :class:`MojoClusterKV`.

    Attributes:
        evict_ratio: Fraction of low-scoring clusters to evict.
    """

    evict_ratio: float = 0.5


class MojoClusterKV:
    """Mojo-backed ClusterKV attention-weight aggregation and eviction.

    Uses ``parallelize`` over cluster buckets for score accumulation.
    Falls back to NumPy when the Mojo runtime is absent.
    """

    def __init__(self, config: Optional[ClusterKVMojoConfig] = None) -> None:
        self._cfg = config or ClusterKVMojoConfig()

    def score_clusters(
        self,
        assignments: np.ndarray,
        attn_weights: np.ndarray,
        n_clusters: int,
    ) -> np.ndarray:
        """Aggregate per-token attention weights into cluster scores.

        Args:
            assignments:  ``(S,)`` int32 cluster index per KV slot.
            attn_weights: ``(S,)`` float32 attention weights.
            n_clusters:   Total number of clusters.

        Returns:
            ``(n_clusters,)`` float32 cluster importance scores.

        Raises:
            ValueError: If shapes are inconsistent.
        """
        a = np.ascontiguousarray(assignments, dtype=np.int32).ravel()
        w = np.ascontiguousarray(attn_weights, dtype=np.float32).ravel()
        if a.shape[0] != w.shape[0]:
            raise ValueError(
                f"assignments length {a.shape[0]} != attn_weights length {w.shape[0]}"
            )
        if _score_kernel is not None:
            out = np.zeros(n_clusters, dtype=np.float32)
            _score_kernel(a.ctypes.data, w.ctypes.data, out.ctypes.data, len(a), n_clusters)
            return out
        return _numpy_score(a, w, n_clusters)

    def evict_mask(
        self,
        assignments: np.ndarray,
        attn_weights: np.ndarray,
        n_clusters: int,
        evict_ratio: Optional[float] = None,
    ) -> np.ndarray:
        """Return boolean eviction mask for KV slots.

        Args:
            assignments:  ``(S,)`` int32 cluster index per slot.
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
        a = np.ascontiguousarray(assignments, dtype=np.int32).ravel()
        return np.array([int(x) in evict_ids for x in a], dtype=bool)

    def backend(self) -> str:
        return "mojo" if _score_kernel is not None else "numpy"
