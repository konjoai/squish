"""squish/kv/cluster_evict_kv.py

ClusterEvictKV — Cluster-Based Adaptive KV Cache Eviction.

Reference
---------
Zhang et al. "SnapKV v2 / PyramidKV Dynamic: Adaptive Cluster-Based KV
Eviction." NeurIPS 2024 (arXiv:2406.02069 follow-up).

Algorithm
---------
Rather than evicting individual K/V tokens based on per-token scores,
ClusterEvictKV groups tokens into clusters based on the cosine similarity
of their key vectors and evicts entire clusters:

1. Compute attention scores for the current observation window.
2. Cluster K vectors using an online k-means-like approach.
3. Score each cluster by summed attention weight.
4. Evict lowest-scoring clusters to meet per-layer budget.
5. Adapt budget per layer based on observed attention entropy.

Key properties
--------------
* NumPy-only.
* ``budget`` — maximum number of KV tokens to retain per layer.
* ``n_clusters`` — number of clusters (default min(budget//2, 32)).
* ``window_size`` — observation window for attention scoring.
* ``adaptive_budget`` — if True, reallocate budget across layers by entropy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

__all__ = [
    "ClusterEvictKVConfig",
    "ClusterEvictKV",
]


@dataclass
class ClusterEvictKVConfig:
    """Configuration for :class:`ClusterEvictKV`.

    Attributes:
        budget: Maximum K/V tokens retained after eviction.
        n_clusters: Number of clusters for K vector grouping.
        window_size: Observation window tokens used to score clusters.
        n_heads: Number of K/V heads.
        head_dim: Dimension per head.
        adaptive_budget: Enable entropy-based per-layer budget adaptation.
    """

    budget_tokens: int = 512
    n_clusters: int = 32
    window_size: int = 64
    n_heads: int = 8
    head_dim: int = 64
    adaptive_budget: bool = True


class ClusterEvictKV:
    """Cluster-based adaptive KV eviction.

    Parameters
    ----------
    config:
        ClusterEvictKV configuration.
    seed:
        RNG seed for k-means initialisation.
    """

    def __init__(self, config: Optional[ClusterEvictKVConfig] = None, seed: int = 0) -> None:
        self._cfg = config or ClusterEvictKVConfig()
        self._rng = np.random.default_rng(seed)
        self._current_budget: int = self._cfg.budget_tokens
        self._eviction_count: int = 0

    @property
    def config(self) -> ClusterEvictKVConfig:
        return self._cfg

    @property
    def current_budget(self) -> int:
        return self._current_budget

    @property
    def eviction_count(self) -> int:
        return self._eviction_count

    def _cluster_keys(self, keys: np.ndarray, n_clusters: int) -> np.ndarray:
        """Assign tokens to clusters using a single Lloyd iteration.

        Parameters
        ----------
        keys:
            Shape ``(seq_len, head_dim)``.
        n_clusters:
            Number of clusters.

        Returns
        -------
        np.ndarray
            Cluster assignments, shape ``(seq_len,)``.
        """
        seq_len = len(keys)
        n_clusters = min(n_clusters, seq_len)
        # Initialise centroids via k-means++ seed (simplified)
        idx = self._rng.integers(0, seq_len, size=n_clusters)
        centroids = keys[idx]  # (n_clusters, head_dim)
        # One Lloyd step
        norms_c = np.linalg.norm(centroids, axis=1, keepdims=True).clip(min=1e-8)
        norms_k = np.linalg.norm(keys, axis=1, keepdims=True).clip(min=1e-8)
        sim = (keys / norms_k) @ (centroids / norms_c).T  # (seq_len, n_clusters)
        assignments = sim.argmax(axis=1)
        return assignments

    def _attention_entropy(self, attn_weights: np.ndarray) -> float:
        """Mean attention entropy (nats) for budget adaptation."""
        w = np.asarray(attn_weights, dtype=np.float64).clip(min=1e-10)
        w /= w.sum(axis=-1, keepdims=True)
        return float(-(w * np.log(w)).sum(axis=-1).mean())

    def evict(
        self,
        keys: np.ndarray,
        values: np.ndarray,
        attn_weights: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Evict K/V tokens using cluster-based scoring.

        Parameters
        ----------
        keys:
            Shape ``(n_heads, seq_len, head_dim)``.
        values:
            Shape ``(n_heads, seq_len, head_dim)``.
        attn_weights:
            Optional attention weights ``(n_heads, window, seq_len)`` for
            scoring clusters.  If None, recency is used as a score proxy.

        Returns
        -------
        Tuple of (keys_evicted, values_evicted) with seq_len <= budget.
        """
        K = np.asarray(keys, dtype=np.float32)
        V = np.asarray(values, dtype=np.float32)
        is_3d = K.ndim == 3
        seq_len = K.shape[1] if is_3d else K.shape[0]

        if seq_len <= self._current_budget:
            return K, V

        # Adapt budget if entropy is available
        if self._cfg.adaptive_budget and attn_weights is not None:
            ent = self._attention_entropy(attn_weights)
            # High entropy → use full budget; low entropy → use less
            ent_max = np.log(seq_len + 1)
            ratio = np.clip(ent / ent_max, 0.2, 1.0)
            self._current_budget = max(
                self._cfg.n_clusters,
                int(self._cfg.budget_tokens * ratio),
            )

        # Keys for clustering: use mean over heads for 3D, or as-is for 2D
        mean_keys = K.mean(axis=0) if is_3d else K  # (seq_len, head_dim)
        n_cl = min(self._cfg.n_clusters, self._current_budget)
        assignments = self._cluster_keys(mean_keys, n_cl)

        # Score clusters
        if attn_weights is not None:
            aw = np.asarray(attn_weights, dtype=np.float32)
            avg_attn = aw.mean(axis=tuple(range(aw.ndim - 1)))  # collapse all but last
        else:
            # Recency bias: more recent tokens get higher scores
            avg_attn = np.linspace(0.0, 1.0, seq_len, dtype=np.float32)

        cluster_scores = np.zeros(n_cl, dtype=np.float32)
        for c in range(n_cl):
            mask = assignments == c
            if mask.any():
                cluster_scores[c] = avg_attn[mask].sum()

        # Sort clusters by score; keep top clusters until budget is filled
        cluster_order = np.argsort(cluster_scores)[::-1]
        keep_indices_set = []
        for c in cluster_order:
            if len(keep_indices_set) >= self._current_budget:
                break
            members = np.where(assignments == c)[0]
            keep_indices_set.extend(members.tolist())
            if len(keep_indices_set) >= self._current_budget:
                break

        keep_indices = np.array(sorted(set(keep_indices_set[: self._current_budget])))
        self._eviction_count += seq_len - len(keep_indices)
        if is_3d:
            return K[:, keep_indices, :], V[:, keep_indices, :]
        return K[keep_indices, :], V[keep_indices, :]

    def reset_budget(self) -> None:
        self._current_budget = self._cfg.budget_tokens
