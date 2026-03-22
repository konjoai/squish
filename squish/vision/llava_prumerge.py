"""LLaVAPruMerge: adaptive spatial clustering and merging of visual patch tokens.

Shang et al. (CVPR 2024, arXiv 2403.15388) cluster patch tokens by spatial
position combined with key-similarity and merge each cluster into a single
weighted-average token before the tokens enter the LLM decoder.  This differs
from pruning (which discards tokens) — all spatial information is preserved in
the merged representation, giving LLaVAPruMerge a quality advantage on
spatially demanding tasks such as document-OCR and chart QA.

The NumPy implementation uses an exact K-means with a fixed number of
iterations; production deployments replace this with a Metal-accelerated
approximate K-means.

Reference: Shang et al., "LLaVA-PruMerge: Adaptive Token Reduction for
Efficient Large Multimodal Models", CVPR 2024.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

__all__ = [
    "LLaVAPruMergeConfig",
    "LLaVAPruMergeResult",
    "LLaVAPruMerge",
]


@dataclass
class LLaVAPruMergeConfig:
    """Configuration for :class:`LLaVAPruMerge`.

    Attributes:
        n_clusters: Target number of merged tokens.  When ``adaptive=True``
            this is the *maximum*; fewer clusters may be used for low-entropy
            images.
        adaptive: If True, reduce cluster count for images whose spatial
            key-entropy is below ``entropy_threshold``.
        entropy_threshold: Key-entropy threshold below which cluster count is
            halved (used only when ``adaptive=True``).
        position_weight: How strongly spatial position influences cluster
            assignment vs key-vector similarity.
        km_iters: Number of K-means iterations.
        seed: RNG seed for K-means initialisation.
    """

    n_clusters: int = 64
    adaptive: bool = True
    entropy_threshold: float = 2.0
    position_weight: float = 0.5
    km_iters: int = 10
    seed: int = 0

    def __post_init__(self) -> None:
        if self.n_clusters < 1:
            raise ValueError(f"n_clusters must be ≥ 1, got {self.n_clusters}")
        if not (0.0 <= self.position_weight <= 1.0):
            raise ValueError(
                f"position_weight must be in [0, 1], got {self.position_weight}"
            )
        if self.km_iters < 1:
            raise ValueError(f"km_iters must be ≥ 1, got {self.km_iters}")


@dataclass
class LLaVAPruMergeResult:
    """Output of one :meth:`LLaVAPruMerge.merge` call.

    Attributes:
        merged_tokens: The merged token matrix of shape ``(n_out, hidden_dim)``.
        cluster_labels: Per-input-token cluster assignment.
        n_clusters_used: Actual number of clusters (may be < ``n_clusters``
            when ``adaptive=True``).
    """

    merged_tokens: np.ndarray
    cluster_labels: np.ndarray
    n_clusters_used: int

    @property
    def compression_ratio(self) -> float:
        n_in = self.cluster_labels.size
        return self.n_clusters_used / n_in if n_in > 0 else 0.0


class LLaVAPruMerge:
    """Cluster visual patch tokens and replace each cluster with a weighted mean.

    Usage::

        cfg = LLaVAPruMergeConfig(n_clusters=64)
        prumerge = LLaVAPruMerge(cfg)
        # keys: (n_patches, key_dim)
        # positions: (n_patches, 2)  [row, col] normalised to [0, 1]
        result = prumerge.merge(keys, positions)
        # result.merged_tokens: (n_clusters, key_dim)
    """

    def __init__(self, config: LLaVAPruMergeConfig) -> None:
        self.config = config
        self._rng = np.random.default_rng(config.seed)

    # ------------------------------------------------------------------
    # Primary API
    # ------------------------------------------------------------------

    def merge(
        self,
        keys: np.ndarray,
        positions: Optional[np.ndarray] = None,
    ) -> LLaVAPruMergeResult:
        """Merge *keys* into cluster centroids.

        Parameters
        ----------
        keys:
            Visual patch key vectors of shape ``(n_patches, key_dim)``.
        positions:
            Optional spatial positions of shape ``(n_patches, 2)`` normalised
            to ``[0, 1]``.  When None, a grid layout is assumed.
        """
        keys = np.asarray(keys, dtype=np.float32)
        n, d = keys.shape

        if positions is None:
            side = int(np.ceil(np.sqrt(n)))
            idx = np.arange(n)
            rows = idx // side / max(side - 1, 1)
            cols = idx % side / max(side - 1, 1)
            positions = np.stack([rows, cols], axis=1).astype(np.float32)
        else:
            positions = np.asarray(positions, dtype=np.float32)

        # Determine effective cluster count
        k = self.config.n_clusters
        if self.config.adaptive:
            entropy = self._key_entropy(keys)
            if entropy < self.config.entropy_threshold:
                k = max(1, k // 2)
        k = min(k, n)

        # Build the feature matrix: normalised keys + weighted positions
        pw = self.config.position_weight
        keys_norm = self._l2_normalise(keys)
        pos_norm = positions  # already in [0,1]
        features = np.concatenate(
            [(1.0 - pw) * keys_norm, pw * pos_norm], axis=1
        )

        labels = self._kmeans(features, k)
        merged = self._weighted_average_centroids(keys, labels, k)

        return LLaVAPruMergeResult(
            merged_tokens=merged,
            cluster_labels=labels,
            n_clusters_used=k,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _kmeans(self, features: np.ndarray, k: int) -> np.ndarray:
        """Simple Lloyd K-means; returns cluster labels."""
        n = features.shape[0]
        # Initialise centroids by random selection
        init_idx = self._rng.choice(n, size=k, replace=False)
        centroids = features[init_idx].copy()

        labels = np.zeros(n, dtype=np.int32)
        for _ in range(self.config.km_iters):
            # Assignment step
            dists = np.linalg.norm(features[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2)
            labels = dists.argmin(axis=1)
            # Update step
            for c in range(k):
                mask = labels == c
                if mask.any():
                    centroids[c] = features[mask].mean(axis=0)
        return labels

    def _weighted_average_centroids(
        self, keys: np.ndarray, labels: np.ndarray, k: int
    ) -> np.ndarray:
        """Return per-cluster weighted-average key vector (uniform weights)."""
        d = keys.shape[1]
        merged = np.zeros((k, d), dtype=np.float32)
        for c in range(k):
            mask = labels == c
            if mask.any():
                merged[c] = keys[mask].mean(axis=0)
        return merged

    @staticmethod
    def _l2_normalise(x: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(x, axis=1, keepdims=True).clip(min=1e-8)
        return x / norms

    @staticmethod
    def _key_entropy(keys: np.ndarray) -> float:
        """Rough entropy proxy: mean absolute deviation of normalised keys."""
        norms = np.linalg.norm(keys, axis=1)
        if norms.std() < 1e-8:
            return 0.0
        normed = norms / (norms.mean() + 1e-8)
        return float(np.mean(np.abs(normed - 1.0)))
