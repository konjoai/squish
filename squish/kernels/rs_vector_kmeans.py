"""rs_vector_kmeans.py — Rust-accelerated vector K-means codebook fitting.

Wraps ``squish_quant.vector_kmeans_fit_f32``, ``vector_kmeans_assign_f32``,
and ``vector_kmeans_reconstruct_f32`` (Wave 58a).  Falls back to pure-NumPy
K-means++ when the Rust extension is unavailable.

RustVectorKMeans eliminates the ``O(N × K × D)`` broadcast-expansion
temporary that VPTQ, AQLM, and CodecKV all create inside their K-means
inner loops, replacing them with a Rayon parallel chunk-wise argmin +
scatter-add that avoids any ``(N, K, D)`` allocation.

Reference:
  Liu et al. (NeurIPS 2024) — VPTQ (arXiv:2409.17066).
  Tseng et al. (ICLR 2024) — AQLM (arXiv:2401.06118).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

try:
    import squish_quant as _sq

    _RUST_AVAILABLE = all(
        hasattr(_sq, fn)
        for fn in (
            "vector_kmeans_fit_f32",
            "vector_kmeans_assign_f32",
            "vector_kmeans_reconstruct_f32",
        )
    )
except ImportError:
    _RUST_AVAILABLE = False

__all__ = ["VectorKMeansConfig", "RustVectorKMeans"]


@dataclass
class VectorKMeansConfig:
    """Configuration for RustVectorKMeans.

    Attributes:
        n_clusters: Number of centroids (codebook size).
        n_iter:     Number of Lloyd EM iterations.
    """

    n_clusters: int = 256
    n_iter: int = 25


class RustVectorKMeans:
    """Rust-accelerated K-means codebook fitting for vector quantization.

    Covers the codebook-fitting inner loop shared by VPTQ, AQLM, and
    CodecKV — a single ``(N, K, D)``-broadcast-free Rayon implementation.

    Usage::

        km = RustVectorKMeans(VectorKMeansConfig(n_clusters=256, n_iter=20))
        data = np.random.randn(10000, 8).astype(np.float32)
        centroids = km.fit(data)            # (256, 8)
        codes     = km.assign(data, centroids)   # (10000,)
        recon     = km.reconstruct(codes, centroids)  # (10000, 8)
    """

    def __init__(self, config: VectorKMeansConfig | None = None) -> None:
        self._cfg = config or VectorKMeansConfig()

    def fit(self, data: np.ndarray, n_clusters: int | None = None, n_iter: int | None = None) -> np.ndarray:
        """Fit K-means on *data* and return centroids ``(K, D)`` float32.

        Args:
            data:       Float32 array ``(N, D)`` of input vectors.
            n_clusters: Override config n_clusters.
            n_iter:     Override config n_iter.

        Returns:
            Float32 array ``(K, D)`` of fitted centroids.
        """
        data = np.ascontiguousarray(data, dtype=np.float32)
        k = n_clusters if n_clusters is not None else self._cfg.n_clusters
        it = n_iter if n_iter is not None else self._cfg.n_iter
        if _RUST_AVAILABLE:
            return np.asarray(_sq.vector_kmeans_fit_f32(data, k, it), dtype=np.float32)
        return self._numpy_fit(data, k, it)

    def assign(self, data: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """Assign each vector to the nearest centroid, return ``(N,)`` int32.

        Args:
            data:      Float32 array ``(N, D)``.
            centroids: Float32 array ``(K, D)``.

        Returns:
            Int32 array ``(N,)`` of nearest centroid indices.
        """
        data = np.ascontiguousarray(data, dtype=np.float32)
        centroids = np.ascontiguousarray(centroids, dtype=np.float32)
        if _RUST_AVAILABLE:
            return np.asarray(_sq.vector_kmeans_assign_f32(data, centroids), dtype=np.int32)
        return self._numpy_assign(data, centroids)

    def reconstruct(self, indices: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """Reconstruct vectors from centroid indices, return ``(N, D)`` float32.

        Args:
            indices:   Int32 array ``(N,)`` of centroid indices.
            centroids: Float32 array ``(K, D)``.

        Returns:
            Float32 array ``(N, D)`` reconstructed vectors.
        """
        indices = np.ascontiguousarray(indices, dtype=np.int32)
        centroids = np.ascontiguousarray(centroids, dtype=np.float32)
        if _RUST_AVAILABLE:
            return np.asarray(_sq.vector_kmeans_reconstruct_f32(indices, centroids), dtype=np.float32)
        return self._numpy_reconstruct(indices, centroids)

    def n_clusters(self) -> int:
        """Return configured number of clusters."""
        return self._cfg.n_clusters

    def backend(self) -> str:
        """Return 'rust' if Rust extension available, else 'numpy'."""
        return "rust" if _RUST_AVAILABLE else "numpy"

    # ── NumPy fallbacks ────────────────────────────────────────────────────

    @staticmethod
    def _numpy_fit(data: np.ndarray, n_clusters: int, n_iter: int) -> np.ndarray:
        rng = np.random.default_rng(42)
        n, d = data.shape
        # K-means++ seeding
        idx = [0]
        for _ in range(1, n_clusters):
            dists = np.array([
                min(np.sum((data[i] - data[c]) ** 2) for c in idx)
                for i in range(n)
            ])
            idx.append(int(np.argmax(dists)))
        centroids = data[idx].copy()
        # Lloyd iterations
        for _ in range(n_iter):
            sq_dists = np.sum((data[:, None, :] - centroids[None, :, :]) ** 2, axis=-1)
            assignments = np.argmin(sq_dists, axis=1)
            for c in range(n_clusters):
                mask = assignments == c
                if mask.any():
                    centroids[c] = data[mask].mean(axis=0)
        return centroids.astype(np.float32)

    @staticmethod
    def _numpy_assign(data: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        sq_dists = np.sum((data[:, None, :] - centroids[None, :, :]) ** 2, axis=-1)
        return np.argmin(sq_dists, axis=1).astype(np.int32)

    @staticmethod
    def _numpy_reconstruct(indices: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        return centroids[np.clip(indices, 0, len(centroids) - 1)].astype(np.float32)
