"""squish/kernels/rs_pq_cache_fit.py — Rust-backed PQ sub-codebook centroid fitting.

Wraps ``squish_quant_rs.pq_cache_fit_f32`` with a NumPy fallback.

Product Quantisation (PQ) compresses KV-cache vectors by splitting each head
dimension into sub-vectors and learning a small codebook for each sub-space.
This module fits the sub-codebook centroids via Lloyd's algorithm and provides
an ``encode`` helper that assigns sub-vectors to their nearest centroid.

Reference: Jégou et al., "Product Quantization for Nearest Neighbor Search,"
IEEE TPAMI 2011.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

__all__ = [
    "PQCacheFitConfig",
    "RustPQCacheFit",
]

try:
    import squish_quant as _sq
    _HAS_RUST = hasattr(_sq, "pq_cache_fit_f32")
except ImportError:
    _sq = None  # type: ignore[assignment]
    _HAS_RUST = False


# ── NumPy fallback ────────────────────────────────────────────────────────────


def _numpy_fit(
    sub_vecs: np.ndarray,
    k: int,
    n_iters: int,
    seed: int,
) -> np.ndarray:
    """Lloyd clustering for PQ centroid fitting.

    Args:
        sub_vecs: ``(N, sub_dim)`` float32 sub-vectors.
        k:        Number of centroids.
        n_iters:  Lloyd iterations.
        seed:     Random seed.

    Returns:
        ``(k, sub_dim)`` float32 centroids.
    """
    rng = np.random.default_rng(seed)
    n, sub_dim = sub_vecs.shape
    k = min(k, n)
    # Evenly-spaced init
    idx = np.linspace(0, n - 1, k, dtype=int)
    centroids = sub_vecs[idx].copy().astype(np.float32)
    for _ in range(n_iters):
        diffs = sub_vecs[:, None, :] - centroids[None, :, :]
        assignments = (diffs ** 2).sum(axis=-1).argmin(axis=1)
        for c in range(k):
            mask = assignments == c
            centroids[c] = sub_vecs[mask].mean(axis=0) if mask.any() else sub_vecs[rng.integers(n)]
    return centroids.astype(np.float32)


# ── Config / wrapper ──────────────────────────────────────────────────────────


@dataclass
class PQCacheFitConfig:
    """Configuration for :class:`RustPQCacheFit`.

    Attributes:
        n_iters: Lloyd iterations.
        seed:    Random seed.
    """

    n_iters: int = 50
    seed: int = 42


class RustPQCacheFit:
    """Rust-accelerated PQ sub-codebook centroid fitting.

    Fits a ``(K, sub_dim)`` codebook via Lloyd's algorithm, then provides
    nearest-centroid encoding of sub-vectors.  Falls back to NumPy when
    ``squish_quant_rs`` is unavailable.

    Example::

        pq = RustPQCacheFit()
        centroids = pq.fit(sub_vecs, K=256)
        indices = pq.encode(sub_vecs, centroids)
    """

    def __init__(self, config: Optional[PQCacheFitConfig] = None) -> None:
        self._cfg = config or PQCacheFitConfig()

    def fit(
        self,
        sub_vecs: np.ndarray,
        k: int,
        n_iters: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """Fit a PQ sub-codebook from sub-vectors.

        Args:
            sub_vecs: ``(N, sub_dim)`` float32 sub-vectors.
            k:        Number of centroids.
            n_iters:  Lloyd iterations (overrides config).
            seed:     Random seed (overrides config).

        Returns:
            ``(k, sub_dim)`` float32 centroids.

        Raises:
            ValueError: If ``sub_vecs`` is not 2-D or ``k < 1``.
        """
        sv = np.ascontiguousarray(sub_vecs, dtype=np.float32)
        if sv.ndim != 2:
            raise ValueError(f"sub_vecs must be 2-D (N, sub_dim), got {sv.shape}")
        if k < 1:
            raise ValueError(f"k must be ≥ 1, got {k}")
        ni = int(n_iters) if n_iters is not None else self._cfg.n_iters
        sd = int(seed) if seed is not None else self._cfg.seed
        if _HAS_RUST:
            return np.asarray(_sq.pq_cache_fit_f32(sv, min(k, len(sv)), ni, sd), dtype=np.float32)
        return _numpy_fit(sv, k, ni, sd)

    def encode(
        self,
        sub_vecs: np.ndarray,
        centroids: np.ndarray,
    ) -> np.ndarray:
        """Assign sub-vectors to their nearest centroid.

        Args:
            sub_vecs:  ``(N, sub_dim)`` float32 sub-vectors.
            centroids: ``(K, sub_dim)`` float32 centroids from :meth:`fit`.

        Returns:
            ``(N,)`` int32 centroid assignment indices.

        Raises:
            ValueError: If shapes are incompatible.
        """
        sv = np.asarray(sub_vecs, dtype=np.float32)
        cb = np.asarray(centroids, dtype=np.float32)
        if sv.ndim != 2 or cb.ndim != 2:
            raise ValueError(f"sub_vecs and centroids must be 2-D; got {sv.shape}, {cb.shape}")
        if sv.shape[1] != cb.shape[1]:
            raise ValueError(f"sub_dim mismatch: sub_vecs {sv.shape[1]}, centroids {cb.shape[1]}")
        diffs = sv[:, None, :] - cb[None, :, :]  # (N, K, sub_dim)
        return (diffs ** 2).sum(axis=-1).argmin(axis=1).astype(np.int32)

    def decode(
        self,
        indices: np.ndarray,
        centroids: np.ndarray,
    ) -> np.ndarray:
        """Reconstruct sub-vectors from centroid indices.

        Args:
            indices:   ``(N,)`` int32 centroid indices.
            centroids: ``(K, sub_dim)`` float32 centroids.

        Returns:
            ``(N, sub_dim)`` float32 reconstructed sub-vectors.
        """
        return np.asarray(centroids, dtype=np.float32)[np.asarray(indices)]

    def backend(self) -> str:
        return "rust" if _HAS_RUST else "numpy"
