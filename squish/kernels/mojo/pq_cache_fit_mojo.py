"""squish/kernels/mojo/pq_cache_fit_mojo.py — Mojo-backed PQ sub-codebook centroid fitting.

Wraps the ``pq_cache_fit_kernel`` Mojo stub via MojoBridge with a NumPy fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from squish.kernels.mojo.mojo_bridge import MojoBridge

__all__ = [
    "PQCacheFitMojoConfig",
    "MojoPQCacheFit",
]

_bridge = MojoBridge()
_fit_kernel = _bridge.load_kernel("pq_cache_fit_kernel")


# ── NumPy fallback ────────────────────────────────────────────────────────────


def _numpy_fit(sub_vecs: np.ndarray, k: int, n_iters: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n, sub_dim = sub_vecs.shape
    k = min(k, n)
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
class PQCacheFitMojoConfig:
    """Configuration for :class:`MojoPQCacheFit`.

    Attributes:
        n_iters: Lloyd iterations.
        seed:    Random seed.
    """

    n_iters: int = 50
    seed: int = 42


class MojoPQCacheFit:
    """Mojo-backed PQ sub-codebook centroid fitting.

    Falls back to NumPy when the Mojo runtime is absent.

    Example::

        pq = MojoPQCacheFit()
        centroids = pq.fit(sub_vecs, K=256)
        indices = pq.encode(sub_vecs, centroids)
    """

    def __init__(self, config: Optional[PQCacheFitMojoConfig] = None) -> None:
        self._cfg = config or PQCacheFitMojoConfig()

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
        if _fit_kernel is not None:
            n, sub_dim = sv.shape
            out = np.zeros((min(k, n), sub_dim), dtype=np.float32)
            _fit_kernel(sv.ctypes.data, out.ctypes.data, n, sub_dim, min(k, n), ni, sd)
            return out
        return _numpy_fit(sv, k, ni, sd)

    def encode(self, sub_vecs: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """Assign sub-vectors to their nearest centroid.

        Args:
            sub_vecs:  ``(N, sub_dim)`` float32 sub-vectors.
            centroids: ``(K, sub_dim)`` float32 centroids.

        Returns:
            ``(N,)`` int32 centroid indices.
        """
        sv = np.asarray(sub_vecs, dtype=np.float32)
        cb = np.asarray(centroids, dtype=np.float32)
        diffs = sv[:, None, :] - cb[None, :, :]
        return (diffs ** 2).sum(axis=-1).argmin(axis=1).astype(np.int32)

    def decode(self, indices: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """Reconstruct sub-vectors from centroid indices.

        Args:
            indices:   ``(N,)`` int32 centroid indices.
            centroids: ``(K, sub_dim)`` float32 centroids.

        Returns:
            ``(N, sub_dim)`` float32.
        """
        return np.asarray(centroids, dtype=np.float32)[np.asarray(indices)]

    def backend(self) -> str:
        return "mojo" if _fit_kernel is not None else "numpy"
