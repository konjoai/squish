"""squish/kernels/rs_qmoe_compress.py — Rust-backed QMoE block compression.

Wraps ``squish_quant_rs.qmoe_compress_iter_f32`` with a NumPy fallback.

QMoE quantises Mixture-of-Experts weight matrices by treating each
``block_size``-element block as a vector and running k-means EM to learn a
shared codebook across all N blocks.  Assignment lookups allow near-exact
weight reconstruction from a small codebook + integer indices.

Reference: Frantar & Alistarh, "QMoE: Practical Sub-1-Bit Compression of
Trillion-Parameter Models," arXiv 2310.16795, 2023.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

__all__ = [
    "QMoECompressConfig",
    "RustQMoECompress",
]

try:
    import squish_quant as _sq
    _HAS_RUST = hasattr(_sq, "qmoe_compress_iter_f32")
except ImportError:
    _sq = None  # type: ignore[assignment]
    _HAS_RUST = False


# ── NumPy fallback ────────────────────────────────────────────────────────────


def _numpy_compress(
    blocks: np.ndarray,
    k: int,
    n_iter: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Lloyd k-means on ``blocks``.

    Args:
        blocks:  ``(N, block_size)`` float32.
        k:       Codebook size.
        n_iter:  EM iterations.
        seed:    Random seed.

    Returns:
        Tuple of codebook ``(k, block_size)`` float32,
        assignments ``(N,)`` int32.
    """
    rng = np.random.default_rng(seed)
    N, bs = blocks.shape
    k = min(k, N)
    idx = rng.choice(N, size=k, replace=False)
    centroids = blocks[idx].copy()
    assignments = np.zeros(N, dtype=np.int32)
    for _ in range(n_iter):
        # E-step
        diffs = blocks[:, None, :] - centroids[None, :, :]  # (N, k, bs)
        dists = (diffs ** 2).sum(axis=-1)  # (N, k)
        assignments = dists.argmin(axis=1).astype(np.int32)
        # M-step
        new_centroids = np.zeros_like(centroids)
        counts = np.zeros(k, dtype=np.int64)
        for c in range(k):
            mask = assignments == c
            if mask.any():
                new_centroids[c] = blocks[mask].mean(axis=0)
                counts[c] = mask.sum()
            else:
                new_centroids[c] = blocks[rng.integers(N)]
        centroids = new_centroids
    return centroids.astype(np.float32), assignments


# ── Config / wrapper ──────────────────────────────────────────────────────────


@dataclass
class QMoECompressConfig:
    """Configuration for :class:`RustQMoECompress`.

    Attributes:
        n_iter: EM iterations.
        seed:   Random seed for centroid initialisation.
    """

    n_iter: int = 50
    seed: int = 42


class RustQMoECompress:
    """Rust-accelerated QMoE per-expert block compression.

    Runs Rayon-parallelised Lloyd k-means on ``(N, block_size)`` blocks to
    learn a ``k``-entry codebook, then assigns each block to its nearest
    centroid.  Falls back to NumPy when ``squish_quant_rs`` is unavailable.

    Example::

        comp = RustQMoECompress()
        codebook, assignments = comp.compress(blocks=w, k=256)
        reconstructed = codebook[assignments]
    """

    def __init__(self, config: Optional[QMoECompressConfig] = None) -> None:
        self._cfg = config or QMoECompressConfig()

    def compress(
        self,
        blocks: np.ndarray,
        k: int,
        n_iter: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compress blocks into a shared codebook.

        Args:
            blocks:  ``(N, block_size)`` float32 expert weight blocks.
            k:       Codebook size (entries).
            n_iter:  EM iterations (overrides config).
            seed:    Random seed (overrides config).

        Returns:
            Tuple of:
            - ``codebook``: ``(k, block_size)`` float32 centroids.
            - ``assignments``: ``(N,)`` int32 indices into codebook.

        Raises:
            ValueError: If ``blocks`` is not 2-D or ``k < 1``.
        """
        b = np.ascontiguousarray(blocks, dtype=np.float32)
        if b.ndim != 2:
            raise ValueError(f"blocks must be 2-D (N, block_size), got {b.shape}")
        if k < 1:
            raise ValueError(f"k must be ≥ 1, got {k}")
        k = min(k, len(b))  # clamp to number of blocks
        ni = int(n_iter) if n_iter is not None else self._cfg.n_iter
        sd = int(seed) if seed is not None else self._cfg.seed
        if _HAS_RUST:
            cb, asgn = _sq.qmoe_compress_iter_f32(b, k, ni, sd)
            return np.asarray(cb, dtype=np.float32), np.asarray(asgn, dtype=np.int32)
        return _numpy_compress(b, k, ni, sd)

    def reconstruct(
        self,
        assignments: np.ndarray,
        codebook: np.ndarray,
    ) -> np.ndarray:
        """Reconstruct blocks from assignments and codebook.

        Args:
            assignments: ``(N,)`` int32 codebook indices.
            codebook:    ``(k, block_size)`` float32 centroids.

        Returns:
            ``(N, block_size)`` float32 reconstructed weight blocks.
        """
        return codebook[assignments]

    def backend(self) -> str:
        return "rust" if _HAS_RUST else "numpy"
