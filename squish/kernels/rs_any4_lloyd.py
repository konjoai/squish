"""squish/kernels/rs_any4_lloyd.py — Rust-backed Any4 Lloyd k-means codebook calibration.

Wraps ``squish_quant_rs.any4_lloyd_step_f32`` with a NumPy fallback.

Any4 learns a per-tensor floating-point codebook of 2^bits=16 centroids via
iterative Lloyd's algorithm on calibration weight samples.  The inner E-step
(distance computation + argmin) is parallelised over value chunks; the M-step
(per-centroid mean) uses parallel scatter-reduce.

Reference: Liao et al., "Any-Precision LLM: Low-Cost Deployment of Multiple,
Different-Sized LLMs," arXiv 2402.10517, 2024.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

__all__ = [
    "Any4LloydConfig",
    "RustAny4Lloyd",
]

try:
    import squish_quant as _sq
    _HAS_RUST = hasattr(_sq, "any4_lloyd_step_f32")
except ImportError:
    _sq = None  # type: ignore[assignment]
    _HAS_RUST = False


# ── NumPy fallback ────────────────────────────────────────────────────────────


def _numpy_lloyd(
    values: np.ndarray,
    centroids: np.ndarray,
    n_iter: int,
) -> np.ndarray:
    """Lloyd k-means codebook update.

    Args:
        values:    ``(N,)`` float32 weight values.
        centroids: ``(k,)`` float32 initial centroids.
        n_iter:    Number of iterations.

    Returns:
        ``(k,)`` float32 final centroids.
    """
    c = centroids.copy()
    k = len(c)
    for _ in range(n_iter):
        dists = np.abs(values[:, np.newaxis] - c[np.newaxis, :])  # (N, k)
        asgn = np.argmin(dists, axis=1)
        for ci in range(k):
            mask = asgn == ci
            if mask.any():
                c[ci] = values[mask].mean()
    return c


# ── Config / wrapper ──────────────────────────────────────────────────────────


@dataclass
class Any4LloydConfig:
    """Configuration for :class:`RustAny4Lloyd`.

    Attributes:
        n_iter:      Lloyd iterations.
        codebook_k:  Number of centroids (2^bits for Any4).
    """

    n_iter: int = 100
    codebook_k: int = 16


class RustAny4Lloyd:
    """Rust-accelerated Any4 Lloyd k-means codebook calibration.

    Runs ``n_iter`` rounds of Lloyd's algorithm on a flat array of weight
    values to produce an optimised ``k``-entry floating-point codebook.
    The E-step is parallelised over value chunks via Rayon.
    Falls back to NumPy when ``squish_quant_rs`` is unavailable.
    """

    def __init__(self, config: Optional[Any4LloydConfig] = None) -> None:
        self._cfg = config or Any4LloydConfig()

    def lloyd_step(
        self,
        values: np.ndarray,
        centroids_init: Optional[np.ndarray] = None,
        n_iter: Optional[int] = None,
    ) -> np.ndarray:
        """Run Lloyd k-means to convergence.

        Args:
            values:         ``(N,)`` float32 weight values.
            centroids_init: ``(k,)`` float32 initial centroids.
                            If ``None``, evenly-spaced over ``[min, max]``.
            n_iter:         Iterations (overrides config).

        Returns:
            ``(k,)`` float32 optimised centroids.

        Raises:
            ValueError: If ``values`` is not 1-D.
        """
        v = np.ascontiguousarray(values, dtype=np.float32).ravel()
        if v.ndim != 1:
            raise ValueError(f"values must be 1-D, got shape {v.shape}")
        k = self._cfg.codebook_k
        if centroids_init is None:
            lo, hi = float(v.min()), float(v.max())
            c = np.linspace(lo, hi, k, dtype=np.float32)
        else:
            c = np.ascontiguousarray(centroids_init, dtype=np.float32)
            k = len(c)
        n_it = int(n_iter) if n_iter is not None else self._cfg.n_iter
        if _HAS_RUST:
            return np.asarray(_sq.any4_lloyd_step_f32(v, c, n_it), dtype=np.float32)
        return _numpy_lloyd(v, c, n_it)

    def quantize(
        self,
        weight: np.ndarray,
        centroids: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Quantize a weight matrix using the Any4 codebook.

        Flattens weight, fits (or reuses) codebook, then assigns
        each value to its nearest centroid.

        Args:
            weight:    ``(out_f, in_f)`` float32.
            centroids: Pre-fitted ``(k,)`` codebook; if ``None`` it is fitted
                       from ``weight``.

        Returns:
            ``(indices, codebook)`` where ``indices`` is ``(N,)`` int32 and
            ``codebook`` is ``(k,)`` float32.
        """
        flat = weight.ravel().astype(np.float32)
        if centroids is None:
            codebook = self.lloyd_step(flat)
        else:
            codebook = np.asarray(centroids, dtype=np.float32)
        dists = np.abs(flat[:, np.newaxis] - codebook[np.newaxis, :])
        indices = np.argmin(dists, axis=1).astype(np.int32)
        return indices, codebook

    def backend(self) -> str:
        return "rust" if _HAS_RUST else "numpy"
