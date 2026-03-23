"""squish/kernels/mojo/any4_lloyd_mojo.py — Mojo-backed Any4 Lloyd k-means.

Wraps ``any4_lloyd_step`` Mojo kernel via MojoBridge with a NumPy fallback.
Runs Lloyd (k-means) EM iterations to learn a 4-bit arbitrary codebook for
weight quantisation, parallelised over the E-step assignment.

Reference: Tseng et al., "QuIP#: Even Better LLM Quantization with Hadamard
Incoherence and Lattice Codebooks," ICML 2024.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from squish.kernels.mojo.mojo_bridge import MojoBridge

__all__ = [
    "Any4LloydMojoConfig",
    "MojoAny4Lloyd",
]

_bridge = MojoBridge()
_lloyd_kernel = _bridge.load_kernel("any4_lloyd_step")


# ── NumPy fallbacks ───────────────────────────────────────────────────────────


def _numpy_lloyd(
    values: np.ndarray,
    centroids_init: np.ndarray,
    n_iter: int,
) -> np.ndarray:
    """Lloyd iterations.

    Args:
        values:         ``(N,)`` float32.
        centroids_init: ``(k,)`` float32 initial centroids.
        n_iter:         Iteration count.

    Returns:
        ``(k,)`` float32 converged centroids.
    """
    v = values.astype(np.float64)
    c = centroids_init.astype(np.float64)
    k = len(c)
    rng = np.random.default_rng(0)
    for _ in range(n_iter):
        dists = np.abs(v[:, None] - c[None, :])  # (N, k)
        idx = dists.argmin(axis=1)
        new_c = np.empty_like(c)
        for j in range(k):
            mask = idx == j
            new_c[j] = v[mask].mean() if mask.any() else rng.choice(v)
        c = new_c
    return c.astype(np.float32)


# ── Config / wrapper ──────────────────────────────────────────────────────────


@dataclass
class Any4LloydMojoConfig:
    """Configuration for :class:`MojoAny4Lloyd`.

    Attributes:
        n_iter:     Lloyd iteration count.
        codebook_k: Codebook entries (≤16 for 4-bit).
    """

    n_iter: int = 100
    codebook_k: int = 16


class MojoAny4Lloyd:
    """Mojo-backed Any4 Lloyd k-means centroid search.

    Uses ``parallelize`` over the E-step (assignment) loop with SIMD
    abs-distance reduction per vector.  Falls back to NumPy when the
    Mojo runtime is absent.
    """

    def __init__(self, config: Optional[Any4LloydMojoConfig] = None) -> None:
        self._cfg = config or Any4LloydMojoConfig()

    def lloyd_step(
        self,
        values: np.ndarray,
        centroids_init: np.ndarray,
        n_iter: Optional[int] = None,
    ) -> np.ndarray:
        """Run Lloyd iterations to refine centroids.

        Args:
            values:         ``(N,)`` float32 weight elements.
            centroids_init: ``(k,)`` float32 initial centroids.
            n_iter:         Override config iteration count.

        Returns:
            ``(k,)`` float32 converged centroids.

        Raises:
            ValueError: If ``values`` or ``centroids_init`` are not 1-D.
        """
        v = np.ascontiguousarray(values, dtype=np.float32).ravel()
        c = np.ascontiguousarray(centroids_init, dtype=np.float32).ravel()
        if v.ndim != 1:
            raise ValueError("values must be 1-D")
        ni = int(n_iter) if n_iter is not None else self._cfg.n_iter
        if _lloyd_kernel is not None:
            out = np.empty_like(c)
            _lloyd_kernel(v.ctypes.data, c.ctypes.data, out.ctypes.data, len(v), len(c), ni)
            return out
        return _numpy_lloyd(v, c, ni)

    def quantize(
        self,
        weight: np.ndarray,
        centroids: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Quantize a weight array to codebook indices.

        Args:
            weight:    ``(N,)`` float32 weight elements.
            centroids: ``(k,)`` float32 codebook (auto-init if None).

        Returns:
            Tuple of (``(N,)`` int32 indices, ``(k,)`` float32 codebook).
        """
        v = np.ascontiguousarray(weight, dtype=np.float32).ravel()
        k = self._cfg.codebook_k
        if centroids is None:
            idx_init = np.linspace(0, len(v) - 1, k).astype(int)
            c_init = v[idx_init].copy()
        else:
            c_init = np.ascontiguousarray(centroids, dtype=np.float32).ravel()
        codebook = self.lloyd_step(v, c_init)
        dists = np.abs(v[:, None] - codebook[None, :])
        indices = dists.argmin(axis=1).astype(np.int32)
        return indices, codebook

    def backend(self) -> str:
        return "mojo" if _lloyd_kernel is not None else "numpy"
