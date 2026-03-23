"""squish/kernels/mojo/shadow_kv_fit_mojo.py — Mojo-backed ShadowKV SVD fit.

Wraps ``shadow_kv_svd_fit`` and ``shadow_kv_store_batch`` Mojo kernels via
MojoBridge with NumPy fallbacks.  ShadowKV compresses the KV cache by storing
low-rank projections of key matrices.

Reference: Sun et al., "ShadowKV: KV Cache in Shadows for High-Throughput
Long-Context LLM Inference," arXiv 2410.21465, 2024.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from squish.kernels.mojo.mojo_bridge import MojoBridge

__all__ = [
    "ShadowKVFitMojoConfig",
    "MojoShadowKVFit",
]

_bridge = MojoBridge()
_fit_kernel = _bridge.load_kernel("shadow_kv_svd_fit")
_store_kernel = _bridge.load_kernel("shadow_kv_store_batch")


# ── NumPy fallbacks ───────────────────────────────────────────────────────────


def _numpy_fit_svd(keys: np.ndarray, rank: int) -> np.ndarray:
    """Fit per-head V-matrices.

    Args:
        keys: ``(H, T, D)`` float32.
        rank: Target rank.

    Returns:
        ``(H, rank, D)`` float32 right-singular-vector matrices.
    """
    H, T, D = keys.shape
    rank = min(rank, min(T, D))
    v_mat = np.empty((H, rank, D), dtype=np.float32)
    for h in range(H):
        _, _, vt = np.linalg.svd(keys[h], full_matrices=False)
        v_mat[h] = vt[:rank].astype(np.float32)
    return v_mat


def _numpy_store_batch(keys: np.ndarray, v_mat: np.ndarray) -> np.ndarray:
    """Project keys into low-rank space.

    Args:
        keys:  ``(H, T, D)`` float32.
        v_mat: ``(H, rank, D)`` float32.

    Returns:
        ``(H, T, rank)`` float32 compressed representations.
    """
    return np.einsum("htd,hrd->htr", keys, v_mat)


# ── Config / wrapper ──────────────────────────────────────────────────────────


@dataclass
class ShadowKVFitMojoConfig:
    """Configuration for :class:`MojoShadowKVFit`.

    Attributes:
        rank: Low-rank projection dimension.
    """

    rank: int = 16


class MojoShadowKVFit:
    """Mojo-backed ShadowKV SVD fitting and batch projection.

    Uses ``parallelize`` over heads for both the SVD fit and projection
    steps.  Falls back to NumPy when the Mojo runtime is absent.
    """

    def __init__(self, config: Optional[ShadowKVFitMojoConfig] = None) -> None:
        self._cfg = config or ShadowKVFitMojoConfig()

    def fit_svd(
        self,
        keys: np.ndarray,
        rank: Optional[int] = None,
    ) -> np.ndarray:
        """Learn per-head low-rank projection matrices.

        Args:
            keys: ``(H, T, D)`` float32 key cache.
            rank: Projection rank (overrides config).

        Returns:
            ``(H, rank, D)`` float32 V-matrix stack.

        Raises:
            ValueError: If ``keys`` is not 3-D.
        """
        k = np.ascontiguousarray(keys, dtype=np.float32)
        if k.ndim != 3:
            raise ValueError(f"keys must be 3-D (H,T,D), got {k.shape}")
        H, T, D = k.shape
        r = int(rank) if rank is not None else self._cfg.rank
        r = min(r, min(T, D))
        if _fit_kernel is not None:
            out = np.empty((H, r, D), dtype=np.float32)
            _fit_kernel(k.ctypes.data, out.ctypes.data, H, T, D, r)
            return out
        return _numpy_fit_svd(k, r)

    def store_batch(
        self,
        keys: np.ndarray,
        v_mat: np.ndarray,
    ) -> np.ndarray:
        """Project token keys into low-rank shadow space.

        Args:
            keys:  ``(H, T, D)`` float32.
            v_mat: ``(H, rank, D)`` float32 V-matrices.

        Returns:
            ``(H, T, rank)`` float32 projected keys.

        Raises:
            ValueError: If ``keys`` or ``v_mat`` are not 3-D.
        """
        k = np.ascontiguousarray(keys, dtype=np.float32)
        v = np.ascontiguousarray(v_mat, dtype=np.float32)
        if k.ndim != 3 or v.ndim != 3:
            raise ValueError("keys and v_mat must both be 3-D")
        H, T, D = k.shape
        _, rank, _ = v.shape
        if _store_kernel is not None:
            out = np.empty((H, T, rank), dtype=np.float32)
            _store_kernel(k.ctypes.data, v.ctypes.data, out.ctypes.data, H, T, D, rank)
            return out
        return _numpy_store_batch(k, v)

    def backend(self) -> str:
        return "mojo" if (_fit_kernel is not None and _store_kernel is not None) else "numpy"
