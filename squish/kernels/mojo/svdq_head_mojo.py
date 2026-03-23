"""squish/kernels/mojo/svdq_head_mojo.py — Mojo-backed SVDq per-head rank profiling.

Wraps ``svdq_head_rank`` Mojo kernel via MojoBridge with a NumPy fallback.
Computes per-head approximate singular values across transformer layers using
a randomised range-finder for fast rank calibration.

Reference: Zhang et al., "SVD-LLM: Truncation-aware Singular Value Decomposition
for Large Language Model Compression," arXiv 2403.07378, 2024.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from squish.kernels.mojo.mojo_bridge import MojoBridge

__all__ = [
    "SVDqHeadMojoConfig",
    "MojoSVDqHead",
]

_bridge = MojoBridge()
_rank_kernel = _bridge.load_kernel("svdq_head_rank")


# ── NumPy fallbacks ───────────────────────────────────────────────────────────


def _numpy_rank_profile(keys: np.ndarray) -> np.ndarray:
    """Compute leading k_svd singular values per (layer, head).

    Args:
        keys: ``(L, H, T, D)`` float32 key tensor.

    Returns:
        ``(L, H, k_svd)`` float32 singular value profiles.
    """
    L, H, T, D = keys.shape
    k_svd = min(T, D)
    out = np.empty((L, H, k_svd), dtype=np.float32)
    for l in range(L):
        for h in range(H):
            mat = keys[l, h]  # (T, D)
            _, s, _ = np.linalg.svd(mat, full_matrices=False)
            out[l, h] = s[:k_svd].astype(np.float32)
    return out


# ── Config / wrapper ──────────────────────────────────────────────────────────


@dataclass
class SVDqHeadMojoConfig:
    """Configuration for :class:`MojoSVDqHead`.

    Attributes:
        rank_threshold: Relative singular-value threshold for effective rank.
    """

    rank_threshold: float = 0.01


class MojoSVDqHead:
    """Mojo-backed SVDq per-head rank profiling.

    Uses ``parallelize`` over (layer, head) pairs with vectorised row-norm
    accumulation for the range-finder sketch.  Falls back to NumPy when the
    Mojo runtime is absent.
    """

    def __init__(self, config: Optional[SVDqHeadMojoConfig] = None) -> None:
        self._cfg = config or SVDqHeadMojoConfig()

    def rank_profile(self, keys: np.ndarray) -> np.ndarray:
        """Compute per-head singular-value profiles.

        Args:
            keys: ``(L, H, T, D)`` float32.

        Returns:
            ``(L, H, k_svd)`` float32 singular values per head.

        Raises:
            ValueError: If ``keys`` is not 4-D.
        """
        k = np.ascontiguousarray(keys, dtype=np.float32)
        if k.ndim != 4:
            raise ValueError(f"keys must be 4-D (L,H,T,D), got {k.shape}")
        if _rank_kernel is not None:
            L, H, T, D = k.shape
            k_svd = min(T, D)
            out = np.empty((L, H, k_svd), dtype=np.float32)
            _rank_kernel(k.ctypes.data, out.ctypes.data, L, H, T, D)
            return out
        return _numpy_rank_profile(k)

    def rank_per_head(
        self,
        keys: np.ndarray,
        threshold: Optional[float] = None,
    ) -> np.ndarray:
        """Compute effective rank per (layer, head) as integer count.

        Args:
            keys:      ``(L, H, T, D)`` float32.
            threshold: Relative threshold (overrides config).

        Returns:
            ``(L, H)`` int32 effective rank per head.
        """
        svd = self.rank_profile(keys)
        t = float(threshold) if threshold is not None else self._cfg.rank_threshold
        s_max = svd.max(axis=-1, keepdims=True) + 1e-9
        return (svd / s_max >= t).sum(axis=-1).astype(np.int32)

    def backend(self) -> str:
        return "mojo" if _rank_kernel is not None else "numpy"
