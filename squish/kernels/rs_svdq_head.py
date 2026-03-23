"""squish/kernels/rs_svdq_head.py — Rust-backed SVDq per-head rank profiling.

Wraps ``squish_quant_rs.svdq_head_rank_f32`` with a NumPy fallback.

SVDq calibration searches per-(layer, head) singular-value spectra to
determine the optimal low-rank decomposition rank for each attention head.
The inner double-loop (n_layers × n_heads independent SVDs) is parallelised
over the flattened grid via Rayon ``into_par_iter``.

Reference: Liu et al., "SVD-LLM: Truncation-aware Singular Value
Decomposition for Large Language Models," arXiv 2403.07378, 2024.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

__all__ = [
    "SVDqHeadConfig",
    "RustSVDqHead",
]

try:
    import squish_quant as _sq
    _HAS_RUST = hasattr(_sq, "svdq_head_rank_f32")
except ImportError:
    _sq = None  # type: ignore[assignment]
    _HAS_RUST = False


# ── NumPy fallback ────────────────────────────────────────────────────────────


def _numpy_svdq_head_rank(keys: np.ndarray) -> np.ndarray:
    """Compute per-head singular values via numpy.linalg.svd.

    Args:
        keys: ``(n_layers, n_heads, T, D)`` float32.

    Returns:
        ``(n_layers, n_heads, min(T,D))`` float32 singular values (descending).
    """
    n_layers, n_heads, t_len, head_dim = keys.shape
    k_svd = min(t_len, head_dim)
    out = np.zeros((n_layers, n_heads, k_svd), dtype=np.float32)
    for li in range(n_layers):
        for hi in range(n_heads):
            sv = np.linalg.svd(keys[li, hi], compute_uv=False)
            out[li, hi, :len(sv)] = sv[:k_svd]
    return out


# ── Config / wrapper ──────────────────────────────────────────────────────────


@dataclass
class SVDqHeadConfig:
    """Configuration for :class:`RustSVDqHead`.

    Attributes:
        rank_threshold: Minimum relative singular-value ratio to consider a
            rank significant (used in :meth:`rank_per_head`).
    """

    rank_threshold: float = 0.01


class RustSVDqHead:
    """Rust-accelerated SVDq per-head rank profiling.

    Computes singular-value spectra for each (layer, head) pair in parallel.
    Used during calibration to determine optimal low-rank decomposition ranks
    for SVD-based quantisation.
    Falls back to NumPy when ``squish_quant_rs`` is unavailable.
    """

    def __init__(self, config: Optional[SVDqHeadConfig] = None) -> None:
        self._cfg = config or SVDqHeadConfig()

    def rank_profile(self, keys: np.ndarray) -> np.ndarray:
        """Compute singular-value spectrum for every (layer, head) pair.

        Args:
            keys: Stacked key tensors ``(n_layers, n_heads, T, D)`` float32.

        Returns:
            Singular values ``(n_layers, n_heads, min(T,D))`` float32,
            sorted descending within each head.

        Raises:
            ValueError: If ``keys`` is not 4-D or not float32-compatible.
        """
        k = np.ascontiguousarray(keys, dtype=np.float32)
        if k.ndim != 4:
            raise ValueError(f"keys must be 4-D, got shape {k.shape}")
        if _HAS_RUST:
            return np.asarray(_sq.svdq_head_rank_f32(k), dtype=np.float32)
        return _numpy_svdq_head_rank(k)

    def rank_per_head(
        self,
        keys: np.ndarray,
        threshold: Optional[float] = None,
    ) -> np.ndarray:
        """Determine effective rank per head from the singular-value profile.

        Counts singular values above ``threshold × s_max`` per head.

        Args:
            keys:      ``(n_layers, n_heads, T, D)`` float32.
            threshold: Relative threshold (overrides config).

        Returns:
            ``(n_layers, n_heads)`` int32 effective rank per head.
        """
        thr = threshold if threshold is not None else self._cfg.rank_threshold
        sv = self.rank_profile(keys)  # (L, H, k_svd)
        s_max = sv.max(axis=-1, keepdims=True) + 1e-12
        return (sv > thr * s_max).sum(axis=-1).astype(np.int32)

    def backend(self) -> str:
        return "rust" if _HAS_RUST else "numpy"
