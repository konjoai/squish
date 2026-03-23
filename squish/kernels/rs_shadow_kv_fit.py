"""squish/kernels/rs_shadow_kv_fit.py — Rust-backed ShadowKV SVD fit + batch store.

Wraps ``squish_quant_rs.shadow_kv_svd_fit_f32`` and
``squish_quant_rs.shadow_kv_store_batch_f32`` with NumPy fallbacks.

ShadowKV builds per-head low-rank projections of the key cache at every
context window boundary.  The fitting step (thin SVD per head) and the
batch-store step (project each token into low-rank space) are both
parallelised over heads via Rayon.

Reference: Sun et al., "ShadowKV: KV Cache in Shadows for High-Throughput
Long-Context LLM Inference," arXiv 2410.21465, 2024.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

__all__ = [
    "ShadowKVFitConfig",
    "RustShadowKVFit",
]

try:
    import squish_quant as _sq
    _HAS_RUST = hasattr(_sq, "shadow_kv_svd_fit_f32") and hasattr(
        _sq, "shadow_kv_store_batch_f32"
    )
except ImportError:
    _sq = None  # type: ignore[assignment]
    _HAS_RUST = False


# ── NumPy fallbacks ───────────────────────────────────────────────────────────


def _numpy_svd_fit(keys: np.ndarray, rank: int) -> np.ndarray:
    """Thin SVD fit per head.

    Args:
        keys: ``(H, T, D)`` float32.
        rank: Number of right singular vectors to retain.

    Returns:
        ``(H, rank, D)`` float32 V-matrices.
    """
    n_heads, t_len, head_dim = keys.shape
    r = min(rank, head_dim, t_len)
    out = np.zeros((n_heads, r, head_dim), dtype=np.float32)
    for h in range(n_heads):
        _, _, vh = np.linalg.svd(keys[h], full_matrices=False)
        out[h] = vh[:r]
    return out


def _numpy_store_batch(keys: np.ndarray, v_mat: np.ndarray) -> np.ndarray:
    """Project key tokens into low-rank space.

    Args:
        keys:  ``(H, T, D)`` float32.
        v_mat: ``(H, rank, D)`` float32 right singular vectors.

    Returns:
        ``(H, T, rank)`` float32 projected keys.
    """
    # projected[h, i, r] = sum_d keys[h, i, d] * v_mat[h, r, d]
    return np.einsum("htd,hrd->htr", keys, v_mat)


# ── Config / wrapper ──────────────────────────────────────────────────────────


@dataclass
class ShadowKVFitConfig:
    """Configuration for :class:`RustShadowKVFit`.

    Attributes:
        rank: Number of singular vectors to retain per head.
    """

    rank: int = 16


class RustShadowKVFit:
    """Rust-accelerated ShadowKV per-head SVD fit and low-rank key projection.

    Provides two operations:
    * :meth:`fit_svd`     — compute V-matrices from a key cache snapshot.
    * :meth:`store_batch` — project a batch of keys into the low-rank space.

    Both operations parallelise over attention heads via Rayon.
    Falls back to NumPy when ``squish_quant_rs`` is unavailable.
    """

    def __init__(self, config: Optional[ShadowKVFitConfig] = None) -> None:
        self._cfg = config or ShadowKVFitConfig()

    def fit_svd(
        self,
        keys: np.ndarray,
        rank: Optional[int] = None,
    ) -> np.ndarray:
        """Fit per-head low-rank projections from the key cache.

        Args:
            keys: Key cache ``(n_heads, T, head_dim)`` float32.
            rank: Number of singular vectors (overrides config).

        Returns:
            V-matrices ``(n_heads, rank, head_dim)`` float32.

        Raises:
            ValueError: If ``keys`` is not 3-D.
        """
        k = np.ascontiguousarray(keys, dtype=np.float32)
        if k.ndim != 3:
            raise ValueError(f"keys must be 3-D (H, T, D), got {k.shape}")
        r = int(rank) if rank is not None else self._cfg.rank
        if _HAS_RUST:
            return np.asarray(_sq.shadow_kv_svd_fit_f32(k, r), dtype=np.float32)
        return _numpy_svd_fit(k, r)

    def store_batch(
        self,
        keys: np.ndarray,
        v_mat: np.ndarray,
    ) -> np.ndarray:
        """Project a batch of keys into the low-rank ShadowKV space.

        Args:
            keys:  New key tokens ``(n_heads, n_tokens, head_dim)`` float32.
            v_mat: V-matrices from :meth:`fit_svd`
                   ``(n_heads, rank, head_dim)`` float32.

        Returns:
            Projected tokens ``(n_heads, n_tokens, rank)`` float32.

        Raises:
            ValueError: If head counts or head dims are inconsistent.
        """
        k = np.ascontiguousarray(keys, dtype=np.float32)
        v = np.ascontiguousarray(v_mat, dtype=np.float32)
        if k.ndim != 3 or v.ndim != 3:
            raise ValueError("keys and v_mat must both be 3-D")
        if k.shape[0] != v.shape[0]:
            raise ValueError(
                f"n_heads mismatch: keys={k.shape[0]}, v_mat={v.shape[0]}"
            )
        if k.shape[2] != v.shape[2]:
            raise ValueError(
                f"head_dim mismatch: keys={k.shape[2]}, v_mat={v.shape[2]}"
            )
        if _HAS_RUST:
            return np.asarray(_sq.shadow_kv_store_batch_f32(k, v), dtype=np.float32)
        return _numpy_store_batch(k, v)

    def backend(self) -> str:
        return "rust" if _HAS_RUST else "numpy"
