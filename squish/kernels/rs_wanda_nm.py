"""squish/kernels/rs_wanda_nm.py — Rust-backed Wanda N:M sparsity kernel.

Wraps ``squish_quant_rs.wanda_importance_f32`` and
``squish_quant_rs.wanda_nm_mask_f32`` with NumPy fallbacks.

Wanda (Pruning by Weights and Activations) computes element-wise
importance as |W| × activation_rms, then applies N:M structured
sparsity by keeping the top-N entries per M-element column block.

Reference: Sun et al., "A Simple and Effective Pruning Approach for
Large Language Models." ICLR 2024.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

__all__ = [
    "WandaNMConfig",
    "RustWandaNM",
]

try:
    import squish_quant as _sq
    _HAS_RUST = hasattr(_sq, "wanda_importance_f32") and hasattr(
        _sq, "wanda_nm_mask_f32"
    )
except ImportError:
    _sq = None  # type: ignore[assignment]
    _HAS_RUST = False


# ── NumPy fallbacks ───────────────────────────────────────────────────────────


def _numpy_importance(weight: np.ndarray, rms: np.ndarray) -> np.ndarray:
    """Element-wise importance: |W| × rms broadcast."""
    return np.abs(weight) * rms[np.newaxis, :]


def _numpy_nm_mask(importance: np.ndarray, n: int, m: int) -> np.ndarray:
    """N:M structured sparsity mask — keep top-n per m-element block."""
    rows, cols = importance.shape
    mask = np.zeros_like(importance, dtype=np.uint8)
    n_blocks = (cols + m - 1) // m
    for bi in range(n_blocks):
        col_start = bi * m
        col_end = min(col_start + m, cols)
        block = importance[:, col_start:col_end]
        # Top-n indices per row within the block
        top_idx = np.argsort(-block, axis=1)[:, :n]
        for row in range(rows):
            for idx in top_idx[row]:
                mask[row, col_start + idx] = 1
    return mask


# ── Config / wrapper ──────────────────────────────────────────────────────────


@dataclass
class WandaNMConfig:
    """Configuration for :class:`RustWandaNM`.

    Attributes:
        n: Number of non-zero entries per M-element block.
        m: Block size for N:M sparsity (e.g., 2:4 → n=2, m=4).
    """

    n: int = 2
    m: int = 4


class RustWandaNM:
    """Rust-accelerated Wanda N:M importance scoring and mask generation.

    Computes element-wise activation-weighted importance for pruning and
    generates N:M structured sparsity masks.  Rayon parallelises over rows.
    Falls back to NumPy when ``squish_quant_rs`` is unavailable.
    """

    def __init__(self, config: Optional[WandaNMConfig] = None) -> None:
        self._cfg = config or WandaNMConfig()

    def importance(
        self,
        weight: np.ndarray,
        activation_rms: np.ndarray,
    ) -> np.ndarray:
        """Compute per-element Wanda importance: |W[r,c]| × activation_rms[c].

        Args:
            weight:          FP32 weight matrix ``(out_f, in_f)``.
            activation_rms:  Per-column activation RMS ``(in_f,)`` float32.

        Returns:
            Importance matrix ``(out_f, in_f)`` float32.
        """
        w = np.ascontiguousarray(weight, dtype=np.float32)
        rms = np.ascontiguousarray(activation_rms, dtype=np.float32).ravel()
        if w.shape[1] != rms.shape[0]:
            raise ValueError(
                f"weight in_features={w.shape[1]} != rms length={rms.shape[0]}"
            )
        if _HAS_RUST:
            return np.asarray(_sq.wanda_importance_f32(w, rms), dtype=np.float32)
        return _numpy_importance(w, rms)

    def nm_mask(
        self,
        importance: np.ndarray,
        n: Optional[int] = None,
        m: Optional[int] = None,
    ) -> np.ndarray:
        """Generate N:M structured sparsity mask from importance scores.

        Args:
            importance: Importance matrix ``(out_f, in_f)`` float32.
            n:          Non-zero count per block (overrides config).
            m:          Block size (overrides config).

        Returns:
            Binary mask ``(out_f, in_f)`` uint8 — 1 = keep, 0 = prune.

        Raises:
            ValueError: If n > m.
        """
        n_ = int(n) if n is not None else self._cfg.n
        m_ = int(m) if m is not None else self._cfg.m
        if n_ > m_:
            raise ValueError(f"n={n_} must be <= m={m_}")
        imp = np.ascontiguousarray(importance, dtype=np.float32)
        if _HAS_RUST:
            return np.asarray(_sq.wanda_nm_mask_f32(imp, n_, m_), dtype=np.uint8)
        return _numpy_nm_mask(imp, n_, m_)

    def prune(
        self,
        weight: np.ndarray,
        activation_rms: np.ndarray,
        n: Optional[int] = None,
        m: Optional[int] = None,
    ) -> np.ndarray:
        """Score, mask, and prune a weight matrix in one step.

        Returns a float32 weight matrix with pruned entries zeroed out.
        """
        imp = self.importance(weight, activation_rms)
        mask = self.nm_mask(imp, n=n, m=m)
        w = np.asarray(weight, dtype=np.float32)
        return w * mask.astype(np.float32)

    def backend(self) -> str:
        return "rust" if _HAS_RUST else "numpy"
