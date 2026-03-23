"""rs_sparse_act_gemv.py — Rust-accelerated sparse activation GEMV.

Wraps ``sparse_act_gemv_f32`` from ``squish_quant_rs`` (Wave 59a).
Falls back to a dense NumPy dot when the Rust extension is unavailable.

RustSparseActGEMV skips zero (or near-zero) activation indices:
1. SIMD comparison pass builds non-zero index set.
2. Rayon output-row parallel compressed dot-product.

~2× effective throughput at 50% sparsity for 4096→11008 FFN projections.

Hooks into ``native_sparse_attn.py`` FFN gate and any module with
ReLU² / ReLU-gated projections where 30–60% of activations are zero.

Reference:
    Mirzadeh et al. (NeurIPS 2023, arXiv 2310.04564) — Relufication.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    import squish_quant as _sq
    _HAS_RUST = hasattr(_sq, "sparse_act_gemv_f32")
except ImportError:
    _HAS_RUST = False

__all__ = ["SparseActGEMVConfig", "RustSparseActGEMV"]


@dataclass
class SparseActGEMVConfig:
    threshold: float = 0.0      # values with |act| <= threshold treated as zero


class RustSparseActGEMV:
    """Sparse activation GEMV: ``output = W @ act`` skipping zero activations.

    Args:
        config: :class:`SparseActGEMVConfig`.
    """

    def __init__(self, config: Optional[SparseActGEMVConfig] = None) -> None:
        self._cfg = config or SparseActGEMVConfig()

    # ------------------------------------------------------------------
    def gemv(
        self,
        weight: np.ndarray,
        activation: np.ndarray,
        threshold: Optional[float] = None,
    ) -> np.ndarray:
        """Compute ``W @ activation`` exploiting activation sparsity.

        Args:
            weight: ``(out_features, in_features)`` float32 weight matrix.
            activation: ``(in_features,)`` float32 activation vector.
            threshold: Override config threshold (absolute value cutoff).

        Returns:
            ``(out_features,)`` float32 output vector.
        """
        w = np.ascontiguousarray(weight, dtype=np.float32)
        a = np.ascontiguousarray(activation.ravel(), dtype=np.float32)
        thr = float(threshold) if threshold is not None else self._cfg.threshold
        if w.shape[1] != a.shape[0]:
            raise ValueError(
                f"weight in_features={w.shape[1]} != activation length={a.shape[0]}"
            )
        if _HAS_RUST:
            return _sq.sparse_act_gemv_f32(w, a, thr)
        # NumPy fallback: apply threshold mask then dense matmul
        a_masked = np.where(np.abs(a) > thr, a, 0.0)
        return (w @ a_masked).astype(np.float32)

    def threshold(self) -> float:
        return self._cfg.threshold

    def sparsity(self, activation: np.ndarray) -> float:
        """Return the fraction of zero activations (for diagnostics)."""
        a = np.asarray(activation, dtype=np.float32).ravel()
        return float((np.abs(a) <= self._cfg.threshold).mean())

    def backend(self) -> str:
        return "rust" if _HAS_RUST else "numpy"
