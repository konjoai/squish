"""squish/kernels/rs_ternary_gemv.py — Rust-backed ternary GEMV kernel.

Wraps ``squish_quant_rs.ternary_gemv_i8`` with a NumPy fallback.
Implements INT8 {-1, 0, +1} weight × FP32 activation matrix-vector product
(BitNet b1.58 inference path).

Reference: Ma et al., "The Era of 1-bit LLMs: All Large Language Models are
in 1.58 Bits." arXiv 2402.17764.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

__all__ = [
    "TernaryGEMVConfig",
    "RustTernaryGEMV",
]

try:
    import squish_quant as _sq
    _HAS_RUST = hasattr(_sq, "ternary_gemv_i8")
except ImportError:
    _sq = None  # type: ignore[assignment]
    _HAS_RUST = False


# ── NumPy fallback ────────────────────────────────────────────────────────────


def _numpy_ternary_gemv(
    weight: np.ndarray,    # (out_features, in_features) int8 {-1,0,+1}
    activation: np.ndarray,  # (in_features,) float32
    scale: float,
) -> np.ndarray:
    # Exploit ternary structure: separate +1 and -1 masks
    pos_mask = weight == 1
    neg_mask = weight == -1
    out = (pos_mask @ activation - neg_mask @ activation) * scale
    return out.astype(np.float32)


# ── Config / wrapper ──────────────────────────────────────────────────────────


@dataclass
class TernaryGEMVConfig:
    """Configuration for :class:`RustTernaryGEMV`.

    Attributes:
        scale: Default dequantisation scale (mean(|W_orig|)).
    """

    scale: float = 1.0


class RustTernaryGEMV:
    """Rust-accelerated ternary GEMV: INT8 {-1,0,+1} × FP32.

    Uses the {-1, 0, +1} weight structure to skip zero elements and
    replace multiplications with additions/subtractions.
    Rayon parallelises over output rows.
    Falls back to NumPy when ``squish_quant_rs`` is unavailable.
    """

    def __init__(self, config: Optional[TernaryGEMVConfig] = None) -> None:
        self._cfg = config or TernaryGEMVConfig()

    def gemv(
        self,
        weight: np.ndarray,
        activation: np.ndarray,
        scale: Optional[float] = None,
    ) -> np.ndarray:
        """Compute W @ a with ternary SIMD acceleration.

        Args:
            weight:     INT8 ternary weight matrix
                        ``(out_features, in_features)`` — values in {-1, 0, +1}.
            activation: FP32 activation vector ``(in_features,)``.
            scale:      Override dequantisation scale.

        Returns:
            Output vector ``(out_features,)`` float32.

        Raises:
            ValueError: If weight columns != activation length.
        """
        w = np.ascontiguousarray(weight, dtype=np.int8)
        a = np.ascontiguousarray(activation, dtype=np.float32).ravel()
        if w.shape[1] != a.shape[0]:
            raise ValueError(
                f"weight in_features={w.shape[1]} != activation length={a.shape[0]}"
            )
        s = float(scale) if scale is not None else self._cfg.scale
        if _HAS_RUST:
            raw = _sq.ternary_gemv_i8(w, a, s)
            return np.asarray(raw, dtype=np.float32)
        return _numpy_ternary_gemv(w, a, s)

    def sparsity(self, weight: np.ndarray) -> float:
        """Return the fraction of zero weights (diagnostic)."""
        w = np.asarray(weight, dtype=np.int8).ravel()
        return float((w == 0).mean())

    # ── properties ───────────────────────────────────────────────────────────

    def scale(self) -> float:
        return self._cfg.scale

    def backend(self) -> str:
        return "rust" if _HAS_RUST else "numpy"
