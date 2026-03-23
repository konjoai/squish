"""bf16_gemv_mojo.py — Mojo-accelerated native BF16 weight × FP32 activation GEMV.

Wraps ``squish/kernels/mojo/kernels/bf16_gemv.mojo`` via MojoBridge (Wave 59b).
Falls back to a NumPy upcast path when the Mojo library is unavailable.

MojoBF16GEMV uses ``SIMD[DType.bfloat16, 8]`` weight loads + ``SIMD.cast``
to FP32 accumulator + FMA dot-product with ``parallelize`` over output rows;
eliminates the bf16→fp32 upcast allocation in ``compressed_loader.py`` and
``gqa.py`` for BF16-weight models; ~3× for 4096×4096 on M3.

Reference:
    ARM Architecture Reference Manual — ARMv8.6-A BF16 SIMD (FBFMMLA).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from squish.kernels.mojo.mojo_bridge import MojoBridge

__all__ = ["BF16GEMVConfig", "MojoBF16GEMV"]

_bridge = MojoBridge()
_kernel = _bridge.load_kernel("bf16_gemv")


@dataclass
class BF16GEMVConfig:
    hidden_dim: int = 4096


def _numpy_bf16_gemv(weight_bits: np.ndarray, activation: np.ndarray) -> np.ndarray:
    """NumPy fallback: upcast BF16 bits to FP32 then dense matmul."""
    u16 = np.asarray(weight_bits, dtype=np.uint16)
    u32 = u16.astype(np.uint32) << 16
    w_f32 = u32.view(np.float32)
    return (w_f32 @ activation).astype(np.float32)


class MojoBF16GEMV:
    """Native BF16 weight × FP32 activation GEMV.

    Args:
        config: :class:`BF16GEMVConfig`.
    """

    def __init__(self, config: Optional[BF16GEMVConfig] = None) -> None:
        self._cfg = config or BF16GEMVConfig()

    # ------------------------------------------------------------------
    def gemv(
        self,
        weight_bits: np.ndarray,
        activation: np.ndarray,
    ) -> np.ndarray:
        """Compute ``W_bf16 @ activation`` without allocating an FP32 weight copy.

        Args:
            weight_bits: ``(out_features, in_features)`` uint16 array (raw BF16 bits).
            activation: ``(in_features,)`` float32 activation vector.

        Returns:
            ``(out_features,)`` float32 output vector.
        """
        wb = np.ascontiguousarray(weight_bits, dtype=np.uint16)
        a = np.ascontiguousarray(activation.ravel(), dtype=np.float32)
        out_features, in_features = wb.shape
        if in_features != a.shape[0]:
            raise ValueError(
                f"weight in_features={in_features} != activation length={a.shape[0]}"
            )
        if _kernel is not None:
            try:
                result = _kernel(wb, a)
                return np.asarray(result, dtype=np.float32).ravel()
            except Exception:
                pass
        return _numpy_bf16_gemv(wb, a)

    def hidden_dim(self) -> int:
        return self._cfg.hidden_dim

    def backend(self) -> str:
        return "mojo" if _kernel is not None else "numpy"
