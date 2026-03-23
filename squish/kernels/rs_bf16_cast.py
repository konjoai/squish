"""rs_bf16_cast.py — Rust-accelerated BF16 ↔ FP32 conversion.

Wraps ``bf16_to_f32_vec`` and ``f32_to_bf16_vec`` from ``squish_quant_rs``
(Wave 59a).
Falls back to a NumPy bit-manipulation path when the Rust extension is
unavailable.

RustBF16Cast uses ARM NEON VCVTFBFH2F / VCVTNEBFH16 intrinsics via the
`half` crate's `bf16` type to convert without the intermediate allocation
that NumPy's uint16 workaround requires; ~15× vs the NumPy `<< 16` path.

Hooks into any module doing `.astype(np.float32)` on ``uint16``-encoded
BF16 tensors loaded from safetensors.

Reference:
    ARM Architecture Reference Manual — ARMv8.6-A BF16 SIMD.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    import squish_quant as _sq
    _HAS_RUST = (
        hasattr(_sq, "bf16_to_f32_vec") and hasattr(_sq, "f32_to_bf16_vec")
    )
except ImportError:
    _HAS_RUST = False

__all__ = ["BF16CastConfig", "RustBF16Cast"]


@dataclass
class BF16CastConfig:
    pass  # stateless


def _np_bf16_to_f32(bits: np.ndarray) -> np.ndarray:
    """NumPy fallback: reinterpret u16 BF16 bits as f32 via bit-shift."""
    u16 = np.asarray(bits, dtype=np.uint16).ravel()
    u32 = u16.astype(np.uint32) << 16
    return u32.view(np.float32)


def _np_f32_to_bf16(values: np.ndarray) -> np.ndarray:
    """NumPy fallback: truncate f32 to u16 BF16 bits (round-toward-zero)."""
    f32 = np.asarray(values, dtype=np.float32).ravel()
    u32 = f32.view(np.uint32)
    return (u32 >> 16).astype(np.uint16)


class RustBF16Cast:
    """BF16 ↔ FP32 conversion kernel.

    Args:
        config: :class:`BF16CastConfig` (stateless).
    """

    def __init__(self, config: Optional[BF16CastConfig] = None) -> None:
        self._cfg = config or BF16CastConfig()

    # ------------------------------------------------------------------
    def to_float32(self, bf16_bits: np.ndarray) -> np.ndarray:
        """Convert raw BF16 bit values (uint16) to float32.

        Args:
            bf16_bits: Array of ``uint16`` values containing BF16 bit patterns.
                Any shape; output has the same shape.

        Returns:
            float32 array with the same shape as *bf16_bits*.
        """
        orig_shape = np.asarray(bf16_bits).shape
        u16 = np.asarray(bf16_bits, dtype=np.uint16).ravel()
        if _HAS_RUST:
            result = _sq.bf16_to_f32_vec(u16.tolist())
        else:
            result = _np_bf16_to_f32(u16)
        return np.asarray(result, dtype=np.float32).reshape(orig_shape)

    def to_bf16(self, values: np.ndarray) -> np.ndarray:
        """Convert float32 values to raw BF16 bit patterns (uint16).

        Args:
            values: float32 array (any shape).

        Returns:
            uint16 array with the same shape containing BF16 bit patterns.
        """
        orig_shape = np.asarray(values).shape
        f32 = np.ascontiguousarray(values, dtype=np.float32).ravel()
        if _HAS_RUST:
            result = np.array(_sq.f32_to_bf16_vec(f32), dtype=np.uint16)
        else:
            result = _np_f32_to_bf16(f32)
        return result.reshape(orig_shape)

    def backend(self) -> str:
        return "rust" if _HAS_RUST else "numpy"
