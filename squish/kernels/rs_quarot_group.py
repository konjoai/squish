"""rs_quarot_group.py — Rust-accelerated QuaRot group quantization / dequantization.

Wraps ``quarot_group_quant_f32`` and ``quarot_group_dequant_f32`` from
``squish_quant_rs`` (Wave 59a).
Falls back to NumPy when the Rust extension is unavailable.

RustQuaRotGroup replaces the ``for g in range(n_groups):`` loop in
``quarot_quant.py`` ``QuaRotQuantizer.quantise()``/``.dequantise()``,
operating on FP32-rotated weight matrices with symmetric or asymmetric
INT quantization; ~4× for 4096×4096, group_size=128.

Reference:
    Ashkboos et al. (NeurIPS 2024, arXiv 2404.00456) — QuaRoT.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    import squish_quant as _sq
    _HAS_RUST = hasattr(_sq, "quarot_group_quant_f32")
except ImportError:
    _HAS_RUST = False

__all__ = ["QuaRotGroupConfig", "RustQuaRotGroup"]


@dataclass
class QuaRotGroupConfig:
    group_size: int = 128
    q_max: float = 7.0
    symmetric: bool = True


def _numpy_quant(
    weight: np.ndarray,
    group_size: int,
    q_max: float,
    symmetric: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows, cols = weight.shape
    gs = max(1, group_size)
    n_groups = (cols + gs - 1) // gs
    codes = np.zeros((rows, cols), dtype=np.int32)
    scales = np.ones(n_groups, dtype=np.float32)
    zeros = np.zeros(n_groups, dtype=np.float32)
    for g in range(n_groups):
        c0, c1 = g * gs, min((g + 1) * gs, cols)
        block = weight[:, c0:c1].astype(np.float32)
        if symmetric:
            abs_max = float(np.abs(block).max())
            scale = abs_max / q_max if abs_max > 1e-9 else 1.0
            scales[g] = scale
            codes[:, c0:c1] = np.round(block / scale).clip(-q_max, q_max).astype(np.int32)
        else:
            vmin, vmax = float(block.min()), float(block.max())
            rng = max(vmax - vmin, 1e-9)
            scale = rng / (2.0 * q_max)
            zero = -vmin / scale
            scales[g] = scale
            zeros[g] = zero
            codes[:, c0:c1] = (
                np.round(block / scale + zero).clip(0.0, 2.0 * q_max).astype(np.int32)
            )
    return codes, scales, zeros


def _numpy_dequant(
    codes: np.ndarray,
    scales: np.ndarray,
    zeros: np.ndarray,
    rows: int,
    cols: int,
    group_size: int,
) -> np.ndarray:
    gs = max(1, group_size)
    out = np.empty((rows, cols), dtype=np.float32)
    for j in range(cols):
        g = j // gs
        out[:, j] = (codes[:, j].astype(np.float32) - zeros[g]) * scales[g]
    return out


class RustQuaRotGroup:
    """QuaRot group quantization and dequantization for rotated weight matrices.

    Args:
        config: :class:`QuaRotGroupConfig`.
    """

    def __init__(self, config: Optional[QuaRotGroupConfig] = None) -> None:
        self._cfg = config or QuaRotGroupConfig()

    # ------------------------------------------------------------------
    def quantize(
        self,
        weight: np.ndarray,
        group_size: Optional[int] = None,
        q_max: Optional[float] = None,
        symmetric: Optional[bool] = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Quantize *weight* into (codes, scales, zeros).

        Returns:
            ``(codes (rows, cols) int32, scales (n_groups,) float32,
               zeros (n_groups,) float32)``
        """
        w = np.ascontiguousarray(weight, dtype=np.float32)
        rows, cols = w.shape
        gs = int(group_size) if group_size is not None else self._cfg.group_size
        qm = float(q_max) if q_max is not None else self._cfg.q_max
        sym = bool(symmetric) if symmetric is not None else self._cfg.symmetric
        if _HAS_RUST:
            codes_flat, scales, zeros = _sq.quarot_group_quant_f32(w, gs, qm, sym)
            return codes_flat.reshape(rows, cols), scales, zeros
        return _numpy_quant(w, gs, qm, sym)

    def dequantize(
        self,
        codes: np.ndarray,
        scales: np.ndarray,
        zeros: np.ndarray,
        group_size: Optional[int] = None,
    ) -> np.ndarray:
        """Reconstruct float32 weights from quantized representation.

        Args:
            codes: ``(rows, cols)`` int32 or float32 code array.
            scales: ``(n_groups,)`` float32.
            zeros: ``(n_groups,)`` float32.

        Returns:
            ``(rows, cols)`` float32 weight matrix.
        """
        rows, cols = codes.shape
        gs = int(group_size) if group_size is not None else self._cfg.group_size
        codes_i = np.ascontiguousarray(codes.ravel(), dtype=np.int32)
        s = np.ascontiguousarray(scales, dtype=np.float32)
        z = np.ascontiguousarray(zeros, dtype=np.float32)
        if _HAS_RUST:
            out_flat = _sq.quarot_group_dequant_f32(codes_i, s, z, rows, cols, gs)
            return out_flat.reshape(rows, cols)
        return _numpy_dequant(codes.reshape(rows, cols), s, z, rows, cols, gs)

    def group_size(self) -> int:
        return self._cfg.group_size

    def q_max(self) -> float:
        return self._cfg.q_max

    def backend(self) -> str:
        return "rust" if _HAS_RUST else "numpy"
