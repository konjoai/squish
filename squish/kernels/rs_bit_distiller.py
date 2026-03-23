"""squish/kernels/rs_bit_distiller.py — Rust-backed BitDistiller weight quantisation.

Wraps ``squish_quant_rs.bit_distiller_quant_f32`` and
``squish_quant_rs.bit_distiller_refine_f32`` with NumPy fallbacks.

BitDistiller applies knowledge-distillation-aware scale refinement during
post-training quantisation: an initial per-group min/max quantisation step is
followed by iterative KL-guided scale re-fitting using teacher soft logits.

Reference: Du et al., "BitDistiller: Unleashing the Potential of Sub-4-Bit
LLMs via Self-Distillation," arXiv 2402.10631, 2024.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

__all__ = [
    "BitDistillerConfig",
    "RustBitDistiller",
]

try:
    import squish_quant as _sq
    _HAS_QUANT = hasattr(_sq, "bit_distiller_quant_f32")
    _HAS_REFINE = hasattr(_sq, "bit_distiller_refine_f32")
    _HAS_RUST = _HAS_QUANT and _HAS_REFINE
except ImportError:
    _sq = None  # type: ignore[assignment]
    _HAS_QUANT = _HAS_REFINE = _HAS_RUST = False


# ── NumPy fallbacks ───────────────────────────────────────────────────────────


def _numpy_quant(
    w: np.ndarray,
    bits: int,
    group_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-group min/max asymmetric quantisation.

    Args:
        w:          ``(rows, cols)`` float32.
        bits:       Quantisation bits.
        group_size: Elements per group.

    Returns:
        Tuple of quantised ``(rows, cols)`` int8,
        scales ``(n_blocks,)`` float32,
        zeros ``(n_blocks,)`` float32.
    """
    rows, cols = w.shape
    gs = max(1, group_size)
    levels = float((1 << bits) - 1)
    q = np.empty_like(w, dtype=np.int8)
    scales_list: list[float] = []
    zeros_list: list[float] = []
    for r in range(rows):
        row = w[r]
        for g_start in range(0, cols, gs):
            chunk = row[g_start:g_start + gs]
            mn = float(chunk.min())
            mx = float(chunk.max())
            rng = max(mx - mn, 1e-8)
            scale = rng / levels
            zero = -mn / scale
            scales_list.append(scale)
            zeros_list.append(zero)
            qchunk = np.round((chunk - mn) / rng * levels).clip(0, levels).astype(np.int8)
            q[r, g_start:g_start + gs] = qchunk
    return q, np.array(scales_list, dtype=np.float32), np.array(zeros_list, dtype=np.float32)


def _numpy_refine(
    w: np.ndarray,
    teacher: np.ndarray,
    bits: int,
    n_steps: int,
    temperature: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """KL-distillation scale refinement fallback.

    Args:
        w:           ``(rows, cols)`` float32 student weights.
        teacher:     ``(rows, cols)`` float32 teacher reference.
        bits:        Quantisation bits.
        n_steps:     Refinement iterations.
        temperature: Softmax temperature.

    Returns:
        Tuple of refined scales ``(n_blocks,)`` float32,
        zeros ``(n_blocks,)`` float32.
    """
    rows, cols = w.shape
    gs = min(128, cols)
    levels = float((1 << bits) - 1)
    temp = max(float(temperature), 1e-6)
    # Initial quant
    scales, zeros = [], []
    for r in range(rows):
        for g_start in range(0, cols, gs):
            chunk = w[r, g_start:g_start + gs]
            mn = float(chunk.min())
            mx = float(chunk.max())
            rng = max(mx - mn, 1e-8)
            scales.append(rng / levels)
            zeros.append(-mn / (rng / levels))
    scales_arr = np.array(scales, dtype=np.float32)
    zeros_arr = np.array(zeros, dtype=np.float32)
    groups_per_row = (cols + gs - 1) // gs
    for _ in range(n_steps):
        for r in range(rows):
            t_row = teacher[r]
            w_row = w[r]
            for g_idx, g_start in enumerate(range(0, cols, gs)):
                wchunk = w_row[g_start:g_start + gs]
                tchunk = t_row[g_start:g_start + gs]
                scale = max(scales_arr[r * groups_per_row + g_idx], 1e-8)
                zero = zeros_arr[r * groups_per_row + g_idx]
                tsoft = np.exp((tchunk - tchunk.max()) / temp)
                tsoft /= max(tsoft.sum(), 1e-8)
                qv = (np.round(wchunk / scale + zero).clip(0, levels) - zero) * scale
                wsum = float((tsoft * qv).sum())
                wsum2 = float((tsoft * qv * qv).sum())
                var = max(wsum2 - wsum * wsum, 0.0)
                new_scale = max(var ** 0.5 * 2.0 / levels, 1e-8)
                scales_arr[r * groups_per_row + g_idx] = new_scale
                zeros_arr[r * groups_per_row + g_idx] = -float(wchunk.min()) / new_scale
    return scales_arr, zeros_arr


# ── Config / wrapper ──────────────────────────────────────────────────────────


@dataclass
class BitDistillerConfig:
    """Configuration for :class:`RustBitDistiller`.

    Attributes:
        bits:        Default quantisation bits.
        group_size:  Default elements per quantisation group.
        n_steps:     Scale-refinement iterations.
        temperature: KL softmax temperature.
    """

    bits: int = 4
    group_size: int = 128
    n_steps: int = 5
    temperature: float = 1.0


class RustBitDistiller:
    """Rust-accelerated BitDistiller weight quantisation.

    Provides per-group asymmetric quantisation and optional KL-distillation
    scale refinement.  Falls back to NumPy when ``squish_quant_rs`` is
    unavailable.

    Example::

        bd = RustBitDistiller()
        q, scales, zeros = bd.quantize(W)
        scales, zeros = bd.refine(W, teacher=teacher_W)
        W_hat = bd.dequantize(q, scales, zeros)
    """

    def __init__(self, config: Optional[BitDistillerConfig] = None) -> None:
        self._cfg = config or BitDistillerConfig()

    def quantize(
        self,
        w: np.ndarray,
        bits: Optional[int] = None,
        group_size: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Quantise a weight matrix.

        Args:
            w:          ``(rows, cols)`` float32 weight matrix.
            bits:       Quantisation bits (overrides config).
            group_size: Elements per group (overrides config).

        Returns:
            Tuple of:
            - ``quantized``: ``(rows, cols)`` int8.
            - ``scales``:    ``(n_blocks,)`` float32.
            - ``zeros``:     ``(n_blocks,)`` float32.

        Raises:
            ValueError: If ``w`` is not 2-D.
        """
        arr = np.ascontiguousarray(w, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(f"w must be 2-D (rows, cols), got {arr.shape}")
        b = int(bits) if bits is not None else self._cfg.bits
        gs = int(group_size) if group_size is not None else self._cfg.group_size
        if _HAS_QUANT:
            q, s, z = _sq.bit_distiller_quant_f32(arr, b, gs)
            return (
                np.asarray(q, dtype=np.int8),
                np.asarray(s, dtype=np.float32),
                np.asarray(z, dtype=np.float32),
            )
        return _numpy_quant(arr, b, gs)

    def refine(
        self,
        w: np.ndarray,
        teacher: np.ndarray,
        bits: Optional[int] = None,
        n_steps: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """KL-distillation scale refinement.

        Args:
            w:           ``(rows, cols)`` float32 student weight matrix.
            teacher:     ``(rows, cols)`` float32 teacher reference.
            bits:        Quantisation bits (overrides config).
            n_steps:     Refinement iterations (overrides config).
            temperature: KL softmax temperature (overrides config).

        Returns:
            Tuple of refined scales ``(n_blocks,)`` float32,
            zeros ``(n_blocks,)`` float32.

        Raises:
            ValueError: If shapes are incompatible.
        """
        wa = np.ascontiguousarray(w, dtype=np.float32)
        ta = np.ascontiguousarray(teacher, dtype=np.float32)
        if wa.ndim != 2 or ta.shape != wa.shape:
            raise ValueError(f"w and teacher must be 2-D with matching shapes; got {wa.shape}, {ta.shape}")
        b = int(bits) if bits is not None else self._cfg.bits
        ns = int(n_steps) if n_steps is not None else self._cfg.n_steps
        tc = float(temperature) if temperature is not None else self._cfg.temperature
        if _HAS_REFINE:
            s, z = _sq.bit_distiller_refine_f32(wa, ta, b, ns, tc)
            return np.asarray(s, dtype=np.float32), np.asarray(z, dtype=np.float32)
        return _numpy_refine(wa, ta, b, ns, tc)

    def dequantize(
        self,
        quantized: np.ndarray,
        scales: np.ndarray,
        zeros: np.ndarray,
        group_size: Optional[int] = None,
    ) -> np.ndarray:
        """Reconstruct float32 weights from quantised form.

        Args:
            quantized:  ``(rows, cols)`` int8.
            scales:     ``(n_blocks,)`` float32.
            zeros:      ``(n_blocks,)`` float32.
            group_size: Elements per group (overrides config).

        Returns:
            ``(rows, cols)`` float32 reconstructed weights.
        """
        q = np.asarray(quantized, dtype=np.float32)
        rows, cols = q.shape
        gs = int(group_size) if group_size is not None else self._cfg.group_size
        gs = max(1, gs)
        groups_per_row = (cols + gs - 1) // gs
        out = np.empty_like(q)
        for r in range(rows):
            for g_idx, g_start in enumerate(range(0, cols, gs)):
                scale = scales[r * groups_per_row + g_idx]
                zero = zeros[r * groups_per_row + g_idx]
                out[r, g_start:g_start + gs] = (
                    q[r, g_start:g_start + gs] - zero
                ) * scale
        return out.astype(np.float32)

    def backend(self) -> str:
        return "rust" if _HAS_RUST else "numpy"
