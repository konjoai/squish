"""squish/kernels/rs_flute_lut.py — Rust-backed FLUTE LUT quantization kernel.

Wraps ``squish_quant_rs.flute_lut_encode_f32`` and
``squish_quant_rs.flute_lut_decode_u8`` with NumPy fallbacks.

FLUTE (Flexible Lookup Table Engine) stores per-group codebooks and
encodes/decodes weight values via nearest-entry lookup.  Rayon
parallelises encode/decode over column groups.

Reference: Guo et al., "FLUTE: Flexible Lookup Table Engine for LUTbased
Network Inference." CVPR 2024.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

__all__ = [
    "FluteLUTConfig",
    "RustFluteLUT",
]

try:
    import squish_quant as _sq
    _HAS_RUST = hasattr(_sq, "flute_lut_encode_f32") and hasattr(
        _sq, "flute_lut_decode_u8"
    )
except ImportError:
    _sq = None  # type: ignore[assignment]
    _HAS_RUST = False


# ── NumPy fallbacks ───────────────────────────────────────────────────────────


def _numpy_encode(
    weight: np.ndarray,
    codebook: np.ndarray,
    group_size: int,
) -> np.ndarray:
    """Encode weights to codebook indices (argmin L1 per element)."""
    rows, cols = weight.shape
    n_groups = codebook.shape[0]
    cb_size = codebook.shape[1]
    codes = np.zeros((rows, cols), dtype=np.uint8)
    for g in range(n_groups):
        col_start = g * group_size
        col_end = min(col_start + group_size, cols)
        cb_g = codebook[g]  # (cb_size,)
        for c in range(col_start, col_end):
            diff = np.abs(weight[:, c : c + 1] - cb_g[np.newaxis, :])  # (rows, cb_size)
            codes[:, c] = np.argmin(diff, axis=1).astype(np.uint8)
    return codes


def _numpy_decode(
    codes: np.ndarray,
    codebook: np.ndarray,
    group_size: int,
) -> np.ndarray:
    """Decode codebook indices to float32 weight values."""
    rows, cols = codes.shape
    n_groups = codebook.shape[0]
    out = np.zeros((rows, cols), dtype=np.float32)
    for g in range(n_groups):
        col_start = g * group_size
        col_end = min(col_start + group_size, cols)
        cb_g = codebook[g]  # (cb_size,)
        out[:, col_start:col_end] = cb_g[codes[:, col_start:col_end].astype(int)]
    return out


# ── Config / wrapper ──────────────────────────────────────────────────────────


@dataclass
class FluteLUTConfig:
    """Configuration for :class:`RustFluteLUT`.

    Attributes:
        group_size: Number of weight columns per codebook group.
        cb_size:    Codebook entries per group (≤ 256 for uint8 codes).
    """

    group_size: int = 128
    cb_size: int = 16


class RustFluteLUT:
    """Rust-accelerated FLUTE per-group LUT quantization.

    Encodes weight matrices to codebook indices and decodes them back,
    implementing the FLUTE group-LUT quantization scheme.
    Rayon parallelises over column groups.
    Falls back to NumPy when ``squish_quant_rs`` is unavailable.
    """

    def __init__(self, config: Optional[FluteLUTConfig] = None) -> None:
        self._cfg = config or FluteLUTConfig()

    def encode(
        self,
        weight: np.ndarray,
        codebook: np.ndarray,
        group_size: Optional[int] = None,
    ) -> np.ndarray:
        """Encode a weight matrix to per-group LUT code indices.

        Args:
            weight:     FP32 weight matrix ``(rows, cols)``.
            codebook:   FP32 per-group codebook ``(n_groups, cb_size)``.
                        Number of groups = ceil(cols / group_size).
            group_size: Column group size (overrides config).

        Returns:
            Code index matrix ``(rows, cols)`` uint8.

        Raises:
            ValueError: If codebook has wrong number of groups.
        """
        w = np.ascontiguousarray(weight, dtype=np.float32)
        cb = np.ascontiguousarray(codebook, dtype=np.float32)
        gs = int(group_size) if group_size is not None else self._cfg.group_size
        n_groups_expected = (w.shape[1] + gs - 1) // gs
        if cb.shape[0] != n_groups_expected:
            raise ValueError(
                f"Expected {n_groups_expected} codebook groups "
                f"for cols={w.shape[1]}, group_size={gs}; got {cb.shape[0]}"
            )
        if _HAS_RUST:
            return np.asarray(_sq.flute_lut_encode_f32(w, cb, gs), dtype=np.uint8)
        return _numpy_encode(w, cb, gs)

    def decode(
        self,
        codes: np.ndarray,
        codebook: np.ndarray,
        group_size: Optional[int] = None,
    ) -> np.ndarray:
        """Decode LUT code indices back to floating-point weight values.

        Args:
            codes:      Code index matrix ``(rows, cols)`` uint8.
            codebook:   FP32 per-group codebook ``(n_groups, cb_size)``.
            group_size: Column group size (overrides config).

        Returns:
            Reconstructed weight matrix ``(rows, cols)`` float32.
        """
        c = np.ascontiguousarray(codes, dtype=np.uint8)
        cb = np.ascontiguousarray(codebook, dtype=np.float32)
        gs = int(group_size) if group_size is not None else self._cfg.group_size
        if _HAS_RUST:
            return np.asarray(_sq.flute_lut_decode_u8(c, cb, gs), dtype=np.float32)
        return _numpy_decode(c, cb, gs)

    def roundtrip_error(
        self,
        weight: np.ndarray,
        codebook: np.ndarray,
        group_size: Optional[int] = None,
    ) -> float:
        """Mean absolute error of encode → decode roundtrip."""
        codes = self.encode(weight, codebook, group_size)
        decoded = self.decode(codes, codebook, group_size)
        return float(np.abs(weight.astype(np.float32) - decoded).mean())

    def backend(self) -> str:
        return "rust" if _HAS_RUST else "numpy"
