"""squish/kernels/mojo/flute_lut_mojo.py — Mojo-backed FLUTE LUT kernel.

Wraps ``flute_lut_encode`` / ``flute_lut_decode`` Mojo kernels via
MojoBridge with NumPy fallbacks.  Implements per-group codebook encoding
and decoding with SIMD-vectorised argmin search.

Reference: Guo et al., "FLUTE: Flexible Lookup Table Engine for LUT-based
Network Inference." CVPR 2024.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from squish.kernels.mojo.mojo_bridge import MojoBridge

__all__ = [
    "FluteLUTMojoConfig",
    "MojoFluteLUT",
]

_bridge = MojoBridge()
_encode_kernel = _bridge.load_kernel("flute_lut_encode")
_decode_kernel = _bridge.load_kernel("flute_lut_decode")


# ── NumPy fallbacks ───────────────────────────────────────────────────────────


def _numpy_encode(weight: np.ndarray, codebook: np.ndarray, gs: int) -> np.ndarray:
    rows, cols = weight.shape
    codes = np.zeros((rows, cols), dtype=np.uint8)
    for g in range(codebook.shape[0]):
        cs = g * gs
        ce = min(cs + gs, cols)
        cb = codebook[g]
        for c in range(cs, ce):
            codes[:, c] = np.abs(weight[:, c:c+1] - cb[np.newaxis, :]).argmin(axis=1)
    return codes


def _numpy_decode(codes: np.ndarray, codebook: np.ndarray, gs: int) -> np.ndarray:
    rows, cols = codes.shape
    out = np.zeros((rows, cols), dtype=np.float32)
    for g in range(codebook.shape[0]):
        cs = g * gs
        ce = min(cs + gs, cols)
        cb = codebook[g]
        out[:, cs:ce] = cb[codes[:, cs:ce].astype(int)]
    return out


# ── Config / wrapper ──────────────────────────────────────────────────────────


@dataclass
class FluteLUTMojoConfig:
    """Configuration for :class:`MojoFluteLUT`.

    Attributes:
        group_size: Number of weight columns per codebook group.
        cb_size:    Number of entries in each codebook group.
    """

    group_size: int = 128
    cb_size: int = 16


class MojoFluteLUT:
    """Mojo-backed FLUTE per-group LUT quantization encoder/decoder.

    Provides SIMD-vectorised nearest-codebook-entry search for encoding and
    gather-based decode.  ``parallelize`` over column groups.
    Falls back to NumPy when the Mojo runtime is absent.
    """

    def __init__(self, config: Optional[FluteLUTMojoConfig] = None) -> None:
        self._cfg = config or FluteLUTMojoConfig()

    def encode(
        self,
        weight: np.ndarray,
        codebook: np.ndarray,
        group_size: Optional[int] = None,
    ) -> np.ndarray:
        """Encode weights to per-group LUT code indices.

        Args:
            weight:     ``(rows, cols)`` float32.
            codebook:   ``(n_groups, cb_size)`` float32.
            group_size: Column group size (overrides config).

        Returns:
            ``(rows, cols)`` uint8 code indices.
        """
        w = np.ascontiguousarray(weight, dtype=np.float32)
        cb = np.ascontiguousarray(codebook, dtype=np.float32)
        gs = int(group_size) if group_size is not None else self._cfg.group_size
        if _encode_kernel is not None:
            out = np.zeros(w.shape, dtype=np.uint8)
            _encode_kernel(
                w.ctypes.data, cb.ctypes.data, out.ctypes.data,
                w.shape[0], w.shape[1], cb.shape[0], cb.shape[1], gs,
            )
            return out
        return _numpy_encode(w, cb, gs)

    def decode(
        self,
        codes: np.ndarray,
        codebook: np.ndarray,
        group_size: Optional[int] = None,
    ) -> np.ndarray:
        """Decode LUT code indices to float32 weights.

        Args:
            codes:      ``(rows, cols)`` uint8 code indices.
            codebook:   ``(n_groups, cb_size)`` float32.
            group_size: Column group size (overrides config).

        Returns:
            ``(rows, cols)`` float32 reconstructed weights.
        """
        c = np.ascontiguousarray(codes, dtype=np.uint8)
        cb = np.ascontiguousarray(codebook, dtype=np.float32)
        gs = int(group_size) if group_size is not None else self._cfg.group_size
        if _decode_kernel is not None:
            out = np.zeros(c.shape, dtype=np.float32)
            _decode_kernel(
                c.ctypes.data, cb.ctypes.data, out.ctypes.data,
                c.shape[0], c.shape[1], cb.shape[0], cb.shape[1], gs,
            )
            return out
        return _numpy_decode(c, cb, gs)

    def roundtrip_error(
        self,
        weight: np.ndarray,
        codebook: np.ndarray,
        group_size: Optional[int] = None,
    ) -> float:
        """Mean absolute error of encode→decode roundtrip.

        Args:
            weight:     ``(rows, cols)`` float32.
            codebook:   ``(n_groups, cb_size)`` float32.
            group_size: Column group size (overrides config).

        Returns:
            Scalar float MAE.
        """
        codes = self.encode(weight, codebook, group_size=group_size)
        reconstructed = self.decode(codes, codebook, group_size=group_size)
        return float(np.abs(weight.astype(np.float32) - reconstructed).mean())

    def backend(self) -> str:
        if _encode_kernel is not None or _decode_kernel is not None:
            return "mojo"
        return "numpy"
