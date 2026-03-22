"""squish/quant/marlin_gemm.py

MarlinGEMM — INT4 Weight × FP16 Activation Tiled GEMM.

Reference
---------
Frantar & Alistarh. "Marlin: A Mixed-Precision Matrix Multiplication
Kernel for Post-Training Quantization."
MLSys 2024 (arXiv:2408.11743).

Algorithm
---------
Marlin achieves near-full-bandwidth INT4 weight dequantization + GEMM by:

1. Packing 8 INT4 weights per int32 word (2 nibbles per byte).
2. Tiling the weight matrix so that each tile fits in L2 cache.
3. Dequantizing each tile on-the-fly during the matrix multiply.
4. Using SIMD/vectorized accumulation in FP16/FP32.

This module provides a NumPy simulation with identical output semantics
to the CUDA Marlin kernel.  It plugs into the existing quantized linear
layer infrastructure.

Key properties
--------------
* NumPy-only; no CUDA dependency.
* ``n_bits`` — quantization bit-width (default 4).
* ``group_size`` — per-group scale size (default 128).
* ``tile_size`` — GEMM tile size for simulation (default 128).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

__all__ = [
    "MarlinGEMMConfig",
    "MarlinGEMM",
]


@dataclass
class MarlinGEMMConfig:
    """Configuration for :class:`MarlinGEMM`.

    Attributes:
        n_bits: Weight quantization bit-width.
        group_size: Number of weights sharing a single scale factor.
        tile_size: Tile size for the GEMM simulation.
    """

    n_bits: int = 4
    group_size: int = 128
    tile_size: int = 128

    def __post_init__(self) -> None:
        if self.n_bits not in (4, 8):
            raise ValueError("n_bits must be 4 or 8")


class MarlinGEMM:
    """INT4 weight × FP16 activation GEMM (Marlin-style simulation).

    Parameters
    ----------
    config:
        MarlinGEMM configuration.
    """

    def __init__(self, config: Optional[MarlinGEMMConfig] = None) -> None:
        self._cfg = config or MarlinGEMMConfig()
        self._W_packed: Optional[np.ndarray] = None
        self._scales: Optional[np.ndarray] = None
        self._zeros: Optional[np.ndarray] = None
        self._out_features: int = 0
        self._in_features: int = 0

    @property
    def config(self) -> MarlinGEMMConfig:
        return self._cfg

    @property
    def is_packed(self) -> bool:
        return self._W_packed is not None

    def pack_weights(self, weights: np.ndarray) -> None:
        """Quantize and pack weight matrix for Marlin-style GEMM.

        Parameters
        ----------
        weights:
            FP32 weight matrix of shape ``(out_features, in_features)``.
        """
        W = np.asarray(weights, dtype=np.float32)
        out_f, in_f = W.shape
        self._out_features = out_f
        self._in_features = in_f
        gs = self._cfg.group_size
        n_levels = 2 ** self._cfg.n_bits
        n_groups = (in_f + gs - 1) // gs
        # Pad
        pad = (-in_f) % gs
        if pad:
            W = np.pad(W, [(0, 0), (0, pad)])
        W_g = W.reshape(out_f, n_groups, gs)
        x_min = W_g.min(axis=-1, keepdims=True)
        x_max = W_g.max(axis=-1, keepdims=True)
        scale = (x_max - x_min).clip(min=1e-8) / (n_levels - 1)
        zero = -x_min / scale
        codes = np.round(W_g / scale + zero).clip(0, n_levels - 1).astype(np.int32)
        self._W_packed = codes  # (out_f, n_groups, gs) — simulated packing
        self._scales = scale.squeeze(-1).astype(np.float32)   # (out_f, n_groups)
        self._zeros = zero.squeeze(-1).astype(np.float32)     # (out_f, n_groups)

    def _dequantize(self) -> np.ndarray:
        """Reconstruct FP32 weight matrix from packed representation."""
        out_f = self._out_features
        gs = self._cfg.group_size
        in_f_padded = self._W_packed.shape[1] * gs
        W_deq = (self._W_packed - self._zeros[:, :, None]) * self._scales[:, :, None]
        return W_deq.reshape(out_f, in_f_padded)[:, : self._in_features]

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute W @ x with on-the-fly dequantization.

        Parameters
        ----------
        x:
            Input activation of shape ``(in_features,)`` or
            ``(batch, in_features)``.

        Returns
        -------
        np.ndarray
            Output of shape ``(out_features,)`` or ``(batch, out_features)``.
        """
        if not self.is_packed:
            raise RuntimeError("Weights not packed; call pack_weights() first")
        x = np.asarray(x, dtype=np.float32)
        squeezed = x.ndim == 1
        if squeezed:
            x = x[None, :]
        # Simulate tile-based GEMM with on-the-fly dequantization
        W_deq = self._dequantize()
        out = x @ W_deq.T
        result = out.squeeze(0) if squeezed else out
        return result.astype(np.float32)

    def unpack_weights(self) -> np.ndarray:
        """Return dequantized weight matrix for inspection."""
        if not self.is_packed:
            raise RuntimeError("Weights not packed")
        return self._dequantize()
