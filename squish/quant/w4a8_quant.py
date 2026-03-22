"""squish/quant/w4a8_quant.py

W4A8QuantRuntime — W4 weight + A8 activation hybrid quantization.

Reference
---------
Lin et al. "QServe: W4A8KV4 Quantization and System Co-design for Efficient
LLM Serving." NeurIPS 2024. arXiv:2405.04532.

Algorithm
---------
Weight path (W4):
  - Per-group INT4 quantization with group_size=128 (same as AWQ).
  - Progressive scaling: compute per-group scale from max(|W_group|) / 7.
  - Packed as uint8 nibbles (two INT4s per byte).

Activation path (A8):
  - Per-tensor dynamic INT8 quantization: scale = max(|X|) / 127.
  - Quantize, compute integer GEMM, dequantize result.

Combined GEMM simulation (NumPy float32):
  - Dequantize W to float32 on-the-fly (per-group scale expansion).
  - Dequantize A to float32 (per-tensor scale).
  - Execute float32 GEMM as simulation of the integer kernel.

This sits between W4A16 (weight-only INT4) and W8A8 in the speed–memory
Pareto frontier. On Ampere/Hopper GPUs it achieves 3× vs TensorRT-LLM W8A8
at ISO-quality.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class W4A8Config:
    """Configuration for W4A8QuantRuntime.

    Parameters
    ----------
    group_size:
        Number of weight elements per quantization group.
    w_bits:
        Weight quantization bits (default 4).
    a_bits:
        Activation quantization bits (default 8).
    symmetric:
        If True, zero-point is fixed to 0 (symmetric INT4/INT8).
    """

    group_size: int = 128
    w_bits: int = 4
    a_bits: int = 8
    symmetric: bool = True

    def __post_init__(self) -> None:
        if self.w_bits not in (1, 2, 3, 4, 8):
            raise ValueError("w_bits must be 1, 2, 3, 4, or 8")
        if self.a_bits not in (4, 8):
            raise ValueError("a_bits must be 4 or 8")


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

@dataclass
class W4WeightResult:
    """Packed INT4 weight result.

    Parameters
    ----------
    W_packed:
        Packed weight bytes, shape ``(out_features, ceil(in_features/2))``.
    scale:
        Per-group scales, shape ``(out_features, n_groups)``.
    zero:
        Per-group zero-points (0 if symmetric), same shape as scale.
    original_shape:
        ``(out_features, in_features)``.
    """

    W_packed: np.ndarray
    scale: np.ndarray
    zero: np.ndarray
    original_shape: tuple[int, int]

    def dequantize(self) -> np.ndarray:
        """Dequantize packed INT4 weights to float32.

        Returns
        -------
        np.ndarray
            Shape ``(out_features, in_features)``.
        """
        out_f, in_f = self.original_shape
        # Unpack nibbles
        packed = self.W_packed.astype(np.uint8)
        lo = (packed & 0x0F).astype(np.float32)           # (out_f, ceil(in_f/2))
        hi = ((packed >> 4) & 0x0F).astype(np.float32)
        # Interleave lo/hi back to in_f width
        W_int = np.empty((out_f, lo.shape[1] + hi.shape[1]), dtype=np.float32)
        W_int[:, 0::2] = lo
        W_int[:, 1::2] = hi
        W_int = W_int[:, :in_f]  # trim padding

        # Expand scale/zero to in_f
        n_groups = self.scale.shape[1]
        gs = in_f // n_groups + (1 if in_f % n_groups else 0)
        scale_full = np.repeat(self.scale, gs, axis=1)[:, :in_f]
        zero_full = np.repeat(self.zero, gs, axis=1)[:, :in_f]

        return (W_int - zero_full) * scale_full


@dataclass
class A8ActivationResult:
    """Dynamic INT8 quantized activation.

    Parameters
    ----------
    X_int8:
        Quantized int8 values.
    scale:
        Per-tensor scale (float32 scalar).
    original_shape:
        Original tensor shape.
    """

    X_int8: np.ndarray
    scale: float
    original_shape: tuple

    def dequantize(self) -> np.ndarray:
        return self.X_int8.astype(np.float32) * self.scale


# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------

class W4A8QuantRuntime:
    """W4 weight + A8 activation quantized linear layer.

    Parameters
    ----------
    config:
        W4A8 configuration.
    """

    def __init__(self, config: Optional[W4A8Config] = None) -> None:
        self._cfg = config or W4A8Config()
        self._w_result: Optional[W4WeightResult] = None

    @property
    def config(self) -> W4A8Config:
        return self._cfg

    # ------------------------------------------------------------------
    # Weight quantization
    # ------------------------------------------------------------------

    def quantize_weight(self, W: np.ndarray) -> W4WeightResult:
        """INT4 weight quantization with per-group scaling.

        Parameters
        ----------
        W:
            Shape ``(out_features, in_features)``.

        Returns
        -------
        W4WeightResult
        """
        W = np.asarray(W, dtype=np.float32)
        out_f, in_f = W.shape
        gs = self._cfg.group_size
        w_max = (1 << self._cfg.w_bits) - 1  # 15 for INT4

        # Pad in_f to multiple of group_size
        pad = (gs - in_f % gs) % gs
        if pad:
            W_pad = np.pad(W, ((0, 0), (0, pad)))
        else:
            W_pad = W

        n_groups = W_pad.shape[1] // gs
        W_groups = W_pad.reshape(out_f, n_groups, gs)

        if self._cfg.symmetric:
            max_vals = np.abs(W_groups).max(axis=2)  # (out_f, n_groups)
            scale = max_vals / (w_max / 2.0)
            scale = np.where(scale == 0, 1.0, scale)
            zero = np.zeros_like(scale)
            W_quant = np.round(W_groups / scale[:, :, np.newaxis]).clip(
                -(w_max // 2 + 1), w_max // 2
            ).astype(np.int8)
            W_quant_u = (W_quant + (w_max // 2 + 1)).astype(np.uint8)
        else:
            min_vals = W_groups.min(axis=2)
            max_vals = W_groups.max(axis=2)
            scale = (max_vals - min_vals) / w_max
            scale = np.where(scale == 0, 1.0, scale)
            zero = np.round(-min_vals / scale).clip(0, w_max).astype(np.float32)
            W_quant_u = np.round(
                W_groups / scale[:, :, np.newaxis] + zero[:, :, np.newaxis]
            ).clip(0, w_max).astype(np.uint8)

        W_flat = W_quant_u.reshape(out_f, -1)  # (out_f, n_groups*gs)
        W_flat = W_flat[:, :in_f]              # trim back to in_f

        # Pack two nibbles per byte
        if in_f % 2 == 1:
            W_flat = np.pad(W_flat, ((0, 0), (0, 1)))
        lo = W_flat[:, 0::2] & 0x0F
        hi = (W_flat[:, 1::2] & 0x0F) << 4
        W_packed = (lo | hi).astype(np.uint8)

        self._w_result = W4WeightResult(
            W_packed=W_packed,
            scale=scale,
            zero=zero,
            original_shape=(out_f, in_f),
        )
        return self._w_result

    # ------------------------------------------------------------------
    # Activation quantization
    # ------------------------------------------------------------------

    def quantize_activation(self, X: np.ndarray) -> A8ActivationResult:
        """Dynamic INT8 per-tensor activation quantization.

        Parameters
        ----------
        X:
            Input activations ``(batch, in_features)`` or ``(in_features,)``.

        Returns
        -------
        A8ActivationResult
        """
        X = np.asarray(X, dtype=np.float32)
        a_max = (1 << (self._cfg.a_bits - 1)) - 1  # 127 for INT8
        abs_max = float(np.abs(X).max()) + 1e-8
        scale = abs_max / a_max
        X_int = np.round(X / scale).clip(-a_max - 1, a_max).astype(np.int8)
        return A8ActivationResult(X_int8=X_int, scale=scale, original_shape=X.shape)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self, X: np.ndarray, W: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Simulate W4A8 quantized linear layer forward pass.

        Parameters
        ----------
        X:
            Input activations ``(batch, in_features)`` or ``(in_features,)``.
        W:
            Weight matrix ``(out_features, in_features)``.  If None, uses
            the last quantize_weight() result.

        Returns
        -------
        np.ndarray
            Output ``(batch, out_features)`` or ``(out_features,)``.
        """
        if W is not None:
            self.quantize_weight(W)
        if self._w_result is None:
            raise RuntimeError("Call quantize_weight() or pass W before forward().")

        act_result = self.quantize_activation(X)
        W_fp = self._w_result.dequantize()
        X_fp = act_result.dequantize()
        return (X_fp @ W_fp.T).astype(np.float32)
