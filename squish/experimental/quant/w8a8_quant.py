"""W8A8QuantRuntime — dual INT8 weight-and-activation quantisation.

Implements the W8A8 (Weight INT8 + Activation INT8) inference path described
in the TensorRT-LLM and vLLM production references (2024).

On NVIDIA Ampere (A100) and Hopper (H100) GPUs the ``INT8 GEMM`` instruction
(``I8DP4A`` / ``IMMA``) runs at approximately **2× the throughput** of FP16
GEMM, because each 32-bit accumulator lane holds four INT8 multiply-adds.
W8A8 unlocks this instruction whereas weight-only INT4 (W4A16) still performs
FP16 matmuls and only benefits from reduced memory bandwidth.

This module provides:

* :class:`W8A8Config` — configuration (per-tensor / per-channel scales,
  zero-point, symmetric vs asymmetric).
* :class:`W8A8Tensor` — compressed weight container.
* :class:`W8A8QuantRuntime` — encode weights, quantise activations, and
  perform INT8 matmul with float32 simulation on all platforms.

The simulation path uses ``numpy.int32`` accumulation to exactly replicate
the INT8 GEMM semantics.

Reference:
    NVIDIA TRT-LLM W8A8 kernel reference (2024);
    vLLM production INT8 serving (2024).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

__all__ = [
    "W8A8Config",
    "W8A8Tensor",
    "W8A8QuantRuntime",
]

# ── Config ────────────────────────────────────────────────────────────────────


@dataclass
class W8A8Config:
    """Configuration for W8A8QuantRuntime.

    Attributes:
        per_channel_weight: If True, use one scale per output channel for
            weights (more accurate).  If False, use per-tensor scaling.
        symmetric: If True, use symmetric quantisation (zero-point = 0).
            If False, use asymmetric quantisation with a per-scale zero-point.
        weight_bits: Bit-width for weight quantisation (8 or 4).
        act_bits: Bit-width for activation quantisation (must be 8).
        epsilon: Small constant to prevent division by zero in scale calc.
    """

    per_channel_weight: bool = True
    symmetric: bool = True
    weight_bits: int = 8
    act_bits: int = 8
    epsilon: float = 1e-5

    def __post_init__(self) -> None:
        if self.weight_bits not in (4, 8):
            raise ValueError(
                f"weight_bits must be 4 or 8; got {self.weight_bits}"
            )
        if self.act_bits != 8:
            raise ValueError(
                f"act_bits must be 8 for W8A8; got {self.act_bits}"
            )
        if self.epsilon <= 0:
            raise ValueError(f"epsilon must be positive; got {self.epsilon}")


# ── Compressed tensor ─────────────────────────────────────────────────────────


@dataclass
class W8A8Tensor:
    """W8A8-quantised weight tensor.

    Attributes:
        codes: ``np.int8`` weight codes of shape ``(C_out, C_in)``.
        scale: Per-channel scale factors of shape ``(C_out,)`` or scalar.
        zero: Per-channel zero-points (int8) or scalar 0 for symmetric.
        shape: Original float32 weight shape.
        config: The :class:`W8A8Config` used.
    """

    codes: np.ndarray
    scale: np.ndarray
    zero: np.ndarray
    shape: tuple[int, ...]
    config: W8A8Config


# ── Main class ────────────────────────────────────────────────────────────────


class W8A8QuantRuntime:
    """Dual INT8 weight + activation quantisation runtime.

    Example::

        cfg     = W8A8Config(per_channel_weight=True, symmetric=True)
        runtime = W8A8QuantRuntime(cfg)
        qtensor = runtime.quantise_weight(weight)        # one-time
        out     = runtime.linear(x, qtensor)             # per-inference
        # Equivalent full-precision reference:
        ref     = x @ weight.T

    Args:
        config: :class:`W8A8Config` (optional; defaults to per-channel symmetric INT8).
    """

    def __init__(self, config: Optional[W8A8Config] = None) -> None:
        self.config: W8A8Config = config or W8A8Config()

    # ── Weight quantisation ───────────────────────────────────────────────────

    def quantise_weight(self, weight: np.ndarray) -> W8A8Tensor:
        """Quantise a float32 weight matrix to INT8.

        Args:
            weight: ``(C_out, C_in)`` float32 weight.

        Returns:
            :class:`W8A8Tensor`.

        Raises:
            ValueError: If weight is not 2-D.
        """
        W = np.asarray(weight, dtype=np.float32)
        if W.ndim != 2:
            raise ValueError(f"weight must be 2-D; got {W.shape}")

        cfg = self.config
        qmax = float(2 ** (cfg.weight_bits - 1) - 1)
        eps = cfg.epsilon

        if cfg.per_channel_weight:
            if cfg.symmetric:
                scale = (np.abs(W).max(axis=1) + eps) / qmax  # (C_out,)
                zero = np.zeros(W.shape[0], dtype=np.int8)
                codes = np.clip(
                    np.round(W / scale[:, np.newaxis]), -qmax - 1, qmax
                ).astype(np.int8)
            else:
                w_min = W.min(axis=1)  # (C_out,)
                w_max = W.max(axis=1)
                span = np.maximum(w_max - w_min, eps)
                levels = 2 ** cfg.weight_bits - 1
                scale = span / levels
                zero = (-np.round(w_min / scale)).astype(np.int8)
                codes = np.clip(
                    np.round(W / scale[:, np.newaxis]) + zero[:, np.newaxis],
                    0, levels,
                ).astype(np.int8)
        else:
            if cfg.symmetric:
                amax = float(np.abs(W).max()) + eps
                scale = np.float32(amax / qmax)
                zero = np.int8(0)
                codes = np.clip(np.round(W / scale), -qmax - 1, qmax).astype(np.int8)
            else:
                w_min, w_max = float(W.min()), float(W.max())
                levels = 2 ** cfg.weight_bits - 1
                scale = np.float32((w_max - w_min + eps) / levels)
                zero = np.int32(int(np.clip(np.round(-w_min / scale), 0, 255)))
                codes = np.clip(np.round(W / scale) + int(zero), 0, levels).astype(np.int8)

        return W8A8Tensor(
            codes=codes,
            scale=np.atleast_1d(np.asarray(scale, dtype=np.float32)),
            zero=np.atleast_1d(np.asarray(zero, dtype=np.int32)),
            shape=W.shape,
            config=cfg,
        )

    # ── Activation quantisation ───────────────────────────────────────────────

    def quantise_activation(
        self, x: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Quantise activations to INT8 (per-tensor, symmetric).

        Args:
            x: Float32 activation of any shape.

        Returns:
            ``(x_int8, scale)`` where ``x_int8`` is ``np.int8`` and
            ``scale`` is a float32 scalar.
        """
        x = np.asarray(x, dtype=np.float32)
        qmax = 127.0
        amax = float(np.abs(x).max()) + self.config.epsilon
        scale = np.float32(amax / qmax)
        x_q = np.clip(np.round(x / scale), -128, 127).astype(np.int8)
        return x_q, scale

    # ── INT8 linear ───────────────────────────────────────────────────────────

    def linear(
        self,
        x: np.ndarray,
        qtensor: W8A8Tensor,
        bias: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Perform INT8 weight + INT8 activation matmul.

        Args:
            x: Float32 ``(..., C_in)`` activation.
            qtensor: Quantised weight from :meth:`quantise_weight`.
            bias: Optional float32 ``(C_out,)`` bias vector.

        Returns:
            Float32 ``(..., C_out)`` output.
        """
        x = np.asarray(x, dtype=np.float32)
        x_q, act_scale = self.quantise_activation(x)

        batch_shape = x.shape[:-1]
        flat = x_q.astype(np.int32).reshape(-1, x.shape[-1])  # (N, C_in)
        W_q = qtensor.codes.astype(np.int32)                   # (C_out, C_in)

        # INT8 GEMM → int32 accumulator
        out_int32 = flat @ W_q.T  # (N, C_out)

        # De-quantise
        w_scale = qtensor.scale  # (C_out,) or scalar
        w_zero = qtensor.zero.astype(np.float32)

        out = out_int32.astype(np.float32)
        cfg = self.config
        if cfg.per_channel_weight and cfg.symmetric:
            out = out * (act_scale * w_scale[np.newaxis, :])
        elif cfg.per_channel_weight and not cfg.symmetric:
            out = (out - flat.sum(axis=-1, keepdims=True) * w_zero[np.newaxis, :]) * (
                act_scale * w_scale[np.newaxis, :]
            )
        else:
            scalar_scale = float(w_scale[0]) if w_scale.ndim > 0 else float(w_scale)
            out = out * (act_scale * scalar_scale)

        out = out.reshape(batch_shape + (qtensor.shape[0],))
        if bias is not None:
            out = out + np.asarray(bias, dtype=np.float32)
        return out

    def relative_error(
        self, reference: np.ndarray, output: np.ndarray
    ) -> float:
        """Relative L2 error between reference output and INT8-quantized output.

        Args:
            reference: Float32 reference (full-precision matmul).
            output: Float32 output from :meth:`linear`.

        Returns:
            Relative error scalar.
        """
        ref = np.asarray(reference, dtype=np.float32)
        out = np.asarray(output, dtype=np.float32)
        norm = float(np.linalg.norm(ref))
        if norm == 0:
            return 0.0
        return float(np.linalg.norm(ref - out)) / norm

    def __repr__(self) -> str:
        return (
            f"W8A8QuantRuntime(per_channel={self.config.per_channel_weight}, "
            f"symmetric={self.config.symmetric}, "
            f"weight_bits={self.config.weight_bits})"
        )
