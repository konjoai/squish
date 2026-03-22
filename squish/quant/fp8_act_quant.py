"""squish/quant/fp8_act_quant.py

FP8ActQuant — W8A8 FP8 E4M3/E5M2 Dynamic Activation Quantization.

Reference
---------
Micikevicius et al. "FP8 Formats for Deep Learning." arXiv:2209.05433.
Transformer Engine (NVIDIA) FP8 training/inference framework.

Algorithm
---------
W8A8 FP8 uses:
* Weights stored in FP8-E4M3 (4 exponent, 3 mantissa bits — higher precision,
  range ±448).
* Activations quantized dynamically per-tensor in FP8-E4M3 or FP8-E5M2
  (5 exp, 2 mantissa — wider range ±57344, used for gradients).

Per-tensor dynamic scaling:
  scale = max(|x|) / fp8_max
  q = round_to_fp8(x / scale) * scale

This module provides a NumPy simulation:
* ``quantize_weights(W)`` quantizes weights to FP8-E4M3.
* ``quantize_activations(x)`` quantizes activations to the configured format.
* ``forward(x, W)`` performs W8A8 fused matmul (simulated in FP32).

Key properties
--------------
* No real FP8 hardware intrinsics — pure NumPy simulation.
* ``FP8Format.E4M3`` and ``FP8Format.E5M2`` enumeration.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

import numpy as np

__all__ = [
    "FP8Format",
    "FP8ActQuantConfig",
    "FP8ActQuantResult",
    "FP8ActQuant",
]


class FP8Format(str, Enum):
    E4M3 = "e4m3"
    E5M2 = "e5m2"


# Maximum representable value for each format
_FP8_MAX: dict = {
    FP8Format.E4M3: 448.0,
    FP8Format.E5M2: 57344.0,
}


@dataclass
class FP8ActQuantConfig:
    """Configuration for :class:`FP8ActQuant`.

    Attributes:
        weight_format: FP8 format for weights (default E4M3).
        activation_format: FP8 format for activations (default E4M3).
        stochastic_rounding: Use stochastic rounding (adds ±0.5 ULP noise).
    """

    weight_format: FP8Format = FP8Format.E4M3
    activation_format: FP8Format = FP8Format.E4M3
    stochastic_rounding: bool = False


@dataclass
class FP8ActQuantResult:
    """Quantized tensor result.

    Attributes:
        q_data: FP8-quantized values stored as float32 (simulated).
        scale: Scalar dequantization scale.
        fmt: FP8 format used.
    """

    q_data: np.ndarray
    scale: float
    fmt: FP8Format

    def dequantize(self) -> np.ndarray:
        return self.q_data * self.scale


class FP8ActQuant:
    """FP8 E4M3/E5M2 W8A8 quantizer.

    Parameters
    ----------
    config:
        FP8ActQuantConfig.
    """

    def __init__(self, config: Optional[FP8ActQuantConfig] = None) -> None:
        self._cfg = config or FP8ActQuantConfig()
        self._rng = np.random.default_rng(0)

    @property
    def config(self) -> FP8ActQuantConfig:
        return self._cfg

    def _quantize(self, x: np.ndarray, fmt: FP8Format) -> FP8ActQuantResult:
        fp8_max = _FP8_MAX[fmt]
        abs_max = float(np.abs(x).max())
        scale = abs_max / fp8_max if abs_max > 0 else 1.0
        x_scaled = np.asarray(x, dtype=np.float32) / scale
        if self._cfg.stochastic_rounding:
            noise = self._rng.uniform(-0.5, 0.5, size=x_scaled.shape).astype(np.float32)
            x_scaled = x_scaled + noise
        q = np.clip(np.round(x_scaled), -fp8_max, fp8_max).astype(np.float32)
        return FP8ActQuantResult(q_data=q, scale=scale, fmt=fmt)

    def quantize_weights(self, W: np.ndarray) -> FP8ActQuantResult:
        """Quantize weight matrix to FP8-E4M3."""
        return self._quantize(W, self._cfg.weight_format)

    def quantize_activations(self, x: np.ndarray) -> FP8ActQuantResult:
        """Quantize activation tensor with dynamic per-tensor scale."""
        return self._quantize(x, self._cfg.activation_format)

    def forward(
        self,
        x: np.ndarray,
        W: np.ndarray,
        bias: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, FP8ActQuantResult, FP8ActQuantResult]:
        """W8A8 FP8 linear forward pass (simulated).

        Parameters
        ----------
        x:
            Input activations, shape ``(batch, in_features)``.
        W:
            Weight matrix, shape ``(out_features, in_features)``.
        bias:
            Optional bias, shape ``(out_features,)``.

        Returns
        -------
        (output, q_x, q_W)
        """
        q_x = self.quantize_activations(x)
        q_W = self.quantize_weights(W)
        out = q_x.dequantize() @ q_W.dequantize().T
        if bias is not None:
            out = out + bias[None, :]
        return out, q_x, q_W
