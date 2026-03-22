"""squish/kernels/fused_bias_gelu.py

FusedBiasGELU — Fused Bias-Add + GELU Activation Kernel (NumPy reference).

Reference
---------
Hendrycks & Gimpel. "Gaussian Error Linear Units (GELUs)." arXiv:1606.08415.
Megatron-LM fused kernels:
  github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/fusions/fused_bias_gelu.py

Algorithm
---------
The GELU activation is computed as:
  GELU(x) = x · Φ(x) = x · 0.5 · (1 + erf(x / √2))

The *fast* approximation (used by Megatron and GPT-2) is:
  GELU_approx(x) = 0.5 · x · (1 + tanh(√(2/π) · (x + 0.044715 · x^3)))

Fusing bias-add and GELU into a single kernel avoids a write-back to memory
after the linear layer and reduces DRAM bandwidth.

This module provides:
* ``FusedBiasGELU.forward(x, bias)`` — fused bias + GELU (or exact).
* ``FusedBiasGELU.backward(grad_out, x, bias)`` → (grad_x, grad_bias).
* Both exact (erf-based) and fast (tanh-based) approximation modes.

Key properties
--------------
* NumPy-only.
* ``approximate=True`` uses the tanh approximation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

__all__ = [
    "FusedBiasGELUConfig",
    "FusedBiasGELU",
]

_SQRT_2 = float(np.sqrt(2.0))
_SQRT_2_OVER_PI = float(np.sqrt(2.0 / np.pi))
_GELU_COEFF = 0.044715


@dataclass
class FusedBiasGELUConfig:
    """Configuration for :class:`FusedBiasGELU`.

    Attributes:
        approximate: Use tanh approximation (faster, slightly less accurate).
    """

    approximate: bool = True


class FusedBiasGELU:
    """Fused bias-add + GELU activation.

    Parameters
    ----------
    config:
        FusedBiasGELUConfig.
    """

    def __init__(self, config: Optional[FusedBiasGELUConfig] = None) -> None:
        self._cfg = config or FusedBiasGELUConfig()

    @property
    def config(self) -> FusedBiasGELUConfig:
        return self._cfg

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: np.ndarray, bias: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute fused bias-add + GELU.

        Parameters
        ----------
        x:
            Input tensor (any shape), typically ``(batch, features)``.
        bias:
            Optional bias, shape ``(features,)`` or broadcastable.

        Returns
        -------
        GELU(x + bias) of same shape as ``x``.
        """
        out = np.asarray(x, dtype=np.float32)
        if bias is not None:
            out = out + np.asarray(bias, dtype=np.float32)
        return self._gelu(out)

    def backward(
        self,
        grad_out: np.ndarray,
        x: np.ndarray,
        bias: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Compute gradients through fused bias-add + GELU.

        Parameters
        ----------
        grad_out:
            Upstream gradient, same shape as forward output.
        x:
            Original pre-activation input.
        bias:
            Bias used in the forward pass (may be None).

        Returns
        -------
        (grad_x, grad_bias) — grad_bias is None if no bias was used.
        """
        z = np.asarray(x, dtype=np.float32)
        if bias is not None:
            z = z + np.asarray(bias, dtype=np.float32)
        dgelu = self._gelu_grad(z)
        grad_x = grad_out * dgelu
        grad_bias = grad_x.sum(axis=0) if bias is not None else None
        return grad_x, grad_bias

    # ------------------------------------------------------------------
    # GELU implementations
    # ------------------------------------------------------------------

    def _gelu(self, x: np.ndarray) -> np.ndarray:
        if self._cfg.approximate:
            return self._gelu_tanh(x)
        return self._gelu_exact(x)

    def _gelu_grad(self, x: np.ndarray) -> np.ndarray:
        if self._cfg.approximate:
            return self._gelu_tanh_grad(x)
        return self._gelu_exact_grad(x)

    @staticmethod
    def _gelu_exact(x: np.ndarray) -> np.ndarray:
        from math import erf
        return x * 0.5 * (1.0 + np.vectorize(erf)(x / _SQRT_2))

    @staticmethod
    def _gelu_exact_grad(x: np.ndarray) -> np.ndarray:
        phi = 0.5 * (1.0 + np.vectorize(lambda v: float(__import__('math').erf(v / _SQRT_2)))(x))
        pdf = (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * x ** 2)
        return phi + x * pdf

    @staticmethod
    def _gelu_tanh(x: np.ndarray) -> np.ndarray:
        inner = _SQRT_2_OVER_PI * (x + _GELU_COEFF * x ** 3)
        return 0.5 * x * (1.0 + np.tanh(inner))

    @staticmethod
    def _gelu_tanh_grad(x: np.ndarray) -> np.ndarray:
        inner = _SQRT_2_OVER_PI * (x + _GELU_COEFF * x ** 3)
        tanh_val = np.tanh(inner)
        sech2 = 1.0 - tanh_val ** 2
        d_inner = _SQRT_2_OVER_PI * (1.0 + 3.0 * _GELU_COEFF * x ** 2)
        return 0.5 * (1.0 + tanh_val) + 0.5 * x * sech2 * d_inner
