"""squish/quant/fp4_quant.py

FP4 (E2M1) Weight Quantizer.

FP4 is a 4-bit floating-point format with 1 sign bit, 2 exponent bits, and
1 mantissa bit.  The representable non-zero finite values are:

    ±{0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}

plus the value 0, giving 15 unique finite values.  (± inf and NaN are
excluded, matching the NVIDIA NF4/FP4 spec for inference.)

Usage in practice:
* Adopted in NVIDIA Blackwell (B100/B200) GPU hardware (FP4 tensor cores).
* Supported in mlx >= 0.22 for Apple Silicon.
* Typical accuracy loss: ~0.1-0.2 PPL gap vs FP16 on 7B models.

Reference
---------
NVIDIA Blackwell Architecture Whitepaper, 2024.
arXiv:2310.16836 (MLX FP4 variant).
"""

from __future__ import annotations

__all__ = ["FP4Config", "FP4Quantizer"]

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from numpy import ndarray


# ---------------------------------------------------------------------------
# FP4 E2M1 value table
# ---------------------------------------------------------------------------

# Representable E2M1 finite values (unsigned, 0-indexed)
# Index 0 = 0; indices 1-7 = positive; indices 8-14 = negative
_FP4_POS_VALUES = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)
_FP4_NEG_VALUES = -_FP4_POS_VALUES[1:]  # 7 negative values (excluding 0)
# Full table: index 0 = 0, 1-7 = positive 0.5..6, 8-14 = negative -0.5..-6
_FP4_LOOKUP = np.concatenate([_FP4_POS_VALUES, _FP4_NEG_VALUES])  # 15 values


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class FP4Config:
    """Configuration for FP4Quantizer.

    Parameters
    ----------
    per_channel:
        When ``True``, one scale per output channel (row of W is used).
        When ``False``, one global scale per tensor.
    seed:
        RNG seed (reserved).
    """

    per_channel: bool = True
    seed: int = 0


# ---------------------------------------------------------------------------
# Quantizer
# ---------------------------------------------------------------------------

class FP4Quantizer:
    """FP4 E2M1 weight quantizer.

    Weights are scaled to fit the representable FP4 range then nearest
    E2M1 value is selected.  Only ``uint8`` code indices are stored
    (values 0-14), not the actual 4-bit packed form.

    Parameters
    ----------
    config:
        ``FP4Config`` instance.
    """

    def __init__(self, config: FP4Config | None = None) -> None:
        self.config = config or FP4Config()

    def fp4_values(self) -> ndarray:
        """Return the 15 representable FP4 E2M1 finite values."""
        return _FP4_LOOKUP.copy()

    def quantize(self, W: ndarray) -> Tuple[ndarray, ndarray]:
        """Quantize a weight matrix to FP4 E2M1 code indices.

        Parameters
        ----------
        W:
            2-D float weight matrix, shape ``(rows, cols)``.

        Returns
        -------
        W_q_indices:
            uint8 index array (values 0–14), same shape as ``W``.
        scales:
            Scale factors.  Shape ``(rows,)`` for per-channel; scalar
            array of shape ``(1,)`` for per-tensor.
        """
        W = np.asarray(W, dtype=np.float32)
        max_fp4 = 6.0  # Maximum absolute value in E2M1 table

        if self.config.per_channel:
            abs_max = np.abs(W).max(axis=1, keepdims=True)  # (rows, 1)
            scales = (abs_max.squeeze(axis=1) / max_fp4).astype(np.float32)
            scale_bcast = abs_max / max_fp4
        else:
            abs_max = float(np.abs(W).max())
            scales = np.array([abs_max / max_fp4], dtype=np.float32)
            scale_bcast = float(scales[0])

        # Normalise weights to [-6, +6]
        scale_safe = np.where(
            np.asarray(scale_bcast) > 1e-8, scale_bcast, np.ones_like(scale_bcast)
        )
        W_norm = W / scale_safe

        # Nearest-neighbour lookup against FP4 table
        # Expand dims for broadcasting: (rows, cols, 1) vs (15,)
        W_exp = W_norm[..., np.newaxis]  # (..., 1)
        diffs = np.abs(W_exp - _FP4_LOOKUP)  # (..., 15)
        W_q_indices = np.argmin(diffs, axis=-1).astype(np.uint8)
        return W_q_indices, scales

    def dequantize(self, W_q_indices: ndarray, scales: ndarray) -> ndarray:
        """Reconstruct approximate float weights from FP4 indices.

        Parameters
        ----------
        W_q_indices:
            uint8 index array (values 0–14).
        scales:
            Scale factors from :meth:`quantize`.

        Returns
        -------
        Approximate float32 weight matrix.
        """
        W_q_indices = np.asarray(W_q_indices, dtype=np.uint8)
        fp4_vals = _FP4_LOOKUP[W_q_indices]  # look up fp4 value per index
        scales = np.asarray(scales, dtype=np.float32)

        if scales.ndim == 1 and scales.shape[0] == W_q_indices.shape[0]:
            # per-channel
            return (fp4_vals * scales[:, np.newaxis]).astype(np.float32)
        # per-tensor
        return (fp4_vals * float(scales.flat[0])).astype(np.float32)

    def matmul(
        self,
        x: ndarray,
        W_q_indices: ndarray,
        scales: ndarray,
    ) -> ndarray:
        """Compute ``y = x @ W_dequant.T``.

        Parameters
        ----------
        x:
            Input activations, shape ``(batch, cols)``.

        Returns
        -------
        Output activations, shape ``(batch, rows)``.
        """
        W_float = self.dequantize(W_q_indices, scales)
        return (np.asarray(x, dtype=np.float32) @ W_float.T).astype(np.float32)

    def ppl_gap(self, baseline_ppl: float, quantized_ppl: float) -> float:
        """Compute the perplexity gap between baseline and quantized models.

        Parameters
        ----------
        baseline_ppl:
            Perplexity of the original (float16/bfloat16) model.
        quantized_ppl:
            Perplexity of the FP4-quantized model.

        Returns
        -------
        Absolute PPL degradation (quantized − baseline).  Positive means
        the quantized model is worse; negative means better (unexpected).
        """
        return float(quantized_ppl) - float(baseline_ppl)
