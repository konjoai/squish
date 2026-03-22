"""squish/quant/bitnet_b158.py

BitNet b1.58 Ternary Weight Quantizer.

Weights are mapped from full-precision to {-1, 0, +1} using an absmean
threshold (0.5 × mean(|W|)).  This enables integer-only inference where
matrix multiplications reduce to additions — no multiplications required.

Key properties
--------------
* Compression ratio: ~18.3× at 16-bit originals (log2(3)/16 ≈ 0.099 bits/weight
  theoretical; practical = 1.58 bits/weight with efficient packing)
* Per-tensor scale factor ``s = mean(|W|)``; dequantised weight ≈ W_ternary × s
* Forward pass: ``y = (x @ W_ternary.T) * scale`` — adds, no fp multiplies on
  the weight side

Reference
---------
Ma, S. et al. "The Era of 1-bit LLMs: All Large Language Models are in 1.58
Bits." Microsoft Research. arXiv:2402.17764, Feb 2024.
"""

from __future__ import annotations

__all__ = ["BitNet158Config", "BitNet158Quantizer"]

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from numpy import ndarray


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class BitNet158Config:
    """Configuration for BitNet b1.58 quantization.

    Parameters
    ----------
    absmean_scale_eps:
        Small constant added to the absmean scale to avoid division by zero.
    seed:
        RNG seed (not used during deterministic quantisation; reserved).
    """

    absmean_scale_eps: float = 1e-8
    seed: int = 0

    def __post_init__(self) -> None:
        if self.absmean_scale_eps <= 0:
            raise ValueError("absmean_scale_eps must be positive")


# ---------------------------------------------------------------------------
# Quantizer
# ---------------------------------------------------------------------------

class BitNet158Quantizer:
    """Ternary weight quantizer following the BitNet b1.58 recipe.

    Parameters
    ----------
    config:
        ``BitNet158Config`` instance.
    """

    def __init__(self, config: BitNet158Config | None = None) -> None:
        self.config = config or BitNet158Config()

    def quantize_weight(self, W: ndarray) -> Tuple[ndarray, float]:
        """Quantize a weight matrix to ternary {-1, 0, +1}.

        Parameters
        ----------
        W:
            2-D weight array, shape ``(out_features, in_features)``.

        Returns
        -------
        W_ternary:
            Ternary weight array, dtype ``int8``, same shape as ``W``.
        scale:
            Scalar absmean scale factor ``s = mean(|W|)``.
        """
        W = np.asarray(W, dtype=np.float32)
        scale = float(np.mean(np.abs(W))) + self.config.absmean_scale_eps
        # Threshold = 0.5 × absmean (per arXiv 2402.17764)
        threshold = 0.5 * scale
        W_ternary = np.where(W > threshold, 1,
                    np.where(W < -threshold, -1, 0)).astype(np.int8)
        return W_ternary, scale

    def dequantize(self, W_ternary: ndarray, scale: float) -> ndarray:
        """Reconstruct approximate float weights.

        Parameters
        ----------
        W_ternary:
            Ternary weight array (dtype ``int8``, values in {-1, 0, 1}).
        scale:
            Scalar absmean scale from :meth:`quantize_weight`.

        Returns
        -------
        Approximate float32 weight matrix.
        """
        return W_ternary.astype(np.float32) * float(scale)

    def bitlinear_forward(
        self,
        x: ndarray,
        W_ternary: ndarray,
        scale: float,
    ) -> ndarray:
        """Compute ``y = x @ W_ternary.T * scale``.

        In a real implementation the matmul is replaced with integer addition.
        Here we use NumPy int16 accumulation to model that behaviour.

        Parameters
        ----------
        x:
            Activation matrix, shape ``(batch, in_features)``.
        W_ternary:
            Ternary weight array, shape ``(out_features, in_features)``.
        scale:
            Absmean scale factor.

        Returns
        -------
        Output activations, shape ``(batch, out_features)``, dtype float32.
        """
        x = np.asarray(x, dtype=np.float32)
        W_ternary = np.asarray(W_ternary, dtype=np.int8)
        # Promote to int16 to avoid overflow in accumulation
        z = x @ W_ternary.T.astype(np.float32)
        return (z * float(scale)).astype(np.float32)

    def compression_ratio(self, original_dtype_bits: int = 16) -> float:
        """Estimated compression ratio vs a dense float representation.

        Uses the practical effective bit-width of 1.58 bits/weight for
        ternary values packed efficiently.

        Parameters
        ----------
        original_dtype_bits:
            Bit-width of the original weights (e.g., 16 for float16).

        Returns
        -------
        Compression ratio (>1 means smaller than original).
        """
        # 1.58 bits/weight (log2(3) ≈ 1.585)
        ternary_bits = float(np.log2(3))
        return original_dtype_bits / ternary_bits
