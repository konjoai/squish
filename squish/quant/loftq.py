"""squish/quant/loftq.py

LoFTQ — LoRA-Fine-Tuning-Aware Quantization.

Reference
---------
Li et al. "LoFTQ: LoRA-Fine-Tuning-Aware Quantization for Large Language
Models." ICLR 2024 (arXiv:2310.08659).

Algorithm
---------
LoFTQ jointly optimizes quantization and LoRA initialization:

1. Quantize weight W to INT-n using standard symmetric quantization.
2. Compute the residual R = W - dequant(quant(W)).
3. Initialize LoRA matrices A, B via SVD of R such that A @ B ≈ R.
4. This gives: W ≈ W_q + A @ B, where A @ B compensates quant error.

At inference, the effective weight is (W_q + A @ B) which converges to
better quality than QLoRA at the same bit-width.

Key properties
--------------
* NumPy-only.
* ``n_bits`` — quantization bit-width (default 4).
* ``rank`` — LoRA adapter rank (default 16).
* ``n_iterations`` — number of alternating optimization rounds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

__all__ = [
    "LoFTQConfig",
    "LoFTQResult",
    "LoFTQ",
]


@dataclass
class LoFTQConfig:
    """Configuration for :class:`LoFTQ`.

    Attributes:
        n_bits: Quantization bit-width.
        rank: LoRA adapter rank.
        n_iterations: Number of alternating quantize / SVD rounds.
    """

    n_bits: int = 4
    rank: int = 16
    n_iterations: int = 5

    def __post_init__(self) -> None:
        if self.n_bits not in (2, 4, 8):
            raise ValueError("n_bits must be 2, 4, or 8")
        if self.rank < 1:
            raise ValueError("rank must be >= 1")


@dataclass
class LoFTQResult:
    """Result of the LoFTQ optimization.

    Attributes:
        W_q: Quantized weight codes, shape ``(out, in)``.
        scale: Per-column dequantisation scale.
        A: LoRA A matrix, shape ``(out, rank)``.
        B: LoRA B matrix, shape ``(rank, in)``.
        residual_norm: Frobenius norm of the final residual (lower = better).
    """

    W_q: np.ndarray
    scale: np.ndarray
    A: np.ndarray
    B: np.ndarray
    residual_norm: float

    def effective_weight(self) -> np.ndarray:
        """Return W_q_deq + A @ B."""
        W_deq = (self.W_q.astype(np.float32) - (2 ** (len(self.scale.shape) - 1))) * self.scale
        return W_deq + self.A @ self.B

    @property
    def lora_A(self) -> np.ndarray:
        """LoRA A matrix (rank, in_features) — same as B field."""
        return self.B

    @property
    def lora_B(self) -> np.ndarray:
        """LoRA B matrix (out_features, rank) — same as A field."""
        return self.A

    def dequantize(self) -> np.ndarray:
        """Reconstruct approximate FP32 weight (alias for effective_weight)."""
        return self.effective_weight()


class LoFTQ:
    """LoRA-fine-tuning-aware quantizer.

    Parameters
    ----------
    config:
        LoFTQ configuration.
    """

    def __init__(self, config: Optional[LoFTQConfig] = None) -> None:
        self._cfg = config or LoFTQConfig()

    @property
    def config(self) -> LoFTQConfig:
        return self._cfg

    def _quantize_symmetric(self, W: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Symmetric per-column INT-n quantization."""
        n_levels = 2 ** self._cfg.n_bits
        half = n_levels // 2
        col_max = np.abs(W).max(axis=0, keepdims=True).clip(min=1e-8)
        scale = col_max / (half - 1)
        codes = np.round(W / scale + half).clip(0, n_levels - 1).astype(np.int32)
        return codes, scale.squeeze(0)

    def _dequantize(self, codes: np.ndarray, scale: np.ndarray) -> np.ndarray:
        half = 2 ** (self._cfg.n_bits - 1)
        return (codes.astype(np.float32) - half) * scale[None, :]

    def quantize(self, weights: np.ndarray) -> LoFTQResult:
        """Run alternating LoFTQ optimization.

        Parameters
        ----------
        weights:
            FP32 weight matrix, shape ``(out_features, in_features)``.

        Returns
        -------
        LoFTQResult
        """
        W = np.asarray(weights, dtype=np.float32)
        rank = min(self._cfg.rank, min(W.shape))

        W_current = W.copy()
        A = np.zeros((W.shape[0], rank), dtype=np.float32)
        B = np.zeros((rank, W.shape[1]), dtype=np.float32)

        for _ in range(self._cfg.n_iterations):
            # Subtract current LoRA approximation
            W_target = W - A @ B
            # Quantize
            codes, scale = self._quantize_symmetric(W_target)
            W_q_deq = self._dequantize(codes, scale)
            # Compute residual and update LoRA via truncated SVD
            residual = W - W_q_deq
            u, s, vt = np.linalg.svd(residual, full_matrices=False)
            A = u[:, :rank] * s[:rank][None, :]
            B = vt[:rank, :]

        final_residual = W - self._dequantize(codes, scale) - A @ B
        residual_norm = float(np.linalg.norm(final_residual, "fro"))

        return LoFTQResult(
            W_q=codes,
            scale=scale,
            A=A,
            B=B,
            residual_norm=residual_norm,
        )
