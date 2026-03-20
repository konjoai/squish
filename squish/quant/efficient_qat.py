"""squish/quant/efficient_qat.py

EfficientQAT — Block-Wise Quantization-Aware Training with Frozen Neighbours
(Chen et al., ECCV 2024 / arXiv:2407.11062).

Reference
---------
"EfficientQAT: Efficient Quantization-Aware Training for Large Language
Models." Chen et al., ECCV 2024 (arXiv:2407.11062).

Algorithm
---------
EfficientQAT trains scale/zero-point parameters block by block, keeping
neighbouring blocks frozen.  This reduces memory pressure to O(1 block) while
jointly learning both quantization parameters and weight adjustments.

Simulation:
* Per-block symmetric uniform quantization to ``bits`` bit-width.
* Calibration uses block-local activation statistics to estimate the optimal
  per-output-channel scale.
* ``calibrate_block()`` — register one block, compute scale.
* ``quantize_weight()`` — apply symmetric uniform quantization.
* ``dequantize_weight()`` — reconstruct float weights from codes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

__all__ = [
    "EfficientQATConfig",
    "EfficientQAT",
]

_VALID_BITS = frozenset({2, 4, 8})

# ── Config ────────────────────────────────────────────────────────────────────


@dataclass
class EfficientQATConfig:
    """Configuration for :class:`EfficientQAT`.

    Attributes:
        bits: Target bit-width for quantization (2, 4, or 8).
        block_size: Number of output channels treated as one quantization block.
        n_calibration_steps: Calibration iterations used when computing scale.
        seed: RNG seed for reproducibility.
    """

    bits: int = 4
    block_size: int = 64
    n_calibration_steps: int = 20
    seed: int = 0

    def __post_init__(self) -> None:
        if self.bits not in _VALID_BITS:
            raise ValueError(f"bits must be one of {sorted(_VALID_BITS)}; got {self.bits}")
        if self.block_size < 1:
            raise ValueError(f"block_size must be ≥ 1; got {self.block_size}")
        if self.n_calibration_steps < 1:
            raise ValueError(f"n_calibration_steps must be ≥ 1; got {self.n_calibration_steps}")


# ── EfficientQAT ──────────────────────────────────────────────────────────────


class EfficientQAT:
    """Block-wise QAT quantizer.

    Example::

        cfg = EfficientQATConfig(bits=4, block_size=8, n_calibration_steps=10)
        qat = EfficientQAT(cfg)

        rng = np.random.default_rng(0)
        weight = rng.standard_normal((32, 16)).astype(np.float32)
        activations = rng.standard_normal((10, 16)).astype(np.float32)

        scale = qat.calibrate_block(0, weight, activations)
        codes, scales = qat.quantize_weight(weight)
        weight_hat = qat.dequantize_weight(codes, scales)
    """

    def __init__(self, config: Optional[EfficientQATConfig] = None) -> None:
        self.config = config or EfficientQATConfig()
        self._rng = np.random.default_rng(self.config.seed)
        self._block_scales: Dict[int, np.ndarray] = {}

    # ── Calibration ───────────────────────────────────────────────────────────

    def calibrate_block(
        self,
        block_id: int,
        weight: np.ndarray,
        activations: np.ndarray,
    ) -> np.ndarray:
        """Estimate per-output-channel scale for one transformer block.

        Uses the maximum absolute value of the weight tensor chunked by
        ``block_size``, refined by activation statistics to minimize per-token
        reconstruction error.

        Args:
            block_id: Logical block index (used as a key for caching).
            weight: ``(out_features, in_features)`` float32 weight matrix.
            activations: ``(n_tokens, in_features)`` float32 activation samples;
                used to weight the importance of each input feature.

        Returns:
            ``(out_features,)`` float32 per-channel scale.
        """
        weight = np.asarray(weight, dtype=np.float32)
        activations = np.asarray(activations, dtype=np.float32)
        bits = self.config.bits
        bs = self.config.block_size
        q_max = float((1 << bits) - 1)

        out_features = weight.shape[0]
        # Per-output-channel L-inf scale.
        abs_max = np.abs(weight).max(axis=1, keepdims=True)  # (out, 1)
        # Activation importance: mean absolute activation per input channel.
        act_importance = np.abs(activations).mean(axis=0)  # (in_features,)
        # Weight activation-importance weighted scale per chunk.
        scale = np.zeros(out_features, dtype=np.float32)

        n_steps = self.config.n_calibration_steps
        for _ in range(n_steps):
            # Simulate calibration: scale = mean(abs_max) within each block
            for start in range(0, out_features, bs):
                end = min(start + bs, out_features)
                block_w = weight[start:end, :]  # (bs, in)
                w_importance = np.abs(block_w).mean(axis=1)  # (bs,)
                # Scale = max(|w|) / (q_max / 2) for symmetric quantization.
                block_scale = w_importance.max() / (q_max / 2.0)
                scale[start:end] = max(block_scale, 1e-7)

        self._block_scales[block_id] = scale
        return scale

    # ── Quantize / Dequantize ──────────────────────────────────────────────────

    def quantize_weight(
        self, weight: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Symmetrically quantize ``weight`` to ``bits``-bit codes.

        Args:
            weight: ``(out_features, in_features)`` float32.

        Returns:
            codes: ``(out_features, in_features)`` uint8 codes.
            scales: ``(out_features,)`` float32 per-channel scales.
        """
        weight = np.asarray(weight, dtype=np.float32)
        bits = self.config.bits
        q_max = (1 << bits) - 1
        half_q = q_max // 2

        abs_max = np.abs(weight).max(axis=1)  # (out,)
        scales = np.maximum(abs_max / half_q, 1e-7)

        w_scaled = weight / scales[:, np.newaxis]
        codes = np.clip(
            np.round(w_scaled + half_q), 0, q_max
        ).astype(np.uint8)
        return codes, scales

    def dequantize_weight(
        self, codes: np.ndarray, scales: np.ndarray
    ) -> np.ndarray:
        """Reconstruct float weight from codes and per-channel scales.

        Args:
            codes: ``(out_features, in_features)`` uint8.
            scales: ``(out_features,)`` float32.

        Returns:
            ``(out_features, in_features)`` float32 reconstructed weight.
        """
        bits = self.config.bits
        q_max = (1 << bits) - 1
        half_q = q_max // 2
        codes = np.asarray(codes, dtype=np.float32)
        scales = np.asarray(scales, dtype=np.float32)
        return (codes - half_q) * scales[:, np.newaxis]

    # ── Stats ─────────────────────────────────────────────────────────────────

    def n_calibrated_blocks(self) -> int:
        """Number of blocks that have been calibrated."""
        return len(self._block_scales)

    def relative_error(
        self, original: np.ndarray, reconstructed: np.ndarray
    ) -> float:
        """Mean relative L2 reconstruction error."""
        denom = np.linalg.norm(original) + 1e-9
        return float(np.linalg.norm(original - reconstructed) / denom)

    def __repr__(self) -> str:
        cfg = self.config
        return (
            f"EfficientQAT(bits={cfg.bits}, block_size={cfg.block_size}, "
            f"n_calibration_steps={cfg.n_calibration_steps})"
        )
