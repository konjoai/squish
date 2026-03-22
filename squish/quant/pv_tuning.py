"""squish/quant/pv_tuning.py

PVTuning — Proximal-Gradient Quantized Weight Optimization.

Reference
---------
Malinovskiy et al. "PV-Tuning: Beyond Straight-Through Estimation for
Extreme LLM Compression." NeurIPS 2024 (arXiv:2405.14852).

Algorithm
---------
Standard quantization-aware training uses the straight-through estimator
(STE) to pass gradients through the non-differentiable round() operation.
PV-Tuning replaces STE with a proximal-gradient update:

1. Start from the quantized weight Q(W).
2. Compute a gradient step: W_new = W_current - lr * grad_L(W_current).
3. Apply a quantization proximity projection: W_prox = Q(W_new).
4. Repeat for several steps.

This leads to significantly lower PPL at W1–W2 compression vs QuIP# and
QuaRot at the same bit-width.

Key properties
--------------
* NumPy-only simulation.
* ``n_bits`` — quantization bit-width (1–4; specialty at 1–2).
* ``n_steps`` — number of proximal gradient steps.
* ``lr`` — proximal gradient step size.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

__all__ = [
    "PVTuningConfig",
    "PVTuningResult",
    "PVTuning",
]


@dataclass
class PVTuningConfig:
    """Configuration for :class:`PVTuning`.

    Attributes:
        n_bits: Quantization bit-width (1–4).
        n_steps: Number of proximal gradient iterations.
        lr: Proximal gradient step size.
        group_size: Group size for per-group scale computation.
    """

    n_bits: int = 2
    n_steps: int = 20
    lr: float = 5e-4
    group_size: int = 128

    def __post_init__(self) -> None:
        if not 1 <= self.n_bits <= 8:
            raise ValueError("n_bits must be 1–8")


@dataclass
class PVTuningResult:
    """Result of PVTuning compression.

    Attributes:
        W_q: Final quantized weight codes.
        scale: Per-group dequantization scale.
        zero: Per-group zero point.
        n_steps_taken: Number of iterations executed.
        final_error: Frobenius norm of W - W_deq.
    """

    W_q: np.ndarray
    scale: np.ndarray
    zero: np.ndarray
    n_steps_taken: int
    final_error: float

    def dequantize(self) -> np.ndarray:
        """Reconstruct approximate FP32 weight matrix."""
        # W_q: (out_f, n_groups, gs) → flatten to (out_f, n_groups * gs)
        W_flat = self.W_q.reshape(self.W_q.shape[0], -1)
        out_f, total = W_flat.shape
        n_groups = self.scale.shape[1]
        gs = total // n_groups
        scale_exp = np.repeat(self.scale, gs, axis=1)  # (out_f, n_groups * gs)
        zero_exp = np.repeat(self.zero, gs, axis=1)    # (out_f, n_groups * gs)
        return (W_flat.astype(np.float32) - zero_exp) * scale_exp


class PVTuning:
    """Proximal-gradient quantized weight optimizer.

    Parameters
    ----------
    config:
        PVTuning configuration.
    """

    def __init__(self, config: Optional[PVTuningConfig] = None) -> None:
        self._cfg = config or PVTuningConfig()

    @property
    def config(self) -> PVTuningConfig:
        return self._cfg

    def _quantize_group(self, w_group: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """Quantize a weight group to INT-n_bits.

        Returns (codes_flat, scale, zero).
        """
        n_levels = 2 ** self._cfg.n_bits
        w_min = float(w_group.min())
        w_max = float(w_group.max())
        scale = max((w_max - w_min) / (n_levels - 1), 1e-8)
        zero = -w_min / scale
        codes = np.round(w_group / scale + zero).clip(0, n_levels - 1).astype(np.int32)
        return codes, scale, zero

    def _dequantize_group(self, codes: np.ndarray, scale: float, zero: float) -> np.ndarray:
        return (codes.astype(np.float32) - zero) * scale

    def compress(self, weights: np.ndarray) -> PVTuningResult:
        """Run PV-Tuning proximal gradient compression.

        Parameters
        ----------
        weights:
            FP32 weight matrix, shape ``(out_features, in_features)``.

        Returns
        -------
        PVTuningResult
        """
        W = np.asarray(weights, dtype=np.float32)
        out_f, in_f = W.shape
        gs = self._cfg.group_size
        n_groups = (in_f + gs - 1) // gs
        pad = (-in_f) % gs
        if pad:
            W_padded = np.pad(W, [(0, 0), (0, pad)])
        else:
            W_padded = W.copy()

        W_current = W_padded.copy()
        scales = np.zeros((out_f, n_groups), dtype=np.float32)
        zeros = np.zeros((out_f, n_groups), dtype=np.float32)
        codes_final = np.zeros((out_f, n_groups, gs), dtype=np.int32)

        for step in range(self._cfg.n_steps):
            # Proximal gradient update (simulated: gradient = W_current - W_target)
            grad = W_current - W_padded  # minimize ||W_current - W_orig||
            W_current = W_current - self._cfg.lr * grad

            # Quantization proximity projection
            W_groups = W_current.reshape(out_f, n_groups, gs)
            for i in range(out_f):
                for g in range(n_groups):
                    codes, sc, zr = self._quantize_group(W_groups[i, g])
                    codes_final[i, g] = codes
                    scales[i, g] = sc
                    zeros[i, g] = zr
                    W_groups[i, g] = self._dequantize_group(codes, sc, zr)
            W_current = W_groups.reshape(out_f, n_groups * gs)

        final_error = float(np.linalg.norm(W_padded[:, :in_f] - W_current[:, :in_f], "fro"))
        return PVTuningResult(
            W_q=codes_final,
            scale=scales,
            zero=zeros,
            n_steps_taken=self._cfg.n_steps,
            final_error=final_error,
        )
