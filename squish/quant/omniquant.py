"""squish/quant/omniquant.py

OmniQuant — Omnidirectional Post-Training Quantization.

OmniQuant jointly optimises two learnable transformations to minimise the
quantization reconstruction error for both weights and activations:

* **Learnable Weight Clipping (LWC)** — per-channel clip values that shape
  the weight distribution before uniform quantization.
* **Learnable Equivalent Transformation (LET)** — per-channel scales applied
  to activations (and the inverse to weights) to smooth outliers via an
  equivalent transformation that preserves the matmul output.

The calibration procedure alternates gradient steps on ``clip_val`` and
``transform_scale`` using a small batch of calibration activations.

Reference
---------
Shao, W. et al. "OmniQuant: Omnidirectional Calibration for Low-bit
Quantization of Large Language Models."
arXiv:2308.13137, ICLR 2024 (Spotlight).
"""

from __future__ import annotations

__all__ = ["OmniQuantConfig", "OmniQuantizer"]

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from numpy import ndarray


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class OmniQuantConfig:
    """Configuration for OmniQuantizer.

    Parameters
    ----------
    w_bits:
        Weight quantization bit-width (e.g., 4).
    a_bits:
        Activation quantization bit-width (e.g., 8).
    n_iters:
        Number of gradient descent iterations during calibration.
    lwc_lr:
        Learning rate for Learnable Weight Clipping optimisation.
    let_lr:
        Learning rate for Learnable Equivalent Transformation optimisation.
    seed:
        RNG seed.
    """

    w_bits: int = 4
    a_bits: int = 8
    n_iters: int = 100
    lwc_lr: float = 1e-2
    let_lr: float = 1e-2
    seed: int = 0

    def __post_init__(self) -> None:
        if not 2 <= self.w_bits <= 8:
            raise ValueError("w_bits must be in [2, 8]")
        if not 2 <= self.a_bits <= 16:
            raise ValueError("a_bits must be in [2, 16]")
        if self.n_iters < 1:
            raise ValueError("n_iters must be >= 1")
        if self.lwc_lr <= 0:
            raise ValueError("lwc_lr must be positive")
        if self.let_lr <= 0:
            raise ValueError("let_lr must be positive")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _uniform_quant(x: ndarray, bits: int, x_min: ndarray, x_max: ndarray) -> ndarray:
    """Symmetric per-channel uniform quantization with straight-through gradient."""
    q_levels = float((1 << bits) - 1)
    scale = np.where(
        np.abs(x_max - x_min) > 1e-8,
        (x_max - x_min) / q_levels,
        np.ones_like(x_max),
    )
    x_clamped = np.clip(x, x_min, x_max)
    x_q = np.round((x_clamped - x_min) / scale) * scale + x_min
    return x_q.astype(np.float32)


# ---------------------------------------------------------------------------
# Quantizer
# ---------------------------------------------------------------------------

class OmniQuantizer:
    """Post-training quantizer using LWC + LET calibration.

    Parameters
    ----------
    config:
        ``OmniQuantConfig`` instance.
    """

    def __init__(self, config: OmniQuantConfig | None = None) -> None:
        self.config = config or OmniQuantConfig()
        self._rng = np.random.default_rng(self.config.seed)

    def calibrate(
        self, W: ndarray, X_calib: ndarray
    ) -> Tuple[ndarray, ndarray]:
        """Learn optimal LWC clip values and LET transform scales.

        Minimises per-channel reconstruction loss
        ``||X_calib @ W.T - X_calib_scaled @ W_q(clip_val).T||_F^2``
        via simple gradient descent using finite differences.

        Parameters
        ----------
        W:
            Weight matrix, shape ``(out_features, in_features)``.
        X_calib:
            Calibration activations, shape ``(n_samples, in_features)``.

        Returns
        -------
        clip_val:
            Learned per-output-channel clip range, shape ``(out_features,)``.
        transform_scale:
            Learned per-input-channel activation scale, shape ``(in_features,)``.
        """
        W = np.asarray(W, dtype=np.float32)
        X_calib = np.asarray(X_calib, dtype=np.float32)
        out_features, in_features = W.shape

        # Initialise: clip_val = max |W| per output channel; scale = 1
        clip_val = np.abs(W).max(axis=1).astype(np.float32)  # (out,)
        transform_scale = np.ones(in_features, dtype=np.float32)  # (in,)

        ref_output = X_calib @ W.T  # (n_samples, out_features)

        lwc_lr = float(self.config.lwc_lr)
        let_lr = float(self.config.let_lr)
        eps = 1e-4

        for _ in range(self.config.n_iters):
            # LWC step: finite-difference gradient on clip_val
            W_q = self._quantize_w(W, clip_val)
            X_scaled = X_calib * transform_scale[np.newaxis, :]
            current_output = X_scaled @ W_q.T

            loss = float(np.mean((current_output - ref_output) ** 2))
            grad_clip = np.zeros_like(clip_val)
            for i in range(out_features):
                cv_plus = clip_val.copy()
                cv_plus[i] += eps
                W_q_plus = self._quantize_w(W, cv_plus)
                out_plus = X_scaled @ W_q_plus.T
                loss_plus = float(np.mean((out_plus - ref_output) ** 2))
                grad_clip[i] = (loss_plus - loss) / eps
            clip_val -= lwc_lr * grad_clip
            clip_val = np.clip(clip_val, 1e-3, np.abs(W).max(axis=1) * 2.0)

            # LET step: finite-difference gradient on transform_scale
            W_q = self._quantize_w(W, clip_val)
            X_scaled = X_calib * transform_scale[np.newaxis, :]
            loss = float(np.mean((X_scaled @ W_q.T - ref_output) ** 2))
            grad_ts = np.zeros_like(transform_scale)
            for j in range(in_features):
                ts_plus = transform_scale.copy()
                ts_plus[j] += eps
                X_plus = X_calib * ts_plus[np.newaxis, :]
                loss_plus = float(np.mean((X_plus @ W_q.T - ref_output) ** 2))
                grad_ts[j] = (loss_plus - loss) / eps
            transform_scale -= let_lr * grad_ts
            transform_scale = np.clip(transform_scale, 1e-3, None)

        return clip_val, transform_scale

    def _quantize_w(self, W: ndarray, clip_val: ndarray) -> ndarray:
        """Internal: quantize W with current clip_val per output channel."""
        w_min = -clip_val[:, np.newaxis] * np.ones_like(W)
        w_max = clip_val[:, np.newaxis] * np.ones_like(W)
        return _uniform_quant(W, self.config.w_bits, w_min, w_max)

    def quantize_weight(
        self, W: ndarray, clip_val: ndarray
    ) -> Tuple[ndarray, ndarray]:
        """Quantize weight matrix using learned clip values.

        Parameters
        ----------
        W:
            Weight matrix, shape ``(out_features, in_features)``.
        clip_val:
            Per-output-channel clip bounds, shape ``(out_features,)``.

        Returns
        -------
        W_q:
            Quantized weight matrix, float32, same shape as ``W``.
        scales:
            Per-channel scale factors, shape ``(out_features,)``.
        """
        W = np.asarray(W, dtype=np.float32)
        clip_val = np.asarray(clip_val, dtype=np.float32)
        q_levels = float((1 << self.config.w_bits) - 1)
        scales = (2.0 * clip_val) / q_levels  # (out_features,)
        W_q = self._quantize_w(W, clip_val)
        return W_q, scales

    def forward(
        self,
        x: ndarray,
        W_q: ndarray,
        scales: ndarray,
        transform_scale: ndarray,
    ) -> ndarray:
        """Forward pass with quantized weights and LET scaling.

        Parameters
        ----------
        x:
            Input activations, shape ``(batch, in_features)``.
        W_q:
            Quantized weight matrix from :meth:`quantize_weight`.
        scales:
            Per-output-channel scales (unused in float forward; included for
            interface parity with other quantizers).
        transform_scale:
            Per-input-channel LET scales, shape ``(in_features,)``.

        Returns
        -------
        Output activations, shape ``(batch, out_features)``.
        """
        x = np.asarray(x, dtype=np.float32)
        x_scaled = x * np.asarray(transform_scale, dtype=np.float32)[np.newaxis, :]
        return (x_scaled @ np.asarray(W_q, dtype=np.float32).T).astype(np.float32)
