"""
squish/quant/quarot_quant.py

QuaRotQuantizer: Random-Hadamard Rotation for Outlier-Free INT4 Quantization.

Reference
---------
Ashkboos et al. "QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs."
NeurIPS 2024.

Algorithm
---------
Neural network weight matrices (and activations) often contain large outlier
values in specific channels.  These outliers inflate the quantisation range and
force other values into narrow codes, degrading quality.

QuaRot applies a *random Hadamard rotation* to the weight / activation matrices
before quantisation:

  1. Generate a random diagonal sign matrix ``D`` (d_i ∈ {+1, −1}).
  2. Apply ``R = H_n @ D`` where ``H_n`` is the normalised Walsh–Hadamard
     matrix of order n (nearest power-of-two).
  3. Quantise ``W @ R^T`` (for weights) or ``x @ R`` (for activations).
  4. At inference, the rotation is absorbed into the matrix multiply; the
     effective computation is identical to the original but the matrices have
     dramatically reduced per-channel variance ("outlier migration").

This module provides:
  * ``QuaRotQuantizer.rotate(x)`` — apply the random Hadamard rotation.
  * ``QuaRotQuantizer.unrotate(x)`` — invert the rotation.
  * ``QuaRotQuantizer.quantise(weight)`` — rotate + INT4-quantise.
  * ``QuaRotQuantizer.dequantise(codes, scale, zero)`` — INT4 → FP32.

Key properties
--------------
* ``bits`` — quantisation bits (default 4; W4A4 supported).
* ``group_size`` — per-group scale factors (default 128).
* ``seed`` — random seed for the sign matrix.
* NumPy-only; no external dependencies.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class QuaRotConfig:
    """Configuration for QuaRotQuantizer."""

    bits: int = 4
    """Quantisation bits (4 or 8)."""

    group_size: int = 128
    """Columns per scale-factor group."""

    seed: int = 0
    """Random seed for the Hadamard sign diagonal."""

    symmetric: bool = False
    """Symmetric (zero-point = 0) vs asymmetric quantisation."""

    def __post_init__(self) -> None:
        if self.bits not in (4, 8):
            raise ValueError(f"bits must be 4 or 8; got {self.bits}")
        if self.group_size < 1:
            raise ValueError("group_size must be >= 1")


@dataclass
class QuaRotStats:
    """Runtime counters for QuaRotQuantizer."""

    rotate_calls: int = 0
    quantise_calls: int = 0
    dequantise_calls: int = 0


class QuaRotQuantizer:
    """Hadamard-rotation outlier migration + INT4/INT8 quantizer.

    Usage
    -----
    ::

        qr = QuaRotQuantizer()
        codes, scale, zero = qr.quantise(weight)
        weight_approx = qr.dequantise(codes, scale, zero)
    """

    def __init__(self, config: Optional[QuaRotConfig] = None) -> None:
        self.config = config or QuaRotConfig()
        self.stats = QuaRotStats()
        self._R: Optional[np.ndarray] = None   # cached rotation matrix
        self._sign_diag: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Hadamard utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _hadamard(n: int) -> np.ndarray:
        """Return the normalised Walsh-Hadamard matrix of order n (power of 2)."""
        H = np.array([[1.0]], dtype=np.float32)
        while H.shape[0] < n:
            H = np.block([[H, H], [H, -H]])
        return H / math.sqrt(H.shape[0])

    def _get_rotation(self, dim: int) -> np.ndarray:
        """Return (or build) the rotation matrix R for dimension ``dim``."""
        if self._R is not None and self._R.shape[0] == dim:
            return self._R
        n = 1
        while n < dim:
            n <<= 1
        rng = np.random.default_rng(self.config.seed)
        signs = rng.choice([-1.0, 1.0], size=n).astype(np.float32)
        H = self._hadamard(n)
        R_full = H * signs[None, :]  # broadcast along rows
        # Slice to actual dimension
        self._R = R_full[:dim, :dim]
        self._sign_diag = signs[:dim]
        return self._R

    # ------------------------------------------------------------------
    # Rotation API
    # ------------------------------------------------------------------

    def rotate(self, x: np.ndarray) -> np.ndarray:
        """Apply the random Hadamard rotation to the last dimension of x.

        Parameters
        ----------
        x:
            ``(..., dim)`` float32 array.

        Returns
        -------
        x_rot:
            ``(..., dim)`` float32 — rotated.
        """
        self.stats.rotate_calls += 1
        dim = x.shape[-1]
        R = self._get_rotation(dim)
        return (x.astype(np.float32) @ R.T)

    def unrotate(self, x: np.ndarray) -> np.ndarray:
        """Invert the rotation (R is orthogonal: R^{-1} = R^T).

        Parameters
        ----------
        x:
            ``(..., dim)`` float32 — rotated array.

        Returns
        -------
        x_orig:
            ``(..., dim)`` float32 — un-rotated.
        """
        dim = x.shape[-1]
        R = self._get_rotation(dim)
        return (x.astype(np.float32) @ R)  # R^T @ R = I

    # ------------------------------------------------------------------
    # Quantisation
    # ------------------------------------------------------------------

    def quantise(
        self, weight: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Rotate then quantise a weight matrix.

        Parameters
        ----------
        weight:
            Shape ``(out, in)`` float32.

        Returns
        -------
        codes:
            Shape ``(out, in)`` uint8 (values in [0, 2^bits - 1]).
        scale:
            Shape ``(out, n_groups)`` float32 per-group scale.
        zero:
            Shape ``(out, n_groups)`` float32 per-group zero point.
        """
        self.stats.quantise_calls += 1
        w = self.rotate(weight.astype(np.float32))
        rows, cols = w.shape
        gs = min(self.config.group_size, cols)
        n_groups = math.ceil(cols / gs)
        max_val = float(2 ** self.config.bits - 1)

        codes = np.empty((rows, cols), dtype=np.uint8)
        scale = np.empty((rows, n_groups), dtype=np.float32)
        zero = np.zeros((rows, n_groups), dtype=np.float32)

        for g in range(n_groups):
            c0 = g * gs
            c1 = min(c0 + gs, cols)
            group = w[:, c0:c1]  # (rows, gs_actual)

            if self.config.symmetric:
                abs_max = np.abs(group).max(axis=1, keepdims=True).clip(min=1e-8)
                s = (2 * abs_max) / max_val
                zp = np.zeros_like(abs_max)
                codes[:, c0:c1] = np.clip(
                    np.round(group / s + max_val / 2), 0, max_val
                ).astype(np.uint8)
                scale[:, g] = s[:, 0]
                zero[:, g] = zp[:, 0]
            else:
                g_min = group.min(axis=1, keepdims=True)
                g_max = group.max(axis=1, keepdims=True)
                s = (g_max - g_min) / max_val
                s = np.where(s < 1e-8, np.ones_like(s) * 1e-8, s)
                zp = -g_min / s
                codes[:, c0:c1] = np.clip(
                    np.round(group / s + zp), 0, max_val
                ).astype(np.uint8)
                scale[:, g] = s[:, 0]
                zero[:, g] = zp[:, 0]

        return codes, scale, zero

    def dequantise(
        self,
        codes: np.ndarray,
        scale: np.ndarray,
        zero: np.ndarray,
    ) -> np.ndarray:
        """Decode INT codes back to rotated FP32, then un-rotate.

        Parameters
        ----------
        codes:
            Shape ``(rows, cols)`` uint8.
        scale:
            Shape ``(rows, n_groups)`` float32.
        zero:
            Shape ``(rows, n_groups)`` float32.

        Returns
        -------
        weight_approx:
            Shape ``(rows, cols)`` float32 — dequantised and un-rotated.
        """
        self.stats.dequantise_calls += 1
        rows, cols = codes.shape
        gs = math.ceil(cols / scale.shape[1])
        n_groups = scale.shape[1]
        w_rot = np.empty((rows, cols), dtype=np.float32)

        for g in range(n_groups):
            c0 = g * gs
            c1 = min(c0 + gs, cols)
            s = scale[:, g : g + 1]
            zp = zero[:, g : g + 1]
            w_rot[:, c0:c1] = (codes[:, c0:c1].astype(np.float32) - zp) * s

        return self.unrotate(w_rot)
