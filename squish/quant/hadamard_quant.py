"""squish/quant/hadamard_quant.py

HadamardQuant — Random Hadamard Rotation + INT4 Quantization.

Reference
---------
Ashkboos et al. "QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs."
NeurIPS 2024 (arXiv:2404.00456).

Also used in:
  SpinQuant (Liu et al. 2024), GPTQ+Hadamard variants.

Algorithm
---------
A random Walsh–Hadamard transform (WHT) with a random sign flip is applied
to the weight matrix columns before symmetric INT4 quantization.  The
rotation distributes weight outliers uniformly, eliminating the few extreme
columns that dominate quantization error.

Key properties
--------------
* The rotation is orthogonal, so it does not change the model's outputs if
  applied consistently to activations as well.
* NumPy-only simulation (CPU reference).
* ``group_size`` — per-group scale; -1 → per-row scale.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

__all__ = [
    "HadamardQuantConfig",
    "HadamardQuantResult",
    "HadamardQuant",
]


def _next_power_of_two(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


def _hadamard_matrix(n: int) -> np.ndarray:
    """Construct a normalised 2^k × 2^k Walsh–Hadamard matrix."""
    H = np.array([[1.0]])
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]])
    return H / np.sqrt(H.shape[0])


@dataclass
class HadamardQuantConfig:
    """Configuration for :class:`HadamardQuant`.

    Attributes:
        n_bits: Quantization bit-width (2–8).
        group_size: Columns per quantization group; -1 → per-row.
        hadamard_seed: NumPy seed for the random sign vector.
    """

    n_bits: int = 4
    group_size: int = 128
    hadamard_seed: int = 42

    def __post_init__(self) -> None:
        if not 1 <= self.n_bits <= 8:
            raise ValueError("n_bits must be 1–8")


@dataclass
class HadamardQuantResult:
    """Result of HadamardQuant.

    Attributes:
        W_q: Quantized weight codes (INT-n_bits), shape (out_f, in_f_padded).
        scale: Per-group dequantization scale.
        zero: Per-group zero point.
        H: Hadamard matrix used for rotation (in_f_padded × in_f_padded).
        sign_vec: Random sign vector applied before WHT.
        in_features: Original (unpadded) number of input features.
    """

    W_q: np.ndarray
    scale: np.ndarray
    zero: np.ndarray
    H: np.ndarray
    sign_vec: np.ndarray
    in_features: int

    def dequantize(self) -> np.ndarray:
        """Return the de-quantized weight in the *rotated* space."""
        # W_q: (out_f, n_groups * gs), scale/zero: (out_f, n_groups)
        out_f = self.W_q.shape[0]
        n_groups = self.scale.shape[1]
        gs = self.W_q.shape[1] // n_groups
        scale_exp = np.repeat(self.scale, gs, axis=1)  # (out_f, n_groups * gs)
        zero_exp = np.repeat(self.zero, gs, axis=1)    # (out_f, n_groups * gs)
        return (self.W_q.astype(np.float32) - zero_exp) * scale_exp

    def dequantize_unrotated(self) -> np.ndarray:
        """Return the de-quantized weight in the *original* space, trimmed."""
        W_rot = self.dequantize()  # (out_f, n_groups * gs)
        n_pad = self.H.shape[0]
        # Pad or trim to match Hadamard rotation dimension
        if W_rot.shape[1] < n_pad:
            W_rot = np.pad(W_rot, [(0, 0), (0, n_pad - W_rot.shape[1])])
        elif W_rot.shape[1] > n_pad:
            W_rot = W_rot[:, :n_pad]
        # Un-rotate: W_orig ≈ W_rot @ H^T * sign (H is orthogonal)
        W_unrot = (W_rot @ self.H.T) * self.sign_vec[None, :]
        return W_unrot[:, : self.in_features]


class HadamardQuant:
    """Weight quantizer using random Hadamard rotation to suppress outliers.

    Parameters
    ----------
    config:
        HadamardQuant configuration.
    """

    def __init__(self, config: Optional[HadamardQuantConfig] = None) -> None:
        self._cfg = config or HadamardQuantConfig()
        self._rng = np.random.default_rng(self._cfg.hadamard_seed)

    @property
    def config(self) -> HadamardQuantConfig:
        return self._cfg

    def _make_rotation(self, n_pad: int) -> Tuple[np.ndarray, np.ndarray]:
        """Build H and random sign vector for dimension ``n_pad``."""
        H = _hadamard_matrix(n_pad)
        signs = self._rng.choice([-1.0, 1.0], size=n_pad).astype(np.float32)
        return H.astype(np.float32), signs

    def _quantize_per_group(
        self, W_rot: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Quantize ``W_rot`` per-group and return (codes, scale, zero)."""
        out_f, in_f = W_rot.shape
        gs = self._cfg.group_size if self._cfg.group_size > 0 else in_f
        n_groups = (in_f + gs - 1) // gs
        pad = (-in_f) % gs
        if pad:
            W_rot = np.pad(W_rot, [(0, 0), (0, pad)])

        n_levels = 2 ** self._cfg.n_bits
        W_g = W_rot.reshape(out_f, n_groups, gs)
        w_min = W_g.min(axis=-1)
        w_max = W_g.max(axis=-1)
        scale = np.maximum((w_max - w_min) / (n_levels - 1), 1e-8)
        zero = -w_min / scale
        codes = np.round(W_g / scale[:, :, None] + zero[:, :, None])
        codes = codes.clip(0, n_levels - 1).astype(np.int32)
        # Flatten group dimension for storage
        codes_flat = codes.reshape(out_f, n_groups * gs)[:, :in_f + pad]
        return codes_flat, scale, zero

    def quantize(self, weights: np.ndarray) -> HadamardQuantResult:
        """Apply Hadamard rotation and INT4 quantize.

        Parameters
        ----------
        weights:
            FP32 weight matrix, shape ``(out_features, in_features)``.
        """
        W = np.asarray(weights, dtype=np.float32)
        out_f, in_f = W.shape
        n_pad = _next_power_of_two(in_f)

        H, signs = self._make_rotation(n_pad)
        W_padded = np.pad(W, [(0, 0), (0, n_pad - in_f)])
        W_rot = (W_padded * signs[None, :]) @ H

        codes, scale, zero = self._quantize_per_group(W_rot[:, :in_f])
        return HadamardQuantResult(
            W_q=codes,
            scale=scale,
            zero=zero,
            H=H,
            sign_vec=signs,
            in_features=in_f,
        )
