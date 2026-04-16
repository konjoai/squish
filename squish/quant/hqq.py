"""HQQQuantizer — Half-Quadratic Quantization (calibration-free PTQ).

Implements the HQQ algorithm (Badri & Shaji, arXiv:2309.15531, 2024).

HQQ frames weight quantisation as a proximal-point optimisation problem,
avoiding the need for calibration data entirely.  The objective is:

    min_W_q  ||W - W_q||² + λ · proximal_penalty(W_q)

where the proximal penalty encourages the quantised weights to lie on the
uniform grid.  A fast iterative solve (alternating half-quadratic splitting)
reaches a good solution in just a handful of iterations, making it 10× faster
than GPTQ while matching or exceeding its accuracy on INT2/INT4.

This implementation provides a NumPy simulation of HQQ, suitable for
off-line weight compression targeting any precision from 1-bit (binary)
through INT8, including non-integer widths such as 1.5-bit (3 levels),
2.5-bit (6 levels), and 3.5-bit (11 levels).  The resulting compressed
tensors can be stored and loaded back without GPU dependencies.

Reference:
    Badri & Shaji, "HQQ: Half-Quadratic Quantization of Large Machine
    Learning Models", arXiv:2309.15531 v3 (2024).
"""

from __future__ import annotations

__all__ = [
    "HQQConfig",
    "HQQTensor",
    "HQQQuantizer",
]

from dataclasses import dataclass
from typing import Optional

import numpy as np

# ── Helpers ───────────────────────────────────────────────────────────────────

# Any float nbits in [1.0, 8.0] is accepted.
# Integer widths (1–8) map to exact power-of-two levels.
# Fractional widths (e.g. 2.5) map to round(2**nbits) levels:
#   1.0 →  2 levels (binary)
#   1.5 →  3 levels
#   2.0 →  4 levels
#   2.5 →  6 levels
#   3.0 →  8 levels
#   3.5 → 11 levels
#   4.0 → 16 levels
#   8.0 → 256 levels
_BITS_MIN: float = 1.0
_BITS_MAX: float = 8.0


def _grid_levels(nbits: float) -> int:
    """Return the number of quantisation levels for *nbits* (any float in [1,8]).

    Uses ``max(2, round(2**nbits))`` so that sub-integer widths like 2.5-bit
    yield 6 levels and 1-bit yields 2 levels (binary).
    """
    return max(2, round(2.0 ** nbits))


# ── Config ────────────────────────────────────────────────────────────────────


@dataclass
class HQQConfig:
    """Configuration for HQQQuantizer.

    Attributes:
        bits: Target bit-width.  Accepts any **float** in ``[1.0, 8.0]``,
            including non-integer widths such as ``1.5``, ``2.5``, or ``3.5``
            (see :func:`_grid_levels`).  Passing an integer (e.g. ``4``) is
            fully backward-compatible; it is coerced to ``float`` internally.
        group_size: Number of weights per quantisation group.  ``-1`` means
            use the full row (per-row quantisation).
        lambda_scale: Proximal penalty weight (λ).  Higher values push the
            solution harder onto the grid at the cost of slightly more
            reconstruction error.
        max_iter: Number of half-quadratic solver iterations.  10 is
            typically more than sufficient.
        axis: Axis along which groups are formed (0 = per-row, 1 = per-col).
    """

    bits: float = 4.0
    group_size: int = 128
    lambda_scale: float = 1.0
    max_iter: int = 10
    axis: int = 0

    def __post_init__(self) -> None:
        # Coerce integer bits to float for uniform handling.
        self.bits = float(self.bits)
        if not (_BITS_MIN <= self.bits <= _BITS_MAX):
            raise ValueError(
                f"bits must be in [{_BITS_MIN}, {_BITS_MAX}]; got {self.bits}"
            )
        if self.group_size != -1 and self.group_size < 1:
            raise ValueError(
                f"group_size must be ≥ 1 or -1 (full row); got {self.group_size}"
            )
        if self.lambda_scale <= 0:
            raise ValueError(
                f"lambda_scale must be positive; got {self.lambda_scale}"
            )
        if self.max_iter < 1:
            raise ValueError(
                f"max_iter must be ≥ 1; got {self.max_iter}"
            )
        if self.axis not in (0, 1):
            raise ValueError(
                f"axis must be 0 or 1; got {self.axis}"
            )

    @property
    def n_levels(self) -> int:
        """Number of quantisation levels derived from ``bits``."""
        return _grid_levels(self.bits)


# ── Quantised tensor container ────────────────────────────────────────────────


@dataclass
class HQQTensor:
    """Compressed tensor returned by :meth:`HQQQuantizer.encode`.

    Attributes:
        codes: Integer codes array (uint8 for bits ≤ 8).
        scale: Per-group scale factors (float32).
        zero: Per-group zero-points (float32).
        shape: Original weight shape.
        config: The :class:`HQQConfig` used.
    """

    codes: np.ndarray
    scale: np.ndarray
    zero: np.ndarray
    shape: tuple[int, ...]
    config: HQQConfig

    def nbytes(self) -> int:
        """Approximate storage size in bytes (codes + scales + zeros)."""
        return int(
            self.codes.nbytes + self.scale.nbytes + self.zero.nbytes
        )


# ── Main class ────────────────────────────────────────────────────────────────


class HQQQuantizer:
    """Calibration-free weight quantizer via half-quadratic splitting.

    Supports any bit-width in ``[1.0, 8.0]``, including fractional widths
    such as 1.5-bit (3 levels), 2.5-bit (6 levels), and 3.5-bit (11 levels).

    Example::

        cfg = HQQConfig(bits=3, group_size=64)       # integer bits (classic)
        cfg = HQQConfig(bits=2.5, group_size=64)     # 6-level fractional
        cfg = HQQConfig(bits=1.0, group_size=128)    # 1-bit (binary)
        quant = HQQQuantizer(cfg)
        tensor = quant.encode(weight)        # → HQQTensor
        W_hat  = quant.decode(tensor)        # → float32 numpy array
        err    = quant.relative_error(weight, W_hat)
        snr    = quant.quantisation_error_db(weight, W_hat)

    Args:
        config: :class:`HQQConfig` (optional; defaults to 4-bit, group=128).
    """

    def __init__(self, config: Optional[HQQConfig] = None) -> None:
        self.config: HQQConfig = config or HQQConfig()

    # ── Encode ────────────────────────────────────────────────────────────────

    def encode(self, weight: np.ndarray) -> HQQTensor:
        """Quantise a weight matrix to the target bit-width.

        Args:
            weight: Float32 weight of shape ``(C_out, C_in)``; any 2-D array.

        Returns:
            :class:`HQQTensor` containing integer codes, scales, and zeros.

        Raises:
            ValueError: If ``weight`` is not 2-D.
        """
        W = np.asarray(weight, dtype=np.float32)
        if W.ndim != 2:
            raise ValueError(
                f"weight must be 2-D; got shape {W.shape}"
            )

        cfg = self.config
        n_levels = _grid_levels(cfg.bits)  # works for float nbits
        qmax = float(n_levels - 1)

        rows, cols = W.shape
        group_size = (
            cfg.group_size
            if cfg.group_size != -1
            else (cols if cfg.axis == 0 else rows)
        )

        # Work along axis 0 (rows) by default; transpose for axis 1
        if cfg.axis == 0:
            dim_size = cols
        else:
            dim_size = rows
            W = W.T

        n_groups = max(1, (dim_size + group_size - 1) // group_size)
        padded = n_groups * group_size
        if padded > dim_size:
            pad_width = [(0, 0)] * W.ndim
            pad_width[-1] = (0, padded - dim_size)
            W_pad = np.pad(W, pad_width, mode="constant")
        else:
            W_pad = W

        other_dim = W_pad.shape[0]
        W_groups = W_pad.reshape(other_dim, n_groups, group_size)

        # Per-group min/max → initial scale + zero
        g_min = W_groups.min(axis=-1, keepdims=True)
        g_max = W_groups.max(axis=-1, keepdims=True)
        span = np.maximum(g_max - g_min, 1e-6)
        scales = span / qmax          # (other_dim, n_groups, 1)
        zeros = g_min                  # (other_dim, n_groups, 1)

        codes_f = (W_groups - zeros) / scales
        codes = np.clip(np.round(codes_f), 0, qmax)

        # Half-quadratic iterative refinement (alternating scale/zero/code updates)
        lam = cfg.lambda_scale
        for _ in range(cfg.max_iter):
            c = codes
            c2 = (c ** 2).sum(axis=-1, keepdims=True) + lam
            scales = (
                (c * (W_groups - zeros)).sum(axis=-1, keepdims=True) + lam * scales
            ) / c2
            zeros = (W_groups - codes * scales).mean(axis=-1, keepdims=True)
            codes_f = (W_groups - zeros) / np.maximum(np.abs(scales), 1e-8)
            codes = np.clip(np.round(codes_f), 0, qmax)

        codes_trim = codes.reshape(other_dim, -1)[:, :dim_size]
        scales_out = scales.squeeze(-1)   # (other_dim, n_groups)
        zeros_out = zeros.squeeze(-1)

        if cfg.axis == 1:
            codes_trim = codes_trim.T

        return HQQTensor(
            codes=codes_trim.astype(np.uint8),
            scale=scales_out.astype(np.float32),
            zero=zeros_out.astype(np.float32),
            shape=tuple(weight.shape),
            config=cfg,
        )

    # ── Decode ────────────────────────────────────────────────────────────────

    def decode(self, tensor: HQQTensor) -> np.ndarray:
        """Reconstruct float32 weights from an :class:`HQQTensor`.

        Args:
            tensor: Output of :meth:`encode`.

        Returns:
            Float32 weight array of the original shape.
        """
        cfg = tensor.config

        codes = tensor.codes.astype(np.float32)
        rows, cols = tensor.shape

        if cfg.axis == 0:
            dim_size = cols
            other_dim = rows
        else:
            dim_size = rows
            other_dim = cols

        n_groups = tensor.scale.shape[-1]
        group_size_actual = max(1, (dim_size + n_groups - 1) // n_groups)

        padded = n_groups * group_size_actual
        if codes.shape[-1] < padded:
            codes_pad = np.zeros((other_dim, padded), dtype=np.float32)
            codes_pad[:, : codes.shape[-1]] = codes
        else:
            codes_pad = codes

        codes_g = codes_pad.reshape(other_dim, n_groups, group_size_actual)
        scales = tensor.scale[:, :, np.newaxis]   # (O, G, 1)
        zeros = tensor.zero[:, :, np.newaxis]
        W_hat = codes_g * scales + zeros

        W_hat_flat = W_hat.reshape(other_dim, -1)[:, :dim_size]
        if cfg.axis == 1:
            W_hat_flat = W_hat_flat.T

        return W_hat_flat.astype(np.float32)

    # ── Quality metrics ───────────────────────────────────────────────────────

    def relative_error(
        self, original: np.ndarray, reconstructed: np.ndarray
    ) -> float:
        """Mean relative L2 reconstruction error ``||W - Ŵ|| / ||W||``."""
        orig = np.asarray(original, dtype=np.float32)
        recon = np.asarray(reconstructed, dtype=np.float32)
        norm_orig = float(np.linalg.norm(orig))
        if norm_orig == 0:
            return 0.0
        return float(np.linalg.norm(orig - recon)) / norm_orig

    def quantisation_error_db(
        self, original: np.ndarray, reconstructed: np.ndarray
    ) -> float:
        """Signal-to-noise ratio in dB (higher = better)."""
        orig = np.asarray(original, dtype=np.float32)
        recon = np.asarray(reconstructed, dtype=np.float32)
        signal_power = float(np.mean(orig ** 2))
        noise_power = float(np.mean((orig - recon) ** 2)) + 1e-20
        return 10.0 * np.log10(signal_power / noise_power)

    def __repr__(self) -> str:
        return (
            f"HQQQuantizer(bits={self.config.bits}, "
            f"group_size={self.config.group_size}, "
            f"max_iter={self.config.max_iter})"
        )
