"""squish/attention/yarn_rope.py

YaRNRoPE — NTK-by-parts Rotary Position Embedding with Temperature Correction.

Reference
---------
Peng et al. "YaRN: Efficient Context Window Extension of Large Language
Models." ICLR 2024 (arXiv:2309.00071).

Algorithm
---------
YaRN extends RoPE to very long contexts by:

1. **NTK-by-parts scaling** — instead of uniformly scaling all frequencies
   (NTK-aware) or uniformly interpolating (PositionInterpolation), YaRN
   identifies a *ramp* from ``alpha`` to ``beta`` in frequency space:
   * Low-frequency dimensions: linear interpolation (scale by 1/factor).
   * High-frequency dimensions: extrapolation (no change).
   * Mid-range: blended via a smooth ramp function.

2. **Temperature (attention scaling)** — a correction factor ``t``
   counteracts the entropy increase from context extension:
   ``softmax_scale *= t ≈ 0.1 ln(s) + 1`` where s = target / original length.

Key properties
--------------
* ``build_freqs(seq_len)`` returns the YaRN-adjusted sin/cos tables.
* ``apply(x, offset)`` rotates hidden states with positional offset.
* Pure NumPy; no ML framework dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

__all__ = [
    "YaRNRoPEConfig",
    "YaRNRoPE",
]


@dataclass
class YaRNRoPEConfig:
    """Configuration for :class:`YaRNRoPE`.

    Attributes:
        dim: Head dimension (must be even).
        original_max_len: Original training context length (e.g., 4096).
        target_max_len: Desired extended context length (e.g., 131072).
        base: RoPE base frequency (typically 10000).
        alpha: Low-freq ramp parameter (default 1).
        beta: High-freq ramp parameter (default 32).
    """

    dim: int = 128
    original_max_len: int = 4096
    target_max_len: int = 131072
    base: float = 10000.0
    alpha: float = 1.0
    beta: float = 32.0

    def __post_init__(self) -> None:
        if self.dim % 2 != 0:
            raise ValueError("dim must be even")


class YaRNRoPE:
    """YaRN NTK-by-parts RoPE with temperature correction.

    Parameters
    ----------
    config:
        YaRNRoPEConfig.
    """

    def __init__(self, config: Optional[YaRNRoPEConfig] = None) -> None:
        self._cfg = config or YaRNRoPEConfig()
        self._scale_factor = self._cfg.target_max_len / self._cfg.original_max_len
        self._temperature = float(0.1 * np.log(self._scale_factor) + 1.0)
        self._inv_freq = self._build_inv_freq()

    @property
    def config(self) -> YaRNRoPEConfig:
        return self._cfg

    @property
    def scale_factor(self) -> float:
        return self._scale_factor

    @property
    def temperature(self) -> float:
        """Attention scale correction factor (multiply softmax scale by this)."""
        return self._temperature

    def _build_inv_freq(self) -> np.ndarray:
        """Compute NTK-by-parts inverse frequencies."""
        cfg = self._cfg
        half_dim = cfg.dim // 2
        i = np.arange(half_dim, dtype=np.float64)
        # Standard RoPE inv_freq
        inv_freq_std = 1.0 / (cfg.base ** (i / half_dim))

        # Linear interpolation inv_freq
        inv_freq_lin = inv_freq_std / self._scale_factor

        # Per-dimension ramp: 0 = high-freq (no change), 1 = low-freq (interpolate)
        # ramp_fn from YaRN paper
        low = np.log(cfg.original_max_len / (2 * np.pi * cfg.alpha)) / np.log(cfg.base)
        high = np.log(cfg.original_max_len / (2 * np.pi * cfg.beta)) / np.log(cfg.base)
        # Dimensionless frequency index clipped to [0, 1]
        ramp = np.clip((i / half_dim - low) / (high - low), 0.0, 1.0)

        inv_freq_yarn = (1.0 - ramp) * inv_freq_lin + ramp * inv_freq_std
        return inv_freq_yarn.astype(np.float32)

    def build_freqs(self, seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
        """Build sin/cos tables for ``seq_len`` positions.

        Returns
        -------
        (cos_table, sin_table) each of shape ``(seq_len, dim // 2)``.
        """
        positions = np.arange(seq_len, dtype=np.float32)
        freqs = np.outer(positions, self._inv_freq)  # (seq_len, dim//2)
        return np.cos(freqs), np.sin(freqs)

    def apply(
        self,
        x: np.ndarray,
        offset: int = 0,
    ) -> np.ndarray:
        """Apply YaRN RoPE to a batch of hidden states.

        Parameters
        ----------
        x:
            Query or key tensor, shape ``(batch, seq, dim)`` or
            ``(seq, dim)``.
        offset:
            Starting position index for this chunk.

        Returns
        -------
        Rotated tensor with the same shape as input.
        """
        orig_shape = x.shape
        if x.ndim == 2:
            x = x[None, ...]
        batch, seq, dim = x.shape
        if dim != self._cfg.dim:
            raise ValueError(f"Expected dim={self._cfg.dim}, got {dim}")

        cos_t, sin_t = self.build_freqs(offset + seq)
        cos_t = cos_t[offset:, :]  # (seq, dim//2)
        sin_t = sin_t[offset:, :]

        x1 = x[..., : dim // 2]
        x2 = x[..., dim // 2 :]
        rot = np.concatenate([-x2, x1], axis=-1)
        # Tile cos/sin to full dim
        cos_full = np.concatenate([cos_t, cos_t], axis=-1)[None, ...]  # (1, seq, dim)
        sin_full = np.concatenate([sin_t, sin_t], axis=-1)[None, ...]
        out = x * cos_full + rot * sin_full
        return out.reshape(orig_shape)
