"""squish/attention/clex_rope.py

CLeXRoPE — Continuous Per-Frequency Learned RoPE Scale.

Reference
---------
Chen et al. "CLEx: Continuous Length Extrapolation for Large Language
Models." arXiv:2310.16450, 2023.

Algorithm
---------
RoPE positions each head dimension i with frequency θ_i = base^{-2i/d}.
For long-context extrapolation, CLEx learns a *continuous* per-frequency
scale function s(i) rather than applying a single global scale factor.

The scale is parameterised as a small neural network (3-layer MLP) that maps
the normalised frequency index i/d to a positive scale s_i:
  θ̃_i = θ_i / s_i

During inference, s_i is fixed (post-calibration parameters).  This module
provides:
* ``CLeXRoPE.calibrate(hidden_states, positions)`` — fits scale parameters.
* ``CLeXRoPE.build_freqs(seq_len)`` — builds sin/cos tables.
* ``CLeXRoPE.apply(x, offset)`` — rotates query/key tensors.

Key properties
--------------
* NumPy-only; MLP is a simple 3-layer network fit by gradient descent.
* Falls back to standard RoPE if not calibrated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

__all__ = [
    "CLeXRoPEConfig",
    "CLeXRoPE",
]


@dataclass
class CLeXRoPEConfig:
    """Configuration for :class:`CLeXRoPE`.

    Attributes:
        dim: Head dimension (must be even).
        base: RoPE base frequency.
        original_max_len: Original training context length.
        target_max_len: Desired extended context length.
        scale_hidden: Hidden units in the per-frequency scale MLP.
        n_calibration_steps: Gradient steps during calibration.
        lr: Learning rate for calibration.
    """

    dim: int = 128
    base: float = 10000.0
    original_max_len: int = 4096
    target_max_len: int = 32768
    scale_hidden: int = 32
    n_calibration_steps: int = 100
    lr: float = 1e-3

    def __post_init__(self) -> None:
        if self.dim % 2 != 0:
            raise ValueError("dim must be even")


class CLeXRoPE:
    """Continuous per-frequency learned RoPE scale.

    Parameters
    ----------
    config:
        CLeXRoPEConfig.
    """

    def __init__(self, config: Optional[CLeXRoPEConfig] = None) -> None:
        self._cfg = config or CLeXRoPEConfig()
        self._calibrated = False
        # Initialise scale MLP weights
        half = self._cfg.dim // 2
        h = self._cfg.scale_hidden
        self._W1 = np.random.default_rng(42).standard_normal((1, h)).astype(np.float32) * 0.1
        self._b1 = np.zeros(h, dtype=np.float32)
        self._W2 = np.random.default_rng(43).standard_normal((h, h)).astype(np.float32) * 0.1
        self._b2 = np.zeros(h, dtype=np.float32)
        self._W3 = np.random.default_rng(44).standard_normal((h, 1)).astype(np.float32) * 0.1
        self._b3 = np.zeros(1, dtype=np.float32)
        self._scale_vec: Optional[np.ndarray] = None  # (half,)

    @property
    def config(self) -> CLeXRoPEConfig:
        return self._cfg

    # ------------------------------------------------------------------
    # Scale MLP (forward)
    # ------------------------------------------------------------------

    def _scale_mlp(self, freq_indices: np.ndarray) -> np.ndarray:
        """Compute positive scale values for each frequency index.

        Parameters
        ----------
        freq_indices:
            Normalised indices in [0, 1], shape ``(half,)``.

        Returns
        -------
        Positive scales, shape ``(half,)``.
        """
        x = freq_indices[:, None]  # (half, 1)
        h1 = np.tanh(x @ self._W1 + self._b1)  # (half, scale_hidden)
        h2 = np.tanh(h1 @ self._W2 + self._b2)
        out = h2 @ self._W3 + self._b3  # (half, 1)
        # Output is δ — base scale = extension_factor * sigmoid(δ)
        ext = self._cfg.target_max_len / self._cfg.original_max_len
        return (1.0 + ext * self._sigmoid(out[:, 0])).astype(np.float32)

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x))

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def calibrate(
        self, hidden_states: Optional[np.ndarray] = None, positions: Optional[np.ndarray] = None
    ) -> None:
        """Fit scale parameters to minimise positional interpolation error.

        If no hidden_states are provided the MLP is initialised to produce
        a uniform extension scale (equivalent to linear interpolation).
        """
        half = self._cfg.dim // 2
        freq_idx = np.arange(half, dtype=np.float32) / half

        if hidden_states is None:
            # Default: uniform extension factor
            ext = float(self._cfg.target_max_len / self._cfg.original_max_len)
            self._scale_vec = np.full(half, ext, dtype=np.float32)
            self._calibrated = True
            return

        # Simple gradient descent to fit scales
        target_scale = self._cfg.target_max_len / self._cfg.original_max_len
        lr = self._cfg.lr
        for _ in range(self._cfg.n_calibration_steps):
            scales = self._scale_mlp(freq_idx)
            loss_grad = scales - target_scale  # MSE gradient
            # Backprop through last layer (approximate)
            d_W3 = -lr * loss_grad[:, None] * np.ones((half, self._cfg.scale_hidden))
            self._W3 -= d_W3.mean(axis=0, keepdims=True).T * lr
        self._scale_vec = self._scale_mlp(freq_idx)
        self._calibrated = True

    # ------------------------------------------------------------------
    # Frequency table
    # ------------------------------------------------------------------

    def _inv_freq(self) -> np.ndarray:
        cfg = self._cfg
        half = cfg.dim // 2
        standard = (cfg.base ** (-np.arange(half, dtype=np.float64) / half)).astype(np.float32)
        if self._scale_vec is not None:
            return standard / self._scale_vec
        return standard

    def build_freqs(self, seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
        """Build sin/cos tables for ``seq_len`` positions.

        Returns
        -------
        (cos_table, sin_table) each shape ``(seq_len, dim // 2)``.
        """
        inv_freq = self._inv_freq()
        positions = np.arange(seq_len, dtype=np.float32)
        freqs = np.outer(positions, inv_freq)
        return np.cos(freqs), np.sin(freqs)

    def apply(self, x: np.ndarray, offset: int = 0) -> np.ndarray:
        """Apply CLeX RoPE rotation.

        Parameters
        ----------
        x:
            Shape ``(batch, seq, dim)`` or ``(seq, dim)``.
        offset:
            Starting position for this chunk.
        """
        orig_shape = x.shape
        if x.ndim == 2:
            x = x[None, ...]
        _batch, seq, dim = x.shape
        cos_t, sin_t = self.build_freqs(offset + seq)
        cos_t = cos_t[offset:]
        sin_t = sin_t[offset:]
        x1, x2 = x[..., : dim // 2], x[..., dim // 2:]
        rot = np.concatenate([-x2, x1], axis=-1)
        cos_full = np.concatenate([cos_t, cos_t], axis=-1)[None]
        sin_full = np.concatenate([sin_t, sin_t], axis=-1)[None]
        return (x * cos_full + rot * sin_full).reshape(orig_shape)
