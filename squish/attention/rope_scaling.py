# [Experimental] This module is part of Squish v6+ (Waves 19–26).
# Proof-of-concept quality: API and behaviour may change without notice.
# For stable core functionality see Waves 1–12 in MODULES.md.
#!/usr/bin/env python3
"""
squish/rope_scaling.py

RoPEScaling — Dynamic RoPE context extension (NTK-aware / YaRN / LongRoPE).

This module implements three families of Rotary Position Embedding (RoPE)
context-extension techniques that allow models trained on short contexts to
extrapolate to longer sequences without fine-tuning (or with minimal fine-tuning):

NTK-aware scaling
    Adjusts the RoPE base frequency so that high-frequency components are
    unchanged and low-frequency components are smoothly stretched.

        "NTK-Aware Scaled RoPE allows LLaMA models to have extended context
        size without any fine-tuning and minimal perplexity degradation."
        Reddit post by u/bloc97, 2023.
        https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/

YaRN
    Interpolates per-dimension between linear position interpolation (for low
    frequencies) and no scaling (for high frequencies) using a smooth ramp.

        Peng et al., "YaRN: Efficient Context Window Extension of Large
        Language Models", ICLR 2024.
        https://arxiv.org/abs/2309.00071

LongRoPE
    Applies per-dimension rescaling factors that vary smoothly across the
    frequency spectrum, assigning more context extension to lower-frequency
    dimensions and leaving high-frequency dimensions unscaled.

        Ding et al., "LongRoPE: Extending LLM Context Window Beyond 2 Million
        Tokens", 2024.
        https://arxiv.org/abs/2402.13753

All implementations are pure NumPy; no MLX or PyTorch required.

Example usage::

    import numpy as np
    from squish.attention.rope_scaling import RoPEConfig, create_rope_scaler

    config = RoPEConfig(
        head_dim=128,
        base_theta=10000.0,
        original_max_len=4096,
        target_max_len=32768,
        method="yarn",
    )
    scaler = create_rope_scaler(config)

    x = np.random.randn(16, 32, 128).astype(np.float32)  # (seq, heads, dim)
    position_ids = np.arange(16)
    out = scaler.apply(x, position_ids)  # same shape as x
"""

from __future__ import annotations

__all__ = [
    "RoPEConfig",
    "RoPEScaler",
    "NTKScaler",
    "YaRNScaler",
    "LongRoPEScaler",
    "create_rope_scaler",
]

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

# Valid method identifiers.
_VALID_METHODS = ("ntk", "yarn", "longrope")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class RoPEConfig:
    """Configuration for RoPE context-extension scaling.

    Attributes:
        head_dim:          Head dimension (must be even).
        base_theta:        Base frequency for the RoPE sinusoidal functions.
                           Typically 10 000 (original) or 500 000 (Llama 3).
        original_max_len:  The sequence length the model was originally
                           trained on (e.g. 4096 for Llama 2-7B).
        target_max_len:    The desired extended context length.
        method:            Scaling algorithm: ``"ntk"``, ``"yarn"``, or
                           ``"longrope"``.
        scale_factor:      Ratio ``target_max_len / original_max_len``.
                           Computed automatically when ``None``.
    """

    head_dim: int = 128
    base_theta: float = 10000.0
    original_max_len: int = 4096
    target_max_len: int = 32768
    method: str = "ntk"
    scale_factor: float | None = None

    def __post_init__(self) -> None:
        if self.head_dim < 2 or self.head_dim % 2 != 0:
            raise ValueError(
                f"head_dim must be a positive even integer, got {self.head_dim}"
            )
        if self.base_theta <= 0.0:
            raise ValueError(f"base_theta must be > 0, got {self.base_theta}")
        if self.original_max_len < 1:
            raise ValueError(
                f"original_max_len must be >= 1, got {self.original_max_len}"
            )
        if self.target_max_len < self.original_max_len:
            raise ValueError(
                f"target_max_len ({self.target_max_len}) must be >= "
                f"original_max_len ({self.original_max_len})"
            )
        if self.method not in _VALID_METHODS:
            raise ValueError(
                f"method must be one of {_VALID_METHODS}, got {self.method!r}"
            )
        if self.scale_factor is None:
            self.scale_factor = self.target_max_len / self.original_max_len
        if self.scale_factor < 1.0:
            raise ValueError(
                f"scale_factor must be >= 1.0, got {self.scale_factor}"
            )


# ---------------------------------------------------------------------------
# Base scaler
# ---------------------------------------------------------------------------

class RoPEScaler:
    """Base class for RoPE scalers.

    Subclasses override :meth:`get_freqs` to implement their specific
    frequency-scaling strategy.  The :meth:`apply` method is shared and
    delegates to :meth:`get_freqs` for angle computation.

    The rotation uses the *split-half* convention (as in HuggingFace
    transformers): the head vector is split into two halves ``(x1, x2)``,
    and the rotation is applied as::

        rotate_half(x) = concat(−x2, x1)
        output = x * cos(θ) + rotate_half(x) * sin(θ)

    Args:
        config: A :class:`RoPEConfig` instance.
    """

    def __init__(self, config: RoPEConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------
    # Public API — override get_freqs in subclasses
    # ------------------------------------------------------------------

    def get_freqs(self, seq_len: int) -> np.ndarray:
        """Compute angle values for positions ``[0, seq_len)``.

        Args:
            seq_len: Number of sequential positions to compute angles for.

        Returns:
            Array of shape ``(seq_len, head_dim // 2)`` where entry ``[p, i]``
            is the angle ``position_p × theta_i`` after any scaling.
        """
        freqs = self._unscaled_base_freqs()           # (half,)
        positions = np.arange(seq_len, dtype=np.float64)
        return np.outer(positions, freqs)             # (seq_len, half)

    def apply(self, x: np.ndarray, position_ids: np.ndarray) -> np.ndarray:
        """Apply rotary position embeddings to *x*.

        Args:
            x:            Shape ``(seq_len, n_heads, head_dim)``.
            position_ids: 1-D integer array of length ``seq_len`` giving the
                          absolute position of each token.

        Returns:
            Array of the same shape and dtype as *x* with RoPE applied.
        """
        seq_len, n_heads, head_dim = x.shape
        half = head_dim // 2

        pos_ids = np.asarray(position_ids, dtype=np.int64)
        if pos_ids.ndim != 1 or pos_ids.shape[0] != seq_len:
            raise ValueError(
                f"position_ids must be 1-D with length {seq_len}, "
                f"got shape {pos_ids.shape}"
            )

        # Retrieve angles for the required positions via get_freqs.
        max_pos = int(pos_ids.max()) + 1
        all_freqs = self.get_freqs(max_pos)           # (max_pos, half)
        angles = all_freqs[pos_ids]                   # (seq_len, half)

        cos_a = np.cos(angles).astype(x.dtype)       # (seq_len, half)
        sin_a = np.sin(angles).astype(x.dtype)

        # Broadcast over the n_heads axis.
        cos_a = cos_a[:, np.newaxis, :]               # (seq_len, 1, half)
        sin_a = sin_a[:, np.newaxis, :]

        # Tile cos/sin to cover both halves of head_dim identically.
        cos_full = np.concatenate([cos_a, cos_a], axis=-1)  # (seq_len, 1, head_dim)
        sin_full = np.concatenate([sin_a, sin_a], axis=-1)

        # Split-half rotation: rotate_half(x) = concat(−x[..., half:], x[..., :half])
        x1 = x[..., :half]
        x2 = x[..., half:]
        rotated = np.concatenate([-x2, x1], axis=-1) # (seq_len, n_heads, head_dim)

        return (x * cos_full + rotated * sin_full).astype(x.dtype)

    # ------------------------------------------------------------------
    # Protected helpers
    # ------------------------------------------------------------------

    def _unscaled_base_freqs(self) -> np.ndarray:
        """Standard RoPE per-dimension frequencies without any scaling.

        Returns an array of shape ``(head_dim // 2,)`` where entry ``i`` is::

            theta_i = 1 / base_theta ^ (2i / head_dim)
        """
        dim = self.config.head_dim
        exponents = np.arange(0, dim, 2, dtype=np.float64) / dim
        return 1.0 / (self.config.base_theta ** exponents)


# ---------------------------------------------------------------------------
# NTK-aware scaling
# ---------------------------------------------------------------------------

class NTKScaler(RoPEScaler):
    """NTK-aware RoPE scaling.

    Rescales the base frequency by raising the base theta by
    ``scale_factor ^ (dim / (dim - 2))``.  This leaves high-frequency
    components (small dimension index) unchanged and progressively stretches
    lower-frequency components, matching the intuition from Neural Tangent
    Kernel theory.

    Effective scaled theta per dimension::

        new_base = base_theta * scale_factor ^ (dim / (dim - 2))
        theta_i  = 1 / new_base ^ (2i / dim)

    At ``i = 0`` (highest frequency): theta_0 = 1 (unchanged).
    At ``i = dim/2 − 1`` (lowest frequency): theta_i ≈ original / scale_factor.
    """

    def get_freqs(self, seq_len: int) -> np.ndarray:
        cfg = self.config
        dim = cfg.head_dim
        assert cfg.scale_factor is not None  # resolved in RoPEConfig.__post_init__

        # Rescale base theta via the NTK formula.
        scaled_base = cfg.base_theta * (cfg.scale_factor ** (dim / (dim - 2)))

        exponents = np.arange(0, dim, 2, dtype=np.float64) / dim
        freqs = 1.0 / (scaled_base ** exponents)          # (half,)

        positions = np.arange(seq_len, dtype=np.float64)
        return np.outer(positions, freqs)                  # (seq_len, half)


# ---------------------------------------------------------------------------
# YaRN scaling
# ---------------------------------------------------------------------------

class YaRNScaler(RoPEScaler):
    """YaRN RoPE scaling (Yet Another RoPE extensioN).

    Partitions frequency dimensions into three regions based on wavelength:

    * **High-frequency** (short wavelength): no scaling applied.
    * **Low-frequency**  (long wavelength):  full position interpolation
      (equivalent to dividing frequency by ``scale_factor``).
    * **Transition region**: smooth linear ramp between the two extremes.

    The ramp is defined in terms of the dimension's wavelength
    ``λ_i = 2π / theta_i`` relative to the original context length:

    * ``λ_i < original_max_len / high_freq_factor`` → high-freq, scale = 1
    * ``λ_i > original_max_len / low_freq_factor``  → low-freq,  scale = 1/s
    * Otherwise: ``smooth = (original_max_len/λ_i − low_freq_factor) /
                             (high_freq_factor − low_freq_factor)``
      and ``scaled_freq = (1 − smooth) × freq/s + smooth × freq``

    Args:
        config:           A :class:`RoPEConfig` instance.
        low_freq_factor:  Wavelength threshold dividing mid/low freq regions.
        high_freq_factor: Wavelength threshold dividing high/mid freq regions.
    """

    def __init__(
        self,
        config: RoPEConfig,
        low_freq_factor: float = 1.0,
        high_freq_factor: float = 4.0,
    ) -> None:
        super().__init__(config)
        if low_freq_factor <= 0.0:
            raise ValueError(
                f"low_freq_factor must be > 0, got {low_freq_factor}"
            )
        if high_freq_factor <= low_freq_factor:
            raise ValueError(
                f"high_freq_factor ({high_freq_factor}) must be greater than "
                f"low_freq_factor ({low_freq_factor})"
            )
        self.low_freq_factor = low_freq_factor
        self.high_freq_factor = high_freq_factor

    def get_freqs(self, seq_len: int) -> np.ndarray:
        cfg = self.config
        original_freqs = self._unscaled_base_freqs()     # (half,)

        low_wavelen = cfg.original_max_len / self.low_freq_factor
        high_wavelen = cfg.original_max_len / self.high_freq_factor

        # Wavelength per dimension: λ_i = 2π / theta_i.
        wavelens = (2.0 * math.pi) / original_freqs      # (half,)

        # Ramp value in [0, 1]: 0 → full interpolation, 1 → no scaling.
        smooth = (cfg.original_max_len / wavelens - self.low_freq_factor) / (
            self.high_freq_factor - self.low_freq_factor
        )
        smooth = np.clip(smooth, 0.0, 1.0)

        # Per-dimension scaled frequencies.
        interp_freqs = original_freqs / cfg.scale_factor  # full interpolation
        scaled_freqs = (1.0 - smooth) * interp_freqs + smooth * original_freqs

        # Override with exact per-region values for clean boundary behaviour.
        scaled_freqs = np.where(wavelens > low_wavelen, interp_freqs, scaled_freqs)
        scaled_freqs = np.where(wavelens < high_wavelen, original_freqs, scaled_freqs)

        positions = np.arange(seq_len, dtype=np.float64)
        return np.outer(positions, scaled_freqs)          # (seq_len, half)


# ---------------------------------------------------------------------------
# LongRoPE scaling
# ---------------------------------------------------------------------------

class LongRoPEScaler(RoPEScaler):
    """LongRoPE — position-dependent per-dimension RoPE rescaling.

    The published LongRoPE paper finds per-dimension rescaling factors via
    evolutionary search on calibration data.  This implementation uses a
    closed-form approximation: a monotonically increasing ramp of scale
    factors across the frequency spectrum.

    Specifically, each dimension ``i ∈ [0, half)`` receives a scale factor::

        dim_scale[i] = 1 + (scale_factor − 1) × i / (half − 1)

    This assigns scale = 1 (no change) to the highest-frequency dimension
    (i = 0) and scale = scale_factor (full extension) to the lowest-frequency
    dimension (i = half − 1), with a linear ramp in between.  The effective
    frequency is then::

        scaled_freq[i] = original_freq[i] / dim_scale[i]

    This distributes context extension non-uniformly, concentrating the
    position stretching where the model benefits most (low-frequency
    components that encode long-range structure).
    """

    def get_freqs(self, seq_len: int) -> np.ndarray:
        cfg = self.config
        half = cfg.head_dim // 2
        assert cfg.scale_factor is not None  # resolved in RoPEConfig.__post_init__
        original_freqs = self._unscaled_base_freqs()  # (half,)

        # Per-dimension scale factors: linear ramp from 1 → scale_factor.
        if half > 1:
            dim_scale = 1.0 + (cfg.scale_factor - 1.0) * (
                np.arange(half, dtype=np.float64) / (half - 1)
            )
        else:
            dim_scale = np.array([cfg.scale_factor], dtype=np.float64)

        # Compress positions in each dimension by its assigned scale factor.
        scaled_freqs = original_freqs / dim_scale      # (half,)

        positions = np.arange(seq_len, dtype=np.float64)
        return np.outer(positions, scaled_freqs)        # (seq_len, half)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_rope_scaler(config: RoPEConfig) -> RoPEScaler:
    """Instantiate the appropriate :class:`RoPEScaler` for *config*.

    Args:
        config: A fully-validated :class:`RoPEConfig` instance.

    Returns:
        An :class:`NTKScaler`, :class:`YaRNScaler`, or
        :class:`LongRoPEScaler` depending on ``config.method``.

    Raises:
        ValueError: if ``config.method`` is not a known method identifier
                    (this should have been caught by :class:`RoPEConfig`
                    ``__post_init__``, but is re-checked for safety).
    """
    if config.method == "ntk":
        return NTKScaler(config)
    if config.method == "yarn":
        return YaRNScaler(config)
    if config.method == "longrope":
        return LongRoPEScaler(config)
    raise ValueError(  # pragma: no cover
        f"Unknown RoPE scaling method {config.method!r}. "
        f"Expected one of {_VALID_METHODS}."
    )
