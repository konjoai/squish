"""adaptive_rope.py — Adaptive Rotary Position Embedding (RoPE) Scaling

Standard RoPE uses a fixed base frequency (θ = 10000) calibrated for sequence
lengths seen during training. This sub-optimal for:

  - Short sequences (< 512 tokens): rotations resolve too slowly → under-rotated
  - Long sequences (> training context): rotations wrap/aliase → extrapolation fails

AdaptiveRoPE selects the optimal frequency base per request at runtime:

    seq_len < SHORT_THRESHOLD (512):
        base = SHORT_BASE (500) — tighter rotation, better short-range encoding

    seq_len in [512, long_threshold (4096)]:
        base = STANDARD_BASE (10000) — nominal training configuration

    seq_len > long_threshold:
        YaRN-style scaling: base ← base × (seq_len / max_trained_len) ^ (d/(d-2))
        NTK interpolation:  base ← base_ntk × (seq_len / trained_len) ^ (2d/(d-2))

The class recomputes cos/sin caches lazily and caches them per (seq_len, d_model)
to avoid recomputation within a streaming decode.

Scale modes:
  STANDARD: fixed θ = 10000 regardless of sequence length
  DYNAMIC:  select base from SHORT/STANDARD/YARN rules above (recommended)
  YARN:     always use YaRN scaling (forces YaRN for all lengths)
  NTK:      always use NTK interpolation

Based on:
  - RoPE: "RoFormer: Enhanced Transformer with Rotary Position Embedding"
    Su et al., 2021
  - YaRN: "YaRN: Efficient Context Window Extension of Large Language Models"
    Peng et al., 2023
  - NTK-aware interpolation: https://reddit.com/r/LocalLLaMA/comments/14lz7j5
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

import numpy as np


class RoPEScaleMode(str, Enum):
    STANDARD = "standard"   # Fixed θ = 10000 everywhere
    DYNAMIC = "dynamic"     # Auto-select by seq_len (recommended)
    YARN = "yarn"           # Always apply YaRN scaling
    NTK = "ntk"             # Always apply NTK interpolation


@dataclass
class AdaptiveRoPEConfig:
    """Configuration for AdaptiveRoPE.

    Args:
        mode:             Scale mode (see RoPEScaleMode).
        base:             Standard RoPE base frequency (default 10000).
        short_base:       Base for seq_len < short_threshold.
        short_threshold:  Sequence length below which short_base is used.
        long_threshold:   Sequence length above which long-context scaling kicks in.
        max_trained_len:  Training context length (needed for YaRN/NTK scaling).
        dim:              Head dimension (d_k). Required when computing caches.
        yarn_beta_fast:   YaRN β_fast parameter (default 32).
        yarn_beta_slow:   YaRN β_slow parameter (default 1).
        yarn_scale:       YaRN scale factor s (default 1 = auto from seq_len ratio).
                          If 0 or None, computed from seq_len / max_trained_len.
    """
    mode: RoPEScaleMode = RoPEScaleMode.DYNAMIC
    base: float = 10000.0
    short_base: float = 500.0
    short_threshold: int = 512
    long_threshold: int = 4096
    max_trained_len: int = 4096
    dim: int = 128
    yarn_beta_fast: float = 32.0
    yarn_beta_slow: float = 1.0
    yarn_scale: Optional[float] = None


class AdaptiveRoPE:
    """Per-request RoPE with dynamic base frequency selection.

    Usage:
        rope = AdaptiveRoPE(AdaptiveRoPEConfig(dim=128))
        cos, sin = rope.get_cos_sin(seq_len=2048, dtype=np.float32)
        # cos, sin: (seq_len, dim//2) each

        # Apply to a query matrix q: (batch, n_heads, seq_len, d_k)
        q_rotated = rope.apply(q, cos, sin)
    """

    def __init__(self, config: Optional[AdaptiveRoPEConfig] = None) -> None:
        self.config = config or AdaptiveRoPEConfig()
        # Cache: (seq_len, effective_base) → (cos, sin) arrays
        self._cache: dict[tuple[int, float], tuple[np.ndarray, np.ndarray]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_rope_scale(self, seq_len: int) -> float:
        """Return the effective RoPE base for a given sequence length.

        Returns:
            float — the effective θ to use for frequency computation.
        """
        cfg = self.config
        mode = cfg.mode

        if mode == RoPEScaleMode.STANDARD:
            return cfg.base

        if mode == RoPEScaleMode.DYNAMIC:
            if seq_len < cfg.short_threshold:
                return cfg.short_base
            if seq_len <= cfg.long_threshold:
                return cfg.base
            return self._yarn_base(seq_len)

        if mode == RoPEScaleMode.YARN:
            return self._yarn_base(seq_len)

        if mode == RoPEScaleMode.NTK:
            return self._ntk_base(seq_len)

        return cfg.base

    def get_cos_sin(
        self,
        seq_len: int,
        dtype: type = np.float32,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute (or return cached) cos/sin position tables.

        Args:
            seq_len: Maximum sequence length to compute tables for.
            dtype:   Output dtype (default float32).

        Returns:
            cos, sin: each shape (seq_len, dim // 2)
        """
        base = self.get_rope_scale(seq_len)
        key = (seq_len, base)
        if key not in self._cache:
            self._cache[key] = self._build_cos_sin(seq_len, base, dtype)
        return self._cache[key]

    def apply(
        self,
        x: np.ndarray,       # (..., seq_len, d_k)
        cos: np.ndarray,     # (seq_len, d_k // 2)
        sin: np.ndarray,     # (seq_len, d_k // 2)
    ) -> np.ndarray:
        """Apply rotary embeddings to x.

        Args:
            x:   Query or Key matrix, shape (..., seq_len, d_k).
            cos: Cosine table (seq_len, d_k // 2).
            sin: Sine table   (seq_len, d_k // 2).

        Returns:
            x_rotated with same shape as x.
        """
        d_k = x.shape[-1]
        if cos.shape[-1] * 2 != d_k:
            raise ValueError(
                f"cos has d={cos.shape[-1]*2}, but x has d_k={d_k}"
            )
        # Split into even/odd pairs
        x1 = x[..., : d_k // 2]    # even dimensions
        x2 = x[..., d_k // 2 :]    # odd dimensions
        # Broadcast cos/sin over batch dimensions
        cos_b = cos[..., :]   # (seq_len, d_k//2)
        sin_b = sin[..., :]
        x_rot = np.concatenate(
            [
                x1 * cos_b - x2 * sin_b,
                x1 * sin_b + x2 * cos_b,
            ],
            axis=-1,
        )
        return x_rot.astype(x.dtype)

    def clear_cache(self) -> None:
        """Evict all cached cos/sin tables."""
        self._cache.clear()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build_cos_sin(
        self,
        seq_len: int,
        base: float,
        dtype: type,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Build cos/sin position tables for a given base and seq_len."""
        cfg = self.config
        d = cfg.dim
        if d % 2 != 0:
            raise ValueError(f"dim must be even, got {d}")

        # θ_i = 1 / base^(2i/d),  i = 0 … d//2-1
        half_d = d // 2
        inv_freq = 1.0 / (
            base ** (np.arange(0, d, 2, dtype=np.float64) / d)
        )  # (half_d,)

        positions = np.arange(seq_len, dtype=np.float64)   # (seq_len,)
        angles = np.outer(positions, inv_freq)               # (seq_len, half_d)
        cos = np.cos(angles).astype(dtype)
        sin = np.sin(angles).astype(dtype)
        return cos, sin

    def _yarn_base(self, seq_len: int) -> float:
        """YaRN-style base scaling for long sequences."""
        cfg = self.config
        s = cfg.yarn_scale
        if not s:
            s = seq_len / cfg.max_trained_len
        if s <= 1.0:
            return cfg.base
        d = cfg.dim
        # YaRN base scaling: base_yarn = base × s^(d/(d-2))
        exponent = d / max(d - 2, 1)
        return cfg.base * (s ** exponent)

    def _ntk_base(self, seq_len: int) -> float:
        """NTK-aware interpolation base scaling."""
        cfg = self.config
        s = seq_len / cfg.max_trained_len
        if s <= 1.0:
            return cfg.base
        d = cfg.dim
        # NTK interpolation: base_ntk = base × s^(2d/(d-2))
        exponent = (2 * d) / max(d - 2, 1)
        return cfg.base * (s ** exponent)

    def __repr__(self) -> str:
        cfg = self.config
        return (
            f"AdaptiveRoPE(mode={cfg.mode.value}, base={cfg.base}, "
            f"dim={cfg.dim}, cache_entries={len(self._cache)})"
        )
