"""squish/attention/grouped_rope.py

GroupedRoPE — Per-Head Frequency Grouping (Llama 3 / DeepSeek Style).

Reference
---------
Touvron et al. "Llama 3 Technical Report." Meta AI, 2024.
Su et al. "RoFormer: Enhanced Transformer with Rotary Position Embedding."
Neurocomputing 2023 (arXiv:2104.09864).

Algorithm
---------
Standard RoPE applies the same frequency schedule to all attention heads.
GroupedRoPE partitions heads into groups, applying a distinct base frequency
(or scale factor) to each group.  This enables different heads to specialise
in different positional resolution ranges:

* Low-frequency heads: coarse global context.
* High-frequency heads: fine local context.

DeepSeek-V2/V3 uses a similar trick — "Multi-head Latent Attention" (MLA)
with per-head rope theta.

Key properties
--------------
* ``n_heads`` total heads split into ``n_groups`` groups.
* ``group_bases`` — per-group base frequency (length ``n_groups``).
* ``build_all_freqs(seq_len)`` returns sin/cos tables per head.
* ``apply(x, offset)`` rotates the full query/key tensor.
* NumPy-only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

__all__ = [
    "GroupedRoPEConfig",
    "GroupedRoPE",
]


@dataclass
class GroupedRoPEConfig:
    """Configuration for :class:`GroupedRoPE`.

    Attributes:
        n_heads: Total number of attention heads.
        head_dim: Dimension per head (must be even).
        n_groups: Number of head frequency groups.
        group_bases: Per-group RoPE base frequency.  Must have length
            ``n_groups``.  Defaults to geometrically spaced 1000–100000.
    """

    n_heads: int = 32
    head_dim: int = 128
    n_groups: int = 4
    group_bases: Optional[List[float]] = None

    def __post_init__(self) -> None:
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even")
        if self.n_heads % self.n_groups != 0:
            raise ValueError("n_heads must be divisible by n_groups")
        if self.group_bases is None:
            # Geometrically spaced bases
            self.group_bases = list(
                np.geomspace(1000.0, 100000.0, self.n_groups).tolist()
            )
        if len(self.group_bases) != self.n_groups:
            raise ValueError("group_bases must have length n_groups")


class GroupedRoPE:
    """Per-head frequency group RoPE.

    Parameters
    ----------
    config:
        GroupedRoPEConfig.
    """

    def __init__(self, config: Optional[GroupedRoPEConfig] = None) -> None:
        self._cfg = config or GroupedRoPEConfig()
        self._heads_per_group = self._cfg.n_heads // self._cfg.n_groups
        # Precompute per-group inv_freq
        self._inv_freqs = self._build_inv_freqs()

    @property
    def config(self) -> GroupedRoPEConfig:
        return self._cfg

    def _build_inv_freqs(self) -> List[np.ndarray]:
        """Build one inv_freq vector per group."""
        half = self._cfg.head_dim // 2
        inv_freqs = []
        for base in self._cfg.group_bases:  # type: ignore[union-attr]
            i = np.arange(half, dtype=np.float64)
            inv_freq = (base ** (-i / half)).astype(np.float32)
            inv_freqs.append(inv_freq)
        return inv_freqs

    def build_all_freqs(self, seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
        """Build sin/cos tables covering all heads.

        Returns
        -------
        (cos_all, sin_all) each of shape ``(n_heads, seq_len, head_dim // 2)``.
        """
        positions = np.arange(seq_len, dtype=np.float32)
        cos_all = np.zeros((self._cfg.n_heads, seq_len, self._cfg.head_dim // 2), dtype=np.float32)
        sin_all = np.zeros_like(cos_all)
        for g, inv_freq in enumerate(self._inv_freqs):
            freqs = np.outer(positions, inv_freq)  # (seq, half)
            h_start = g * self._heads_per_group
            h_end = h_start + self._heads_per_group
            cos_all[h_start:h_end] = np.cos(freqs)[None, ...]
            sin_all[h_start:h_end] = np.sin(freqs)[None, ...]
        return cos_all, sin_all

    def apply(self, x: np.ndarray, offset: int = 0) -> np.ndarray:
        """Apply grouped RoPE to query or key tensor.

        Parameters
        ----------
        x:
            Shape ``(batch, n_heads, seq, head_dim)``.
        offset:
            Starting position index.

        Returns
        -------
        Rotated tensor of same shape.
        """
        batch, n_heads, seq, head_dim = x.shape
        if n_heads != self._cfg.n_heads:
            raise ValueError(f"Expected n_heads={self._cfg.n_heads}, got {n_heads}")

        cos_all, sin_all = self.build_all_freqs(offset + seq)
        cos_all = cos_all[:, offset:, :]  # (H, seq, D//2)
        sin_all = sin_all[:, offset:, :]

        x1 = x[..., : head_dim // 2]
        x2 = x[..., head_dim // 2:]
        rot = np.concatenate([-x2, x1], axis=-1)  # (B, H, seq, D)

        cos_full = np.concatenate([cos_all, cos_all], axis=-1)[None, ...]  # (1, H, seq, D)
        sin_full = np.concatenate([sin_all, sin_all], axis=-1)[None, ...]
        return x * cos_full + rot * sin_full
