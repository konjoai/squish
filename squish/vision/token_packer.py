"""TokenPacker: efficient fixed-size visual projector via region-to-point attention.

Li et al. (arXiv 2407.09985, 2024) propose a cross-attention visual projector
that maps N variable-count patch tokens to a fixed M-token compact representation
through a learned set of M anchor tokens.  Each anchor attends over its assigned
spatial region of the patch sequence; the result is a resolution-invariant M-token
visual embedding ready for any LLM decoder — enabling InternVL2-style arbitrary-
resolution inputs without quadratic token count.

This NumPy stub implements the cross-attention computation faithfully.

Reference: Li et al., "TokenPacker: Efficient Visual Projector for Multimodal
LLM", arXiv 2407.09985, 2024.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

__all__ = [
    "TokenPackerConfig",
    "TokenPackerResult",
    "TokenPacker",
]


@dataclass
class TokenPackerConfig:
    """Configuration for :class:`TokenPacker`.

    Attributes:
        n_anchor: Number of output anchor tokens M (default 64).
        hidden_dim: Dimension of patch and anchor token embeddings.
        n_heads: Number of attention heads for the region-to-point cross-attention.
        dropout: Attention dropout probability (applied only during forward passes
            that accept a training-mode flag; stubs here ignore it).
        seed: RNG seed for anchor initialisation.
    """

    n_anchor: int = 64
    hidden_dim: int = 256
    n_heads: int = 4
    dropout: float = 0.0
    seed: int = 0

    def __post_init__(self) -> None:
        if self.n_anchor < 1:
            raise ValueError(f"n_anchor must be ≥ 1, got {self.n_anchor}")
        if self.hidden_dim < 1:
            raise ValueError(f"hidden_dim must be ≥ 1, got {self.hidden_dim}")
        if self.n_heads < 1:
            raise ValueError(f"n_heads must be ≥ 1, got {self.n_heads}")
        if self.hidden_dim % self.n_heads != 0:
            raise ValueError(
                f"hidden_dim ({self.hidden_dim}) must be divisible by n_heads ({self.n_heads})"
            )


@dataclass
class TokenPackerResult:
    """Output of one :meth:`TokenPacker.pack` call.

    Attributes:
        packed: Anchor token matrix of shape ``(n_anchor, hidden_dim)``.
        attn_weights: Attention weights of shape ``(n_heads, n_anchor, n_patches)``.
    """

    packed: np.ndarray
    attn_weights: np.ndarray

    @property
    def n_anchor(self) -> int:
        return self.packed.shape[0]

    @property
    def n_patches_in(self) -> int:
        return self.attn_weights.shape[2]


class TokenPacker:
    """Map variable-count patch tokens to a fixed M-anchor representation.

    The packer maintains a fixed set of M learnable anchor queries that cross-
    attend over the N patch tokens via scaled dot-product attention.

    Usage::

        cfg = TokenPackerConfig(n_anchor=64, hidden_dim=256, n_heads=4)
        packer = TokenPacker(cfg)  # anchors randomly initialised
        # patches: (N, hidden_dim)
        result = packer.pack(patches)
        # result.packed: (64, hidden_dim)  ← fixed size regardless of N
    """

    def __init__(
        self,
        config: TokenPackerConfig,
        anchors: Optional[np.ndarray] = None,
    ) -> None:
        self.config = config
        rng = np.random.default_rng(config.seed)
        scale = 1.0 / np.sqrt(config.hidden_dim)
        if anchors is not None:
            self._anchors = np.asarray(anchors, dtype=np.float32)
        else:
            self._anchors = (
                rng.standard_normal((config.n_anchor, config.hidden_dim)).astype(np.float32) * scale
            )
        # Projection weights (Q, K, V) — each (hidden_dim, hidden_dim)
        self._W_q = rng.standard_normal((config.hidden_dim, config.hidden_dim)).astype(np.float32) * scale
        self._W_k = rng.standard_normal((config.hidden_dim, config.hidden_dim)).astype(np.float32) * scale
        self._W_v = rng.standard_normal((config.hidden_dim, config.hidden_dim)).astype(np.float32) * scale
        self._W_o = rng.standard_normal((config.hidden_dim, config.hidden_dim)).astype(np.float32) * scale

    # ------------------------------------------------------------------
    # Primary API
    # ------------------------------------------------------------------

    def pack(self, patches: np.ndarray) -> TokenPackerResult:
        """Map *patches* to the fixed anchor representation.

        Parameters
        ----------
        patches:
            Visual patch tokens of shape ``(n_patches, hidden_dim)``.
        """
        patches = np.asarray(patches, dtype=np.float32)
        n_p, d = patches.shape
        n_a = self.config.n_anchor
        n_h = self.config.n_heads
        head_dim = d // n_h

        # Linear projections
        Q = self._anchors @ self._W_q   # (n_anchor, d)
        K = patches @ self._W_k          # (n_patches, d)
        V = patches @ self._W_v          # (n_patches, d)

        # Reshape to multi-head
        Q = Q.reshape(n_a, n_h, head_dim).transpose(1, 0, 2)    # (h, n_a, hd)
        K = K.reshape(n_p, n_h, head_dim).transpose(1, 0, 2)    # (h, n_p, hd)
        V = V.reshape(n_p, n_h, head_dim).transpose(1, 0, 2)    # (h, n_p, hd)

        scale = 1.0 / np.sqrt(head_dim)
        attn_logits = np.einsum("hqd,hkd->hqk", Q, K) * scale    # (h, n_a, n_p)
        attn_weights = self._softmax(attn_logits)                  # (h, n_a, n_p)
        out = np.einsum("hqk,hkd->hqd", attn_weights, V)          # (h, n_a, hd)
        out = out.transpose(1, 0, 2).reshape(n_a, d)              # (n_a, d)
        packed = out @ self._W_o                                   # (n_a, d)

        return TokenPackerResult(packed=packed, attn_weights=attn_weights)

    def set_anchors(self, anchors: np.ndarray) -> None:
        """Replace the learnable anchor vectors."""
        self._anchors = np.asarray(anchors, dtype=np.float32)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        x_max = x.max(axis=-1, keepdims=True)
        e = np.exp(x - x_max)
        return e / (e.sum(axis=-1, keepdims=True) + 1e-9)
