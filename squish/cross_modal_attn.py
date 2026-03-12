"""squish/cross_modal_attn.py

CrossModalAttn — Scaled dot-product cross-attention from text query features
to vision key-value features for vision-language model fusion.

In encoder-decoder and cross-modal fusion architectures, text decoder tokens
must attend over the output of a visual encoder (e.g., a ViT patch sequence)
to ground language generation in image content.  This cross-attention
mechanism differs from self-attention in that the queries originate from the
text stream while the keys and values originate from the vision stream, which
may have a different sequence length and be produced by an entirely separate
encoder.

CrossModalAttn implements standard scaled dot-product attention across
modalities.  For each attention head *h* the attention logits are computed as
``scores[h] = text_q[h] @ vision_k[h].T * softmax_scale``, yielding a
``(seq_text, seq_vision)`` logit matrix.  Numerically stable softmax is
applied row-wise by subtracting the per-row maximum before exponentiation,
preventing overflow at long vision sequence lengths.  The output is computed
as ``output[h] = softmax(scores[h]) @ vision_v[h]``, producing a
``(seq_text, head_dim)`` tensor that replaces the text representation with
vision-grounded features.

The current implementation requires ``n_text_heads == n_vision_heads`` — a
simplification that matches the most common deployment pattern where the
language model and visual encoder share the same head count after a linear
projection.  Grouped-query cross-attention (GQA-style, where fewer vision
heads service more text heads) is left for a future extension.

Example usage::

    import numpy as np
    from squish.cross_modal_attn import CrossModalConfig, CrossModalAttention

    cfg  = CrossModalConfig(n_text_heads=8, n_vision_heads=8, head_dim=64)
    attn = CrossModalAttention(cfg)

    text_q   = np.random.randn(8, 32, 64).astype(np.float32)   # 32 text tokens
    vision_k = np.random.randn(8, 196, 64).astype(np.float32)  # 196 patch tokens
    vision_v = np.random.randn(8, 196, 64).astype(np.float32)
    out      = attn.forward(text_q, vision_k, vision_v)
    print(f"output shape: {out.shape}")   # (8, 32, 64)
    print(attn.stats)
"""

from __future__ import annotations

__all__ = ["CrossModalConfig", "CrossModalAttention", "CrossModalStats"]

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class CrossModalConfig:
    """Configuration for cross-modal scaled dot-product attention.

    Attributes:
        n_text_heads:   Number of text query heads.
        n_vision_heads: Number of vision key/value heads.  Must equal
                        ``n_text_heads`` in this implementation.
        head_dim:       Dimension of each attention head.
        softmax_scale:  Scaling factor applied to attention logits before
                        softmax.  Defaults to ``1 / sqrt(head_dim)`` when
                        ``None``.
    """

    n_text_heads: int = 8
    n_vision_heads: int = 8
    head_dim: int = 64
    softmax_scale: Optional[float] = None

    def __post_init__(self) -> None:
        if self.n_text_heads < 1:
            raise ValueError(
                f"n_text_heads must be >= 1, got {self.n_text_heads}"
            )
        if self.n_vision_heads < 1:
            raise ValueError(
                f"n_vision_heads must be >= 1, got {self.n_vision_heads}"
            )
        if self.head_dim < 1:
            raise ValueError(f"head_dim must be >= 1, got {self.head_dim}")
        if self.n_vision_heads != self.n_text_heads:
            raise ValueError(
                f"n_vision_heads ({self.n_vision_heads}) must equal "
                f"n_text_heads ({self.n_text_heads}) in this implementation."
            )
        if self.softmax_scale is None:
            self.softmax_scale = 1.0 / math.sqrt(self.head_dim)


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


@dataclass
class CrossModalStats:
    """Aggregate statistics for a :class:`CrossModalAttention` instance.

    Attributes:
        total_forward_calls:  Total number of :meth:`~CrossModalAttention.forward`
                              invocations.
        total_text_tokens:    Cumulative text query token count across all
                              forward calls.
        total_vision_tokens:  Cumulative vision key/value token count across
                              all forward calls.
    """

    total_forward_calls: int = 0
    total_text_tokens: int = 0
    total_vision_tokens: int = 0


# ---------------------------------------------------------------------------
# Attention module
# ---------------------------------------------------------------------------


class CrossModalAttention:
    """Cross-attention from text queries to vision keys and values.

    Implements numerically stable scaled dot-product attention without masking.
    Each text query token attends over all vision key-value pairs, fusing
    visual context into the text representation.

    Args:
        config: A :class:`CrossModalConfig` instance.
    """

    def __init__(self, config: CrossModalConfig) -> None:
        self._cfg = config
        self._total_forward_calls: int = 0
        self._total_text_tokens:   int = 0
        self._total_vision_tokens: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forward(
        self,
        text_q: np.ndarray,
        vision_k: np.ndarray,
        vision_v: np.ndarray,
    ) -> np.ndarray:
        """Compute cross-modal attention output.

        Args:
            text_q:   Shape ``(n_text_heads, seq_text, head_dim)``.
            vision_k: Shape ``(n_vision_heads, seq_vision, head_dim)``.
            vision_v: Shape ``(n_vision_heads, seq_vision, head_dim)``.

        Returns:
            Shape ``(n_text_heads, seq_text, head_dim)`` — text tokens
            grounded by vision context.

        Raises:
            ValueError: If any input tensor has an incorrect shape or if
                        the sequence / head dimensions are inconsistent.
        """
        text_q   = np.asarray(text_q,   dtype=np.float32)
        vision_k = np.asarray(vision_k, dtype=np.float32)
        vision_v = np.asarray(vision_v, dtype=np.float32)

        self._validate_shapes(text_q, vision_k, vision_v)

        seq_text = text_q.shape[1]
        seq_vis  = vision_k.shape[1]
        scale    = float(self._cfg.softmax_scale)  # type: ignore[arg-type]

        # Compute attention logits: (n_heads, seq_text, seq_vision)
        # scores[h, t, v] = text_q[h, t, :] · vision_k[h, v, :]
        scores = np.einsum("htd,hvd->htv", text_q, vision_k) * scale

        # Numerically stable softmax over the vision dimension.
        scores_max = scores.max(axis=2, keepdims=True)           # (n_heads, seq_text, 1)
        exp_scores = np.exp(scores - scores_max)                 # (n_heads, seq_text, seq_vision)
        attn_weights = exp_scores / exp_scores.sum(axis=2, keepdims=True)

        # Weighted sum of vision values: (n_heads, seq_text, head_dim)
        output = np.einsum("htv,hvd->htd", attn_weights, vision_v)

        self._total_forward_calls += 1
        self._total_text_tokens   += seq_text
        self._total_vision_tokens += seq_vis

        return output

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def stats(self) -> CrossModalStats:
        """Return a snapshot of cumulative forward-pass statistics."""
        return CrossModalStats(
            total_forward_calls=self._total_forward_calls,
            total_text_tokens=self._total_text_tokens,
            total_vision_tokens=self._total_vision_tokens,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_shapes(
        self,
        text_q: np.ndarray,
        vision_k: np.ndarray,
        vision_v: np.ndarray,
    ) -> None:
        """Raise ``ValueError`` if any tensor has an unexpected shape."""
        nh  = self._cfg.n_text_heads
        vnH = self._cfg.n_vision_heads
        dh  = self._cfg.head_dim

        if text_q.ndim != 3:
            raise ValueError(
                f"text_q must be 3-D (n_text_heads, seq_text, head_dim), "
                f"got shape {text_q.shape}."
            )
        if text_q.shape[0] != nh or text_q.shape[2] != dh:
            raise ValueError(
                f"text_q must have shape ({nh}, seq_text, {dh}), "
                f"got {text_q.shape}."
            )
        if vision_k.ndim != 3:
            raise ValueError(
                f"vision_k must be 3-D (n_vision_heads, seq_vision, head_dim), "
                f"got shape {vision_k.shape}."
            )
        if vision_k.shape[0] != vnH or vision_k.shape[2] != dh:
            raise ValueError(
                f"vision_k must have shape ({vnH}, seq_vision, {dh}), "
                f"got {vision_k.shape}."
            )
        if vision_v.ndim != 3:
            raise ValueError(
                f"vision_v must be 3-D (n_vision_heads, seq_vision, head_dim), "
                f"got shape {vision_v.shape}."
            )
        if vision_v.shape[0] != vnH or vision_v.shape[2] != dh:
            raise ValueError(
                f"vision_v must have shape ({vnH}, seq_vision, {dh}), "
                f"got {vision_v.shape}."
            )
        if vision_k.shape[1] != vision_v.shape[1]:
            raise ValueError(
                f"vision_k seq_vision ({vision_k.shape[1]}) must equal "
                f"vision_v seq_vision ({vision_v.shape[1]})."
            )
