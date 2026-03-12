"""BinaryAttn — Sign-binarised query/key attention approximation.

Approximates scaled-dot-product attention by binarising query and key vectors
to {−1, +1} via the sign function before computing the similarity matrix.
Because all elements are ±1, the inner product becomes equivalent to a POPCOUNT
(XOR-POPCOUNT in XNOR-Net terminology), which can be computed with integer
hardware at very high throughput.

Values (V) are kept in float32, so the attention output preserves the full
value precision.  The binary approximation trades a modest quality loss for
extreme memory efficiency: Q and K require only 1 bit per element instead of
16 or 32.

Reference:
    Rastegari et al., "XNOR-Net: ImageNet Classification Using Binary
    Convolutional Neural Networks", ECCV 2016.
    https://arxiv.org/abs/1603.05279

Usage::

    import numpy as np
    from squish.binary_attn import BinaryConfig, BinaryAttention

    cfg  = BinaryConfig(n_heads=8, head_dim=64)
    attn = BinaryAttention(cfg)

    rng = np.random.default_rng(42)
    q   = rng.standard_normal((8, 16, 64)).astype(np.float32)
    k   = rng.standard_normal((8, 32, 64)).astype(np.float32)
    v   = rng.standard_normal((8, 32, 64)).astype(np.float32)

    out = attn.forward(q, k, v)   # (8, 16, 64)
    print(attn.stats.total_tokens_processed)  # 16
"""

from __future__ import annotations

__all__ = [
    "BinaryConfig",
    "BinaryAttention",
    "BinaryStats",
]

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class BinaryConfig:
    """Configuration for binary-quantised attention.

    Attributes:
        n_heads: Number of attention heads.  Must be >= 1.
        head_dim: Dimension of each attention head.  Must be >= 1.
        softmax_scale: Explicit scale applied to the dot-product scores before
            softmax.  Defaults to ``1 / sqrt(head_dim)`` when *None*.
    """

    n_heads: int = 8
    head_dim: int = 64
    softmax_scale: Optional[float] = None

    def __post_init__(self) -> None:
        if self.n_heads < 1:
            raise ValueError(
                f"n_heads must be >= 1; got {self.n_heads}"
            )
        if self.head_dim < 1:
            raise ValueError(
                f"head_dim must be >= 1; got {self.head_dim}"
            )
        if self.softmax_scale is not None and self.softmax_scale <= 0.0:
            raise ValueError(
                f"softmax_scale must be positive when provided; "
                f"got {self.softmax_scale}"
            )

    @property
    def effective_scale(self) -> float:
        """Resolved softmax scale (default: ``1 / sqrt(head_dim)``)."""
        if self.softmax_scale is not None:
            return float(self.softmax_scale)
        return 1.0 / math.sqrt(self.head_dim)


# ── Stats ─────────────────────────────────────────────────────────────────────

@dataclass
class BinaryStats:
    """Cumulative statistics for a :class:`BinaryAttention` session.

    Attributes:
        total_forward_calls: Number of times :meth:`BinaryAttention.forward`
            has been invoked.
        total_tokens_processed: Cumulative number of query tokens processed
            (sum of ``seq_q`` across all forward calls).
    """

    total_forward_calls: int = 0
    total_tokens_processed: int = 0


# ── Attention ─────────────────────────────────────────────────────────────────

class BinaryAttention:
    """Multi-head binary-quantised attention.

    Queries and keys are binarised to ±1 before the similarity computation,
    reducing Q/K memory to 1 bit per element.  Values remain in float32 so
    the weighted aggregation retains full precision.

    The dot-product is normalised by ``head_dim`` in addition to
    ``softmax_scale`` so that the scores are comparable to ordinary attention
    regardless of head dimension.

    Args:
        config: :class:`BinaryConfig` specifying head count, head dimension,
            and optional custom softmax scale.
    """

    def __init__(self, config: BinaryConfig) -> None:
        self.config = config
        self._stats = BinaryStats()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forward(
        self,
        q: np.ndarray,
        k: np.ndarray,
        v: np.ndarray,
    ) -> np.ndarray:
        """Compute binarised multi-head attention.

        Queries and keys are binarised element-wise: ``q_bin = +1`` where
        ``q >= 0``, otherwise ``−1``.  The attention logits are computed as:

            ``scores = q_bin @ k_bin.T * softmax_scale / head_dim``

        A numerically-stable softmax is applied along the key axis, and the
        result is multiplied by *v* (kept in float32).

        Args:
            q: Query tensor of shape ``(n_heads, seq_q, head_dim)`` float32.
            k: Key tensor of shape ``(n_heads, seq_k, head_dim)`` float32.
            v: Value tensor of shape ``(n_heads, seq_k, head_dim)`` float32.

        Returns:
            Output tensor of shape ``(n_heads, seq_q, head_dim)`` float32.

        Raises:
            ValueError: If any shape constraint is violated.
        """
        cfg = self.config

        q = np.asarray(q, dtype=np.float32)
        k = np.asarray(k, dtype=np.float32)
        v = np.asarray(v, dtype=np.float32)

        if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
            raise ValueError(
                "q, k, v must each be 3-D "
                f"(n_heads, seq, head_dim); got shapes "
                f"{q.shape}, {k.shape}, {v.shape}"
            )
        if q.shape[0] != cfg.n_heads:
            raise ValueError(
                f"q.shape[0]={q.shape[0]} != n_heads={cfg.n_heads}"
            )
        if k.shape[0] != cfg.n_heads:
            raise ValueError(
                f"k.shape[0]={k.shape[0]} != n_heads={cfg.n_heads}"
            )
        if v.shape[0] != cfg.n_heads:
            raise ValueError(
                f"v.shape[0]={v.shape[0]} != n_heads={cfg.n_heads}"
            )
        if q.shape[2] != cfg.head_dim:
            raise ValueError(
                f"q.shape[2]={q.shape[2]} != head_dim={cfg.head_dim}"
            )
        if k.shape[2] != cfg.head_dim:
            raise ValueError(
                f"k.shape[2]={k.shape[2]} != head_dim={cfg.head_dim}"
            )
        if k.shape != v.shape:
            raise ValueError(
                f"k and v must have identical shapes; "
                f"got k={k.shape}, v={v.shape}"
            )

        # Binarise queries and keys: map to {−1, +1}
        # Values at zero are treated as positive (+1).
        q_bin = np.where(q >= 0.0, 1.0, -1.0).astype(np.float32)
        k_bin = np.where(k >= 0.0, 1.0, -1.0).astype(np.float32)

        # Attention logits: (n_heads, seq_q, seq_k)
        # Normalise by head_dim so magnitude is independent of dimensionality.
        scores = (
            np.matmul(q_bin, k_bin.transpose(0, 2, 1))
            * cfg.effective_scale
            / cfg.head_dim
        )

        # Numerically stable softmax along the key axis
        scores_shifted = scores - np.max(scores, axis=-1, keepdims=True)
        exp_scores = np.exp(scores_shifted)
        attn_weights = exp_scores / (
            np.sum(exp_scores, axis=-1, keepdims=True) + 1e-9
        )

        # Weighted aggregation over float32 values
        output = np.matmul(attn_weights, v)

        self._stats.total_forward_calls     += 1
        self._stats.total_tokens_processed  += q.shape[1]

        return output

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def stats(self) -> BinaryStats:
        """Cumulative attention statistics for this instance."""
        return self._stats
