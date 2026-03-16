# [Experimental] This module is part of Squish v6+ (Waves 19–26).
# Proof-of-concept quality: API and behaviour may change without notice.
# For stable core functionality see Waves 1–12 in MODULES.md.
"""FlashPrefill — Chunked causal-attention prefill with O(chunk²) memory.

Standard prefill computes the full ``seq_len × seq_len`` attention matrix,
requiring O(seq²) memory.  :class:`FlashPrefillKernel` processes the sequence
in ``chunk_size`` blocks, processing one query chunk at a time and attending
over all causally visible key/value positions.  Peak working memory is
O(seq × chunk_size), making very long sequences feasible without tiling to
hardware.

Inspired by the chunked-attention algorithm described in:
    Dao et al., "FlashAttention-2: Faster Attention with Better Parallelism
    and Work Partitioning", ICLR 2024.
    https://arxiv.org/abs/2307.08691

Usage::

    import numpy as np
    from squish.attention.flash_prefill import FlashPrefillKernel, PrefillConfig

    cfg    = PrefillConfig(chunk_size=512, n_heads=8, head_dim=64)
    kernel = FlashPrefillKernel(cfg)

    rng = np.random.default_rng(0)
    seq_len = 1024
    q = rng.standard_normal((8, seq_len, 64)).astype(np.float32)
    k = rng.standard_normal((8, seq_len, 64)).astype(np.float32)
    v = rng.standard_normal((8, seq_len, 64)).astype(np.float32)

    output = kernel.prefill(q, k, v)  # (8, 1024, 64)
    print(kernel.stats)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

__all__ = ["PrefillConfig", "PrefillStats", "FlashPrefillKernel"]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class PrefillConfig:
    """Configuration for the chunked flash-prefill kernel.

    Attributes:
        chunk_size: Number of query positions processed per chunk iteration.
            Smaller values use less peak memory; larger values amortise
            Python loop overhead.
        n_heads: Number of attention heads.
        head_dim: Dimension per attention head.
        softmax_scale: Attention scale factor applied to raw dot-product
            scores.  Defaults to ``1 / sqrt(head_dim)`` when ``None``.
    """

    chunk_size: int = 512
    n_heads: int = 8
    head_dim: int = 64
    softmax_scale: float | None = None

    def __post_init__(self) -> None:
        if self.chunk_size < 1:
            raise ValueError(f"chunk_size must be >= 1; got {self.chunk_size}")
        if self.n_heads < 1:
            raise ValueError(f"n_heads must be >= 1; got {self.n_heads}")
        if self.head_dim < 1:
            raise ValueError(f"head_dim must be >= 1; got {self.head_dim}")
        if self.softmax_scale is None:
            object.__setattr__(
                self, "softmax_scale", 1.0 / math.sqrt(self.head_dim)
            )

    @property
    def effective_scale(self) -> float:
        """Resolved attention scale (always set after ``__post_init__``)."""
        return self.softmax_scale  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


@dataclass
class PrefillStats:
    """Accumulated statistics for :class:`FlashPrefillKernel`.

    Attributes:
        total_chunks_processed: Sum of chunk counts across all prefill calls.
        total_tokens: Sum of sequence lengths across all prefill calls.
        total_prefill_calls: Number of :meth:`FlashPrefillKernel.prefill`
            invocations.
    """

    total_chunks_processed: int = 0
    total_tokens: int = 0
    total_prefill_calls: int = 0


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------


class FlashPrefillKernel:
    """Chunked causal attention prefill kernel.

    For each query chunk ``[q_start : q_end]`` the kernel attends over all
    key/value positions ``[0 : q_end]`` (causal constraint: query at absolute
    position ``i`` may only attend to positions ``<= i``).  Within the query
    chunk the causal mask zeroes the logits for ``k_pos > q_pos``.

    The algorithm accumulates the output in-place without materialising the
    full ``(n_heads, seq_len, seq_len)`` score matrix.

    Args:
        config: :class:`PrefillConfig` instance.
    """

    def __init__(self, config: PrefillConfig) -> None:
        self.config = config
        self._stats = PrefillStats()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def prefill(
        self,
        q: np.ndarray,
        k: np.ndarray,
        v: np.ndarray,
    ) -> np.ndarray:
        """Run chunked causal attention over the full input sequence.

        Args:
            q: Query tensor, shape ``(n_heads, seq_len, head_dim)`` float32.
            k: Key tensor, shape ``(n_heads, seq_len, head_dim)`` float32.
            v: Value tensor, shape ``(n_heads, seq_len, head_dim)`` float32.

        Returns:
            Attention output, shape ``(n_heads, seq_len, head_dim)`` float32.

        Raises:
            ValueError: If any tensor has an unexpected shape.
        """
        cfg = self.config
        if q.ndim != 3:
            raise ValueError(
                f"q must be 3-D (n_heads, seq_len, head_dim); got shape {q.shape}"
            )
        n_heads, seq_len, head_dim = q.shape
        if n_heads != cfg.n_heads:
            raise ValueError(
                f"Expected {cfg.n_heads} heads; got {n_heads}"
            )
        if head_dim != cfg.head_dim:
            raise ValueError(
                f"Expected head_dim {cfg.head_dim}; got {head_dim}"
            )
        if k.shape != q.shape:
            raise ValueError(
                f"k shape {k.shape} must match q shape {q.shape}"
            )
        if v.shape != q.shape:
            raise ValueError(
                f"v shape {v.shape} must match q shape {q.shape}"
            )

        q_f = q.astype(np.float32)
        k_f = k.astype(np.float32)
        v_f = v.astype(np.float32)
        scale = cfg.effective_scale

        output = np.zeros((n_heads, seq_len, head_dim), dtype=np.float32)
        n_chunks = 0

        q_start = 0
        while q_start < seq_len:
            q_end = min(q_start + cfg.chunk_size, seq_len)

            # Query slice: (n_heads, chunk_q, head_dim)
            q_chunk = q_f[:, q_start:q_end, :]
            # Key/value slice causally visible to this query chunk: [0, q_end)
            k_chunk = k_f[:, :q_end, :]  # (n_heads, q_end, head_dim)
            v_chunk = v_f[:, :q_end, :]  # (n_heads, q_end, head_dim)

            # Attention scores: (n_heads, chunk_q, q_end)
            # scores[h, i, j] = scale * q_chunk[h,i] · k_chunk[h,j]
            scores = np.einsum("hid,hjd->hij", q_chunk, k_chunk) * scale

            # Causal mask: query at absolute position (q_start + i) may not
            # attend to key at position j > (q_start + i).
            q_positions = np.arange(q_start, q_end)[:, None]  # (chunk_q, 1)
            k_positions = np.arange(q_end)[None, :]            # (1, q_end)
            causal_mask = k_positions > q_positions            # (chunk_q, q_end)
            # Broadcast mask over heads dimension
            scores = np.where(causal_mask[np.newaxis, :, :], np.float32(-1e30), scores)

            # Numerically stable softmax
            scores_max = scores.max(axis=-1, keepdims=True)
            exp_scores = np.exp(scores - scores_max)
            denom = exp_scores.sum(axis=-1, keepdims=True) + np.float32(1e-30)
            weights = exp_scores / denom  # (n_heads, chunk_q, q_end)

            # Weighted sum of values: (n_heads, chunk_q, head_dim)
            output[:, q_start:q_end, :] = np.einsum(
                "hij,hjd->hid", weights, v_chunk
            )

            q_start = q_end
            n_chunks += 1

        self._stats.total_chunks_processed += n_chunks
        self._stats.total_tokens += seq_len
        self._stats.total_prefill_calls += 1
        return output

    def reset_stats(self) -> None:
        """Reset accumulated prefill statistics to zero."""
        self._stats = PrefillStats()

    @property
    def stats(self) -> PrefillStats:
        """Current accumulated :class:`PrefillStats`."""
        return self._stats
