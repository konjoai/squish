# [Experimental] This module is part of Squish v6+ (Waves 19–26).
# Proof-of-concept quality: API and behaviour may change without notice.
# For stable core functionality see Waves 1–12 in MODULES.md.
"""CrossDocAttn — Chunked cross-document attention for multi-document QA.

Enables a query to attend over multiple document key-value chunks *without*
concatenating all documents into a single flat sequence.  Each document is
processed independently: per-document raw attention logits, maximum, and
weighted-value sums are computed, then merged with a numerically stable
log-sum-exp normalisation to produce the exact same result as if all tokens
had been concatenated.

The merge formula for N documents is:

    global_max  = max(max_i  for each doc i)
    numerator   = Σ_i  weighted_v_i · exp(max_i − global_max)
    denominator = Σ_i  sum_exp_i  · exp(max_i − global_max)
    output      = numerator / denominator

This is the standard "online softmax" / FlashAttention identity applied
across document chunks rather than across sequence tiles.

Typical usage::

    from squish.attention.cross_doc_attn import CrossDocConfig, CrossDocAttention
    import numpy as np

    cfg  = CrossDocConfig(n_heads=4, head_dim=32, max_docs=8)
    attn = CrossDocAttention(cfg)

    query     = np.random.randn(4, 2, 32).astype(np.float32)
    doc_keys  = [np.random.randn(4, 16, 32).astype(np.float32) for _ in range(3)]
    doc_vals  = [np.random.randn(4, 16, 32).astype(np.float32) for _ in range(3)]

    out = attn.forward(query, doc_keys, doc_vals)  # (4, 2, 32)
    print(attn.stats)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "CrossDocConfig",
    "CrossDocAttention",
    "CrossDocStats",
]

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class CrossDocConfig:
    """Configuration for :class:`CrossDocAttention`.

    Parameters
    ----------
    n_heads : int
        Number of attention heads.
    head_dim : int
        Dimension of each attention head.
    max_docs : int
        Maximum number of documents that can be attended in a single call.
    softmax_scale : float or None
        Scaling factor applied to raw attention logits before softmax.
        If ``None``, defaults to ``1 / sqrt(head_dim)``.
    """

    n_heads: int = 8
    head_dim: int = 64
    max_docs: int = 16
    softmax_scale: float | None = None

    def __post_init__(self) -> None:
        if self.n_heads < 1:
            raise ValueError("n_heads must be >= 1")
        if self.head_dim < 1:
            raise ValueError("head_dim must be >= 1")
        if self.max_docs < 1:
            raise ValueError("max_docs must be >= 1")
        if self.softmax_scale is not None and self.softmax_scale <= 0.0:
            raise ValueError("softmax_scale must be positive when provided")


# ---------------------------------------------------------------------------
# Stats dataclass
# ---------------------------------------------------------------------------


@dataclass
class CrossDocStats:
    """Aggregate statistics for :class:`CrossDocAttention`.

    Attributes
    ----------
    total_forward_calls : int
        Total number of :meth:`~CrossDocAttention.forward` calls.
    total_docs_attended : int
        Sum of document counts across all forward calls.
    """

    total_forward_calls: int = 0
    total_docs_attended: int = 0

    @property
    def avg_docs_per_call(self) -> float:
        """Mean number of documents per forward call."""
        if self.total_forward_calls == 0:
            return 0.0
        return self.total_docs_attended / self.total_forward_calls


# ---------------------------------------------------------------------------
# CrossDocAttention
# ---------------------------------------------------------------------------


class CrossDocAttention:
    """Chunked cross-document attention with exact softmax normalisation.

    The query attends over all document key-value chunks via the online-softmax
    merge identity, returning the same result as full-sequence attention over
    all documents concatenated.

    Parameters
    ----------
    config : CrossDocConfig
        Attention configuration.
    """

    def __init__(self, config: CrossDocConfig) -> None:
        self._cfg = config
        self._scale: float = (
            config.softmax_scale
            if config.softmax_scale is not None
            else float(config.head_dim) ** -0.5
        )
        self._stats = CrossDocStats()

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def forward(
        self,
        query: np.ndarray,
        doc_keys: list[np.ndarray],
        doc_values: list[np.ndarray],
    ) -> np.ndarray:
        """Compute cross-document attention.

        Parameters
        ----------
        query : np.ndarray
            Query tensor of shape ``(n_heads, seq_q, head_dim)``.
        doc_keys : list[np.ndarray]
            Per-document key tensors, each of shape
            ``(n_heads, seq_k_i, head_dim)``.
        doc_values : list[np.ndarray]
            Per-document value tensors, each of shape
            ``(n_heads, seq_v_i, head_dim)``.  Must have the same length as
            *doc_keys* and each ``seq_v_i`` must match the corresponding
            ``seq_k_i``.

        Returns
        -------
        np.ndarray
            Output tensor of shape ``(n_heads, seq_q, head_dim)``.

        Raises
        ------
        ValueError
            If ``len(doc_keys) != len(doc_values)``, if the number of
            documents exceeds ``config.max_docs``, or if tensor shapes are
            inconsistent.
        """
        if len(doc_keys) != len(doc_values):
            raise ValueError(
                f"doc_keys and doc_values must have the same length; "
                f"got {len(doc_keys)} vs {len(doc_values)}"
            )
        n_docs = len(doc_keys)
        if n_docs > self._cfg.max_docs:
            raise ValueError(
                f"Number of documents {n_docs} exceeds max_docs={self._cfg.max_docs}"
            )
        if n_docs == 0:
            raise ValueError("doc_keys must be non-empty")

        query = np.asarray(query, dtype=np.float32)
        n_heads, seq_q, head_dim = query.shape
        if n_heads != self._cfg.n_heads or head_dim != self._cfg.head_dim:
            raise ValueError(
                f"query shape must be ({self._cfg.n_heads}, seq_q, {self._cfg.head_dim}); "
                f"got {query.shape}"
            )

        # Per-document intermediates.
        max_logits: list[np.ndarray] = []    # each (n_heads, seq_q)
        sum_exps:   list[np.ndarray] = []    # each (n_heads, seq_q)
        weighted_vs: list[np.ndarray] = []   # each (n_heads, seq_q, head_dim)

        for i, (dk, dv) in enumerate(zip(doc_keys, doc_values)):
            dk = np.asarray(dk, dtype=np.float32)
            dv = np.asarray(dv, dtype=np.float32)

            if dk.ndim != 3 or dk.shape[0] != n_heads or dk.shape[2] != head_dim:
                raise ValueError(
                    f"doc_keys[{i}] must have shape (n_heads, seq_k, head_dim); "
                    f"got {dk.shape}"
                )
            if dv.shape != dk.shape:
                raise ValueError(
                    f"doc_values[{i}] shape {dv.shape} must match doc_keys[{i}] "
                    f"shape {dk.shape}"
                )

            # logits: (n_heads, seq_q, seq_k)
            logits = np.einsum("hqd,hkd->hqk", query, dk) * self._scale

            # Numerically stable softmax ingredients per document.
            max_i = logits.max(axis=-1)           # (n_heads, seq_q)
            exp_i = np.exp(logits - max_i[..., np.newaxis])  # (n_heads, seq_q, seq_k)
            sum_i = exp_i.sum(axis=-1)             # (n_heads, seq_q)
            wv_i  = np.einsum("hqk,hkd->hqd", exp_i, dv)    # (n_heads, seq_q, head_dim)

            max_logits.append(max_i)
            sum_exps.append(sum_i)
            weighted_vs.append(wv_i)

        # Global merge across documents.
        # global_max: (n_heads, seq_q)
        global_max = np.stack(max_logits, axis=0).max(axis=0)

        numerator   = np.zeros((n_heads, seq_q, head_dim), dtype=np.float32)
        denominator = np.zeros((n_heads, seq_q),           dtype=np.float32)

        for max_i, sum_i, wv_i in zip(max_logits, sum_exps, weighted_vs):
            # Scale factor compensates for shifting to the global max.
            scale_factor = np.exp(max_i - global_max)          # (n_heads, seq_q)
            numerator   += wv_i  * scale_factor[..., np.newaxis]
            denominator += sum_i * scale_factor

        # Avoid division by zero (should not happen with valid inputs).
        output = numerator / (denominator[..., np.newaxis] + 1e-30)

        self._stats.total_forward_calls += 1
        self._stats.total_docs_attended += n_docs
        return output

    # ------------------------------------------------------------------ #
    # Properties                                                           #
    # ------------------------------------------------------------------ #

    @property
    def stats(self) -> CrossDocStats:
        """Current aggregate statistics."""
        return self._stats

    def __repr__(self) -> str:
        return (
            f"CrossDocAttention(n_heads={self._cfg.n_heads}, "
            f"head_dim={self._cfg.head_dim}, max_docs={self._cfg.max_docs})"
        )
