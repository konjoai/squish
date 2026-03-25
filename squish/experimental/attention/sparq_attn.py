"""sparq_attn.py — SparQ Attention: Bandwidth-Efficient Sparse Query Attention

During autoregressive decode, for each query token:
  1. Select top-r query dimensions (largest absolute magnitude) → sparse Q
  2. Estimate approximate dot-product scores using only those r dimensions
  3. Select top-k KV pairs by estimated score (coarse selection)
  4. Compute exact attention over the top-k pairs only
  5. Gather weighted sum → output token

Bandwidth savings vs standard full attention:
  - KV read: k/seq_len fraction (instead of full KV)
  - Q dimensions used: r/d_k fraction (for score estimation)

Quality at typical settings (r=d_k//4, k=seq_len//8): <1% perplexity delta.

Based on: "SparQ Attention: Bandwidth-Efficient LLM Inference"
          (Riddell et al., 2023, NeurIPS Efficient NLP Workshop)

Usage:
    cfg = SparQConfig(r_dims=32, top_k_kv=128)
    attn = SparQAttention(cfg)
    # keys: (seq, n_heads, d_k)  values: (seq, n_heads, d_v)  query: (n_heads, d_k)
    output = attn.decode_step(query, keys, values)  # -> (n_heads, d_v)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class SparQConfig:
    """Configuration for SparQ sparse-query attention.

    Args:
        r_dims:     Number of query dimensions (top-r by magnitude) used for
                    approximate score estimation.  None → use full dimension.
        top_k_kv:   Number of KV pairs selected for exact attention.
                    None → use full sequence (no sparsity).
        score_scale: If True, rescale approximate scores by sqrt(d_k/r_dims)
                     to compensate for the partial-dimension dot product.
        fallback_full: If True, fall back to full attention when seq < top_k_kv.
    """
    r_dims: Optional[int] = 32
    top_k_kv: Optional[int] = 128
    score_scale: bool = True
    fallback_full: bool = True


@dataclass
class SparQStats:
    """Accumulated statistics for a SparQAttention instance."""
    decode_calls: int = 0
    total_seq_len: int = 0
    total_k_selected: int = 0
    full_fallbacks: int = 0

    @property
    def mean_compression_ratio(self) -> float:
        """Fraction of KV pairs actually read (lower = more sparse)."""
        if self.total_seq_len == 0:
            return 1.0
        return self.total_k_selected / self.total_seq_len

    @property
    def mean_seq_len(self) -> float:
        if self.decode_calls == 0:
            return 0.0
        return self.total_seq_len / self.decode_calls


class SparQAttention:
    """Sparse-Q decode-time attention with approximate KV selection.

    Assumes causal attention; optimised for single-token decode steps.
    Keys and values should be pre-transposed to (seq, n_heads, d_kv) layout.
    """

    def __init__(self, config: Optional[SparQConfig] = None) -> None:
        self.config = config or SparQConfig()
        self.stats = SparQStats()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def decode_step(
        self,
        query: np.ndarray,   # (n_heads, d_k)
        keys: np.ndarray,    # (seq_len, n_heads, d_k)
        values: np.ndarray,  # (seq_len, n_heads, d_v)
        mask: Optional[np.ndarray] = None,  # (seq_len,) bool — True=valid
    ) -> np.ndarray:
        """Compute one decode-step attention output using SparQ.

        Returns:
            np.ndarray of shape (n_heads, d_v).
        """
        if query.ndim != 2:
            raise ValueError(f"query must be 2-D (n_heads, d_k), got {query.shape}")
        if keys.ndim != 3:
            raise ValueError(f"keys must be 3-D (seq, n_heads, d_k), got {keys.shape}")
        if values.ndim != 3:
            raise ValueError(f"values must be 3-D (seq, n_heads, d_v), got {values.shape}")

        seq_len, n_heads, d_k = keys.shape
        cfg = self.config

        effective_r = cfg.r_dims if cfg.r_dims is not None else d_k
        effective_k = cfg.top_k_kv if cfg.top_k_kv is not None else seq_len
        effective_r = min(effective_r, d_k)
        effective_k = min(effective_k, seq_len)

        self.stats.decode_calls += 1
        self.stats.total_seq_len += seq_len

        # Fallback to full attention when sequence is short enough
        if cfg.fallback_full and seq_len <= effective_k:
            self.stats.total_k_selected += seq_len
            self.stats.full_fallbacks += 1
            return self._full_attention(query, keys, values, mask)

        # ---- Stage 1: select top-r query dimensions ----
        # query: (n_heads, d_k)
        q_magnitude = np.abs(query)  # (n_heads, d_k)
        # Average magnitude across heads for shared dim selection
        q_mean_mag = q_magnitude.mean(axis=0)  # (d_k,)
        r_indices = np.argpartition(q_mean_mag, -effective_r)[-effective_r:]
        r_indices = np.sort(r_indices)  # keep sorted for cache locality

        # ---- Stage 2: approximate scores via partial dot product ----
        # sparse_q: (n_heads, r) from query dims
        sparse_q = query[:, r_indices]  # (n_heads, r)
        # sparse_k: (seq_len, n_heads, r)
        sparse_k = keys[:, :, r_indices]  # (seq_len, n_heads, r)

        # Approximate dot product: (seq_len, n_heads)
        # einsum 'hr,snr->sn'  →  sum over r dimension
        approx_scores = np.einsum("hr,snr->sn", sparse_q, sparse_k)  # (seq_len, n_heads)

        if cfg.score_scale and effective_r < d_k:
            scale = math.sqrt(d_k / effective_r)
            approx_scores = approx_scores * scale

        # Average approximate score across heads for unified KV selection
        mean_approx = approx_scores.mean(axis=1)  # (seq_len,)

        # Apply pad mask before selection
        if mask is not None:
            mean_approx = np.where(mask, mean_approx, -np.inf)

        # ---- Stage 3: select top-k KV pairs ----
        if effective_k >= seq_len:
            top_k_indices = np.arange(seq_len)
        else:
            top_k_indices = np.argpartition(mean_approx, -effective_k)[-effective_k:]
            top_k_indices = np.sort(top_k_indices)

        self.stats.total_k_selected += len(top_k_indices)

        # ---- Stage 4: exact attention over top-k ----
        sel_keys = keys[top_k_indices]    # (k, n_heads, d_k)
        sel_vals = values[top_k_indices]  # (k, n_heads, d_v)
        sel_mask = mask[top_k_indices] if mask is not None else None

        return self._full_attention(query, sel_keys, sel_vals, sel_mask)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _full_attention(
        query: np.ndarray,   # (n_heads, d_k)
        keys: np.ndarray,    # (seq, n_heads, d_k)
        values: np.ndarray,  # (seq, n_heads, d_v)
        mask: Optional[np.ndarray] = None,  # (seq,) bool
    ) -> np.ndarray:
        """Standard scaled dot-product attention for a single query token."""
        d_k = query.shape[-1]
        scale = 1.0 / math.sqrt(d_k)

        # scores: einsum 'hd,shd->sh' → (seq, n_heads)
        scores = np.einsum("hd,shd->sh", query, keys) * scale  # (seq, n_heads)

        if mask is not None:
            scores = np.where(mask[:, None], scores, -1e9)

        # Softmax over seq dimension
        scores_max = scores.max(axis=0, keepdims=True)
        exp_s = np.exp(scores - scores_max)
        weights = exp_s / (exp_s.sum(axis=0, keepdims=True) + 1e-9)  # (seq, n_heads)

        # Weighted sum of values: einsum 'sh,shv->hv'
        output = np.einsum("sh,shv->hv", weights, values)  # (n_heads, d_v)
        return output

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def reset_stats(self) -> None:
        """Reset accumulated statistics."""
        self.stats = SparQStats()

    def expected_speedup(self, seq_len: int, d_k: int) -> float:
        """Estimate bandwidth reduction factor vs full attention.

        Returns value > 1 meaning attention is this many times cheaper.
        """
        cfg = self.config
        r = min(cfg.r_dims or d_k, d_k)
        k = min(cfg.top_k_kv or seq_len, seq_len)
        if seq_len == 0:
            return 1.0
        # Phase 1: r/d_k of KV for approx scores; Phase 2: k/seq exact
        approx_cost = seq_len * r
        exact_cost = k * d_k
        full_cost = seq_len * d_k
        return full_cost / (approx_cost + exact_cost + 1e-9)
