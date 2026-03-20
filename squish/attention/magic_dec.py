"""
squish/attention/magic_dec.py

MagicDecAttention: Sink + Recent + Landmark Sparse Decode.

Reference
---------
He et al. "MagicDec: Breaking the Latency-Throughput Tradeoff for Long Context
Generation with Speculative Decoding." NeurIPS 2024.

Algorithm
---------
At decode time, instead of attending to the full KV cache, each decoding step
attends to three disjoint KV subsets ("topology"):

  1. **Sink tokens** — the first ``n_sinks`` positions.  Attention-sink theory
     (StreamingLLM) shows that LLMs consistently assign high attention mass to
     the very first tokens; keeping them prevents catastrophic quality loss.

  2. **Recent tokens** — the last ``n_recent`` positions in the KV cache,
     representing the immediate local context.

  3. **Landmark tokens** — evenly-spaced "landmark" positions sampled from the
     middle of the cache at stride ``landmark_stride``.  These act as
     compressed global context representatives.

Together these three groups cover the semantically important regions with
O(n_sinks + n_recent + context_len/landmark_stride) tokens — sublinear in the
full context length.

Key properties
--------------
* ``n_sinks`` — number of initial sink positions (default 4).
* ``n_recent`` — size of the sliding recent window (default 256).
* ``landmark_stride`` — stride between global landmark tokens (default 32).
* Exact fallback for short contexts (< min_length tokens).
* NumPy-only; no external dependencies.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class MagicDecConfig:
    """Configuration for MagicDecAttention."""

    n_sinks: int = 4
    """Number of initial attention-sink tokens always kept."""

    n_recent: int = 256
    """Size of the sliding recent-token window."""

    landmark_stride: int = 32
    """Stride at which landmark tokens are sampled from the middle of the cache."""

    head_dim: int = 64
    """Attention head dimension (used for score normalisation)."""

    min_length: int = 64
    """Contexts shorter than this use exact attention."""

    def __post_init__(self) -> None:
        if self.n_sinks < 0:
            raise ValueError("n_sinks must be >= 0")
        if self.n_recent < 1:
            raise ValueError("n_recent must be >= 1")
        if self.landmark_stride < 1:
            raise ValueError("landmark_stride must be >= 1")


@dataclass
class MagicDecStats:
    """Runtime counters for MagicDecAttention."""

    attn_calls: int = 0
    sparse_calls: int = 0
    exact_calls: int = 0
    total_tokens_attended: int = 0
    total_context_tokens: int = 0

    @property
    def mean_sparsity(self) -> float:
        if self.total_context_tokens == 0:
            return 0.0
        return 1.0 - self.total_tokens_attended / self.total_context_tokens


class MagicDecAttention:
    """Sink + recent + landmark sparse decode attention.

    Usage
    -----
    ::

        md = MagicDecAttention()
        output = md.attend(query, keys, values)
    """

    def __init__(self, config: Optional[MagicDecConfig] = None) -> None:
        self.config = config or MagicDecConfig()
        self.stats = MagicDecStats()

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        x = x - x.max()
        e = np.exp(x)
        return e / (e.sum() + 1e-9)

    def _build_mask(self, seq_len: int) -> np.ndarray:
        """Return boolean mask of shape (seq_len,) marking selected positions."""
        cfg = self.config
        mask = np.zeros(seq_len, dtype=bool)

        # 1. Sinks
        sink_end = min(cfg.n_sinks, seq_len)
        mask[:sink_end] = True

        # 2. Recent window
        recent_start = max(0, seq_len - cfg.n_recent)
        mask[recent_start:] = True

        # 3. Landmarks in the middle (between sinks and recent window)
        middle_end = recent_start
        middle_start = sink_end
        if middle_end > middle_start and cfg.landmark_stride > 0:
            landmarks = np.arange(middle_start, middle_end, cfg.landmark_stride)
            mask[landmarks] = True

        return mask

    def attend(
        self,
        query: np.ndarray,
        keys: np.ndarray,
        values: np.ndarray,
    ) -> np.ndarray:
        """Sparse decode-time attention.

        Parameters
        ----------
        query:
            Shape ``(head_dim,)``.
        keys:
            Shape ``(seq_len, head_dim)``.
        values:
            Shape ``(seq_len, head_dim)``.

        Returns
        -------
        output:
            Shape ``(head_dim,)``.
        """
        self.stats.attn_calls += 1
        seq_len = keys.shape[0]
        self.stats.total_context_tokens += seq_len
        scale = 1.0 / math.sqrt(self.config.head_dim)

        if seq_len <= self.config.min_length:
            self.stats.exact_calls += 1
            self.stats.total_tokens_attended += seq_len
            weights = self._softmax((keys @ query) * scale)
            return weights @ values

        self.stats.sparse_calls += 1
        mask = self._build_mask(seq_len)
        k_sel = keys[mask]
        v_sel = values[mask]
        self.stats.total_tokens_attended += int(mask.sum())

        weights = self._softmax((k_sel @ query) * scale)
        return weights @ v_sel

    def reset_stats(self) -> None:
        """Reset runtime counters."""
        self.stats = MagicDecStats()
