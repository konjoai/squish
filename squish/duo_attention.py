"""
squish/duo_attention.py

DuoAttention — efficient long-context LLM inference via retrieval/streaming
head separation.

Based on:
  "DuoAttention: Efficient Long-Context LLM Inference with Retrieval and
   Streaming Heads" — Xiao et al., MIT 2024  (arXiv:2410.10819)

Key insight
-----------
Not all attention heads contribute equally to long-range token retrieval.
"Retrieval heads" (≈30–40% of heads) maintain meaningful attention patterns
across the full context; the remaining "streaming heads" almost exclusively
attend to:
  1. Attention sinks — the very first tokens (positions 0 … sink_tokens-1)
  2. A local recency window — the most-recent ``local_window`` tokens

By caching only sink + window tokens for streaming heads, KV memory is reduced
by ~50–60% with negligible quality loss on long-context benchmarks (≥1 M tokens).

This module provides:
  * ``DuoAttentionConfig`` — sink size, window, classification threshold.
  * ``HeadCalibration`` — accumulates attention-mass statistics over samples.
  * ``HeadClassifier`` — classifies each head as retrieval vs streaming.
  * ``StreamingKVWindow`` — circular-buffer KV store for streaming heads.
  * ``DuoKVManager`` — unified store: full history for retrieval,
    sink+window for streaming.

Usage::

    from squish.duo_attention import (
        DuoAttentionConfig, HeadClassifier, DuoKVManager,
    )

    cfg = DuoAttentionConfig(num_layers=28, num_heads=32)

    clf = HeadClassifier(cfg)
    clf.record(layer_idx=0, attn_weights=attn_np)   # (n_heads, q_len, k_len)
    labels = clf.classify()                          # {(layer, head): "retrieval"|"streaming"}

    mgr = DuoKVManager(cfg, labels)
    mgr.store_kv(layer=0, head=3, pos=42, key=k_vec, value=v_vec)
    keys, vals = mgr.load_kv(layer=0, head=3)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

__all__ = [
    "DuoAttentionConfig",
    "HeadCalibration",
    "HeadClassifier",
    "StreamingKVWindow",
    "DuoKVManager",
]

HeadLabel = str          # "retrieval" or "streaming"
HeadKey   = Tuple[int, int]  # (layer_idx, head_idx)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class DuoAttentionConfig:
    """Configuration for DuoAttention head-level KV-cache strategy.

    Parameters
    ----------
    num_layers : int
        Total transformer layers (e.g. 28 for Qwen2.5-7B).
    num_heads : int
        Attention heads per layer.
    head_dim : int
        Dimension of each K/V head vector.
    sink_tokens : int
        Number of initial token positions always retained ("attention sinks").
    local_window : int
        Recent-token window kept for streaming heads (non-sink positions).
    retrieval_threshold : float
        Out-of-window attention mass that qualifies a head as a retrieval head.
        Lower → more retrieval heads → larger cache, better recall quality.
    min_retrieval_fraction : float
        Lower bound on the fraction of heads forced into the retrieval class
        even if their calibration scores fall below ``retrieval_threshold``.
    """

    num_layers:             int   = 32
    num_heads:              int   = 32
    head_dim:               int   = 128
    sink_tokens:            int   = 4
    local_window:           int   = 256
    retrieval_threshold:    float = 0.05
    min_retrieval_fraction: float = 0.25

    def __post_init__(self) -> None:
        if self.num_layers < 1:
            raise ValueError("num_layers must be ≥ 1")
        if self.num_heads < 1:
            raise ValueError("num_heads must be ≥ 1")
        if self.head_dim < 1:
            raise ValueError("head_dim must be ≥ 1")
        if self.sink_tokens < 0:
            raise ValueError("sink_tokens must be ≥ 0")
        if self.local_window < 1:
            raise ValueError("local_window must be ≥ 1")
        if not 0.0 < self.retrieval_threshold < 1.0:
            raise ValueError("retrieval_threshold must be in (0, 1)")
        if not 0.0 <= self.min_retrieval_fraction <= 1.0:
            raise ValueError("min_retrieval_fraction must be in [0, 1]")


# ---------------------------------------------------------------------------
# Head calibration
# ---------------------------------------------------------------------------

class HeadCalibration:
    """Accumulate per-head attention statistics to classify head types.

    For each head we measure the fraction of attention mass that falls
    *outside* the sink + local-window region.  High out-of-window mass
    signals a retrieval head.

    Parameters
    ----------
    config : DuoAttentionConfig
    """

    def __init__(self, config: DuoAttentionConfig) -> None:
        self._cfg     = config
        self._sum:   Dict[HeadKey, float] = {}
        self._count: Dict[HeadKey, int]   = {}

    def record(self, layer_idx: int, attn_weights: np.ndarray) -> None:
        """Record one attention tensor for calibration.

        Parameters
        ----------
        layer_idx : int
            Transformer layer index (0-based).
        attn_weights : ndarray of shape ``(n_heads, q_len, k_len)``
            Row-normalised attention weights (softmax output).
        """
        attn = np.asarray(attn_weights, dtype=np.float32)
        if attn.ndim != 3:
            raise ValueError("attn_weights must be (n_heads, q_len, k_len)")
        n_heads, q_len, k_len = attn.shape

        sink = self._cfg.sink_tokens
        win  = self._cfg.local_window

        # Build a mask of "in-window" key positions.
        in_window           = np.zeros(k_len, dtype=bool)
        in_window[:min(sink, k_len)] = True
        win_start = max(0, k_len - win)
        in_window[win_start:] = True
        out_mask = ~in_window                              # (k_len,)

        for h in range(n_heads):
            # Average out-of-window mass over all query positions
            out_mass_avg = float(attn[h, :, :].sum(axis=0)[out_mask].sum()) / max(q_len, 1)
            key = (layer_idx, h)
            self._sum[key]   = self._sum.get(key, 0.0) + out_mass_avg
            self._count[key] = self._count.get(key, 0) + 1

    def mean_scores(self) -> Dict[HeadKey, float]:
        """Return mean out-of-window attention score per head."""
        return {k: self._sum[k] / self._count[k] for k in self._sum}

    def classify(self) -> Dict[HeadKey, HeadLabel]:
        """Return head labels from accumulated data.

        Returns
        -------
        dict mapping ``(layer_idx, head_idx)`` → ``"retrieval"`` | ``"streaming"``.
        Heads with no recorded data default to ``"streaming"``.
        """
        cfg    = self._cfg
        scores = self.mean_scores()

        n_total       = cfg.num_layers * cfg.num_heads
        n_min_ret     = max(1, int(n_total * cfg.min_retrieval_fraction))

        sorted_keys   = sorted(scores, key=lambda k: scores[k], reverse=True)
        retrieval_set = {k for k in sorted_keys if scores[k] >= cfg.retrieval_threshold}

        # Enforce minimum retrieval fraction
        for k in sorted_keys:
            if len(retrieval_set) >= n_min_ret:
                break
            retrieval_set.add(k)

        labels: Dict[HeadKey, HeadLabel] = {}
        for layer in range(cfg.num_layers):
            for h in range(cfg.num_heads):
                key = (layer, h)
                labels[key] = "retrieval" if key in retrieval_set else "streaming"
        return labels


class HeadClassifier:
    """Convenience façade: calibrate then classify heads.

    Parameters
    ----------
    config : DuoAttentionConfig
    """

    def __init__(self, config: DuoAttentionConfig) -> None:
        self._cal = HeadCalibration(config)
        self._cfg = config

    def record(self, layer_idx: int, attn_weights: np.ndarray) -> None:
        """Forward to :meth:`HeadCalibration.record`."""
        self._cal.record(layer_idx, attn_weights)

    def classify(self) -> Dict[HeadKey, HeadLabel]:
        """Return head labels; see :meth:`HeadCalibration.classify`."""
        return self._cal.classify()

    def retrieval_fraction(self) -> float:
        """Fraction of (layer, head) pairs classified as retrieval heads."""
        labels = self.classify()
        n_ret  = sum(1 for v in labels.values() if v == "retrieval")
        return n_ret / max(len(labels), 1)


# ---------------------------------------------------------------------------
# Streaming KV window (circular buffer = sinks + recent window)
# ---------------------------------------------------------------------------

class StreamingKVWindow:
    """Circular-buffer KV cache for streaming heads (sink + recent window).

    Sink positions (0 … sink_tokens-1) are stored permanently.
    Non-sink tokens beyond ``window`` capacity are evicted in FIFO order.

    Parameters
    ----------
    sink_tokens : int
        Number of sink positions to keep permanently.
    window : int
        Maximum recent non-sink positions to retain.
    head_dim : int
        Dimensionality of K/V vectors.
    """

    def __init__(self, sink_tokens: int, window: int, head_dim: int) -> None:
        if sink_tokens < 0:
            raise ValueError("sink_tokens must be ≥ 0")
        if window < 1:
            raise ValueError("window must be ≥ 1")
        if head_dim < 1:
            raise ValueError("head_dim must be ≥ 1")
        self._sink  = sink_tokens
        self._win   = window
        self._dim   = head_dim
        self._sink_k: List[np.ndarray] = []
        self._sink_v: List[np.ndarray] = []
        self._buf_k: List[Optional[np.ndarray]] = [None] * window
        self._buf_v: List[Optional[np.ndarray]] = [None] * window
        self._ptr:   int = 0
        self._n_rec: int = 0   # number of recent tokens stored (≤ window)

    def push(self, pos: int, key: np.ndarray, value: np.ndarray) -> None:
        """Insert a token's KV pair.

        Parameters
        ----------
        pos : int — token's absolute position in the sequence.
        key, value : (head_dim,) float arrays.
        """
        k = np.asarray(key,   dtype=np.float32)
        v = np.asarray(value, dtype=np.float32)
        if pos < self._sink:
            self._sink_k.append(k)
            self._sink_v.append(v)
        else:
            self._buf_k[self._ptr] = k
            self._buf_v[self._ptr] = v
            self._ptr   = (self._ptr + 1) % self._win
            self._n_rec = min(self._n_rec + 1, self._win)

    def get_kv(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return all cached (keys, values) as ``(n_cached, head_dim)`` arrays."""
        all_k = list(self._sink_k)
        all_v = list(self._sink_v)
        if self._n_rec > 0:
            # Reconstruct insertion order from circular buffer
            start = (self._ptr - self._n_rec) % self._win
            for i in range(self._n_rec):
                idx = (start + i) % self._win
                if self._buf_k[idx] is not None:  # pragma: no branch
                    all_k.append(self._buf_k[idx])
                    all_v.append(self._buf_v[idx])
        if not all_k:
            empty = np.empty((0, self._dim), dtype=np.float32)
            return empty, empty.copy()
        return np.stack(all_k), np.stack(all_v)

    def __len__(self) -> int:
        return len(self._sink_k) + self._n_rec


# ---------------------------------------------------------------------------
# DuoKVManager
# ---------------------------------------------------------------------------

class DuoKVManager:
    """Unified per-head KV store with dual-budget management.

    * **Retrieval heads**: store the full KV history (unbounded list).
    * **Streaming heads**: ``StreamingKVWindow`` (sink + recency window).

    Parameters
    ----------
    config : DuoAttentionConfig
    labels : dict mapping ``(layer, head)`` → ``"retrieval"`` | ``"streaming"``.
        Defaults all heads to retrieval when not provided.
    """

    def __init__(
        self,
        config: DuoAttentionConfig,
        labels: Optional[Dict[HeadKey, HeadLabel]] = None,
    ) -> None:
        self._cfg    = config
        self._labels = labels or {}
        # retrieval: (layer, head) → ([keys…], [values…])
        self._full: Dict[HeadKey, Tuple[List[np.ndarray], List[np.ndarray]]] = {}
        # streaming: (layer, head) → StreamingKVWindow
        self._win: Dict[HeadKey, StreamingKVWindow] = {}

    def _label(self, layer: int, head: int) -> HeadLabel:
        return self._labels.get((layer, head), "retrieval")

    def store_kv(
        self,
        layer: int,
        head: int,
        pos: int,
        key: np.ndarray,
        value: np.ndarray,
    ) -> None:
        """Store one token's K/V for the given (layer, head).

        Parameters
        ----------
        layer, head : int — head coordinates.
        pos : int — absolute token position in sequence.
        key, value : (head_dim,) float arrays.
        """
        hk = (layer, head)
        if self._label(layer, head) == "retrieval":
            if hk not in self._full:
                self._full[hk] = ([], [])
            self._full[hk][0].append(np.asarray(key,   dtype=np.float32))
            self._full[hk][1].append(np.asarray(value, dtype=np.float32))
        else:
            if hk not in self._win:
                c = self._cfg
                self._win[hk] = StreamingKVWindow(c.sink_tokens, c.local_window, c.head_dim)
            self._win[hk].push(pos, key, value)

    def load_kv(self, layer: int, head: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return (keys, values) shaped ``(n_cached, head_dim)`` for the head."""
        hk  = (layer, head)
        dim = self._cfg.head_dim
        empty = np.empty((0, dim), dtype=np.float32)
        if self._label(layer, head) == "retrieval":
            if hk not in self._full:
                return empty, empty.copy()
            ks, vs = self._full[hk]
            return np.stack(ks), np.stack(vs)
        else:
            if hk not in self._win:
                return empty, empty.copy()
            return self._win[hk].get_kv()

    def cache_size_tokens(self) -> Dict[str, int]:
        """Return total cached tokens per head type."""
        ret_total = sum(len(v[0]) for v in self._full.values())
        str_total = sum(len(w) for w in self._win.values())
        return {"retrieval": ret_total, "streaming": str_total}

    def clear(self) -> None:
        """Drop all cached KV data."""
        self._full.clear()
        self._win.clear()
