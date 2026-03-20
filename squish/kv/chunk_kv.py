"""
ChunkKV — Semantic Chunk-Based KV Cache Eviction.

Inspired by: "ChunkKV: Semantic Chunk-Based KV Cache Compression" (NeurIPS 2025).
Key ideas:
  1.  Treat semantic chunks (contiguous token windows) as the basic eviction unit
      rather than individual tokens — this preserves linguistic coherence.
  2.  Layer-wise index reuse: if adjacent layers select similar chunks, reuse the
      top-layer index set rather than recomputing, gaining ~26.5% throughput.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ChunkKVConfig:
    """Configuration for ChunkKV eviction."""
    chunk_size: int = 16             # tokens per semantic chunk
    budget_ratio: float = 0.5        # fraction of chunks to retain after eviction
    layer_reuse: bool = True         # enable cross-layer index reuse
    reuse_window: int = 2            # reuse indices within this many consecutive layers
    score_fn: str = "max_attn"       # "max_attn" | "mean_attn" | "norm"

    def __post_init__(self) -> None:
        if self.chunk_size < 1:
            raise ValueError(f"chunk_size must be >= 1, got {self.chunk_size}")
        if not (0.0 < self.budget_ratio <= 1.0):
            raise ValueError(f"budget_ratio must be in (0, 1], got {self.budget_ratio}")
        if self.score_fn not in ("max_attn", "mean_attn", "norm"):
            raise ValueError(f"Unknown score_fn '{self.score_fn}'")


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class ChunkScore:
    """Score record for a single chunk."""
    chunk_idx: int
    score: float
    token_start: int
    token_end: int  # exclusive


@dataclass
class ChunkKVStats:
    """Runtime statistics for ChunkKV."""
    eviction_calls: int = 0
    reuse_hits: int = 0
    reuse_misses: int = 0
    tokens_evicted: int = 0
    tokens_kept: int = 0

    @property
    def reuse_hit_rate(self) -> float:
        total = self.reuse_hits + self.reuse_misses
        return self.reuse_hits / total if total > 0 else 0.0

    @property
    def eviction_rate(self) -> float:
        total = self.tokens_evicted + self.tokens_kept
        return self.tokens_evicted / total if total > 0 else 0.0

    def __repr__(self) -> str:
        return (
            f"ChunkKVStats(evictions={self.eviction_calls}, "
            f"reuse_hit={self.reuse_hit_rate:.1%}, "
            f"eviction_rate={self.eviction_rate:.1%})"
        )


# ---------------------------------------------------------------------------
# Core manager
# ---------------------------------------------------------------------------

class ChunkKVManager:
    """Manages semantic-chunk KV eviction for a single attention layer."""

    def __init__(self, config: ChunkKVConfig) -> None:
        self.config = config
        self.stats = ChunkKVStats()
        # cache of most-recent eviction indices for cross-layer reuse
        self._cached_indices: Optional[np.ndarray] = None
        self._cached_layer: Optional[int] = None

    # ------------------------------------------------------------------
    # Chunk scoring
    # ------------------------------------------------------------------

    def score_chunks(
        self,
        key: np.ndarray,
        query: Optional[np.ndarray] = None,
    ) -> List[ChunkScore]:
        """Score every chunk in the KV sequence.

        Args:
            key:   (seq_len, n_heads, d_k) or (seq_len, d_k) key tensor.
            query: (n_q_heads, d_k) current query vector (used for max_attn /
                   mean_attn scoring).  If None, falls back to norm scoring.

        Returns:
            List of ChunkScore ordered by chunk index (low → high).
        """
        # Flatten multi-head if needed: (seq, d)
        key_2d = key.reshape(key.shape[0], -1).astype(np.float32)
        seq_len = key_2d.shape[0]
        cs = self.config.chunk_size

        n_chunks = math.ceil(seq_len / cs)
        scores: List[ChunkScore] = []

        for c in range(n_chunks):
            t_start = c * cs
            t_end = min((c + 1) * cs, seq_len)
            chunk_keys = key_2d[t_start:t_end]           # (cs, d)

            if self.config.score_fn == "norm":
                score = float(np.linalg.norm(chunk_keys, axis=1).mean())
            elif query is not None:
                q_flat = query.reshape(-1).astype(np.float32)
                d = min(chunk_keys.shape[1], q_flat.shape[0])
                attn = chunk_keys[:, :d] @ q_flat[:d]    # (cs,)
                if self.config.score_fn == "max_attn":
                    score = float(attn.max())
                else:
                    score = float(attn.mean())
            else:
                # fallback to norm when no query provided
                score = float(np.linalg.norm(chunk_keys, axis=1).mean())

            scores.append(ChunkScore(
                chunk_idx=c,
                score=score,
                token_start=t_start,
                token_end=t_end,
            ))

        return scores

    # ------------------------------------------------------------------
    # Eviction
    # ------------------------------------------------------------------

    def evict(
        self,
        key: np.ndarray,
        value: np.ndarray,
        scores: Optional[List[ChunkScore]] = None,
        query: Optional[np.ndarray] = None,
        layer_idx: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Evict low-importance chunks from key/value tensors.

        Args:
            key:       (seq_len, ...) key tensor.
            value:     (seq_len, ...) value tensor.
            scores:    pre-computed chunk scores (optional).
            query:     current query for attention-based scoring (optional).
            layer_idx: if given, may use or populate the cross-layer reuse cache.

        Returns:
            (kept_key, kept_value, kept_indices) where kept_indices is a 1-D
            int64 array of the token positions that were retained.
        """
        seq_len = key.shape[0]
        cs = self.config.chunk_size

        # --- try cross-layer index reuse ---
        if (
            self.config.layer_reuse
            and layer_idx is not None
            and self._cached_layer is not None
            and abs(layer_idx - self._cached_layer) <= self.config.reuse_window
            and self._cached_indices is not None
        ):
            kept_idx = self._cached_indices
            # guard against cached index from a different seq length
            if kept_idx.max() < seq_len:
                self.stats.reuse_hits += 1
                kept_key = key[kept_idx]
                kept_val = value[kept_idx]
                self.stats.eviction_calls += 1
                self.stats.tokens_kept += len(kept_idx)
                self.stats.tokens_evicted += seq_len - len(kept_idx)
                return kept_key, kept_val, kept_idx

        self.stats.reuse_misses += 1

        # --- compute scores if not provided ---
        if scores is None:
            scores = self.score_chunks(key, query)

        n_chunks = len(scores)
        n_keep = max(1, round(n_chunks * self.config.budget_ratio))

        top_chunks = sorted(scores, key=lambda s: s.score, reverse=True)[:n_keep]
        top_chunks_sorted = sorted(top_chunks, key=lambda s: s.chunk_idx)

        kept_idx = np.concatenate([
            np.arange(c.token_start, c.token_end, dtype=np.int64)
            for c in top_chunks_sorted
        ])
        kept_idx = kept_idx[kept_idx < seq_len]   # safety clip

        # cache for layer reuse
        if self.config.layer_reuse and layer_idx is not None:
            self._cached_indices = kept_idx
            self._cached_layer = layer_idx

        kept_key = key[kept_idx]
        kept_val = value[kept_idx]

        self.stats.eviction_calls += 1
        self.stats.tokens_kept += len(kept_idx)
        self.stats.tokens_evicted += seq_len - len(kept_idx)

        return kept_key, kept_val, kept_idx

    # ------------------------------------------------------------------
    # Manual cache control
    # ------------------------------------------------------------------

    def invalidate_reuse_cache(self) -> None:
        """Clear cross-layer index cache (call at the start of each new request)."""
        self._cached_indices = None
        self._cached_layer = None

    def set_reuse_indices(self, layer_idx: int, indices: np.ndarray) -> None:
        """Manually seed the reuse cache (used by upper-level orchestrator)."""
        self._cached_indices = indices
        self._cached_layer = layer_idx

    def get_reuse_indices(self, layer_idx: int) -> Optional[np.ndarray]:
        """Return cached indices if still applicable for layer_idx."""
        if (
            self._cached_layer is not None
            and abs(layer_idx - self._cached_layer) <= self.config.reuse_window
        ):
            return self._cached_indices
        return None

    def __repr__(self) -> str:
        return (
            f"ChunkKVManager(chunk_size={self.config.chunk_size}, "
            f"budget={self.config.budget_ratio:.0%}, {self.stats})"
        )


# ---------------------------------------------------------------------------
# Multi-layer orchestrator
# ---------------------------------------------------------------------------

class ChunkKVOrchestrator:
    """Coordinates ChunkKV eviction across all transformer layers."""

    def __init__(self, config: ChunkKVConfig, n_layers: int) -> None:
        self.config = config
        self.n_layers = n_layers
        self._managers: Dict[int, ChunkKVManager] = {
            i: ChunkKVManager(config) for i in range(n_layers)
        }

    def evict_layer(
        self,
        layer_idx: int,
        key: np.ndarray,
        value: np.ndarray,
        query: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Evict for a specific layer, with cross-layer index propagation."""
        mgr = self._managers[layer_idx]

        # propagate reuse cache from adjacent layer
        if self.config.layer_reuse and layer_idx > 0:
            prev_mgr = self._managers[layer_idx - 1]
            prev_idx = prev_mgr.get_reuse_indices(layer_idx - 1)
            if prev_idx is not None and prev_idx.max() < key.shape[0]:
                mgr.set_reuse_indices(layer_idx - 1, prev_idx)

        return mgr.evict(key, value, query=query, layer_idx=layer_idx)

    def reset_request(self) -> None:
        """Clear all cross-layer reuse caches (call between requests)."""
        for mgr in self._managers.values():
            mgr.invalidate_reuse_cache()

    @property
    def aggregate_stats(self) -> ChunkKVStats:
        agg = ChunkKVStats()
        for mgr in self._managers.values():
            s = mgr.stats
            agg.eviction_calls += s.eviction_calls
            agg.reuse_hits += s.reuse_hits
            agg.reuse_misses += s.reuse_misses
            agg.tokens_evicted += s.tokens_evicted
            agg.tokens_kept += s.tokens_kept
        return agg

    def __repr__(self) -> str:
        return (
            f"ChunkKVOrchestrator(layers={self.n_layers}, "
            f"chunk_size={self.config.chunk_size}, "
            f"budget={self.config.budget_ratio:.0%})"
        )


import math  # noqa: E402 (already imported but also needed in score_chunks)
