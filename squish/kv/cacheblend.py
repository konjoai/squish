"""squish/kv/cacheblend.py

CacheBlend — Partial KV prefix reuse for RAG context.

Reference
---------
Yao et al. "CacheBlend: Fast Large Language Model Serving for RAG with
Cached Knowledge Fusion." EuroSys 2025. arXiv:2405.16444.

Algorithm
---------
In RAG pipelines the same document chunks appear repeatedly as context
across many queries.  Prefilling their KV entries from scratch for every
request wastes compute.

CacheBlend stores the pre-computed KV blocks for known documents and reuses
them, but recomputes only the *divergent region* — the short suffix where
the system prompt or query interacts with the document boundary — to
maintain attention quality:

1. Cache lookup: find the longest cached KV prefix for the input ids.
2. Identify divergence point d: the first position where input ids differ
   from the cached prefix.
3. Recompute the KV suffix from d onward (async, non-blocking).
4. Splice: concatenate cached_kv[:d] with freshly computed kv[d:].
5. Return blended KV sequence.

The divergence recomputation is typically < 5–10% of full prefill, giving
a 95%+ prefill FLOP reduction on repeated-context RAG queries.

This module is distinct from the semantic_cache (which checks embedding
similarity) — CacheBlend operates at the exact token-id level.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class CacheBlendConfig:
    """Configuration for CacheBlend.

    Parameters
    ----------
    n_heads:
        Number of attention heads.
    head_dim:
        Dimension per head.
    max_cached_seqs:
        Maximum number of document KV sequences to keep in the store.
    blend_overlap:
        Number of tokens before the divergence point to recompute for
        smooth blending (default 4).
    """

    n_heads: int = 32
    head_dim: int = 128
    max_cached_seqs: int = 256
    blend_overlap: int = 4

    def __post_init__(self) -> None:
        if self.n_heads < 1:
            raise ValueError(f"n_heads must be >= 1; got {self.n_heads}")
        if self.head_dim < 1:
            raise ValueError(f"head_dim must be >= 1; got {self.head_dim}")
        if self.max_cached_seqs < 1:
            raise ValueError(f"max_cached_seqs must be >= 1; got {self.max_cached_seqs}")
        if self.blend_overlap < 1:
            raise ValueError(f"blend_overlap must be >= 1; got {self.blend_overlap}")


# ---------------------------------------------------------------------------
# Cached entry
# ---------------------------------------------------------------------------

@dataclass
class CachedKVEntry:
    """A stored pre-computed KV sequence for a document.

    Parameters
    ----------
    token_ids:
        Token ids that generated this KV sequence.
    K:
        Key tensor ``(seq_len, n_heads, head_dim)``.
    V:
        Value tensor ``(seq_len, n_heads, head_dim)``.
    """

    token_ids: list[int]
    K: np.ndarray
    V: np.ndarray

    @property
    def seq_len(self) -> int:
        return len(self.token_ids)


# ---------------------------------------------------------------------------
# Blend result
# ---------------------------------------------------------------------------

@dataclass
class CacheBlendResult:
    """Result of a CacheBlend operation.

    Parameters
    ----------
    K:
        Blended key sequence ``(total_len, n_heads, head_dim)``.
    V:
        Blended value sequence ``(total_len, n_heads, head_dim)``.
    cached_tokens:
        Number of tokens served from cache.
    recomputed_tokens:
        Number of tokens that required fresh attention computation.
    divergence_point:
        Token index where the cached and live prefixes diverged.
    """

    K: np.ndarray
    V: np.ndarray
    cached_tokens: int
    recomputed_tokens: int
    divergence_point: int

    @property
    def cache_hit_ratio(self) -> float:
        total = self.cached_tokens + self.recomputed_tokens
        return self.cached_tokens / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Module
# ---------------------------------------------------------------------------

class CacheBlend:
    """Partial-KV prefix reuse for RAG context.

    Parameters
    ----------
    config:
        CacheBlend configuration.
    """

    def __init__(self, config: Optional[CacheBlendConfig] = None) -> None:
        self._cfg = config or CacheBlendConfig()
        # KV store: list of CachedKVEntry (oldest evicted first)
        self._store: list[CachedKVEntry] = []

    @property
    def config(self) -> CacheBlendConfig:
        return self._cfg

    @property
    def n_cached(self) -> int:
        """Number of document KV sequences in the store."""
        return len(self._store)

    # ------------------------------------------------------------------
    # Store management
    # ------------------------------------------------------------------

    def store_kv(
        self,
        token_ids: list[int],
        K: np.ndarray,
        V: np.ndarray,
    ) -> None:
        """Store a pre-computed KV sequence.

        Parameters
        ----------
        token_ids:
            Token ids for the document.
        K:
            Shape ``(seq_len, n_heads, head_dim)``.
        V:
            Shape ``(seq_len, n_heads, head_dim)``.
        """
        if len(self._store) >= self._cfg.max_cached_seqs:
            self._store.pop(0)  # LRU eviction (simple FIFO approximation)
        self._store.append(
            CachedKVEntry(
                token_ids=list(token_ids),
                K=np.asarray(K, dtype=np.float32),
                V=np.asarray(V, dtype=np.float32),
            )
        )

    def _find_longest_prefix(
        self, token_ids: list[int]
    ) -> tuple[Optional[CachedKVEntry], int]:
        """Find the cached entry with the longest matching prefix.

        Returns
        -------
        tuple[Optional[CachedKVEntry], int]
            (best_entry, match_len) where match_len is the number of
            matching leading tokens.
        """
        best_entry = None
        best_len = 0
        for entry in self._store:
            match_len = 0
            for a, b in zip(token_ids, entry.token_ids):
                if a == b:
                    match_len += 1
                else:
                    break
            if match_len > best_len:
                best_len = match_len
                best_entry = entry
        return best_entry, best_len

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def blend(
        self,
        token_ids: list[int],
        fresh_K: np.ndarray,
        fresh_V: np.ndarray,
    ) -> CacheBlendResult:
        """Produce a blended KV sequence using cached prefix where possible.

        Parameters
        ----------
        token_ids:
            Full token id sequence for the current query+context.
        fresh_K:
            Freshly computed K for the *full* sequence
            ``(total_len, n_heads, head_dim)``.  The blended result
            replaces the prefix with the cached version.
        fresh_V:
            Freshly computed V — same shape as fresh_K.

        Returns
        -------
        CacheBlendResult
        """
        fresh_K = np.asarray(fresh_K, dtype=np.float32)
        fresh_V = np.asarray(fresh_V, dtype=np.float32)
        total_len = len(token_ids)

        entry, match_len = self._find_longest_prefix(token_ids)

        # Compute divergence point (with overlap)
        overlap = self._cfg.blend_overlap
        if entry is None or match_len == 0:
            # No usable cache — return fresh KV entirely
            return CacheBlendResult(
                K=fresh_K,
                V=fresh_V,
                cached_tokens=0,
                recomputed_tokens=total_len,
                divergence_point=0,
            )

        divergence = max(0, match_len - overlap)

        # Splice: cached prefix + fresh suffix
        K_blended = fresh_K.copy()
        V_blended = fresh_V.copy()
        K_blended[:divergence] = entry.K[:divergence]
        V_blended[:divergence] = entry.V[:divergence]

        return CacheBlendResult(
            K=K_blended,
            V=V_blended,
            cached_tokens=divergence,
            recomputed_tokens=total_len - divergence,
            divergence_point=divergence,
        )
