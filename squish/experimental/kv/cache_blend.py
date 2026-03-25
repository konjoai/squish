"""squish/kv/cache_blend.py

CacheBlend — Fast LLM Serving with Cached Knowledge Fusion.

Reference
---------
Yao et al. "CacheBlend: Fast Large Language Model Serving with Cached
Knowledge Fusion." EuroSys 2025 (arXiv:2405.16444).

Algorithm
---------
In RAG-style workloads the same document chunks are repeatedly prefilled.
CacheBlend reuses pre-computed KV caches from previously seen documents and
*blends* them with a small amount of selective recomputation so that
cross-document attention is correctly captured.

The key insight: only the tokens whose attention scores to the *new* query
context change significantly between the cached version and the live context
need recomputation.  CacheBlend selects a ``recompute_ratio`` fraction of
positions per document chunk, recomputes their K/V, and splices them back into
the cached block.

Key properties
--------------
* NumPy-only; no GPU dependency.
* ``recompute_ratio`` — fraction of KV positions refreshed per blend step.
* ``importance_fn`` — how to score which positions need recomputation:
  ``"l2"`` (distance to running mean), or ``"random"`` (baseline).
* ``max_blocks`` — maximum number of cached KV blocks stored.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

__all__ = [
    "CacheBlendConfig",
    "KVBlock",
    "CacheBlendKV",
]

# ── Config ────────────────────────────────────────────────────────────────────


@dataclass
class CacheBlendConfig:
    """Configuration for :class:`CacheBlendKV`.

    Attributes:
        recompute_ratio: Fraction of KV positions recomputed per blend call.
        importance_fn: How to pick positions to recompute (``"l2"`` or
            ``"random"``).
        max_blocks: Maximum number of document KV blocks to cache.
        n_heads: Attention heads per layer.
        head_dim: Dimension per head.
    """

    recompute_ratio: float = 0.1
    importance_fn: str = "l2"
    max_blocks: int = 64
    n_heads: int = 8
    head_dim: int = 64

    def __post_init__(self) -> None:
        if not (0.0 < self.recompute_ratio <= 1.0):
            raise ValueError(
                f"recompute_ratio must be in (0, 1]; got {self.recompute_ratio}"
            )
        if self.importance_fn not in ("l2", "random"):
            raise ValueError(
                f"importance_fn must be 'l2' or 'random'; got '{self.importance_fn}'"
            )
        if self.max_blocks < 1:
            raise ValueError(f"max_blocks must be ≥ 1; got {self.max_blocks}")
        if self.n_heads < 1:
            raise ValueError(f"n_heads must be ≥ 1; got {self.n_heads}")
        if self.head_dim < 1:
            raise ValueError(f"head_dim must be ≥ 1; got {self.head_dim}")


# ── Data container ─────────────────────────────────────────────────────────────


@dataclass
class KVBlock:
    """A cached KV block for a document chunk.

    Attributes:
        block_id: Unique string identifier.
        K: ``(n_heads, S, head_dim)`` cached key tensor.
        V: ``(n_heads, S, head_dim)`` cached value tensor.
    """

    block_id: str
    K: np.ndarray
    V: np.ndarray


# ── Core class ─────────────────────────────────────────────────────────────────


class CacheBlendKV:
    """KV block reuse with selective recomputation for RAG / shared-prefix workloads.

    Example::

        cfg   = CacheBlendConfig(recompute_ratio=0.1)
        blender = CacheBlendKV(cfg)

        # Store a pre-computed KV block
        K_doc = np.random.randn(4, 64, 16).astype(np.float32)
        V_doc = np.random.randn(4, 64, 16).astype(np.float32)
        blender.store("doc_42", K_doc, V_doc)

        # Blend: return K/V with important positions refreshed
        K_new = np.random.randn(4, 64, 16).astype(np.float32)
        V_new = np.random.randn(4, 64, 16).astype(np.float32)
        K_out, V_out = blender.blend("doc_42", K_new, V_new)
    """

    def __init__(
        self,
        config: Optional[CacheBlendConfig] = None,
        seed: int = 0,
    ) -> None:
        self.config = config or CacheBlendConfig()
        self._blocks: Dict[str, KVBlock] = {}
        self._rng = np.random.default_rng(seed)
        self._n_blends: int = 0

    # ── Store ─────────────────────────────────────────────────────────────────

    def store(self, block_id: str, K: np.ndarray, V: np.ndarray) -> None:
        """Cache a document's KV block.

        Args:
            block_id: Unique document / chunk identifier.
            K: ``(n_heads, S, head_dim)`` key tensor.
            V: ``(n_heads, S, head_dim)`` value tensor.
        """
        if len(self._blocks) >= self.config.max_blocks and block_id not in self._blocks:
            # Evict oldest entry (FIFO)
            oldest = next(iter(self._blocks))
            del self._blocks[oldest]
        self._blocks[block_id] = KVBlock(
            block_id=block_id,
            K=np.asarray(K, dtype=np.float32).copy(),
            V=np.asarray(V, dtype=np.float32).copy(),
        )

    # ── Blend ─────────────────────────────────────────────────────────────────

    def blend(
        self,
        block_id: str,
        K_fresh: np.ndarray,
        V_fresh: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Blend cached KV with freshly computed KV.

        ``recompute_ratio`` positions are identified as the most important
        (by L2 distance between cached and fresh), and the cached block is
        updated at those positions with the fresh values.

        Args:
            block_id: Identifier of the cached block to blend.
            K_fresh: ``(n_heads, S, head_dim)`` freshly computed keys.
            V_fresh: ``(n_heads, S, head_dim)`` freshly computed values.

        Returns:
            ``(K_blended, V_blended)`` each ``(n_heads, S, head_dim)``.

        Raises:
            KeyError: If ``block_id`` is not in the cache.
        """
        if block_id not in self._blocks:
            raise KeyError(f"Block '{block_id}' not found in cache")

        cached = self._blocks[block_id]
        K_c = cached.K.copy()
        V_c = cached.V.copy()
        K_f = np.asarray(K_fresh, dtype=np.float32)
        V_f = np.asarray(V_fresh, dtype=np.float32)

        H, S, d = K_c.shape
        n_recompute = max(1, int(S * self.config.recompute_ratio))

        # Score positions across heads by L2 distance (or random)
        if self.config.importance_fn == "l2":
            diff = ((K_f - K_c) ** 2).sum(axis=(0, 2))  # (S,)
            top_idx = np.argsort(-diff)[:n_recompute]
        else:
            top_idx = self._rng.choice(S, size=n_recompute, replace=False)

        K_c[:, top_idx, :] = K_f[:, top_idx, :]
        V_c[:, top_idx, :] = V_f[:, top_idx, :]

        self._n_blends += 1
        return K_c, V_c

    # ── Introspection ─────────────────────────────────────────────────────────

    def has_block(self, block_id: str) -> bool:
        """Return True if a block with this id is cached."""
        return block_id in self._blocks

    def n_blocks(self) -> int:
        """Return number of cached blocks."""
        return len(self._blocks)

    def n_blends(self) -> int:
        """Return number of blend operations performed."""
        return self._n_blends

    def evict(self, block_id: str) -> None:
        """Remove a block from the cache."""
        self._blocks.pop(block_id, None)

    def clear(self) -> None:
        """Remove all cached blocks."""
        self._blocks.clear()
        self._n_blends = 0

    def __repr__(self) -> str:
        return (
            f"CacheBlendKV(n_blocks={self.n_blocks()}, "
            f"recompute_ratio={self.config.recompute_ratio}, "
            f"n_blends={self._n_blends})"
        )
