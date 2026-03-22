"""squish/kv/inf_memory.py

InfMemory — Training-Free Long-Context External Block Memory.

Reference
---------
Xiao et al. "InfLLM: Training-Free Long-Context Extrapolation for LLMs
with an Efficient Context Memory." NeurIPS 2024 (arXiv 2402.04617).

Algorithm
---------
Distant context blocks are compressed to a small set of *representative
tokens* (the mean of each block's key vectors) and stored in an external
memory. At decode time, query vectors retrieve the most relevant blocks via
dot-product similarity; the full K/V of those blocks is loaded back for
attention.

This module provides:

1. ``InfMemory.store_block(K, V)`` — compress and store a KV block.
2. ``InfMemory.retrieve(Q, top_k)`` — return top-k block K/V pairs.
3. ``InfMemory.compress_block(K)`` — compute block representative (mean key).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

__all__ = [
    "InfMemoryConfig",
    "MemoryBlock",
    "InfMemory",
]


@dataclass
class InfMemoryConfig:
    """Configuration for :class:`InfMemory`.

    Attributes:
        n_heads: Number of attention heads.
        head_dim: Dimension per head.
        block_size: Tokens per memory block.
        max_blocks: Maximum stored blocks (oldest evicted first).
        top_k_retrieve: Default number of blocks returned per query.
    """

    n_heads: int = 32
    head_dim: int = 128
    block_size: int = 64
    max_blocks: int = 1024
    top_k_retrieve: int = 8

    def __post_init__(self) -> None:
        if self.n_heads < 1:
            raise ValueError(f"n_heads must be >= 1; got {self.n_heads}")
        if self.head_dim < 1:
            raise ValueError(f"head_dim must be >= 1; got {self.head_dim}")
        if self.block_size < 1:
            raise ValueError(f"block_size must be >= 1; got {self.block_size}")
        if self.max_blocks < 1:
            raise ValueError(f"max_blocks must be >= 1; got {self.max_blocks}")
        if self.top_k_retrieve < 1:
            raise ValueError(f"top_k_retrieve must be >= 1; got {self.top_k_retrieve}")


@dataclass
class MemoryBlock:
    """A compressed KV block stored in external memory.

    Attributes:
        K: Full key tensor ``(block_size, n_heads, head_dim)``.
        V: Full value tensor ``(block_size, n_heads, head_dim)``.
        representative: Mean key per head ``(n_heads, head_dim)``.
        block_id: Global insertion index.
    """

    K: np.ndarray
    V: np.ndarray
    representative: np.ndarray
    block_id: int

    @property
    def seq_len(self) -> int:
        return self.K.shape[0]


class InfMemory:
    """Block-level external KV memory with representative-token retrieval.

    Example::

        cfg = InfMemoryConfig(n_heads=4, head_dim=16, block_size=8)
        mem = InfMemory(cfg)

        K = np.random.randn(8, 4, 16).astype(np.float32)
        V = np.random.randn(8, 4, 16).astype(np.float32)
        mem.store_block(K, V)

        Q = np.random.randn(4, 16).astype(np.float32)   # (n_heads, head_dim)
        blocks = mem.retrieve(Q, top_k=1)
    """

    def __init__(self, config: Optional[InfMemoryConfig] = None) -> None:
        self._cfg = config or InfMemoryConfig()
        self._blocks: List[MemoryBlock] = []
        self._next_id: int = 0

    @property
    def config(self) -> InfMemoryConfig:
        return self._cfg

    @property
    def n_blocks(self) -> int:
        """Number of blocks currently in memory."""
        return len(self._blocks)

    def compress_block(self, K: np.ndarray) -> np.ndarray:
        """Compute the representative for a KV block.

        Args:
            K: ``(seq_len, n_heads, head_dim)``.

        Returns:
            ``(n_heads, head_dim)`` mean key per head.
        """
        K = np.asarray(K, dtype=np.float32)
        return K.mean(axis=0)

    def store_block(self, K: np.ndarray, V: np.ndarray) -> int:
        """Compress and store a KV block.

        Args:
            K: ``(seq_len, n_heads, head_dim)``.
            V: ``(seq_len, n_heads, head_dim)``.

        Returns:
            Block id assigned to this block.
        """
        K = np.asarray(K, dtype=np.float32)
        V = np.asarray(V, dtype=np.float32)
        rep = self.compress_block(K)
        if len(self._blocks) >= self._cfg.max_blocks:
            self._blocks.pop(0)
        block = MemoryBlock(K=K, V=V, representative=rep, block_id=self._next_id)
        self._blocks.append(block)
        self._next_id += 1
        return block.block_id

    def retrieve(
        self,
        Q: np.ndarray,
        top_k: Optional[int] = None,
    ) -> List[MemoryBlock]:
        """Retrieve the top-k blocks most relevant to the query.

        Args:
            Q: ``(n_heads, head_dim)`` query per head.
            top_k: Number of blocks to return; defaults to config value.

        Returns:
            List of up to ``top_k`` :class:`MemoryBlock` objects, ordered
            by descending similarity score.
        """
        Q = np.asarray(Q, dtype=np.float32)
        if top_k is None:
            top_k = self._cfg.top_k_retrieve
        if not self._blocks:
            return []
        top_k = min(top_k, len(self._blocks))

        # Score each block: mean dot-product between Q and representative
        # Q: (n_heads, head_dim), rep: (n_heads, head_dim)
        scores = np.array(
            [(Q * b.representative).sum() for b in self._blocks],
            dtype=np.float32,
        )
        indices = np.argsort(scores)[::-1][:top_k]
        return [self._blocks[i] for i in indices]

    def retrieve_kv(
        self, Q: np.ndarray, top_k: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Retrieve and concatenate K/V tensors for the top-k blocks.

        Returns:
            ``(K_concat, V_concat)`` both ``(total_tokens, n_heads, head_dim)``.
            Returns zero-length arrays if memory is empty.
        """
        blocks = self.retrieve(Q, top_k)
        if not blocks:
            cfg = self._cfg
            empty = np.zeros((0, cfg.n_heads, cfg.head_dim), dtype=np.float32)
            return empty, empty.copy()
        K_all = np.concatenate([b.K for b in blocks], axis=0)
        V_all = np.concatenate([b.V for b in blocks], axis=0)
        return K_all, V_all

    def reset(self) -> None:
        """Clear all stored blocks."""
        self._blocks.clear()
        self._next_id = 0
