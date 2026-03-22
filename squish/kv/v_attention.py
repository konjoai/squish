"""squish/kv/v_attention.py

vAttentionKV — Virtual-Memory KV Cache with OS-style Page Management.

Reference
---------
Prabhu et al. "vAttention: Dynamic Memory Management for Serving LLMs
without PagedAttention." OSDI 2024.

Algorithm
---------
KV cache is backed by a contiguous *physical* pool split into fixed-size
pages.  Each sequence holds a logical page table mapping logical positions
to physical pages.  Unlike copy-on-write paging, allocation here is eager
but zero-copy within a page once written.  Pages are allocated on demand
and freed when a sequence is evicted, giving lower fragmentation than
naive pre-allocation.

This module provides:

1. ``vAttentionKV.allocate(seq_id, n_tokens)`` — reserve pages.
2. ``vAttentionKV.store_token(seq_id, pos, k, v)`` — write a single token.
3. ``vAttentionKV.get_kv(seq_id)`` — read full K/V for a sequence.
4. ``vAttentionKV.free(seq_id)`` — release pages back to the pool.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

__all__ = [
    "vAttentionConfig",
    "vAttentionKV",
]


@dataclass
class vAttentionConfig:
    """Configuration for :class:`vAttentionKV`.

    Attributes:
        page_size: Tokens per page.
        max_pages: Total pages in the physical pool.
        n_heads: Attention head count.
        head_dim: Dimension per head.
    """

    page_size: int = 16
    max_pages: int = 1024
    n_heads: int = 32
    head_dim: int = 128

    def __post_init__(self) -> None:
        if self.page_size < 1:
            raise ValueError(f"page_size must be >= 1; got {self.page_size}")
        if self.max_pages < 1:
            raise ValueError(f"max_pages must be >= 1; got {self.max_pages}")
        if self.n_heads < 1:
            raise ValueError(f"n_heads must be >= 1; got {self.n_heads}")
        if self.head_dim < 1:
            raise ValueError(f"head_dim must be >= 1; got {self.head_dim}")


class vAttentionKV:
    """Virtual-memory KV cache manager.

    Physical KV storage is a single pre-allocated pool partitioned into
    pages.  Each sequence has a logical page table.  Pages are allocated on
    first write and returned to the free list on ``free()``.

    Example::

        cfg = vAttentionConfig(page_size=4, max_pages=16, n_heads=2, head_dim=8)
        kv = vAttentionKV(cfg)
        kv.allocate("seq0", n_tokens=6)
        k0 = np.ones((2, 8))  # (n_heads, head_dim)
        v0 = np.zeros((2, 8))
        kv.store_token("seq0", pos=0, k=k0, v=v0)
        K, V = kv.get_kv("seq0")
    """

    def __init__(self, config: Optional[vAttentionConfig] = None) -> None:
        self._cfg = config or vAttentionConfig()
        c = self._cfg
        # Physical pool: (max_pages, page_size, n_heads, head_dim)
        self._K_pool = np.zeros(
            (c.max_pages, c.page_size, c.n_heads, c.head_dim), dtype=np.float32
        )
        self._V_pool = np.zeros_like(self._K_pool)
        self._free_pages: List[int] = list(range(c.max_pages))
        # seq_id -> list of physical page indices
        self._page_tables: Dict[str, List[int]] = {}
        # seq_id -> logical length (tokens written so far)
        self._seq_lengths: Dict[str, int] = {}

    @property
    def config(self) -> vAttentionConfig:
        return self._cfg

    @property
    def n_allocated_pages(self) -> int:
        """Pages currently assigned to active sequences."""
        return sum(len(v) for v in self._page_tables.values())

    @property
    def n_free_pages(self) -> int:
        """Pages available for allocation."""
        return len(self._free_pages)

    @property
    def fragmentation_ratio(self) -> float:
        """Fraction of allocated tokens that are padding (unfilled slots).

        Returns 0.0 if nothing is allocated.
        """
        total_slots = self.n_allocated_pages * self._cfg.page_size
        if total_slots == 0:
            return 0.0
        used = sum(self._seq_lengths.values())
        return (total_slots - used) / total_slots

    def _n_pages_needed(self, n_tokens: int) -> int:
        ps = self._cfg.page_size
        return (n_tokens + ps - 1) // ps

    def allocate(self, seq_id: str, n_tokens: int) -> List[int]:
        """Reserve pages for *seq_id* capable of holding *n_tokens*.

        If *seq_id* already exists, extra pages are appended as needed.

        Args:
            seq_id: Sequence identifier.
            n_tokens: Total tokens to accommodate (logical length).

        Returns:
            Full physical page list for the sequence.

        Raises:
            MemoryError: If the free pool is exhausted.
        """
        if seq_id not in self._page_tables:
            self._page_tables[seq_id] = []
            self._seq_lengths[seq_id] = 0

        current_pages = len(self._page_tables[seq_id])
        needed = self._n_pages_needed(n_tokens) - current_pages
        if needed > len(self._free_pages):
            raise MemoryError(
                f"Not enough free pages: need {needed}, have {len(self._free_pages)}"
            )
        for _ in range(needed):
            page = self._free_pages.pop()
            self._page_tables[seq_id].append(page)
        return list(self._page_tables[seq_id])

    def store_token(
        self, seq_id: str, pos: int, k: np.ndarray, v: np.ndarray
    ) -> None:
        """Write key/value for token at logical position *pos*.

        Args:
            seq_id: Sequence identifier (must have been allocated).
            pos: Zero-based token position.
            k: ``(n_heads, head_dim)`` key vector.
            v: ``(n_heads, head_dim)`` value vector.
        """
        if seq_id not in self._page_tables:
            raise KeyError(f"seq_id {seq_id!r} not allocated; call allocate() first")
        pages = self._page_tables[seq_id]
        ps = self._cfg.page_size
        page_idx = pos // ps
        slot_idx = pos % ps
        if page_idx >= len(pages):
            raise IndexError(
                f"pos={pos} exceeds allocated capacity for seq {seq_id!r}"
            )
        phys = pages[page_idx]
        self._K_pool[phys, slot_idx] = np.asarray(k, dtype=np.float32)
        self._V_pool[phys, slot_idx] = np.asarray(v, dtype=np.float32)
        if pos >= self._seq_lengths[seq_id]:
            self._seq_lengths[seq_id] = pos + 1

    def get_kv(self, seq_id: str) -> Tuple[np.ndarray, np.ndarray]:
        """Retrieve the full K/V tensors for *seq_id*.

        Returns:
            ``(K, V)`` each ``(seq_len, n_heads, head_dim)`` where
            *seq_len* is the number of stored tokens.
        """
        if seq_id not in self._page_tables:
            raise KeyError(f"seq_id {seq_id!r} not found")
        pages = self._page_tables[seq_id]
        seq_len = self._seq_lengths[seq_id]
        if seq_len == 0 or not pages:
            c = self._cfg
            empty = np.zeros((0, c.n_heads, c.head_dim), dtype=np.float32)
            return empty, empty.copy()
        K_chunks = [self._K_pool[p] for p in pages]
        V_chunks = [self._V_pool[p] for p in pages]
        K_all = np.concatenate(K_chunks, axis=0)[:seq_len]
        V_all = np.concatenate(V_chunks, axis=0)[:seq_len]
        return K_all, V_all

    def free(self, seq_id: str) -> None:
        """Release all pages owned by *seq_id* back to the free pool.

        No-op if *seq_id* is not present.
        """
        pages = self._page_tables.pop(seq_id, [])
        self._seq_lengths.pop(seq_id, None)
        self._free_pages.extend(pages)
