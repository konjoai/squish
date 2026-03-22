"""squish/kv/paged_attn.py

PagedAttention — Physical-Page Block Manager for K/V Tensors.

Reference
---------
Kwon et al. "Efficient Memory Management for Large Language Model Serving
with PagedAttention." SOSP 2023 / production 2024.

Algorithm
---------
PagedAttention maps the logical KV cache of each sequence onto a set of
fixed-size physical *pages* (blocks), eliminating internal fragmentation:

1. The allocator maintains a pool of free physical blocks.
2. Each sequence is assigned a *block table*: a list of physical block IDs.
3. When a sequence needs more K/V storage, a new block is allocated from
   the free pool and appended to its block table.
4. When a sequence finishes, its blocks are returned to the free pool.
5. Cross-sequence prefix sharing: multiple block tables can share the same
   physical block IDs for a common prefix.

Key properties
--------------
* NumPy-only simulation.
* ``block_size`` — tokens per physical block (default 16).
* ``n_blocks`` — total physical blocks in the pool.
* ``n_heads`` — number of K/V heads.
* ``head_dim`` — dimension per head.
* Ref-counted blocks for prefix sharing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

import numpy as np

__all__ = [
    "PagedAttnConfig",
    "PagedAttnBlock",
    "PagedAttention",
]


# ── Config ────────────────────────────────────────────────────────────────────


@dataclass
class PagedAttnConfig:
    """Configuration for :class:`PagedAttention`.

    Attributes:
        max_blocks: Total number of physical blocks in the pool.
        block_size: Number of KV tokens per block.
        n_heads: Number of K/V attention heads.
        head_dim: Dimension per head.
    """

    max_blocks: int = 256
    block_size: int = 16
    n_heads: int = 8
    head_dim: int = 64


# ── Block ─────────────────────────────────────────────────────────────────────


@dataclass
class PagedAttnBlock:
    """A single physical KV page.

    Attributes:
        block_id: Unique block identifier.
        keys: Shape ``(n_heads, block_size, head_dim)``.
        values: Shape ``(n_heads, block_size, head_dim)``.
        fill: Number of valid token slots used.
        ref_count: Number of sequences sharing this block.
    """

    block_id: int
    keys: np.ndarray
    values: np.ndarray
    fill: int = 0
    ref_count: int = 1

    @property
    def is_full(self) -> bool:
        return self.fill >= self.keys.shape[1]

    def write_token(self, k: np.ndarray, v: np.ndarray) -> int:
        """Write one token's K/V at position ``fill``.  Returns the slot index."""
        slot = self.fill
        self.keys[:, slot, :] = k
        self.values[:, slot, :] = v
        self.fill += 1
        return slot


# ── Module ────────────────────────────────────────────────────────────────────


class PagedAttention:
    """Physical-page block manager for KV cache.

    Parameters
    ----------
    config:
        PagedAttention configuration.
    """

    def __init__(self, config: Optional[PagedAttnConfig] = None) -> None:
        self._cfg = config or PagedAttnConfig()
        h, bs, d = self._cfg.n_heads, self._cfg.block_size, self._cfg.head_dim
        # Pre-allocate the block pool
        self._blocks: List[PagedAttnBlock] = [
            PagedAttnBlock(
                block_id=i,
                keys=np.zeros((h, bs, d), dtype=np.float32),
                values=np.zeros((h, bs, d), dtype=np.float32),
            )
            for i in range(self._cfg.max_blocks)
        ]
        self._free: Set[int] = set(range(self._cfg.max_blocks))
        # Sequence → list of block IDs
        self._seq_tables: Dict[int, List[int]] = {}
        self._next_seq_id: int = 0

    # ── Allocator API ─────────────────────────────────────────────────────────

    @property
    def config(self) -> PagedAttnConfig:
        return self._cfg

    @property
    def free_blocks(self) -> int:
        return len(self._free)

    @property
    def used_blocks(self) -> int:
        return self._cfg.max_blocks - len(self._free)

    def _alloc_block(self) -> int:
        if not self._free:
            raise MemoryError("PagedAttention: no free blocks available")
        bid = next(iter(self._free))
        self._free.remove(bid)
        block = self._blocks[bid]
        block.fill = 0
        block.ref_count = 1
        return bid

    def _free_block(self, bid: int) -> None:
        block = self._blocks[bid]
        block.ref_count -= 1
        if block.ref_count <= 0:
            self._free.add(bid)

    # ── Sequence API ──────────────────────────────────────────────────────────

    def create_sequence(self) -> int:
        """Allocate a new sequence slot.

        Returns
        -------
        int
            Sequence ID.
        """
        seq_id = self._next_seq_id
        self._next_seq_id += 1
        self._seq_tables[seq_id] = []
        return seq_id

    def append_token(self, seq_id: int, key: np.ndarray, value: np.ndarray) -> None:
        """Append one token's KV to a sequence.

        Parameters
        ----------
        seq_id: Sequence ID from :meth:`create_sequence`.
        key, value: Shape ``(n_heads, head_dim)``.
        """
        table = self._seq_tables[seq_id]
        # If no blocks or current block is full, allocate a new one
        if not table or self._blocks[table[-1]].is_full:
            bid = self._alloc_block()
            table.append(bid)
        k = np.asarray(key, dtype=np.float32)
        v = np.asarray(value, dtype=np.float32)
        self._blocks[table[-1]].write_token(k, v)

    def free_sequence(self, seq_id: int) -> None:
        """Release all blocks belonging to a sequence."""
        table = self._seq_tables.pop(seq_id, [])
        for bid in table:
            self._free_block(bid)

    def get_kv(self, seq_id: int) -> tuple[np.ndarray, np.ndarray]:
        """Reconstruct the full KV tensors for a sequence.

        Returns
        -------
        Tuple of ``(keys, values)`` each with shape
        ``(seq_len, n_heads, head_dim)``.
        """
        table = self._seq_tables.get(seq_id, [])
        if not table:
            h, d = self._cfg.n_heads, self._cfg.head_dim
            return np.zeros((0, h, d), dtype=np.float32), np.zeros((0, h, d), dtype=np.float32)
        key_chunks = []
        val_chunks = []
        for bid in table:
            blk = self._blocks[bid]
            key_chunks.append(blk.keys[:, : blk.fill, :])
            val_chunks.append(blk.values[:, : blk.fill, :])
        K = np.concatenate(key_chunks, axis=1)  # (n_heads, seq_len, head_dim)
        V = np.concatenate(val_chunks, axis=1)
        return K.transpose(1, 0, 2), V.transpose(1, 0, 2)  # (seq_len, n_heads, head_dim)

    def seq_length(self, seq_id: int) -> int:
        """Return the number of tokens stored for a sequence."""
        table = self._seq_tables.get(seq_id, [])
        total = 0
        for bid in table:
            total += self._blocks[bid].fill
        return total

    def share_prefix(self, source_seq_id: int, new_seq_id: int, prefix_len: int) -> None:
        """Share a prefix from an existing sequence with a new one.

        Increments ref-count on shared blocks rather than copying them.

        Parameters
        ----------
        source_seq_id: Sequence whose first ``prefix_len`` tokens are shared.
        new_seq_id: New sequence that will share those blocks.
        prefix_len: Number of prefix tokens to share.
        """
        src_table = self._seq_tables[source_seq_id]
        dst_table = self._seq_tables[new_seq_id]
        bs = self._cfg.block_size
        shared_blocks = (prefix_len + bs - 1) // bs
        for bid in src_table[:shared_blocks]:
            self._blocks[bid].ref_count += 1
            dst_table.append(bid)
