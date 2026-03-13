#!/usr/bin/env python3
"""
squish/paged_kv.py

PagedKV — vLLM-style paged KV cache with virtual block table.

This module implements a paged KV cache following the architecture described in:

    Kwon et al., "Efficient Memory Management for Large Language Model Serving
    with PagedAttention", SOSP 2023. https://arxiv.org/abs/2309.06180

Physical memory is divided into fixed-size blocks (pages). Each sequence
maintains a logical block table that maps its token positions to physical
blocks, enabling non-contiguous memory layout and eliminating fragmentation
from variable-length requests.

The key insight is that logical and physical memory are decoupled: multiple
sequences can share physical blocks (copy-on-write) and the pool is reused
across requests without requiring compaction.

Example usage::

    import numpy as np
    from squish.paged_kv import PagedKVConfig, PagedKVCache

    config = PagedKVConfig(block_size=16, n_blocks=256, n_heads=32, head_dim=128)
    cache = PagedKVCache(config)

    key = np.random.randn(config.kv_n_heads, config.head_dim).astype(np.float32)
    val = np.random.randn(config.kv_n_heads, config.head_dim).astype(np.float32)
    cache.append(seq_id=0, key=key, value=val)

    keys, values = cache.gather(seq_id=0)  # (kv_n_heads, n_tokens, head_dim)
    print(f"utilization={cache.utilization:.2%}, sequences={cache.n_sequences}")
    cache.free(seq_id=0)
"""

from __future__ import annotations

__all__ = [
    "PagedKVConfig",
    "KVBlock",
    "BlockTable",
    "PagedKVCache",
    "PagedKVStats",
]

from dataclasses import dataclass
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class PagedKVConfig:
    """Configuration for the paged KV cache.

    Attributes:
        block_size:  Number of token slots per physical block.
        n_blocks:    Total physical blocks in the memory pool.
        n_heads:     Number of query attention heads (used as default for
                     ``kv_n_heads`` under MHA).
        head_dim:    Dimension of each attention head vector.
        kv_n_heads:  Number of KV heads.  Defaults to ``n_heads`` when
                     ``None`` (multi-head attention).  Set lower for GQA/MQA.
    """

    block_size: int = 16
    n_blocks: int = 512
    n_heads: int = 32
    head_dim: int = 128
    kv_n_heads: Optional[int] = None

    def __post_init__(self) -> None:
        if self.kv_n_heads is None:
            self.kv_n_heads = self.n_heads
        if self.block_size < 1:
            raise ValueError(f"block_size must be >= 1, got {self.block_size}")
        if self.n_blocks < 1:
            raise ValueError(f"n_blocks must be >= 1, got {self.n_blocks}")
        if self.n_heads < 1:
            raise ValueError(f"n_heads must be >= 1, got {self.n_heads}")
        if self.head_dim < 1:
            raise ValueError(f"head_dim must be >= 1, got {self.head_dim}")
        if self.kv_n_heads < 1:
            raise ValueError(f"kv_n_heads must be >= 1, got {self.kv_n_heads}")
        if self.n_heads % self.kv_n_heads != 0:
            raise ValueError(
                f"n_heads ({self.n_heads}) must be divisible by "
                f"kv_n_heads ({self.kv_n_heads})"
            )


# ---------------------------------------------------------------------------
# Physical block
# ---------------------------------------------------------------------------

@dataclass
class KVBlock:
    """A single physical memory block holding key and value tensors.

    Each block stores up to ``block_size`` tokens across all KV heads.

    Attributes:
        keys:    Shape ``(kv_n_heads, block_size, head_dim)``, float32.
        values:  Shape ``(kv_n_heads, block_size, head_dim)``, float32.
        fill:    Number of valid token slots currently written (0..block_size).
    """

    keys: np.ndarray
    values: np.ndarray
    fill: int = 0

    @classmethod
    def empty(cls, kv_n_heads: int, block_size: int, head_dim: int) -> KVBlock:
        """Allocate a zero-initialised block of the given dimensions."""
        return cls(
            keys=np.zeros((kv_n_heads, block_size, head_dim), dtype=np.float32),
            values=np.zeros((kv_n_heads, block_size, head_dim), dtype=np.float32),
            fill=0,
        )

    def reset(self) -> None:
        """Zero all data and reset the fill counter (return to pool)."""
        self.keys[:] = 0.0
        self.values[:] = 0.0
        self.fill = 0


# ---------------------------------------------------------------------------
# Block table
# ---------------------------------------------------------------------------

class BlockTable:
    """Maps logical sequence token positions to physical block indices.

    Maintains a free-list of physical block IDs and assigns them to sequences
    on demand.  A sequence's logical address space is partitioned into windows
    of ``block_size`` tokens; each window maps to exactly one physical block.

    Args:
        n_blocks:   Total number of physical blocks in the pool.
        block_size: Tokens per physical block.
    """

    def __init__(self, n_blocks: int, block_size: int) -> None:
        self._n_blocks = n_blocks
        self._block_size = block_size
        # Unordered set of free physical block IDs.
        self._free: set[int] = set(range(n_blocks))
        # seq_id → ordered list of assigned physical block IDs.
        self._seq_blocks: dict[int, list[int]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def allocate(self, seq_id: int, n_tokens: int) -> None:
        """Ensure *seq_id* owns enough physical blocks for *n_tokens* tokens.

        Existing blocks are preserved; new blocks are appended only when the
        current capacity is insufficient.  Raises ``MemoryError`` when the
        pool is exhausted.

        Args:
            seq_id:   Sequence identifier (arbitrary non-negative integer).
            n_tokens: Total tokens the sequence must be able to hold.
        """
        if seq_id not in self._seq_blocks:
            self._seq_blocks[seq_id] = []

        blocks_required = _ceil_div(n_tokens, self._block_size)
        to_alloc = blocks_required - len(self._seq_blocks[seq_id])
        if to_alloc <= 0:
            return

        if len(self._free) < to_alloc:
            raise MemoryError(
                f"Physical block pool exhausted: need {to_alloc} additional "
                f"block(s), only {len(self._free)} free."
            )

        for _ in range(to_alloc):
            phys_id = self._free.pop()
            self._seq_blocks[seq_id].append(phys_id)

    def free(self, seq_id: int) -> None:
        """Return all physical blocks owned by *seq_id* to the free pool.

        No-op if *seq_id* is not currently tracked.
        """
        if seq_id not in self._seq_blocks:
            return
        for phys_id in self._seq_blocks.pop(seq_id):
            self._free.add(phys_id)

    def get_block_indices(self, seq_id: int) -> list[int]:
        """Return the ordered list of physical block IDs assigned to *seq_id*.

        Raises:
            KeyError: if *seq_id* has not been allocated.
        """
        if seq_id not in self._seq_blocks:
            raise KeyError(f"seq_id {seq_id!r} not found in block table")
        return list(self._seq_blocks[seq_id])

    @property
    def n_free_blocks(self) -> int:
        """Number of physical blocks currently available for allocation."""
        return len(self._free)

    @property
    def n_allocated_blocks(self) -> int:
        """Number of physical blocks currently assigned to active sequences."""
        return self._n_blocks - len(self._free)


# ---------------------------------------------------------------------------
# Paged KV cache
# ---------------------------------------------------------------------------

class PagedKVCache:
    """Full paged KV cache backed by a fixed-size pool of physical blocks.

    Tokens are appended one at a time (decode step) or in bulk (prefill via
    repeated ``append`` calls).  Physical blocks are lazily allocated as
    sequences grow and immediately returned to the pool on ``free``.

    Args:
        config: A :class:`PagedKVConfig` instance describing the pool shape.
    """

    def __init__(self, config: PagedKVConfig) -> None:
        self._cfg = config
        self._block_table = BlockTable(config.n_blocks, config.block_size)
        # Pre-allocate the entire physical pool up front to avoid GC pressure.
        self._pool: list[KVBlock] = [
            KVBlock.empty(config.kv_n_heads, config.block_size, config.head_dim)
            for _ in range(config.n_blocks)
        ]
        # Logical token count per active sequence.
        self._seq_len: dict[int, int] = {}
        # Cumulative statistics.
        self._stats = PagedKVStats(n_total_blocks=config.n_blocks)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def append(self, seq_id: int, key: np.ndarray, value: np.ndarray) -> None:
        """Append one token's KV vectors to the end of sequence *seq_id*.

        A new physical block is allocated automatically when the current last
        block becomes full.

        Args:
            seq_id: Arbitrary integer sequence identifier.
            key:    Shape ``(kv_n_heads, head_dim)``, any float dtype.
            value:  Shape ``(kv_n_heads, head_dim)``, any float dtype.

        Raises:
            ValueError: if ``key`` or ``value`` have unexpected shapes.
            MemoryError: if the physical block pool is exhausted.
        """
        self._validate_kv(key, value)

        if seq_id not in self._seq_len:
            self._seq_len[seq_id] = 0
            self._stats.total_allocations += 1

        cur_len = self._seq_len[seq_id]
        new_len = cur_len + 1

        # Ensure the block table has capacity for the new length.
        self._block_table.allocate(seq_id, new_len)

        # Compute which physical block and slot to write into.
        block_idx = cur_len // self._cfg.block_size
        slot_idx = cur_len % self._cfg.block_size

        phys_ids = self._block_table.get_block_indices(seq_id)
        block = self._pool[phys_ids[block_idx]]
        block.keys[:, slot_idx, :] = key
        block.values[:, slot_idx, :] = value
        block.fill = slot_idx + 1

        self._seq_len[seq_id] = new_len

        # Track peak utilisation.
        in_use = self._block_table.n_allocated_blocks
        if in_use > self._stats.peak_blocks_used:
            self._stats.peak_blocks_used = in_use

    def gather(self, seq_id: int) -> tuple[np.ndarray, np.ndarray]:
        """Collect all stored KV tensors for *seq_id* into dense arrays.

        Iterates the physical blocks in logical order and copies valid token
        slots into a contiguous output buffer.

        Returns:
            ``(keys, values)`` — each of shape
            ``(kv_n_heads, n_tokens, head_dim)``, dtype float32.

        Raises:
            KeyError: if *seq_id* is not currently active.
        """
        if seq_id not in self._seq_len:
            raise KeyError(f"seq_id {seq_id!r} has no active KV data")

        n_tokens = self._seq_len[seq_id]
        cfg = self._cfg
        keys_out = np.empty((cfg.kv_n_heads, n_tokens, cfg.head_dim), dtype=np.float32)
        vals_out = np.empty((cfg.kv_n_heads, n_tokens, cfg.head_dim), dtype=np.float32)

        phys_ids = self._block_table.get_block_indices(seq_id)
        dst_offset = 0
        for block_num, phys_id in enumerate(phys_ids):
            block_start = block_num * cfg.block_size
            block_end = min(block_start + cfg.block_size, n_tokens)
            n_valid = block_end - block_start
            if n_valid <= 0:  # pragma: no cover
                break
            block = self._pool[phys_id]
            keys_out[:, dst_offset : dst_offset + n_valid, :] = block.keys[:, :n_valid, :]
            vals_out[:, dst_offset : dst_offset + n_valid, :] = block.values[:, :n_valid, :]
            dst_offset += n_valid

        return keys_out, vals_out

    def free(self, seq_id: int) -> None:
        """Release all resources held by sequence *seq_id*.

        Physical blocks are zeroed and returned to the pool.
        No-op if *seq_id* is not currently tracked.
        """
        if seq_id not in self._seq_len:
            return
        for phys_id in self._block_table.get_block_indices(seq_id):
            self._pool[phys_id].reset()
        self._block_table.free(seq_id)
        del self._seq_len[seq_id]
        self._stats.total_frees += 1

    @property
    def utilization(self) -> float:
        """Fraction of physical blocks currently in use (0.0–1.0)."""
        return self._block_table.n_allocated_blocks / self._cfg.n_blocks

    @property
    def n_sequences(self) -> int:
        """Number of sequences currently holding cached data."""
        return len(self._seq_len)

    @property
    def stats(self) -> PagedKVStats:
        """Cumulative cache statistics (updated in place)."""
        return self._stats

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _validate_kv(self, key: np.ndarray, value: np.ndarray) -> None:
        cfg = self._cfg
        expected = (cfg.kv_n_heads, cfg.head_dim)
        if key.shape != expected:
            raise ValueError(
                f"key shape {key.shape} does not match expected {expected}"
            )
        if value.shape != expected:
            raise ValueError(
                f"value shape {value.shape} does not match expected {expected}"
            )


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

@dataclass
class PagedKVStats:
    """Cumulative statistics collected by a :class:`PagedKVCache` instance.

    Attributes:
        total_allocations:  Number of distinct sequences ever inserted.
        total_frees:        Number of sequences explicitly freed.
        peak_blocks_used:   Maximum simultaneous physical blocks in use.
        n_total_blocks:     Pool size set at cache construction time.
    """

    total_allocations: int = 0
    total_frees: int = 0
    peak_blocks_used: int = 0
    n_total_blocks: int = 0

    @property
    def utilization_rate(self) -> float:
        """Peak physical-block utilisation as a fraction of pool capacity.

        Returns ``peak_blocks_used / n_total_blocks``, or 0.0 when the pool
        size is unknown.
        """
        if self.n_total_blocks == 0:
            return 0.0
        return self.peak_blocks_used / self.n_total_blocks


# ---------------------------------------------------------------------------
# Internal utilities
# ---------------------------------------------------------------------------

def _ceil_div(numerator: int, denominator: int) -> int:
    """Integer ceiling division without floating-point conversion."""
    return (numerator + denominator - 1) // denominator
