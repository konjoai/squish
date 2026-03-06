"""Phase 2A — Paged KV Cache (PagedAttention-style block table).

Stores K/V tensors in fixed-size physical blocks (PAGE_SIZE = 16 tokens each).
Multiple requests can share prefix blocks via copy-on-write ref-counting.
The RadixTree (radix_cache.py) stores which blocks correspond to which token
prefixes; when a prefix match is found the corresponding blocks can be forked
into a new request's PageBlockTable so prefill is skipped for those tokens.

Key classes
-----------
PhysicalBlock   — lightweight page metadata (index, ref_count, last_access)
BlockAllocator  — pre-allocated pool management + LRU eviction
PageBlockTable  — per-request logical→physical block mapping
PagedKVCache    — numpy backing store + per-layer scatter/gather API

Integration model
-----------------
A single ``PagedKVCache`` instance lives in the server (instantiated at startup
when ``--paged-attention`` is set).  The existing ``QuantizedKVCache``
generation path remains the *hot* path inside the model forward pass.

After each prefill/decode step the calling code records K/V tensors via
``store_token()`` / ``advance_token()``.  On a RadixTree prefix hit, the
matching blocks are forked into the new request and the KV tensors for those
prefix tokens are loaded back into the ``QuantizedKVCache`` layers via
``load_prefix_into_quantized()``, eliminating the prefill step for those tokens.
"""

from __future__ import annotations

import time
import threading
import numpy as np
from collections import deque
from dataclasses import dataclass, field

__all__ = [
    "PAGE_SIZE",
    "PhysicalBlock",
    "BlockAllocator",
    "PageBlockTable",
    "PagedKVCache",
]

PAGE_SIZE: int = 16  # tokens per physical page


# ── Physical block ────────────────────────────────────────────────────────────


@dataclass
class PhysicalBlock:
    """Metadata for one physical KV page."""

    idx: int
    ref_count: int = 0
    last_access: float = field(default_factory=time.monotonic)

    def touch(self) -> None:
        self.last_access = time.monotonic()


# ── Block allocator ──────────────────────────────────────────────────────────


class BlockAllocator:
    """
    Manages a fixed pool of physical blocks numbered 0..num_blocks-1.

    Thread-safe via an internal lock.  The eviction policy is LRU by
    ``last_access`` timestamp which allows evicting unreferenced blocks
    when the pool is exhausted.
    """

    def __init__(self, num_blocks: int) -> None:
        self._num_blocks = num_blocks
        self._blocks: list[PhysicalBlock] = [
            PhysicalBlock(idx=i) for i in range(num_blocks)
        ]
        # All blocks start free
        self._free: deque[int] = deque(range(num_blocks))
        self._lock = threading.Lock()

    # ── Allocation ────────────────────────────────────────────────────────────

    def alloc(self) -> int | None:
        """
        Allocate one block from the free pool.
        Returns the physical block index, or ``None`` if the pool is exhausted.
        """
        with self._lock:
            if not self._free:
                return None
            idx = self._free.popleft()
            b = self._blocks[idx]
            b.ref_count = 1
            b.touch()
            return idx

    def free(self, idx: int) -> None:
        """
        Decrement ref_count for block *idx*.
        When it reaches zero the block is returned to the free pool.
        """
        with self._lock:
            b = self._blocks[idx]
            b.ref_count = max(0, b.ref_count - 1)
            if b.ref_count == 0:
                self._free.append(idx)

    def fork(self, idx: int) -> None:
        """
        Increment ref_count for block *idx* (copy-on-write prefix sharing).
        Forked blocks are NOT returned to the free pool until all holders free
        them.
        """
        with self._lock:
            self._blocks[idx].ref_count += 1
            self._blocks[idx].touch()

    # ── Introspection ─────────────────────────────────────────────────────────

    @property
    def free_count(self) -> int:
        with self._lock:
            return len(self._free)

    @property
    def total_blocks(self) -> int:
        return self._num_blocks

    @property
    def used_count(self) -> int:
        return self._num_blocks - self.free_count

    def evict_lru(self, n: int) -> list[int]:
        """
        Evict up to *n* unreferenced blocks ordered by oldest ``last_access``.
        Returns the list of evicted block indices (now free).

        Note: blocks already in the free deque are skipped (they have
        ``ref_count == 0`` by invariant but are already accounted for).
        """
        with self._lock:
            free_set = set(self._free)
            evictable = sorted(
                (b for b in self._blocks if b.ref_count == 0 and b.idx not in free_set),
                key=lambda b: b.last_access,
            )
            evicted: list[int] = []
            for b in evictable[:n]:
                self._free.append(b.idx)
                evicted.append(b.idx)
            return evicted


# ── Per-request block table ──────────────────────────────────────────────────


class PageBlockTable:
    """
    Maps logical page indices to physical block indices for one request.

    A page is "full" when it contains exactly ``PAGE_SIZE`` token positions.
    Append-only during generation; blocks are freed in bulk via ``free_all()``.
    """

    __slots__ = ("_blocks", "_write_offset", "_allocator")

    def __init__(self, allocator: BlockAllocator) -> None:
        self._allocator = allocator
        self._blocks: list[int] = []      # logical_page_idx → physical_block_idx
        self._write_offset: int = PAGE_SIZE  # sentinel: force alloc on first append

    # ── Space management ──────────────────────────────────────────────────────

    def ensure_space(self) -> int | None:
        """
        Ensure the current page has room for one token.
        Allocates a new block if the current page is full.
        Returns the physical block index for the current write slot,
        or ``None`` if the allocator is exhausted.
        """
        if self._write_offset >= PAGE_SIZE:
            idx = self._allocator.alloc()
            if idx is None:
                return None
            self._blocks.append(idx)
            self._write_offset = 0
        return self._blocks[-1]

    def advance(self) -> None:
        """Advance the write pointer by one token slot."""
        self._write_offset += 1

    # ── Introspection ─────────────────────────────────────────────────────────

    @property
    def block_refs(self) -> list[int]:
        """Physical block indices in logical order (copy)."""
        return list(self._blocks)

    @property
    def write_offset(self) -> int:
        """Slot index within the current (last) block (0..PAGE_SIZE-1)."""
        return self._write_offset

    @property
    def n_tokens_stored(self) -> int:
        """Total token positions stored so far."""
        if not self._blocks:
            return 0
        return (len(self._blocks) - 1) * PAGE_SIZE + self._write_offset

    # ── Prefix forking (copy-on-write) ────────────────────────────────────────

    def fork_blocks(self, existing_blocks: list[int]) -> None:
        """
        Initialize this table with *existing_blocks* from a cached prefix
        (copy-on-write).  Increments each block's ref_count via the allocator.

        The write pointer is placed at the start of a NEW block so the next
        ``ensure_space()`` call allocates a fresh block without mutating the
        shared prefix pages.
        """
        for idx in existing_blocks:
            self._allocator.fork(idx)
        self._blocks = list(existing_blocks)
        # Force a new block alloc on next write
        self._write_offset = PAGE_SIZE

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def free_all(self) -> None:
        """Release all blocks held by this table."""
        for idx in self._blocks:
            self._allocator.free(idx)
        self._blocks = []
        self._write_offset = PAGE_SIZE


# ── Paged KV cache ───────────────────────────────────────────────────────────


class PagedKVCache:
    """
    NumPy-backed paged KV cache for all transformer layers.

    Backing store shape::

        [num_blocks, PAGE_SIZE, n_layers, 2, n_kv_heads, head_dim]

    where ``axis[3]`` is ``0=K, 1=V``.  This layout keeps each physical block
    contiguous in memory, which minimises scatter/gather I/O.

    Usage
    -----
    At startup (when ``--paged-attention`` is set)::

        paged = PagedKVCache.from_model(model, metal_fraction=0.25)

    Per request::

        table = paged.new_request(req_id)
        # ... during prefill for each token t, each layer l:
        paged.store_token(req_id, layer_l, k_np, v_np)
        # ... after all layers for token t:
        paged.advance_token(req_id)

    On a RadixTree prefix hit::

        new_table = paged.fork_request(req_id, existing_block_refs)
        k_np, v_np = paged.get_kv_for_layer(req_id, layer_l)
        # load k_np / v_np into QuantizedKVCache to skip prefill

    Cleanup::

        paged.free_request(req_id)
    """

    def __init__(
        self,
        num_blocks: int,
        n_layers: int,
        n_kv_heads: int,
        head_dim: int,
        dtype: np.dtype = np.float16,
    ) -> None:
        self._allocator  = BlockAllocator(num_blocks)
        self._n_layers   = n_layers
        self._n_heads    = n_kv_heads
        self._head_dim   = head_dim
        self._dtype      = np.dtype(dtype)
        self._num_blocks = num_blocks
        self._lock       = threading.Lock()

        # Backing store — allocated lazily to avoid RAM pressure at import time
        self._store: np.ndarray | None = None

        # Active request tables: req_id → PageBlockTable
        self._tables: dict[str, PageBlockTable] = {}

    # ── Lazy backing allocation ───────────────────────────────────────────────

    def _get_store(self) -> np.ndarray:
        if self._store is None:
            self._store = np.zeros(
                (self._num_blocks, PAGE_SIZE, self._n_layers, 2,
                 self._n_heads, self._head_dim),
                dtype=self._dtype,
            )
        return self._store

    # ── Request lifecycle ─────────────────────────────────────────────────────

    def new_request(self, req_id: str) -> PageBlockTable:
        """Register a new request and return its ``PageBlockTable``."""
        table = PageBlockTable(self._allocator)
        with self._lock:
            self._tables[req_id] = table
        return table

    def free_request(self, req_id: str) -> None:
        """Release all blocks held by *req_id*."""
        with self._lock:
            table = self._tables.pop(req_id, None)
        if table is not None:
            table.free_all()

    def fork_request(self, new_req_id: str, existing_blocks: list[int]) -> PageBlockTable:
        """
        Start a new request pre-populated with shared prefix blocks.
        All blocks receive a forked ref (copy-on-write semantics).
        """
        table = PageBlockTable(self._allocator)
        table.fork_blocks(existing_blocks)
        with self._lock:
            self._tables[new_req_id] = table
        return table

    # ── KV store ─────────────────────────────────────────────────────────────

    def store_token(
        self,
        req_id: str,
        layer_idx: int,
        k: np.ndarray,   # (n_kv_heads, head_dim)
        v: np.ndarray,   # (n_kv_heads, head_dim)
    ) -> bool:
        """
        Write one token's K/V for *layer_idx* into the current page slot.
        Returns ``True`` on success, ``False`` if the allocator is exhausted.

        Call ``advance_token()`` after writing **all** layers for one token.
        """
        store = self._get_store()
        with self._lock:
            table = self._tables.get(req_id)
            if table is None:
                return False
            block_idx = table.ensure_space()
            if block_idx is None:
                return False
            slot = table.write_offset
        store[block_idx, slot, layer_idx, 0] = k
        store[block_idx, slot, layer_idx, 1] = v
        return True

    def advance_token(self, req_id: str) -> None:
        """
        Advance the write pointer after all layers have been stored for one
        token.  Must be called exactly once per token (after iterating all
        layers).
        """
        with self._lock:
            table = self._tables.get(req_id)
            if table is not None:
                table.advance()

    # ── KV load ──────────────────────────────────────────────────────────────

    def get_kv_for_layer(
        self,
        req_id: str,
        layer_idx: int,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """
        Return ``(K, V)`` arrays of shape ``(n_tokens, n_kv_heads, head_dim)``
        for *layer_idx*, gathering all pages in logical order.
        Returns ``None`` if the request is not registered.
        """
        store = self._get_store()
        with self._lock:
            table = self._tables.get(req_id)
            if table is None:
                return None
            blocks      = table.block_refs
            n_full      = len(blocks) - 1 if blocks else 0
            last_offset = table.write_offset

        if not blocks:
            empty = np.empty((0, self._n_heads, self._head_dim), dtype=self._dtype)
            return empty, empty.copy()

        k_parts: list[np.ndarray] = []
        v_parts: list[np.ndarray] = []
        for i, blk in enumerate(blocks):
            slots = PAGE_SIZE if i < n_full else last_offset
            if slots == 0:
                continue
            k_parts.append(store[blk, :slots, layer_idx, 0])
            v_parts.append(store[blk, :slots, layer_idx, 1])

        k = (np.concatenate(k_parts, axis=0) if k_parts
             else np.empty((0, self._n_heads, self._head_dim), dtype=self._dtype))
        v = (np.concatenate(v_parts, axis=0) if v_parts else k.copy())
        return k, v

    def get_block_refs(self, req_id: str) -> list[int]:
        """Return the current list of physical block indices for *req_id*."""
        with self._lock:
            table = self._tables.get(req_id)
            return table.block_refs if table else []

    def n_tokens_stored(self, req_id: str) -> int:
        """Return the number of token positions stored for *req_id*."""
        with self._lock:
            table = self._tables.get(req_id)
            return table.n_tokens_stored if table else 0

    # ── Prefix snapshot ───────────────────────────────────────────────────────

    def snapshot_prefix(
        self,
        req_id: str,
        n_prefix_tokens: int,
        snapshot_req_id: str,
    ) -> list[int]:
        """
        Fork the first *n_prefix_tokens* of *req_id*'s block table into a new
        entry under *snapshot_req_id*.  Used by the RadixTree when recording a
        new prefix entry after a completed request.

        Returns the forked block refs (empty list on failure).

        The snapshot entry can later be used for ``fork_request()`` when a new
        request shares this prefix.
        """
        with self._lock:
            src = self._tables.get(req_id)
            if src is None:
                return []
            all_blocks = src.block_refs

        # How many full pages does n_prefix_tokens span?
        n_pages = min(len(all_blocks), (n_prefix_tokens + PAGE_SIZE - 1) // PAGE_SIZE)
        prefix_blocks = all_blocks[:n_pages]

        new_table = PageBlockTable(self._allocator)
        new_table.fork_blocks(prefix_blocks)
        with self._lock:
            self._tables[snapshot_req_id] = new_table
        return list(prefix_blocks)

    # ── Memory pressure relief ────────────────────────────────────────────────

    def evict_lru_blocks(self, n: int) -> int:
        """
        Evict up to *n* unreferenced blocks by LRU.
        Returns the count of blocks actually freed.
        """
        freed = self._allocator.evict_lru(n)
        return len(freed)

    # ── Stats ─────────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        alloc = self._allocator
        bytes_per_block = (
            PAGE_SIZE * self._n_layers * 2
            * self._n_heads * self._head_dim
            * self._dtype.itemsize
        )
        return {
            "total_blocks":    alloc.total_blocks,
            "free_blocks":     alloc.free_count,
            "used_blocks":     alloc.used_count,
            "active_requests": len(self._tables),
            "memory_mb":       round(
                self._num_blocks * bytes_per_block / 1_048_576, 1
            ),
            "page_size":       PAGE_SIZE,
            "n_layers":        self._n_layers,
            "n_kv_heads":      self._n_heads,
            "head_dim":        self._head_dim,
        }

    # ── Convenience constructor ───────────────────────────────────────────────

    @classmethod
    def from_model(
        cls,
        model,
        metal_fraction: float = 0.25,
        dtype: np.dtype = np.float16,
    ) -> "PagedKVCache":
        """
        Construct a ``PagedKVCache`` sized to *metal_fraction* of available
        unified memory by introspecting the model config.

        Falls back to conservative defaults (128 blocks) if the model config
        is not readable.

        Parameters
        ----------
        model           : mlx_lm model with a ``.config`` or ``.args`` attribute
        metal_fraction  : fraction of total RAM to budget for paged KV storage
        dtype           : storage dtype (default float16)
        """
        try:
            import psutil
            ram_bytes = psutil.virtual_memory().total
        except ImportError:
            ram_bytes = 16 * 1_073_741_824  # assume 16 GB

        # Default dims for Qwen3-8B / similar GQA models
        n_layers  = 32
        n_heads   = 8      # kv heads (GQA)
        head_dim  = 128

        cfg = getattr(model, "config", None) or getattr(model, "args", None)
        if cfg is not None:
            n_layers = int(getattr(cfg, "num_hidden_layers",
                           getattr(cfg, "n_layers", n_layers)))
            n_heads  = int(getattr(cfg, "num_key_value_heads",
                           getattr(cfg, "n_kv_heads",
                           getattr(cfg, "num_heads", n_heads))))
            head_dim = int(getattr(cfg, "head_dim",
                           getattr(cfg, "hidden_size",
                                   head_dim * n_heads) // n_heads))

        budget_bytes    = int(ram_bytes * metal_fraction)
        bytes_per_block = (PAGE_SIZE * n_layers * 2 * n_heads * head_dim
                           * np.dtype(dtype).itemsize)
        num_blocks      = max(64, budget_bytes // bytes_per_block)

        return cls(
            num_blocks=num_blocks,
            n_layers=n_layers,
            n_kv_heads=n_heads,
            head_dim=head_dim,
            dtype=dtype,
        )
