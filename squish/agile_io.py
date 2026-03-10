"""
squish/agile_io.py

AGILE-inspired Async I/O Manager for NVMe-backed KV-cache tiers.

Based on:
  "AGILE: A GPU-Centric Asynchronous NVMe I/O Library for LLM Inference"
  arXiv:2504.19365 (SC'25)
  github.com/arc-research-lab/AGILE

Problem
-------
Squish's cold KV tier uses ``numpy.memmap`` (CPU-mediated NVMe reads).  This
means the GPU must stall until the CPU has finished reading from NVMe, which
becomes the dominant bottleneck as the KV tier grows.  AGILE solves this on
CUDA+NVMe systems by initiating I/O directly from GPU threads with a lock-free
async completion model.

This module provides the *architectural equivalent* for Apple Silicon:

  NVMe  →  AgileIOManager (background thread pool)  →  in-memory buffer cache
          ↑                                                     ↓
       prefetch()                                           get() blocks only
       called during                                        on final miss
       prev step compute

On Apple Silicon the "GPU-initiated DMA" analogy maps to:
  - prefetch() runs on a background IO thread (not CPU main thread)
  - Metal blit encoder overlap is achieved by running model compute on the
    main thread while prefetch runs concurrently in the io_thread_pool
  - A future Metal-native backend can replace _io_thread_pool with
    MTLCommandBuffer async blit encoders for true GPU-driven DMA

The module is **framework-agnostic** (no mlx/torch dependency): it operates
on raw file paths and returns bytes / np.ndarray.  The caller is responsible
for numpy-loading the result.

Design: Lock-Free Async Transaction Model (AGILE §3.3)
-------------------------------------------------------
AGILE's key insight: prior GPU-initiated NVMe approaches (BaM) hold locks on
NVMe submission queues, causing deadlock when all queue slots are occupied.
AGILE uses a *service thread* that polls the completion queue and releases
entries, so submitters never stall waiting for queue space.

This module mirrors that design with Python's concurrent.futures:
  - ThreadPoolExecutor acts as the AGILE submission queue
  - Future.done() polling replaces the NVMe CQ poll
  - LRU buffer cache replaces AGILE's HBM software cache

Provides
--------
  AgileIOConfig       — configuration parameters
  AgileIOManager      — prefetch + get interface with in-memory LRU
  AgileIOStats        — I/O accounting
"""

from __future__ import annotations

import io
import threading
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

__all__ = [
    "AgileIOConfig",
    "AgileIOManager",
    "AgileIOStats",
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class AgileIOConfig:
    """Configuration for AgileIOManager.

    Parameters
    ----------
    n_worker_threads:
        Number of background I/O threads (analogous to AGILE's async NVMe
        request depth).  Default 4 — enough to pipeline multiple block reads
        concurrently while the model computes.
    cache_size_mb:
        Maximum total bytes kept in the in-memory LRU read buffer cache.
        Default 256 MB.  Increase for warm-tier KV blocks that are repeatedly
        accessed.
    prefetch_ahead:
        Number of blocks to prefetch ahead at each ``prefetch()`` call.
        Default 2.
    """

    n_worker_threads: int   = 4
    cache_size_mb:    int   = 256
    prefetch_ahead:   int   = 2

    def __post_init__(self) -> None:
        if self.n_worker_threads < 1:
            raise ValueError(
                f"n_worker_threads must be >= 1, got {self.n_worker_threads}"
            )
        if self.cache_size_mb < 1:
            raise ValueError(
                f"cache_size_mb must be >= 1, got {self.cache_size_mb}"
            )
        if self.prefetch_ahead < 0:
            raise ValueError(
                f"prefetch_ahead must be >= 0, got {self.prefetch_ahead}"
            )

    @property
    def cache_size_bytes(self) -> int:
        return self.cache_size_mb * 1024 * 1024


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

@dataclass
class AgileIOStats:
    """I/O accounting for AgileIOManager.

    Attributes
    ----------
    reads_total:   Total ``get()`` calls.
    reads_hit:     Calls satisfied from the in-memory LRU cache.
    reads_miss:    Calls that required a disk read.
    prefetches:    Total ``prefetch()`` calls enqueued.
    bytes_read:    Total bytes read from disk.
    """

    reads_total:  int = 0
    reads_hit:    int = 0
    reads_miss:   int = 0
    prefetches:   int = 0
    bytes_read:   int = 0

    @property
    def hit_rate(self) -> float:
        return self.reads_hit / self.reads_total if self.reads_total else 0.0

    def reset(self) -> None:
        self.reads_total = 0
        self.reads_hit   = 0
        self.reads_miss  = 0
        self.prefetches  = 0
        self.bytes_read  = 0


# ---------------------------------------------------------------------------
# AgileIOManager
# ---------------------------------------------------------------------------

class AgileIOManager:
    """Background-thread async I/O manager with in-memory LRU cache.

    Mimics the AGILE lock-free async NVMe model:
      - ``prefetch(path)`` enqueues a background read (non-blocking).
      - ``get(path)`` returns the bytes from cache or blocks for at most one
        synchronous read if the prefetch has not completed yet.
      - The internal LRU cache prevents re-reading recently-used blocks.

    Parameters
    ----------
    config:
        AgileIOConfig instance.  Defaults to :class:`AgileIOConfig` defaults.
    """

    def __init__(self, config: AgileIOConfig | None = None) -> None:
        self._config = config or AgileIOConfig()
        self._pool   = ThreadPoolExecutor(
            max_workers=self._config.n_worker_threads,
            thread_name_prefix="agile_io",
        )
        self._lock    = threading.Lock()
        self._futures: dict[str, Future[bytes]] = {}   # in-flight reads
        # LRU cache: path → bytes  (ordered oldest → newest)
        self._cache: OrderedDict[str, bytes] = OrderedDict()
        self._cache_bytes = 0
        self._stats = AgileIOStats()

    # ── Public API ─────────────────────────────────────────────────────────────

    @property
    def config(self) -> AgileIOConfig:
        return self._config

    @property
    def stats(self) -> AgileIOStats:
        return AgileIOStats(
            reads_total=self._stats.reads_total,
            reads_hit=self._stats.reads_hit,
            reads_miss=self._stats.reads_miss,
            prefetches=self._stats.prefetches,
            bytes_read=self._stats.bytes_read,
        )

    def prefetch(self, path: str | Path) -> None:
        """Enqueue an asynchronous background read for *path*.

        Non-blocking.  If *path* is already in cache or already being fetched,
        this is a no-op.

        Parameters
        ----------
        path:
            Absolute or relative file path to pre-load.
        """
        key = str(path)
        with self._lock:
            if key in self._cache:
                return  # already cached
            if key in self._futures and not self._futures[key].done():
                return  # already in flight
            fut = self._pool.submit(self._read_file, key)
            self._futures[key] = fut
            self._stats.prefetches += 1

    def get(self, path: str | Path) -> bytes:
        """Return the file contents of *path*.

        If a prefetch is in flight, blocks until it completes.
        If not in cache and no prefetch, initiates a synchronous read.

        Parameters
        ----------
        path:
            Absolute or relative file path to load.

        Returns
        -------
        bytes — raw file contents.

        Raises
        ------
        FileNotFoundError if the path does not exist.
        """
        key = str(path)
        with self._lock:
            self._stats.reads_total += 1
            if key in self._cache:
                self._stats.reads_hit += 1
                self._cache.move_to_end(key)
                return self._cache[key]

        # Not in cache — check in-flight future
        fut: Future[bytes] | None = None
        with self._lock:
            fut = self._futures.get(key)

        if fut is not None:
            data = fut.result()  # blocks until done
        else:
            self._stats.reads_miss += 1
            data = self._read_file(key)

        self._insert_cache(key, data)
        return data

    def get_npy(self, path: str | Path) -> np.ndarray:
        """Return a numpy array loaded from *path* (a ``.npy`` file).

        Calls :meth:`get` internally; the deserialization is zero-copy when
        the array can be interpreted directly from the bytes buffer.

        Parameters
        ----------
        path:
            Path to a ``.npy`` file.

        Returns
        -------
        np.ndarray loaded from the file.
        """
        raw = self.get(path)
        return np.load(io.BytesIO(raw), allow_pickle=False)

    def prefetch_sequence(
        self,
        paths: list[str | Path],
        start_idx: int = 0,
    ) -> None:
        """Prefetch the next ``config.prefetch_ahead`` paths starting at *start_idx*.

        Convenience helper for layer-streaming inference: call
        ``prefetch_sequence(layer_paths, next_layer_idx)`` while executing
        the current layer, and the next N layers will be in cache when needed.

        Parameters
        ----------
        paths:
            List of all layer file paths in execution order.
        start_idx:
            First index in *paths* to (possibly) prefetch.
        """
        end = min(start_idx + self._config.prefetch_ahead, len(paths))
        for i in range(start_idx, end):
            self.prefetch(paths[i])

    def evict(self, path: str | Path) -> bool:
        """Remove *path* from the LRU cache if present.

        Parameters
        ----------
        path:
            File path to evict.

        Returns
        -------
        bool — True if an entry was removed.
        """
        key = str(path)
        with self._lock:
            if key not in self._cache:
                return False
            data = self._cache.pop(key)
            self._cache_bytes -= len(data)
            return True

    def cache_info(self) -> dict:
        """Return a snapshot of current cache state.

        Returns
        -------
        dict with keys ``entries``, ``bytes_used``, ``bytes_limit``,
        ``hit_rate``.
        """
        with self._lock:
            return {
                "entries":     len(self._cache),
                "bytes_used":  self._cache_bytes,
                "bytes_limit": self._config.cache_size_bytes,
                "hit_rate":    self._stats.hit_rate,
            }

    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the background thread pool.

        Parameters
        ----------
        wait:
            If True (default), block until all in-flight reads complete.
        """
        self._pool.shutdown(wait=wait)

    # ── Private ────────────────────────────────────────────────────────────────

    @staticmethod
    def _read_file(path: str) -> bytes:
        """Read file at *path* and return raw bytes."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"AgileIOManager: file not found: {path}")
        return p.read_bytes()

    def _insert_cache(self, key: str, data: bytes) -> None:
        """Insert *data* into the LRU cache under *key*, evicting if needed."""
        size = len(data)
        limit = self._config.cache_size_bytes

        with self._lock:
            # Evict LRU entries until we have room
            while self._cache and self._cache_bytes + size > limit:
                _old_key, _old_val = self._cache.popitem(last=False)
                self._cache_bytes -= len(_old_val)

            if size <= limit:
                self._cache[key] = data
                self._cache.move_to_end(key)
                self._cache_bytes += size
                self._stats.bytes_read += size

            # Clean up the completed future
            self._futures.pop(key, None)

    def __repr__(self) -> str:
        return (
            f"AgileIOManager(workers={self._config.n_worker_threads}, "
            f"cache_mb={self._config.cache_size_mb}, "
            f"entries={len(self._cache)})"
        )
