"""
squish/kernels/mem_pool.py

Pre-allocated numpy buffer pool for GC-pressure elimination.

Each autoregressive decode step allocates temporary numpy arrays (logit
buffers, attention masks, sampling scratch space).  On Python 3.12 / macOS
ARM64 these allocations typically take 50–300 µs per decode step — a
non-trivial fraction of total step latency at small batch sizes.

This module provides a thread-safe pool of pre-allocated numpy buffers.
Callers acquire a buffer, use it, and return it.  The pool re-uses buffers
across calls, eliminating malloc()/free() and GC pressure during the hot
decode loop.

Design
------
* Fixed-size pre-allocated buffers: ``pool_size`` arrays of shape
  ``max_shape`` and the configured dtype.
* ``acquire(shape)`` returns a buffer large enough to provide a view of
  ``shape`` (zero-copies via reshaping if the buffer is large enough).
* ``release(buf_id)`` returns the buffer to the available pool.
* Thread-safe via ``threading.Lock``.
* Context manager ``borrow(shape)`` for RAII-style usage.
* Statistics: ``hits``, ``misses``, ``active_count``.

Usage
-----
::

    pool = NumpyMemPool(PoolConfig(pool_size=64, max_shape=(4096,)))

    # Explicit acquire/release:
    buf_id, buf = pool.acquire((2048,))
    buf[:] = 0
    # … use buf …
    pool.release(buf_id)

    # Context manager:
    with pool.borrow((2048,)) as buf:
        buf[:] = compute_logits()
"""

from __future__ import annotations

import contextlib
import threading
from dataclasses import dataclass, field
from typing import Generator, List, Optional, Set, Tuple

import numpy as np


@dataclass
class PoolConfig:
    """Configuration for the numpy buffer pool.

    Parameters
    ----------
    pool_size:
        Total number of pre-allocated buffers.
    max_shape:
        Shape of each buffer.  ``acquire`` requests of any shape whose total
        element count ≤ ``max_elements`` will be served from pool.
    dtype:
        NumPy dtype of all pooled buffers (e.g. np.float32).
    overflow_policy:
        ``"allocate"`` — on pool exhaustion, allocate a fresh buffer (not
        returned to pool).  ``"raise"`` — raise RuntimeError if pool is empty.
    """

    pool_size: int = 32
    max_shape: Tuple[int, ...] = (2048, 4096)
    dtype: type = np.float32
    overflow_policy: str = "allocate"

    def __post_init__(self) -> None:
        if self.pool_size < 1:
            raise ValueError("pool_size must be >= 1")
        if not self.max_shape:
            raise ValueError("max_shape must be a non-empty tuple")
        if any(d < 1 for d in self.max_shape):
            raise ValueError("All max_shape dimensions must be >= 1")
        if self.overflow_policy not in ("allocate", "raise"):
            raise ValueError(
                f"overflow_policy must be 'allocate' or 'raise', "
                f"got {self.overflow_policy!r}"
            )

    @property
    def max_elements(self) -> int:
        """Total number of elements in each pooled buffer."""
        result = 1
        for d in self.max_shape:
            result *= d
        return result


class _PoolBuffer:
    """Internal wrapper around a pre-allocated numpy array."""

    __slots__ = ("buf_id", "data", "available")

    def __init__(self, buf_id: int, data: np.ndarray) -> None:
        self.buf_id = buf_id
        self.data = data
        self.available = True


class NumpyMemPool:
    """Thread-safe pre-allocated numpy buffer pool.

    All buffers have identical dtype and total element count.  Views of the
    requested shape are returned, avoiding extra allocations.
    """

    def __init__(self, config: Optional[PoolConfig] = None) -> None:
        self.config = config or PoolConfig()
        self._lock = threading.Lock()
        self._buffers: List[_PoolBuffer] = [
            _PoolBuffer(
                i,
                np.empty(self.config.max_elements, dtype=self.config.dtype),
            )
            for i in range(self.config.pool_size)
        ]
        self._free_ids: List[int] = list(range(self.config.pool_size))
        self._hits: int = 0
        self._misses: int = 0

    # ------------------------------------------------------------------
    # Pool operations
    # ------------------------------------------------------------------

    def acquire(self, shape: Tuple[int, ...]) -> Tuple[int, np.ndarray]:
        """Acquire a buffer that can be viewed as *shape*.

        Parameters
        ----------
        shape:
            Desired shape.  The returned array will have exactly this shape.

        Returns
        -------
        (buf_id, array):
            ``buf_id`` must be passed to :meth:`release` when done.
            ``array`` is a writable numpy array of the requested shape.

        Raises
        ------
        RuntimeError:
            If the pool is empty and ``overflow_policy == "raise"``.
        ValueError:
            If the requested element count exceeds ``max_elements``.
        """
        n_elements = 1
        for d in shape:
            n_elements *= d

        if n_elements > self.config.max_elements:
            raise ValueError(
                f"Requested shape {shape} ({n_elements} elements) exceeds "
                f"pool max_elements={self.config.max_elements}"
            )

        with self._lock:
            if self._free_ids:
                buf_id = self._free_ids.pop()
                self._buffers[buf_id].available = False
                self._hits += 1
                return buf_id, self._buffers[buf_id].data[:n_elements].reshape(shape)

        # Pool exhausted
        self._misses += 1
        if self.config.overflow_policy == "raise":
            raise RuntimeError(
                f"NumpyMemPool exhausted: all {self.config.pool_size} buffers "
                "are in use.  Increase pool_size or release buffers promptly."
            )
        # Allocate a temporary buffer outside the pool (buf_id = -1)
        return -1, np.empty(shape, dtype=self.config.dtype)

    def release(self, buf_id: int) -> None:
        """Return a buffer to the pool.

        Parameters
        ----------
        buf_id:
            The ``buf_id`` returned by :meth:`acquire`.  If ``-1`` (overflow
            buffer), this is a no-op.
        """
        if buf_id < 0:
            return  # overflow buffer, not pooled
        with self._lock:
            buf = self._buffers[buf_id]
            if buf.available:
                raise RuntimeError(
                    f"NumpyMemPool: buffer {buf_id} released twice."
                )
            buf.available = True
            self._free_ids.append(buf_id)

    @contextlib.contextmanager
    def borrow(self, shape: Tuple[int, ...]) -> Generator[np.ndarray, None, None]:
        """Context manager that acquires and auto-releases a buffer.

        ::

            with pool.borrow((1024,)) as buf:
                buf[:] = 0.0
                # use buf
            # buf is released here
        """
        buf_id, arr = self.acquire(shape)
        try:
            yield arr
        finally:
            self.release(buf_id)

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    @property
    def hits(self) -> int:
        """Number of successful pool acquisitions (no extra allocation)."""
        return self._hits

    @property
    def misses(self) -> int:
        """Number of overflow acquisitions (pool was empty)."""
        return self._misses

    @property
    def active_count(self) -> int:
        """Number of buffers currently checked out from the pool."""
        with self._lock:
            return self.config.pool_size - len(self._free_ids)

    @property
    def capacity(self) -> int:
        """Total number of pooled buffers."""
        return self.config.pool_size

    @property
    def free_count(self) -> int:
        """Number of buffers available for immediate acquisition."""
        with self._lock:
            return len(self._free_ids)

    def reset_stats(self) -> None:
        """Reset hit/miss counters."""
        with self._lock:
            self._hits = 0
            self._misses = 0
