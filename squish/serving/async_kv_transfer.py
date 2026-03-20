"""AsyncKVTransfer — non-blocking KV block migration for disaggregated serving.

Implements the LMCache-style KV block migration strategy (Gao et al.,
MLSys 2025 / arXiv:2401.02669).

In **prefill–decode disaggregation** (PD-disagg), a separate high-throughput
prefill worker fills the KV cache for a prompt, then the decode worker takes
over.  Without asynchronous transfer the decode worker stalls waiting for PCIe
/ NVLink copies to complete.

AsyncKVTransfer solves this by:

1. Enqueueing KV block transfers as soon as the prefill worker emits them.
2. Dispatching copies in a background "transfer coroutine" (simulated here
   with immediate execution, since we have no real GPU/PCIe).
3. Exposing a polling / await interface so the decode worker can check
   readiness and retrieve blocks without blocking its hot path.

This module is a pure-Python / NumPy implementation that precisely models the
asynchronous lifecycle.  In production it is replaced by CUDA stream copies or
``asyncio``-wrapped DMA operations.

Reference:
    Gao et al., "LMCache: Enabling Efficient KV Cache Reuse for LLM Serving
    with Disaggregated Architecture", MLSys 2025 (arXiv:2401.02669).
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

import numpy as np

__all__ = [
    "TransferStatus",
    "KVBlock",
    "TransferHandle",
    "AsyncKVTransferConfig",
    "AsyncKVTransfer",
]

# ── Status enum ───────────────────────────────────────────────────────────────


class TransferStatus(Enum):
    """Lifecycle state of a KV block transfer."""

    QUEUED = auto()       # Enqueued, not yet started
    IN_FLIGHT = auto()    # Transfer in progress
    COMPLETE = auto()     # Data available on destination
    FAILED = auto()       # Transfer error


# ── KV block ──────────────────────────────────────────────────────────────────


@dataclass
class KVBlock:
    """A single KV cache block (page).

    Attributes:
        block_id: Unique integer identifier.
        layer_id: Transformer layer index.
        tokens: Sequence of token IDs this block covers.
        keys: ``(n_tokens, n_heads, head_dim)`` float32 key tensor.
        values: ``(n_tokens, n_heads, head_dim)`` float32 value tensor.
    """

    block_id: int
    layer_id: int
    tokens: list[int]
    keys: np.ndarray
    values: np.ndarray

    def byte_size(self) -> int:
        """Return total bytes occupied by keys + values."""
        return int(self.keys.nbytes + self.values.nbytes)


# ── Transfer handle ───────────────────────────────────────────────────────────


class TransferHandle:
    """Handle returned by :meth:`AsyncKVTransfer.enqueue`.

    Callers use this to poll or wait for transfer completion.

    Args:
        handle_id: Unique integer handle ID.
        block: The :class:`KVBlock` being transferred.
    """

    def __init__(self, handle_id: int, block: KVBlock) -> None:
        self.handle_id: int = handle_id
        self.block: KVBlock = block
        self.status: TransferStatus = TransferStatus.QUEUED
        self._done_event: threading.Event = threading.Event()

    def is_ready(self) -> bool:
        """Return True if the transfer has completed successfully."""
        return self.status == TransferStatus.COMPLETE

    def wait(self, timeout: Optional[float] = None) -> bool:
        """Block until the transfer completes or times out.

        Args:
            timeout: Maximum seconds to wait; None waits indefinitely.

        Returns:
            True if completed within timeout, False if timed out.
        """
        return self._done_event.wait(timeout=timeout)

    def _mark_complete(self) -> None:
        self.status = TransferStatus.COMPLETE
        self._done_event.set()

    def _mark_failed(self) -> None:
        self.status = TransferStatus.FAILED
        self._done_event.set()

    def __repr__(self) -> str:
        return f"TransferHandle(id={self.handle_id}, status={self.status.name})"


# ── Config ────────────────────────────────────────────────────────────────────


@dataclass
class AsyncKVTransferConfig:
    """Configuration for AsyncKVTransfer.

    Attributes:
        max_inflight: Maximum simultaneous in-flight transfers.
        simulated_latency_ms: Artificial latency added per transfer in the
            simulation path (0 = synchronous, > 0 = async).
        max_queue_depth: Maximum pending transfers before backpressure.
        bandwidth_gbps: Simulated transfer bandwidth in GB/s (for latency
            estimation in the telemetry only).
    """

    max_inflight: int = 4
    simulated_latency_ms: float = 0.0
    max_queue_depth: int = 64
    bandwidth_gbps: float = 16.0

    def __post_init__(self) -> None:
        if self.max_inflight < 1:
            raise ValueError(
                f"max_inflight must be ≥ 1; got {self.max_inflight}"
            )
        if self.max_queue_depth < 1:
            raise ValueError(
                f"max_queue_depth must be ≥ 1; got {self.max_queue_depth}"
            )
        if self.bandwidth_gbps <= 0:
            raise ValueError(
                f"bandwidth_gbps must be positive; got {self.bandwidth_gbps}"
            )


# ── Main class ────────────────────────────────────────────────────────────────


class AsyncKVTransfer:
    """Non-blocking KV block migration between prefill and decode workers.

    Thread-safe.  A background thread drains the transfer queue.  When
    ``simulated_latency_ms=0`` transfers complete before :meth:`enqueue`
    returns (synchronous simulation); otherwise they complete asynchronously.

    Example::

        cfg      = AsyncKVTransferConfig(max_inflight=2)
        transfer = AsyncKVTransfer(cfg)
        transfer.start()

        block  = KVBlock(0, 0, [1,2,3], keys, values)
        handle = transfer.enqueue(block)
        handle.wait(timeout=1.0)
        assert handle.is_ready()

        transfer.stop()

    Args:
        config: :class:`AsyncKVTransferConfig` (optional).
    """

    def __init__(self, config: Optional[AsyncKVTransferConfig] = None) -> None:
        self.config: AsyncKVTransferConfig = config or AsyncKVTransferConfig()
        self._queue: list[TransferHandle] = []
        self._lock: threading.Lock = threading.Lock()
        self._worker: Optional[threading.Thread] = None
        self._running: bool = False
        self._next_handle_id: int = 0
        self._n_completed: int = 0
        self._n_failed: int = 0
        self._total_bytes: int = 0

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the background transfer worker thread."""
        if self._running:
            return
        self._running = True
        self._worker = threading.Thread(
            target=self._drain_loop, name="AsyncKVTransfer", daemon=True
        )
        self._worker.start()

    def stop(self, timeout: float = 5.0) -> None:
        """Stop the background worker and wait for it to finish.

        Args:
            timeout: Maximum seconds to wait for clean shutdown.
        """
        self._running = False
        if self._worker is not None:
            self._worker.join(timeout=timeout)
            self._worker = None

    # ── Enqueue ───────────────────────────────────────────────────────────────

    def enqueue(self, block: KVBlock) -> TransferHandle:
        """Enqueue a KV block for async transfer.

        If the queue is full (``max_queue_depth`` exceeded) the oldest pending
        handle is evicted and marked FAILED.

        Args:
            block: :class:`KVBlock` to transfer.

        Returns:
            :class:`TransferHandle` that can be polled or awaited.
        """
        with self._lock:
            handle_id = self._next_handle_id
            self._next_handle_id += 1
            handle = TransferHandle(handle_id, block)

            if len(self._queue) >= self.config.max_queue_depth:
                # Evict oldest
                evicted = self._queue.pop(0)
                evicted._mark_failed()
                self._n_failed += 1

            self._queue.append(handle)

        # Synchronous simulation: drain immediately when latency=0
        if self.config.simulated_latency_ms == 0.0 and not self._running:
            self._execute_transfer(handle)

        return handle

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def get_ready_blocks(self) -> list[KVBlock]:
        """Return all blocks whose transfers have completed.

        Completed handles are removed from tracking.

        Returns:
            List of :class:`KVBlock` objects ready for use.
        """
        with self._lock:
            ready = [h.block for h in list(self._queue) if h.is_ready()]
            self._queue = [h for h in self._queue if not h.is_ready()]
        return ready

    def pending_count(self) -> int:
        """Return the number of transfers not yet complete."""
        with self._lock:
            return sum(1 for h in self._queue if not h.is_ready())

    # ── Stats ─────────────────────────────────────────────────────────────────

    @property
    def n_completed(self) -> int:
        """Total completed transfers since construction."""
        return self._n_completed

    @property
    def n_failed(self) -> int:
        """Total failed/evicted transfers since construction."""
        return self._n_failed

    @property
    def total_bytes_transferred(self) -> int:
        """Total bytes transferred since construction."""
        return self._total_bytes

    # ── Background worker ─────────────────────────────────────────────────────

    def _drain_loop(self) -> None:
        """Background thread: drain the transfer queue."""
        while self._running:
            handle: Optional[TransferHandle] = None
            with self._lock:
                for h in self._queue:
                    if h.status == TransferStatus.QUEUED:
                        h.status = TransferStatus.IN_FLIGHT
                        handle = h
                        break

            if handle is not None:
                if self.config.simulated_latency_ms > 0:
                    time.sleep(self.config.simulated_latency_ms / 1000.0)
                self._execute_transfer(handle)
            else:
                time.sleep(0.0001)  # yield

    def _execute_transfer(self, handle: TransferHandle) -> None:
        """Simulate the actual DMA copy (no-op: data already in memory)."""
        try:
            handle.status = TransferStatus.IN_FLIGHT
            # Simulate bandwidth cost
            expected_s = handle.block.byte_size() / (self.config.bandwidth_gbps * 1e9)
            if expected_s > 0 and self.config.simulated_latency_ms > 0:
                time.sleep(min(expected_s, self.config.simulated_latency_ms / 1000.0))
            with self._lock:
                self._n_completed += 1
                self._total_bytes += handle.block.byte_size()
            handle._mark_complete()
        except Exception:
            with self._lock:
                self._n_failed += 1
            handle._mark_failed()

    def __repr__(self) -> str:
        return (
            f"AsyncKVTransfer("
            f"max_inflight={self.config.max_inflight}, "
            f"running={self._running}, "
            f"completed={self._n_completed})"
        )
