"""squish/serving/sarathi_scheduler.py

SarathiScheduler — Chunked-prefill scheduler that piggybacks decode tokens onto
idle prefill budget slots (Agrawal et al., OSDI 2024 / arXiv:2308.16369).

Reference
---------
"Sarathi-Serve: Efficient LLM Inference by Piggybacking Decodes on Chunked
Prefills." Agrawal et al., OSDI 2024 (arXiv:2308.16369).

Algorithm
---------
Classical serving either:
  * Processes full prefill (fills all KV slots) then decodes — TTFT is long.
  * Processes full decode batch — prefill stalls decode.

Sarathi splits prefill into fixed-size ``chunk_size`` token slices.  Each
scheduler tick has a token budget of ``chunk_size``.  Decode tokens are
``1 token × n_decoding_requests`` and fill any remaining budget.  This ensures
decode never starves while prefill completes in O(prompt_len / chunk_size) ticks.

This simulation:
* Maintains two queues: ``_prefill_queue`` (requests with remaining prompt) and
  ``_decode_queue`` (requests whose prefill is complete, awaiting decodes).
* ``schedule()`` returns a :class:`SarathiTick` with the tokens to process.
* Each request is represented by a :class:`SarathiRequest`.

Key properties
--------------
* NumPy-only simulation (no real model calls).
* ``chunk_size`` — prefill tokens per tick (default 512).
* ``max_decode_tokens`` — decode tokens per tick across all active requests.
* Thread-safe via internal lock.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional

__all__ = [
    "SarathiConfig",
    "SarathiRequest",
    "SarathiTick",
    "SarathiScheduler",
]

# ── Config ────────────────────────────────────────────────────────────────────


@dataclass
class SarathiConfig:
    """Configuration for :class:`SarathiScheduler`.

    Attributes:
        chunk_size: Maximum prefill tokens processed per scheduler tick.
        max_decode_tokens: Maximum decode tokens per tick across all active requests.
        max_batch_size: Maximum concurrent requests in the decode queue.
    """

    chunk_size: int = 512
    max_decode_tokens: int = 512
    max_batch_size: int = 16

    def __post_init__(self) -> None:
        if self.chunk_size < 1:
            raise ValueError(f"chunk_size must be ≥ 1; got {self.chunk_size}")
        if self.max_decode_tokens < 1:
            raise ValueError(f"max_decode_tokens must be ≥ 1; got {self.max_decode_tokens}")
        if self.max_batch_size < 1:
            raise ValueError(f"max_batch_size must be ≥ 1; got {self.max_batch_size}")


# ── Request ───────────────────────────────────────────────────────────────────


@dataclass
class SarathiRequest:
    """A single request in the Sarathi scheduler.

    Attributes:
        request_id: Unique string identifier.
        prompt_tokens: Total number of prompt token IDs.
        max_new_tokens: Maximum decode steps for this request.
        prefill_done: Tokens of prompt already processed.
        decode_done: Decode steps completed.
    """

    request_id: str
    prompt_tokens: int
    max_new_tokens: int = 256
    prefill_done: int = 0
    decode_done: int = 0

    @property
    def prefill_remaining(self) -> int:
        return max(0, self.prompt_tokens - self.prefill_done)

    @property
    def decode_remaining(self) -> int:
        return max(0, self.max_new_tokens - self.decode_done)

    @property
    def is_prefill_complete(self) -> bool:
        return self.prefill_remaining == 0

    @property
    def is_complete(self) -> bool:
        return self.is_prefill_complete and self.decode_remaining == 0


# ── Tick ──────────────────────────────────────────────────────────────────────


@dataclass
class SarathiTick:
    """Output of one :meth:`SarathiScheduler.schedule` call.

    Attributes:
        prefill_chunks: List of ``(request_id, chunk_token_count)`` tuples.
        decode_requests: List of ``request_id`` strings for decode step.
        total_tokens: Total tokens processed this tick (prefill + decode).
        idle: True if nothing was scheduled (all queues empty).
    """

    prefill_chunks: List[tuple]
    decode_requests: List[str]
    total_tokens: int
    idle: bool


# ── Scheduler ─────────────────────────────────────────────────────────────────


class SarathiScheduler:
    """Chunked-prefill / piggybacked-decode scheduler.

    Example::

        cfg = SarathiConfig(chunk_size=128, max_decode_tokens=64, max_batch_size=4)
        sched = SarathiScheduler(cfg)
        sched.add_request(SarathiRequest("r1", prompt_tokens=512, max_new_tokens=64))
        tick = sched.schedule()   # processes first 128 prefill tokens of r1
        tick2 = sched.schedule()  # second chunk ...
    """

    def __init__(self, config: Optional[SarathiConfig] = None) -> None:
        self.config = config or SarathiConfig()
        self._lock = threading.Lock()
        self._prefill_queue: List[SarathiRequest] = []
        self._decode_queue: List[SarathiRequest] = []
        self._completed: List[str] = []
        self._n_ticks = 0
        self._total_prefill_tokens = 0
        self._total_decode_tokens = 0

    # ── Public API ────────────────────────────────────────────────────────────

    def add_request(self, request: SarathiRequest) -> None:
        """Enqueue a new request for scheduling."""
        with self._lock:
            self._prefill_queue.append(request)

    def schedule(self) -> SarathiTick:
        """Run one scheduler tick.

        1. Process up to ``chunk_size`` prefill tokens from the head of the
           prefill queue.  Completed prefills move to the decode queue.
        2. Fill remaining token budget with decode tokens from the decode queue.

        Returns:
            :class:`SarathiTick` describing what was scheduled.
        """
        with self._lock:
            cfg = self.config
            budget = cfg.chunk_size
            prefill_chunks: List[tuple] = []
            decode_requests: List[str] = []

            # ── Prefill phase ──────────────────────────────────────────────────
            i = 0
            while i < len(self._prefill_queue) and budget > 0:
                req = self._prefill_queue[i]
                chunk = min(req.prefill_remaining, budget)
                req.prefill_done += chunk
                budget -= chunk
                prefill_chunks.append((req.request_id, chunk))
                if req.is_prefill_complete and len(self._decode_queue) < cfg.max_batch_size:
                    self._prefill_queue.pop(i)
                    self._decode_queue.append(req)
                else:
                    i += 1
                self._total_prefill_tokens += chunk

            # ── Decode phase ───────────────────────────────────────────────────
            decode_budget = min(cfg.max_decode_tokens, len(self._decode_queue))
            to_remove = []
            for req in self._decode_queue[:decode_budget]:
                if req.decode_remaining > 0:
                    req.decode_done += 1
                    decode_requests.append(req.request_id)
                    self._total_decode_tokens += 1
                    if req.is_complete:
                        to_remove.append(req)

            for req in to_remove:
                self._decode_queue.remove(req)
                self._completed.append(req.request_id)

            total = sum(c for _, c in prefill_chunks) + len(decode_requests)
            idle = total == 0
            self._n_ticks += 1

            return SarathiTick(
                prefill_chunks=prefill_chunks,
                decode_requests=decode_requests,
                total_tokens=total,
                idle=idle,
            )

    def n_inflight(self) -> int:
        """Number of requests currently in prefill or decode queues."""
        with self._lock:
            return len(self._prefill_queue) + len(self._decode_queue)

    def n_completed(self) -> int:
        """Number of requests that have finished decode."""
        with self._lock:
            return len(self._completed)

    def stats(self) -> dict:
        """Return scheduler statistics."""
        with self._lock:
            return {
                "n_ticks": self._n_ticks,
                "total_prefill_tokens": self._total_prefill_tokens,
                "total_decode_tokens": self._total_decode_tokens,
                "n_inflight": len(self._prefill_queue) + len(self._decode_queue),
                "n_completed": len(self._completed),
            }

    def __repr__(self) -> str:
        cfg = self.config
        return (
            f"SarathiScheduler(chunk_size={cfg.chunk_size}, "
            f"max_decode={cfg.max_decode_tokens}, "
            f"inflight={self.n_inflight()}, completed={self.n_completed()})"
        )
