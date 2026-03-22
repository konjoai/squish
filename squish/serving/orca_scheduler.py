"""squish/serving/orca_scheduler.py

OrcaScheduler — Iteration-Level Preemptive Continuous Batching.

Reference
---------
Yu et al. "Orca: A Distributed Serving System for Transformer-Based
Generative Models." OSDI 2022.

Algorithm
---------
Traditional serving batches entire requests together — a short request is
blocked until the longest one finishes.  Orca implements *iteration-level*
scheduling:

1. Every decode iteration, the scheduler selects a *batch* of requests whose
   next token can be processed.
2. Newly arrived requests are added as soon as they fit in KV memory.
3. Preemption: if a high-priority request arrives and memory is full, the
   scheduler may swap a low-priority request to CPU memory.

Key properties
--------------
* ``submit(request)`` adds a request to the waiting queue.
* ``step() → batch`` picks the next batch of token sequences to decode.
* ``complete(request_id)`` removes a request from the active set.
* NumPy-free; no ML dependencies — scheduling logic only.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Dict, List, Optional

__all__ = [
    "RequestStatus",
    "OrcaRequest",
    "OrcaSchedulerConfig",
    "OrcaScheduler",
]


class RequestStatus:
    WAITING = "waiting"
    RUNNING = "running"
    SWAPPED = "swapped"
    COMPLETED = "completed"


@dataclass
class OrcaRequest:
    """A single inference request tracked by :class:`OrcaScheduler`.

    Attributes:
        request_id: Unique identifier.
        prompt_len: Number of prompt tokens.
        max_new_tokens: Maximum tokens to generate.
        priority: Lower = higher priority (min-heap).
        tokens_generated: Counter incremented each decode step.
        status: Current request status.
    """

    request_id: int
    prompt_len: int
    max_new_tokens: int
    priority: int = 0
    tokens_generated: int = 0
    status: str = RequestStatus.WAITING

    def __lt__(self, other: "OrcaRequest") -> bool:
        return self.priority < other.priority

    @property
    def kv_slots_needed(self) -> int:
        """Current KV slot usage = prompt_len + tokens_generated."""
        return self.prompt_len + self.tokens_generated

    @property
    def is_finished(self) -> bool:
        return self.tokens_generated >= self.max_new_tokens


@dataclass
class OrcaSchedulerConfig:
    """Configuration for :class:`OrcaScheduler`.

    Attributes:
        max_batch_size: Maximum requests per iteration.
        max_kv_slots: Total KV slot capacity across all layers.
        preemption_enabled: Whether to allow swapping to make room.
    """

    max_batch_size: int = 32
    max_kv_slots: int = 16384
    preemption_enabled: bool = True


class OrcaScheduler:
    """Iteration-level preemptive continuous-batching scheduler.

    Parameters
    ----------
    config:
        OrcaSchedulerConfig.
    """

    def __init__(self, config: Optional[OrcaSchedulerConfig] = None) -> None:
        self._cfg = config or OrcaSchedulerConfig()
        self._waiting: List[OrcaRequest] = []  # min-heap by priority
        self._running: Dict[int, OrcaRequest] = {}
        self._swapped: Dict[int, OrcaRequest] = {}
        self._next_id: int = 0

    @property
    def config(self) -> OrcaSchedulerConfig:
        return self._cfg

    def submit(
        self,
        prompt_len: int,
        max_new_tokens: int,
        priority: int = 0,
    ) -> int:
        """Submit a new inference request.

        Returns
        -------
        Assigned request_id.
        """
        req = OrcaRequest(
            request_id=self._next_id,
            prompt_len=prompt_len,
            max_new_tokens=max_new_tokens,
            priority=priority,
        )
        heapq.heappush(self._waiting, req)
        self._next_id += 1
        return req.request_id

    def _kv_used(self) -> int:
        return sum(r.kv_slots_needed for r in self._running.values())

    def step(self) -> List[OrcaRequest]:
        """Run one scheduling iteration.

        1. Move waiting requests to running if capacity allows.
        2. If capacity is exhausted and preemption is enabled, swap lowest-
           priority running request to free space for a higher-priority waiter.
        3. Return the current running batch.
        """
        cfg = self._cfg

        # Try to admit waiting requests
        while self._waiting:
            if len(self._running) >= cfg.max_batch_size:
                break
            candidate = self._waiting[0]
            slots_after = self._kv_used() + candidate.kv_slots_needed
            if slots_after > cfg.max_kv_slots:
                if cfg.preemption_enabled:
                    self._try_preempt(candidate)
                break
            heapq.heappop(self._waiting)
            candidate.status = RequestStatus.RUNNING
            self._running[candidate.request_id] = candidate

        # Restore any swapped requests if there's room
        for rid in list(self._swapped.keys()):
            req = self._swapped[rid]
            if (
                len(self._running) < cfg.max_batch_size
                and self._kv_used() + req.kv_slots_needed <= cfg.max_kv_slots
            ):
                del self._swapped[rid]
                req.status = RequestStatus.RUNNING
                self._running[rid] = req

        return list(self._running.values())

    def advance(self, request_id: int) -> bool:
        """Increment token counter for a running request.

        Returns True if the request has finished.
        """
        req = self._running.get(request_id)
        if req is None:
            return False
        req.tokens_generated += 1
        if req.is_finished:
            req.status = RequestStatus.COMPLETED
            del self._running[request_id]
            return True
        return False

    def _try_preempt(self, incoming: OrcaRequest) -> None:
        """Swap the lowest-priority running request to free KV slots."""
        if not self._running:
            return
        victim = max(self._running.values(), key=lambda r: r.priority)
        if victim.priority <= incoming.priority:
            return  # Incoming is not higher priority
        del self._running[victim.request_id]
        victim.status = RequestStatus.SWAPPED
        self._swapped[victim.request_id] = victim

    @property
    def waiting_count(self) -> int:
        return len(self._waiting)

    @property
    def running_count(self) -> int:
        return len(self._running)

    @property
    def swapped_count(self) -> int:
        return len(self._swapped)
