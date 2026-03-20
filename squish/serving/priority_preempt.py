"""
squish/serving/priority_preempt.py

PriorityPreemptScheduler: SLO-Aware Preemption + Chunked-Prefill Scheduling.

Reference
---------
Agrawal et al. "Sarathi-Serve: Efficient LLM Serving by Chunked-Prefill and
Decode Pull Scheduling." OSDI 2024.

Algorithm
---------
PriorityPreemptScheduler manages a queue of in-flight LLM requests with two
scheduling innovations:

  1. **Chunked prefill** — long prompts are split into fixed-size chunks
     (``chunk_size`` tokens).  Each scheduler tick fills exactly one chunk
     worth of prefill compute, preventing a single long prompt from blocking
     the entire decode pipeline.

  2. **Priority-based preemption** — active decode requests have a *priority
     score* computed from age (time spent in the system) and an explicit
     priority tier.  When VRAM overflows, the lowest-priority active request
     is *preempted* (its KV cache is evicted) and re-queued for later
     re-prefill, rather than dropped.

Priority score formula (higher = serve sooner):
  ``score = priority_tier * tier_weight + age_ms * age_weight``

Key properties
--------------
* ``chunk_size`` — prefill chunk size in tokens (default 256).
* ``max_active`` — maximum concurrently active requests (default 8).
* ``tier_weight`` / ``age_weight`` — priority formula weights.
* Pure Python; no external dependencies.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class SchedulerConfig:
    """Configuration for PriorityPreemptScheduler."""

    chunk_size: int = 256
    """Tokens per prefill chunk."""

    max_active: int = 8
    """Maximum concurrently active decode requests."""

    tier_weight: float = 1000.0
    """Score contribution per unit of priority tier."""

    age_weight: float = 1.0
    """Score contribution per millisecond of age in the system."""

    def __post_init__(self) -> None:
        if self.chunk_size < 1:
            raise ValueError("chunk_size must be >= 1")
        if self.max_active < 1:
            raise ValueError("max_active must be >= 1")
        if self.tier_weight < 0.0:
            raise ValueError("tier_weight must be >= 0")
        if self.age_weight < 0.0:
            raise ValueError("age_weight must be >= 0")


@dataclass
class RequestEntry:
    """A single request tracked by the scheduler."""

    request_id: str
    prompt_tokens: int
    priority_tier: int = 0
    enqueue_time: float = field(default_factory=time.monotonic)
    prefill_done_tokens: int = 0
    decode_tokens: int = 0
    state: str = "pending"  # "pending" | "prefilling" | "decoding" | "done" | "preempted"

    @property
    def prefill_remaining(self) -> int:
        return max(0, self.prompt_tokens - self.prefill_done_tokens)

    @property
    def age_ms(self) -> float:
        return (time.monotonic() - self.enqueue_time) * 1e3


@dataclass
class SchedulerStats:
    """Runtime counters for PriorityPreemptScheduler."""

    enqueued: int = 0
    prefill_chunks: int = 0
    decode_steps: int = 0
    preemptions: int = 0
    completed: int = 0


class PriorityPreemptScheduler:
    """SLO-aware chunked-prefill + preemptive decode scheduler.

    Usage
    -----
    ::

        sched = PriorityPreemptScheduler()
        sched.enqueue("req-1", prompt_tokens=512, priority_tier=2)
        sched.enqueue("req-2", prompt_tokens=128, priority_tier=0)

        while not sched.all_done():
            work = sched.tick()
            # work is a list of (request_id, action, tokens) tuples
    """

    def __init__(self, config: Optional[SchedulerConfig] = None) -> None:
        self.config = config or SchedulerConfig()
        self.stats = SchedulerStats()
        self._requests: Dict[str, RequestEntry] = {}
        self._queue: List[str] = []        # pending / preempted request ids
        self._active: List[str] = []       # currently prefilling or decoding

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def enqueue(
        self,
        request_id: str,
        prompt_tokens: int,
        priority_tier: int = 0,
    ) -> None:
        """Add a new request to the scheduler queue.

        Parameters
        ----------
        request_id:
            Unique identifier for the request.
        prompt_tokens:
            Length of the prompt in tokens.
        priority_tier:
            Integer priority tier (higher = more urgent).
        """
        entry = RequestEntry(
            request_id=request_id,
            prompt_tokens=prompt_tokens,
            priority_tier=priority_tier,
        )
        self._requests[request_id] = entry
        self._queue.append(request_id)
        self.stats.enqueued += 1

    def tick(self) -> List[Tuple[str, str, int]]:
        """Advance the scheduler by one tick.

        Returns
        -------
        work_items:
            List of ``(request_id, action, tokens)`` tuples where
            ``action`` is one of "prefill_chunk", "decode_step", "preempt".
        """
        work: List[Tuple[str, str, int]] = []
        cfg = self.config

        # Admit pending requests up to max_active
        self._admit_from_queue()

        for rid in list(self._active):
            entry = self._requests[rid]

            if entry.state == "prefilling":
                chunk = min(cfg.chunk_size, entry.prefill_remaining)
                entry.prefill_done_tokens += chunk
                work.append((rid, "prefill_chunk", chunk))
                self.stats.prefill_chunks += 1

                if entry.prefill_remaining == 0:
                    entry.state = "decoding"

            elif entry.state == "decoding":
                entry.decode_tokens += 1
                work.append((rid, "decode_step", 1))
                self.stats.decode_steps += 1

        # Enforce max_active via priority-based preemption if over capacity
        while len(self._active) > cfg.max_active:
            victim_id = self._lowest_priority_active()
            if victim_id is None:
                break
            self._preempt(victim_id)
            work.append((victim_id, "preempt", 0))

        return work

    def complete(self, request_id: str) -> None:
        """Mark a request as completed and remove it from active set."""
        entry = self._requests.get(request_id)
        if entry is not None:
            entry.state = "done"
            self.stats.completed += 1
        if request_id in self._active:
            self._active.remove(request_id)

    def all_done(self) -> bool:
        """True when all enqueued requests have been completed."""
        return all(
            e.state == "done"
            for e in self._requests.values()
        )

    def queue_depth(self) -> int:
        """Number of requests waiting to be admitted."""
        return len(self._queue)

    def active_count(self) -> int:
        """Number of currently active requests."""
        return len(self._active)

    def reset(self) -> None:
        """Clear all state."""
        self._requests.clear()
        self._queue.clear()
        self._active.clear()
        self.stats = SchedulerStats()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _priority_score(self, rid: str) -> float:
        entry = self._requests[rid]
        cfg = self.config
        return entry.priority_tier * cfg.tier_weight + entry.age_ms * cfg.age_weight

    def _admit_from_queue(self) -> None:
        """Move pending requests from the queue into the active set."""
        cfg = self.config
        # Sort queue by priority score descending
        self._queue.sort(key=lambda r: -self._priority_score(r))
        while self._queue and len(self._active) < cfg.max_active:
            rid = self._queue.pop(0)
            entry = self._requests[rid]
            if entry.state in ("pending", "preempted"):
                entry.state = "prefilling"
                self._active.append(rid)

    def _lowest_priority_active(self) -> Optional[str]:
        """Return the active request with the lowest priority score."""
        if not self._active:
            return None
        return min(self._active, key=self._priority_score)

    def _preempt(self, request_id: str) -> None:
        """Preempt an active request — reset its prefill and re-queue it."""
        entry = self._requests[request_id]
        entry.state = "preempted"
        entry.prefill_done_tokens = 0  # must re-prefill from scratch
        if request_id in self._active:
            self._active.remove(request_id)
        self._queue.append(request_id)
        self.stats.preemptions += 1
