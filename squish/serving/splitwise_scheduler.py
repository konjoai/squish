"""squish/serving/splitwise_scheduler.py

SplitwiseScheduler — Prefill/Decode Phase Disaggregation
(Patel et al., ISCA 2024 / arXiv:2311.18677).

Reference
---------
"Splitwise: Efficient Generative LLM Inference Using Phase Splitting." Patel
et al., ISCA 2024 (arXiv:2311.18677).

Design
------
Splitwise routes each request through two independent worker pools:

* **Prefill pool** (`prefill_workers` slots) — compute-bound phase.
* **Decode pool** (`decode_workers` slots) — memory-bandwidth-bound phase.

A request lifecycle:

1. ``submit(request)`` → added to the *prefill queue*.
2. ``schedule_prefill()`` → returns up to ``max_prefill_batch`` requests from
   the prefill queue to run on the prefill pool.  Marks them ``prefilling``.
3. ``complete_prefill(request_id)`` → moves the request to the *decode queue*.
4. ``schedule_decode()`` → returns up to ``max_decode_batch`` requests from
   the decode queue to run on the decode pool.  Marks them ``decoding``.
5. ``complete_decode(request_id)`` → marks the request ``done``.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional

__all__ = [
    "SplitwiseConfig",
    "SplitwiseRequest",
    "SplitwiseScheduler",
]

# ── Config ────────────────────────────────────────────────────────────────────


@dataclass
class SplitwiseConfig:
    """Configuration for :class:`SplitwiseScheduler`.

    Attributes:
        prefill_workers: Logical prefill worker concurrency slots.
        decode_workers: Logical decode worker concurrency slots.
        max_prefill_batch: Max requests dispatched per ``schedule_prefill`` call.
        max_decode_batch: Max requests dispatched per ``schedule_decode`` call.
    """

    prefill_workers: int = 2
    decode_workers: int = 4
    max_prefill_batch: int = 8
    max_decode_batch: int = 32

    def __post_init__(self) -> None:
        if self.prefill_workers < 1:
            raise ValueError(f"prefill_workers must be ≥ 1; got {self.prefill_workers}")
        if self.decode_workers < 1:
            raise ValueError(f"decode_workers must be ≥ 1; got {self.decode_workers}")
        if self.max_prefill_batch < 1:
            raise ValueError(f"max_prefill_batch must be ≥ 1; got {self.max_prefill_batch}")
        if self.max_decode_batch < 1:
            raise ValueError(f"max_decode_batch must be ≥ 1; got {self.max_decode_batch}")


# ── Request State ─────────────────────────────────────────────────────────────


class _ReqState(Enum):
    PENDING_PREFILL = auto()
    PREFILLING = auto()
    PENDING_DECODE = auto()
    DECODING = auto()
    DONE = auto()


@dataclass
class SplitwiseRequest:
    """A single inference request tracked by :class:`SplitwiseScheduler`.

    Attributes:
        request_id: Unique string identifier.
        prompt_tokens: Number of prompt tokens (prefill cost).
        max_new_tokens: Maximum tokens to generate in the decode phase.
    """

    request_id: str
    prompt_tokens: int
    max_new_tokens: int = 256
    _state: _ReqState = field(default=_ReqState.PENDING_PREFILL, repr=False, init=False)

    @property
    def state(self) -> str:
        return self._state.name.lower()

    @property
    def is_done(self) -> bool:
        return self._state == _ReqState.DONE


# ── SplitwiseScheduler ────────────────────────────────────────────────────────


class SplitwiseScheduler:
    """Disaggregated prefill–decode scheduler.

    Example::

        cfg = SplitwiseConfig(prefill_workers=2, decode_workers=4)
        sched = SplitwiseScheduler(cfg)
        req = SplitwiseRequest("req-1", prompt_tokens=128)
        sched.submit(req)
        batch = sched.schedule_prefill()   # [req]
        sched.complete_prefill("req-1")
        decode_batch = sched.schedule_decode()   # [req]
        sched.complete_decode("req-1")
    """

    def __init__(self, config: Optional[SplitwiseConfig] = None) -> None:
        self.config = config or SplitwiseConfig()
        self._requests: Dict[str, SplitwiseRequest] = {}
        self._prefill_queue: "OrderedDict[str, SplitwiseRequest]" = OrderedDict()
        self._decode_queue: "OrderedDict[str, SplitwiseRequest]" = OrderedDict()
        self._n_completed = 0

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def submit(self, request: SplitwiseRequest) -> None:
        """Enqueue a new request for prefill."""
        if request.request_id in self._requests:
            raise ValueError(f"Duplicate request_id: {request.request_id!r}")
        request._state = _ReqState.PENDING_PREFILL
        self._requests[request.request_id] = request
        self._prefill_queue[request.request_id] = request

    def schedule_prefill(self) -> List[SplitwiseRequest]:
        """Return up to ``max_prefill_batch`` requests for the prefill pool."""
        cfg = self.config
        limit = min(cfg.max_prefill_batch, cfg.prefill_workers)
        # Count currently prefilling.
        in_flight = sum(
            1 for r in self._requests.values() if r._state == _ReqState.PREFILLING
        )
        available_slots = max(0, limit - in_flight)
        batch: List[SplitwiseRequest] = []
        for req_id in list(self._prefill_queue.keys()):
            if len(batch) >= available_slots:
                break
            req = self._prefill_queue.pop(req_id)
            req._state = _ReqState.PREFILLING
            batch.append(req)
        return batch

    def complete_prefill(self, request_id: str) -> None:
        """Mark a request's prefill as done; move to decode queue."""
        req = self._requests[request_id]
        if req._state != _ReqState.PREFILLING:
            raise RuntimeError(
                f"Request {request_id!r} is not in PREFILLING state (state={req.state})"
            )
        req._state = _ReqState.PENDING_DECODE
        self._decode_queue[request_id] = req

    def schedule_decode(self) -> List[SplitwiseRequest]:
        """Return up to ``max_decode_batch`` requests for the decode pool."""
        cfg = self.config
        limit = min(cfg.max_decode_batch, cfg.decode_workers)
        in_flight = sum(
            1 for r in self._requests.values() if r._state == _ReqState.DECODING
        )
        available_slots = max(0, limit - in_flight)
        batch: List[SplitwiseRequest] = []
        for req_id in list(self._decode_queue.keys()):
            if len(batch) >= available_slots:
                break
            req = self._decode_queue.pop(req_id)
            req._state = _ReqState.DECODING
            batch.append(req)
        return batch

    def complete_decode(self, request_id: str) -> None:
        """Mark a request as fully completed."""
        req = self._requests[request_id]
        if req._state != _ReqState.DECODING:
            raise RuntimeError(
                f"Request {request_id!r} is not in DECODING state (state={req.state})"
            )
        req._state = _ReqState.DONE
        self._n_completed += 1

    # ── Stats ─────────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        """Return current scheduler statistics."""
        states: Dict[str, int] = {}
        for r in self._requests.values():
            states[r.state] = states.get(r.state, 0) + 1
        return {
            "total_submitted": len(self._requests),
            "n_completed": self._n_completed,
            "prefill_queue": len(self._prefill_queue),
            "decode_queue": len(self._decode_queue),
            "state_counts": states,
        }

    def n_inflight(self) -> int:
        """Number of requests in PREFILLING or DECODING state."""
        return sum(
            1
            for r in self._requests.values()
            if r._state in (_ReqState.PREFILLING, _ReqState.DECODING)
        )

    def __repr__(self) -> str:
        cfg = self.config
        return (
            f"SplitwiseScheduler(prefill_workers={cfg.prefill_workers}, "
            f"decode_workers={cfg.decode_workers})"
        )
