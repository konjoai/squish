#!/usr/bin/env python3
"""
squish/multi_tenant_sched.py

MultiTenantSched — Fair per-tenant QoS scheduling for multi-tenant LLM serving.

Each tenant is assigned a fractional weight that governs its share of the server's
capacity under contention.  The scheduler implements Weighted Fair Queuing (WFQ):
every tenant maintains a virtual clock that advances by ``n_tokens_est / weight``
per dispatched request.  When selecting the next request, the scheduler picks the
head request from the tenant queue whose virtual finish time is smallest, ensuring
that high-weight tenants receive proportionally more throughput while low-weight
tenants are never fully starved.

SLO tracking compares each request's actual service latency against the tenant's
``target_latency_ms`` threshold and increments a per-tenant violation counter when
the threshold is breached.

Example usage::

    from squish.multi_tenant_sched import (
        TenantConfig, TenantRequest, TenantScheduler,
    )
    import time

    tenants = [
        TenantConfig(tenant_id="alice", weight=2.0, max_concurrent=4, target_latency_ms=200.0),
        TenantConfig(tenant_id="bob",   weight=1.0, max_concurrent=2, target_latency_ms=500.0),
    ]
    sched = TenantScheduler(tenants)

    sched.submit(TenantRequest("req-1", "alice", n_tokens_est=128, submitted_at=time.monotonic()))
    sched.submit(TenantRequest("req-2", "bob",   n_tokens_est=64,  submitted_at=time.monotonic()))

    req = sched.next_request()
    sched.complete(req.request_id, actual_latency_ms=95.0)
    print(sched.stats)
"""

from __future__ import annotations

__all__ = [
    "TenantConfig",
    "TenantRequest",
    "TenantScheduler",
    "TenantStats",
]

from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np  # noqa: F401  — imported for dtype compatibility in future extensions


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class TenantConfig:
    """Per-tenant capacity and quality-of-service parameters.

    Attributes:
        tenant_id:          Unique string identifier for this tenant.
        weight:             Fair-share weight used for Weighted Fair Queuing.
                            A tenant with weight 2.0 receives twice the server
                            throughput of a tenant with weight 1.0 under
                            contention.  Must be strictly positive.
        max_concurrent:     Maximum number of requests from this tenant that
                            may be in-flight simultaneously.  Must be >= 1.
        target_latency_ms:  SLO target in milliseconds.  Requests that complete
                            with an actual latency exceeding this value are
                            counted as SLO violations.  Must be > 0.
    """

    tenant_id: str
    weight: float = 1.0
    max_concurrent: int = 4
    target_latency_ms: float = 200.0

    def __post_init__(self) -> None:
        if not self.tenant_id:
            raise ValueError("tenant_id must be a non-empty string")
        if self.weight <= 0.0:
            raise ValueError(f"weight must be > 0, got {self.weight}")
        if self.max_concurrent < 1:
            raise ValueError(f"max_concurrent must be >= 1, got {self.max_concurrent}")
        if self.target_latency_ms <= 0.0:
            raise ValueError(
                f"target_latency_ms must be > 0, got {self.target_latency_ms}"
            )


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------


@dataclass
class TenantRequest:
    """A single request submitted by a tenant for scheduling.

    Attributes:
        request_id:    Globally unique string identifier for this request.
        tenant_id:     The tenant that owns this request.
        n_tokens_est:  Estimated output token count.  Used by the WFQ virtual
                       clock to proportion the bandwidth consumed.  Must be >= 1.
        submitted_at:  Wall-clock timestamp (e.g. ``time.monotonic()``) when the
                       request was submitted.  Used for latency accounting.
    """

    request_id: str
    tenant_id: str
    n_tokens_est: int
    submitted_at: float

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("request_id must be a non-empty string")
        if not self.tenant_id:
            raise ValueError("tenant_id must be a non-empty string")
        if self.n_tokens_est < 1:
            raise ValueError(f"n_tokens_est must be >= 1, got {self.n_tokens_est}")


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


@dataclass
class TenantStats:
    """Aggregate scheduling statistics across all tenants.

    Attributes:
        total_submitted:  Total number of requests submitted since creation.
        total_completed:  Total number of requests marked complete.
        slo_violations:   Number of completions whose actual latency exceeded
                          the tenant's ``target_latency_ms``.
    """

    total_submitted: int = 0
    total_completed: int = 0
    slo_violations: int = 0

    @property
    def slo_violation_rate(self) -> float:
        """Fraction of completed requests that violated their SLO (0.0–1.0).

        Returns 0.0 when no requests have been completed yet.
        """
        if self.total_completed == 0:
            return 0.0
        return self.slo_violations / self.total_completed


# ---------------------------------------------------------------------------
# Internal per-tenant state
# ---------------------------------------------------------------------------


@dataclass
class _TenantState:
    """Mutable runtime state maintained per tenant by the scheduler (private)."""

    config: TenantConfig
    queue: deque = field(default_factory=deque)
    # WFQ virtual finish time for the last dispatched request.
    virtual_time: float = 0.0
    # Number of requests currently in-flight (dispatched but not completed).
    concurrent: int = 0
    # Per-tenant SLO violation counter.
    slo_violations: int = 0


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------


class TenantScheduler:
    """Weighted Fair Queue scheduler for multi-tenant LLM request dispatch.

    Requests submitted by each tenant are enqueued independently.  On each call
    to :meth:`next_request`, the scheduler selects the head request from the
    tenant queue whose WFQ virtual finish time is smallest, subject to each
    tenant's ``max_concurrent`` concurrency cap.  This guarantees that
    high-weight tenants receive proportionally more throughput under load, while
    lower-weight tenants remain schedulable.

    Args:
        tenants: List of :class:`TenantConfig` instances describing the tenants
                 known at construction time.  Each ``tenant_id`` must be unique.

    Raises:
        ValueError: if ``tenants`` is empty or contains duplicate tenant IDs.
    """

    def __init__(self, tenants: list[TenantConfig]) -> None:
        if not tenants:
            raise ValueError("tenants list must contain at least one TenantConfig")

        ids = [t.tenant_id for t in tenants]
        if len(ids) != len(set(ids)):
            raise ValueError(
                f"Duplicate tenant IDs detected: {[x for x in ids if ids.count(x) > 1]}"
            )

        self._states: dict[str, _TenantState] = {
            t.tenant_id: _TenantState(config=t) for t in tenants
        }
        # Index active requests by request_id for O(1) completion lookup.
        self._active: dict[str, tuple[str, float]] = {}  # request_id → (tenant_id, target_ms)
        self._stats = TenantStats()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def submit(self, req: TenantRequest) -> None:
        """Enqueue *req* into its tenant's queue for future dispatch.

        Args:
            req: The :class:`TenantRequest` to schedule.

        Raises:
            KeyError:   if ``req.tenant_id`` is not a registered tenant.
            ValueError: if ``req.request_id`` is already queued or in-flight.
        """
        if req.tenant_id not in self._states:
            raise KeyError(
                f"Unknown tenant '{req.tenant_id}'.  "
                f"Registered tenants: {list(self._states)}"
            )
        if req.request_id in self._active:
            raise ValueError(
                f"request_id '{req.request_id}' is already active"
            )
        # Check that the same request_id is not already in the queue.
        state = self._states[req.tenant_id]
        for queued in state.queue:
            if queued.request_id == req.request_id:
                raise ValueError(
                    f"request_id '{req.request_id}' is already queued for "
                    f"tenant '{req.tenant_id}'"
                )
        state.queue.append(req)
        self._stats.total_submitted += 1

    def next_request(self) -> Optional[TenantRequest]:
        """Dispatch the highest-priority request using Weighted Fair Queuing.

        Eligibility requires that a tenant has at least one queued request *and*
        has not exceeded its ``max_concurrent`` concurrency cap.  Among all
        eligible tenants, the one with the smallest ``virtual_time`` is selected
        (fair-share scheduling).

        Returns:
            The next :class:`TenantRequest` to process, or ``None`` if no
            eligible request exists.
        """
        best_state: Optional[_TenantState] = None
        best_vt: float = float("inf")

        for state in self._states.values():
            if not state.queue:
                continue
            if state.concurrent >= state.config.max_concurrent:
                continue
            if state.virtual_time < best_vt:
                best_vt = state.virtual_time
                best_state = state

        if best_state is None:
            return None

        req = best_state.queue.popleft()
        # Advance virtual clock by the normalised token cost.
        best_state.virtual_time += req.n_tokens_est / best_state.config.weight
        best_state.concurrent += 1
        self._active[req.request_id] = (
            req.tenant_id,
            best_state.config.target_latency_ms,
        )
        return req

    def complete(self, request_id: str, actual_latency_ms: float) -> None:
        """Mark *request_id* as finished and update SLO accounting.

        Args:
            request_id:         The ID of the request that has finished.
            actual_latency_ms:  End-to-end latency observed for this request.

        Raises:
            KeyError:   if ``request_id`` is not currently in-flight.
            ValueError: if ``actual_latency_ms`` is negative.
        """
        if request_id not in self._active:
            raise KeyError(
                f"request_id '{request_id}' is not currently in-flight"
            )
        if actual_latency_ms < 0.0:
            raise ValueError(
                f"actual_latency_ms must be >= 0, got {actual_latency_ms}"
            )

        tenant_id, target_ms = self._active.pop(request_id)
        state = self._states[tenant_id]
        state.concurrent = max(0, state.concurrent - 1)

        if actual_latency_ms > target_ms:
            state.slo_violations += 1
            self._stats.slo_violations += 1

        self._stats.total_completed += 1

    @property
    def stats(self) -> TenantStats:
        """Aggregate scheduling statistics across all tenants."""
        return self._stats

    # ------------------------------------------------------------------
    # Inspection helpers
    # ------------------------------------------------------------------

    def queue_depth(self, tenant_id: str) -> int:
        """Return the number of requests currently queued for *tenant_id*.

        Raises:
            KeyError: if *tenant_id* is not registered.
        """
        if tenant_id not in self._states:
            raise KeyError(f"Unknown tenant '{tenant_id}'")
        return len(self._states[tenant_id].queue)

    def tenant_slo_violations(self, tenant_id: str) -> int:
        """Return the number of SLO violations recorded for *tenant_id*.

        Raises:
            KeyError: if *tenant_id* is not registered.
        """
        if tenant_id not in self._states:
            raise KeyError(f"Unknown tenant '{tenant_id}'")
        return self._states[tenant_id].slo_violations
