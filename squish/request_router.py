#!/usr/bin/env python3
"""
squish/request_router.py

RequestRouter — Load-aware request routing across model replicas.

The router maintains a live view of how many requests each replica is currently
serving and directs incoming requests to the least-loaded replica, weighted by
the replica's declared throughput weight.  The effective load is computed as
``n_active / (max_concurrent * weight)``, so a heavier replica can absorb
proportionally more traffic before it is considered equally loaded to a lighter
one.

Replicas can be added at runtime via :meth:`ReplicaRegistry.register` and their
load can be updated externally (e.g. from a heartbeat) via
:meth:`ReplicaRegistry.update_load`.  The router itself tracks which replica was
assigned to each request so that :meth:`RequestRouter.complete` can decrement
the correct counter.

Example usage::

    from squish.request_router import ReplicaConfig, ReplicaRegistry, RequestRouter

    registry = ReplicaRegistry([
        ReplicaConfig(replica_id="gpu-0", max_concurrent=8, weight=2.0),
        ReplicaConfig(replica_id="gpu-1", max_concurrent=4, weight=1.0),
    ])
    router = RequestRouter(registry)

    replica_id = router.route("req-001")
    print(f"req-001 routed to {replica_id}")
    router.complete("req-001")
    print(router.stats)
"""

from __future__ import annotations

__all__ = [
    "ReplicaConfig",
    "ReplicaRegistry",
    "RequestRouter",
    "RouterStats",
]

from dataclasses import dataclass
from typing import Optional

import numpy as np  # noqa: F401  — imported for dtype compatibility in future extensions


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class ReplicaConfig:
    """Static configuration for a single model replica.

    Attributes:
        replica_id:     Unique identifier for this replica.
        max_concurrent: Maximum number of requests the replica can serve
                        simultaneously.  Used as the denominator when computing
                        fractional load.  Must be >= 1.
        weight:         Relative throughput weight.  A replica with weight 2.0
                        is treated as having twice the capacity of one with
                        weight 1.0 when load-balancing.  Must be > 0.
    """

    replica_id: str
    max_concurrent: int = 8
    weight: float = 1.0

    def __post_init__(self) -> None:
        if not self.replica_id:
            raise ValueError("replica_id must be a non-empty string")
        if self.max_concurrent < 1:
            raise ValueError(f"max_concurrent must be >= 1, got {self.max_concurrent}")
        if self.weight <= 0.0:
            raise ValueError(f"weight must be > 0, got {self.weight}")


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class ReplicaRegistry:
    """Live registry of known replicas and their current load.

    Args:
        replicas: Initial list of :class:`ReplicaConfig` instances.  Each
                  ``replica_id`` must be unique within the registry.

    Raises:
        ValueError: if *replicas* is empty or contains duplicate IDs.
    """

    def __init__(self, replicas: list[ReplicaConfig]) -> None:
        if not replicas:
            raise ValueError("replicas list must contain at least one ReplicaConfig")
        self._configs: dict[str, ReplicaConfig] = {}
        # Mutable load counter per replica.
        self._load: dict[str, int] = {}
        for r in replicas:
            self.register(r)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register(self, r: ReplicaConfig) -> None:
        """Add *r* to the registry.

        If a replica with the same ``replica_id`` already exists, it is
        replaced by *r* and its load counter is reset to zero.

        Args:
            r: The :class:`ReplicaConfig` to register.
        """
        self._configs[r.replica_id] = r
        self._load.setdefault(r.replica_id, 0)

    def update_load(self, replica_id: str, n_active: int) -> None:
        """Overwrite the active-request count for *replica_id*.

        Used by external heartbeat or monitoring systems to synchronise the
        router's view of replica load with ground truth.

        Args:
            replica_id: ID of the replica to update.
            n_active:   Current number of in-flight requests on that replica.
                        Must be >= 0.

        Raises:
            KeyError:   if *replica_id* is not registered.
            ValueError: if *n_active* is negative.
        """
        if replica_id not in self._configs:
            raise KeyError(f"Unknown replica '{replica_id}'")
        if n_active < 0:
            raise ValueError(f"n_active must be >= 0, got {n_active}")
        self._load[replica_id] = n_active

    def get_replica(self, replica_id: str) -> Optional[ReplicaConfig]:
        """Return the :class:`ReplicaConfig` for *replica_id*, or ``None``.

        Args:
            replica_id: The ID to look up.
        """
        return self._configs.get(replica_id)

    def effective_load(self, replica_id: str) -> float:
        """Fractional load for *replica_id* as ``n_active / (max_concurrent * weight)``.

        A value of 0.0 means the replica is completely idle; 1.0 means it is
        at its weighted capacity.

        Raises:
            KeyError: if *replica_id* is not registered.
        """
        if replica_id not in self._configs:
            raise KeyError(f"Unknown replica '{replica_id}'")
        cfg = self._configs[replica_id]
        capacity = cfg.max_concurrent * cfg.weight
        return self._load[replica_id] / capacity

    @property
    def replica_ids(self) -> list[str]:
        """Sorted list of all registered replica IDs."""
        return sorted(self._configs)

    @property
    def n_replicas(self) -> int:
        """Number of registered replicas."""
        return len(self._configs)

    # ------------------------------------------------------------------
    # Package-private access for the router
    # ------------------------------------------------------------------

    def _increment(self, replica_id: str) -> None:
        self._load[replica_id] = self._load.get(replica_id, 0) + 1

    def _decrement(self, replica_id: str) -> None:
        self._load[replica_id] = max(0, self._load.get(replica_id, 0) - 1)

    def _all_loads(self) -> dict[str, float]:
        """Return effective load for every replica."""
        return {rid: self.effective_load(rid) for rid in self._configs}


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


@dataclass
class RouterStats:
    """Cumulative routing statistics.

    Attributes:
        total_routed:    Total number of requests dispatched since creation.
        total_completed: Total number of requests marked complete.
    """

    total_routed: int = 0
    total_completed: int = 0

    @property
    def avg_load(self) -> float:
        """Average in-flight requests per completed routing slot.

        Returns ``total_routed / total_completed`` as a rough measure of
        concurrency utilisation, or 0.0 when nothing has completed yet.
        """
        if self.total_completed == 0:
            return 0.0
        return self.total_routed / self.total_completed


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


class RequestRouter:
    """Load-aware request router using least-loaded weighted scheduling.

    On each :meth:`route` call the router selects the replica with the lowest
    ``effective_load`` (``n_active / (max_concurrent * weight)``).  Ties are
    broken by ``replica_id`` lexicographic order for determinism.

    The router tracks which replica was assigned to each active request so that
    :meth:`complete` can atomically decrement the correct load counter.

    Args:
        registry: A populated :class:`ReplicaRegistry` instance.
    """

    def __init__(self, registry: ReplicaRegistry) -> None:
        self._registry = registry
        # Maps request_id → replica_id for in-flight requests.
        self._assignments: dict[str, str] = {}
        self._stats = RouterStats()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def route(self, request_id: str) -> str:
        """Route *request_id* to the least-loaded replica.

        Args:
            request_id: A unique identifier for the incoming request.

        Returns:
            The ``replica_id`` of the selected replica.

        Raises:
            ValueError: if *request_id* is already in-flight.
            RuntimeError: if no replicas are registered.
        """
        if not request_id:
            raise ValueError("request_id must be a non-empty string")
        if request_id in self._assignments:
            raise ValueError(
                f"request_id '{request_id}' is already in-flight on replica "
                f"'{self._assignments[request_id]}'"
            )
        if self._registry.n_replicas == 0:
            raise RuntimeError("No replicas registered in the registry")

        loads = self._registry._all_loads()
        # Deterministic tie-breaking: sort by (effective_load, replica_id).
        best_id = min(loads, key=lambda rid: (loads[rid], rid))

        self._registry._increment(best_id)
        self._assignments[request_id] = best_id
        self._stats.total_routed += 1
        return best_id

    def complete(self, request_id: str) -> None:
        """Mark *request_id* as finished and release its replica slot.

        Args:
            request_id: The ID of the completed request.

        Raises:
            KeyError: if *request_id* is not currently in-flight.
        """
        if request_id not in self._assignments:
            raise KeyError(
                f"request_id '{request_id}' is not currently in-flight"
            )
        replica_id = self._assignments.pop(request_id)
        self._registry._decrement(replica_id)
        self._stats.total_completed += 1

    @property
    def stats(self) -> RouterStats:
        """Cumulative routing statistics."""
        return self._stats

    # ------------------------------------------------------------------
    # Inspection helpers
    # ------------------------------------------------------------------

    def assigned_replica(self, request_id: str) -> Optional[str]:
        """Return the replica currently serving *request_id*, or ``None``."""
        return self._assignments.get(request_id)

    @property
    def n_active(self) -> int:
        """Total number of requests currently in-flight across all replicas."""
        return len(self._assignments)
