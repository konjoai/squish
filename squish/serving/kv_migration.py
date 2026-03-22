"""squish/serving/kv_migration.py

KVMigrationManager — Live KV shard migration between serving workers.

Manages block-page KV shard transfers between workers.  Each worker has a
bounded KV page capacity.  When a worker's free headroom drops below
``low_watermark``, rebalancing migrates KV pages from overloaded workers to
workers with spare capacity.

This is a CPU-simulation of the KV-page migration subsystem used in
distributed LLM serving frameworks (e.g., vLLM, DistServe).

Reference
---------
Zhong et al., "DistServe: Disaggregating Prefill and Decoding for Goodput-
Optimized Large Language Model Serving." OSDI 2024.
"""

from __future__ import annotations

__all__ = ["KVMigrationConfig", "MigrationRecord", "KVMigrationManager"]

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy import ndarray


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class KVMigrationConfig:
    """Configuration for KVMigrationManager.

    Parameters
    ----------
    page_size:
        Number of tokens per KV page.
    low_watermark:
        Headroom fraction below which rebalancing is triggered.
    seed:
        RNG seed (reserved for future stochastic tie-breaking).
    """

    page_size: int = 16
    low_watermark: float = 0.15
    seed: int = 0

    def __post_init__(self) -> None:
        if self.page_size < 1:
            raise ValueError("page_size must be >= 1")
        if not (0.0 < self.low_watermark < 1.0):
            raise ValueError("low_watermark must be in (0, 1)")


# ---------------------------------------------------------------------------
# Migration record
# ---------------------------------------------------------------------------

@dataclass
class MigrationRecord:
    """Record of a completed KV page migration.

    Attributes
    ----------
    src_worker:
        Source worker ID.
    dst_worker:
        Destination worker ID.
    session_id:
        Session / request whose KV pages were moved.
    n_pages:
        Number of pages transferred.
    timestamp:
        Unix timestamp at migration completion.
    """

    src_worker: str
    dst_worker: str
    session_id: str
    n_pages: int
    timestamp: float = field(default_factory=time.time)

    @property
    def bytes_transferred(self) -> int:
        """Simulated bytes transferred (n_pages × page_size × token_size).

        Uses a fixed 2048-byte-per-page estimate (32 tokens × 64-byte KV vec).
        """
        BYTES_PER_PAGE = 2048
        return self.n_pages * BYTES_PER_PAGE


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------

class KVMigrationManager:
    """Coordinate KV page allocation and migration across workers.

    Parameters
    ----------
    config:
        ``KVMigrationConfig`` instance.
    """

    def __init__(self, config: KVMigrationConfig) -> None:
        self.config = config
        # worker_id → capacity (max pages)
        self._capacity: Dict[str, int] = {}
        # worker_id → session_id → n_pages_allocated
        self._allocations: Dict[str, Dict[str, int]] = {}
        self._migration_log: List[MigrationRecord] = []

    def register_worker(self, worker_id: str, capacity_pages: int) -> None:
        """Register a worker with a given KV page capacity.

        Parameters
        ----------
        worker_id:
            Unique string identifier for the worker.
        capacity_pages:
            Total number of KV pages the worker can hold.
        """
        if capacity_pages < 1:
            raise ValueError("capacity_pages must be >= 1")
        self._capacity[worker_id] = capacity_pages
        self._allocations.setdefault(worker_id, {})

    def worker_headroom(self, worker_id: str) -> float:
        """Return the fraction of free capacity for ``worker_id``.

        Returns
        -------
        Headroom in ``[0, 1]``.  1.0 = completely empty; 0.0 = full.
        """
        if worker_id not in self._capacity:
            raise KeyError(f"Unknown worker: {worker_id!r}")
        cap = self._capacity[worker_id]
        used = sum(self._allocations[worker_id].values())
        return max(0.0, (cap - used) / cap)

    def allocate_pages(
        self, worker_id: str, session_id: str, n_pages: int
    ) -> bool:
        """Attempt to allocate ``n_pages`` for ``session_id`` on ``worker_id``.

        Returns
        -------
        True if allocation succeeded; False if insufficient capacity.
        """
        if worker_id not in self._capacity:
            raise KeyError(f"Unknown worker: {worker_id!r}")
        cap = self._capacity[worker_id]
        used = sum(self._allocations[worker_id].values())
        if used + n_pages > cap:
            return False
        self._allocations[worker_id][session_id] = (
            self._allocations[worker_id].get(session_id, 0) + n_pages
        )
        return True

    def free_pages(self, worker_id: str, session_id: str) -> None:
        """Free all pages allocated for ``session_id`` on ``worker_id``."""
        if worker_id not in self._capacity:
            raise KeyError(f"Unknown worker: {worker_id!r}")
        self._allocations[worker_id].pop(session_id, None)

    def migrate(
        self, src: str, dst: str, session_id: str
    ) -> MigrationRecord:
        """Move all KV pages for ``session_id`` from ``src`` to ``dst``.

        Parameters
        ----------
        src:
            Source worker ID.
        dst:
            Destination worker ID.
        session_id:
            Session whose pages are migrated.

        Returns
        -------
        ``MigrationRecord`` describing the transfer.

        Raises
        ------
        KeyError
            If either worker is not registered, or session has no pages on src.
        ValueError
            If dst has insufficient capacity.
        """
        for w in (src, dst):
            if w not in self._capacity:
                raise KeyError(f"Unknown worker: {w!r}")
        n_pages = self._allocations[src].get(session_id, 0)
        if n_pages == 0:
            raise KeyError(f"Session {session_id!r} has no pages on worker {src!r}")
        dst_used = sum(self._allocations[dst].values())
        if dst_used + n_pages > self._capacity[dst]:
            raise ValueError(
                f"Worker {dst!r} has insufficient capacity for {n_pages} pages"
            )
        # Transfer
        self._allocations[src].pop(session_id)
        self._allocations[dst][session_id] = (
            self._allocations[dst].get(session_id, 0) + n_pages
        )
        record = MigrationRecord(
            src_worker=src,
            dst_worker=dst,
            session_id=session_id,
            n_pages=n_pages,
        )
        self._migration_log.append(record)
        return record

    def rebalance(self, headrooms: Dict[str, float]) -> List[MigrationRecord]:
        """Rebalance workers below ``low_watermark`` headroom.

        Workers whose headroom < low_watermark are treated as overloaded;
        workers with the highest headroom receive their sessions.

        Parameters
        ----------
        headrooms:
            Mapping of worker_id → current headroom fraction.

        Returns
        -------
        List of ``MigrationRecord`` describing all migrations performed.
        """
        records: List[MigrationRecord] = []
        overloaded = [
            w for w, h in headrooms.items()
            if h < self.config.low_watermark and w in self._capacity
        ]
        if not overloaded:
            return records

        for src in overloaded:
            sessions = list(self._allocations.get(src, {}).keys())
            for session_id in sessions:
                # Find worker with most headroom (excluding src)
                candidates = {
                    w: h for w, h in headrooms.items()
                    if w != src and w in self._capacity
                }
                if not candidates:
                    break
                dst = max(candidates, key=lambda w: candidates[w])
                try:
                    rec = self.migrate(src, dst, session_id)
                    records.append(rec)
                    # Update local headroom estimate
                    n = rec.n_pages
                    headrooms[src] = min(1.0, headrooms[src] + n / self._capacity[src])
                    headrooms[dst] = max(0.0, headrooms[dst] - n / self._capacity[dst])
                except (KeyError, ValueError):
                    continue
        return records

    def stats(self) -> dict:
        """Return a summary statistics dict."""
        return {
            "n_workers": len(self._capacity),
            "n_migrations": len(self._migration_log),
            "total_pages_migrated": sum(r.n_pages for r in self._migration_log),
            "worker_headrooms": {w: self.worker_headroom(w) for w in self._capacity},
        }
