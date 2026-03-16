# [Experimental] This module is part of Squish v6+ (Waves 19–26).
# Proof-of-concept quality: API and behaviour may change without notice.
# For stable core functionality see Waves 1–12 in MODULES.md.
#!/usr/bin/env python3
"""
squish/model_pool.py

ModelPool — Hot model pool with lazy loading and LRU eviction.

In multi-model serving, keeping all models resident in memory simultaneously
is impractical.  A hot pool keeps the most recently used (MRU) models ready,
evicting the least recently used (LRU) model when adding a new one would
exceed the configured memory capacity.

Models must first be registered with their size before they can be acquired.
:meth:`ModelPool.acquire` bumps an active reference count so that an in-use
model is never evicted mid-request.  Callers must call :meth:`ModelPool.release`
when they are finished with a model.

Example usage::

    from squish.serving.model_pool import ModelPool, PoolEntry

    pool = ModelPool(capacity_mb=8192.0)
    pool.register("llama-3-8b", size_mb=4096.0)
    pool.register("llama-3-70b", size_mb=35000.0)
    pool.register("phi-3-mini", size_mb=2048.0)

    entry = pool.acquire("llama-3-8b")
    print(f"loaded={pool.loaded_models}, util={pool.utilization:.1%}")
    pool.release("llama-3-8b")

    evicted = pool.evict_lru()
    print(f"evicted={evicted!r}, stats={pool.stats}")
"""

from __future__ import annotations

__all__ = [
    "PoolEntry",
    "ModelPool",
    "PoolStats",
]

import time
from dataclasses import dataclass
from typing import Optional

# ---------------------------------------------------------------------------
# Pool entry
# ---------------------------------------------------------------------------

@dataclass
class PoolEntry:
    """Metadata for a single model managed by :class:`ModelPool`.

    Attributes:
        model_id:     Unique model identifier string.
        size_mb:      Model memory footprint in megabytes.
        last_accessed: Unix timestamp of the most recent :meth:`ModelPool.acquire`.
        access_count: Cumulative number of times this entry has been acquired.
    """

    model_id: str
    size_mb: float
    last_accessed: float = 0.0
    access_count: int = 0

    def __post_init__(self) -> None:
        if not self.model_id:
            raise ValueError("model_id must be a non-empty string")
        if self.size_mb <= 0.0:
            raise ValueError(
                f"size_mb must be > 0, got {self.size_mb}"
            )


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

@dataclass
class PoolStats:
    """Cumulative statistics collected by :class:`ModelPool`.

    Attributes:
        total_acquires:  Total calls to :meth:`ModelPool.acquire`.
        total_evictions: Total models evicted to free capacity.
        cache_hits:      Acquires satisfied from an already-loaded model.
        cache_misses:    Acquires that required loading a model into memory.
    """

    total_acquires: int = 0
    total_evictions: int = 0
    cache_hits: int = 0
    cache_misses: int = 0

    @property
    def hit_rate(self) -> float:
        """Fraction of acquires served from the hot pool (0.0–1.0).

        Returns 0.0 when no acquires have been made yet.
        """
        if self.total_acquires == 0:
            return 0.0
        return self.cache_hits / self.total_acquires


# ---------------------------------------------------------------------------
# Model pool
# ---------------------------------------------------------------------------

class ModelPool:
    """Hot model pool with lazy loading and LRU eviction.

    Maintains a registry of known models and a hot set of currently-loaded
    models bounded by ``capacity_mb``.  Models are loaded on first acquire
    and evicted LRU when capacity would be exceeded.

    A model with an active reference (acquired but not yet released) is
    never eligible for eviction.

    Args:
        capacity_mb: Maximum combined memory footprint (MB) of loaded models.

    Raises:
        ValueError: if ``capacity_mb`` is not positive.
    """

    def __init__(self, capacity_mb: float = 16_384.0) -> None:
        if capacity_mb <= 0.0:
            raise ValueError(
                f"capacity_mb must be > 0, got {capacity_mb}"
            )
        self._capacity_mb = capacity_mb
        # All registered models (loaded or not).
        self._registry: dict[str, PoolEntry] = {}
        # Currently loaded models.
        self._loaded: dict[str, PoolEntry] = {}
        # Active reference counts — must be 0 for eviction eligibility.
        self._active_refs: dict[str, int] = {}
        self._stats = PoolStats()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register(self, model_id: str, size_mb: float) -> None:
        """Add a model to the registry without loading it into memory.

        Re-registering an existing model_id with a different size updates
        the stored size.  If the model is already loaded, its in-memory
        entry is updated accordingly.

        Args:
            model_id: Unique model identifier.
            size_mb:  Memory footprint in megabytes (> 0).

        Raises:
            ValueError: if ``model_id`` is empty or ``size_mb`` is not positive.
        """
        entry = PoolEntry(model_id=model_id, size_mb=size_mb)
        self._registry[model_id] = entry
        if model_id in self._loaded:
            self._loaded[model_id].size_mb = size_mb

    def acquire(self, model_id: str) -> PoolEntry:
        """Mark a model as in-use, loading it if necessary.

        If the model is already loaded (cache hit), its ``last_accessed``
        and ``access_count`` are updated.  On a cache miss, LRU evictions
        are performed until sufficient capacity is available, then the model
        is marked as loaded.

        Args:
            model_id: Identifier of a previously registered model.

        Returns:
            The :class:`PoolEntry` for the acquired model.

        Raises:
            KeyError:    if ``model_id`` has not been registered.
            MemoryError: if the model is too large even after evicting all
                         idle models.
        """
        if model_id not in self._registry:
            raise KeyError(
                f"model_id {model_id!r} is not registered; "
                "call register() first"
            )

        self._stats.total_acquires += 1
        now = time.time()

        if model_id in self._loaded:
            # Cache hit — model is already resident.
            entry = self._loaded[model_id]
            entry.last_accessed = now
            entry.access_count += 1
            self._active_refs[model_id] = self._active_refs.get(model_id, 0) + 1
            self._stats.cache_hits += 1
            return entry

        # Cache miss — need to load the model.
        self._stats.cache_misses += 1
        reg_entry = self._registry[model_id]
        needed_mb = reg_entry.size_mb

        if needed_mb > self._capacity_mb:
            raise MemoryError(
                f"Model {model_id!r} requires {needed_mb:.1f} MB which "
                f"exceeds total pool capacity {self._capacity_mb:.1f} MB"
            )

        # Evict LRU idle models until there is enough room.
        while self._loaded_mb() + needed_mb > self._capacity_mb:
            evicted = self.evict_lru()
            if evicted is None:
                raise MemoryError(
                    f"Cannot load {model_id!r} ({needed_mb:.1f} MB): "
                    f"capacity {self._capacity_mb:.1f} MB, loaded "
                    f"{self._loaded_mb():.1f} MB, all remaining models "
                    "have active references"
                )

        # Create a fresh loaded entry, copying metadata from registry.
        entry = PoolEntry(
            model_id=model_id,
            size_mb=needed_mb,
            last_accessed=now,
            access_count=1,
        )
        self._loaded[model_id] = entry
        self._active_refs[model_id] = 1
        return entry

    def release(self, model_id: str) -> None:
        """Decrement the active reference count for a loaded model.

        After the reference count reaches zero the model remains loaded
        (hot) until it is evicted by a future :meth:`acquire` call.

        Args:
            model_id: Identifier of a previously acquired model.

        Raises:
            KeyError:    if ``model_id`` is not currently loaded.
            RuntimeError: if the reference count is already zero.
        """
        if model_id not in self._loaded:
            raise KeyError(
                f"model_id {model_id!r} is not currently loaded"
            )
        refs = self._active_refs.get(model_id, 0)
        if refs <= 0:
            raise RuntimeError(
                f"model_id {model_id!r} has no active references to release"
            )
        self._active_refs[model_id] = refs - 1

    def evict_lru(self) -> str | None:
        """Evict the least recently used idle model from the hot pool.

        Only models with an active reference count of zero are eligible.

        Returns:
            The ``model_id`` of the evicted model, or ``None`` if no model
            is eligible for eviction.
        """
        candidates = [
            entry
            for model_id, entry in self._loaded.items()
            if self._active_refs.get(model_id, 0) == 0
        ]
        if not candidates:
            return None

        # Select the entry with the oldest last_accessed timestamp.
        lru_entry = min(candidates, key=lambda e: e.last_accessed)
        evicted_id = lru_entry.model_id
        del self._loaded[evicted_id]
        self._active_refs.pop(evicted_id, None)
        self._stats.total_evictions += 1
        return evicted_id

    @property
    def loaded_models(self) -> list[str]:
        """Sorted list of model IDs currently resident in the hot pool."""
        return sorted(self._loaded.keys())

    @property
    def utilization(self) -> float:
        """Fraction of pool capacity currently occupied by loaded models (0.0–1.0)."""
        return self._loaded_mb() / self._capacity_mb

    @property
    def stats(self) -> PoolStats:
        """Cumulative pool statistics (updated in place)."""
        return self._stats

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _loaded_mb(self) -> float:
        """Return total MB consumed by currently loaded models."""
        return sum(entry.size_mb for entry in self._loaded.values())
