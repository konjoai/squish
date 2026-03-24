"""squish/moe/expert_memory_map.py

ExpertMemoryMap — LRU-managed resident set for MoE expert weights.

Maintains a bounded RAM budget across all expert weight tensors.  When a
new expert is requested and the budget is exhausted, the least-recently-used
resident expert is evicted and its numpy arrays released.

Supports optional INT4-packed storage (via :mod:`squish.moe.int4_expert_pack`)
to quadruple the effective resident set without additional RAM.

Design
------
  * Each "expert slot" is identified by (layer_idx, expert_idx).
  * Expert weights are stored as a dict of projection-name → numpy array.
  * LRU tracking is done via an OrderedDict (O(1) move-to-end / popitem).
  * ``pin()`` / ``unpin()`` protect actively-used experts from eviction.

Reference
---------
Eliseev & Panferov, "Fast Inference of Mixture-of-Experts Language Models
with Offloading," arXiv:2312.17238, 2023.

Usage
-----
::

    from squish.moe.expert_memory_map import ExpertMemoryMap, MemoryMapConfig

    cfg = MemoryMapConfig(budget_mb=2048)
    emap = ExpertMemoryMap(cfg)

    # load and pin expert (0, 0)
    emap.put(0, 0, {"gate": gate_array, "up": up_array, "down": down_array})
    weights = emap.get(0, 0)    # → dict or None
    emap.pin(0, 0)
    ...
    emap.unpin(0, 0)

    print(emap.stats())         # MemoryMapStats
"""

from __future__ import annotations

__all__ = [
    "MemoryMapConfig",
    "MemoryMapStats",
    "ExpertMemoryMap",
]

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, Optional, Set, Tuple

import numpy as np

# Type alias for an expert's weight dictionary
ExpertWeights = Dict[str, np.ndarray]
ExpertKey = Tuple[int, int]  # (layer_idx, expert_idx)


# ---------------------------------------------------------------------------
# Config / stats
# ---------------------------------------------------------------------------

@dataclass
class MemoryMapConfig:
    """Configuration for ExpertMemoryMap.

    Attributes
    ----------
    budget_mb:
        Maximum RAM allowed for resident expert weights, in megabytes.
        Default is 4096 MB (4 GB), which holds ~30 INT4-compressed Mixtral
        8x7B experts at once.
    max_experts:
        Hard cap on resident expert count independent of budget.
        0 = unlimited (budget_mb is the only constraint).
    eviction_batch:
        Number of experts to evict at once when budget is exceeded.
        Higher values reduce eviction frequency at the cost of larger
        memory spikes.
    """

    budget_mb: float = 4096.0
    max_experts: int = 0
    eviction_batch: int = 1

    def __post_init__(self) -> None:
        if self.budget_mb <= 0:
            raise ValueError("budget_mb must be > 0")
        if self.max_experts < 0:
            raise ValueError("max_experts must be >= 0")
        if self.eviction_batch < 1:
            raise ValueError("eviction_batch must be >= 1")

    @property
    def budget_bytes(self) -> int:
        return int(self.budget_mb * 1024 * 1024)


@dataclass
class MemoryMapStats:
    """Runtime statistics for an ExpertMemoryMap instance.

    Attributes
    ----------
    n_resident:
        Current number of experts held in the resident set.
    resident_bytes:
        Current bytes used by resident experts.
    budget_bytes:
        Configured budget in bytes.
    n_hits:
        Number of get() calls that found the expert resident.
    n_misses:
        Number of get() calls that found the expert absent.
    n_evictions:
        Total number of individual expert evictions performed.
    n_puts:
        Total number of put() calls.
    pinned_count:
        Number of currently pinned (eviction-protected) experts.
    """

    n_resident: int = 0
    resident_bytes: int = 0
    budget_bytes: int = 0
    n_hits: int = 0
    n_misses: int = 0
    n_evictions: int = 0
    n_puts: int = 0
    pinned_count: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.n_hits + self.n_misses
        return self.n_hits / total if total > 0 else 0.0

    @property
    def resident_mb(self) -> float:
        return self.resident_bytes / (1024 * 1024)

    @property
    def utilisation(self) -> float:
        """Fraction of budget currently used (0–1)."""
        return self.resident_bytes / self.budget_bytes if self.budget_bytes > 0 else 0.0


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _ndarray_bytes(arr: np.ndarray) -> int:
    """Return the byte size of *arr* (handles both contiguous and views)."""
    return arr.nbytes


def _expert_bytes(weights: ExpertWeights) -> int:
    return sum(_ndarray_bytes(v) for v in weights.values())


# ---------------------------------------------------------------------------
# ExpertMemoryMap
# ---------------------------------------------------------------------------

class ExpertMemoryMap:
    """LRU-managed RAM resident set for sparse MoE expert weights.

    Parameters
    ----------
    config:
        Memory budget and eviction settings.
    """

    def __init__(self, config: MemoryMapConfig) -> None:
        self._config = config
        # Ordered insertion/access order: oldest entry is first (popitem(last=False))
        self._lru: OrderedDict[ExpertKey, ExpertWeights] = OrderedDict()
        self._pinned: Set[ExpertKey] = set()
        self._used_bytes: int = 0
        self._n_hits: int = 0
        self._n_misses: int = 0
        self._n_evictions: int = 0
        self._n_puts: int = 0

    # ------------------------------------------------------------------ #
    # Core API
    # ------------------------------------------------------------------ #

    def get(self, layer_idx: int, expert_idx: int) -> Optional[ExpertWeights]:
        """Retrieve expert weights, marking the entry as most-recently-used.

        Returns ``None`` if the expert is not currently resident.

        Parameters
        ----------
        layer_idx:
            Zero-based transformer layer index.
        expert_idx:
            Zero-based expert index.
        """
        key: ExpertKey = (layer_idx, expert_idx)
        if key in self._lru:
            self._lru.move_to_end(key)
            self._n_hits += 1
            return self._lru[key]
        self._n_misses += 1
        return None

    def put(
        self,
        layer_idx: int,
        expert_idx: int,
        weights: ExpertWeights,
    ) -> None:
        """Insert or replace expert weights in the resident set.

        If the expert is already resident, its weights are updated and its
        LRU position is refreshed.  The method evicts LRU unpinned experts
        as needed to respect ``budget_mb`` and ``max_experts``.

        Parameters
        ----------
        layer_idx:
            Zero-based transformer layer index.
        expert_idx:
            Zero-based expert index.
        weights:
            Dict mapping projection name (e.g. "gate", "up", "down") to
            numpy array.
        """
        self._n_puts += 1
        key: ExpertKey = (layer_idx, expert_idx)
        new_bytes = _expert_bytes(weights)

        # If already resident, remove old entry first
        if key in self._lru:
            old_bytes = _expert_bytes(self._lru[key])
            self._used_bytes -= old_bytes
            del self._lru[key]

        # Evict until space is available
        self._evict_to_fit(new_bytes)

        self._lru[key] = weights
        self._lru.move_to_end(key)
        self._used_bytes += new_bytes

    def pin(self, layer_idx: int, expert_idx: int) -> None:
        """Mark an expert as pinned — it will not be evicted automatically.

        The expert must already be resident.  Pinning a non-resident expert
        is a no-op.
        """
        key: ExpertKey = (layer_idx, expert_idx)
        if key in self._lru:
            self._pinned.add(key)

    def unpin(self, layer_idx: int, expert_idx: int) -> None:
        """Remove the eviction-protection pin from an expert."""
        self._pinned.discard((layer_idx, expert_idx))

    def evict(self, layer_idx: int, expert_idx: int) -> bool:
        """Explicitly evict an expert (even if pinned).

        Returns True if the expert was resident and was removed.
        """
        key: ExpertKey = (layer_idx, expert_idx)
        if key not in self._lru:
            return False
        evicted_bytes = _expert_bytes(self._lru[key])
        del self._lru[key]
        self._used_bytes -= evicted_bytes
        self._pinned.discard(key)
        self._n_evictions += 1
        return True

    def is_resident(self, layer_idx: int, expert_idx: int) -> bool:
        return (layer_idx, expert_idx) in self._lru

    def clear(self) -> None:
        """Evict all experts and reset byte counters."""
        self._lru.clear()
        self._pinned.clear()
        self._used_bytes = 0

    # ------------------------------------------------------------------ #
    # Eviction internals
    # ------------------------------------------------------------------ #

    def _evict_to_fit(self, incoming_bytes: int) -> None:
        """Evict LRU unpinned experts until the incoming expert fits.

        If all remaining resident experts are pinned and the budget is still
        exceeded, we proceed anyway rather than looping forever — pinned
        experts are deliberately kept alive and the caller accepts the
        temporary budget overrun.
        """
        budget = self._config.budget_bytes
        max_exp = self._config.max_experts

        # Evict for budget
        while (
            self._used_bytes + incoming_bytes > budget
            and self._lru
        ):
            evicted = self._evict_lru_unpinned()
            if not evicted:
                break  # all remaining experts are pinned; accept budget overrun

        # Evict for max_experts cap
        if max_exp > 0:
            while len(self._lru) >= max_exp and self._lru:
                evicted = self._evict_lru_unpinned()
                if not evicted:
                    break  # all pinned; can't evict further

    def _evict_lru_unpinned(self) -> bool:
        """Evict the oldest unpinned entry.  Returns True if successful."""
        for key in list(self._lru.keys()):  # OrderedDict iteration = LRU order
            if key not in self._pinned:
                evicted_bytes = _expert_bytes(self._lru[key])
                del self._lru[key]
                self._used_bytes -= evicted_bytes
                self._n_evictions += 1
                return True
        return False  # all resident experts are pinned

    # ------------------------------------------------------------------ #
    # Stats / introspection
    # ------------------------------------------------------------------ #

    def stats(self) -> MemoryMapStats:
        return MemoryMapStats(
            n_resident=len(self._lru),
            resident_bytes=self._used_bytes,
            budget_bytes=self._config.budget_bytes,
            n_hits=self._n_hits,
            n_misses=self._n_misses,
            n_evictions=self._n_evictions,
            n_puts=self._n_puts,
            pinned_count=len(self._pinned),
        )

    def resident_keys(self) -> list[ExpertKey]:
        """Return list of (layer_idx, expert_idx) tuples in LRU order (oldest first)."""
        return list(self._lru.keys())

    def __len__(self) -> int:
        return len(self._lru)

    def __repr__(self) -> str:
        s = self.stats()
        return (
            f"ExpertMemoryMap("
            f"resident={s.n_resident}, "
            f"used={s.resident_mb:.1f}/{self._config.budget_mb:.0f} MB, "
            f"hit_rate={s.hit_rate:.1%})"
        )
