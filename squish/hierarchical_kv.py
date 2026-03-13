# [Experimental] This module is part of Squish v6+ (Waves 19–26).
# Proof-of-concept quality: API and behaviour may change without notice.
# For stable core functionality see Waves 1–12 in MODULES.md.
"""HierarchicalKV — Three-tier hot/warm/cold KV cache with transparent promotion.

Manages KV tensors across three tiers whose sizes grow progressively:

  hot  (small, fast)   — most-recently accessed tokens
  warm (medium)        — demoted-from-hot tokens
  cold (large)         — demoted-from-warm tokens; eviction is permanent here

Every ``get`` promotes the hit token to the hot tier, cascading demotions
downwards as needed.  Every ``put`` inserts into hot, with the same cascade.
LRU order within each tier is tracked via a plain ``list``: index 0 is the
oldest (least-recently used) token; new arrivals are appended at the end.

Typical usage::

    from squish.hierarchical_kv import TierConfig, HierarchicalKVStore

    cfg   = TierConfig(hot_capacity=64, warm_capacity=256, cold_capacity=1024)
    store = HierarchicalKVStore(cfg)

    key   = np.zeros((4, 64), dtype=np.float32)   # (n_heads, head_dim)
    value = np.ones((4, 64), dtype=np.float32)

    store.put(pos=0, key=key, value=value)
    kv = store.get(pos=0)   # returns (key, value) and stays in hot tier

    print(store.stats)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "TierConfig",
    "HierarchicalKVStore",
    "HierarchicalKVStats",
]

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class TierConfig:
    """Capacity and shape configuration for :class:`HierarchicalKVStore`.

    Parameters
    ----------
    hot_capacity : int
        Maximum number of token slots in the hot tier.
    warm_capacity : int
        Maximum number of token slots in the warm tier.
    cold_capacity : int
        Maximum number of token slots in the cold tier.
    n_heads : int
        Number of attention heads per KV tensor.
    head_dim : int
        Head dimension for each KV tensor.
    """

    hot_capacity: int = 64
    warm_capacity: int = 256
    cold_capacity: int = 1024
    n_heads: int = 4
    head_dim: int = 64

    def __post_init__(self) -> None:
        if self.hot_capacity < 1:
            raise ValueError("hot_capacity must be >= 1")
        if self.warm_capacity < 1:
            raise ValueError("warm_capacity must be >= 1")
        if self.cold_capacity < 1:
            raise ValueError("cold_capacity must be >= 1")
        if not (self.hot_capacity < self.warm_capacity < self.cold_capacity):
            raise ValueError(
                "Tier capacities must satisfy hot_capacity < warm_capacity < cold_capacity; "
                f"got hot={self.hot_capacity}, warm={self.warm_capacity}, cold={self.cold_capacity}"
            )
        if self.n_heads < 1:
            raise ValueError("n_heads must be >= 1")
        if self.head_dim < 1:
            raise ValueError("head_dim must be >= 1")


# ---------------------------------------------------------------------------
# Stats dataclass
# ---------------------------------------------------------------------------


@dataclass
class HierarchicalKVStats:
    """Aggregate statistics for :class:`HierarchicalKVStore`.

    Attributes
    ----------
    total_puts : int
        Total number of :meth:`~HierarchicalKVStore.put` calls.
    total_gets : int
        Total number of :meth:`~HierarchicalKVStore.get` calls.
    hot_hits : int
        Number of ``get`` hits resolved from the hot tier.
    warm_hits : int
        Number of ``get`` hits resolved from the warm tier.
    cold_hits : int
        Number of ``get`` hits resolved from the cold tier.
    cold_misses : int
        Number of ``get`` calls where the token was not found in any tier.
    total_demotions : int
        Total number of individual token demotion events (hot→warm or warm→cold).
    total_evictions : int
        Total number of permanent cold-tier evictions.
    """

    total_puts: int = 0
    total_gets: int = 0
    hot_hits: int = 0
    warm_hits: int = 0
    cold_hits: int = 0
    cold_misses: int = 0
    total_demotions: int = 0
    total_evictions: int = 0

    @property
    def hit_rate(self) -> float:
        """Fraction of ``get`` calls that found the token in any tier."""
        if self.total_gets == 0:
            return 0.0
        total_hits = self.hot_hits + self.warm_hits + self.cold_hits
        return total_hits / self.total_gets

    @property
    def hot_hit_rate(self) -> float:
        """Fraction of ``get`` calls resolved from hot tier."""
        if self.total_gets == 0:
            return 0.0
        return self.hot_hits / self.total_gets


# ---------------------------------------------------------------------------
# HierarchicalKVStore
# ---------------------------------------------------------------------------


class HierarchicalKVStore:
    """Three-tier KV cache with transparent LRU promotion and demotion.

    Parameters
    ----------
    config : TierConfig
        Tier capacity and tensor shape configuration.

    Internal representation
    -----------------------
    ``_store`` maps ``pos: int`` → ``(key, value, tier_name)`` where
    ``tier_name`` is one of ``"hot"``, ``"warm"``, or ``"cold"``.

    ``_hot_list``, ``_warm_list``, and ``_cold_list`` maintain insertion-order
    lists of token positions in each tier.  Index 0 is the LRU (oldest) token;
    the most-recently used token is at the tail (``[-1]``).

    Demotion cascade
    ----------------
    When a tier overflows its capacity the LRU token (index 0) is moved to the
    head of the next tier's list.  If the next tier also overflows, the cascade
    continues.  Cold-tier overflow results in permanent eviction.
    """

    def __init__(self, config: TierConfig) -> None:
        self._cfg = config
        # pos → (key, value, tier)
        self._store: dict[int, tuple[np.ndarray, np.ndarray, str]] = {}
        # LRU lists: index 0 = oldest, tail = most recent
        self._hot_list: list[int] = []
        self._warm_list: list[int] = []
        self._cold_list: list[int] = []
        self._stats = HierarchicalKVStats()

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def put(self, pos: int, key: np.ndarray, value: np.ndarray) -> None:
        """Insert or update the KV tensors at *pos*, placing them in the hot tier.

        If *pos* is already stored its tensors are updated in place and the
        token is promoted to the most-recently-used position of the hot tier.
        If hot is at capacity the LRU hot token is demoted to warm, cascading
        as needed.

        Parameters
        ----------
        pos : int
            Token position identifier (e.g. absolute sequence index).
        key : np.ndarray
            Key tensor, shape ``(n_heads, head_dim)``.
        value : np.ndarray
            Value tensor, shape ``(n_heads, head_dim)``.

        Raises
        ------
        ValueError
            If *key* or *value* have the wrong shape.
        """
        expected = (self._cfg.n_heads, self._cfg.head_dim)
        if key.shape != expected:
            raise ValueError(
                f"key shape must be {expected}; got {key.shape}"
            )
        if value.shape != expected:
            raise ValueError(
                f"value shape must be {expected}; got {value.shape}"
            )

        if pos in self._store:
            # Already stored: update tensors and promote to hot MRU.
            _, _, tier = self._store[pos]
            self._remove_from_tier_list(pos, tier)
        # Always insert into hot.
        self._store[pos] = (key.copy(), value.copy(), "hot")
        self._hot_list.append(pos)
        self._enforce_hot_capacity()
        self._stats.total_puts += 1

    def get(self, pos: int) -> tuple[np.ndarray, np.ndarray] | None:
        """Retrieve the KV tensors at *pos* and promote the token to hot tier.

        Parameters
        ----------
        pos : int
            Token position to look up.

        Returns
        -------
        tuple[np.ndarray, np.ndarray] or None
            ``(key, value)`` on hit; ``None`` on miss.
        """
        self._stats.total_gets += 1

        if pos not in self._store:
            self._stats.cold_misses += 1
            return None

        key, value, tier = self._store[pos]

        if tier == "hot":
            # Already in hot: move to MRU position.
            self._hot_list.remove(pos)
            self._hot_list.append(pos)
            self._stats.hot_hits += 1
        else:
            # Promote from warm or cold to hot.
            if tier == "warm":
                self._stats.warm_hits += 1
            else:  # cold
                self._stats.cold_hits += 1
            self._remove_from_tier_list(pos, tier)
            self._store[pos] = (key, value, "hot")
            self._hot_list.append(pos)
            self._enforce_hot_capacity()

        return key, value

    # ------------------------------------------------------------------ #
    # Properties                                                           #
    # ------------------------------------------------------------------ #

    @property
    def n_hot(self) -> int:
        """Number of token slots currently in the hot tier."""
        return len(self._hot_list)

    @property
    def n_warm(self) -> int:
        """Number of token slots currently in the warm tier."""
        return len(self._warm_list)

    @property
    def n_cold(self) -> int:
        """Number of token slots currently in the cold tier."""
        return len(self._cold_list)

    @property
    def stats(self) -> HierarchicalKVStats:
        """Current aggregate statistics."""
        return self._stats

    def __len__(self) -> int:
        return len(self._store)

    def __repr__(self) -> str:
        return (
            f"HierarchicalKVStore(hot={self.n_hot}/{self._cfg.hot_capacity}, "
            f"warm={self.n_warm}/{self._cfg.warm_capacity}, "
            f"cold={self.n_cold}/{self._cfg.cold_capacity})"
        )

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _remove_from_tier_list(self, pos: int, tier: str) -> None:
        """Remove *pos* from the list associated with *tier*."""
        if tier == "hot":
            self._hot_list.remove(pos)
        elif tier == "warm":
            self._warm_list.remove(pos)
        else:
            self._cold_list.remove(pos)

    def _enforce_hot_capacity(self) -> None:
        """Demote LRU hot tokens until hot tier is within capacity."""
        while len(self._hot_list) > self._cfg.hot_capacity:
            lru_pos = self._hot_list.pop(0)
            key, value, _ = self._store[lru_pos]
            self._store[lru_pos] = (key, value, "warm")
            self._warm_list.append(lru_pos)
            self._stats.total_demotions += 1
            self._enforce_warm_capacity()

    def _enforce_warm_capacity(self) -> None:
        """Demote LRU warm tokens until warm tier is within capacity."""
        while len(self._warm_list) > self._cfg.warm_capacity:
            lru_pos = self._warm_list.pop(0)
            key, value, _ = self._store[lru_pos]
            self._store[lru_pos] = (key, value, "cold")
            self._cold_list.append(lru_pos)
            self._stats.total_demotions += 1
            self._enforce_cold_capacity()

    def _enforce_cold_capacity(self) -> None:
        """Permanently evict LRU cold tokens until cold tier is within capacity."""
        while len(self._cold_list) > self._cfg.cold_capacity:
            lru_pos = self._cold_list.pop(0)
            del self._store[lru_pos]
            self._stats.total_evictions += 1
