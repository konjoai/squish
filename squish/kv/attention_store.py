"""squish/kv/attention_store.py

AttentionStore — Session-Scoped KV Persistence with Hot/Warm/SSD Tiering
(Sheng et al., ACL 2024 / arXiv:2403.19708).

Reference
---------
"FlexGen: High-Throughput Generative Inference of Large Language Models with
a Single GPU." Sheng et al., arXiv:2403.19708.  The AttentionStore module
implements the tiered KV-storage hierarchy described in Section 3.2.

Architecture
------------
* **Hot tier** (in-process Python dict) — fast-access, limited capacity.
* **Warm tier** (secondary dict, simulated host DRAM) — larger capacity, evicts
  to SSD when full.
* **SSD tier** (Python pickle → byte buffer) — unlimited but slow.

Eviction policy is LRU across hot and warm tiers.
"""

from __future__ import annotations

import io
import pickle
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

__all__ = [
    "AttentionStoreConfig",
    "AttentionStore",
]

_KV = Tuple[np.ndarray, np.ndarray]

# ── Config ────────────────────────────────────────────────────────────────────


@dataclass
class AttentionStoreConfig:
    """Configuration for :class:`AttentionStore`.

    Attributes:
        hot_capacity: Max number of (session, layer) slots in the hot tier.
        warm_capacity: Max slots in the warm tier.
        ssd_enabled: If True, overflow the warm tier to a byte buffer (SSD sim).
    """

    hot_capacity: int = 4096
    warm_capacity: int = 32768
    ssd_enabled: bool = False

    def __post_init__(self) -> None:
        if self.hot_capacity < 1:
            raise ValueError(f"hot_capacity must be ≥ 1; got {self.hot_capacity}")
        if self.warm_capacity < 1:
            raise ValueError(f"warm_capacity must be ≥ 1; got {self.warm_capacity}")


# ── AttentionStore ────────────────────────────────────────────────────────────


class AttentionStore:
    """Three-tier KV cache with session scoping.

    Example::

        cfg = AttentionStoreConfig(hot_capacity=8, warm_capacity=64)
        store = AttentionStore(cfg)
        rng = np.random.default_rng(0)
        K = rng.standard_normal((4, 64))
        V = rng.standard_normal((4, 64))
        store.store("sess-1", 0, K, V)
        K2, V2 = store.load("sess-1", 0)
    """

    def __init__(self, config: Optional[AttentionStoreConfig] = None) -> None:
        self.config = config or AttentionStoreConfig()
        self._hot: "OrderedDict[Tuple[str, int], _KV]" = OrderedDict()
        self._warm: "OrderedDict[Tuple[str, int], _KV]" = OrderedDict()
        self._ssd: Dict[Tuple[str, int], bytes] = {}
        self._hits_hot = 0
        self._hits_warm = 0
        self._hits_ssd = 0
        self._misses = 0

    # ── Write ─────────────────────────────────────────────────────────────────

    def store(
        self,
        session_id: str,
        layer_id: int,
        K: np.ndarray,
        V: np.ndarray,
    ) -> None:
        """Store KV tensors for ``(session_id, layer_id)``."""
        key = (session_id, layer_id)
        pair: _KV = (np.array(K, dtype=np.float32), np.array(V, dtype=np.float32))

        # Evict from hot → warm if hot is full.
        if len(self._hot) >= self.config.hot_capacity:
            self._evict_hot_to_warm()
        self._hot[key] = pair
        self._hot.move_to_end(key)

    # ── Read ──────────────────────────────────────────────────────────────────

    def load(self, session_id: str, layer_id: int) -> _KV:
        """Load KV tensors from the highest available tier.

        Raises:
            KeyError: If the entry is not cached anywhere.
        """
        key = (session_id, layer_id)

        if key in self._hot:
            self._hits_hot += 1
            self._hot.move_to_end(key)
            return self._hot[key]

        if key in self._warm:
            self._hits_warm += 1
            pair = self._warm.pop(key)
            # Promote to hot.
            if len(self._hot) >= self.config.hot_capacity:
                self._evict_hot_to_warm()
            self._hot[key] = pair
            self._hot.move_to_end(key)
            return pair

        if key in self._ssd:
            self._hits_ssd += 1
            pair = pickle.loads(self._ssd.pop(key))  # noqa: S301 — internal only
            if len(self._hot) >= self.config.hot_capacity:
                self._evict_hot_to_warm()
            self._hot[key] = pair
            self._hot.move_to_end(key)
            return pair

        self._misses += 1
        raise KeyError(f"No cached KV for session={session_id!r}, layer={layer_id}")

    # ── Eviction ──────────────────────────────────────────────────────────────

    def evict_session(self, session_id: str) -> int:
        """Remove all cached entries for ``session_id``; return count removed."""
        removed = 0
        for cache in (self._hot, self._warm):
            keys = [k for k in cache if k[0] == session_id]
            for k in keys:
                del cache[k]
                removed += 1
        ssd_keys = [k for k in self._ssd if k[0] == session_id]
        for k in ssd_keys:
            del self._ssd[k]
            removed += 1
        return removed

    # ── Stats ─────────────────────────────────────────────────────────────────

    def hit_rate(self) -> float:
        """Fraction of loads served from any tier (not a cold miss)."""
        total = self._hits_hot + self._hits_warm + self._hits_ssd + self._misses
        if total == 0:
            return 0.0
        return (self._hits_hot + self._hits_warm + self._hits_ssd) / total

    def tiers_used(self) -> Dict[str, int]:
        """Number of entries per tier."""
        return {
            "hot": len(self._hot),
            "warm": len(self._warm),
            "ssd": len(self._ssd),
        }

    def memory_bytes(self) -> int:
        """Approximate bytes consumed by hot + warm tier KV arrays."""
        total = 0
        for pair in list(self._hot.values()) + list(self._warm.values()):
            total += pair[0].nbytes + pair[1].nbytes
        return total

    # ── Internals ─────────────────────────────────────────────────────────────

    def _evict_hot_to_warm(self) -> None:
        """LRU-evict one entry from hot to warm (or SSD)."""
        oldest_key, oldest_val = self._hot.popitem(last=False)
        if len(self._warm) >= self.config.warm_capacity:
            self._evict_warm_to_ssd()
        self._warm[oldest_key] = oldest_val
        self._warm.move_to_end(oldest_key)

    def _evict_warm_to_ssd(self) -> None:
        """LRU-evict one entry from warm to SSD tier (or drop if disabled)."""
        oldest_key, oldest_val = self._warm.popitem(last=False)
        if self.config.ssd_enabled:
            self._ssd[oldest_key] = pickle.dumps(oldest_val)

    def __repr__(self) -> str:
        cfg = self.config
        return (
            f"AttentionStore(hot_capacity={cfg.hot_capacity}, "
            f"warm_capacity={cfg.warm_capacity}, "
            f"ssd_enabled={cfg.ssd_enabled})"
        )
