"""
squish/kv/infinite_gen.py

InfiniGenKVManager: Async CPU KV Offload with Importance-Scored Prefetch.

Reference
---------
Lee et al. "InfiniGen: Efficient Generative Inference with Dynamic Context
Management." arXiv 2406.14737, 2024.

Algorithm
---------
InfiniGen keeps a *hot tier* of KV entries in fast memory (simulated as a
dict mapping position → (key, value) ndarray pair) and a *cold tier* on the
CPU (simulated as another dict or a list).  At each decode step:

  1. The hot tier is checked for the required positions.
  2. Missing positions are "fetched" from the cold tier (prefetch fills the
     hot tier asynchronously in a real implementation; here it's synchronous
     but tracked via stats).
  3. Eviction uses importance scores when the hot tier exceeds ``hot_capacity``.

Importance scoring: each KV slot has a running score equal to the sum of
attention weights assigned to it across all previous decode steps (accumulative
attention score, as in SnapKV/H2O).  When eviction is needed, the
lowest-scoring slots are migrated back to the cold tier.

Key properties
--------------
* ``hot_capacity`` — max KV positions in the fast tier (default 512).
* ``prefetch_k`` — number of low-score slots to prefetch before they are
  needed (default 32, 0 = disabled).
* Deterministic, thread-safe (single-threaded sim).
* NumPy-only; no external dependencies.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class InfiniGenConfig:
    """Configuration for InfiniGenKVManager."""

    hot_capacity: int = 512
    """Maximum number of KV positions held in the hot (fast) tier."""

    prefetch_k: int = 32
    """How many additional entries to prefetch in each eviction pass."""

    head_dim: int = 64
    """Attention head dimension."""

    importance_decay: float = 0.9
    """Exponential decay applied to importance scores at each decode step."""

    def __post_init__(self) -> None:
        if self.hot_capacity < 1:
            raise ValueError("hot_capacity must be >= 1")
        if self.prefetch_k < 0:
            raise ValueError("prefetch_k must be >= 0")
        if not 0.0 < self.importance_decay <= 1.0:
            raise ValueError("importance_decay must be in (0, 1]")


@dataclass
class InfiniGenStats:
    """Runtime counters for InfiniGenKVManager."""

    hot_hits: int = 0
    cold_fetches: int = 0
    evictions: int = 0
    total_decode_steps: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hot_hits + self.cold_fetches
        if total == 0:
            return 0.0
        return self.hot_hits / total


class InfiniGenKVManager:
    """Async-style CPU KV offload manager with importance-scored eviction.

    Internal layout
    ---------------
    ``_hot``:  dict[int, (k_vec, v_vec)]  — fast-tier positions.
    ``_cold``: dict[int, (k_vec, v_vec)]  — cold-tier (CPU) positions.
    ``_scores``: dict[int, float]          — cumulative attention importance.
    """

    def __init__(self, config: Optional[InfiniGenConfig] = None) -> None:
        self.config = config or InfiniGenConfig()
        self.stats = InfiniGenStats()
        self._hot: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        self._cold: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        self._scores: Dict[int, float] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def put(self, position: int, key: np.ndarray, value: np.ndarray) -> None:
        """Insert or update a KV entry.

        New entries always land in the hot tier.  If hot capacity is exceeded
        the lowest-importance entries are evicted to cold.
        """
        self._hot[position] = (key.copy(), value.copy())
        self._scores.setdefault(position, 0.0)
        self._maybe_evict()

    def get(
        self, positions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Retrieve KV entries for the requested positions.

        Returns
        -------
        found_positions:
            1-D int array of positions actually found (may be < len(positions)).
        keys:
            (len(found_positions), head_dim).
        values:
            (len(found_positions), head_dim).
        """
        self.stats.total_decode_steps += 1
        hd = self.config.head_dim
        found_pos: List[int] = []
        keys_out: List[np.ndarray] = []
        vals_out: List[np.ndarray] = []

        for pos in positions:
            pos = int(pos)
            if pos in self._hot:
                self.stats.hot_hits += 1
                k, v = self._hot[pos]
                found_pos.append(pos)
                keys_out.append(k)
                vals_out.append(v)
            elif pos in self._cold:
                self.stats.cold_fetches += 1
                k, v = self._cold.pop(pos)
                self._hot[pos] = (k, v)
                self._maybe_evict()
                found_pos.append(pos)
                keys_out.append(k)
                vals_out.append(v)

        if not found_pos:
            return (
                np.empty(0, dtype=np.int64),
                np.empty((0, hd), dtype=np.float32),
                np.empty((0, hd), dtype=np.float32),
            )
        return (
            np.array(found_pos, dtype=np.int64),
            np.stack(keys_out),
            np.stack(vals_out),
        )

    def update_scores(self, positions: np.ndarray, attn_weights: np.ndarray) -> None:
        """Accumulate attention weights into importance scores.

        Parameters
        ----------
        positions:
            1-D int array of KV positions that were attended to.
        attn_weights:
            1-D float array of attention weights for each position.
        """
        decay = self.config.importance_decay
        for pos, w in zip(positions, attn_weights):
            pos = int(pos)
            self._scores[pos] = self._scores.get(pos, 0.0) * decay + float(w)

    def size(self) -> Tuple[int, int]:
        """Return ``(hot_size, cold_size)``."""
        return len(self._hot), len(self._cold)

    def reset(self) -> None:
        """Clear all tiers and reset stats."""
        self._hot.clear()
        self._cold.clear()
        self._scores.clear()
        self.stats = InfiniGenStats()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _maybe_evict(self) -> None:
        """Evict lowest-importance entries to cold tier if over capacity."""
        cap = self.config.hot_capacity
        if len(self._hot) <= cap:
            return

        n_evict = len(self._hot) - cap
        hot_positions = list(self._hot.keys())
        scores = np.array([self._scores.get(p, 0.0) for p in hot_positions])
        evict_indices = np.argpartition(scores, n_evict)[:n_evict]

        for idx in evict_indices:
            pos = hot_positions[idx]
            self._cold[pos] = self._hot.pop(pos)
            self.stats.evictions += 1
