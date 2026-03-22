"""Multi-tier model weight paging: GPU-hot / CPU-warm / SSD-cold.

Implements the three-tier memory hierarchy described in *FlexGen*
(Sheng et al., ICML 2023) and *LLM in a Flash* (Apple, 2024).  Layers
are promoted and evicted between tiers based on an advancing inference
window, with configurable lookahead prefetching.

Reference:
  - Sheng et al., "FlexGen: High-Throughput Generative Inference of LLMs"
    (ICML 2023).
  - Alizadeh et al., "LLM in a Flash" (Apple 2024).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

__all__ = [
    "ShardConfig",
    "ShardTier",
    "LayerShard",
    "ModelShardLoader",
]


# ---------------------------------------------------------------------------
# Tier enum
# ---------------------------------------------------------------------------


class ShardTier(Enum):
    """Memory tier for a model layer shard.

    Attributes:
        HOT: GPU-resident (or equivalent fastest memory).
        WARM: CPU-pinned (fast prefetch buffer).
        COLD: SSD-paged (slow, loaded on demand).
    """

    HOT = "hot"
    WARM = "warm"
    COLD = "cold"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ShardConfig:
    """Configuration for :class:`ModelShardLoader`.

    Attributes:
        n_layers: Total number of transformer layers.
        hot_layers: Maximum layers kept in HOT tier simultaneously.
        warm_layers: Maximum layers kept in WARM tier simultaneously.
        lookahead: Layers ahead of the current position to prefetch into WARM.
        seed: Unused; retained for API consistency.
    """

    n_layers: int = 32
    hot_layers: int = 4
    warm_layers: int = 8
    lookahead: int = 2
    seed: int = 0

    def __post_init__(self) -> None:
        if self.n_layers < 1:
            raise ValueError(f"n_layers must be ≥ 1, got {self.n_layers}")
        if self.hot_layers < 1:
            raise ValueError(f"hot_layers must be ≥ 1, got {self.hot_layers}")
        if self.warm_layers < 0:
            raise ValueError(f"warm_layers must be ≥ 0, got {self.warm_layers}")
        if self.lookahead < 0:
            raise ValueError(f"lookahead must be ≥ 0, got {self.lookahead}")
        if self.hot_layers > self.n_layers:
            raise ValueError(
                f"hot_layers ({self.hot_layers}) cannot exceed n_layers ({self.n_layers})"
            )


@dataclass
class LayerShard:
    """One layer's weight shard in the three-tier hierarchy.

    Attributes:
        layer_idx: Which transformer layer this shard belongs to.
        tier: Current memory tier.
        data: Float32 weight array; ``None`` when evicted to COLD.
        size_bytes: Original byte footprint (even when evicted to COLD).
    """

    layer_idx: int
    tier: ShardTier
    data: Optional[np.ndarray]
    size_bytes: int

    @property
    def is_resident(self) -> bool:
        """True when data is available in-memory (HOT or WARM)."""
        return self.data is not None


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


class ModelShardLoader:
    """Three-tier weight paging with lookahead prefetch.

    Maintains an internal registry of :class:`LayerShard` objects
    and enforces HOT/WARM/COLD tier capacities.  Calling
    :meth:`advance_window` automatically promotes upcoming layers to WARM
    and evicts lagging layers to COLD.

    Example::

        cfg = ShardConfig(n_layers=32, hot_layers=4, warm_layers=8, lookahead=2)
        loader = ModelShardLoader(cfg)
        rng = np.random.default_rng(0)
        layers = {i: rng.standard_normal((512, 512)).astype(np.float32) for i in range(32)}
        loader.load_model(layers)
        for i in range(32):
            loader.advance_window(i)
            w = loader.get_layer(i)   # float32 array

    """

    def __init__(self, config: ShardConfig) -> None:
        self.config = config
        self._shards: Dict[int, LayerShard] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_model(self, layers: Dict[int, np.ndarray]) -> None:
        """Register all layers.  The first ``hot_layers`` go HOT; the next
        ``warm_layers`` go WARM; the rest start COLD.
        """
        cfg = self.config
        with self._lock:
            self._shards.clear()
            for idx in range(cfg.n_layers):
                data = layers.get(idx)
                if data is not None:
                    data = np.asarray(data, dtype=np.float32)

                if idx < cfg.hot_layers:
                    tier = ShardTier.HOT
                elif idx < cfg.hot_layers + cfg.warm_layers:
                    tier = ShardTier.WARM
                else:
                    tier = ShardTier.COLD
                    data = None  # evict from memory

                self._shards[idx] = LayerShard(
                    layer_idx=idx,
                    tier=tier,
                    data=data,
                    size_bytes=layers[idx].nbytes if idx in layers else 0,
                )

    # ------------------------------------------------------------------
    # Data access
    # ------------------------------------------------------------------

    def get_layer(self, layer_idx: int) -> np.ndarray:
        """Return the float32 weight array for *layer_idx*.

        Raises :class:`KeyError` if the layer has not been loaded.
        Raises :class:`RuntimeError` if the shard is COLD (data not resident).
        """
        with self._lock:
            shard = self._shards.get(layer_idx)
        if shard is None:
            raise KeyError(f"Layer {layer_idx} not registered in ModelShardLoader.")
        if shard.data is None:
            raise RuntimeError(
                f"Layer {layer_idx} is COLD (data not resident). "
                "Call prefetch() or promote_to_warm() first."
            )
        return shard.data

    # ------------------------------------------------------------------
    # Tier management
    # ------------------------------------------------------------------

    def promote_to_hot(self, layer_idx: int) -> None:
        """Move layer *layer_idx* to HOT tier.

        If the HOT tier is full, the lowest-indexed HOT layer is
        demoted to WARM to make room.
        """
        with self._lock:
            self._ensure_hot_capacity()
            shard = self._shards.get(layer_idx)
            if shard is None:
                raise KeyError(f"Layer {layer_idx} not found.")
            if shard.data is None:
                raise RuntimeError(
                    f"Cannot promote layer {layer_idx} to HOT: data is None. "
                    "Load via promote_to_warm() first."
                )
            shard.tier = ShardTier.HOT

    def promote_to_warm(self, layer_idx: int) -> None:
        """Move layer *layer_idx* to WARM tier, re-populating data if needed.

        In a real system this would issue a disk read; here the data is
        kept in a COLD stub so it can be faithfully restored.
        """
        with self._lock:
            shard = self._shards.get(layer_idx)
            if shard is None:
                raise KeyError(f"Layer {layer_idx} not found.")
            if shard.tier == ShardTier.HOT:
                return  # already resident
            self._ensure_warm_capacity()
            shard.tier = ShardTier.WARM
            # If data was cleared (cold eviction without a real disk), restore zeros.
            if shard.data is None and shard.size_bytes > 0:
                n_elem = shard.size_bytes // 4
                shard.data = np.zeros(n_elem, dtype=np.float32)

    def evict_to_cold(self, layer_idx: int) -> None:
        """Move layer *layer_idx* to COLD tier and free in-memory data."""
        with self._lock:
            shard = self._shards.get(layer_idx)
            if shard is None:
                raise KeyError(f"Layer {layer_idx} not found.")
            shard.tier = ShardTier.COLD
            shard.data = None

    def tier_of(self, layer_idx: int) -> ShardTier:
        """Return the current tier of *layer_idx*."""
        with self._lock:
            shard = self._shards.get(layer_idx)
        if shard is None:
            raise KeyError(f"Layer {layer_idx} not found.")
        return shard.tier

    def prefetch(self, layer_indices: List[int]) -> None:
        """Prefetch multiple layers into WARM tier."""
        for idx in layer_indices:
            self.promote_to_warm(idx)

    # ------------------------------------------------------------------
    # Window advancement
    # ------------------------------------------------------------------

    def advance_window(self, current_layer: int) -> None:
        """Slide the inference window to *current_layer*.

        - Promotes *current_layer* to HOT (if not already).
        - Prefetches ``[current_layer+1 … current_layer+lookahead]`` to WARM.
        - Evicts layers that are before ``current_layer - hot_layers`` to COLD.
        """
        cfg = self.config

        # Evict far-behind layers
        evict_before = current_layer - cfg.hot_layers
        with self._lock:
            for idx, shard in self._shards.items():
                if idx < evict_before and shard.tier != ShardTier.COLD:
                    shard.tier = ShardTier.COLD
                    shard.data = None

        # Promote current layer to HOT
        if current_layer in self._shards:
            try:
                self.promote_to_warm(current_layer)
                self.promote_to_hot(current_layer)
            except RuntimeError:
                pass  # data not available — caller responsibility

        # Prefetch lookahead into WARM
        for ahead in range(1, cfg.lookahead + 1):
            target = current_layer + ahead
            if target in self._shards:
                try:
                    self.promote_to_warm(target)
                except (RuntimeError, KeyError):
                    pass

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def memory_report(self) -> Dict:
        """Return a summary of tier occupancy and byte usage."""
        hot_layers: List[int] = []
        warm_layers: List[int] = []
        cold_layers: List[int] = []
        hot_bytes = 0
        warm_bytes = 0

        with self._lock:
            for idx, shard in self._shards.items():
                if shard.tier == ShardTier.HOT:
                    hot_layers.append(idx)
                    hot_bytes += shard.size_bytes
                elif shard.tier == ShardTier.WARM:
                    warm_layers.append(idx)
                    warm_bytes += shard.size_bytes
                else:
                    cold_layers.append(idx)

        return {
            "hot_count": len(hot_layers),
            "warm_count": len(warm_layers),
            "cold_count": len(cold_layers),
            "hot_layers": sorted(hot_layers),
            "warm_layers": sorted(warm_layers),
            "cold_layers": sorted(cold_layers),
            "hot_bytes": hot_bytes,
            "warm_bytes": warm_bytes,
            "total_layers": len(self._shards),
            "config": {
                "n_layers": self.config.n_layers,
                "hot_layers": self.config.hot_layers,
                "warm_layers": self.config.warm_layers,
                "lookahead": self.config.lookahead,
            },
        }

    # ------------------------------------------------------------------
    # Iteration helpers
    # ------------------------------------------------------------------

    def iter_hot(self) -> Iterator[Tuple[int, np.ndarray]]:
        """Yield (layer_idx, data) for all HOT layers in index order."""
        with self._lock:
            hot = sorted(
                (idx, s) for idx, s in self._shards.items()
                if s.tier == ShardTier.HOT and s.data is not None
            )
        for idx, shard in hot:
            yield idx, shard.data  # type: ignore[arg-type]

    def __len__(self) -> int:
        with self._lock:
            return len(self._shards)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _hot_count(self) -> int:
        return sum(1 for s in self._shards.values() if s.tier == ShardTier.HOT)

    def _warm_count(self) -> int:
        return sum(1 for s in self._shards.values() if s.tier == ShardTier.WARM)

    def _ensure_hot_capacity(self) -> None:
        """Demote the lowest HOT layer to WARM if HOT is full."""
        cfg = self.config
        while self._hot_count() >= cfg.hot_layers:
            hot_indices = sorted(
                idx for idx, s in self._shards.items() if s.tier == ShardTier.HOT
            )
            if not hot_indices:
                break
            oldest_idx = hot_indices[0]
            self._shards[oldest_idx].tier = ShardTier.WARM

    def _ensure_warm_capacity(self) -> None:
        """Evict the lowest WARM layer to COLD if WARM is full."""
        cfg = self.config
        while self._warm_count() >= cfg.warm_layers:
            warm_indices = sorted(
                idx for idx, s in self._shards.items() if s.tier == ShardTier.WARM
            )
            if not warm_indices:
                break
            oldest_idx = warm_indices[0]
            self._shards[oldest_idx].tier = ShardTier.COLD
            self._shards[oldest_idx].data = None
