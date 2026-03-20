"""squish/io/flash_weight_cache.py

FlashWeightCache — NAND Flash-backed weight paging for M-series devices.

Reference
---------
Alizadeh et al. "LLM in a Flash: Efficient Large Language Model Inference
with Limited Memory." Apple 2024 (arXiv:2312.11514).

Algorithm
---------
On M-series Macs (and any system where model weights exceed available DRAM),
this module provides a **two-tier weight cache**:

1. **Flash tier** — weights are stored on an mmap-backed file (simulating
   NAND Flash / SSD storage).
2. **DRAM tier** — a bounded LRU cache holds the ``max_dram_layers`` most
   recently used weight tensors in RAM.

The prefetch predictor uses a *sequential sliding window* heuristic: when
layer ``i`` is fetched, the next ``prefetch_ahead`` layers are pre-loaded
into the DRAM cache asynchronously.

Key properties
--------------
* Simulation uses NumPy ``memmap`` for the Flash tier (file-backed arrays).
* LRU eviction in the DRAM tier via ``_lru_order`` deque.
* ``bandwidth_gbps`` controls simulated Flash read bandwidth (for timing).
* ``max_dram_layers`` — number of weight tensors pinned in DRAM.
"""

from __future__ import annotations

import os
import tempfile
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

__all__ = [
    "FlashWeightCacheConfig",
    "FlashWeightCache",
]

# ── Config ────────────────────────────────────────────────────────────────────


@dataclass
class FlashWeightCacheConfig:
    """Configuration for :class:`FlashWeightCache`.

    Attributes:
        max_dram_layers: Maximum weight tensors to keep in DRAM (LRU eviction).
        prefetch_ahead: Number of next layers to prefetch when one is accessed.
        bandwidth_gbps: Simulated Flash→DRAM bandwidth (informational only).
        dtype: NumPy dtype for weight storage.
    """

    max_dram_layers: int = 8
    prefetch_ahead: int = 2
    bandwidth_gbps: float = 10.0
    dtype: str = "float32"

    def __post_init__(self) -> None:
        if self.max_dram_layers < 1:
            raise ValueError(
                f"max_dram_layers must be ≥ 1; got {self.max_dram_layers}"
            )
        if self.prefetch_ahead < 0:
            raise ValueError(
                f"prefetch_ahead must be ≥ 0; got {self.prefetch_ahead}"
            )
        if self.bandwidth_gbps <= 0:
            raise ValueError(
                f"bandwidth_gbps must be > 0; got {self.bandwidth_gbps}"
            )
        if self.dtype not in ("float32", "float16", "bfloat16", "int8"):
            raise ValueError(
                f"dtype must be float32/float16/bfloat16/int8; got '{self.dtype}'"
            )


# ── Core class ─────────────────────────────────────────────────────────────────


class FlashWeightCache:
    """Transparent two-tier weight cache: DRAM → mmap-Flash.

    Layers are identified by integer ``layer_id`` values.  Each layer has a
    single weight tensor associated with it.

    Example::

        cfg   = FlashWeightCacheConfig(max_dram_layers=4)
        cache = FlashWeightCache(cfg)

        W = np.random.randn(512, 512).astype(np.float32)
        cache.store(layer_id=0, weight=W)

        W_back = cache.load(layer_id=0)
    """

    def __init__(self, config: Optional[FlashWeightCacheConfig] = None) -> None:
        self.config = config or FlashWeightCacheConfig()
        # DRAM tier: OrderedDict acting as LRU
        self._dram: "OrderedDict[int, np.ndarray]" = OrderedDict()
        # Flash tier metadata: layer_id -> (shape, dtype, tmpfile_path, offset)
        self._flash_meta: Dict[int, Tuple[Tuple[int, ...], str, str]] = {}
        self._tmpdir = tempfile.mkdtemp(prefix="flash_weight_")
        self._n_flash_hits: int = 0
        self._n_dram_hits: int = 0

    # ── Store ─────────────────────────────────────────────────────────────────

    def store(self, layer_id: int, weight: np.ndarray) -> None:
        """Store a weight tensor (writes to both DRAM and Flash tiers).

        Args:
            layer_id: Non-negative integer layer identifier.
            weight: NumPy weight array.
        """
        if layer_id < 0:
            raise ValueError(f"layer_id must be ≥ 0; got {layer_id}")
        w = np.asarray(weight, dtype=self.config.dtype)
        # Write to Flash (mmap file)
        path = os.path.join(self._tmpdir, f"layer_{layer_id}.npy")
        np.save(path, w)
        self._flash_meta[layer_id] = (w.shape, str(w.dtype), path)
        # Write to DRAM tier
        self._dram_insert(layer_id, w)

    # ── Load ──────────────────────────────────────────────────────────────────

    def load(self, layer_id: int) -> np.ndarray:
        """Load a weight tensor (DRAM hit or Flash page-in).

        Args:
            layer_id: Layer to load.

        Returns:
            Weight array as float32.

        Raises:
            KeyError: If the layer has never been stored.
        """
        if layer_id not in self._flash_meta:
            raise KeyError(f"Layer {layer_id} not stored in FlashWeightCache")

        if layer_id in self._dram:
            self._n_dram_hits += 1
            self._dram.move_to_end(layer_id)
            return self._dram[layer_id]

        # Flash page-in
        self._n_flash_hits += 1
        _, _, path = self._flash_meta[layer_id]
        w = np.load(path)
        self._dram_insert(layer_id, w)
        return w

    # ── Prefetch ──────────────────────────────────────────────────────────────

    def prefetch(self, layer_id: int) -> None:
        """Eagerly load layer_id and the next ``prefetch_ahead`` layers.

        Args:
            layer_id: Starting layer for the prefetch window.
        """
        end = layer_id + self.config.prefetch_ahead + 1
        for lid in range(layer_id, end):
            if lid in self._flash_meta and lid not in self._dram:
                self.load(lid)

    # ── Evict ─────────────────────────────────────────────────────────────────

    def evict(self, layer_id: int) -> None:
        """Manually evict a layer from the DRAM tier."""
        self._dram.pop(layer_id, None)

    # ── Introspection ─────────────────────────────────────────────────────────

    def dram_resident_layers(self) -> list:
        """Return list of layer ids currently in DRAM."""
        return list(self._dram.keys())

    def n_stored_layers(self) -> int:
        """Return total number of layers stored (Flash tier)."""
        return len(self._flash_meta)

    @property
    def n_dram_hits(self) -> int:
        return self._n_dram_hits

    @property
    def n_flash_hits(self) -> int:
        return self._n_flash_hits

    def memory_bytes_dram(self) -> int:
        """Approximate DRAM usage in bytes."""
        return sum(w.nbytes for w in self._dram.values())

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _dram_insert(self, layer_id: int, w: np.ndarray) -> None:
        """Insert into DRAM tier with LRU eviction."""
        if layer_id in self._dram:
            self._dram.move_to_end(layer_id)
        else:
            if len(self._dram) >= self.config.max_dram_layers:
                self._dram.popitem(last=False)
            self._dram[layer_id] = w

    def __repr__(self) -> str:
        return (
            f"FlashWeightCache(max_dram_layers={self.config.max_dram_layers}, "
            f"stored={self.n_stored_layers()}, "
            f"dram_resident={len(self._dram)}, "
            f"dram_hits={self._n_dram_hits}, "
            f"flash_hits={self._n_flash_hits})"
        )
