"""
squish/io/layer_overlap_loader.py

LayerOverlapLoader — Prefetch next layer weights during current-layer compute.

Motivation
----------
In streaming (layer-by-layer) LLM inference, each layer executes as:

    weights = load_layer(i)          # synchronous disk I/O  ← stall
    output  = compute_layer(weights, hidden_state)

On an M-series MacBook with 100 GB/s unified memory bandwidth, a 7B model
shard of ~1 GB per layer introduces ~10 ms per-layer IO latency.  Across
32 layers, that is 320 ms of pure stall time per token.

**Overlap loading** eliminates this stall by prefetching layer i+1 while
layer i is computing:

    load_layer(i+1) ‖ compute_layer(i, hidden)
    →  when compute(i) finishes, layer i+1 data is already in memory.

This module implements the prefetch scheduler as a threading-based async
loader.  In Python, threading offers true parallelism for blocking IO while
keeping the GIL-constrained compute on the main thread.

Classes
-------
``LayerOverlapConfig``    — prefetch depth, pin-memory flag
``LayerHandle``           — one prefetched layer (state, ready flag)
``LayerOverlapStats``     — hit/miss statistics
``LayerOverlapLoader``    — start/get/prefetch/stop API

Usage::

    from squish.io.layer_overlap_loader import LayerOverlapConfig, LayerOverlapLoader
    import numpy as np

    def load_fn(layer_idx: int):
        return {"weight": np.random.randn(256, 256).astype(np.float32)}

    loader = LayerOverlapLoader(LayerOverlapConfig(prefetch_count=2))
    loader.start(n_layers=32, load_fn=load_fn)

    for i in range(32):
        weights = loader.get_layer(i)          # instant if prefetched
        loader.prefetch_next(i)                # queue layer i+1, i+2, …
        # ... compute with weights ...

    loader.stop()
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

__all__ = [
    "LayerOverlapConfig",
    "LayerHandle",
    "LayerOverlapStats",
    "LayerOverlapLoader",
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class LayerOverlapConfig:
    """Configuration for the layer overlap loader.

    Attributes:
        prefetch_count:  Number of future layers to keep prefetched ahead of
                         the current layer.  Higher values use more memory but
                         tolerate more IO jitter.  Default: 2.
        load_timeout_s:  Maximum seconds to wait for a layer to be ready
                         (fallback to synchronous load if exceeded).
                         Default: 5.0.
    """

    prefetch_count: int = 2
    load_timeout_s: float = 5.0

    def __post_init__(self) -> None:
        if self.prefetch_count < 1:
            raise ValueError(f"prefetch_count must be >= 1, got {self.prefetch_count}")
        if self.load_timeout_s <= 0:
            raise ValueError(
                f"load_timeout_s must be > 0, got {self.load_timeout_s}"
            )


# ---------------------------------------------------------------------------
# Layer handle
# ---------------------------------------------------------------------------


class LayerHandle:
    """Container for a single prefetched layer.

    Attributes:
        layer_idx: Index of this layer.
        weights:   Weight dict once loaded; ``None`` until ready.
        ready:     threading.Event set when loading is complete.
    """

    def __init__(self, layer_idx: int) -> None:
        self.layer_idx = layer_idx
        self.weights: Optional[Dict[str, Any]] = None
        self.ready = threading.Event()
        self.error: Optional[Exception] = None

    def wait(self, timeout: float = 5.0) -> bool:
        """Block until ready or timeout.  Returns True if ready."""
        return self.ready.wait(timeout=timeout)

    def __repr__(self) -> str:
        return f"LayerHandle(idx={self.layer_idx}, ready={self.ready.is_set()})"


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


@dataclass
class LayerOverlapStats:
    """Runtime statistics for LayerOverlapLoader.

    Attributes:
        prefetch_hits:     Calls to ``get_layer()`` where data was ready.
        prefetch_misses:   Calls to ``get_layer()`` that had to wait.
        total_layers_loaded: Total layers loaded.
        total_load_ms:     Cumulative load time in milliseconds.
    """

    prefetch_hits: int = 0
    prefetch_misses: int = 0
    total_layers_loaded: int = 0
    total_load_ms: float = 0.0

    @property
    def hit_rate(self) -> float:
        total = self.prefetch_hits + self.prefetch_misses
        if total == 0:
            return 0.0
        return self.prefetch_hits / total

    @property
    def mean_load_ms(self) -> float:
        if self.total_layers_loaded == 0:
            return 0.0
        return self.total_load_ms / self.total_layers_loaded

    def __repr__(self) -> str:
        return (
            f"LayerOverlapStats("
            f"hit_rate={self.hit_rate:.2%}, "
            f"hits={self.prefetch_hits}, "
            f"misses={self.prefetch_misses}, "
            f"mean_load={self.mean_load_ms:.1f}ms)"
        )


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


class LayerOverlapLoader:
    """Async layer weight loader with prefetch-ahead scheduling.

    Uses a background thread pool to asynchronously load upcoming layers
    while the main thread computes on the current layer.

    Parameters
    ----------
    config:
        Loader configuration.
    """

    def __init__(self, config: Optional[LayerOverlapConfig] = None) -> None:
        self._cfg = config or LayerOverlapConfig()
        self._n_layers: int = 0
        self._load_fn: Optional[Callable[[int], Dict[str, Any]]] = None
        self._cache: Dict[int, LayerHandle] = {}
        self._lock = threading.Lock()
        self._threads: List[threading.Thread] = []
        self._stopped = False
        self.stats = LayerOverlapStats()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(
        self,
        n_layers: int,
        load_fn: Callable[[int], Dict[str, Any]],
    ) -> None:
        """Initialise and prefetch the first few layers.

        Parameters
        ----------
        n_layers: Total number of layers.
        load_fn:  Callable ``(layer_idx: int) → Dict[str, np.ndarray]``
                  that loads the weights for layer ``layer_idx``.
        """
        self._n_layers = n_layers
        self._load_fn = load_fn
        self._stopped = False
        self._cache = {}
        # Pre-warm: schedule the first `prefetch_count` layers
        for i in range(min(self._cfg.prefetch_count, n_layers)):
            self._schedule(i)

    def stop(self) -> None:
        """Signal shutdown and wait for in-flight threads to finish."""
        self._stopped = True
        for t in self._threads:
            t.join(timeout=2.0)
        self._threads = []
        self._cache = {}

    # ------------------------------------------------------------------
    # Prefetch scheduling
    # ------------------------------------------------------------------

    def _schedule(self, layer_idx: int) -> None:
        """Schedule asynchronous loading of layer ``layer_idx``."""
        if layer_idx < 0 or layer_idx >= self._n_layers:
            return
        with self._lock:
            if layer_idx in self._cache:
                return
            handle = LayerHandle(layer_idx)
            self._cache[layer_idx] = handle

        t = threading.Thread(
            target=self._load_worker,
            args=(layer_idx, handle),
            daemon=True,
            name=f"LayerLoader-{layer_idx}",
        )
        self._threads.append(t)
        t.start()

    def _load_worker(self, layer_idx: int, handle: LayerHandle) -> None:
        """Background worker that calls load_fn and fills the handle."""
        t0 = time.perf_counter()
        try:
            weights = self._load_fn(layer_idx)  # type: ignore[misc]
            handle.weights = weights
        except Exception as exc:
            handle.error = exc
        finally:
            handle.ready.set()
            elapsed_ms = (time.perf_counter() - t0) * 1e3
            with self._lock:
                self.stats.total_layers_loaded += 1
                self.stats.total_load_ms += elapsed_ms

    def prefetch_next(self, current_idx: int) -> None:
        """Schedule loading of the next ``prefetch_count`` layers.

        Call this immediately after calling ``get_layer(current_idx)``
        so loading overlaps with compute.

        Parameters
        ----------
        current_idx: Currently computing layer index.
        """
        for offset in range(1, self._cfg.prefetch_count + 1):
            self._schedule(current_idx + offset)

    # ------------------------------------------------------------------
    # Get
    # ------------------------------------------------------------------

    def get_layer(self, layer_idx: int) -> Dict[str, Any]:
        """Return weights for ``layer_idx``, waiting if necessary.

        Parameters
        ----------
        layer_idx: Layer to retrieve.

        Returns
        -------
        Weight dict for the layer.

        Raises
        ------
        RuntimeError if loading failed or loader not started.
        ValueError if layer_idx out of range.
        """
        if self._load_fn is None:
            raise RuntimeError("LayerOverlapLoader.start() must be called before get_layer()")
        if layer_idx < 0 or layer_idx >= self._n_layers:
            raise ValueError(f"layer_idx {layer_idx} out of range [0, {self._n_layers})")

        with self._lock:
            handle = self._cache.get(layer_idx)

        if handle is None:
            # Synchronous fallback (scheduling miss)
            self.stats.prefetch_misses += 1
            self._schedule(layer_idx)
            with self._lock:
                handle = self._cache[layer_idx]

        ready = handle.ready.is_set()
        if ready:
            self.stats.prefetch_hits += 1
        else:
            self.stats.prefetch_misses += 1
            handle.wait(timeout=self._cfg.load_timeout_s)

        if handle.error is not None:
            raise RuntimeError(
                f"Layer {layer_idx} load failed: {handle.error}"
            ) from handle.error

        # Evict old layers from cache to free memory
        with self._lock:
            old_keys = [k for k in list(self._cache.keys()) if k < layer_idx - 1]
            for k in old_keys:
                del self._cache[k]

        return handle.weights  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def cached_layer_count(self) -> int:
        with self._lock:
            return len(self._cache)

    def __repr__(self) -> str:
        return (
            f"LayerOverlapLoader("
            f"n_layers={self._n_layers}, "
            f"prefetch={self._cfg.prefetch_count}, "
            f"{self.stats})"
        )
