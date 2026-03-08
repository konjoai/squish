"""
squish/pipo.py

PIPO — Pipelined Inference with Prefetched Offloading for consumer devices.

Inspired by:
  "PIPO: Efficient Pipeline-Parallel Inference with Structured Weight Offloading"
  (Microsoft Research, 2024)

Problem
-------
Running a 70B+ model on a machine with 24 GB VRAM requires weight offloading:
each transformer layer is streamed from CPU/NVMe → GPU just before use.  The
naive approach (load → compute → repeat) means the GPU sits idle while waiting
for PCIe transfer, cutting effective throughput in half.

PIPO Strategy
-------------
Overlap three pipelined stages with separate threads:

  ┌──────────┐    ┌──────────┐    ┌──────────────────┐
  │ Compute  │    │  Weight  │    │     KV-Manage    │
  │  thread  │    │  loader  │    │     thread       │
  │ (GPU /   │    │  thread  │    │ (evict old KV /  │
  │  numpy)  │    │ (PCIe /  │    │  pin next-layer  │
  │          │    │  mmap)   │    │  KV to CPU)      │
  └──────────┘    └──────────┘    └──────────────────┘

  While layer N computes, layer N+1 loads weights, layer N-1 KV is evicted.

Additional optimisation: **INT4 dequant bypass** — for small batch sizes
(< bypass_batch_threshold tokens), the standard INT4 → FP16 dequant → matmul
sequence is memory-bandwidth dominated.  PIPO caches dequantized weights in a
small LRU and skips re-dequant for frequently reused layers.

This module provides:
  ``PIPOConfig``          — all hyperparameters
  ``LayerWeightBuffer``   — double-buffer for layer weights
  ``INT4BypassKernel``    — cached dequant + matmul
  ``PIPOScheduler``       — 2-thread pipeline (compute + prefetch)

Usage::

    from squish.pipo import PIPOConfig, PIPOScheduler

    # weights_store: callable(layer_idx) → (weight_int4, scale) — your loader
    cfg  = PIPOConfig(n_prefetch_layers=2)
    sched = PIPOScheduler(cfg, weight_loader=weights_store, n_layers=32)

    output = np.zeros((1, 4096))
    for layer_idx in range(32):
        output = sched.run_layer(layer_idx, output)

    print(f"Throughput: {sched.throughput_tps:.1f} tokens/s")
"""

from __future__ import annotations

import queue
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

__all__ = [
    "PIPOConfig",
    "LayerWeightBuffer",
    "INT4BypassKernel",
    "PIPOScheduler",
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class PIPOConfig:
    """
    PIPO pipeline configuration.

    Parameters
    ----------
    n_prefetch_layers : int
        How many layers ahead to pre-load weights.
        1 = one layer ahead (minimum overlap); 2 = two layers.
    bypass_batch_threshold : int
        Batch-size threshold below which INT4 dequant-bypass (cached weights)
        is activated.  Set to 0 to always dequant; large value to always use
        cache.  Default 16.
    dequant_cache_size : int
        Maximum number of (layer, weight) pairs to keep in the dequant LRU
        cache.  Default 4.
    group_size : int
        INT4 quantization group size (number of values sharing one scale).
        Default 64.
    """
    n_prefetch_layers:      int = 1
    bypass_batch_threshold: int = 16
    dequant_cache_size:     int = 4
    group_size:             int = 64

    def __post_init__(self) -> None:
        if self.n_prefetch_layers < 1:
            raise ValueError(
                f"n_prefetch_layers must be ≥ 1, got {self.n_prefetch_layers}"
            )
        if self.bypass_batch_threshold < 0:
            raise ValueError(
                f"bypass_batch_threshold must be ≥ 0, got {self.bypass_batch_threshold}"
            )
        if self.dequant_cache_size < 1:
            raise ValueError(
                f"dequant_cache_size must be ≥ 1, got {self.dequant_cache_size}"
            )
        if self.group_size < 1:
            raise ValueError(f"group_size must be ≥ 1, got {self.group_size}")


# ---------------------------------------------------------------------------
# Layer weight buffer (double-buffer)
# ---------------------------------------------------------------------------

class LayerWeightBuffer:
    """
    Double-buffer for async weight pre-loading.

    Two slots alternate: while one slot is used for the active compute step,
    the other is being filled by the prefetch thread.

    Parameters
    ----------
    n_slots : int
        How many pre-load slots to maintain (default 2 = double-buffer).
    """

    def __init__(self, n_slots: int = 2) -> None:
        if n_slots < 1:
            raise ValueError(f"n_slots must be ≥ 1, got {n_slots}")
        self._n_slots  = n_slots
        self._slots: Dict[int, Optional[Tuple[np.ndarray, np.ndarray]]] = {}
        self._lock = threading.Lock()
        self._ready: Dict[int, threading.Event] = {}

    def begin_load(
        self,
        layer_idx: int,
        weight_int4: np.ndarray,
        scale: np.ndarray,
    ) -> None:
        """
        Store a pre-loaded weight (called from the prefetch thread).

        Parameters
        ----------
        layer_idx   : int
        weight_int4 : np.ndarray  uint8 packed INT4 weights
        scale       : np.ndarray  float32 per-group scales
        """
        with self._lock:
            self._slots[layer_idx] = (weight_int4, scale)
            ev = self._ready.setdefault(layer_idx, threading.Event())
        ev.set()

    def wait_ready(
        self,
        layer_idx: int,
        timeout: float = 60.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Block until ``layer_idx`` weights are ready and return them.

        Parameters
        ----------
        layer_idx : int
        timeout   : float — seconds to wait before raising TimeoutError

        Returns
        -------
        (weight_int4, scale)
        """
        with self._lock:
            ev = self._ready.setdefault(layer_idx, threading.Event())
        if not ev.wait(timeout=timeout):
            raise TimeoutError(f"Timed out waiting for layer {layer_idx} weights")
        with self._lock:
            return self._slots[layer_idx]  # type: ignore[return-value]

    def release(self, layer_idx: int) -> None:
        """Free the slot for ``layer_idx`` (called after compute finishes)."""
        with self._lock:
            self._slots.pop(layer_idx, None)
            self._ready.pop(layer_idx, None)

    def is_ready(self, layer_idx: int) -> bool:
        """Non-blocking check whether weights are available."""
        with self._lock:
            ev = self._ready.get(layer_idx)
        return ev is not None and ev.is_set()


# ---------------------------------------------------------------------------
# INT4 dequant bypass kernel
# ---------------------------------------------------------------------------

class INT4BypassKernel:
    """
    Cached INT4 dequantization + matrix multiply.

    For small batch sizes the PCIe transfer + dequant cost is amortised by
    keeping a small LRU cache of recently dequantized weight matrices.

    Parameters
    ----------
    cache_size : int — max cached layers
    group_size : int — INT4 group size
    """

    def __init__(self, cache_size: int = 4, group_size: int = 64) -> None:
        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self.cache_size = cache_size
        self.group_size = group_size

    def _dequantize(
        self,
        weight_int4: np.ndarray,
        scale: np.ndarray,
    ) -> np.ndarray:
        """
        Dequantize packed INT4 weights to float32.

        Parameters
        ----------
        weight_int4 : np.ndarray  uint8  shape (out_features, in_features // 2)
            Each byte packs two INT4 values (nibbles).
        scale : np.ndarray  float32  shape (n_groups,) or (out_features, n_groups)

        Returns
        -------
        np.ndarray  float32  shape (out_features, in_features)
        """
        # Unpack nibbles
        w = weight_int4.astype(np.int16)
        lo = (w & 0x0F).astype(np.int8)  # (out, in//2)
        hi = ((w >> 4) & 0x0F).astype(np.int8)
        # Interleave: [lo0, hi0, lo1, hi1, ...]
        out_features, half_in = lo.shape
        in_features = half_in * 2
        unpacked = np.empty((out_features, in_features), dtype=np.int8)
        unpacked[:, 0::2] = lo
        unpacked[:, 1::2] = hi
        # Sign-extend 4-bit: values 0..7 → 0..7, 8..15 → -8..-1
        unpacked = np.where(unpacked > 7, unpacked - 16, unpacked).astype(np.float32)
        # Apply scales per group
        gs = self.group_size
        n_groups = in_features // gs
        scales = np.asarray(scale, dtype=np.float32)
        if scales.ndim == 1:
            # (n_groups,) → broadcast over out_features
            scales = scales.reshape(1, n_groups)
        # scales: (out_features or 1, n_groups) → (out_features, in_features)
        scale_expanded = np.repeat(scales, gs, axis=-1)  # (*, in_features)
        if scale_expanded.shape[0] == 1:
            scale_expanded = np.broadcast_to(scale_expanded, (out_features, in_features))
        return unpacked * scale_expanded

    def matmul(
        self,
        x: np.ndarray,
        weight_int4: np.ndarray,
        scale: np.ndarray,
        layer_key: int,
        batch_size: int,
        bypass_threshold: int,
    ) -> np.ndarray:
        """
        Compute ``x @ W^T`` using dequantized weights, with optional caching.

        Parameters
        ----------
        x               : np.ndarray  float32  shape (batch, in_features)
        weight_int4     : np.ndarray  uint8    packed INT4
        scale           : np.ndarray  float32  per-group scales
        layer_key       : int         cache key for this layer
        batch_size      : int         current batch size (tokens)
        bypass_threshold: int         if batch_size < threshold → use cache

        Returns
        -------
        np.ndarray  float32  shape (batch, out_features)
        """
        use_cache = batch_size < bypass_threshold

        if use_cache and layer_key in self._cache:
            # Cache hit — avoid re-dequant
            self._cache.move_to_end(layer_key)
            w_f32 = self._cache[layer_key]
        else:
            w_f32 = self._dequantize(weight_int4, scale)
            if use_cache:
                # LRU eviction
                if len(self._cache) >= self.cache_size:
                    self._cache.popitem(last=False)
                self._cache[layer_key] = w_f32

        return np.asarray(x, dtype=np.float32) @ w_f32.T

    def clear_cache(self) -> None:
        """Evict all entries from the dequant LRU cache."""
        self._cache.clear()


# ---------------------------------------------------------------------------
# PIPOScheduler
# ---------------------------------------------------------------------------

class PIPOScheduler:
    """
    Coordinated 2-thread pipeline: compute thread + weight-prefetch thread.

    The scheduler runs a sequential decode-loop pattern:

        sched.run_layer(0, input)   # triggers prefetch of layer 1
        sched.run_layer(1, hidden)  # weights ready immediately (pre-loaded)
        ...

    Parameters
    ----------
    config        : PIPOConfig
    weight_loader : callable(layer_idx) → (weight_int4 np.ndarray, scale np.ndarray)
        Function provided by the caller that loads a layer's INT4 weight and
        per-group scales.  Called asynchronously from the prefetch thread.
    n_layers : int

    Notes
    -----
    ``run_layer`` is synchronous — it blocks until the current layer completes.
    Prefetch of the next layer runs in the background.
    """

    def __init__(
        self,
        config: PIPOConfig,
        weight_loader: Callable[[int], Tuple[np.ndarray, np.ndarray]],
        n_layers: int,
    ) -> None:
        self.config   = config
        self._loader  = weight_loader
        self.n_layers = n_layers

        self._buffer  = LayerWeightBuffer(n_slots=config.n_prefetch_layers + 1)
        self._kernel  = INT4BypassKernel(
            cache_size=config.dequant_cache_size,
            group_size=config.group_size,
        )
        self._total_tokens:  int   = 0
        self._total_time_s:  float = 0.0
        self._prefetch_thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------

    @property
    def throughput_tps(self) -> float:
        """Estimated tokens per second (measured over all run_layer calls)."""
        if self._total_time_s <= 0:
            return 0.0
        return self._total_tokens / self._total_time_s

    # ------------------------------------------------------------------

    def _prefetch_layer(self, layer_idx: int) -> None:
        """Called from background thread: load and buffer a layer's weights."""
        if layer_idx < 0 or layer_idx >= self.n_layers:
            return
        try:
            w, s = self._loader(layer_idx)
            self._buffer.begin_load(layer_idx, w, s)
        except Exception:
            pass  # Prefetch failures are non-fatal; run_layer will retry

    def prefetch_async(self, layer_idx: int) -> None:
        """Start async prefetch of a layer (non-blocking)."""
        t = threading.Thread(
            target=self._prefetch_layer,
            args=(layer_idx,),
            daemon=True,
        )
        t.start()
        self._prefetch_thread = t

    def run_layer(self, layer_idx: int, x: np.ndarray) -> np.ndarray:
        """
        Run forward computation for one transformer layer.

        Blocks until the layer's weights are ready (should already be
        pre-fetched), computes the matmul, then launches async prefetch
        of the next layer.

        Parameters
        ----------
        layer_idx : int
        x         : np.ndarray  float32  shape (batch, in_features)

        Returns
        -------
        np.ndarray  float32  shape (batch, out_features)
        """
        t0 = time.perf_counter()

        # Try to get pre-loaded weights; fall back to sync load
        if not self._buffer.is_ready(layer_idx):
            w, s = self._loader(layer_idx)
            self._buffer.begin_load(layer_idx, w, s)

        w_int4, scale = self._buffer.wait_ready(layer_idx)

        batch_size = x.shape[0] if x.ndim >= 2 else 1
        output = self._kernel.matmul(
            x, w_int4, scale, layer_idx,
            batch_size, self.config.bypass_batch_threshold,
        )

        self._buffer.release(layer_idx)

        # Launch async prefetch of next layer(s)
        for ahead in range(1, self.config.n_prefetch_layers + 1):
            next_idx = layer_idx + ahead
            if next_idx < self.n_layers and not self._buffer.is_ready(next_idx):
                self.prefetch_async(next_idx)

        elapsed = time.perf_counter() - t0
        self._total_tokens  += batch_size
        self._total_time_s  += elapsed

        return output
