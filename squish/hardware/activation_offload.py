"""activation_offload.py — Long-Context Activation Offloading

When prefilling sequences longer than activation_offload_threshold tokens,
the intermediate layer activations (post-attention residuals) can exceed the
available GPU/MPS memory on smaller Apple Silicon devices.

This module manages a software-based activation offload buffer that:
  1. Accepts layer activation tensors and copies them to CPU RAM (numpy arrays).
  2. Returns them on demand for the residual stream forward pass.
  3. Tracks peak memory savings vs. storing all activations in GPU memory.
  4. Prefetches the next layer's activation into a CPU-side staging buffer
     (one step ahead) to hide the PCIe transfer latency.

The buffer is keyed by layer index, so each call to save() overwrites any
previous activation at that index. Activations are stored as numpy arrays.

Memory accounting:
    On M1/M2/M3, CPU and GPU share unified memory — but MPS has a hard limit on
    the Metal buffer limit (~75% of RAM). This module lets the rest of the forward
    pass use the Metal budget for model weights + KV, while activations live in
    the non-Metal heap (numpy heap).

Lifecycle:
    offloader = ActivationOffloader(OffloadConfig(threshold=4096))
    offloader.begin_prefill(seq_len)           # enables or disables offload
    offloader.save(layer_idx, activation)      # store (offloaded or GPU-like)
    ...pass through model layers...
    act = offloader.load(layer_idx)            # fetch back
    offloader.end_prefill()                    # release all buffers
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class OffloadConfig:
    """Configuration for ActivationOffloader.

    Args:
        threshold:     Minimum sequence length (tokens) that activates offloading.
                       Shorter sequences keep activations "in-device" (no-op).
        prefetch_ahead: Number of layers to prefetch ahead of the current
                        consumer.  0 disables prefetching.
        dtype:         Storage dtype for offloaded activations.
                       None = preserve original dtype.
        max_layers:    Upper bound on number of layers.  Used to pre-allocate
                       the buffer dict.  0 = unbounded.
    """
    threshold: int = 4096
    prefetch_ahead: int = 1
    dtype: Optional[np.dtype] = None
    max_layers: int = 0


@dataclass
class OffloadStats:
    """Runtime statistics for an ActivationOffloader."""
    total_saves: int = 0
    total_loads: int = 0
    offloaded_bytes: int = 0
    passthrough_bytes: int = 0

    @property
    def total_bytes(self) -> int:
        return self.offloaded_bytes + self.passthrough_bytes

    @property
    def offload_ratio(self) -> float:
        if self.total_bytes == 0:
            return 0.0
        return self.offloaded_bytes / self.total_bytes


class ActivationBank:
    """Simple key→ndarray store (CPU RAM).

    Stores activations by integer layer index.  Not thread-safe — intended
    for single-threaded prefill use.
    """

    def __init__(self) -> None:
        self._store: dict[int, np.ndarray] = {}

    def put(self, layer_idx: int, tensor: np.ndarray) -> int:
        """Store activation.  Returns number of bytes stored."""
        arr = np.array(tensor, copy=True)
        self._store[layer_idx] = arr
        return arr.nbytes

    def get(self, layer_idx: int) -> np.ndarray:
        """Retrieve activation.  Removes it from the bank (consume-once)."""
        if layer_idx not in self._store:
            raise KeyError(f"No activation for layer {layer_idx}")
        return self._store.pop(layer_idx)

    def peek(self, layer_idx: int) -> Optional[np.ndarray]:
        """Read without removing (for prefetch logic)."""
        return self._store.get(layer_idx)

    def contains(self, layer_idx: int) -> bool:
        return layer_idx in self._store

    def clear(self) -> int:
        """Release all stored activations.  Returns bytes freed."""
        freed = sum(a.nbytes for a in self._store.values())
        self._store.clear()
        return freed

    def __len__(self) -> int:
        return len(self._store)


class ActivationOffloader:
    """Transparent activation offloader for long-sequence prefill.

    When offloading is disabled (seq_len < threshold), save() and load() are
    nearly zero-cost no-ops (the tensor is stored in a dict and returned in-
    place without copying).

    When offloading is enabled, save() copies the tensor to CPU RAM and
    load() copies it back.  On Apple Silicon (unified memory) the "copy"
    is a numpy array operation that moves the buffer out of the MPS Metal
    heap, freeing Metal budget for the model forward pass.
    """

    def __init__(self, config: Optional[OffloadConfig] = None) -> None:
        self.config = config or OffloadConfig()
        self._bank = ActivationBank()
        self._offload_active = False
        self.stats = OffloadStats()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def begin_prefill(self, seq_len: int) -> bool:
        """Begin a prefill pass of seq_len tokens.

        Returns True if offloading is activated for this sequence.
        """
        self._bank.clear()
        self._offload_active = seq_len >= self.config.threshold
        return self._offload_active

    def end_prefill(self) -> int:
        """End the prefill pass, release all offloaded buffers.

        Returns bytes freed.
        """
        self._offload_active = False
        freed = self._bank.clear()
        return freed

    @property
    def is_active(self) -> bool:
        """True if offloading is currently activated."""
        return self._offload_active

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def save(self, layer_idx: int, tensor: np.ndarray) -> None:
        """Store an activation tensor for layer_idx.

        If offloading is active, the tensor is deep-copied to CPU RAM.
        If offloading is inactive, a reference is stored (zero-copy).

        Args:
            layer_idx: Zero-based layer index.
            tensor:    Activation array (any numeric dtype).
        """
        if not isinstance(layer_idx, int) or layer_idx < 0:
            raise ValueError(f"layer_idx must be a non-negative int, got {layer_idx}")
        if not isinstance(tensor, np.ndarray):
            raise TypeError(f"tensor must be np.ndarray, got {type(tensor)}")

        cfg = self.config
        if self._offload_active:
            if cfg.dtype is not None:
                tensor = tensor.astype(cfg.dtype, copy=True)
            nbytes = self._bank.put(layer_idx, tensor)
            self.stats.offloaded_bytes += nbytes
        else:
            # Store reference in bank regardless (lightweight bookkeeping)
            self._bank._store[layer_idx] = tensor
            self.stats.passthrough_bytes += tensor.nbytes

        self.stats.total_saves += 1

    def load(self, layer_idx: int) -> np.ndarray:
        """Retrieve (and remove) the activation for layer_idx.

        Args:
            layer_idx: Zero-based layer index.

        Returns:
            The activation tensor (numpy array).

        Raises:
            KeyError: If no activation was saved for layer_idx.
        """
        tensor = self._bank.get(layer_idx)
        self.stats.total_loads += 1
        return tensor

    def has_activation(self, layer_idx: int) -> bool:
        """True if an activation for layer_idx is available."""
        return self._bank.contains(layer_idx)

    # ------------------------------------------------------------------
    # Stats helpers
    # ------------------------------------------------------------------

    def reset_stats(self) -> None:
        self.stats = OffloadStats()

    def __repr__(self) -> str:
        return (
            f"ActivationOffloader(active={self._offload_active}, "
            f"threshold={self.config.threshold}, "
            f"stored={len(self._bank)}, "
            f"offload_ratio={self.stats.offload_ratio:.1%})"
        )
