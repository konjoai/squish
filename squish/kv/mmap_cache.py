"""
squish/kv/mmap_cache.py — Memory-mapped KV cache backend.

A drop-in storage tier that uses :func:`numpy.memmap` to hold K/V tensors on
disk instead of in resident RAM.  Pages are faulted in only when read, so a
32 K-token context that would otherwise OOM on a 16 GB machine can be served
from a (slow) SSD without process death.

Layout
------
Each layer occupies one directory ``<root>/L{layer_idx}/`` containing:

  k.bin       — uint8 file of (capacity, n_heads, head_dim) fp16 bytes
                (we store fp16 even when the parent cache is quantised; the
                memmap tier is a pre-quant residency tier, not a quant tier)
  v.bin       — same shape/dtype as k.bin
  meta.json   — {"capacity": int, "n_heads": int, "head_dim": int,
                 "dtype": "float16", "n_tokens": int}

Why fp16, not the parent's quant format?  Memory-mapping aims at *capacity*
(turning RAM into disk) — not at compression ratio.  Combining the two
(memmap + INT2 quant) is supported by writing the layer's quantised tensors
into the mmap file; the analyser provides ``store_bytes()`` for that case.

Thread-safety
-------------
Each layer has its own RLock.  Concurrent reads are safe; concurrent writes
must be serialised by the caller (the parent ``QuantizedKVCache`` already
holds a per-layer lock during ``append()`` — the mmap layer reuses it).
"""
from __future__ import annotations

import json
import logging
import shutil
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


_META_FILENAME = "meta.json"


@dataclass(frozen=True)
class MMapLayerMeta:
    capacity: int
    n_heads: int
    head_dim: int
    dtype: str
    n_tokens: int

    def to_dict(self) -> dict:
        return {
            "capacity": self.capacity,
            "n_heads": self.n_heads,
            "head_dim": self.head_dim,
            "dtype": self.dtype,
            "n_tokens": self.n_tokens,
        }


class MMapKVLayer:
    """A single layer's mmap-backed K and V buffers.

    Parameters
    ----------
    root        : directory for this layer's files; created if missing.
    capacity    : maximum tokens the buffer can hold.
    n_heads     : per-token head count.
    head_dim    : per-head dimension.
    dtype       : numpy dtype name (default ``"float16"``).

    Notes
    -----
    The buffers are allocated at construction (``capacity`` tokens × heads ×
    head_dim × itemsize bytes per file).  The OS only physically pages in
    written regions — sparse files keep the disk footprint to the working
    set, not the capacity.
    """

    def __init__(
        self,
        root: str | Path,
        capacity: int,
        n_heads: int,
        head_dim: int,
        dtype: str = "float16",
    ) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity must be > 0, got {capacity}")
        if n_heads <= 0 or head_dim <= 0:
            raise ValueError(
                f"n_heads and head_dim must be > 0, got {n_heads}, {head_dim}"
            )

        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)
        self._capacity = int(capacity)
        self._n_heads = int(n_heads)
        self._head_dim = int(head_dim)
        self._dtype = np.dtype(dtype)
        self._n_tokens = 0
        self._lock = threading.RLock()
        self._closed = False

        shape = (self._capacity, self._n_heads, self._head_dim)
        self._k_path = self._root / "k.bin"
        self._v_path = self._root / "v.bin"
        # Existing meta?  Reload n_tokens; otherwise start at 0.
        meta_path = self._root / _META_FILENAME
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
                if (meta.get("capacity") == self._capacity
                        and meta.get("n_heads") == self._n_heads
                        and meta.get("head_dim") == self._head_dim
                        and meta.get("dtype") == self._dtype.name):
                    self._n_tokens = int(meta.get("n_tokens", 0))
                else:
                    log.warning(
                        "mmap_cache: shape mismatch in %s; reinitialising",
                        self._root,
                    )
            except (json.JSONDecodeError, ValueError, OSError) as e:
                log.warning("mmap_cache: meta load failed for %s: %s", self._root, e)

        # meta.json can outlive the .bin files (crash / partial cleanup). If the
        # data is gone we create fresh zeroed memmaps below, so any restored
        # n_tokens is stale — reset it, otherwise get() would return all-zero K/V.
        bins_present = self._k_path.exists() and self._v_path.exists()
        if not bins_present and self._n_tokens > 0:
            log.warning(
                "mmap_cache: meta reports %d tokens but data files are missing in "
                "%s; resetting to empty", self._n_tokens, self._root,
            )
            self._n_tokens = 0

        # Use mode="w+" if no existing data; otherwise "r+" to preserve.
        mode = "r+" if bins_present else "w+"
        self._k = np.memmap(self._k_path, dtype=self._dtype, mode=mode, shape=shape)
        self._v = np.memmap(self._v_path, dtype=self._dtype, mode=mode, shape=shape)
        self._write_meta()

    # ── meta I/O ──────────────────────────────────────────────────────────

    def _write_meta(self) -> None:
        meta = MMapLayerMeta(
            capacity=self._capacity,
            n_heads=self._n_heads,
            head_dim=self._head_dim,
            dtype=self._dtype.name,
            n_tokens=self._n_tokens,
        )
        (self._root / _META_FILENAME).write_text(json.dumps(meta.to_dict()))

    @property
    def meta(self) -> MMapLayerMeta:
        return MMapLayerMeta(
            capacity=self._capacity,
            n_heads=self._n_heads,
            head_dim=self._head_dim,
            dtype=self._dtype.name,
            n_tokens=self._n_tokens,
        )

    @property
    def n_tokens(self) -> int:
        return self._n_tokens

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def root(self) -> Path:
        return self._root

    # ── core ops ──────────────────────────────────────────────────────────

    def append(self, key: np.ndarray, value: np.ndarray) -> None:
        """Append one token (shape ``(n_heads, head_dim)``).

        Raises :class:`OverflowError` when the buffer is full — the caller is
        expected to ``evict`` (drop the oldest) or to bump ``capacity``.
        """
        self._require_open()
        if key.shape != (self._n_heads, self._head_dim):
            raise ValueError(
                f"key shape {key.shape} != ({self._n_heads}, {self._head_dim})"
            )
        if value.shape != (self._n_heads, self._head_dim):
            raise ValueError(
                f"value shape {value.shape} != ({self._n_heads}, {self._head_dim})"
            )
        with self._lock:
            if self._n_tokens >= self._capacity:
                raise OverflowError(
                    f"mmap layer full at capacity={self._capacity}; "
                    f"call evict_oldest() or recreate with larger capacity"
                )
            slot = self._n_tokens
            self._k[slot] = key.astype(self._dtype, copy=False)
            self._v[slot] = value.astype(self._dtype, copy=False)
            self._n_tokens += 1
            self._write_meta()

    def get(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (key, value) for token ``idx`` as in-memory copies."""
        self._require_open()
        if not 0 <= idx < self._n_tokens:
            raise IndexError(
                f"idx {idx} out of range [0, {self._n_tokens})"
            )
        with self._lock:
            return (
                np.array(self._k[idx], copy=True),
                np.array(self._v[idx], copy=True),
            )

    def get_range(self, start: int, end: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (keys, values) for tokens ``[start, end)`` as copies."""
        self._require_open()
        if not 0 <= start <= end <= self._n_tokens:
            raise IndexError(
                f"range [{start}, {end}) invalid for n_tokens={self._n_tokens}"
            )
        with self._lock:
            return (
                np.array(self._k[start:end], copy=True),
                np.array(self._v[start:end], copy=True),
            )

    def evict_oldest(self, n: int = 1) -> int:
        """Drop the ``n`` oldest tokens by shifting later tokens left.

        Returns the number actually dropped (may be less than ``n`` when the
        buffer was nearly empty).  O(n_tokens × n_heads × head_dim) bytes
        copied — call sparingly.
        """
        self._require_open()
        if n < 0:
            raise ValueError(f"n must be ≥ 0, got {n}")
        with self._lock:
            n = min(n, self._n_tokens)
            if n == 0:
                return 0
            self._k[: self._n_tokens - n] = self._k[n : self._n_tokens]
            self._v[: self._n_tokens - n] = self._v[n : self._n_tokens]
            self._n_tokens -= n
            self._write_meta()
            return n

    def flush(self) -> None:
        """Flush dirty pages to disk."""
        self._require_open()
        with self._lock:
            self._k.flush()
            self._v.flush()
            self._write_meta()

    def close(self) -> None:
        """Flush and release the memmap handles.  Idempotent."""
        if self._closed:
            return
        with self._lock:
            try:
                self._k.flush()
                self._v.flush()
            except (OSError, ValueError) as e:
                log.warning("mmap_cache: flush during close failed: %s", e)
            # numpy.memmap exposes _mmap on the underlying buffer
            del self._k
            del self._v
            self._closed = True

    def __enter__(self) -> "MMapKVLayer":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("MMapKVLayer is closed")

    @property
    def disk_bytes(self) -> int:
        """Total bytes allocated on disk for this layer (k + v)."""
        per_file = (
            self._capacity * self._n_heads * self._head_dim * self._dtype.itemsize
        )
        return 2 * per_file


class MMapKVCache:
    """Multi-layer mmap-backed KV cache.

    A flat container over :class:`MMapKVLayer` — one directory per layer
    under the supplied root.  Designed as a *storage backend*, not as a
    full replacement for :class:`QuantizedKVCache`.

    Use ``MMapKVCache.delete(root)`` to reclaim disk after a session.
    """

    def __init__(
        self,
        root: str | Path,
        n_layers: int,
        capacity: int,
        n_heads: int,
        head_dim: int,
        dtype: str = "float16",
    ) -> None:
        if n_layers <= 0:
            raise ValueError(f"n_layers must be > 0, got {n_layers}")
        self._root = Path(root)
        self._n_layers = int(n_layers)
        self._layers = [
            MMapKVLayer(
                self._root / f"L{i}", capacity=capacity,
                n_heads=n_heads, head_dim=head_dim, dtype=dtype,
            )
            for i in range(self._n_layers)
        ]

    def __len__(self) -> int:
        return self._n_layers

    def __getitem__(self, idx: int) -> MMapKVLayer:
        return self._layers[idx]

    def __iter__(self):
        return iter(self._layers)

    @property
    def root(self) -> Path:
        return self._root

    @property
    def n_layers(self) -> int:
        return self._n_layers

    def append(
        self, layer_idx: int, key: np.ndarray, value: np.ndarray,
    ) -> None:
        self._layers[layer_idx].append(key, value)

    @property
    def disk_bytes(self) -> int:
        return sum(layer.disk_bytes for layer in self._layers)

    @property
    def n_tokens(self) -> int:
        """Tokens in layer 0 (representative — all layers grow in lockstep)."""
        return self._layers[0].n_tokens if self._layers else 0

    def flush(self) -> None:
        for layer in self._layers:
            layer.flush()

    def close(self) -> None:
        for layer in self._layers:
            layer.close()

    @classmethod
    def delete(cls, root: str | Path) -> None:
        """Remove the on-disk cache directory.  No-op if it doesn't exist."""
        p = Path(root)
        if p.exists():
            shutil.rmtree(p)
