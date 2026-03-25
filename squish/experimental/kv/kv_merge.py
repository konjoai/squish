"""kv_merge.py — KVMerge: Shared Read-Only KV Prefix Slabs Across Requests

When multiple concurrent requests share an identical token prefix (e.g., the
same system prompt), their KV caches for those prefix tokens are deduplicated
into a single read-only ``SharedPrefixSlab``. Each request gets a lightweight
``RequestKVView`` that points to the shared slab for the prefix and owns a
private extension slab for its decode tokens.

Benefits:
  - A system prompt of 512 tokens prefilled once → served to N requests at
    O(1/N) KV memory overhead  (N-1 fewer prefill KV allocations)
  - TTFT for the 2nd-through-Nth request drops to the extension compute only
  - Reference-counted: shared slabs are freed when all views are released

Architecture:
  KVMergeRegistry   — global registry of active SharedPrefixSlabs, keyed by
                       prefix_hash (SHA-256 of token IDs).
  SharedPrefixSlab  — immutable KV tensor store for a unique prefix.
  RequestKVView     — per-request handle combining slab + private extension.

Usage:
    registry = KVMergeRegistry()
    # On request arrival:
    view = registry.get_or_create_view(prefix_ids, request_id="req-1")
    # Append newly decoded KV:
    view.append_private(layer_idx, k_vector, v_vector)
    # Read combined KV for attention:
    keys, values = view.read_kv(layer_idx)
    # On request completion:
    registry.release_view(request_id="req-1")
"""

from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _hash_prefix(token_ids: List[int]) -> str:
    """Compute a stable SHA-256 hex digest for a token-ID prefix."""
    raw = ",".join(str(t) for t in token_ids).encode()
    return hashlib.sha256(raw).hexdigest()


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class _LayerKV:
    """KV tensors for a single transformer layer in a slab."""
    keys: np.ndarray    # (prefix_len, n_heads, d_k)
    values: np.ndarray  # (prefix_len, n_heads, d_v)


class SharedPrefixSlab:
    """Immutable KV cache slab shared across requests with the same prefix.

    Thread-safe reference counting; the slab is destroyed when refcount==0.
    """

    def __init__(self, prefix_ids: List[int], prefix_hash: str) -> None:
        self.prefix_ids: List[int] = list(prefix_ids)
        self.prefix_hash: str = prefix_hash
        self.prefix_len: int = len(prefix_ids)
        self._layers: Dict[int, _LayerKV] = {}
        self._refcount: int = 0
        self._lock = threading.Lock()
        self._finalized: bool = False

    # ------------------------------------------------------------------
    # Reference counting
    # ------------------------------------------------------------------

    def acquire(self) -> None:
        with self._lock:
            self._refcount += 1

    def release(self) -> bool:
        """Decrement refcount. Returns True if the slab should be freed."""
        with self._lock:
            if self._refcount <= 0:
                raise ValueError(
                    "refcount underflow: cannot release un-acquired SharedPrefixSlab"
                )
            self._refcount -= 1
            return self._refcount <= 0

    @property
    def refcount(self) -> int:
        with self._lock:
            return self._refcount

    # ------------------------------------------------------------------
    # KV storage
    # ------------------------------------------------------------------

    def store_layer(
        self, layer_idx: int, keys: np.ndarray, values: np.ndarray
    ) -> None:
        """Store pre-computed KV for a layer (called once during prefill).

        Raises RuntimeError if the slab is already finalized.
        """
        with self._lock:
            if self._finalized:
                raise RuntimeError("Cannot write to a finalized SharedPrefixSlab")
            self._layers[layer_idx] = _LayerKV(
                keys=keys.copy(), values=values.copy()
            )

    def finalize(self) -> None:
        """Mark slab as immutable (no further writes allowed)."""
        with self._lock:
            self._finalized = True

    def read_layer(self, layer_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return (keys, values) for the given layer.

        Returns (None, None) if the layer has not been stored.
        """
        entry = self._layers.get(layer_idx)
        if entry is None:
            return (np.empty(0), np.empty(0))
        return entry.keys, entry.values

    @property
    def n_layers_stored(self) -> int:
        return len(self._layers)

    def __repr__(self) -> str:
        return (
            f"SharedPrefixSlab(hash={self.prefix_hash[:8]}…, "
            f"len={self.prefix_len}, refs={self.refcount}, "
            f"layers={self.n_layers_stored})"
        )


# ---------------------------------------------------------------------------
# Per-request view combining shared slab + private extension
# ---------------------------------------------------------------------------

class RequestKVView:
    """Per-request KV handle: shared prefix slab + private decode extension.

    The private extension is a simple list of (k, v) pairs appended one token
    at a time during the decode loop.
    """

    def __init__(
        self,
        slab: SharedPrefixSlab,
        request_id: str,
    ) -> None:
        self.slab = slab
        self.request_id = request_id
        slab.acquire()
        # private_layers[layer_idx] = list of (key_vec, val_vec)
        self._private: Dict[int, Tuple[List[np.ndarray], List[np.ndarray]]] = {}

    # ------------------------------------------------------------------
    # Append decode tokens
    # ------------------------------------------------------------------

    def append_private(
        self,
        layer_idx: int,
        key: np.ndarray,   # (n_heads, d_k) — single decode-step key
        value: np.ndarray, # (n_heads, d_v) — single decode-step value
    ) -> None:
        """Append one decode-step KV pair to the private extension."""
        if layer_idx not in self._private:
            self._private[layer_idx] = ([], [])
        self._private[layer_idx][0].append(key)
        self._private[layer_idx][1].append(value)

    # ------------------------------------------------------------------
    # Read combined KV
    # ------------------------------------------------------------------

    def read_kv(
        self, layer_idx: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return the full (prefix + private extension) KV for a layer.

        Returns:
            keys:   (prefix_len + private_len, n_heads, d_k)
            values: (prefix_len + private_len, n_heads, d_v)
        """
        prefix_k, prefix_v = self.slab.read_layer(layer_idx)
        priv = self._private.get(layer_idx)

        if priv is None or len(priv[0]) == 0:
            return prefix_k, prefix_v

        priv_k = np.stack(priv[0], axis=0)  # (priv_len, n_heads, d_k)
        priv_v = np.stack(priv[1], axis=0)

        if prefix_k.size == 0:
            return priv_k, priv_v

        return (
            np.concatenate([prefix_k, priv_k], axis=0),
            np.concatenate([prefix_v, priv_v], axis=0),
        )

    @property
    def total_kv_len(self) -> int:
        """Total number of KV positions (shared prefix + private tokens)."""
        priv_len = max(
            (len(v[0]) for v in self._private.values()), default=0
        )
        return self.slab.prefix_len + priv_len

    @property
    def private_len(self) -> int:
        return max((len(v[0]) for v in self._private.values()), default=0)

    def release(self) -> bool:
        """Release this view's reference to the slab.

        Returns True if the slab can now be freed.
        """
        self._private.clear()
        return self.slab.release()

    def __repr__(self) -> str:
        return (
            f"RequestKVView(id={self.request_id}, "
            f"prefix={self.slab.prefix_len}, "
            f"private={self.private_len})"
        )


# ---------------------------------------------------------------------------
# Global registry
# ---------------------------------------------------------------------------

@dataclass
class KVMergeStats:
    """Registry-level statistics."""
    total_views_created: int = 0
    total_views_released: int = 0
    total_slabs_created: int = 0
    total_slabs_freed: int = 0
    slab_hits: int = 0       # views that reused an existing slab
    slab_misses: int = 0     # views that required a new slab

    @property
    def hit_rate(self) -> float:
        total = self.slab_hits + self.slab_misses
        return self.slab_hits / total if total > 0 else 0.0


class KVMergeRegistry:
    """Global registry of SharedPrefixSlabs keyed by prefix hash.

    Thread-safe; intended to be a process-level singleton.
    """

    def __init__(self) -> None:
        self._slabs: Dict[str, SharedPrefixSlab] = {}
        self._views: Dict[str, RequestKVView] = {}
        self._lock = threading.Lock()
        self.stats = KVMergeStats()

    # ------------------------------------------------------------------
    # View lifecycle
    # ------------------------------------------------------------------

    def get_or_create_view(
        self,
        prefix_ids: List[int],
        request_id: str,
    ) -> RequestKVView:
        """Return a RequestKVView for the given prefix, reusing an existing
        SharedPrefixSlab if available.

        The caller is responsible for populating the slab's KV data via
        ``view.slab.store_layer()`` and calling ``view.slab.finalize()``
        before using the view for decode.
        """
        if not prefix_ids:
            raise ValueError("prefix_ids must be non-empty")
        h = _hash_prefix(prefix_ids)
        with self._lock:
            if h in self._slabs:
                slab = self._slabs[h]
                self.stats.slab_hits += 1
            else:
                slab = SharedPrefixSlab(prefix_ids=prefix_ids, prefix_hash=h)
                self._slabs[h] = slab
                self.stats.total_slabs_created += 1
                self.stats.slab_misses += 1

            view = RequestKVView(slab=slab, request_id=request_id)
            self._views[request_id] = view
            self.stats.total_views_created += 1
        return view

    def release_view(self, request_id: str) -> None:
        """Release a request's KV view; frees the slab if refcount drops to 0."""
        with self._lock:
            view = self._views.pop(request_id, None)
            if view is None:
                return
            should_free = view.release()
            if should_free:
                self._slabs.pop(view.slab.prefix_hash, None)
                self.stats.total_slabs_freed += 1
            self.stats.total_views_released += 1

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    @property
    def active_slab_count(self) -> int:
        with self._lock:
            return len(self._slabs)

    @property
    def active_view_count(self) -> int:
        with self._lock:
            return len(self._views)

    def get_view(self, request_id: str) -> Optional[RequestKVView]:
        with self._lock:
            return self._views.get(request_id)

    def __repr__(self) -> str:
        return (
            f"KVMergeRegistry(slabs={self.active_slab_count}, "
            f"views={self.active_view_count}, "
            f"hit_rate={self.stats.hit_rate:.1%})"
        )
