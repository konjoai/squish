"""squish/kv/lckv_cache.py

Layer-Condensed KV Cache (LCKV) — cross-layer KV sharing to reduce KV memory.

Reference
---------
Zhang et al. "Layer-Condensed KV Cache for Efficient Inference of Large
Language Models." ACL 2024 (arXiv:2405.10637).

Algorithm
---------
In a standard transformer each layer maintains its own full K/V cache.
LCKV observes that upper layers can reuse the K/V computed at a small set of
**anchor layers** (the lowest ``n_anchor`` layers) without meaningful quality
degradation.

The policy:

* The first ``n_anchor`` layers compute and store their own KV tensors
  (anchor layers).
* Every subsequent layer re-uses the K/V tensors from the *nearest preceding
  anchor layer* below it, which is ``(layer_id % n_anchor)``-th anchor.
* This reduces the total KV buffer from ``n_layers × S × d`` to
  ``n_anchor × S × d``.

Key properties
--------------
* NumPy-only simulation; no GPU dependency.
* ``n_layers`` — total transformer layers.
* ``n_anchor`` — bottom-K anchor layers.
* ``head_dim``, ``n_heads`` — per-layer head size.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

__all__ = [
    "LCKVConfig",
    "LCKVCache",
]

# ── Config ────────────────────────────────────────────────────────────────────


@dataclass
class LCKVConfig:
    """Configuration for :class:`LCKVCache`.

    Attributes:
        n_layers: Total number of transformer layers.
        n_anchor: Number of bottom anchor layers that maintain their own KV.
        n_heads: Attention heads per layer.
        head_dim: Dimension per head.
        max_seq_len: Maximum sequence length (pre-allocated if > 0).
    """

    n_layers: int = 12
    n_anchor: int = 4
    n_heads: int = 8
    head_dim: int = 64
    max_seq_len: int = 0  # 0 = dynamic (no pre-allocation)

    def __post_init__(self) -> None:
        if self.n_layers < 1:
            raise ValueError(f"n_layers must be ≥ 1; got {self.n_layers}")
        if self.n_anchor < 1:
            raise ValueError(f"n_anchor must be ≥ 1; got {self.n_anchor}")
        if self.n_anchor > self.n_layers:
            raise ValueError(
                f"n_anchor ({self.n_anchor}) must be ≤ n_layers ({self.n_layers})"
            )
        if self.n_heads < 1:
            raise ValueError(f"n_heads must be ≥ 1; got {self.n_heads}")
        if self.head_dim < 1:
            raise ValueError(f"head_dim must be ≥ 1; got {self.head_dim}")


# ── Core class ─────────────────────────────────────────────────────────────────


class LCKVCache:
    """Layer-condensed KV cache that shares anchor-layer tensors.

    Example::

        cfg   = LCKVConfig(n_layers=8, n_anchor=2, n_heads=4, head_dim=16)
        cache = LCKVCache(cfg)

        # Anchor layers write their KV
        K0 = np.random.randn(4, 32, 16).astype(np.float32)
        V0 = np.random.randn(4, 32, 16).astype(np.float32)
        cache.write(layer_id=0, K=K0, V=V0)

        # Any layer reads back the nearest anchor KV
        K, V = cache.read(layer_id=5)
    """

    def __init__(self, config: Optional[LCKVConfig] = None) -> None:
        self.config = config or LCKVConfig()
        # Only anchor_layer slots are stored
        self._store: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

    # ── Write ─────────────────────────────────────────────────────────────────

    def write(self, layer_id: int, K: np.ndarray, V: np.ndarray) -> None:
        """Store KV tensors for an anchor layer.

        Args:
            layer_id: Layer index in [0, n_layers).
            K: ``(n_heads, S, head_dim)`` key tensor.
            V: ``(n_heads, S, head_dim)`` value tensor.

        Note: Non-anchor layers still call ``write``; the data is stored under
        the mapped anchor index so ``read`` always finds the right slot.
        """
        if not (0 <= layer_id < self.config.n_layers):
            raise ValueError(
                f"layer_id {layer_id} out of range [0, {self.config.n_layers})"
            )
        anchor_id = self._anchor_for(layer_id)
        self._store[anchor_id] = (
            np.asarray(K, dtype=np.float32),
            np.asarray(V, dtype=np.float32),
        )

    # ── Read ──────────────────────────────────────────────────────────────────

    def read(self, layer_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """Retrieve the KV tensors for a given layer.

        Anchor layers return their own KV.  Non-anchor layers return the KV of
        the nearest preceding anchor.

        Args:
            layer_id: Layer index.

        Returns:
            ``(K, V)`` each ``(n_heads, S, head_dim)``.

        Raises:
            KeyError: If no KV has been written for the resolved anchor yet.
        """
        if not (0 <= layer_id < self.config.n_layers):
            raise ValueError(
                f"layer_id {layer_id} out of range [0, {self.config.n_layers})"
            )
        anchor_id = self._anchor_for(layer_id)
        if anchor_id not in self._store:
            raise KeyError(
                f"No KV written for anchor layer {anchor_id} yet"
            )
        return self._store[anchor_id]

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _anchor_for(self, layer_id: int) -> int:
        """Map any layer to its nearest preceding anchor layer index."""
        return layer_id % self.config.n_anchor

    def is_anchor(self, layer_id: int) -> bool:
        """Return True if this layer is one of the bottom anchor layers."""
        return layer_id < self.config.n_anchor

    def memory_ratio(self) -> float:
        """Return the KV memory fraction relative to a full-layer cache."""
        return self.config.n_anchor / self.config.n_layers

    def clear(self) -> None:
        """Reset all stored KV tensors."""
        self._store.clear()

    def n_slots_filled(self) -> int:
        """Return number of anchor slots that have been written."""
        return len(self._store)

    def __repr__(self) -> str:
        return (
            f"LCKVCache(n_layers={self.config.n_layers}, "
            f"n_anchor={self.config.n_anchor}, "
            f"memory_ratio={self.memory_ratio():.2f}, "
            f"slots_filled={self.n_slots_filled()})"
        )
