"""squish/attention/cla_share.py

CLAShareAttention — Cross-Layer Attention Sharing: adjacent layers share
K/V projections to halve KV memory with minimal quality impact
(Brandon et al., ACL Findings 2024 / arXiv:2405.12981).

Reference
---------
"CLA: Cross-Layer Attention Sharing for Large Language Models."
Brandon et al., ACL Findings 2024 (arXiv:2405.12981).

Algorithm
---------
In standard transformers every layer l has its own K_l and V_l projections.
CLA groups consecutive layers into sharing windows of size ``stride`` and
forces layers within a window to reuse the KV of the first layer in that
window:

    K_i = K_{floor(i/stride) * stride}    for i in the sharing window
    V_i = V_{floor(i/stride) * stride}

Benefit:
* At stride=2 → 50% KV parameter reduction.
* At stride=4 → 75% KV parameter reduction.
* Q projections remain independent per layer.

This simulation:
* Stores a shared KV cache keyed by the anchor layer index.
* ``compute_kv(layer_id, K, V)`` stores only if this layer is an anchor.
* ``get_kv(layer_id)`` returns the anchor KV for layer_id.

Key properties
--------------
* NumPy-only simulation; no GPU dependency.
* ``sharing_stride`` — width of the sharing window (≥ 1).
* ``n_layers``       — total transformer layers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

__all__ = [
    "CLAShareConfig",
    "CLAShareAttention",
]

# ── Config ────────────────────────────────────────────────────────────────────


@dataclass
class CLAShareConfig:
    """Configuration for :class:`CLAShareAttention`.

    Attributes:
        sharing_stride: Number of consecutive layers that share a single
            KV projection. stride=1 means no sharing (each layer is its own
            anchor). stride=2 → 50% KV reduction.
        n_layers: Total number of transformer layers.
        n_heads: Attention heads per layer.
        head_dim: Dimension per head.
    """

    sharing_stride: int = 2
    n_layers: int = 12
    n_heads: int = 8
    head_dim: int = 64

    def __post_init__(self) -> None:
        if self.sharing_stride < 1:
            raise ValueError(f"sharing_stride must be ≥ 1; got {self.sharing_stride}")
        if self.n_layers < 1:
            raise ValueError(f"n_layers must be ≥ 1; got {self.n_layers}")
        if self.n_heads < 1:
            raise ValueError(f"n_heads must be ≥ 1; got {self.n_heads}")
        if self.head_dim < 1:
            raise ValueError(f"head_dim must be ≥ 1; got {self.head_dim}")

    @property
    def share_every(self) -> int:  # server.py compatibility alias
        return self.sharing_stride


# ── CLA ───────────────────────────────────────────────────────────────────────


class CLAShareAttention:
    """Cross-layer K/V sharing manager.

    Example::

        cfg = CLAShareConfig(sharing_stride=2, n_layers=8, n_heads=4, head_dim=8)
        cla = CLAShareAttention(cfg)

        K = np.random.randn(4, 16, 8).astype(np.float32)
        V = np.random.randn(4, 16, 8).astype(np.float32)

        for layer_id in range(8):
            cla.compute_kv(layer_id, K, V)  # only stores at anchor layers
            K_l, V_l = cla.get_kv(layer_id)
            out = softmax_attn(Q_l, K_l, V_l)
    """

    def __init__(self, config: Optional[CLAShareConfig] = None) -> None:
        self.config = config or CLAShareConfig()
        self._kv_store: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

    # ── Public API ────────────────────────────────────────────────────────────

    def anchor_layer(self, layer_id: int) -> int:
        """Return the anchor (first) layer of the sharing window containing *layer_id*.

        Args:
            layer_id: Layer index in [0, n_layers).

        Returns:
            Anchor layer index.
        """
        self._check_layer(layer_id)
        return (layer_id // self.config.sharing_stride) * self.config.sharing_stride

    def is_anchor(self, layer_id: int) -> bool:
        """Return True if *layer_id* is the anchor (first) layer in its window."""
        return layer_id == self.anchor_layer(layer_id)

    def compute_kv(
        self,
        layer_id: int,
        K: np.ndarray,
        V: np.ndarray,
    ) -> None:
        """Store K/V if *layer_id* is an anchor layer; no-op otherwise.

        Args:
            layer_id: Layer index.
            K: ``(n_heads, S, head_dim)`` key tensor.
            V: ``(n_heads, S, head_dim)`` value tensor.
        """
        self._check_layer(layer_id)
        if self.is_anchor(layer_id):
            self._kv_store[layer_id] = (
                np.asarray(K, dtype=np.float32),
                np.asarray(V, dtype=np.float32),
            )

    def get_kv(
        self, layer_id: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Retrieve the K/V tensors shared by the window containing *layer_id*.

        Args:
            layer_id: Layer index.

        Returns:
            ``(K, V)`` from the anchor layer of this window.

        Raises:
            KeyError: If the anchor for this layer has not been computed yet.
        """
        self._check_layer(layer_id)
        anchor = self.anchor_layer(layer_id)
        if anchor not in self._kv_store:
            raise KeyError(
                f"KV for anchor layer {anchor} not yet computed. "
                f"Call compute_kv(layer_id={anchor}, ...) first."
            )
        return self._kv_store[anchor]

    def memory_ratio(self) -> float:
        """Fraction of KV memory used relative to an unshared model.

        Returns:
            ``1 / sharing_stride`` (fraction of full KV budget).
        """
        return 1.0 / self.config.sharing_stride

    def n_anchor_layers(self) -> int:
        """Number of unique KV projections (anchor layers)."""
        import math
        return math.ceil(self.config.n_layers / self.config.sharing_stride)

    def clear(self) -> None:
        """Clear all stored KV tensors."""
        self._kv_store.clear()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _check_layer(self, layer_id: int) -> None:
        if not (0 <= layer_id < self.config.n_layers):
            raise ValueError(
                f"layer_id {layer_id} out of range [0, {self.config.n_layers})"
            )

    def __repr__(self) -> str:
        cfg = self.config
        return (
            f"CLAShareAttention(sharing_stride={cfg.sharing_stride}, "
            f"n_layers={cfg.n_layers}, memory_ratio={self.memory_ratio():.2%})"
        )
