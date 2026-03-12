"""KVRouter — Cross-instance KV routing for disaggregated prefill/decode.

In disaggregated serving architectures, the prefill node computes the KV cache
for a request and must transfer those tensors to the decode node before
auto-regressive generation can begin.  :class:`KVRouteTable` tracks in-flight
KV transfer metadata per sequence, and :class:`KVRouter` selects the target
decode node via consistent hashing so that assignment is stable, balanced,
and requires no coordination state.

Reference:
    Zhong et al., "DistServe: Disaggregating Prefill and Decoding for
    Goodput-Optimized Large Language Model Serving", OSDI 2024.
    https://arxiv.org/abs/2401.09670

Usage::

    import numpy as np
    from squish.kv_router import KVRouter, KVRouteConfig, KVRouteTable

    cfg    = KVRouteConfig(n_nodes=4, n_layers=32, n_heads=8, head_dim=64)
    table  = KVRouteTable(cfg)
    router = KVRouter(cfg, table)

    for seq_id in range(10):
        source = seq_id % cfg.n_nodes
        target = router.route(seq_id, source)
        entry  = table.register(seq_id, source, target, n_tokens=128,
                                layers=(0, 32))
        print(f"seq {seq_id}: {source} → {target}, {entry.size_bytes} bytes")

    print(f"Active routes: {table.n_active}")
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

__all__ = [
    "KVRouteConfig",
    "KVRouteEntry",
    "KVRouteTable",
    "KVRouter",
    "KVRouterStats",
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class KVRouteConfig:
    """Configuration for the KV routing layer.

    Attributes:
        n_nodes: Total number of serving nodes.  Must be >= 2 to allow
            source and target to differ.
        n_layers: Number of transformer layers whose KVs may be routed.
        n_heads: Number of KV heads per layer.
        head_dim: Dimension of each KV head vector (float32).
    """

    n_nodes: int = 4
    n_layers: int = 32
    n_heads: int = 8
    head_dim: int = 64

    def __post_init__(self) -> None:
        if self.n_nodes < 2:
            raise ValueError(f"n_nodes must be >= 2; got {self.n_nodes}")
        if self.n_layers < 1:
            raise ValueError(f"n_layers must be >= 1; got {self.n_layers}")
        if self.n_heads < 1:
            raise ValueError(f"n_heads must be >= 1; got {self.n_heads}")
        if self.head_dim < 1:
            raise ValueError(f"head_dim must be >= 1; got {self.head_dim}")

    @property
    def kv_bytes_per_token(self) -> int:
        """Bytes needed to store one token's full KV cache (float32).

        Computed as ``n_layers × 2 (K+V) × n_heads × head_dim × 4 bytes``.
        """
        return self.n_layers * 2 * self.n_heads * self.head_dim * 4


# ---------------------------------------------------------------------------
# Route entry
# ---------------------------------------------------------------------------


@dataclass
class KVRouteEntry:
    """Metadata record for a single in-flight KV transfer.

    Attributes:
        seq_id: Unique sequence identifier.
        source_node: Index of the prefill node that owns the KV vectors.
        target_node: Index of the decode node that will receive them.
        layer_range: ``(start_layer, end_layer)`` of the transferred KV
            range; ``end_layer`` is exclusive.
        n_tokens: Number of tokens in the sequence.
        size_bytes: Total transfer size in bytes (keys + values, float32).
    """

    seq_id: int
    source_node: int
    target_node: int
    layer_range: tuple[int, int]
    n_tokens: int
    size_bytes: int


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


@dataclass
class KVRouterStats:
    """Accumulated statistics for :class:`KVRouter`.

    Attributes:
        total_routes: Total number of :meth:`KVRouter.route` calls.
        total_bytes_routed: Cumulative bytes registered in routed entries.
        n_active_sessions: Number of currently active KV-route entries
            (updated on every :meth:`KVRouter.route` call).
    """

    total_routes: int = 0
    total_bytes_routed: int = 0
    n_active_sessions: int = 0


# ---------------------------------------------------------------------------
# Route table
# ---------------------------------------------------------------------------


class KVRouteTable:
    """Routing table that tracks active KV-transfer metadata.

    Entries are added by the prefill node via :meth:`register` and removed
    by the decode node via :meth:`remove` once the transfer completes.

    Args:
        config: :class:`KVRouteConfig` instance.
    """

    def __init__(self, config: KVRouteConfig) -> None:
        self.config = config
        self._table: dict[int, KVRouteEntry] = {}
        self._cumulative_bytes: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register(
        self,
        seq_id: int,
        source: int,
        target: int,
        n_tokens: int,
        layers: tuple[int, int],
    ) -> KVRouteEntry:
        """Register a new KV transfer and return its metadata entry.

        Args:
            seq_id: Unique sequence identifier.  Raises ``ValueError`` if
                already registered.
            source: Source (prefill) node index.
            target: Target (decode) node index.
            n_tokens: Number of tokens in the sequence.  Must be >= 1.
            layers: ``(start_layer, end_layer)`` tuple; ``end_layer`` is
                exclusive.  Must satisfy
                ``0 <= start_layer < end_layer <= n_layers``.

        Returns:
            Newly created :class:`KVRouteEntry`.

        Raises:
            ValueError: If any argument is out of range, or ``seq_id`` is
                already registered.
        """
        cfg = self.config
        if seq_id in self._table:
            raise ValueError(f"seq_id {seq_id} is already registered")
        if not (0 <= source < cfg.n_nodes):
            raise ValueError(
                f"source must be in [0, {cfg.n_nodes}); got {source}"
            )
        if not (0 <= target < cfg.n_nodes):
            raise ValueError(
                f"target must be in [0, {cfg.n_nodes}); got {target}"
            )
        if n_tokens < 1:
            raise ValueError(f"n_tokens must be >= 1; got {n_tokens}")

        start_layer, end_layer = layers
        if not (0 <= start_layer < end_layer <= cfg.n_layers):
            raise ValueError(
                f"layers {layers} invalid for n_layers={cfg.n_layers}"
            )

        n_layer_range = end_layer - start_layer
        # Keys + values (×2), float32 (×4 bytes)
        size_bytes = n_tokens * n_layer_range * 2 * cfg.n_heads * cfg.head_dim * 4

        entry = KVRouteEntry(
            seq_id=seq_id,
            source_node=source,
            target_node=target,
            layer_range=layers,
            n_tokens=n_tokens,
            size_bytes=size_bytes,
        )
        self._table[seq_id] = entry
        self._cumulative_bytes += size_bytes
        return entry

    def lookup(self, seq_id: int) -> Optional[KVRouteEntry]:
        """Return the route entry for a sequence, or ``None`` if absent.

        Args:
            seq_id: Sequence identifier to look up.

        Returns:
            :class:`KVRouteEntry` if registered, otherwise ``None``.
        """
        return self._table.get(seq_id)

    def remove(self, seq_id: int) -> None:
        """Remove a completed KV transfer from the routing table.

        Args:
            seq_id: Sequence identifier.

        Raises:
            KeyError: If ``seq_id`` is not currently registered.
        """
        if seq_id not in self._table:
            raise KeyError(f"seq_id {seq_id} not found in routing table")
        del self._table[seq_id]

    @property
    def n_active(self) -> int:
        """Number of currently registered (in-flight) route entries."""
        return len(self._table)

    @property
    def total_bytes_routed(self) -> int:
        """Cumulative bytes registered across all :meth:`register` calls."""
        return self._cumulative_bytes


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


class KVRouter:
    """Stateless KV router that maps sequences to decode nodes.

    Target nodes are selected via consistent hashing: the first 8 bytes of
    the SHA-256 digest of ``str(seq_id)`` are interpreted as a big-endian
    unsigned integer and taken modulo ``n_nodes``.  If the result equals
    ``source_node`` and more than one node is available, the target is
    shifted by one to keep prefill and decode on separate nodes.

    Args:
        config: :class:`KVRouteConfig` instance.
        table: :class:`KVRouteTable` used to track routing decisions.
    """

    def __init__(self, config: KVRouteConfig, table: KVRouteTable) -> None:
        self.config = config
        self.table = table
        self._stats = KVRouterStats()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def route(self, seq_id: int, source_node: int) -> int:
        """Compute the target decode node for a sequence.

        Args:
            seq_id: Unique sequence identifier.
            source_node: Index of the prefill (source) node.

        Returns:
            Target decode node index in ``[0, n_nodes)``.

        Raises:
            ValueError: If ``source_node`` is out of range.
        """
        cfg = self.config
        if not (0 <= source_node < cfg.n_nodes):
            raise ValueError(
                f"source_node must be in [0, {cfg.n_nodes}); got {source_node}"
            )

        # Consistent hashing: SHA-256 of seq_id string → modulo n_nodes
        digest = hashlib.sha256(str(seq_id).encode()).digest()
        raw_target = int.from_bytes(digest[:8], "big") % cfg.n_nodes

        # Prefer a node different from source to separate prefill from decode
        if raw_target == source_node and cfg.n_nodes > 1:
            raw_target = (raw_target + 1) % cfg.n_nodes

        # Update stats snapshot
        self._stats.total_routes += 1
        self._stats.total_bytes_routed = self.table.total_bytes_routed
        self._stats.n_active_sessions = self.table.n_active

        return raw_target

    def reset_stats(self) -> None:
        """Reset accumulated routing statistics to zero."""
        self._stats = KVRouterStats()

    @property
    def stats(self) -> KVRouterStats:
        """Current accumulated :class:`KVRouterStats`."""
        return self._stats
