# [Experimental] This module is part of Squish v6+ (Waves 19–26).
# Proof-of-concept quality: API and behaviour may change without notice.
# For stable core functionality see Waves 1–12 in MODULES.md.
"""WeightSharing — Cross-layer weight tying with low-rank residual deltas.

A single base weight matrix ``W_base`` of shape ``(hidden_dim, hidden_dim)``
is shared across *N* transformer layers.  Each layer ``i`` computes its
effective weight as:

    ``W_i = W_base + U_i @ V_i``

where ``U_i`` is ``(hidden_dim, rank)`` and ``V_i`` is ``(rank, hidden_dim)``
with ``rank << hidden_dim``.  This low-rank additive delta preserves
per-layer adaptability while dramatically reducing the total parameter count
compared to storing *N* independent full matrices.

Memory reduction (element count ratio) vs *N* independent weight matrices:

    ``(hidden_dim² + N × 2 × hidden_dim × rank) / (N × hidden_dim²)``

For ``hidden_dim=4096``, ``N=8``, ``rank=8`` this is approximately 0.127,
i.e. only ~13% of the memory of independent weights.

Usage::

    import numpy as np
    from squish.lora.weight_sharing import SharingConfig, WeightSharer

    cfg    = SharingConfig(hidden_dim=512, n_shared_layers=4, rank=8)
    sharer = WeightSharer(cfg)

    w0 = sharer.get_effective_weight(0)   # (512, 512) float32
    w3 = sharer.get_effective_weight(3)

    print(sharer.memory_bytes())          # much less than 4 × 512² × 4
    print(sharer.stats.memory_ratio)
"""

from __future__ import annotations

__all__ = [
    "SharingConfig",
    "WeightSharer",
    "SharingStats",
]

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class SharingConfig:
    """Configuration for cross-layer weight sharing with low-rank deltas.

    Attributes:
        hidden_dim: Width of the square weight matrix shared across layers.
            Must be >= 1.
        n_shared_layers: Number of transformer layers that share the base
            weight.  Must be >= 1.
        rank: Inner rank of each per-layer additive delta (U @ V).
            Must satisfy ``1 <= rank < hidden_dim``.
    """

    hidden_dim: int = 512
    n_shared_layers: int = 4
    rank: int = 8

    def __post_init__(self) -> None:
        if self.hidden_dim < 1:
            raise ValueError(
                f"hidden_dim must be >= 1; got {self.hidden_dim}"
            )
        if self.n_shared_layers < 1:
            raise ValueError(
                f"n_shared_layers must be >= 1; got {self.n_shared_layers}"
            )
        if self.rank < 1:
            raise ValueError(
                f"rank must be >= 1; got {self.rank}"
            )
        if self.rank >= self.hidden_dim:
            raise ValueError(
                f"rank must be strictly less than hidden_dim; "
                f"got rank={self.rank}, hidden_dim={self.hidden_dim}"
            )


# ── Stats ─────────────────────────────────────────────────────────────────────

@dataclass
class SharingStats:
    """Runtime statistics for a :class:`WeightSharer` session.

    Stores the cumulative call counter and the configuration parameters
    required to compute the live memory compression ratio.

    Attributes:
        total_effective_weight_calls: Number of times
            :meth:`WeightSharer.get_effective_weight` has been called.
        hidden_dim: Width of the shared weight matrix (from config).
        n_shared_layers: Number of layers sharing the base weight.
        rank: Low-rank delta dimension (from config).
    """

    total_effective_weight_calls: int = 0
    hidden_dim: int = field(default=512, repr=False)
    n_shared_layers: int = field(default=1, repr=False)
    rank: int = field(default=8, repr=False)

    @property
    def memory_ratio(self) -> float:
        """Ratio of shared-storage elements to dense-storage elements.

        Computed as:

            ``(hidden_dim² + N × 2 × hidden_dim × rank)
            / (N × hidden_dim²)``

        Values below 1 indicate that the weight-sharing scheme uses fewer
        parameters than storing *N* independent full weight matrices.
        """
        hd = self.hidden_dim
        r  = self.rank
        n  = self.n_shared_layers
        shared_elems = hd * hd + n * 2 * hd * r
        dense_elems  = n * hd * hd
        return shared_elems / max(1, dense_elems)


# ── WeightSharer ──────────────────────────────────────────────────────────────

class WeightSharer:
    """Cross-layer weight store with shared base and per-layer low-rank deltas.

    On construction, a shared base weight matrix and *n_shared_layers* pairs
    of low-rank delta matrices are initialised from Gaussian distributions.
    The base uses ``N(0, 0.02)`` (typical LLM initialisation scale) and the
    deltas use ``N(0, 0.01)`` (small perturbations).

    Args:
        config: :class:`SharingConfig` specifying dimensions.
    """

    def __init__(self, config: SharingConfig) -> None:
        self.config = config
        self._stats_count: int = 0

        rng = np.random.default_rng(0)
        hd  = config.hidden_dim
        r   = config.rank

        # Shared base weight matrix: (hidden_dim, hidden_dim)
        self.base_weight: np.ndarray = rng.normal(
            0.0, 0.02, (hd, hd)
        ).astype(np.float32)

        # Per-layer low-rank deltas: list of (U_i, V_i) tuples.
        # U_i: (hidden_dim, rank),  V_i: (rank, hidden_dim)
        self._deltas: List[Tuple[np.ndarray, np.ndarray]] = [
            (
                rng.normal(0.0, 0.01, (hd, r)).astype(np.float32),
                rng.normal(0.0, 0.01, (r, hd)).astype(np.float32),
            )
            for _ in range(config.n_shared_layers)
        ]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_effective_weight(self, layer_idx: int) -> np.ndarray:
        """Compute the effective weight matrix for the given layer.

        The result is ``W_base + U[layer_idx] @ V[layer_idx]``, cast to
        float32.

        Args:
            layer_idx: Zero-based layer index in
                ``[0, config.n_shared_layers)``.

        Returns:
            Float32 array of shape ``(hidden_dim, hidden_dim)`` representing
            the effective weight for layer *layer_idx*.

        Raises:
            IndexError: If *layer_idx* is out of range.
        """
        if layer_idx < 0 or layer_idx >= self.config.n_shared_layers:
            raise IndexError(
                f"layer_idx must be in [0, {self.config.n_shared_layers}); "
                f"got {layer_idx}"
            )
        U, V = self._deltas[layer_idx]
        effective = (self.base_weight + U @ V).astype(np.float32)
        self._stats_count += 1
        return effective

    def memory_bytes(self) -> int:
        """Total bytes consumed by the shared base and all per-layer deltas.

        Counts 4 bytes per float32 element:

        * Base weight: ``hidden_dim × hidden_dim × 4``
        * Each delta pair (U, V): ``2 × hidden_dim × rank × 4``

        Returns:
            Total storage in bytes.
        """
        hd = self.config.hidden_dim
        r  = self.config.rank
        n  = self.config.n_shared_layers
        base_elems  = hd * hd
        delta_elems = n * 2 * hd * r
        return (base_elems + delta_elems) * 4  # 4 bytes per float32

    def dense_memory_bytes(self) -> int:
        """Bytes for *n_shared_layers* independent full weight matrices.

        Returns:
            ``n_shared_layers × hidden_dim² × 4``
        """
        hd = self.config.hidden_dim
        n  = self.config.n_shared_layers
        return n * hd * hd * 4

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def stats(self) -> SharingStats:
        """Live statistics snapshot including the current memory ratio."""
        cfg = self.config
        return SharingStats(
            total_effective_weight_calls=self._stats_count,
            hidden_dim=cfg.hidden_dim,
            n_shared_layers=cfg.n_shared_layers,
            rank=cfg.rank,
        )
