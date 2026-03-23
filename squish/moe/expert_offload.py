"""squish/moe/expert_offload.py

ExpertOffloader — CPU-offload / expert-weight paging for sparse MoE models.

Maintains a bounded "resident" set of expert weight pairs in fast memory.
When a requested expert is not resident, it is fetched from cold storage and
the least-recently-used resident expert is evicted.  This models GPU-DRAM
paging used in systems like Mixtral 8x7B offloaded inference.

Reference
---------
Engineering practice; see also:
  Eliseev & Panferov, "Fast Inference of Mixture-of-Experts Language Models
  with Offloading." arXiv:2312.17238, 2023.
"""

from __future__ import annotations

__all__ = ["ExpertOffloadConfig", "OffloadState", "ExpertOffloader"]

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
from numpy import ndarray


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class ExpertOffloadConfig:
    """Configuration for ExpertOffloader.

    Parameters
    ----------
    n_experts:
        Total number of experts in the model.
    expert_dim:
        Input / output dimension of each expert.
    ffn_dim:
        Hidden dimension of each expert FFN.
    max_resident:
        Maximum number of expert pairs held in fast storage simultaneously.
    seed:
        RNG seed used to initialise synthetic expert weights.
    """

    n_experts: int = 64
    expert_dim: int = 256
    ffn_dim: int = 512
    max_resident: int = 8
    seed: int = 0

    def __post_init__(self) -> None:
        if self.n_experts < 1:
            raise ValueError("n_experts must be >= 1")
        if self.expert_dim < 1:
            raise ValueError("expert_dim must be >= 1")
        if self.ffn_dim < 1:
            raise ValueError("ffn_dim must be >= 1")
        if self.max_resident < 1:
            raise ValueError("max_resident must be >= 1")
        if self.max_resident > self.n_experts:
            raise ValueError("max_resident must be <= n_experts")


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

@dataclass
class OffloadState:
    """Mutable paging state for ExpertOffloader.

    Attributes
    ----------
    resident:
        Dict mapping expert_id → (W_up, W_down) currently in fast storage.
    lru_order:
        List of expert_ids in least-recently-used order (index 0 is LRU).
    n_fetches:
        Cumulative number of cache misses (cold fetches).
    n_evictions:
        Cumulative number of LRU evictions performed.
    """

    resident: Dict[int, Tuple[ndarray, ndarray]]
    lru_order: List[int]
    n_fetches: int = 0
    n_evictions: int = 0


# ---------------------------------------------------------------------------
# ExpertOffloader
# ---------------------------------------------------------------------------

class ExpertOffloader:
    """Expert-weight pager with LRU eviction policy.

    Parameters
    ----------
    config:
        ``ExpertOffloadConfig`` instance.
    """

    def __init__(self, config: ExpertOffloadConfig) -> None:
        self.config = config
        rng = np.random.default_rng(config.seed)
        scale = float(config.expert_dim) ** -0.5
        # Initialise all expert weights in "cold" storage (CPU / disk sim)
        self._storage: List[Tuple[ndarray, ndarray]] = []
        for _ in range(config.n_experts):
            W_up = rng.standard_normal((config.ffn_dim, config.expert_dim)).astype(
                np.float32
            ) * scale
            W_down = rng.standard_normal((config.expert_dim, config.ffn_dim)).astype(
                np.float32
            ) * scale
            self._storage.append((W_up, W_down))

    def new_state(self) -> OffloadState:
        """Create a fresh empty OffloadState (no experts resident)."""
        return OffloadState(resident={}, lru_order=[])

    def get_expert(
        self, idx: int, state: OffloadState
    ) -> Tuple[ndarray, ndarray, OffloadState]:
        """Return (W_up, W_down) for expert ``idx``, paging in if necessary.

        Parameters
        ----------
        idx:
            Expert index in ``[0, n_experts)``.
        state:
            Current ``OffloadState``.

        Returns
        -------
        W_up:
            Shape ``(ffn_dim, expert_dim)``.
        W_down:
            Shape ``(expert_dim, ffn_dim)``.
        state:
            Updated ``OffloadState``.
        """
        if idx < 0 or idx >= self.config.n_experts:
            raise IndexError(f"Expert index {idx} out of range [0, {self.config.n_experts})")

        n_fetches = state.n_fetches
        n_evictions = state.n_evictions
        resident = dict(state.resident)
        lru_order = list(state.lru_order)

        if idx in resident:
            # Cache hit — move to MRU position
            lru_order.remove(idx)
            lru_order.append(idx)
        else:
            # Cache miss — fetch from cold storage
            n_fetches += 1
            if len(resident) >= self.config.max_resident:
                state_tmp = OffloadState(resident=resident, lru_order=lru_order,
                                         n_fetches=n_fetches, n_evictions=n_evictions)
                state_tmp = self.evict_lru(state_tmp)
                resident = state_tmp.resident
                lru_order = state_tmp.lru_order
                n_evictions = state_tmp.n_evictions

            W_up, W_down = self._storage[idx]
            resident[idx] = (W_up.copy(), W_down.copy())
            lru_order.append(idx)

        new_state = OffloadState(
            resident=resident,
            lru_order=lru_order,
            n_fetches=n_fetches,
            n_evictions=n_evictions,
        )
        W_up, W_down = new_state.resident[idx]
        return W_up, W_down, new_state

    def evict_lru(self, state: OffloadState) -> OffloadState:
        """Evict the least-recently-used expert from the resident set.

        No-op if the resident set is empty.
        """
        if not state.lru_order:
            return state
        resident = dict(state.resident)
        lru_order = list(state.lru_order)
        lru_id = lru_order.pop(0)
        resident.pop(lru_id, None)
        return OffloadState(
            resident=resident,
            lru_order=lru_order,
            n_fetches=state.n_fetches,
            n_evictions=state.n_evictions + 1,
        )

    def stats(self, state: OffloadState) -> dict:
        """Return a dict of paging statistics."""
        total = state.n_fetches
        hit_requests = total - state.n_fetches  # always 0; fetches ARE misses
        return {
            "n_resident": len(state.resident),
            "max_resident": self.config.max_resident,
            "n_fetches": state.n_fetches,
            "n_evictions": state.n_evictions,
            "resident_ids": sorted(state.resident.keys()),
        }


# server.py compatibility alias
ExpertOffloaderConfig = ExpertOffloadConfig
