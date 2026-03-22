"""squish/moe/lazy_expert_load.py

LazyExpertLoader — Just-in-time expert weight materialisation.

Expert weights are not allocated until a routing score for that expert
exceeds ``activation_threshold``.  Experts that have not been used for
``idle_evict_steps`` steps are evicted to free memory.  This models the
40–60 % live-expert footprint reduction observed in sparse MoE serving.

Reference
---------
Engineering practice; see also:
  Aminabadi et al. "DeepSpeed-MoE: Advancing Mixture-of-Experts Inference
  and Training to Power Next-Generation AI Scale." ICML 2022.
  arXiv:2201.05596.
"""

from __future__ import annotations

__all__ = ["LazyExpertConfig", "LazyExpertState", "LazyExpertLoader"]

from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np
from numpy import ndarray


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class LazyExpertConfig:
    """Configuration for LazyExpertLoader.

    Parameters
    ----------
    n_experts:
        Total number of experts.
    expert_dim:
        Input / output dimension.
    ffn_dim:
        Hidden dimension of expert FFN.
    activation_threshold:
        Minimum routing score to trigger weight materialisation.
    idle_evict_steps:
        Number of steps without activation before an expert is evicted.
    seed:
        RNG seed for synthetic weight generation.
    """

    n_experts: int = 64
    expert_dim: int = 256
    ffn_dim: int = 512
    activation_threshold: float = 0.1
    idle_evict_steps: int = 10
    seed: int = 0

    def __post_init__(self) -> None:
        if self.n_experts < 1:
            raise ValueError("n_experts must be >= 1")
        if self.expert_dim < 1:
            raise ValueError("expert_dim must be >= 1")
        if self.ffn_dim < 1:
            raise ValueError("ffn_dim must be >= 1")
        if self.activation_threshold < 0.0:
            raise ValueError("activation_threshold must be >= 0")
        if self.idle_evict_steps < 1:
            raise ValueError("idle_evict_steps must be >= 1")


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

@dataclass
class LazyExpertState:
    """Mutable state for LazyExpertLoader.

    Attributes
    ----------
    materialized:
        Dict mapping expert_id → (W_up, W_down) for currently loaded experts.
    last_used:
        Dict mapping expert_id → step at which it was last activated.
    step:
        Current global step counter.
    n_materializations:
        Cumulative number of expert weight allocations.
    n_evictions:
        Cumulative number of expert weight evictions.
    """

    materialized: Dict[int, Tuple[ndarray, ndarray]]
    last_used: Dict[int, int]
    step: int = 0
    n_materializations: int = 0
    n_evictions: int = 0


# ---------------------------------------------------------------------------
# LazyExpertLoader
# ---------------------------------------------------------------------------

class LazyExpertLoader:
    """JIT expert weight loader with idle-based eviction.

    Parameters
    ----------
    config:
        ``LazyExpertConfig`` instance.
    """

    def __init__(self, config: LazyExpertConfig) -> None:
        self.config = config
        rng = np.random.default_rng(config.seed)
        scale = float(config.expert_dim) ** -0.5
        # Pre-generate all expert weights in cold storage
        self._cold_W_up: ndarray = rng.standard_normal(
            (config.n_experts, config.ffn_dim, config.expert_dim)
        ).astype(np.float32) * scale
        self._cold_W_down: ndarray = rng.standard_normal(
            (config.n_experts, config.expert_dim, config.ffn_dim)
        ).astype(np.float32) * scale

    def new_state(self) -> LazyExpertState:
        """Create a fresh LazyExpertState with no materialised experts."""
        return LazyExpertState(materialized={}, last_used={})

    def forward(
        self,
        x: ndarray,
        expert_idx: int,
        score: float,
        state: LazyExpertState,
    ) -> Tuple[ndarray, LazyExpertState]:
        """Run expert forward pass, materialising weights if score is high enough.

        Parameters
        ----------
        x:
            Input, shape ``(T, expert_dim)`` or ``(expert_dim,)``.
        expert_idx:
            Index of the expert to run.
        score:
            Routing score for this expert; compared to ``activation_threshold``.
        state:
            Current ``LazyExpertState``.

        Returns
        -------
        out:
            Expert output, same shape as ``x``.  If score < threshold, returns
            zeros (expert not materialised).
        state:
            Updated state.
        """
        x = np.asarray(x, dtype=np.float32)
        scalar_input = x.ndim == 1
        if scalar_input:
            x = x[np.newaxis, :]

        materialized = dict(state.materialized)
        last_used = dict(state.last_used)
        n_mat = state.n_materializations

        if score < self.config.activation_threshold:
            out = np.zeros_like(x)
            if scalar_input:
                out = out[0]
            new_state = LazyExpertState(
                materialized=materialized,
                last_used=last_used,
                step=state.step + 1,
                n_materializations=n_mat,
                n_evictions=state.n_evictions,
            )
            new_state = self._maybe_evict(new_state)
            return out, new_state

        # Materialise if not present
        if expert_idx not in materialized:
            materialized[expert_idx] = (
                self._cold_W_up[expert_idx].copy(),
                self._cold_W_down[expert_idx].copy(),
            )
            n_mat += 1

        last_used[expert_idx] = state.step
        W_up, W_down = materialized[expert_idx]

        # FFN: x → relu(x @ W_up.T) @ W_down.T
        h = np.maximum(0.0, x @ W_up.T)
        out = h @ W_down.T

        new_state = LazyExpertState(
            materialized=materialized,
            last_used=last_used,
            step=state.step + 1,
            n_materializations=n_mat,
            n_evictions=state.n_evictions,
        )
        new_state = self._maybe_evict(new_state)

        if scalar_input:
            out = out[0]
        return out, new_state

    def _materialize(self, idx: int, state: LazyExpertState) -> LazyExpertState:
        """Force materialisation of expert ``idx`` regardless of score."""
        materialized = dict(state.materialized)
        if idx not in materialized:
            materialized[idx] = (
                self._cold_W_up[idx].copy(),
                self._cold_W_down[idx].copy(),
            )
            return LazyExpertState(
                materialized=materialized,
                last_used=dict(state.last_used),
                step=state.step,
                n_materializations=state.n_materializations + 1,
                n_evictions=state.n_evictions,
            )
        return state

    def _maybe_evict(self, state: LazyExpertState) -> LazyExpertState:
        """Evict experts idle for >= idle_evict_steps steps."""
        to_evict = [
            idx
            for idx, last in state.last_used.items()
            if (state.step - last) >= self.config.idle_evict_steps
        ]
        if not to_evict:
            return state
        materialized = dict(state.materialized)
        last_used = dict(state.last_used)
        for idx in to_evict:
            materialized.pop(idx, None)
            last_used.pop(idx, None)
        return LazyExpertState(
            materialized=materialized,
            last_used=last_used,
            step=state.step,
            n_materializations=state.n_materializations,
            n_evictions=state.n_evictions + len(to_evict),
        )
