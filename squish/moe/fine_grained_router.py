"""squish/moe/fine_grained_router.py

FineGrainedMoERouter — Aux-loss-free expert load balancing via per-step
router-bias updates (DeepSeek-V3 style).

Instead of an auxiliary load-balancing loss term that degrades model quality,
the router maintains a per-expert bias b_e.  After each forward step, experts
that were overloaded have their bias decreased and underloaded experts have
their bias increased.  The bias shifts the routing scores without changing the
primary softmax probabilities.

Reference
---------
DeepSeek-AI, "DeepSeek-V3 Technical Report." arXiv:2412.19437, 2024.
"""

from __future__ import annotations

__all__ = ["FineGrainedRouterConfig", "RouterBiasState", "FineGrainedMoERouter"]

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np
from numpy import ndarray


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class FineGrainedRouterConfig:
    """Configuration for FineGrainedMoERouter.

    Parameters
    ----------
    d_model:
        Hidden dimension of input tokens.
    n_experts:
        Total number of experts.
    top_k:
        Number of experts each token is dispatched to.
    n_groups:
        Number of expert groups for group-level load monitoring.
    bias_update_lr:
        Step size for the per-expert bias update.
    balance_window:
        Number of steps over which the EMA load estimate is averaged.
    seed:
        RNG seed for router weight initialisation.
    """

    d_model: int = 256
    n_experts: int = 64
    top_k: int = 6
    n_groups: int = 8
    bias_update_lr: float = 0.001
    balance_window: int = 32
    seed: int = 0

    def __post_init__(self) -> None:
        if self.d_model < 1:
            raise ValueError("d_model must be >= 1")
        if self.n_experts < 1:
            raise ValueError("n_experts must be >= 1")
        if self.top_k < 1 or self.top_k > self.n_experts:
            raise ValueError("top_k must be in [1, n_experts]")
        if self.n_groups < 1 or self.n_groups > self.n_experts:
            raise ValueError("n_groups must be in [1, n_experts]")
        if self.bias_update_lr <= 0.0:
            raise ValueError("bias_update_lr must be > 0")
        if self.balance_window < 1:
            raise ValueError("balance_window must be >= 1")


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

@dataclass
class RouterBiasState:
    """Mutable state for FineGrainedMoERouter.

    Attributes
    ----------
    bias:
        Per-expert additive bias, shape ``(n_experts,)``.
    load_ema:
        Exponential moving average of per-expert token load fractions,
        shape ``(n_experts,)``.
    n_steps:
        Number of update steps taken so far.
    """

    bias: ndarray
    load_ema: ndarray
    n_steps: int = 0


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

class FineGrainedMoERouter:
    """Fine-grained MoE router with aux-loss-free load balancing.

    Parameters
    ----------
    config:
        ``FineGrainedRouterConfig`` instance.
    """

    def __init__(self, config: FineGrainedRouterConfig) -> None:
        self.config = config
        rng = np.random.default_rng(config.seed)
        scale = float(config.d_model) ** -0.5
        # Router projection: (n_experts, d_model)
        self._router_w: ndarray = (
            rng.standard_normal((config.n_experts, config.d_model)).astype(np.float32)
            * scale
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def new_state(self) -> RouterBiasState:
        """Create a fresh zeroed RouterBiasState."""
        return RouterBiasState(
            bias=np.zeros(self.config.n_experts, dtype=np.float32),
            load_ema=np.ones(self.config.n_experts, dtype=np.float32)
            / self.config.n_experts,
            n_steps=0,
        )

    def route(
        self, x: ndarray, state: RouterBiasState
    ) -> Tuple[ndarray, ndarray, RouterBiasState]:
        """Route tokens to top-K experts.

        Parameters
        ----------
        x:
            Token embeddings, shape ``(T, d_model)``.
        state:
            Current ``RouterBiasState``.

        Returns
        -------
        indices:
            Shape ``(T, top_k)`` — expert indices.
        weights:
            Shape ``(T, top_k)`` — renormalised softmax weights.
        state:
            Updated state (bias updated via ``update_bias``).
        """
        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != self.config.d_model:
            raise ValueError(
                f"x must be (T, {self.config.d_model}), got {x.shape}"
            )
        # logits: (T, n_experts)
        logits = x @ self._router_w.T + state.bias[np.newaxis, :]
        # softmax per token
        logits_shifted = logits - logits.max(axis=-1, keepdims=True)
        exp_l = np.exp(logits_shifted)
        probs = exp_l / exp_l.sum(axis=-1, keepdims=True)  # (T, n_experts)

        top_k = self.config.top_k
        sorted_idx = np.argsort(-probs, axis=-1)
        indices = sorted_idx[:, :top_k]  # (T, top_k)
        raw_w = np.take_along_axis(probs, indices, axis=-1)
        denom = raw_w.sum(axis=-1, keepdims=True)
        weights = raw_w / np.where(denom > 0, denom, 1.0)

        # Count per-expert load for bias update
        T = x.shape[0]
        load_counts = np.zeros(self.config.n_experts, dtype=np.float32)
        np.add.at(load_counts, indices.ravel(), 1.0)
        load_counts /= max(T * top_k, 1)  # fraction of total dispatches

        state = self.update_bias(load_counts, state)
        return indices, weights, state

    def update_bias(
        self, load_counts: ndarray, state: RouterBiasState
    ) -> RouterBiasState:
        """Update per-expert bias based on observed load.

        Parameters
        ----------
        load_counts:
            Per-expert fractional load observed in the current step,
            shape ``(n_experts,)``.
        state:
            Current ``RouterBiasState``.

        Returns
        -------
        Updated ``RouterBiasState``.
        """
        alpha = 2.0 / (self.config.balance_window + 1)
        new_ema = (1.0 - alpha) * state.load_ema + alpha * load_counts
        # Decrease bias for overloaded, increase for underloaded
        target = 1.0 / self.config.n_experts
        delta = target - new_ema  # positive → underloaded, negative → overloaded
        new_bias = state.bias + self.config.bias_update_lr * delta
        return RouterBiasState(
            bias=new_bias.astype(np.float32),
            load_ema=new_ema.astype(np.float32),
            n_steps=state.n_steps + 1,
        )

    def expert_utilization(self, state: RouterBiasState) -> ndarray:
        """Return per-expert EMA load estimate, shape ``(n_experts,)``."""
        return state.load_ema.copy()


# server.py compatibility alias
FineGrainedMoEConfig = FineGrainedRouterConfig
