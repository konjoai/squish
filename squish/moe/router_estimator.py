"""squish/moe/router_estimator.py

RouterEstimator — Pre-compute full routing schedule before any expert loads.

The key insight for memory-efficient MoE inference: the router gate is tiny
(hidden_size × n_experts per layer, e.g. 4096 × 8 = 131k weights for
Mixtral-8x7B).  By running all routing decisions for a sequence BEFORE
loading any expert weights, we know exactly which experts are needed at each
layer and can pre-warm the ExpertMemoryMap in optimal order.

This avoids the common "cold start" pattern where each expert is loaded only
after the router fires, causing sequential disk-read bubbles.

Algorithm
---------
1. Run the full attention pass (backbone only) to obtain hidden states H₀..H_L.
2. For each layer L, compute gate logits = H_L @ W_gate[L].T
3. Apply top-K selection → indices, weights.
4. Record ExpertSchedule: layer → sorted list of expert indices to activate.

The router estimator uses numpy-only operations and works identically in CPU
and Metal paths.

References
----------
Mistral AI, "Mixtral of Experts," arXiv:2401.04088, 2024.
DeepSeek-AI, "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts
Language Model," arXiv:2405.04434, 2024.

Usage
-----
::

    est = RouterEstimator(n_layers=32, n_experts=8, top_k=2, hidden_size=4096)

    # Router gate weight: shape (n_layers, n_experts, hidden_size)
    est.load_gate_weights(gate_weights)

    # hidden_states: shape (seq_len, hidden_size)  — one set per layer, or
    # a single matrix reused across layers for a quick estimate
    schedule = est.estimate(hidden_states_per_layer)
    print(schedule)  # ExpertSchedule(n_layers=32, ...)

    for layer_idx, expert_ids in schedule:
        ...  # pre-load these experts
"""

from __future__ import annotations

__all__ = [
    "RouterConfig",
    "LayerRouting",
    "ExpertSchedule",
    "RouterEstimator",
]

from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RouterConfig:
    """Configuration for RouterEstimator.

    Attributes
    ----------
    n_layers:
        Number of transformer (MoE) layers.
    n_experts:
        Total experts per layer.
    top_k:
        Number of experts selected per token per layer.
    hidden_size:
        Hidden dimension of the model.
    router_jitter:
        Small uniform noise σ added to gate logits during estimation to
        break ties (disabled when 0.0).
    normalize_weights:
        If True, selected expert weights are softmax-normalised over the
        top-k scores.  If False, raw softmax over all experts is used.
    """

    n_layers: int = 32
    n_experts: int = 8
    top_k: int = 2
    hidden_size: int = 4096
    router_jitter: float = 0.0
    normalize_weights: bool = True

    def __post_init__(self) -> None:
        if self.n_layers < 1:
            raise ValueError("n_layers must be >= 1")
        if self.n_experts < 1:
            raise ValueError("n_experts must be >= 1")
        if self.top_k < 1 or self.top_k > self.n_experts:
            raise ValueError(f"top_k must be in [1, n_experts={self.n_experts}]")
        if self.hidden_size < 1:
            raise ValueError("hidden_size must be >= 1")


@dataclass(frozen=True)
class LayerRouting:
    """Routing decision for a single transformer layer.

    Attributes
    ----------
    layer_idx:
        Zero-based layer index.
    expert_ids:
        Sorted array of unique expert IDs activated by at least one token.
    token_assignments:
        Shape (seq_len, top_k) — expert index for each (token, slot).
    token_weights:
        Shape (seq_len, top_k) — softmax weight for each (token, slot).
    gate_logits:
        Shape (seq_len, n_experts) — raw gate logits (before softmax).
    """

    layer_idx: int
    expert_ids: np.ndarray       # shape (n_active,)  unique sorted expert IDs
    token_assignments: np.ndarray  # shape (seq_len, top_k)
    token_weights: np.ndarray    # shape (seq_len, top_k)
    gate_logits: np.ndarray      # shape (seq_len, n_experts)

    @property
    def n_active_experts(self) -> int:
        return len(self.expert_ids)

    @property
    def seq_len(self) -> int:
        return self.token_assignments.shape[0]


@dataclass
class ExpertSchedule:
    """Complete per-layer routing schedule for a single forward pass.

    Attributes
    ----------
    n_layers:
        Number of layers in the schedule.
    routings:
        Dict mapping layer_idx → LayerRouting.
    """

    n_layers: int
    routings: Dict[int, LayerRouting] = field(default_factory=dict)

    def __iter__(self) -> Iterator[Tuple[int, np.ndarray]]:
        """Yield (layer_idx, expert_ids) in layer order."""
        for li in sorted(self.routings):
            yield li, self.routings[li].expert_ids

    def experts_for_layer(self, layer_idx: int) -> np.ndarray:
        """Return sorted expert IDs for *layer_idx*, or empty array."""
        routing = self.routings.get(layer_idx)
        return routing.expert_ids if routing is not None else np.array([], dtype=np.int32)

    def all_expert_ids(self) -> np.ndarray:
        """Sorted unique expert IDs across ALL layers."""
        ids = []
        for routing in self.routings.values():
            ids.extend(routing.expert_ids.tolist())
        return np.unique(ids).astype(np.int32) if ids else np.array([], dtype=np.int32)

    def expert_activation_frequency(self) -> Dict[int, int]:
        """Return {expert_id: activation_count} summed across all layers."""
        freq: Dict[int, int] = {}
        for routing in self.routings.values():
            for eid in routing.token_assignments.ravel():
                freq[int(eid)] = freq.get(int(eid), 0) + 1
        return freq

    def peak_active_per_layer(self) -> int:
        """Max number of unique experts active in any single layer."""
        return max(
            (r.n_active_experts for r in self.routings.values()), default=0
        )

    def __len__(self) -> int:
        return len(self.routings)

    def __repr__(self) -> str:
        peak = self.peak_active_per_layer()
        return (
            f"ExpertSchedule("
            f"n_layers={self.n_layers}, "
            f"scheduled={len(self.routings)}, "
            f"peak_active_per_layer={peak})"
        )


# ---------------------------------------------------------------------------
# RouterEstimator
# ---------------------------------------------------------------------------

class RouterEstimator:
    """Pre-compute routing decisions for a full forward pass.

    Gate weights must be loaded before calling :meth:`estimate`.

    Parameters
    ----------
    config:
        Router hyper-parameters.
    """

    def __init__(self, config: RouterConfig) -> None:
        self._config = config
        # gate_weights[L] has shape (hidden_size, n_experts)  or  (n_experts, hidden_size)
        # We normalise to (n_experts, hidden_size) internally.
        self._gate_weights: Optional[np.ndarray] = None  # shape (n_layers, n_experts, hidden_size)
        self._rng = np.random.default_rng(seed=0)

    # ------------------------------------------------------------------ #
    # Weight loading
    # ------------------------------------------------------------------ #

    def load_gate_weights(self, weights: np.ndarray) -> None:
        """Set gate projection weights for all layers.

        Parameters
        ----------
        weights:
            Array of shape ``(n_layers, n_experts, hidden_size)`` or
            ``(n_layers, hidden_size, n_experts)`` — the latter is
            transposed automatically.
        """
        cfg = self._config
        if weights.ndim != 3:
            raise ValueError(
                f"gate_weights must be 3-D, got shape {weights.shape}"
            )
        n, a, b = weights.shape
        if n != cfg.n_layers:
            raise ValueError(
                f"Expected n_layers={cfg.n_layers} slices, got {n}"
            )
        # Normalise to (n_layers, n_experts, hidden_size)
        if a == cfg.n_experts and b == cfg.hidden_size:
            self._gate_weights = weights.astype(np.float32)
        elif a == cfg.hidden_size and b == cfg.n_experts:
            self._gate_weights = weights.transpose(0, 2, 1).astype(np.float32)
        else:
            raise ValueError(
                f"weight dims ({a}, {b}) don't match "
                f"n_experts={cfg.n_experts} × hidden_size={cfg.hidden_size}"
            )

    def load_gate_weights_for_layer(
        self, layer_idx: int, weights: np.ndarray
    ) -> None:
        """Load gate weights for a single layer.

        Useful when gate weights are scattered across shard files.  The
        full buffer is allocated on first call.

        Parameters
        ----------
        layer_idx:
            Zero-based layer index (0 ≤ layer_idx < n_layers).
        weights:
            Shape ``(n_experts, hidden_size)`` or ``(hidden_size, n_experts)``.
        """
        cfg = self._config
        if self._gate_weights is None:
            self._gate_weights = np.zeros(
                (cfg.n_layers, cfg.n_experts, cfg.hidden_size), dtype=np.float32
            )
        if weights.ndim == 2:
            a, b = weights.shape
            if a == cfg.hidden_size and b == cfg.n_experts:
                weights = weights.T
        self._gate_weights[layer_idx] = weights.astype(np.float32)

    # ------------------------------------------------------------------ #
    # Routing core
    # ------------------------------------------------------------------ #

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        x_shifted = x - x.max(axis=-1, keepdims=True)
        exp_x = np.exp(x_shifted)
        return exp_x / exp_x.sum(axis=-1, keepdims=True)

    def _route_layer(
        self,
        layer_idx: int,
        hidden: np.ndarray,
    ) -> LayerRouting:
        """Compute routing for a single layer.

        Parameters
        ----------
        layer_idx:
            Layer index.
        hidden:
            Shape ``(seq_len, hidden_size)`` — hidden states entering this layer.
        """
        cfg = self._config
        gate_w = self._gate_weights[layer_idx]  # (n_experts, hidden_size)

        # Gate logits: (seq_len, n_experts)
        logits = hidden @ gate_w.T  # (seq_len, n_experts)

        if cfg.router_jitter > 0.0:
            noise = self._rng.uniform(
                -cfg.router_jitter, cfg.router_jitter, size=logits.shape
            ).astype(np.float32)
            logits = logits + noise

        probs = self._softmax(logits)  # (seq_len, n_experts)

        # Top-K selection along expert dimension
        top_k_indices = np.argsort(-probs, axis=-1)[:, : cfg.top_k]  # (seq_len, top_k)

        if cfg.normalize_weights:
            top_k_probs_raw = np.take_along_axis(probs, top_k_indices, axis=-1)
            row_sum = top_k_probs_raw.sum(axis=-1, keepdims=True)
            top_k_weights = top_k_probs_raw / np.where(row_sum > 0, row_sum, 1.0)
        else:
            top_k_weights = np.take_along_axis(probs, top_k_indices, axis=-1)

        # Unique expert IDs activated in this layer
        expert_ids = np.unique(top_k_indices).astype(np.int32)

        return LayerRouting(
            layer_idx=layer_idx,
            expert_ids=expert_ids,
            token_assignments=top_k_indices.astype(np.int32),
            token_weights=top_k_weights.astype(np.float32),
            gate_logits=logits,
        )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def estimate(
        self,
        hidden_states_per_layer: Sequence[np.ndarray] | np.ndarray,
    ) -> ExpertSchedule:
        """Compute routing schedule for a full forward pass.

        Parameters
        ----------
        hidden_states_per_layer:
            Either:
            * A list of ``n_layers`` arrays, each of shape
              ``(seq_len, hidden_size)`` — one per layer.
            * A single array of shape ``(seq_len, hidden_size)`` — reused
              for every layer (cheap approximation if hidden states don't
              change much across layers).

        Returns
        -------
        ExpertSchedule
            Complete routing plan covering all layers.

        Raises
        ------
        RuntimeError
            If gate weights have not been loaded.
        ValueError
            If hidden state dimensions are incompatible with the config.
        """
        if self._gate_weights is None:
            raise RuntimeError(
                "Gate weights are not loaded. Call load_gate_weights() first."
            )
        cfg = self._config

        # Normalise input to a list of per-layer hidden states
        if isinstance(hidden_states_per_layer, np.ndarray):
            if hidden_states_per_layer.ndim == 2:
                hs_list = [hidden_states_per_layer] * cfg.n_layers
            elif (
                hidden_states_per_layer.ndim == 3
                and hidden_states_per_layer.shape[0] == cfg.n_layers
            ):
                hs_list = [hidden_states_per_layer[i] for i in range(cfg.n_layers)]
            else:
                raise ValueError(
                    f"hidden_states_per_layer ndarray shape {hidden_states_per_layer.shape} "
                    f"is ambiguous; expected (seq_len, hidden_size) or "
                    f"(n_layers, seq_len, hidden_size)"
                )
        else:
            hs_list = list(hidden_states_per_layer)

        if len(hs_list) != cfg.n_layers:
            raise ValueError(
                f"Expected {cfg.n_layers} hidden state arrays, got {len(hs_list)}"
            )

        schedule = ExpertSchedule(n_layers=cfg.n_layers)
        for li, hs in enumerate(hs_list):
            hs = np.asarray(hs, dtype=np.float32)
            if hs.ndim != 2 or hs.shape[1] != cfg.hidden_size:
                raise ValueError(
                    f"Layer {li}: hidden shape {hs.shape} expected "
                    f"(seq_len, {cfg.hidden_size})"
                )
            schedule.routings[li] = self._route_layer(li, hs)

        return schedule

    def estimate_single_layer(
        self,
        layer_idx: int,
        hidden: np.ndarray,
    ) -> LayerRouting:
        """Route a single layer without building a full schedule.

        Useful during actual inference where hidden states are computed
        one layer at a time.
        """
        if self._gate_weights is None:
            raise RuntimeError("Gate weights not loaded")
        if layer_idx < 0 or layer_idx >= self._config.n_layers:
            raise IndexError(
                f"layer_idx={layer_idx} out of range [0, {self._config.n_layers})"
            )
        hidden = np.asarray(hidden, dtype=np.float32)
        return self._route_layer(layer_idx, hidden)

    @property
    def config(self) -> RouterConfig:
        return self._config

    @property
    def gate_weights_loaded(self) -> bool:
        return self._gate_weights is not None

    def __repr__(self) -> str:
        return (
            f"RouterEstimator(n_layers={self._config.n_layers}, "
            f"n_experts={self._config.n_experts}, "
            f"top_k={self._config.top_k}, "
            f"weights_loaded={self.gate_weights_loaded})"
        )
