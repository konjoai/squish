"""squish/moe/pregated_router.py

PreGatedMoERouter — Zero-latency MoE expert routing via previous-layer
hidden state pre-computation (Du et al., EMNLP 2024 / arXiv:2402.05666).

Reference
---------
"Pre-gated MoE: An Algorithm-Efficient Approach for Fast and Scalable
Mixture-of-Expert Inference." Du et al., EMNLP 2024 (arXiv:2402.05666).

Algorithm
---------
Standard MoE gating computes router logits AFTER the current-layer FFN,
inserting a latency-critical scatter-gather on the hot path.

Pre-gated MoE shifts routing to the PREVIOUS layer's hidden state:

1. At layer l-1, compute ``router_logits = hidden @ W_gate^T``.
2. Top-K experts are selected by softmax over router logits.
3. At layer l, tokens are already routed — zero extra latency.
4. Expert outputs are combined with gating weights.

This simulation:
* Stores a ``W_gate`` weight matrix (``n_experts × hidden_dim``).
* ``route(hidden_prev)`` returns expert indices and gating weights.
* ``forward(hidden_cur, expert_fns)`` applies the pre-computed routing.

Key properties
--------------
* NumPy-only simulation; no GPU dependency.
* ``n_experts`` — total number of expert networks.
* ``top_k`` — experts selected per token.
* ``hidden_dim`` — dimension of hidden states.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np

__all__ = [
    "PreGatedMoEConfig",
    "PreGatedMoERouter",
]

# ── Config ────────────────────────────────────────────────────────────────────


@dataclass
class PreGatedMoEConfig:
    """Configuration for :class:`PreGatedMoERouter`.

    Attributes:
        n_experts: Total number of expert networks.
        top_k: Number of experts to select per token.
        hidden_dim: Hidden state dimension.
        seed: Random seed for weight initialisation.
    """

    n_experts: int = 8
    top_k: int = 2
    hidden_dim: int = 256
    seed: int = 0

    def __post_init__(self) -> None:
        if self.n_experts < 1:
            raise ValueError(f"n_experts must be ≥ 1; got {self.n_experts}")
        if self.top_k < 1:
            raise ValueError(f"top_k must be ≥ 1; got {self.top_k}")
        if self.top_k > self.n_experts:
            raise ValueError(
                f"top_k ({self.top_k}) must be ≤ n_experts ({self.n_experts})"
            )
        if self.hidden_dim < 1:
            raise ValueError(f"hidden_dim must be ≥ 1; got {self.hidden_dim}")


# ── Router ────────────────────────────────────────────────────────────────────


class PreGatedMoERouter:
    """Pre-gated MoE router: routes tokens using the previous layer's hidden state.

    Example::

        cfg = PreGatedMoEConfig(n_experts=4, top_k=2, hidden_dim=64)
        router = PreGatedMoERouter(cfg)

        hidden_prev = np.random.randn(8, 64).astype(np.float32)  # (T, d)
        expert_indices, gate_weights = router.route(hidden_prev)

        def expert_fn(expert_id, token_hidden):
            return some_ffn(expert_id, token_hidden)

        hidden_cur = np.random.randn(8, 64).astype(np.float32)
        out = router.forward(hidden_cur, expert_fn, expert_indices, gate_weights)
    """

    def __init__(self, config: Optional[PreGatedMoEConfig] = None) -> None:
        self.config = config or PreGatedMoEConfig()
        rng = np.random.default_rng(self.config.seed)
        scale = 1.0 / np.sqrt(self.config.hidden_dim)
        # Gate weight matrix: (n_experts, hidden_dim)
        self.W_gate: np.ndarray = (
            rng.standard_normal((self.config.n_experts, self.config.hidden_dim)).astype(
                np.float32
            )
            * scale
        )
        self._pre_routed_indices: Optional[np.ndarray] = None
        self._pre_routed_weights: Optional[np.ndarray] = None

    # ── Public API ────────────────────────────────────────────────────────────

    def route(
        self,
        hidden_prev: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Pre-compute expert routing from the previous layer's hidden state.

        Args:
            hidden_prev: ``(T, hidden_dim)`` previous layer hidden states.

        Returns:
            Tuple ``(expert_indices, gate_weights)`` where:

            * ``expert_indices``: ``(T, top_k)`` int32 expert indices.
            * ``gate_weights``: ``(T, top_k)`` float32 softmax gate weights.
        """
        hidden_prev = np.asarray(hidden_prev, dtype=np.float32)
        if hidden_prev.ndim != 2 or hidden_prev.shape[-1] != self.config.hidden_dim:
            raise ValueError(
                f"hidden_prev must be (T, {self.config.hidden_dim}); "
                f"got {hidden_prev.shape}"
            )
        # Router logits: (T, n_experts)
        logits = hidden_prev @ self.W_gate.T
        # Top-K selection
        top_k = self.config.top_k
        top_idx = np.argsort(logits, axis=-1)[:, -top_k:][:, ::-1]  # (T, top_k)
        top_logits = np.take_along_axis(logits, top_idx, axis=-1)   # (T, top_k)
        # Softmax gate weights over selected experts
        top_exp = np.exp(top_logits - top_logits.max(axis=-1, keepdims=True))
        gate_weights = (top_exp / (top_exp.sum(axis=-1, keepdims=True) + 1e-9)).astype(
            np.float32
        )
        self._pre_routed_indices = top_idx.astype(np.int32)
        self._pre_routed_weights = gate_weights
        return top_idx.astype(np.int32), gate_weights

    def forward(
        self,
        hidden_cur: np.ndarray,
        expert_fn: Callable[[int, np.ndarray], np.ndarray],
        expert_indices: Optional[np.ndarray] = None,
        gate_weights: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Apply pre-gated routing to current-layer hidden states.

        Args:
            hidden_cur: ``(T, hidden_dim)`` current layer hidden states.
            expert_fn: Callable ``(expert_id: int, token_hidden: ndarray) -> ndarray``
                       applies a single expert to one token.
            expert_indices: ``(T, top_k)`` from :meth:`route`. If None, uses last
                pre-computed result.
            gate_weights: ``(T, top_k)`` from :meth:`route`. If None, uses last
                pre-computed result.

        Returns:
            ``(T, hidden_dim)`` weighted combination of expert outputs.

        Raises:
            ValueError: If no routing has been pre-computed.
        """
        hidden_cur = np.asarray(hidden_cur, dtype=np.float32)
        if expert_indices is None:
            if self._pre_routed_indices is None:
                raise ValueError(
                    "No pre-computed routing found. Call route() first."
                )
            expert_indices = self._pre_routed_indices
            gate_weights = self._pre_routed_weights

        T, d = hidden_cur.shape
        out = np.zeros_like(hidden_cur)
        for t in range(T):
            for k_idx in range(self.config.top_k):
                eid = int(expert_indices[t, k_idx])
                w = float(gate_weights[t, k_idx])  # type: ignore[index]
                expert_out = np.asarray(expert_fn(eid, hidden_cur[t]), dtype=np.float32)
                out[t] += w * expert_out
        return out

    def load_balancing_loss(
        self, expert_indices: Optional[np.ndarray] = None
    ) -> float:
        """Compute auxiliary load-balancing loss (fraction variance across experts).

        A perfectly balanced router returns 0.0; a fully collapsed router returns ~1.0.

        Args:
            expert_indices: ``(T, top_k)`` int32. Uses last pre-computed if None.

        Returns:
            Scalar load-balancing score.
        """
        if expert_indices is None:
            if self._pre_routed_indices is None:
                raise ValueError("No routing computed. Call route() first.")
            expert_indices = self._pre_routed_indices
        counts = np.bincount(expert_indices.ravel(), minlength=self.config.n_experts).astype(
            np.float32
        )
        counts /= counts.sum() + 1e-9
        uniform = 1.0 / self.config.n_experts
        return float(np.mean((counts - uniform) ** 2))

    def __repr__(self) -> str:
        cfg = self.config
        return (
            f"PreGatedMoERouter(n_experts={cfg.n_experts}, "
            f"top_k={cfg.top_k}, hidden_dim={cfg.hidden_dim})"
        )
