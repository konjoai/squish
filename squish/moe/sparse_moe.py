# [Experimental] This module is part of Squish v6+ (Waves 19–26).
# Proof-of-concept quality: API and behaviour may change without notice.
# For stable core functionality see Waves 1–12 in MODULES.md.
"""SparseMoE — Sparse Mixture-of-Experts routing with load-balance loss.

Each token is routed to the ``top_k`` experts with the highest gating score.
A softmax gate over all ``n_experts`` is computed; the top-k selected experts
receive renormalised weights that sum to 1.0.

An auxiliary load-balancing loss penalises routing collapse (the pathological
case where all tokens flow to a single expert):

    aux_loss = load_balance_weight * n_experts
               * Σ_e mean_prob[e] * freq_fraction[e]

where ``mean_prob[e]`` is the mean softmax probability assigned to expert ``e``
across the batch and ``freq_fraction[e]`` is the fraction of tokens routed to
expert ``e`` by argmax (winner-takes-all, for ease of differentiation).

Reference:
    Fedus et al., "Switch Transformers: Scaling to Trillion Parameter Models
    with Simple and Efficient Sparsity", JMLR 2022.
    https://arxiv.org/abs/2101.03961

Usage::

    import numpy as np
    from squish.moe.sparse_moe import MoEConfig, SparseMoERouter, MoEStats

    cfg    = MoEConfig(n_experts=8, top_k=2, hidden_dim=256)
    router = SparseMoERouter(cfg)

    rng          = np.random.default_rng(42)
    hidden_states = rng.standard_normal((16, 256)).astype(np.float32)

    indices, weights, aux_loss = router.route(hidden_states)
    print(indices.shape)   # (16, 2)
    print(weights.shape)   # (16, 2)
    print(f"aux_loss={aux_loss:.4f}")
    print(router.stats)
"""

from __future__ import annotations

__all__ = ["MoEConfig", "SparseMoERouter", "MoEStats"]

from dataclasses import dataclass

import numpy as np


@dataclass
class MoEConfig:
    """Configuration for Sparse MoE routing.

    Attributes:
        n_experts: Total number of expert networks.
        top_k: Number of experts each token is routed to.
        hidden_dim: Dimension of the input hidden states.
        load_balance_weight: Coefficient for the auxiliary load-balancing loss.
    """

    n_experts:           int   = 8
    top_k:               int   = 2
    hidden_dim:          int   = 256
    load_balance_weight: float = 0.01

    def __post_init__(self) -> None:
        if self.top_k < 1:
            raise ValueError(
                f"top_k must be >= 1; got {self.top_k}"
            )
        if self.n_experts < self.top_k:
            raise ValueError(
                f"n_experts ({self.n_experts}) must be >= top_k ({self.top_k})"
            )
        if self.hidden_dim < 1:
            raise ValueError(
                f"hidden_dim must be >= 1; got {self.hidden_dim}"
            )


@dataclass
class MoEStats:
    """Running statistics for a :class:`SparseMoERouter` session.

    Attributes:
        total_route_calls: Number of :meth:`SparseMoERouter.route` calls.
        total_tokens: Cumulative number of tokens routed.
    """

    total_route_calls: int = 0
    total_tokens:      int = 0


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax along *axis*."""
    x_shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x     = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


class SparseMoERouter:
    """Top-k sparse MoE router with auxiliary load-balancing loss.

    The gate weight matrix ``W_g`` is initialised with Xavier uniform scaling
    appropriate for a ``(hidden_dim → n_experts)`` projection.

    Args:
        config: :class:`MoEConfig` controlling routing behaviour.
    """

    def __init__(self, config: MoEConfig) -> None:
        self._config = config
        self._stats  = MoEStats()

        # Xavier initialisation: scale = sqrt(2 / (fan_in + fan_out))
        fan_in  = config.hidden_dim
        fan_out = config.n_experts
        scale   = np.sqrt(2.0 / (fan_in + fan_out))
        rng = np.random.default_rng(0)
        self.gate_weights: np.ndarray = (
            rng.standard_normal((fan_in, fan_out)).astype(np.float32) * scale
        )

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def route(
        self,
        hidden_states: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Route a batch of hidden states to the top-k experts.

        Args:
            hidden_states: Float32 array of shape ``(batch, hidden_dim)`` or
                ``(seq_len, hidden_dim)``.

        Returns:
            Tuple ``(top_k_indices, top_k_weights, aux_loss)`` where:

            * ``top_k_indices`` — int64 array of shape ``(batch, top_k)``
              containing the selected expert indices for each token.
            * ``top_k_weights`` — float32 array of shape ``(batch, top_k)``
              containing renormalised gating weights (sum per token = 1.0).
            * ``aux_loss`` — scalar float; auxiliary load-balancing penalty.

        Raises:
            ValueError: If ``hidden_states`` has the wrong number of
                dimensions or the feature dimension does not match
                ``config.hidden_dim``.
        """
        hidden_states = np.asarray(hidden_states, dtype=np.float32)
        if hidden_states.ndim != 2:
            raise ValueError(
                f"hidden_states must be 2-D (batch, hidden_dim); "
                f"got shape {hidden_states.shape}"
            )
        batch, hdim = hidden_states.shape
        if hdim != self._config.hidden_dim:
            raise ValueError(
                f"hidden_states feature dim ({hdim}) must match "
                f"hidden_dim ({self._config.hidden_dim})"
            )

        cfg = self._config

        # Gate logits and probabilities
        gate_logits = hidden_states @ self.gate_weights  # (batch, n_experts)
        gate_probs  = _softmax(gate_logits, axis=-1)     # (batch, n_experts)

        # Top-k selection: descending argsort, keep first top_k
        sorted_idx    = np.argsort(-gate_probs, axis=-1)        # (batch, n_experts)
        top_k_indices = sorted_idx[:, :cfg.top_k].astype(np.int64)  # (batch, top_k)

        # Gather top-k probabilities
        top_k_probs = np.take_along_axis(gate_probs, top_k_indices, axis=1)

        # Renormalise so each token's weights sum to 1
        prob_sum     = np.sum(top_k_probs, axis=1, keepdims=True)
        safe_sum     = np.where(prob_sum < 1e-12, 1.0, prob_sum)
        top_k_weights = (top_k_probs / safe_sum).astype(np.float32)

        # ── Auxiliary load-balancing loss ─────────────────────────────
        # mean_probs: average softmax probability per expert (n_experts,)
        mean_probs = np.mean(gate_probs, axis=0)  # (n_experts,)

        # freq_fraction: fraction of tokens routed to each expert by argmax
        expert_assignments = np.argmax(gate_probs, axis=1)  # (batch,)
        freq_fraction = np.zeros(cfg.n_experts, dtype=np.float32)
        for e in range(cfg.n_experts):
            freq_fraction[e] = float(np.sum(expert_assignments == e)) / batch

        # Dot product of mean_probs and freq_fraction, scaled by n_experts
        aux_loss = float(
            cfg.load_balance_weight
            * cfg.n_experts
            * float(np.dot(mean_probs, freq_fraction))
        )

        self._stats.total_route_calls += 1
        self._stats.total_tokens      += batch

        return top_k_indices, top_k_weights, aux_loss

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def stats(self) -> MoEStats:
        """Running routing statistics."""
        return self._stats
