"""squish/moe/expert_choice.py

ExpertChoiceRouter — Expert-selects-tokens routing for MoE.

Reference
---------
Zhou et al. "Mixture-of-Experts with Expert Choice Routing."
NeurIPS 2022. arXiv:2202.09368. (Widely deployed in production 2024.)

Algorithm
---------
In standard top-1/top-2 token-selects-expert routing, some experts may
receive far too many tokens (overload) and others too few (underload),
causing quality degradation via token-dropping.

Expert Choice inverts the selection:

  1. Compute router logits: logits = X @ W_r   (n_tokens × n_experts)
  2. Softmax over expert dimension (per-token expert probabilities).
  3. For each expert e, select the top-k tokens with highest probability
     for that expert (k = expert_capacity = floor(n_tokens * capacity / n_experts)).
  4. Each expert processes exactly k tokens — perfectly load-balanced.
  5. Combine using the routing weights (weighted sum of expert outputs).

Tokens may appear in multiple experts (unlike top-1 routing), which
improves quality but increases computation by ``capacity`` factor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class ExpertChoiceConfig:
    """Configuration for ExpertChoiceRouter.

    Parameters
    ----------
    n_experts:
        Total number of experts.
    capacity_factor:
        Average tokens per expert = floor(n_tokens * capacity_factor / n_experts).
        Typical value: 1.25–2.0.
    hidden_size:
        Hidden dimension fed into the router.
    seed:
        RNG seed for router weight initialization.
    """

    n_experts: int = 8
    capacity_factor: float = 1.25
    hidden_size: int = 4096
    seed: int = 0

    def __post_init__(self) -> None:
        if self.n_experts < 1:
            raise ValueError("n_experts must be >= 1")
        if self.capacity_factor <= 0:
            raise ValueError("capacity_factor must be > 0")


# ---------------------------------------------------------------------------
# Routing result
# ---------------------------------------------------------------------------

@dataclass
class ExpertChoiceResult:
    """Result of expert-choice routing.

    Parameters
    ----------
    token_indices:
        Shape ``(n_experts, expert_capacity)`` — which token each expert slot
        processes.  Values are indices into the token dimension [0, n_tokens).
    routing_weights:
        Shape ``(n_experts, expert_capacity)`` — softmax weight for each
        token-expert assignment.
    expert_capacity:
        Number of tokens per expert.
    router_probs:
        Full softmax probabilities ``(n_tokens, n_experts)``.
    """

    token_indices: np.ndarray
    routing_weights: np.ndarray
    expert_capacity: int
    router_probs: np.ndarray

    def load_balance_loss(self) -> float:
        """Auxiliary load-balancing loss (always 0 - EC is perfectly balanced)."""
        return 0.0

    def tokens_per_expert(self) -> np.ndarray:
        """Number of *unique* tokens assigned to each expert.

        Returns
        -------
        np.ndarray
            Shape ``(n_experts,)``.
        """
        n_experts = self.token_indices.shape[0]
        return np.full(n_experts, self.expert_capacity, dtype=np.int32)


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

class ExpertChoiceRouter:
    """Expert-Choice MoE router.

    Parameters
    ----------
    config:
        ExpertChoiceRouter configuration.
    """

    def __init__(self, config: Optional[ExpertChoiceConfig] = None) -> None:
        self._cfg = config or ExpertChoiceConfig()
        rng = np.random.default_rng(self._cfg.seed)
        # Router projection: hidden_size → n_experts
        self._W_r = rng.standard_normal(
            (self._cfg.hidden_size, self._cfg.n_experts)
        ).astype(np.float32) * (self._cfg.hidden_size ** -0.5)

    @property
    def config(self) -> ExpertChoiceConfig:
        return self._cfg

    @property
    def router_weight(self) -> np.ndarray:
        """Router projection matrix ``(hidden_size, n_experts)``."""
        return self._W_r

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        logits = logits - logits.max(axis=-1, keepdims=True)
        exp = np.exp(logits)
        return exp / exp.sum(axis=-1, keepdims=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def route(self, hidden_states: np.ndarray) -> ExpertChoiceResult:
        """Route a batch of token hidden states via expert-choice.

        Parameters
        ----------
        hidden_states:
            Shape ``(n_tokens, hidden_size)`` or ``(batch, seq, hidden_size)``.
            3-D inputs are flattened to ``(batch*seq, hidden_size)``.

        Returns
        -------
        ExpertChoiceResult
        """
        X = np.asarray(hidden_states, dtype=np.float32)
        if X.ndim == 3:
            X = X.reshape(-1, X.shape[-1])

        n_tokens, hidden = X.shape

        # Router logits and probabilities: (n_tokens, n_experts)
        logits = X @ self._W_r
        probs = self._softmax(logits)  # (n_tokens, n_experts)

        # Expert capacity
        capacity = max(1, int(n_tokens * self._cfg.capacity_factor / self._cfg.n_experts))

        # Each expert selects its top-k tokens
        # probs transposed: (n_experts, n_tokens)
        probs_T = probs.T
        top_k_indices = np.argsort(probs_T, axis=1)[:, -capacity:][:, ::-1]  # (n_experts, capacity)
        top_k_weights = np.take_along_axis(probs_T, top_k_indices, axis=1)   # (n_experts, capacity)

        return ExpertChoiceResult(
            token_indices=top_k_indices.astype(np.int32),
            routing_weights=top_k_weights,
            expert_capacity=capacity,
            router_probs=probs,
        )

    def combine(
        self,
        expert_outputs: np.ndarray,
        routing_result: ExpertChoiceResult,
        n_tokens: int,
    ) -> np.ndarray:
        """Combine expert outputs back into the token dimension.

        Parameters
        ----------
        expert_outputs:
            Shape ``(n_experts, expert_capacity, expert_output_dim)``.
        routing_result:
            ExpertChoiceResult from ``route()``.
        n_tokens:
            Total number of input tokens.

        Returns
        -------
        np.ndarray
            Shape ``(n_tokens, expert_output_dim)``.
        """
        n_experts, capacity, d = expert_outputs.shape
        out = np.zeros((n_tokens, d), dtype=np.float32)
        weights = routing_result.routing_weights   # (n_experts, capacity)
        indices = routing_result.token_indices     # (n_experts, capacity)

        for e in range(n_experts):
            for c in range(capacity):
                tok = int(indices[e, c])
                w = float(weights[e, c])
                out[tok] += w * expert_outputs[e, c]

        return out
