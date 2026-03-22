"""squish/moe/mega_blocks.py

MegaBlocksSparse — Dropless MoE with Block-Sparse GEMM.

Reference
---------
Gale et al. "MegaBlocks: Efficient Sparse Training through Dynamic Sparse
Matrix Multiplication." MLSys 2023.

Algorithm
---------
Standard MoE routers drop tokens when expert capacity is exceeded.
MegaBlocks avoids this with *dropless* routing: every token is routed to
its top-k experts, and a block-sparse matrix multiply handles the
variable-length batches per expert without padding.

This pure-NumPy simulation captures the essential API:

1. ``MegaBlocksSparse.route(hidden_states)`` — top-k routing returning
   ``(expert_ids, routing_weights)`` of shape ``(n_tokens, top_k)``.
2. ``MegaBlocksSparse.forward(hidden_states)`` — full dropless forward pass,
   outputting ``(n_tokens, hidden_size)`` with no token dropped.

Each expert is a two-layer FFN: ``ReLU(x @ W1) @ W2``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

__all__ = [
    "MegaBlocksConfig",
    "MegaBlocksSparse",
]


@dataclass
class MegaBlocksConfig:
    """Configuration for :class:`MegaBlocksSparse`.

    Attributes:
        n_experts: Number of experts.
        hidden_size: Input/output dimension.
        ffn_dim: Expert FFN inner dimension.
        top_k: Tokens per expert in routing.
        seed: RNG seed.
    """

    n_experts: int = 8
    hidden_size: int = 512
    ffn_dim: int = 2048
    top_k: int = 2
    seed: int = 0

    def __post_init__(self) -> None:
        if self.n_experts < 1:
            raise ValueError(f"n_experts must be >= 1; got {self.n_experts}")
        if self.hidden_size < 1:
            raise ValueError(f"hidden_size must be >= 1; got {self.hidden_size}")
        if self.ffn_dim < 1:
            raise ValueError(f"ffn_dim must be >= 1; got {self.ffn_dim}")
        if self.top_k < 1:
            raise ValueError(f"top_k must be >= 1; got {self.top_k}")
        if self.top_k > self.n_experts:
            raise ValueError(
                f"top_k ({self.top_k}) cannot exceed n_experts ({self.n_experts})"
            )


class MegaBlocksSparse:
    """Dropless MoE layer using block-sparse routing.

    Example::

        cfg = MegaBlocksConfig(n_experts=4, hidden_size=16, ffn_dim=64, top_k=2)
        moe = MegaBlocksSparse(cfg)

        tokens = np.random.randn(8, 16).astype(np.float32)
        expert_ids, weights = moe.route(tokens)
        output = moe.forward(tokens)
        assert output.shape == (8, 16)
    """

    def __init__(self, config: Optional[MegaBlocksConfig] = None) -> None:
        self._cfg = config or MegaBlocksConfig()
        c = self._cfg
        rng = np.random.default_rng(c.seed)
        scale = np.sqrt(2.0 / c.hidden_size)
        # Router: (hidden_size, n_experts)
        self._router_weight: np.ndarray = (
            rng.standard_normal((c.hidden_size, c.n_experts)).astype(np.float32) * scale
        )
        # Experts: each is (W1, W2) where W1=(hidden_size, ffn_dim), W2=(ffn_dim, hidden_size)
        self._expert_w1: List[np.ndarray] = []
        self._expert_w2: List[np.ndarray] = []
        for _ in range(c.n_experts):
            self._expert_w1.append(
                rng.standard_normal((c.hidden_size, c.ffn_dim)).astype(np.float32) * scale
            )
            self._expert_w2.append(
                rng.standard_normal((c.ffn_dim, c.hidden_size)).astype(np.float32) * scale
            )

    @property
    def config(self) -> MegaBlocksConfig:
        return self._cfg

    @property
    def router_weight(self) -> np.ndarray:
        """Router projection ``(hidden_size, n_experts)``."""
        return self._router_weight

    @property
    def expert_weights(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """List of ``(W1, W2)`` per expert."""
        return list(zip(self._expert_w1, self._expert_w2))

    def route(
        self, hidden_states: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute top-k routing assignments.

        Args:
            hidden_states: ``(n_tokens, hidden_size)``.

        Returns:
            * ``expert_ids`` — ``(n_tokens, top_k)`` int32, sorted ascending.
            * ``routing_weights`` — ``(n_tokens, top_k)`` float32
              (softmax-normalised over the selected experts).
        """
        x = np.asarray(hidden_states, dtype=np.float32)
        logits = x @ self._router_weight   # (n_tokens, n_experts)
        k = self._cfg.top_k
        # top-k per token
        top_k_idx = np.argpartition(logits, -k, axis=-1)[:, -k:]  # (n_tokens, k)
        top_k_idx = np.sort(top_k_idx, axis=-1)
        # Gather logits for chosen experts, then softmax for routing weights
        n_tokens = x.shape[0]
        row_idx = np.arange(n_tokens)[:, None]
        top_k_logits = logits[row_idx, top_k_idx]               # (n_tokens, k)
        top_k_logits -= top_k_logits.max(axis=-1, keepdims=True)  # numerically stable
        exp_l = np.exp(top_k_logits)
        routing_weights = (exp_l / exp_l.sum(axis=-1, keepdims=True)).astype(np.float32)
        return top_k_idx.astype(np.int32), routing_weights

    def _expert_forward(self, expert_id: int, x: np.ndarray) -> np.ndarray:
        """Run expert *expert_id* on tokens *x* ``(n, hidden_size)``."""
        W1 = self._expert_w1[expert_id]
        W2 = self._expert_w2[expert_id]
        return np.maximum(0.0, x @ W1) @ W2   # (n, hidden_size)

    def forward(self, hidden_states: np.ndarray) -> np.ndarray:
        """Dropless MoE forward.

        Every token hits exactly ``top_k`` experts.  The weighted sum of
        expert outputs is the final representation.

        Args:
            hidden_states: ``(n_tokens, hidden_size)``.

        Returns:
            ``(n_tokens, hidden_size)`` output.
        """
        x = np.asarray(hidden_states, dtype=np.float32)
        n_tokens = x.shape[0]
        expert_ids, routing_weights = self.route(x)   # (n, k), (n, k)

        output = np.zeros_like(x)
        k = self._cfg.top_k

        # Group tokens by expert for batched GEMM simulation
        for e in range(self._cfg.n_experts):
            # token indices and weights for expert e
            mask = (expert_ids == e)  # (n_tokens, k) bool
            if not mask.any():
                continue
            tok_indices, k_slots = np.where(mask)
            weights = routing_weights[tok_indices, k_slots]   # (m,)
            batch = x[tok_indices]                            # (m, hidden_size)
            out_e = self._expert_forward(e, batch)            # (m, hidden_size)
            np.add.at(output, tok_indices, (weights[:, None] * out_e))

        return output
