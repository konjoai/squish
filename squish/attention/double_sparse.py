"""squish/attention/double_sparse.py

DoubleSparsityAttn — Two-axis sparsity: head-level + token-level.

Applies two independent sparsity mechanisms simultaneously:

1. **Head-level sparsity** (offline calibration):
   Taylor-expansion importance scores are accumulated over calibration steps
   to produce a binary head mask.  Only the top ``head_keep_ratio`` fraction
   of heads remain active at inference time.

2. **Token-level sparsity** (online, per step):
   For each *active* head, only the ``token_top_k`` keys with the highest
   query-key dot-product score attend to each query.  Remaining positions
   are set to ``-inf`` before softmax.

Reference
---------
Xiao et al., "DuoAttention: Efficient Long-Context LLM Inference with
Retrieval and Streaming Heads." NeurIPS 2024. arXiv:2410.10819.
(Double-sparsity as formalised in) Zhao et al., arXiv:2408.07092, 2024.
"""

from __future__ import annotations

__all__ = ["DoubleSparseConfig", "DoubleSparseState", "DoubleSparsityAttn"]

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from numpy import ndarray


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class DoubleSparseConfig:
    """Configuration for DoubleSparsityAttn.

    Parameters
    ----------
    n_heads:
        Number of attention heads.
    head_dim:
        Dimension of each head.
    token_top_k:
        Number of key positions each query attends to per active head.
    head_keep_ratio:
        Fraction of heads kept after calibration, in ``(0, 1]``.
    calibration_steps:
        Minimum number of calibration calls before head mask is finalised.
    seed:
        RNG seed (unused in forward; kept for API consistency).
    """

    n_heads: int = 8
    head_dim: int = 64
    token_top_k: int = 64
    head_keep_ratio: float = 0.5
    calibration_steps: int = 32
    seed: int = 0

    def __post_init__(self) -> None:
        if self.n_heads < 1:
            raise ValueError("n_heads must be >= 1")
        if self.head_dim < 1:
            raise ValueError("head_dim must be >= 1")
        if self.token_top_k < 1:
            raise ValueError("token_top_k must be >= 1")
        if not (0.0 < self.head_keep_ratio <= 1.0):
            raise ValueError("head_keep_ratio must be in (0, 1]")
        if self.calibration_steps < 1:
            raise ValueError("calibration_steps must be >= 1")


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

@dataclass
class DoubleSparseState:
    """Mutable state for DoubleSparsityAttn.

    Attributes
    ----------
    head_importance:
        Accumulated Taylor importance scores, shape ``(n_heads,)``.
    head_mask:
        Binary mask (1 = active, 0 = pruned), shape ``(n_heads,)``; all-ones
        until calibration is finalised.
    n_calibration_steps:
        Number of calibration steps accumulated so far.
    """

    head_importance: ndarray
    head_mask: ndarray
    n_calibration_steps: int = 0

    @property
    def is_calibrated(self) -> bool:
        """Return True once the head mask has been finalised."""
        return bool(np.any(self.head_mask == 0))


# ---------------------------------------------------------------------------
# Attention module
# ---------------------------------------------------------------------------

class DoubleSparsityAttn:
    """Double-sparse attention: head pruning + token top-K sparsity.

    Parameters
    ----------
    config:
        ``DoubleSparseConfig`` instance.
    """

    def __init__(self, config: DoubleSparseConfig) -> None:
        self.config = config

    def new_state(self) -> DoubleSparseState:
        """Create a fresh state with all heads active."""
        n = self.config.n_heads
        return DoubleSparseState(
            head_importance=np.zeros(n, dtype=np.float32),
            head_mask=np.ones(n, dtype=np.float32),
        )

    def calibrate(
        self, attn_output_grads: ndarray, state: DoubleSparseState
    ) -> DoubleSparseState:
        """Accumulate Taylor importance scores from attention output gradients.

        Taylor importance ≈ |grad ⊙ activation|; here we use the L2 norm of
        ``attn_output_grads`` per head as a proxy.

        Parameters
        ----------
        attn_output_grads:
            Gradient (or activation) tensor, shape ``(n_heads, ...)``.
        state:
            Current state.

        Returns
        -------
        Updated state with accumulated importance scores.
        """
        grads = np.asarray(attn_output_grads, dtype=np.float32)
        # Flatten all dims except head dimension
        head_scores = np.linalg.norm(
            grads.reshape(self.config.n_heads, -1), axis=-1
        )  # (n_heads,)
        return DoubleSparseState(
            head_importance=state.head_importance + head_scores,
            head_mask=state.head_mask.copy(),
            n_calibration_steps=state.n_calibration_steps + 1,
        )

    def finalise_calibration(self, state: DoubleSparseState) -> DoubleSparseState:
        """Compute head mask from accumulated importance scores.

        Keeps the top ``head_keep_ratio`` fraction of heads.

        Parameters
        ----------
        state:
            State with accumulated calibration scores.

        Returns
        -------
        Updated state with finalised ``head_mask``.
        """
        n_keep = max(1, round(self.config.n_heads * self.config.head_keep_ratio))
        thres_idx = np.argsort(-state.head_importance)[n_keep - 1]
        threshold = state.head_importance[thres_idx]
        mask = (state.head_importance >= threshold).astype(np.float32)
        # Ensure exactly n_keep heads are active (break ties by keeping first)
        active = np.where(mask)[0]
        if len(active) > n_keep:
            mask[active[n_keep:]] = 0.0
        return DoubleSparseState(
            head_importance=state.head_importance.copy(),
            head_mask=mask,
            n_calibration_steps=state.n_calibration_steps,
        )

    def forward(
        self,
        Q: ndarray,
        K: ndarray,
        V: ndarray,
        state: DoubleSparseState,
    ) -> Tuple[ndarray, DoubleSparseState]:
        """Compute doubly-sparse attention.

        Parameters
        ----------
        Q:
            Shape ``(n_heads, T_q, head_dim)``.
        K:
            Shape ``(n_heads, T_k, head_dim)``.
        V:
            Shape ``(n_heads, T_k, head_dim)``.
        state:
            Current ``DoubleSparseState``.

        Returns
        -------
        out:
            Shape ``(n_heads, T_q, head_dim)``.
        state:
            Unchanged state (forward is stateless post-calibration).
        """
        Q = np.asarray(Q, dtype=np.float32)
        K = np.asarray(K, dtype=np.float32)
        V = np.asarray(V, dtype=np.float32)

        n_heads, T_q, head_dim = Q.shape
        n_heads_k, T_k, _ = K.shape
        if n_heads != self.config.n_heads:
            raise ValueError(f"Expected {self.config.n_heads} heads, got {n_heads}")

        scale = head_dim ** -0.5
        token_top_k = min(self.config.token_top_k, T_k)
        out = np.zeros_like(Q)

        for h in range(n_heads):
            if state.head_mask[h] == 0.0:
                continue  # pruned head — output stays zero

            # Token-level top-K sparsity
            scores = scale * (Q[h] @ K[h].T)  # (T_q, T_k)
            # For each query keep only top_k keys
            if token_top_k < T_k:
                thresh_vals = np.sort(scores, axis=-1)[:, -(token_top_k)]  # (T_q,)
                mask = scores < thresh_vals[:, np.newaxis]
                scores = np.where(mask, -1e9, scores)

            # Softmax
            scores_shifted = scores - scores.max(axis=-1, keepdims=True)
            exp_s = np.exp(scores_shifted)
            attn_w = exp_s / exp_s.sum(axis=-1, keepdims=True)

            out[h] = attn_w @ V[h]  # (T_q, head_dim)

        return out, state


# server.py compatibility alias
DoubleSparsityConfig = DoubleSparseConfig
