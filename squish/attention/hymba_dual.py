"""HymbaDualTrack: parallel mini-SSM + attention tracks per head.

Hymba (arXiv 2411.13676) augments each attention head with a low-rank
recurrent SSM stream.  At each token position the SSM output and the
standard softmax-attention output are summed element-wise before the
multi-head output projection.  This gives O(1) amortised memory per
sequence position for the SSM stream while retaining full-context
recall via the attention stream.

Architecture per head (Jh = d_model // n_heads):
  input  x ∈ ℝᴶʰ
  SSM stream: h_new = diag(decay) ⊙ h + B·x;  y_ssm = C·h_new
  Attn stream:  y_attn = softmax(Q·Kᵀ/√Jh) · V  [across T tokens]
  combined:     y = y_ssm + y_attn
  output:       out = W_out · concat(y₀, …, y_{H-1})

All weights are initialised from the given ``seed`` for reproducibility.

Reference: Dong et al., "Hymba: A Hybrid-head Architecture for Small
Language Models", arXiv 2411.13676 (2024).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

__all__ = [
    "HymbaConfig",
    "HymbaState",
    "HymbaDualTrack",
]


@dataclass
class HymbaConfig:
    """Configuration for HymbaDualTrack.

    Attributes:
        d_model: Total model dimension.
        n_heads: Number of attention / SSM heads.
        head_dim: Dimension per head (must satisfy n_heads * head_dim == d_model).
        d_ssm: SSM state dimension (hidden size of the recurrent state per head).
        ssm_rank: Number of SSM input/output projections per head.
        expand_factor: FFN expansion ratio.
        seed: RNG seed for weight initialisation.
    """

    d_model: int = 256
    n_heads: int = 4
    head_dim: int = 64
    d_ssm: int = 32
    ssm_rank: int = 1
    expand_factor: float = 4.0
    seed: int = 0

    def __post_init__(self) -> None:
        if self.n_heads * self.head_dim != self.d_model:
            raise ValueError(
                f"n_heads ({self.n_heads}) * head_dim ({self.head_dim}) "
                f"must equal d_model ({self.d_model})"
            )
        if self.d_ssm < 1:
            raise ValueError("d_ssm must be >= 1")
        if self.ssm_rank < 1:
            raise ValueError("ssm_rank must be >= 1")
        if self.expand_factor <= 0.0:
            raise ValueError("expand_factor must be positive")


@dataclass
class HymbaState:
    """Recurrent state for all SSM heads.

    Attributes:
        h: List of per-head hidden states, each shaped (d_ssm,).
        n_steps: Number of tokens processed so far.
    """

    h: List[np.ndarray]
    n_steps: int = 0

    @property
    def state_bytes(self) -> int:
        """Memory used by all SSM states in bytes."""
        return sum(arr.nbytes for arr in self.h)


class HymbaDualTrack:
    """Hybrid dual-track layer combining per-head SSM + attention.

    Args:
        config: HymbaConfig specifying model dimensions.
    """

    def __init__(self, config: HymbaConfig) -> None:
        self.config = config
        H, Jh, D = config.n_heads, config.head_dim, config.d_ssm
        rng = np.random.default_rng(config.seed)
        scale = 1.0 / np.sqrt(config.d_model)

        # Shared input projection → Q, K, V for attention stream
        self.W_qkv = rng.normal(0, scale, (3 * config.d_model, config.d_model))

        # SSM per-head projections: B (input→state), C (state→output), decay
        # Stored as (H, d_ssm, head_dim) / (H, head_dim, d_ssm) respectively
        self.W_B = rng.normal(0, scale, (H, D, Jh))   # (H, d_ssm, head_dim)
        self.W_C = rng.normal(0, scale, (H, Jh, D))   # (H, head_dim, d_ssm)
        # log-decay initialised to small negatives for stability
        self.log_decay = rng.uniform(-1.0, -0.1, (H, D))  # (H, d_ssm)

        # Output projection
        self.W_out = rng.normal(0, scale, (config.d_model, config.d_model))
        self.b_out = np.zeros(config.d_model)

        # FFN
        d_ff = int(config.d_model * config.expand_factor)
        self.W_ff1 = rng.normal(0, scale, (d_ff, config.d_model))
        self.W_ff2 = rng.normal(0, scale, (config.d_model, d_ff))
        self.b_ff1 = np.zeros(d_ff)
        self.b_ff2 = np.zeros(config.d_model)

        # LayerNorm parameters
        self.ln_w = np.ones(config.d_model)
        self.ln_b = np.zeros(config.d_model)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def new_state(self) -> HymbaState:
        """Return a zeroed initial state."""
        return HymbaState(
            h=[np.zeros(self.config.d_ssm) for _ in range(self.config.n_heads)],
            n_steps=0,
        )

    def forward(
        self,
        x: np.ndarray,
        state: HymbaState,
    ) -> Tuple[np.ndarray, HymbaState]:
        """Forward pass over a sequence.

        Args:
            x: Input tensor of shape ``(T, d_model)``.
            state: Current recurrent HymbaState.

        Returns:
            Tuple of ``(output, new_state)`` where output is ``(T, d_model)``.
        """
        if x.ndim != 2 or x.shape[1] != self.config.d_model:
            raise ValueError(
                f"x must be (T, d_model={self.config.d_model}), got {x.shape}"
            )
        T = x.shape[0]
        H, Jh, D = self.config.n_heads, self.config.head_dim, self.config.d_ssm

        # ---------- attention stream across all T tokens ----------
        qkv = x @ self.W_qkv.T  # (T, 3*d_model)
        Q = qkv[:, : self.config.d_model].reshape(T, H, Jh)
        K = qkv[:, self.config.d_model : 2 * self.config.d_model].reshape(T, H, Jh)
        V = qkv[:, 2 * self.config.d_model :].reshape(T, H, Jh)

        # Multi-head masked self-attention  (T, H, Jh)
        y_attn = self._mh_attention(Q, K, V)  # (T, H, Jh)

        # ---------- SSM stream per token ----------
        decay = np.exp(self.log_decay)  # (H, D) — positive values in (0,1)
        h = [s.copy() for s in state.h]
        y_ssm = np.zeros((T, H, Jh))

        for t in range(T):
            xt = x[t]  # (d_model,)
            # Reshape per head
            xt_h = xt.reshape(H, Jh)  # (H, Jh)
            for i in range(H):
                h[i] = decay[i] * h[i] + self.W_B[i] @ xt_h[i]  # (d_ssm,)
                y_ssm[t, i] = self.W_C[i] @ h[i]  # (Jh,)

        # ---------- combine & output projection ----------
        y_combined = y_attn + y_ssm  # (T, H, Jh)
        y_flat = y_combined.reshape(T, self.config.d_model)
        out = y_flat @ self.W_out.T + self.b_out  # (T, d_model)

        # LayerNorm + FFN
        out = self._layer_norm(out + x)
        ffn = np.maximum(0, out @ self.W_ff1.T + self.b_ff1)  # ReLU
        out = out + ffn @ self.W_ff2.T + self.b_ff2
        out = self._layer_norm(out)

        new_state = HymbaState(h=h, n_steps=state.n_steps + T)
        return out, new_state

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _mh_attention(
        self,
        Q: np.ndarray,
        K: np.ndarray,
        V: np.ndarray,
    ) -> np.ndarray:
        """Masked multi-head scaled-dot-product attention.

        Args:
            Q: ``(T, H, Jh)``
            K: ``(T, H, Jh)``
            V: ``(T, H, Jh)``

        Returns:
            Attention output ``(T, H, Jh)``.
        """
        T, H, Jh = Q.shape
        scale = 1.0 / np.sqrt(Jh)
        # (H, T, T)
        scores = np.einsum("thd,shd->hts", Q, K) * scale
        # causal mask
        mask = np.tril(np.ones((T, T), dtype=bool))
        scores = np.where(mask[np.newaxis], scores, -1e9)
        # softmax
        scores -= scores.max(axis=-1, keepdims=True)
        weights = np.exp(scores)
        weights /= weights.sum(axis=-1, keepdims=True) + 1e-12
        # (H, T, Jh) → (T, H, Jh)
        out = np.einsum("hts,shd->thd", weights, V)
        return out

    @staticmethod
    def _layer_norm(x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        mean = x.mean(axis=-1, keepdims=True)
        std = x.std(axis=-1, keepdims=True)
        return (x - mean) / (std + eps)
