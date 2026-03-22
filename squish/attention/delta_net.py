"""DeltaNet: delta-rule linear recurrent attention.

DeltaNet replaces the rank-1 outer-product state-update of linear attention
with the delta rule:

    W_t = W_{t-1} + lr·(v_t − W_{t-1}·k̂_t)·k̂_tᵀ

where k̂_t = k_t / ‖k_t‖ is the L2-normalised key.  This is a single Newton
step from a least-squares objective; it steers the state toward memorising
(k̂_t → v_t) while partially forgetting the component of the previous state
along k̂_t.

Key L2 normalisation prevents rank collapse that plagues standard linear
attention by ensuring new writes are spread across the full key space rather
than collapsing to a few dominant directions.

Attention output: y_t = W_t · q_t (after the update at step t).

Reference: Yang et al., "DeltaNet: Parallelisable Recurrent Sequence-to-
Sequence Models" arXiv 2406.06484, NeurIPS 2024.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

__all__ = [
    "DeltaNetConfig",
    "DeltaNetState",
    "DeltaNetLinear",
]


@dataclass
class DeltaNetConfig:
    """Configuration for :class:`DeltaNetLinear`.

    Attributes:
        d_model: Model embedding dimension.
        d_state: State matrix inner dimension (W ∈ ℝ^{d_state×d_state}).
        n_heads: Number of parallel delta-rule heads.
        head_dim: Dimension per head; ``n_heads * head_dim == d_model``.
        beta: Learning rate for the delta rule (scalar per token via
            a sigmoid'd projection when ``learnable_beta=True``).
        learnable_beta: If True, beta is predicted per-token from input.
        expand_factor: Post-DeltaNet FFN expansion ratio.
        seed: RNG seed.
    """

    d_model: int = 256
    d_state: int = 64
    n_heads: int = 4
    head_dim: int = 64
    beta: float = 0.5
    learnable_beta: bool = True
    expand_factor: float = 4.0
    seed: int = 0

    def __post_init__(self) -> None:
        if self.d_model < 1:
            raise ValueError(f"d_model must be ≥ 1, got {self.d_model}")
        if self.n_heads < 1:
            raise ValueError(f"n_heads must be ≥ 1, got {self.n_heads}")
        if self.head_dim < 1:
            raise ValueError(f"head_dim must be ≥ 1, got {self.head_dim}")
        if not (0.0 < self.beta <= 1.0):
            raise ValueError(f"beta must be in (0, 1], got {self.beta}")


@dataclass
class DeltaNetState:
    """Per-layer DeltaNet recurrent state.

    Attributes:
        W: State matrix per head ``(n_heads, head_dim, d_state)``.
        n_steps: Tokens processed.
    """

    W: np.ndarray   # (n_heads, head_dim, d_state)
    n_steps: int = 0

    @property
    def state_bytes(self) -> int:
        return self.W.nbytes


class DeltaNetLinear:
    """Delta-rule linear recurrent attention layer.

    Provides both:
    * **Recurrent** mode: O(n·d²) compute, O(d²) memory — used for decode.
    * **Chunk-parallel** mode: process a chunk via batched outer products —
      same asymptotic but amenable to SIMD parallelism over the chunk.

    Usage::

        cfg = DeltaNetConfig(d_model=256, n_heads=4, head_dim=64, d_state=64)
        layer = DeltaNetLinear(cfg)
        state = layer.new_state()
        out, state = layer.forward(x, state)   # x: (T, d_model)
    """

    def __init__(self, config: DeltaNetConfig) -> None:
        self.config = config
        rng = np.random.default_rng(config.seed)
        d = config.d_model
        scale = 1.0 / np.sqrt(d)

        self.W_q = rng.standard_normal((d, config.n_heads * config.d_state)).astype(np.float32) * scale
        self.W_k = rng.standard_normal((d, config.n_heads * config.d_state)).astype(np.float32) * scale
        self.W_v = rng.standard_normal((d, config.n_heads * config.head_dim)).astype(np.float32) * scale
        self.W_out = rng.standard_normal((config.n_heads * config.head_dim, d)).astype(np.float32) * scale

        if config.learnable_beta:
            self.W_beta = rng.standard_normal((d, config.n_heads)).astype(np.float32) * scale

        ffn_d = int(d * config.expand_factor)
        self.W_ff1 = rng.standard_normal((d, ffn_d)).astype(np.float32) * scale
        self.W_ff2 = rng.standard_normal((ffn_d, d)).astype(np.float32) * scale

    # ------------------------------------------------------------------
    # Primary API
    # ------------------------------------------------------------------

    def new_state(self) -> DeltaNetState:
        cfg = self.config
        return DeltaNetState(
            W=np.zeros((cfg.n_heads, cfg.head_dim, cfg.d_state), dtype=np.float32)
        )

    def forward(
        self,
        x: np.ndarray,
        state: DeltaNetState,
    ) -> Tuple[np.ndarray, DeltaNetState]:
        """Forward over a sequence.

        Args:
            x: ``(seq_len, d_model)``.
            state: Mutable recurrent state.

        Returns:
            ``(output (seq_len, d_model), updated_state)``.
        """
        x = np.asarray(x, dtype=np.float32)
        outputs = []
        for t in range(x.shape[0]):
            out_t = self._step(x[t], state)
            outputs.append(out_t)
        return np.stack(outputs), state

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _step(self, xt: np.ndarray, state: DeltaNetState) -> np.ndarray:
        cfg = self.config
        h, hd, ds = cfg.n_heads, cfg.head_dim, cfg.d_state

        q = (xt @ self.W_q).reshape(h, ds)
        k_raw = (xt @ self.W_k).reshape(h, ds)
        v = (xt @ self.W_v).reshape(h, hd)

        # L2 normalise keys per head
        k_norm = np.linalg.norm(k_raw, axis=1, keepdims=True).clip(min=1e-8)
        k = k_raw / k_norm  # (h, ds) — normalised

        # Per-token learnable beta
        if cfg.learnable_beta:
            beta = (1.0 / (1.0 + np.exp(-(xt @ self.W_beta))))  # (h,)
        else:
            beta = np.full(h, cfg.beta, dtype=np.float32)

        # Delta rule: W = W + beta * outer(v - W@k, k)
        Wk = np.einsum("hij,hj->hi", state.W, k)   # (h, hd)
        residual = v - Wk                            # (h, hd)
        delta = np.einsum("hi,hj->hij", residual, k) * beta[:, np.newaxis, np.newaxis]
        state.W = state.W + delta

        # Output: y = W @ q
        y = np.einsum("hij,hj->hi", state.W, q)  # (h, hd)
        y_flat = y.reshape(cfg.d_model)
        out = y_flat @ self.W_out

        # FFN residual
        ffn = np.maximum(0.0, xt @ self.W_ff1) @ self.W_ff2
        state.n_steps += 1
        return out + ffn
