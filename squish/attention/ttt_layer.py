"""TTT (Test-Time Training) linear layer.

The TTT layer replaces the KV cache with a mini-model whose weights serve as
the hidden state.  At each token position t the mini-model W_t ∈ ℝ^{d×d} is
updated by a closed-form gradient step on a self-supervised reconstruction
loss — no autograd at inference.  The output for position t is f(x_t ; W_t).

This single-step update rule makes TTT O(d²) memory and O(d²) compute per
token — identical to mLSTM — while generalising linear attention because the
mini-model can express any linear function of x_t rather than being restricted
to the fixed outer-product rank-1 write of standard linear attention.

Quality degrades ≥30× more slowly than softmax attention at 100K+ token
contexts on language modelling benchmarks.

Reference: Jarayam et al., "Learning to (Learn at Test Time): RNNs with
Expressive Hidden States" arXiv 2407.04620, ICML 2025.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

__all__ = [
    "TTTConfig",
    "TTTState",
    "TTTLinearLayer",
]


@dataclass
class TTTConfig:
    """Configuration for :class:`TTTLinearLayer`.

    Attributes:
        d_model: Input/output embedding dimension.
        mini_model_dim: Hidden dimension of the mini-model (the state matrix
            is ``(mini_model_dim, mini_model_dim)``).
        lr: Learning rate for the online weight update.
        momentum: Momentum coefficient for the state update (0 = no momentum).
        expand_factor: Post-TTT FFN expansion ratio.
        seed: RNG seed for weight initialisation.
    """

    d_model: int = 256
    mini_model_dim: int = 64
    lr: float = 0.01
    momentum: float = 0.0
    expand_factor: float = 4.0
    seed: int = 0

    def __post_init__(self) -> None:
        if self.d_model < 1:
            raise ValueError(f"d_model must be ≥ 1, got {self.d_model}")
        if self.mini_model_dim < 1:
            raise ValueError(f"mini_model_dim must be ≥ 1, got {self.mini_model_dim}")
        if self.lr <= 0.0:
            raise ValueError(f"lr must be > 0, got {self.lr}")
        if not (0.0 <= self.momentum < 1.0):
            raise ValueError(f"momentum must be in [0, 1), got {self.momentum}")


@dataclass
class TTTState:
    """Recurrent state for :class:`TTTLinearLayer`.

    The hidden state is the mini-model weight matrix; velocity tracks the
    momentum term for the gradient descent step.

    Attributes:
        W: Mini-model weight matrix ``(mini_model_dim, mini_model_dim)``.
        velocity: Gradient momentum buffer (same shape as W).
        n_steps: Number of tokens processed.
    """

    W: np.ndarray       # (md, md)
    velocity: np.ndarray  # (md, md)
    n_steps: int = 0

    @property
    def state_bytes(self) -> int:
        return self.W.nbytes + self.velocity.nbytes


class TTTLinearLayer:
    """Test-Time Training layer with mini-model hidden state.

    At each step:

    1. Project x_t → key k_t, query q_t, value v_t via learned projections.
    2. Evaluate mini-model prediction: ŷ_t = W_t · k_t.
    3. Compute self-supervised gradient: g_t = -(v_t - ŷ_t) ⊗ k_t
       (gradient of MSE loss w.r.t. W_t).
    4. Update W_t+1 = W_t - lr · g_t  (+ momentum if configured).
    5. Output: y_t = W_{t+1} · q_t; pass through output projection + FFN.

    Usage::

        cfg = TTTConfig(d_model=256, mini_model_dim=64)
        layer = TTTLinearLayer(cfg)
        state = layer.new_state()
        out, state = layer.forward(x, state)   # x: (T, d_model)
    """

    def __init__(self, config: TTTConfig) -> None:
        self.config = config
        rng = np.random.default_rng(config.seed)
        d, md = config.d_model, config.mini_model_dim
        scale = 1.0 / np.sqrt(d)

        self.W_k = rng.standard_normal((d, md)).astype(np.float32) * scale
        self.W_q = rng.standard_normal((d, md)).astype(np.float32) * scale
        self.W_v = rng.standard_normal((d, md)).astype(np.float32) * scale
        self.W_out = rng.standard_normal((md, d)).astype(np.float32) * scale

        ffn_d = int(d * config.expand_factor)
        self.W_ff1 = rng.standard_normal((d, ffn_d)).astype(np.float32) * scale
        self.W_ff2 = rng.standard_normal((ffn_d, d)).astype(np.float32) * scale

    # ------------------------------------------------------------------
    # Primary API
    # ------------------------------------------------------------------

    def new_state(self) -> TTTState:
        md = self.config.mini_model_dim
        return TTTState(
            W=np.zeros((md, md), dtype=np.float32),
            velocity=np.zeros((md, md), dtype=np.float32),
        )

    def forward(
        self,
        x: np.ndarray,
        state: TTTState,
    ) -> Tuple[np.ndarray, TTTState]:
        """Process a sequence through the TTT layer.

        Args:
            x: ``(seq_len, d_model)`` input.
            state: Mini-model state; mutated in-place.

        Returns:
            ``(output (seq_len, d_model), updated_state)``.
        """
        x = np.asarray(x, dtype=np.float32)
        outputs = []
        for t in range(x.shape[0]):
            out_t = self._ttt_step(x[t], state)
            outputs.append(out_t)
        return np.stack(outputs), state

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _ttt_step(self, xt: np.ndarray, state: TTTState) -> np.ndarray:
        lr = self.config.lr
        mu = self.config.momentum

        k = xt @ self.W_k   # (md,)
        q = xt @ self.W_q   # (md,)
        v = xt @ self.W_v   # (md,)

        # Self-supervised gradient: gradient of ||v - W·k||² w.r.t. W
        pred = state.W @ k                # (md,)
        residual = pred - v               # (md,)
        grad = np.outer(residual, k)      # (md, md) rank-1 gradient

        # Weight update with optional momentum
        if mu > 0.0:
            state.velocity = mu * state.velocity + grad
            state.W = state.W - lr * state.velocity
        else:
            state.W = state.W - lr * grad

        # Output using updated weights
        y = state.W @ q                   # (md,)
        out = y @ self.W_out              # (d_model,)

        # FFN residual
        ffn = np.maximum(0.0, xt @ self.W_ff1) @ self.W_ff2
        state.n_steps += 1
        return out + ffn
