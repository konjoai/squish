"""Hawk/Griffin Real-Gated Linear Recurrence (RGLR) layer.

Hawk is the pure-recurrent sub-model within the Griffin architecture (De et al.,
NeurIPS 2024).  Its state update uses real-valued diagonal matrices — input and
forget gates applied element-wise — making the scan trivially vectorisable
without the complex-eigenvalue bookkeeping of S4/S6.

Griffin interleaves Hawk blocks with local-attention blocks (not implemented
here); this module provides the RGLR Cell that is the core primitive.

Reference: De et al., "Griffin: Mixing Gated Linear Recurrences with Local
Attention for Efficient LLMs" arXiv 2402.19427, NeurIPS 2024.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

__all__ = [
    "HawkConfig",
    "HawkState",
    "HawkLinearRNN",
]


@dataclass
class HawkConfig:
    """Configuration for :class:`HawkLinearRNN`.

    Attributes:
        d_model: Model dimension.
        d_state: Recurrent state dimension per layer (can differ from d_model
            if a linear projection is applied before the cell).
        expand_factor: FFN expansion ratio inside the Hawk block.
        dt_min: Minimum value after softplus for the time-step parameter.
        seed: RNG seed for weight initialisation.
    """

    d_model: int = 512
    d_state: int = 512
    expand_factor: float = 4.0
    dt_min: float = 1e-4
    seed: int = 0

    def __post_init__(self) -> None:
        if self.d_model < 1:
            raise ValueError(f"d_model must be ≥ 1, got {self.d_model}")
        if self.d_state < 1:
            raise ValueError(f"d_state must be ≥ 1, got {self.d_state}")
        if self.dt_min <= 0.0:
            raise ValueError(f"dt_min must be > 0, got {self.dt_min}")


@dataclass
class HawkState:
    """Per-layer recurrent state for the Hawk RGLR cell.

    Attributes:
        h: Hidden state vector ``(d_state,)`` — the sole carried context.
        n_steps: Number of decode steps taken with this state.
    """

    h: np.ndarray  # (d_state,)
    n_steps: int = 0

    @property
    def state_bytes(self) -> int:
        return self.h.nbytes


class HawkLinearRNN:
    """Real-Gated Linear Recurrence layer (Hawk).

    The recurrence is::

        α_t = sigmoid(W_α · x_t + b_α)     # input gate
        β_t = sigmoid(W_β · x_t + b_β)     # forget gate (learned decay)
        h_t = β_t ⊙ h_{t-1} + α_t ⊙ x_proj_t
        y_t = h_t ⊙ sigmoid(W_γ · x_t)    # output gate

    The scan over a prefill sequence runs element-by-element on CPU (a Metal
    SIMD-group parallel scan kernel would accelerate this path in production).

    Usage::

        cfg = HawkConfig(d_model=256, d_state=256)
        hawk = HawkLinearRNN(cfg)
        state = hawk.new_state()
        out, state = hawk.forward(x_seq, state)  # x_seq: (T, d_model)
    """

    def __init__(self, config: HawkConfig) -> None:
        self.config = config
        rng = np.random.default_rng(config.seed)
        d, ds = config.d_model, config.d_state
        scale = 1.0 / np.sqrt(d)
        self.W_input = rng.standard_normal((d, ds)).astype(np.float32) * scale
        self.W_alpha = rng.standard_normal((d, ds)).astype(np.float32) * scale
        self.b_alpha = np.zeros(ds, dtype=np.float32)
        self.W_beta = rng.standard_normal((d, ds)).astype(np.float32) * scale
        self.b_beta = np.zeros(ds, dtype=np.float32)
        self.W_gamma = rng.standard_normal((d, ds)).astype(np.float32) * scale
        self.W_out = rng.standard_normal((ds, d)).astype(np.float32) * scale
        # FFN
        ffn_dim = int(d * config.expand_factor)
        self.W_ff1 = rng.standard_normal((d, ffn_dim)).astype(np.float32) * scale
        self.W_ff2 = rng.standard_normal((ffn_dim, d)).astype(np.float32) * scale

    # ------------------------------------------------------------------
    # Primary API
    # ------------------------------------------------------------------

    def new_state(self) -> HawkState:
        return HawkState(h=np.zeros(self.config.d_state, dtype=np.float32))

    def forward(
        self,
        x: np.ndarray,
        state: HawkState,
    ) -> Tuple[np.ndarray, HawkState]:
        """Forward over a sequence.

        Args:
            x: ``(seq_len, d_model)`` input.
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

    def recurrent_step(
        self, x: np.ndarray, state: HawkState
    ) -> Tuple[np.ndarray, HawkState]:
        """Convenience alias for single-token decode."""
        out = self._step(np.asarray(x, dtype=np.float32), state)
        return out, state

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x))

    def _step(self, xt: np.ndarray, state: HawkState) -> np.ndarray:
        alpha = self._sigmoid(xt @ self.W_alpha + self.b_alpha)
        beta = self._sigmoid(xt @ self.W_beta + self.b_beta)
        x_proj = xt @ self.W_input
        state.h = beta * state.h + alpha * x_proj
        gamma = self._sigmoid(xt @ self.W_gamma)
        y = (state.h * gamma) @ self.W_out
        # FFN residual
        ffn = np.maximum(0.0, xt @ self.W_ff1) @ self.W_ff2
        state.n_steps += 1
        return y + ffn

    def scan_prefill(
        self, x: np.ndarray, h0: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return all hidden states and final state without modifying mutable state.

        Args:
            x: ``(seq_len, d_model)`` input.
            h0: Optional initial state ``(d_state,)``; zeros if not supplied.

        Returns:
            ``(all_outputs (seq_len, d_model), h_final (d_state,))``.
        """
        from typing import Optional  # local import to avoid circular
        state = HawkState(
            h=np.zeros(self.config.d_state, dtype=np.float32) if h0 is None else h0.copy()
        )
        out, _ = self.forward(x, state)
        return out, state.h


from typing import Optional  # noqa: E402  (needed for scan_prefill signature)
