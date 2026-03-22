"""squish/attention/hgrn2.py

HGRN2 — Gated Linear RNN with Toeplitz State Expansion.

Reference
---------
Qin et al. "HGRN2: Gated Linear RNNs with State Expansion."
COLM 2024 (arXiv 2404.07904).

Algorithm
---------
HGRN (Hierarchical Gated Recurrent Network) v2 extends HGRN1 with
state expansion: each scalar hidden dimension h_t is expanded to a vector
via a lower-triangular Toeplitz interaction matrix, giving richer state
dynamics while remaining O(1) per-token at decode time.

    f_t = sigmoid(W_f @ x_t + b_f)     # forget gate
    i_t = (1 - f_t) * x_t_proj         # input gate
    h_t = f_t * h_{t-1} + i_t          # recurrence (expanded)
    y_t = o_t * phi(h_t)               # gated output

State expansion uses a learnable Toeplitz mixing matrix T of shape
(d_expand, d_model) applied to h_t before the output gate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

__all__ = [
    "HGRN2Config",
    "HGRN2State",
    "HGRN2",
]


@dataclass
class HGRN2Config:
    """Configuration for :class:`HGRN2`.

    Attributes:
        d_model:   Input / output dimension.
        d_expand:  State expansion factor (expanded_dim = d_expand * d_model).
        seed:      RNG seed for weight initialisation.
    """

    d_model: int = 512
    d_expand: int = 2
    seed: int = 0

    def __post_init__(self) -> None:
        if self.d_model < 1:
            raise ValueError(f"d_model must be >= 1; got {self.d_model}")
        if self.d_expand < 1:
            raise ValueError(f"d_expand must be >= 1; got {self.d_expand}")


@dataclass
class HGRN2State:
    """Per-layer recurrent hidden state.

    Attributes:
        h: ``(d_model,)`` hidden state vector.
    """

    h: np.ndarray


class HGRN2:
    """Gated linear RNN with Toeplitz state expansion (HGRN2).

    Example::

        cfg   = HGRN2Config(d_model=128, d_expand=2)
        rnn   = HGRN2(cfg)

        # Prefill
        x   = np.random.randn(16, 128).astype(np.float32)
        y, st = rnn.forward(x)           # y: (16, 128), st: HGRN2State

        # Decode
        x_t = np.random.randn(128).astype(np.float32)
        y_t, st = rnn.step(x_t, st)     # y_t: (128,)
    """

    def __init__(self, config: Optional[HGRN2Config] = None) -> None:
        self._cfg = config or HGRN2Config()
        cfg = self._cfg
        rng = np.random.default_rng(cfg.seed)
        d = cfg.d_model
        de = d * cfg.d_expand
        s = 0.02

        # Forget gate, input proj, output gate
        self._W_f = rng.standard_normal((d, d)).astype(np.float32) * s
        self._b_f = np.zeros(d, dtype=np.float32)
        self._W_i = rng.standard_normal((d, d)).astype(np.float32) * s
        self._W_o = rng.standard_normal((d, d)).astype(np.float32) * s
        self._b_o = np.zeros(d, dtype=np.float32)

        # State expansion: Toeplitz-inspired mixing matrix (d_expand*d, d)
        self._T_mix = rng.standard_normal((de, d)).astype(np.float32) * s
        # Output down-project
        self._W_down = rng.standard_normal((de, d)).astype(np.float32) * s

    @property
    def config(self) -> HGRN2Config:
        return self._cfg

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

    def _silu(self, x: np.ndarray) -> np.ndarray:
        return x * self._sigmoid(x)

    def _recurrent_step(
        self, x_t: np.ndarray, h: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Single token step; returns (y_t, h_new)."""
        f = self._sigmoid(x_t @ self._W_f.T + self._b_f)
        i = (1.0 - f) * (x_t @ self._W_i.T)
        h_new = f * h + i
        # State expansion
        h_exp = self._silu(h_new @ self._T_mix.T)  # (d_expand * d,)
        o = self._sigmoid(x_t @ self._W_o.T + self._b_o)
        # Down-project expanded state and apply output gate
        y = o * (h_exp @ self._W_down)  # (d,)
        return y, h_new

    def forward(
        self,
        x: np.ndarray,
        initial_state: Optional[HGRN2State] = None,
    ) -> Tuple[np.ndarray, HGRN2State]:
        """Sequence forward pass.

        Args:
            x: ``(seq_len, d_model)``.
            initial_state: Optional starting hidden state.

        Returns:
            ``(output, HGRN2State)`` where output is ``(seq_len, d_model)``.
        """
        x = np.asarray(x, dtype=np.float32)
        T = x.shape[0]
        h = initial_state.h.copy() if initial_state else np.zeros(self._cfg.d_model, dtype=np.float32)
        outs = np.zeros((T, self._cfg.d_model), dtype=np.float32)
        for t in range(T):
            outs[t], h = self._recurrent_step(x[t], h)
        return outs, HGRN2State(h=h)

    def step(
        self,
        x_t: np.ndarray,
        state: Optional[HGRN2State] = None,
    ) -> Tuple[np.ndarray, HGRN2State]:
        """Single-token decode step.

        Args:
            x_t: ``(d_model,)`` input token embedding.
            state: Previous hidden state, or None for fresh state.

        Returns:
            ``(output, HGRN2State)`` where output is ``(d_model,)``.
        """
        x_t = np.asarray(x_t, dtype=np.float32).ravel()
        h = state.h.copy() if state else np.zeros(self._cfg.d_model, dtype=np.float32)
        y, h_new = self._recurrent_step(x_t, h)
        return y.astype(np.float32), HGRN2State(h=h_new)

    def init_state(self) -> HGRN2State:
        """Return a fresh zero hidden state."""
        return HGRN2State(h=np.zeros(self._cfg.d_model, dtype=np.float32))
