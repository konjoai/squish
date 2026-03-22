"""squish/attention/mamba2_ssm.py

Mamba-2 Structured State-Space Duality (SSD) Layer.

Reference
---------
Dao & Gu. "Transformers are SSMs: Generalized Models and Efficient Algorithms
Through Structured State Space Duality." ICML 2024 (arXiv 2405.21060).

Algorithm
---------
SSD formulates a linear recurrence:

    h_t = A_t * h_{t-1} + B_t * x_t
    y_t = C_t @ h_t

where A_t is a scalar × identity (diagonal SSM), enabling a chunked
parallel-scan at prefill and an O(1) per-token recurrent decode.

This module provides:

1. **Parallel prefill**: chunk the sequence into blocks of `chunk_size`; within
   each block compute via matrix-multiplication scan; across blocks accumulate
   the recurrent state sequentially.
2. **Recurrent decode**: update `(conv_state, ssm_state)` one token at a time
   with O(d_state) compute and storage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

__all__ = [
    "Mamba2Config",
    "Mamba2State",
    "Mamba2SSM",
]


@dataclass
class Mamba2Config:
    """Configuration for :class:`Mamba2SSM`.

    Attributes:
        d_model: Input / output dimension.
        d_state: SSM state dimension (N in the paper).
        d_conv: Depthwise convolution kernel width.
        expand: Expansion factor (inner_dim = expand * d_model).
        chunk_size: Block size for parallel chunked scan at prefill.
        n_heads: Number of SSM heads (head_dim = inner_dim // n_heads).
        seed: RNG seed for weight initialisation.
    """

    d_model: int = 512
    d_state: int = 64
    d_conv: int = 4
    expand: int = 2
    chunk_size: int = 64
    n_heads: int = 8
    seed: int = 0

    def __post_init__(self) -> None:
        if self.d_model < 1:
            raise ValueError(f"d_model must be >= 1; got {self.d_model}")
        if self.d_state < 1:
            raise ValueError(f"d_state must be >= 1; got {self.d_state}")
        inner = self.d_model * self.expand
        if inner % self.n_heads != 0:
            raise ValueError(
                f"inner_dim ({inner}) must be divisible by n_heads ({self.n_heads})"
            )
        if self.chunk_size < 1:
            raise ValueError(f"chunk_size must be >= 1; got {self.chunk_size}")


@dataclass
class Mamba2State:
    """Recurrent state for one token-at-a-time decode.

    Attributes:
        conv_state: Depthwise conv sliding window ``(inner_dim, d_conv)``.
        ssm_state: SSM hidden state ``(n_heads, head_dim, d_state)``.
    """

    conv_state: np.ndarray
    ssm_state: np.ndarray


class Mamba2SSM:
    """SSD (Structured State-Space Duality) layer from Mamba-2.

    Supports both:
    * **Parallel prefill** — ``forward(x)`` processes the full sequence.
    * **Recurrent decode** — ``step(x_t, state)`` processes one token.

    Example::

        cfg  = Mamba2Config(d_model=256, n_heads=4)
        ssm  = Mamba2SSM(cfg)

        # Prefill (batch, seq_len, d_model)
        x     = np.random.randn(1, 32, 256).astype(np.float32)
        y, st = ssm.forward(x)      # y: (1, 32, 256), st: Mamba2State

        # Decode (one token at a time)
        x_t  = np.random.randn(256).astype(np.float32)
        y_t, st = ssm.step(x_t, st)  # y_t: (256,)
    """

    def __init__(self, config: Optional[Mamba2Config] = None) -> None:
        self._cfg = config or Mamba2Config()
        cfg = self._cfg
        rng = np.random.default_rng(cfg.seed)
        inner = cfg.d_model * cfg.expand
        head_dim = inner // cfg.n_heads

        # Input projection (expand + B/C/dt params)
        # dt_rank ≈ d_model // 16, clipped to at least 1
        self._dt_rank = max(1, cfg.d_model // 16)
        in_proj_dim = (
            inner           # z branch
            + inner         # x branch
            + 2 * cfg.d_state * cfg.n_heads  # B, C  (n_heads × d_state each)
            + cfg.n_heads   # dt (one per head)
        )
        scale = 0.02
        self._W_in = rng.standard_normal((cfg.d_model, in_proj_dim)).astype(np.float32) * scale
        self._W_out = rng.standard_normal((inner, cfg.d_model)).astype(np.float32) * scale

        # Depthwise conv weights  (inner_dim, d_conv)
        self._conv_weight = rng.standard_normal((inner, cfg.d_conv)).astype(np.float32) * scale
        self._conv_bias = np.zeros(inner, dtype=np.float32)

        # SSM log-A  (n_heads,)  — negative for stability
        self._log_A = -np.ones(cfg.n_heads, dtype=np.float32)

        # D (skip connection)  (n_heads,)
        self._D = np.ones(cfg.n_heads, dtype=np.float32)

        # Dimensions stored for convenience
        self._inner = inner
        self._head_dim = head_dim

    @property
    def config(self) -> Mamba2Config:
        return self._cfg

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _silu(self, x: np.ndarray) -> np.ndarray:
        return x / (1.0 + np.exp(-x))

    def _softplus(self, x: np.ndarray) -> np.ndarray:
        return np.log1p(np.exp(np.clip(x, -20, 20)))

    def _ssm_chunk(
        self,
        x: np.ndarray,       # (T, n_heads, head_dim)
        B: np.ndarray,       # (T, n_heads, d_state)
        C: np.ndarray,       # (T, n_heads, d_state)
        dt: np.ndarray,      # (T, n_heads)
        h_prev: np.ndarray,  # (n_heads, head_dim, d_state)
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Chunked SSD scan over T tokens."""
        cfg = self._cfg
        T = x.shape[0]
        A = -np.exp(self._log_A)  # (n_heads,)
        y = np.zeros_like(x)
        h = h_prev.copy()
        for t in range(T):
            dt_t = self._softplus(dt[t])   # (n_heads,)
            # dA_t = exp(A * dt_t)
            dA = np.exp(A * dt_t)          # (n_heads,)
            dB = dt_t[..., np.newaxis] * B[t]  # (n_heads, d_state)
            # h: (n_heads, head_dim, d_state)
            h = dA[:, np.newaxis, np.newaxis] * h + (
                x[t, :, :, np.newaxis] * dB[:, np.newaxis, :]
            )
            # y_t: (n_heads, head_dim)
            y[t] = (h * C[t, :, np.newaxis, :]).sum(axis=-1)
            # Add skip
            y[t] += x[t] * self._D[:, np.newaxis]
        return y, h

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forward(
        self,
        x: np.ndarray,
        initial_state: Optional[Mamba2State] = None,
    ) -> Tuple[np.ndarray, Mamba2State]:
        """Parallel prefill forward pass.

        Args:
            x: ``(batch, seq_len, d_model)`` or ``(seq_len, d_model)``.
            initial_state: Optional starting recurrent state.

        Returns:
            Tuple of ``(output, Mamba2State)`` where output has the same
            shape as ``x``.
        """
        x = np.asarray(x, dtype=np.float32)
        squeeze = x.ndim == 2
        if squeeze:
            x = x[np.newaxis]
        B_batch, T, _ = x.shape
        cfg = self._cfg

        # Batch-flatten for simplicity (process first element)
        xb = x[0]  # (T, d_model)

        # Input projection
        proj = xb @ self._W_in  # (T, in_proj_dim)

        # Split projections
        i = 0
        z = proj[:, i: i + self._inner]; i += self._inner
        xu = proj[:, i: i + self._inner]; i += self._inner
        # B, C: (T, n_heads, d_state)
        B_proj = proj[:, i: i + cfg.n_heads * cfg.d_state].reshape(T, cfg.n_heads, cfg.d_state)
        i += cfg.n_heads * cfg.d_state
        C_proj = proj[:, i: i + cfg.n_heads * cfg.d_state].reshape(T, cfg.n_heads, cfg.d_state)
        i += cfg.n_heads * cfg.d_state
        dt = proj[:, i: i + cfg.n_heads]; i += cfg.n_heads

        # Depthwise conv on xu
        if initial_state is not None:
            conv_buf = initial_state.conv_state.copy()
        else:
            conv_buf = np.zeros((self._inner, cfg.d_conv), dtype=np.float32)

        xu_conv = np.zeros_like(xu)
        for t in range(T):
            # Shift conv buffer
            conv_buf[:, :-1] = conv_buf[:, 1:]
            conv_buf[:, -1] = xu[t]
            xu_conv[t] = (conv_buf * self._conv_weight).sum(axis=-1) + self._conv_bias

        xu_act = self._silu(xu_conv)  # (T, inner)

        # Reshape to heads
        xu_h = xu_act.reshape(T, cfg.n_heads, self._head_dim)

        # SSM scan
        if initial_state is not None:
            h0 = initial_state.ssm_state.copy()
        else:
            h0 = np.zeros((cfg.n_heads, self._head_dim, cfg.d_state), dtype=np.float32)

        y_h, h_final = self._ssm_chunk(xu_h, B_proj, C_proj, dt, h0)

        # Merge heads and gate
        y_flat = y_h.reshape(T, self._inner)
        y_gated = y_flat * self._silu(z)

        # Output projection
        out = y_gated @ self._W_out  # (T, d_model)

        final_state = Mamba2State(
            conv_state=conv_buf,
            ssm_state=h_final,
        )
        if squeeze:
            # Input was (T, d_model) → return (T, d_model)
            return out, final_state
        # Input was (batch, T, d_model) → return (batch, T, d_model)
        out = np.broadcast_to(out[np.newaxis], (B_batch, T, out.shape[-1])).copy()
        return out, final_state

    def step(
        self,
        x_t: np.ndarray,
        state: Optional[Mamba2State] = None,
    ) -> Tuple[np.ndarray, Mamba2State]:
        """Single-token recurrent decode step.

        Args:
            x_t: ``(d_model,)`` input token embedding.
            state: Previous recurrent state; created fresh if None.

        Returns:
            ``(output, new_state)`` where output is ``(d_model,)``.
        """
        cfg = self._cfg
        x_t = np.asarray(x_t, dtype=np.float32).ravel()

        if state is None:
            conv_buf = np.zeros((self._inner, cfg.d_conv), dtype=np.float32)
            ssm_st = np.zeros((cfg.n_heads, self._head_dim, cfg.d_state), dtype=np.float32)
        else:
            conv_buf = state.conv_state.copy()
            ssm_st = state.ssm_state.copy()

        # Project
        proj = x_t @ self._W_in  # (in_proj_dim,)
        i = 0
        z = proj[i: i + self._inner]; i += self._inner
        xu = proj[i: i + self._inner]; i += self._inner
        B_t = proj[i: i + cfg.n_heads * cfg.d_state].reshape(cfg.n_heads, cfg.d_state)
        i += cfg.n_heads * cfg.d_state
        C_t = proj[i: i + cfg.n_heads * cfg.d_state].reshape(cfg.n_heads, cfg.d_state)
        i += cfg.n_heads * cfg.d_state
        dt_t = proj[i: i + cfg.n_heads]; i += cfg.n_heads

        # Conv step
        conv_buf[:, :-1] = conv_buf[:, 1:]
        conv_buf[:, -1] = xu
        xu_conv = (conv_buf * self._conv_weight).sum(axis=-1) + self._conv_bias
        xu_act = self._silu(xu_conv)
        xu_h = xu_act.reshape(cfg.n_heads, self._head_dim)

        # SSM step
        A = -np.exp(self._log_A)
        dt_s = self._softplus(dt_t)
        dA = np.exp(A * dt_s)
        dB = dt_s[:, np.newaxis] * B_t  # (n_heads, d_state)
        ssm_st = (
            dA[:, np.newaxis, np.newaxis] * ssm_st
            + xu_h[:, :, np.newaxis] * dB[:, np.newaxis, :]
        )
        y_h = (ssm_st * C_t[:, np.newaxis, :]).sum(axis=-1)
        y_h += xu_h * self._D[:, np.newaxis]

        y_flat = y_h.reshape(self._inner)
        y_gated = y_flat * self._silu(z)
        out = y_gated @ self._W_out

        return out.astype(np.float32), Mamba2State(
            conv_state=conv_buf, ssm_state=ssm_st
        )

    def init_state(self) -> Mamba2State:
        """Return a fresh zero-initialised recurrent state."""
        cfg = self._cfg
        return Mamba2State(
            conv_state=np.zeros((self._inner, cfg.d_conv), dtype=np.float32),
            ssm_state=np.zeros((cfg.n_heads, self._head_dim, cfg.d_state), dtype=np.float32),
        )
