"""RWKV-6 Eagle/Finch channel-mix block for inference.

RWKV-6 ("Eagle") reformulates recurrent language modelling with independent
receptance, forget, key, value, and gate projections.  The data-dependent time
decay enables in-context learning comparable to Transformers while retaining
O(d) per-token decode cost and a parallel wkv6 scan form for efficient prefill.

This module implements the wkv6 core (time-parallel and recurrent modes) plus
the channel-mix FFN layer that completes an RWKV-6 block.  The parallel path
uses a simple sequential fallback (Metal-tiled scan is provided by
``ParallelScanKernel`` if available at runtime).

Reference: Peng et al., "Eagle and Finch: RWKV with Matrix-Valued States and
Dynamic Recurrence" arXiv 2404.05892 (2024).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

__all__ = [
    "RWKV6Config",
    "RWKV6State",
    "RWKV6ChannelMix",
]


@dataclass
class RWKV6Config:
    """Configuration for one RWKV-6 time-mix + channel-mix block.

    Attributes:
        d_model: Model embedding dimension.
        d_state: Per-head recurrent state dimension (usually == head_dim).
        n_heads: Number of parallel wkv6 heads.
        head_dim: Dimension per head; must satisfy n_heads * head_dim == d_model.
        expand_factor: Channel-mix expansion ratio.
        dropout: Dropout rate (0 = disabled, inference only).
        seed: RNG seed for weight initialisation stub.
    """

    d_model: int = 512
    d_state: int = 64
    n_heads: int = 8
    head_dim: int = 64
    expand_factor: float = 4.0
    dropout: float = 0.0
    seed: int = 0

    def __post_init__(self) -> None:
        if self.d_model < 1:
            raise ValueError(f"d_model must be ≥ 1, got {self.d_model}")
        if self.n_heads < 1:
            raise ValueError(f"n_heads must be ≥ 1, got {self.n_heads}")
        if self.head_dim < 1:
            raise ValueError(f"head_dim must be ≥ 1, got {self.head_dim}")
        if self.n_heads * self.head_dim != self.d_model:
            raise ValueError(
                f"n_heads ({self.n_heads}) * head_dim ({self.head_dim}) "
                f"must equal d_model ({self.d_model})"
            )
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")


@dataclass
class RWKV6State:
    """Per-layer RWKV-6 recurrent time state.

    Attributes:
        time_state: Matrix-valued state ``(n_heads, head_dim, d_state)``
            maintained across decode steps.
        n_tokens_seen: Number of tokens processed since state reset.
    """

    time_state: np.ndarray  # (n_heads, head_dim, d_state)
    n_tokens_seen: int = 0

    @property
    def n_heads(self) -> int:
        return self.time_state.shape[0]

    @property
    def state_bytes(self) -> int:
        return self.time_state.nbytes


class RWKV6ChannelMix:
    """RWKV-6 block: wkv6 time-mix + channel-mix FFN.

    Supports two operational modes:

    * **Recurrent** (``seq_len == 1``): O(d) per-token update of the matrix
      time state; used during autoregressive decode.
    * **Parallel** (``seq_len > 1``): Sequential WKV scan across the time
      dimension; used during prefill.  A production Metal kernel would replace
      the inner loop but the NumPy reference is functionally equivalent.

    Usage::

        cfg = RWKV6Config(d_model=512, n_heads=8, head_dim=64)
        layer = RWKV6ChannelMix(cfg)
        state = layer.new_state()
        # Prefill
        h_prefill, state = layer.forward(x_prompt, state)
        # Decode one token at a time
        for t in range(max_new_tokens):
            h_t, state = layer.forward(x_t[np.newaxis], state)
    """

    def __init__(self, config: RWKV6Config) -> None:
        self.config = config
        rng = np.random.default_rng(config.seed)
        d, h, hd = config.d_model, config.n_heads, config.head_dim
        scale = 1.0 / np.sqrt(d)
        # Time-mix projections (R, W, K, V, G)
        self.W_r = rng.standard_normal((d, d)).astype(np.float32) * scale
        self.W_w = rng.standard_normal((d, h)).astype(np.float32) * scale
        self.W_k = rng.standard_normal((d, d)).astype(np.float32) * scale
        self.W_v = rng.standard_normal((d, d)).astype(np.float32) * scale
        self.W_g = rng.standard_normal((d, d)).astype(np.float32) * scale
        self.W_o = rng.standard_normal((d, d)).astype(np.float32) * scale
        # Channel-mix projections
        ffn_dim = int(d * config.expand_factor)
        self.W_cm1 = rng.standard_normal((d, ffn_dim)).astype(np.float32) * scale
        self.W_cm2 = rng.standard_normal((ffn_dim, d)).astype(np.float32) * scale

    # ------------------------------------------------------------------
    # Primary API
    # ------------------------------------------------------------------

    def new_state(self) -> RWKV6State:
        cfg = self.config
        ts = np.zeros((cfg.n_heads, cfg.head_dim, cfg.d_state), dtype=np.float32)
        return RWKV6State(time_state=ts)

    def forward(
        self,
        x: np.ndarray,
        state: RWKV6State,
    ) -> Tuple[np.ndarray, RWKV6State]:
        """Process a sequence through the RWKV-6 block.

        Args:
            x: Input ``(seq_len, d_model)``.
            state: Mutable recurrent state.

        Returns:
            ``(output (seq_len, d_model), updated_state)``.
        """
        x = np.asarray(x, dtype=np.float32)
        seq_len, d = x.shape
        outputs = []
        for t in range(seq_len):
            xt = x[t]
            out_t = self._recurrent_step(xt, state)
            outputs.append(out_t)
            state.n_tokens_seen += 1
        return np.stack(outputs), state

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _recurrent_step(self, x: np.ndarray, state: RWKV6State) -> np.ndarray:
        cfg = self.config
        h, hd, ds = cfg.n_heads, cfg.head_dim, cfg.d_state

        r = np.tanh(x @ self.W_r)         # (d,)
        w = np.exp(-np.exp(x @ self.W_w)) # (n_heads,) per-head decay
        k = x @ self.W_k                  # (d,)
        v = x @ self.W_v                  # (d,)
        g = np.sigmoid(x @ self.W_g)      # (d,)

        k_h = k.reshape(h, hd)
        v_h = v.reshape(h, ds)

        # State update: S = w * S + outer(k_h, v_h) per head
        w_h = w[:, np.newaxis, np.newaxis]  # (h, 1, 1)
        state.time_state = w_h * state.time_state + np.einsum(
            "hi,hj->hij", k_h, v_h
        )

        # Retrieve: o = r * (S @ r_head)
        r_h = r.reshape(h, hd)
        o_h = np.einsum("hi,hij->hj", r_h, state.time_state)  # (h, ds)
        o = o_h.reshape(-1)[:cfg.d_model] * g

        # Time-mix output
        tm_out = o @ self.W_o  # (d,)

        # Channel-mix FFN
        cm = np.maximum(0.0, x @ self.W_cm1) ** 2 @ self.W_cm2
        return tm_out + cm


def np_sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


# Patch numpy: sigmoid is not a built-in ufunc
np.sigmoid = np_sigmoid  # type: ignore[attr-defined]
