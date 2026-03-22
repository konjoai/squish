"""xLSTM block: sLSTM (scalar memory) and mLSTM (matrix memory) cells.

xLSTM extends classical LSTM with two key innovations:
  * **sLSTM** — scalar memory cell with exponential gating and max-shift
    numerical stabilization; multiple heads in parallel; O(1) decode.
  * **mLSTM** — matrix-valued covariance memory cell C ∈ ℝ^{d×d} with
    normalisation; full outer-product write, inner-product read; O(1) per
    token at inference (O(d²) memory per layer per session).

A unified :class:`xLSTMBlock` wraps both cell types with a configurable
sLSTM:mLSTM ratio and a post-cell LayerNorm + FFN.

Reference: Beck et al., "xLSTM: Extended Long Short-Term Memory"
arXiv 2405.04517, NeurIPS 2024.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

__all__ = [
    "xLSTMConfig",
    "sLSTMState",
    "mLSTMState",
    "xLSTMState",
    "xLSTMBlock",
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class xLSTMConfig:
    """Configuration for :class:`xLSTMBlock`.

    Attributes:
        d_model: Embedding dimension.
        n_slstm_heads: Number of sLSTM heads.
        slstm_head_dim: State dim per sLSTM head.
        mlstm_dim: Dimension of the mLSTM covariance state (square matrix).
        expand_factor: FFN expansion ratio.
        seed: RNG seed.
    """

    d_model: int = 256
    n_slstm_heads: int = 4
    slstm_head_dim: int = 64
    mlstm_dim: int = 64
    expand_factor: float = 4.0
    seed: int = 0

    def __post_init__(self) -> None:
        if self.d_model < 1:
            raise ValueError(f"d_model must be ≥ 1, got {self.d_model}")
        if self.n_slstm_heads < 1:
            raise ValueError(f"n_slstm_heads must be ≥ 1, got {self.n_slstm_heads}")
        if self.slstm_head_dim < 1:
            raise ValueError(f"slstm_head_dim must be ≥ 1, got {self.slstm_head_dim}")
        if self.mlstm_dim < 1:
            raise ValueError(f"mlstm_dim must be ≥ 1, got {self.mlstm_dim}")


# ---------------------------------------------------------------------------
# State dataclasses
# ---------------------------------------------------------------------------


@dataclass
class sLSTMState:
    """Per-head scalar-valued LSTM state.

    Attributes:
        c: Cell memory ``(n_heads, head_dim)``.
        n: Normaliser ``(n_heads, head_dim)``.
        m: Max-shift tracker ``(n_heads,)`` for numerical stability.
        n_steps: Tokens processed.
    """

    c: np.ndarray   # (n_heads, head_dim)
    n: np.ndarray   # (n_heads, head_dim)
    m: np.ndarray   # (n_heads,)
    n_steps: int = 0

    @property
    def state_bytes(self) -> int:
        return self.c.nbytes + self.n.nbytes + self.m.nbytes


@dataclass
class mLSTMState:
    """Matrix-valued LSTM state.

    Attributes:
        C: Covariance memory matrix ``(d, d)``.
        n: Normaliser vector ``(d,)``.
        m: Scalar max-shift for stability.
        n_steps: Tokens processed.
    """

    C: np.ndarray   # (d, d)
    n: np.ndarray   # (d,)
    m: float = 0.0
    n_steps: int = 0

    @property
    def state_bytes(self) -> int:
        return self.C.nbytes + self.n.nbytes


@dataclass
class xLSTMState:
    """Combined state for one :class:`xLSTMBlock`."""

    slstm: sLSTMState
    mlstm: mLSTMState


# ---------------------------------------------------------------------------
# Block implementation
# ---------------------------------------------------------------------------


class xLSTMBlock:
    """Unified xLSTM block with sLSTM + mLSTM sub-layers.

    Processes a sequence in-order.  Each step:
      1. sLSTM update (scalar heads)
      2. mLSTM update (matrix memory)
      3. Concatenate + project back to d_model
      4. LayerNorm + FFN

    Usage::

        cfg = xLSTMConfig(d_model=256, n_slstm_heads=4, slstm_head_dim=64, mlstm_dim=64)
        block = xLSTMBlock(cfg)
        state = block.new_state()
        out, state = block.forward(x, state)   # x: (T, d_model)
    """

    def __init__(self, config: xLSTMConfig) -> None:
        self.config = config
        rng = np.random.default_rng(config.seed)
        d = config.d_model
        scale = 1.0 / np.sqrt(d)
        h, hd = config.n_slstm_heads, config.slstm_head_dim
        md = config.mlstm_dim

        # sLSTM projections (i, f, z, o) × n_heads
        slstm_out = h * hd
        self.sW_i = rng.standard_normal((d, slstm_out)).astype(np.float32) * scale
        self.sW_f = rng.standard_normal((d, slstm_out)).astype(np.float32) * scale
        self.sW_z = rng.standard_normal((d, slstm_out)).astype(np.float32) * scale
        self.sW_o = rng.standard_normal((d, slstm_out)).astype(np.float32) * scale

        # mLSTM projections
        self.mW_q = rng.standard_normal((d, md)).astype(np.float32) * scale
        self.mW_k = rng.standard_normal((d, md)).astype(np.float32) * scale
        self.mW_v = rng.standard_normal((d, md)).astype(np.float32) * scale
        self.mW_i = rng.standard_normal((d, 1)).astype(np.float32) * scale
        self.mW_f = rng.standard_normal((d, 1)).astype(np.float32) * scale
        self.mW_o = rng.standard_normal((d, md)).astype(np.float32) * scale

        # Output projection (slstm_out + md → d)
        self.W_proj = rng.standard_normal((slstm_out + md, d)).astype(np.float32) * scale

        # LayerNorm parameters
        self.ln_g = np.ones(d, dtype=np.float32)
        self.ln_b = np.zeros(d, dtype=np.float32)

        # FFN
        ffn_d = int(d * config.expand_factor)
        self.W_ff1 = rng.standard_normal((d, ffn_d)).astype(np.float32) * scale
        self.W_ff2 = rng.standard_normal((ffn_d, d)).astype(np.float32) * scale

    # ------------------------------------------------------------------
    # Primary API
    # ------------------------------------------------------------------

    def new_state(self) -> xLSTMState:
        cfg = self.config
        ss = sLSTMState(
            c=np.zeros((cfg.n_slstm_heads, cfg.slstm_head_dim), dtype=np.float32),
            n=np.zeros((cfg.n_slstm_heads, cfg.slstm_head_dim), dtype=np.float32),
            m=np.zeros(cfg.n_slstm_heads, dtype=np.float32),
        )
        ms = mLSTMState(
            C=np.zeros((cfg.mlstm_dim, cfg.mlstm_dim), dtype=np.float32),
            n=np.zeros(cfg.mlstm_dim, dtype=np.float32),
        )
        return xLSTMState(slstm=ss, mlstm=ms)

    def forward(
        self,
        x: np.ndarray,
        state: xLSTMState,
    ) -> Tuple[np.ndarray, xLSTMState]:
        """Forward over a sequence.

        Args:
            x: ``(seq_len, d_model)``.
            state: Mutable block state.

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

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def _layer_norm(
        x: np.ndarray, g: np.ndarray, b: np.ndarray, eps: float = 1e-5
    ) -> np.ndarray:
        mu = x.mean()
        std = np.sqrt(x.var() + eps)
        return g * (x - mu) / std + b

    def _slstm_step(self, xt: np.ndarray, ss: sLSTMState) -> np.ndarray:
        h, hd = self.config.n_slstm_heads, self.config.slstm_head_dim
        i_raw = (xt @ self.sW_i).reshape(h, hd)
        f_raw = (xt @ self.sW_f).reshape(h, hd)
        z = np.tanh((xt @ self.sW_z).reshape(h, hd))
        o = self._sigmoid((xt @ self.sW_o).reshape(h, hd))

        # Stabilised exponential gating (max-shift per head)
        m_new = np.maximum(f_raw.max(axis=1) + ss.m, i_raw.max(axis=1))  # (h,)
        f = np.exp(f_raw + ss.m[:, np.newaxis] - m_new[:, np.newaxis])
        i = np.exp(i_raw - m_new[:, np.newaxis])

        ss.c = f * ss.c + i * z
        ss.n = f * ss.n + i
        ss.m = m_new
        h_out = o * ss.c / np.maximum(np.abs(ss.n), 1.0)
        ss.n_steps += 1
        return h_out.reshape(-1)  # (h*hd,)

    def _mlstm_step(self, xt: np.ndarray, ms: mLSTMState) -> np.ndarray:
        q = xt @ self.mW_q              # (md,)
        k = xt @ self.mW_k             # (md,)
        v = xt @ self.mW_v             # (md,)
        i_raw = float((xt @ self.mW_i).squeeze())
        f_raw = float((xt @ self.mW_f).squeeze())
        o = self._sigmoid(xt @ self.mW_o)  # (md,)

        # Stabilised gates
        m_new = max(f_raw + ms.m, i_raw)
        f = np.exp(f_raw + ms.m - m_new)
        i = np.exp(i_raw - m_new)

        ms.C = f * ms.C + i * np.outer(v, k)    # rank-1 update
        ms.n = f * ms.n + i * k
        ms.m = m_new

        h_hat = ms.C @ q
        denom = max(float(np.abs(ms.n @ q)), 1.0)
        h_out = o * (h_hat / denom)
        ms.n_steps += 1
        return h_out  # (md,)

    def _step(self, xt: np.ndarray, state: xLSTMState) -> np.ndarray:
        s_out = self._slstm_step(xt, state.slstm)
        m_out = self._mlstm_step(xt, state.mlstm)
        combined = np.concatenate([s_out, m_out])
        projected = combined @ self.W_proj  # (d_model,)
        normed = self._layer_norm(projected + xt, self.ln_g, self.ln_b)
        ffn = np.maximum(0.0, normed @ self.W_ff1) @ self.W_ff2
        return normed + ffn
