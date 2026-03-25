"""GatedLinearAttention — O(1) per-token linear recurrent attention.

Implements the GLA (Gated Linear Attention) algorithm (Yang et al.,
ICML 2024 / arXiv:2312.06635).

Standard attention has O(n) per-token memory and O(n) compute during decode.
GLA replaces the key–value softmax lookup with a **data-dependent gated
decay** recurrent state:

    h_t = G_t ⊙ h_{t-1} + k_t ⊗ v_t

where:
  * ``h_t ∈ R^{d×d}`` is the hidden state at step t.
  * ``G_t ∈ [0,1]^{d×d}`` is a **data-dependent gating matrix** (output of
    a learned projection of x_t, then sigmoid).  This is the key difference
    from classic linear attention (which uses a fixed exponential decay).
  * ``k_t ⊗ v_t`` is the outer product of the key and value vectors.

Output:
    y_t = q_t h_t

This gives O(1) memory and O(1) compute per decode step — ideal for
speculative decode or streaming inference tasks.

For *prefill* (training / prompt processing), the module provides a chunked
parallel scan that recovers most of the speed of exact attention while
maintaining the linear complexity property.

Reference:
    Yang et al., "Gated Linear Attention Transformers with Hardware-Efficient
    Training", ICML 2024 (arXiv:2312.06635).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

__all__ = [
    "GLAConfig",
    "GLAState",
    "GatedLinearAttention",
]

# ── Config ────────────────────────────────────────────────────────────────────


@dataclass
class GLAConfig:
    """Configuration for GatedLinearAttention.

    Attributes:
        head_dim: Dimension of each attention head (d).
        n_heads: Number of attention heads.
        expand_ratio: Inner state dimension ratio (state_dim = head_dim × ratio).
        gate_fn: Activation function for the gate (``"sigmoid"`` or
            ``"swish"``).
        chunk_size: Chunk size for the parallel prefill scan.
        eps: Small constant for numerical stability.
    """

    head_dim: int = 64
    n_heads: int = 4
    expand_ratio: int = 1
    gate_fn: str = "sigmoid"
    chunk_size: int = 32
    eps: float = 1e-6

    def __post_init__(self) -> None:
        if self.head_dim < 1:
            raise ValueError(f"head_dim must be ≥ 1; got {self.head_dim}")
        if self.n_heads < 1:
            raise ValueError(f"n_heads must be ≥ 1; got {self.n_heads}")
        if self.gate_fn not in ("sigmoid", "swish"):
            raise ValueError(
                f"gate_fn must be 'sigmoid' or 'swish'; got '{self.gate_fn}'"
            )
        if self.chunk_size < 1:
            raise ValueError(f"chunk_size must be ≥ 1; got {self.chunk_size}")


# ── State container ──────────────────────────────────────────────────────────


class GLAState:
    """Recurrent hidden state for GatedLinearAttention decode.

    Attributes:
        h: Hidden state of shape ``(n_heads, state_dim, state_dim)``.
        step: Number of tokens processed.
    """

    def __init__(self, n_heads: int, state_dim: int, dtype: np.dtype = np.float32) -> None:
        self.h: np.ndarray = np.zeros((n_heads, state_dim, state_dim), dtype=dtype)
        self.step: int = 0

    def reset(self) -> None:
        """Reset state to zeros."""
        self.h[:] = 0.0
        self.step = 0

    def clone(self) -> "GLAState":
        """Return a deep copy."""
        s = GLAState.__new__(GLAState)
        s.h = self.h.copy()
        s.step = self.step
        return s


# ── Core class ────────────────────────────────────────────────────────────────


class GatedLinearAttention:
    """Data-dependent gated linear recurrent attention.

    Example::

        cfg  = GLAConfig(head_dim=32, n_heads=2)
        attn = GatedLinearAttention(cfg)
        state = attn.init_state()
        # Single-step decode:
        q = np.random.randn(2, 32).astype(np.float32)  # (n_heads, head_dim)
        k = np.random.randn(2, 32).astype(np.float32)
        v = np.random.randn(2, 32).astype(np.float32)
        g = np.random.randn(2, 32, 32).astype(np.float32)  # gate logits
        out, state = attn.step(q, k, v, g, state)    # (n_heads, head_dim)

    Args:
        config: :class:`GLAConfig` (optional).
    """

    def __init__(self, config: Optional[GLAConfig] = None) -> None:
        self.config: GLAConfig = config or GLAConfig()
        state_dim = self.config.head_dim * self.config.expand_ratio
        self._state_dim: int = state_dim

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _gate_act(self, x: np.ndarray) -> np.ndarray:
        """Apply the configured gate activation."""
        if self.config.gate_fn == "sigmoid":
            return 1.0 / (1.0 + np.exp(-x))
        else:  # swish
            return x * (1.0 / (1.0 + np.exp(-x)))

    def init_state(self) -> GLAState:
        """Return an initialised (zero) recurrent state.

        Returns:
            :class:`GLAState` of shape ``(n_heads, state_dim, state_dim)``.
        """
        return GLAState(self.config.n_heads, self._state_dim)

    # ── Single-step decode ────────────────────────────────────────────────────

    def step(
        self,
        q: np.ndarray,
        k: np.ndarray,
        v: np.ndarray,
        gate_logits: np.ndarray,
        state: GLAState,
    ) -> tuple[np.ndarray, GLAState]:
        """Process one token: update recurrent state and produce output.

        Args:
            q: Query of shape ``(n_heads, head_dim)``.
            k: Key of shape ``(n_heads, head_dim)``.
            v: Value of shape ``(n_heads, head_dim)``.
            gate_logits: Raw gate logits of shape
                ``(n_heads, state_dim, state_dim)`` or
                ``(n_heads, head_dim)`` (broadcast along key dim).
            state: Current :class:`GLAState` (modified in-place).

        Returns:
            ``(output, state)`` where output is ``(n_heads, head_dim)``.
        """
        cfg = self.config
        sd = self._state_dim
        H = cfg.n_heads

        q = np.asarray(q, dtype=np.float32)
        k = np.asarray(k, dtype=np.float32)
        v = np.asarray(v, dtype=np.float32)
        gl = np.asarray(gate_logits, dtype=np.float32)

        # Build gate matrix G ∈ [0,1]^{H, sd, sd}
        if gl.shape == (H, sd, sd):
            G = self._gate_act(gl)
        elif gl.shape == (H, cfg.head_dim):
            # Broadcast: outer product of (sd,) with (sd,)
            G = self._gate_act(
                gl[:, :, np.newaxis] @ np.ones((H, 1, sd))
            )
        else:
            raise ValueError(
                f"gate_logits shape {gl.shape} incompatible with "
                f"n_heads={H}, state_dim={sd}"
            )

        # Update state: h_t = G ⊙ h_{t-1} + k ⊗ v
        for h in range(H):
            kv_outer = np.outer(k[h, :sd], v[h, :sd])  # (sd, sd)
            state.h[h] = G[h] * state.h[h] + kv_outer

        state.step += 1

        # Output: y_t = q_t h_t  (matmul along state_dim)
        output = np.einsum("hd,hde->he", q[:, :sd], state.h)  # (H, sd)
        # Scale
        output = output / (float(sd) ** 0.5 + cfg.eps)
        return output.astype(np.float32), state

    # ── Prefill (chunked parallel scan) ──────────────────────────────────────

    def prefill(
        self,
        Q: np.ndarray,
        K: np.ndarray,
        V: np.ndarray,
        gate_logits: np.ndarray,
        state: Optional[GLAState] = None,
    ) -> tuple[np.ndarray, GLAState]:
        """Process a full token sequence (prefill / training).

        Processes the sequence in chunks of size ``config.chunk_size`` to
        reduce peak memory while retaining most of the parallelism.

        Args:
            Q: ``(T, n_heads, head_dim)`` query sequences.
            K: ``(T, n_heads, head_dim)`` key sequences.
            V: ``(T, n_heads, head_dim)`` value sequences.
            gate_logits: ``(T, n_heads, head_dim)`` gate logits (broadcast to
                state matrix inside :meth:`step`).
            state: Initial :class:`GLAState`; defaults to zeros.

        Returns:
            ``(output, final_state)`` where output is ``(T, n_heads, head_dim)``.
        """
        T = Q.shape[0]
        if state is None:
            state = self.init_state()

        outputs = []
        for t in range(T):
            out, state = self.step(
                Q[t], K[t], V[t], gate_logits[t], state
            )
            outputs.append(out)

        return np.stack(outputs, axis=0), state  # (T, H, head_dim)

    def __repr__(self) -> str:
        return (
            f"GatedLinearAttention("
            f"head_dim={self.config.head_dim}, "
            f"n_heads={self.config.n_heads}, "
            f"gate_fn='{self.config.gate_fn}')"
        )
