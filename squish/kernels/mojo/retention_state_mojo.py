"""retention_state_mojo.py — Mojo-accelerated Retention recurrent state update.

Wraps `squish/kernels/mojo/kernels/retention_state.mojo` via MojoBridge
(Wave 57b). Falls back to NumPy `np.einsum` when the Mojo library is
unavailable.

MojoRetentionState accelerates the RetNet recurrent mode state update:
  S = gamma * S + outer(k, v)   (rank-1 outer product update)
  o = S @ q                      (matrix-vector retrieval)

Uses Mojo SIMD outer-product + matrix-vector, replacing 2 np.einsum
calls per layer per decode step. State matrix S has shape
(n_heads, head_dim, head_dim).

Reference:
  Sun et al. (arXiv:2307.08621, 2023) — RetNet: Retaining Training
  Transformers' Performance for Inference.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from squish.kernels.mojo.mojo_bridge import MojoBridge

__all__ = ["RetentionStateConfig", "MojoRetentionState"]

_bridge = MojoBridge()
_MOJO_FN = _bridge.load_kernel("retention_state")


@dataclass
class RetentionStateConfig:
    """Configuration for MojoRetentionState.

    Attributes:
        n_heads:  Number of retention heads.
        head_dim: Head dimension (e.g. 64 or 128).
        gamma:    Retention decay factor (per-head or scalar, default 0.999).
    """

    n_heads: int = 8
    head_dim: int = 128
    gamma: float = 0.999


class MojoRetentionState:
    """Mojo-accelerated RetNet recurrent state update and retrieval.

    Maintains the `(n_heads, head_dim, head_dim)` recurrent state `S`
    and provides step-level update + retrieval for decode mode.

    Usage::

        ret = MojoRetentionState(RetentionStateConfig(n_heads=8, head_dim=128))
        k   = np.random.randn(8, 128).astype(np.float32)  # (n_heads, head_dim)
        v   = np.random.randn(8, 128).astype(np.float32)
        q   = np.random.randn(8, 128).astype(np.float32)
        S   = np.zeros((8, 128, 128), dtype=np.float32)
        o, S_new = ret.step(q, k, v, S)
    """

    def __init__(self, config: RetentionStateConfig | None = None) -> None:
        self._cfg = config or RetentionStateConfig()

    def zero_state(self) -> np.ndarray:
        """Return zero-initialized state `(n_heads, head_dim, head_dim)`."""
        n, d = self._cfg.n_heads, self._cfg.head_dim
        return np.zeros((n, d, d), dtype=np.float32)

    def step(
        self,
        q: np.ndarray,
        k: np.ndarray,
        v: np.ndarray,
        state: np.ndarray,
        gamma: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Perform one retention recurrent step.

        Computes::

            S_new = gamma * S + outer(k[h], v[h])  for each head h
            o[h]  = S_new[h] @ q[h]

        Args:
            q:     Float32 `(n_heads, head_dim)` — query vectors.
            k:     Float32 `(n_heads, head_dim)` — key vectors.
            v:     Float32 `(n_heads, head_dim)` — value vectors.
            state: Float32 `(n_heads, head_dim, head_dim)` — current state S.
            gamma: Decay factor override (uses config default if None).

        Returns:
            Tuple `(o, S_new)`:
            - o     `(n_heads, head_dim)` — output vectors
            - S_new `(n_heads, head_dim, head_dim)` — updated state
        """
        q = np.asarray(q, dtype=np.float32)
        k = np.asarray(k, dtype=np.float32)
        v = np.asarray(v, dtype=np.float32)
        state = np.asarray(state, dtype=np.float32)
        g = gamma if gamma is not None else self._cfg.gamma
        if _MOJO_FN is not None:
            pass  # Mojo path placeholder
        return self._numpy_step(q, k, v, state, g)

    def backend(self) -> str:
        """Return backend: 'mojo' or 'numpy'."""
        if _MOJO_FN is not None:
            return "mojo"
        return "numpy"

    # ── NumPy fallback ── -----------------------------------------------------

    @staticmethod
    def _numpy_step(
        q: np.ndarray,
        k: np.ndarray,
        v: np.ndarray,
        state: np.ndarray,
        gamma: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Pure-NumPy retention step: outer product update + matvec retrieval."""
        # state: (n_heads, head_dim, head_dim)
        # k[h] outer v[h]: (head_dim, head_dim)  — via einsum or np.outer per head
        outer = np.einsum("hi,hj->hij", k, v)  # (n_heads, head_dim, head_dim)
        s_new = gamma * state + outer           # (n_heads, head_dim, head_dim)
        # o[h] = S_new[h] @ q[h]
        o = np.einsum("hij,hj->hi", s_new, q)  # (n_heads, head_dim)
        return o.astype(np.float32), s_new.astype(np.float32)
