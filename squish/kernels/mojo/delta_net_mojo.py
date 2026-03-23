"""squish/kernels/mojo/delta_net_mojo.py — Mojo-backed DeltaNet recurrent scan.

Wraps the ``delta_net_recurrence`` Mojo kernel via MojoBridge with a
NumPy fallback.  Implements the sequential delta-rule recurrent scan
over T tokens with SIMD outer-product updates per head.

Reference: Schlag et al., "Linear Transformers Are Secretly Fast Weight
Programmers," NeurIPS 2021; Yang et al., "Parallelizing Linear
Transformers with the Delta Rule over Sequence Length," 2024.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from squish.kernels.mojo.mojo_bridge import MojoBridge

__all__ = [
    "DeltaNetMojoConfig",
    "MojoDeltaNet",
]

_bridge = MojoBridge()
_kernel = _bridge.load_kernel("delta_net_recurrence")


# ── NumPy fallback ────────────────────────────────────────────────────────────


def _numpy_scan(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    beta: np.ndarray,
) -> np.ndarray:
    t_len, n_heads, head_dim = q.shape
    state = np.zeros((n_heads, head_dim, head_dim), dtype=np.float32)
    out = np.zeros_like(q)
    for t in range(t_len):
        for h in range(n_heads):
            k_norm = k[t, h] / (np.linalg.norm(k[t, h]) + 1e-8)
            wk = state[h] @ k_norm
            residual = v[t, h] - wk
            state[h] += beta[t, h] * np.outer(residual, k_norm)
            out[t, h] = state[h] @ q[t, h]
    return out


# ── Config / wrapper ──────────────────────────────────────────────────────────


@dataclass
class DeltaNetMojoConfig:
    """Configuration for :class:`MojoDeltaNet`.

    Attributes:
        eps: Denominator epsilon for key normalisation.
    """

    eps: float = 1e-8


class MojoDeltaNet:
    """Mojo-backed DeltaNet sequential delta-rule recurrent scan.

    Processes tokens sequentially; per-head outer-product update uses
    ``vectorize`` over the state matrix.  ``parallelize`` over heads.
    Falls back to NumPy when the Mojo runtime is absent.
    """

    def __init__(self, config: Optional[DeltaNetMojoConfig] = None) -> None:
        self._cfg = config or DeltaNetMojoConfig()

    def scan(
        self,
        q: np.ndarray,
        k: np.ndarray,
        v: np.ndarray,
        beta: np.ndarray,
    ) -> np.ndarray:
        """Run the DeltaNet delta-rule recurrent scan.

        Args:
            q:    ``(T, n_heads, head_dim)`` float32.
            k:    ``(T, n_heads, head_dim)`` float32.
            v:    ``(T, n_heads, head_dim)`` float32.
            beta: ``(T, n_heads)`` float32 learning rates.

        Returns:
            Output ``(T, n_heads, head_dim)`` float32.
        """
        q_ = np.ascontiguousarray(q, dtype=np.float32)
        k_ = np.ascontiguousarray(k, dtype=np.float32)
        v_ = np.ascontiguousarray(v, dtype=np.float32)
        b_ = np.ascontiguousarray(beta, dtype=np.float32)
        if q_.shape != k_.shape or q_.shape != v_.shape:
            raise ValueError("q/k/v must share the same shape")
        if b_.shape != (q_.shape[0], q_.shape[1]):
            raise ValueError(f"beta must be (T, H); got {b_.shape}")
        if _kernel is not None:
            t_len, n_heads, head_dim = q_.shape
            out = np.zeros_like(q_)
            _kernel(
                q_.ctypes.data, k_.ctypes.data, v_.ctypes.data, b_.ctypes.data,
                out.ctypes.data, t_len, n_heads, head_dim,
            )
            return out
        return _numpy_scan(q_, k_, v_, b_)

    def backend(self) -> str:
        return "mojo" if _kernel is not None else "numpy"
