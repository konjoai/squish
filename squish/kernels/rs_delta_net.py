"""squish/kernels/rs_delta_net.py — Rust-backed DeltaNet recurrent scan kernel.

Wraps ``squish_quant_rs.delta_net_scan_f32`` with a NumPy fallback.

DeltaNet is a linear-recurrent sequence model that applies the online
delta rule: for each timestep the recurrent state W is updated with an
outer-product rank-1 correction driven by the prediction residual.
Rayon parallelises the per-head state update at each timestep.

Reference: Schlag et al., "Linear Transformers Are Secretly Fast Weight
Programmers," NeurIPS 2021; Yang et al., "Parallelizing Linear
Transformers with the Delta Rule over Sequence Length," 2024.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

__all__ = [
    "DeltaNetConfig",
    "RustDeltaNet",
]

try:
    import squish_quant as _sq
    _HAS_RUST = hasattr(_sq, "delta_net_scan_f32")
except ImportError:
    _sq = None  # type: ignore[assignment]
    _HAS_RUST = False


# ── NumPy fallback ────────────────────────────────────────────────────────────


def _numpy_delta_net_scan(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    beta: np.ndarray,
) -> np.ndarray:
    """Sequential delta-rule scan — reference implementation.

    Args:
        q:    ``(T, H, D)`` float32 query vectors.
        k:    ``(T, H, D)`` float32 key vectors.
        v:    ``(T, H, D)`` float32 value (target) vectors.
        beta: ``(T, H)``    float32 per-head learning rates.

    Returns:
        Output ``(T, H, D)`` float32.
    """
    t_len, n_heads, head_dim = q.shape
    state = np.zeros((n_heads, head_dim, head_dim), dtype=np.float32)
    out = np.zeros_like(q)
    for t in range(t_len):
        k_t = k[t]  # (H, D)
        v_t = v[t]  # (H, D)
        q_t = q[t]  # (H, D)
        for h in range(n_heads):
            k_norm = k_t[h] / (np.linalg.norm(k_t[h]) + 1e-8)
            wk = state[h] @ k_norm          # (D,)
            residual = v_t[h] - wk           # (D,)
            state[h] += beta[t, h] * np.outer(residual, k_norm)
            out[t, h] = state[h] @ q_t[h]
    return out


# ── Config / wrapper ──────────────────────────────────────────────────────────


@dataclass
class DeltaNetConfig:
    """Configuration for :class:`RustDeltaNet`.

    Attributes:
        eps: Denominator epsilon for key normalisation.
    """

    eps: float = 1e-8


class RustDeltaNet:
    """Rust-accelerated DeltaNet sequential delta-rule recurrent scan.

    Processes T tokens sequentially.  At each step the per-head state matrix
    W is updated with a rank-1 outer-product correction, and the output is
    computed as W @ q.  Rayon parallelises the per-head update.
    Falls back to NumPy when ``squish_quant_rs`` is unavailable.
    """

    def __init__(self, config: Optional[DeltaNetConfig] = None) -> None:
        self._cfg = config or DeltaNetConfig()

    def scan(
        self,
        q: np.ndarray,
        k: np.ndarray,
        v: np.ndarray,
        beta: np.ndarray,
    ) -> np.ndarray:
        """Run the DeltaNet delta-rule recurrent scan.

        Args:
            q:    Query tensor ``(T, n_heads, head_dim)`` float32.
            k:    Key tensor   ``(T, n_heads, head_dim)`` float32.
            v:    Value tensor ``(T, n_heads, head_dim)`` float32.
            beta: Learning-rate tensor ``(T, n_heads)`` float32.

        Returns:
            Output tensor ``(T, n_heads, head_dim)`` float32.

        Raises:
            ValueError: If input shapes are inconsistent.
        """
        q_ = np.ascontiguousarray(q, dtype=np.float32)
        k_ = np.ascontiguousarray(k, dtype=np.float32)
        v_ = np.ascontiguousarray(v, dtype=np.float32)
        b_ = np.ascontiguousarray(beta, dtype=np.float32)
        if q_.shape != k_.shape or q_.shape != v_.shape:
            raise ValueError(
                f"q/k/v must share the same shape; "
                f"got q={q_.shape}, k={k_.shape}, v={v_.shape}"
            )
        if b_.shape != (q_.shape[0], q_.shape[1]):
            raise ValueError(
                f"beta must be (T={q_.shape[0]}, H={q_.shape[1]}); got {b_.shape}"
            )
        if _HAS_RUST:
            return np.asarray(_sq.delta_net_scan_f32(q_, k_, v_, b_), dtype=np.float32)
        return _numpy_delta_net_scan(q_, k_, v_, b_)

    def backend(self) -> str:
        return "rust" if _HAS_RUST else "numpy"
