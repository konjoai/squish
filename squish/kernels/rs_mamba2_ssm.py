"""squish/kernels/rs_mamba2_ssm.py — Rust-backed Mamba-2 SSD scan kernels.

Wraps ``squish_quant_rs.mamba2_ssm_scan_f32`` and
``squish_quant_rs.mamba2_ssm_decode_f32`` with a NumPy fallback that is
semantically identical to the pure-Python implementation in
``squish.attention.mamba2_ssm``.

Reference: Dao & Gu, "Transformers are SSMs: Generalized Models and Efficient
Algorithms Through Structured State Space Duality." ICML 2024.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

__all__ = [
    "Mamba2ScanConfig",
    "RustMamba2SSM",
]

try:
    import squish_quant as _sq
    _HAS_RUST = hasattr(_sq, "mamba2_ssm_scan_f32")
except ImportError:
    _sq = None  # type: ignore[assignment]
    _HAS_RUST = False


# ── NumPy fallbacks ───────────────────────────────────────────────────────────


def _numpy_scan(
    a: np.ndarray,        # (T,) log-A
    b: np.ndarray,        # (T, d_state)
    c: np.ndarray,        # (T, d_state)
    x: np.ndarray,        # (T,)
    h0: np.ndarray,       # (d_state,)
) -> Tuple[np.ndarray, np.ndarray]:
    t_len, d_state = b.shape
    h = h0.copy()
    out = np.empty(t_len, dtype=np.float32)
    for t in range(t_len):
        a_t = float(np.exp(a[t]))
        h = a_t * h + b[t] * x[t]
        out[t] = float(np.dot(c[t], h))
    return out, h.astype(np.float32)


def _numpy_decode(
    a_scalar: float,
    b_vec: np.ndarray,
    c_vec: np.ndarray,
    x_scalar: float,
    state: np.ndarray,
) -> Tuple[float, np.ndarray]:
    new_h = a_scalar * state + b_vec * x_scalar
    y = float(np.dot(c_vec, new_h))
    return y, new_h.astype(np.float32)


# ── Config / wrapper ──────────────────────────────────────────────────────────


@dataclass
class Mamba2ScanConfig:
    """Configuration for :class:`RustMamba2SSM`.

    Attributes:
        d_state: SSM state dimension.
        d_model: Model / input dimension.
    """

    d_state: int = 64
    d_model: int = 512


class RustMamba2SSM:
    """Rust-accelerated Mamba-2 SSD scan (prefill + decode).

    Falls back to a NumPy implementation when ``squish_quant_rs`` is not
    available.
    """

    def __init__(self, config: Optional[Mamba2ScanConfig] = None) -> None:
        self._cfg = config or Mamba2ScanConfig()

    # ── prefill scan ─────────────────────────────────────────────────────────

    def scan(
        self,
        a: np.ndarray,
        b: np.ndarray,
        c: np.ndarray,
        x: np.ndarray,
        h0: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run the SSD chunked scan over a full sequence.

        Args:
            a:  Log-A scalars ``(T,)`` float32.
            b:  B matrices ``(T, d_state)`` float32.
            c:  C matrices ``(T, d_state)`` float32.
            x:  Input sequence ``(T,)`` float32.
            h0: Initial state ``(d_state,)``; zeros if *None*.

        Returns:
            ``(output (T,), final_state (d_state,))`` both float32.
        """
        a_f = np.ascontiguousarray(a, dtype=np.float32).ravel()
        b_f = np.ascontiguousarray(b, dtype=np.float32)
        c_f = np.ascontiguousarray(c, dtype=np.float32)
        x_f = np.ascontiguousarray(x, dtype=np.float32).ravel()
        d_state = b_f.shape[1]
        h0_f = (
            np.ascontiguousarray(h0, dtype=np.float32).ravel()
            if h0 is not None
            else np.zeros(d_state, dtype=np.float32)
        )
        if _HAS_RUST:
            out, fs = _sq.mamba2_ssm_scan_f32(a_f, b_f, c_f, x_f, h0_f)
            return np.asarray(out, dtype=np.float32), np.asarray(fs, dtype=np.float32)
        return _numpy_scan(a_f, b_f, c_f, x_f, h0_f)

    # ── per-token decode step ─────────────────────────────────────────────────

    def decode_step(
        self,
        a_scalar: float,
        b_vec: np.ndarray,
        c_vec: np.ndarray,
        x_scalar: float,
        state: np.ndarray,
    ) -> Tuple[float, np.ndarray]:
        """Single-token O(d_state) recurrent update.

        Args:
            a_scalar: Pre-computed exp(log_A) scalar for this step.
            b_vec:    B vector ``(d_state,)`` float32.
            c_vec:    C vector ``(d_state,)`` float32.
            x_scalar: Input scalar for this token.
            state:    Current recurrent state ``(d_state,)`` float32.

        Returns:
            ``(y_scalar, new_state (d_state,))``.
        """
        b_f = np.ascontiguousarray(b_vec, dtype=np.float32).ravel()
        c_f = np.ascontiguousarray(c_vec, dtype=np.float32).ravel()
        s_f = np.ascontiguousarray(state, dtype=np.float32).ravel()
        if _HAS_RUST:
            y, ns = _sq.mamba2_ssm_decode_f32(
                float(a_scalar), b_f, c_f, float(x_scalar), s_f
            )
            return float(y), np.asarray(ns, dtype=np.float32)
        return _numpy_decode(float(a_scalar), b_f, c_f, float(x_scalar), s_f)

    # ── properties ───────────────────────────────────────────────────────────

    def d_state(self) -> int:
        """SSM state dimension."""
        return self._cfg.d_state

    def d_model(self) -> int:
        """Model input dimension."""
        return self._cfg.d_model

    def backend(self) -> str:
        """Return ``'rust'`` or ``'numpy'`` depending on availability."""
        return "rust" if _HAS_RUST else "numpy"
