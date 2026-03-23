"""squish/kernels/rs_hawk_rglr.py — Rust-backed Hawk RGLR recurrent scan.

Wraps ``squish_quant_rs.hawk_rglr_scan_f32`` with a NumPy fallback.

Reference: De et al., "Griffin: Mixing Gated Linear Recurrences with Local
Attention for Efficient LLMs." arXiv 2402.19427 / NeurIPS 2024.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

__all__ = [
    "HawkRGLRConfig",
    "RustHawkRGLR",
]

try:
    import squish_quant as _sq
    _HAS_RUST = hasattr(_sq, "hawk_rglr_scan_f32")
except ImportError:
    _sq = None  # type: ignore[assignment]
    _HAS_RUST = False


# ── NumPy fallback ────────────────────────────────────────────────────────────


def _numpy_rglr_scan(
    x: np.ndarray,          # (T, d_state)
    dt: np.ndarray,         # (T, d_state)
    lambda_log: np.ndarray, # (d_state,)
    h0: np.ndarray,         # (d_state,)
) -> Tuple[np.ndarray, np.ndarray]:
    t_len, d_state = x.shape
    h = h0.copy()
    out = np.empty((t_len, d_state), dtype=np.float32)
    for t in range(t_len):
        sp = np.log1p(np.exp(dt[t]))           # softplus(dt)
        decay = np.exp(-np.exp(lambda_log) * sp)  # f_t
        input_gate = np.sqrt(np.maximum(1.0 - decay ** 2, 0.0))  # i_t
        h = decay * h + input_gate * x[t]
        out[t] = h
    return out.astype(np.float32), h.astype(np.float32)


# ── Config / wrapper ──────────────────────────────────────────────────────────


@dataclass
class HawkRGLRConfig:
    """Configuration for :class:`RustHawkRGLR`.

    Attributes:
        d_state: Recurrent state dimension.
        d_model: Model input dimension.
    """

    d_state: int = 512
    d_model: int = 512


class RustHawkRGLR:
    """Rust-accelerated Hawk Real-Gated Linear Recurrence scan.

    Implements the RGLR state-update recurrence that is the core primitive
    of Hawk/Griffin models.  Falls back to a NumPy sequential scan when
    ``squish_quant_rs`` is unavailable.
    """

    def __init__(self, config: Optional[HawkRGLRConfig] = None) -> None:
        self._cfg = config or HawkRGLRConfig()

    def scan(
        self,
        x: np.ndarray,
        dt: np.ndarray,
        lambda_log: np.ndarray,
        h0: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run the RGLR scan over a full sequence.

        Args:
            x:          Projected input ``(T, d_state)`` float32.
            dt:         Delta-time inputs ``(T, d_state)`` float32.
            lambda_log: Log-eigenvalues ``(d_state,)`` float32.
            h0:         Initial state ``(d_state,)``; zeros if *None*.

        Returns:
            ``(outputs (T, d_state), final_state (d_state,))`` float32.
        """
        x_f = np.ascontiguousarray(x, dtype=np.float32)
        dt_f = np.ascontiguousarray(dt, dtype=np.float32)
        lam_f = np.ascontiguousarray(lambda_log, dtype=np.float32).ravel()
        d_state = lam_f.shape[0]
        h0_f = (
            np.ascontiguousarray(h0, dtype=np.float32).ravel()
            if h0 is not None
            else np.zeros(d_state, dtype=np.float32)
        )
        if _HAS_RUST:
            out_flat, fs = _sq.hawk_rglr_scan_f32(x_f, dt_f, lam_f, h0_f)
            t_len = x_f.shape[0]
            return (
                np.asarray(out_flat, dtype=np.float32).reshape(t_len, d_state),
                np.asarray(fs, dtype=np.float32),
            )
        return _numpy_rglr_scan(x_f, dt_f, lam_f, h0_f)

    # ── properties ───────────────────────────────────────────────────────────

    def d_state(self) -> int:
        return self._cfg.d_state

    def d_model(self) -> int:
        return self._cfg.d_model

    def backend(self) -> str:
        return "rust" if _HAS_RUST else "numpy"
