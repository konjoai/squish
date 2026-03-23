"""squish/kernels/mojo/hawk_rglr_mojo.py — Mojo-backed Hawk RGLR scan.

Wraps the ``hawk_rglr`` Mojo kernel via MojoBridge with a NumPy fallback.

Reference: De et al., "Griffin: Mixing Gated Linear Recurrences with Local
Attention for Efficient LLMs." arXiv 2402.19427 / NeurIPS 2024.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from squish.kernels.mojo.mojo_bridge import MojoBridge

__all__ = [
    "HawkRGLRMojoConfig",
    "MojoHawkRGLR",
]

_bridge = MojoBridge()
_kernel = _bridge.load_kernel("hawk_rglr")


# ── NumPy fallback ────────────────────────────────────────────────────────────


def _numpy_rglr(
    x: np.ndarray,
    dt: np.ndarray,
    lambda_log: np.ndarray,
    h0: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    t_len, d_state = x.shape
    h = h0.copy()
    out = np.empty((t_len, d_state), dtype=np.float32)
    for t in range(t_len):
        sp = np.log1p(np.exp(dt[t]))
        decay = np.exp(-np.exp(lambda_log) * sp)
        i_gate = np.sqrt(np.maximum(1.0 - decay ** 2, 0.0))
        h = decay * h + i_gate * x[t]
        out[t] = h
    return out.astype(np.float32), h.astype(np.float32)


# ── Config / wrapper ──────────────────────────────────────────────────────────


@dataclass
class HawkRGLRMojoConfig:
    """Configuration for :class:`MojoHawkRGLR`.

    Attributes:
        d_state: Recurrent state dimension.
    """

    d_state: int = 512


class MojoHawkRGLR:
    """Mojo-backed Hawk Real-Gated Linear Recurrence scan.

    Uses a Mojo kernel with ``parallelize`` over d_state channels and
    vectorised element-wise gate computation.
    Falls back to NumPy when the Mojo runtime is absent.
    """

    def __init__(self, config: Optional[HawkRGLRMojoConfig] = None) -> None:
        self._cfg = config or HawkRGLRMojoConfig()

    def scan(
        self,
        x: np.ndarray,
        dt: np.ndarray,
        lambda_log: np.ndarray,
        h0: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run the RGLR scan over a full sequence.

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
        if _kernel is not None:
            t_len = x_f.shape[0]
            out_buf = np.zeros((t_len, d_state), dtype=np.float32)
            state_buf = h0_f.copy()
            _kernel(
                x_f.ctypes.data, dt_f.ctypes.data, lam_f.ctypes.data,
                state_buf.ctypes.data, out_buf.ctypes.data, t_len, d_state,
            )
            return out_buf, state_buf
        return _numpy_rglr(x_f, dt_f, lam_f, h0_f)

    def d_state(self) -> int:
        return self._cfg.d_state

    def backend(self) -> str:
        return "mojo" if _kernel is not None else "numpy"
