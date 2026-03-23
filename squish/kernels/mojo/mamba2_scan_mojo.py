"""squish/kernels/mojo/mamba2_scan_mojo.py — Mojo-backed Mamba-2 SSD scan.

Wraps the ``mamba2_scan`` Mojo kernel via MojoBridge with a NumPy fallback.

Reference: Dao & Gu, "Transformers are SSMs: Generalized Models and Efficient
Algorithms Through Structured State Space Duality." ICML 2024.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from squish.kernels.mojo.mojo_bridge import MojoBridge

__all__ = [
    "Mamba2ScanMojoConfig",
    "MojoMamba2Scan",
]

_bridge = MojoBridge()
_kernel = _bridge.load_kernel("mamba2_scan")


# ── NumPy fallback ────────────────────────────────────────────────────────────


def _numpy_scan(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    x: np.ndarray,
    h0: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    t_len, d_state = b.shape
    h = h0.copy()
    out = np.empty(t_len, dtype=np.float32)
    for t in range(t_len):
        h = float(np.exp(a[t])) * h + b[t] * x[t]
        out[t] = float(np.dot(c[t], h))
    return out, h.astype(np.float32)


# ── Config / wrapper ──────────────────────────────────────────────────────────


@dataclass
class Mamba2ScanMojoConfig:
    """Configuration for :class:`MojoMamba2Scan`.

    Attributes:
        d_state: SSM state dimension.
        chunk_size: Chunk size for parallel scan.
    """

    d_state: int = 64
    chunk_size: int = 64


class MojoMamba2Scan:
    """Mojo-backed Mamba-2 SSD chunked scan.

    Uses a Mojo kernel with ``parallelize`` over batch × head dimension.
    Falls back to NumPy sequential scan when the Mojo runtime is absent.
    """

    def __init__(self, config: Optional[Mamba2ScanMojoConfig] = None) -> None:
        self._cfg = config or Mamba2ScanMojoConfig()

    def scan(
        self,
        a: np.ndarray,
        b: np.ndarray,
        c: np.ndarray,
        x: np.ndarray,
        h0: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run the SSD scan. See :class:`RustMamba2SSM` for arg details."""
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
        if _kernel is not None:
            t_len = int(a_f.shape[0])
            out_buf = np.zeros(t_len, dtype=np.float32)
            state_buf = h0_f.copy()
            _kernel(
                a_f.ctypes.data, b_f.ctypes.data, c_f.ctypes.data,
                x_f.ctypes.data, state_buf.ctypes.data,
                out_buf.ctypes.data, t_len, d_state,
            )
            return out_buf, state_buf
        return _numpy_scan(a_f, b_f, c_f, x_f, h0_f)

    def d_state(self) -> int:
        return self._cfg.d_state

    def chunk_size(self) -> int:
        return self._cfg.chunk_size

    def backend(self) -> str:
        return "mojo" if _kernel is not None else "numpy"
