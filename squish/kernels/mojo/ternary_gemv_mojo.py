"""squish/kernels/mojo/ternary_gemv_mojo.py — Mojo-backed ternary GEMV.

Wraps the ``ternary_gemv`` Mojo kernel via MojoBridge with a NumPy fallback.
Implements INT8 {-1, 0, +1} weight × FP32 activation GEMV (BitNet b1.58 path).

Reference: Ma et al., "The Era of 1-bit LLMs: All Large Language Models are
in 1.58 Bits." arXiv 2402.17764.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from squish.kernels.mojo.mojo_bridge import MojoBridge

__all__ = [
    "TernaryGEMVMojoConfig",
    "MojoTernaryGEMV",
]

_bridge = MojoBridge()
_kernel = _bridge.load_kernel("ternary_gemv")


# ── NumPy fallback ────────────────────────────────────────────────────────────


def _numpy_ternary_gemv(
    weight: np.ndarray,
    activation: np.ndarray,
    scale: float,
) -> np.ndarray:
    pos = weight == 1
    neg = weight == -1
    return ((pos @ activation - neg @ activation) * scale).astype(np.float32)


# ── Config / wrapper ──────────────────────────────────────────────────────────


@dataclass
class TernaryGEMVMojoConfig:
    """Configuration for :class:`MojoTernaryGEMV`.

    Attributes:
        scale: Default dequantisation scale (mean(|W_orig|)).
    """

    scale: float = 1.0


class MojoTernaryGEMV:
    """Mojo-backed ternary GEMV: INT8 {-1,0,+1} × FP32.

    Uses SIMD signed-integer comparison to avoid multiplication, accumulates
    in FP32.  ``parallelize`` over output rows.
    Falls back to NumPy when the Mojo runtime is absent.
    """

    def __init__(self, config: Optional[TernaryGEMVMojoConfig] = None) -> None:
        self._cfg = config or TernaryGEMVMojoConfig()

    def gemv(
        self,
        weight: np.ndarray,
        activation: np.ndarray,
        scale: Optional[float] = None,
    ) -> np.ndarray:
        """Compute W @ a with ternary SIMD acceleration.

        Args:
            weight:     INT8 ternary weight matrix
                        ``(out_features, in_features)`` in {-1, 0, +1}.
            activation: FP32 activation vector ``(in_features,)``.
            scale:      Override dequantisation scale.

        Returns:
            Output vector ``(out_features,)`` float32.

        Raises:
            ValueError: If weight columns != activation length.
        """
        w = np.ascontiguousarray(weight, dtype=np.int8)
        a = np.ascontiguousarray(activation, dtype=np.float32).ravel()
        if w.shape[1] != a.shape[0]:
            raise ValueError(
                f"weight in_features={w.shape[1]} != activation length={a.shape[0]}"
            )
        s = float(scale) if scale is not None else self._cfg.scale
        if _kernel is not None:
            out_buf = np.zeros(w.shape[0], dtype=np.float32)
            _kernel(w.ctypes.data, a.ctypes.data, out_buf.ctypes.data, w.shape[0], w.shape[1], s)
            return out_buf
        return _numpy_ternary_gemv(w, a, s)

    def sparsity(self, weight: np.ndarray) -> float:
        """Return fraction of zero weights."""
        return float((np.asarray(weight, dtype=np.int8).ravel() == 0).mean())

    def scale(self) -> float:
        return self._cfg.scale

    def backend(self) -> str:
        return "mojo" if _kernel is not None else "numpy"
