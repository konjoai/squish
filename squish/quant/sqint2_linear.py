"""SQINT2Linear: coherent INT2 linear layer with NF2 codebook and optional MLX backend.

W103.4c — two implementations:
- SQINT2LinearNumPy: always available (reference / CPU)
- SQINT2LinearMLX:   Apple Silicon only (sys.platform == "darwin" and mlx installed)

IMPORTANT: MLX imports are deferred to __init__/__call__ (never at module level)
to avoid triggering Metal GPU initialisation during import, which causes SIGABRT
on CI runners where the Metal context is unavailable.
"""
from __future__ import annotations

import sys

import numpy as np

# ---------------------------------------------------------------------------
# NF2 codebook (4 levels, fits INT2)
# ---------------------------------------------------------------------------
NF2_CODEBOOK = np.array([-1.5, -0.5, 0.5, 1.5], dtype=np.float32)


def _nf2_dequantize_numpy(
    packed: np.ndarray,
    scale: float = 1.0,
    zero_point: float = 0.0,
) -> np.ndarray:
    """Dequantize 2-bit packed indices using the NF2 codebook → float32."""
    indices = packed.astype(np.int32) & 0x3
    weights = NF2_CODEBOOK[indices] * scale + zero_point
    return weights.astype(np.float32)


def _residual_gemv_numpy(x: np.ndarray, L: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Low-rank residual: x @ (L @ R)ᵀ without materialising L@R."""
    h = x @ R.T
    return h @ L.T


# ---------------------------------------------------------------------------
# NumPy reference implementation (always available)
# ---------------------------------------------------------------------------

class SQINT2LinearNumPy:
    """INT2 linear layer backed by NumPy — CPU reference, no extra deps."""

    def __init__(
        self,
        packed_weight: np.ndarray,
        scale: float = 1.0,
        zero_point: float = 0.0,
        residual_L: np.ndarray | None = None,
        residual_R: np.ndarray | None = None,
    ) -> None:
        self.packed_weight = packed_weight.astype(np.uint8)
        self.scale = float(scale)
        self.zero_point = float(zero_point)
        self.residual_L = residual_L
        self.residual_R = residual_R
        self._weight = _nf2_dequantize_numpy(self.packed_weight, scale, zero_point)

    @property
    def weight(self) -> np.ndarray:
        return self._weight

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute INT2 GEMV: x @ Wᵀ + residual."""
        out = x @ self._weight.T
        if self.residual_L is not None and self.residual_R is not None:
            out = out + _residual_gemv_numpy(x, self.residual_L, self.residual_R)
        return out

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)


# ---------------------------------------------------------------------------
# MLX implementation — Apple Silicon only
#
# The class is always defined (no module-level `if _mlx_available()` branch)
# so that `from squish.quant.sqint2_linear import SQINT2LinearMLX` never
# triggers Metal initialisation.  The real vs. stub behaviour is selected
# lazily inside __init__ by checking _mlx_available() at call time.
# ---------------------------------------------------------------------------

def _mlx_available() -> bool:
    if sys.platform != "darwin":
        return False
    try:
        import mlx.core  # noqa: F401
        return True
    except ImportError:
        return False


class SQINT2LinearMLX:
    """INT2 linear layer backed by MLX (Apple Silicon / Metal) when available.

    On non-Darwin platforms or when MLX is not installed, construction raises
    ImportError immediately.  MLX is imported lazily inside __init__ so that
    merely *importing* this module never triggers Metal GPU initialisation.
    """

    def __init__(
        self,
        packed_weight: np.ndarray,
        scale: float = 1.0,
        zero_point: float = 0.0,
        residual_L: np.ndarray | None = None,
        residual_R: np.ndarray | None = None,
    ) -> None:
        if not _mlx_available():
            raise ImportError(
                "SQINT2LinearMLX requires Apple Silicon with MLX installed. "
                "pip install mlx (macOS only)."
            )
        # Defer MLX imports to here — Metal context is only initialised on
        # actual construction, never on a bare module import.
        import mlx.core as mx  # noqa: PLC0415
        import mlx.nn as nn_mlx  # noqa: PLC0415

        # Store mx for use in __call__ without a class-level import.
        self._mx = mx

        dequant = _nf2_dequantize_numpy(packed_weight, scale, zero_point)
        self.weight = mx.array(dequant)
        self.scale = scale
        self.zero_point = zero_point
        self.residual_L = mx.array(residual_L) if residual_L is not None else None
        self.residual_R = mx.array(residual_R) if residual_R is not None else None

    def __call__(self, x: object) -> object:
        mx = self._mx
        out = x @ self.weight.T
        if self.residual_L is not None and self.residual_R is not None:
            h = x @ self.residual_R.T
            out = out + h @ self.residual_L.T
        return out


def get_sqint2_linear_info() -> dict:
    """Return availability info for SQINT2Linear backends."""
    return {
        "sqint2_linear_numpy": True,
        "sqint2_linear_mlx": _mlx_available(),
        "platform": sys.platform,
    }


__all__ = [
    "NF2_CODEBOOK",
    "SQINT2LinearNumPy",
    "SQINT2LinearMLX",
    "_nf2_dequantize_numpy",
    "_residual_gemv_numpy",
    "get_sqint2_linear_info",
]
