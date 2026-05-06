"""SQINT2Linear: coherent INT2 linear layer with NF2 codebook and optional MLX backend.

W103.4c — two implementations:
- SQINT2LinearNumPy: always available (reference / CPU)
- SQINT2LinearMLX:   Apple Silicon only (sys.platform == "darwin" and mlx installed)
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
# ---------------------------------------------------------------------------

def _mlx_available() -> bool:
    if sys.platform != "darwin":
        return False
    try:
        import mlx.core  # noqa: F401
        return True
    except ImportError:
        return False


if _mlx_available():
    import mlx.core as mx
    import mlx.nn as nn_mlx

    class SQINT2LinearMLX(nn_mlx.Module):  # type: ignore[no-redef]
        """INT2 linear layer backed by MLX — Apple Silicon (Metal) only."""

        def __init__(
            self,
            packed_weight: np.ndarray,
            scale: float = 1.0,
            zero_point: float = 0.0,
            residual_L: np.ndarray | None = None,
            residual_R: np.ndarray | None = None,
        ) -> None:
            super().__init__()
            dequant = _nf2_dequantize_numpy(packed_weight, scale, zero_point)
            self.weight = mx.array(dequant)
            self.scale = scale
            self.zero_point = zero_point
            self.residual_L = mx.array(residual_L) if residual_L is not None else None
            self.residual_R = mx.array(residual_R) if residual_R is not None else None

        def __call__(self, x: mx.array) -> mx.array:  # type: ignore[override]
            out = x @ self.weight.T
            if self.residual_L is not None and self.residual_R is not None:
                h = x @ self.residual_R.T
                out = out + h @ self.residual_L.T
            return out

else:
    class SQINT2LinearMLX:  # type: ignore[no-redef]
        """Stub — MLX not available on this platform."""

        def __init__(self, *args, **kwargs) -> None:
            raise ImportError(
                "SQINT2LinearMLX requires Apple Silicon with MLX installed. "
                "pip install mlx (macOS only)."
            )

        def __call__(self, *args, **kwargs):  # type: ignore[override]
            raise ImportError("SQINT2LinearMLX is not available on this platform.")


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
