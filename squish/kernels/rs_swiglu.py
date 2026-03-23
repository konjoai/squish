"""rs_swiglu.py — Rust-accelerated SwiGLU / SiLU activation kernels.

Wraps `squish_quant.swiglu_f32` and `squish_quant.silu_f32` (Wave 57a).
Falls back to pure-NumPy implementations when the Rust extension is
unavailable.

RustSwiGLU fuses the SiLU gate activation and element-wise multiply
into one Rayon SIMD pass, eliminating an intermediate `silu_out` array
allocation and two NumPy ufunc dispatches (~3–4× speedup at ffn_dim=14336).

Reference:
  Shazeer (2020) — GLU Variants Improve Transformer (arXiv:2002.05202).
  Alizadeh et al. (ICML 2024) — LLM in a Flash (arXiv:2312.11514).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import squish_quant as _sq

    _RUST_AVAILABLE = all(
        hasattr(_sq, fn) for fn in ("swiglu_f32", "silu_f32")
    )
except ImportError:
    _RUST_AVAILABLE = False

__all__ = ["SwiGLUConfig", "RustSwiGLU"]


@dataclass
class SwiGLUConfig:
    """Configuration for RustSwiGLU.

    Attributes:
        ffn_dim: FFN intermediate dimension (e.g. 14336 for Llama-3-8B).
    """

    ffn_dim: int = 14336


class RustSwiGLU:
    """Rust-accelerated SwiGLU and SiLU activation kernels.

    Usage::

        swiglu = RustSwiGLU()
        gate = np.random.randn(14336).astype(np.float32)
        up   = np.random.randn(14336).astype(np.float32)
        out  = swiglu.forward(gate, up)   # shape (14336,)
        act  = swiglu.silu(gate)           # shape (14336,)
    """

    def __init__(self, config: SwiGLUConfig | None = None) -> None:
        self._cfg = config or SwiGLUConfig()

    def forward(
        self,
        gate: np.ndarray,
        up: np.ndarray,
    ) -> np.ndarray:
        """Compute fused SwiGLU: `SiLU(gate) * up`.

        Equivalent to `gate / (1 + exp(-gate)) * up`.

        Args:
            gate: Float32 1-D array `(N,)` — gate projection output.
            up:   Float32 1-D array `(N,)` — up projection output.

        Returns:
            Float32 1-D array `(N,)` — fused activation output.
        """
        gate = np.asarray(gate, dtype=np.float32)
        up = np.asarray(up, dtype=np.float32)
        if _RUST_AVAILABLE:
            return np.asarray(_sq.swiglu_f32(gate, up), dtype=np.float32)
        return self._numpy_swiglu(gate, up)

    def silu(self, x: np.ndarray) -> np.ndarray:
        """Compute fused SiLU activation: `x * sigmoid(x)`.

        Args:
            x: Float32 1-D array `(N,)`.

        Returns:
            Float32 1-D array `(N,)`.
        """
        x = np.asarray(x, dtype=np.float32)
        if _RUST_AVAILABLE:
            return np.asarray(_sq.silu_f32(x), dtype=np.float32)
        return self._numpy_silu(x)

    def ffn_dim(self) -> int:
        """Return configured FFN dimension."""
        return self._cfg.ffn_dim

    def backend(self) -> str:
        """Return which backend is being used: 'rust' or 'numpy'."""
        return "rust" if _RUST_AVAILABLE else "numpy"

    # ── NumPy fallbacks ── ----------------------------------------------------

    @staticmethod
    def _numpy_silu(x: np.ndarray) -> np.ndarray:
        """Pure-NumPy SiLU: x * sigmoid(x)."""
        return (x / (1.0 + np.exp(-x))).astype(np.float32)

    @staticmethod
    def _numpy_swiglu(gate: np.ndarray, up: np.ndarray) -> np.ndarray:
        """Pure-NumPy SwiGLU: SiLU(gate) * up."""
        return (gate / (1.0 + np.exp(-gate)) * up).astype(np.float32)
