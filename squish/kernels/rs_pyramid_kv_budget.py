"""squish/kernels/rs_pyramid_kv_budget.py — Rust-backed PyramidKV budget compute.

Wraps ``squish_quant_rs.pyramid_kv_budget_f32`` with a NumPy fallback.

PyramidKV allocates fewer KV-cache slots to deeper layers (those produce
more redundant attention) using a simple linear-decay schedule:

    budget[l] = max(min_budget, round(base * (1 − alpha * l / (L − 1))))

Parallelised over layers with Rayon.

Reference: Cai et al., "PyramidKV: Dynamic KV Cache Compression based on
Pyramidal Information Funneling," arXiv 2406.02069, 2024.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

__all__ = [
    "PyramidKVBudgetConfig",
    "RustPyramidKVBudget",
]

try:
    import squish_quant as _sq
    _HAS_RUST = hasattr(_sq, "pyramid_kv_budget_f32")
except ImportError:
    _sq = None  # type: ignore[assignment]
    _HAS_RUST = False


# ── NumPy fallback ────────────────────────────────────────────────────────────


def _numpy_budget(
    base: float,
    alpha: float,
    n_layers: int,
    min_budget: int,
) -> np.ndarray:
    """Compute per-layer KV budgets with linear decay.

    Args:
        base:      Top-layer budget (tokens).
        alpha:     Decay strength ∈ [0, 1].
        n_layers:  Number of transformer layers.
        min_budget: Floor budget.

    Returns:
        ``(n_layers,)`` int32 budgets.
    """
    budgets = np.empty(n_layers, dtype=np.int32)
    for l in range(n_layers):
        frac = l / (n_layers - 1) if n_layers > 1 else 0.0
        val = base * (1.0 - alpha * frac)
        budgets[l] = max(min_budget, round(val))
    return budgets


# ── Config / wrapper ──────────────────────────────────────────────────────────


@dataclass
class PyramidKVBudgetConfig:
    """Configuration for :class:`RustPyramidKVBudget`.

    Attributes:
        alpha:      Decay strength in [0, 1].
        min_budget: Minimum tokens per layer.
    """

    alpha: float = 0.5
    min_budget: int = 32


class RustPyramidKVBudget:
    """Rust-accelerated PyramidKV budget computation.

    Computes a linearly-decaying token budget per transformer layer for
    KV-cache compression.  Rayon parallelises the per-layer computation.

    Falls back to NumPy when ``squish_quant_rs`` is unavailable.

    Example::

        planner = RustPyramidKVBudget()
        budgets = planner.compute(base=512.0, n_layers=32)
        # budgets[0] ≈ 512,  budgets[-1] ≈ min_budget
    """

    def __init__(self, config: Optional[PyramidKVBudgetConfig] = None) -> None:
        self._cfg = config or PyramidKVBudgetConfig()

    def compute(
        self,
        base: float,
        n_layers: int,
        alpha: Optional[float] = None,
        min_budget: Optional[int] = None,
    ) -> np.ndarray:
        """Compute per-layer KV budgets with linear decay.

        Args:
            base:       Budget for the shallowest layer (tokens).
            n_layers:   Total transformer layer count.
            alpha:      Decay strength (overrides config).
            min_budget: Floor value (overrides config).

        Returns:
            ``(n_layers,)`` int32 budgets — monotonically non-increasing.

        Raises:
            ValueError: If ``n_layers < 1`` or ``base < 0``.
        """
        if n_layers < 1:
            raise ValueError(f"n_layers must be ≥ 1, got {n_layers}")
        if base < 0:
            raise ValueError(f"base must be non-negative, got {base}")
        a = float(alpha) if alpha is not None else self._cfg.alpha
        mb = int(min_budget) if min_budget is not None else self._cfg.min_budget
        if _HAS_RUST:
            return np.asarray(
                _sq.pyramid_kv_budget_f32(float(base), a, int(n_layers), mb),
                dtype=np.int32,
            )
        return _numpy_budget(float(base), a, int(n_layers), mb)

    def total(
        self,
        base: float,
        n_layers: int,
        alpha: Optional[float] = None,
        min_budget: Optional[int] = None,
    ) -> int:
        """Return total KV tokens across all layers.

        Args:
            base:       Budget for the shallowest layer (tokens).
            n_layers:   Total transformer layer count.
            alpha:      Decay strength (overrides config).
            min_budget: Floor value (overrides config).

        Returns:
            Sum of all per-layer budgets as a Python int.
        """
        return int(self.compute(base, n_layers, alpha, min_budget).sum())

    def backend(self) -> str:
        return "rust" if _HAS_RUST else "numpy"
