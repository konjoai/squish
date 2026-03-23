"""squish/kernels/mojo/pyramid_kv_budget_mojo.py — Mojo-backed PyramidKV budget.

Wraps ``pyramid_kv_budget`` Mojo kernel via MojoBridge with a NumPy fallback.
Computes per-layer KV-cache token budgets using a linear-decay schedule
parallelised over layers.

Reference: Cai et al., "PyramidKV: Dynamic KV Cache Compression based on
Pyramidal Information Funneling," arXiv 2406.02069, 2024.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from squish.kernels.mojo.mojo_bridge import MojoBridge

__all__ = [
    "PyramidKVBudgetMojoConfig",
    "MojoPyramidKVBudget",
]

_bridge = MojoBridge()
_budget_kernel = _bridge.load_kernel("pyramid_kv_budget")


# ── NumPy fallback ────────────────────────────────────────────────────────────


def _numpy_budget(base: float, alpha: float, n_layers: int, min_budget: int) -> np.ndarray:
    """Compute linearly-decaying per-layer budgets.

    Args:
        base:       Top-layer budget.
        alpha:      Decay factor.
        n_layers:   Number of layers.
        min_budget: Floor value.

    Returns:
        ``(n_layers,)`` int32 budgets.
    """
    out = np.empty(n_layers, dtype=np.int32)
    for l in range(n_layers):
        frac = l / (n_layers - 1) if n_layers > 1 else 0.0
        out[l] = max(min_budget, round(base * (1.0 - alpha * frac)))
    return out


# ── Config / wrapper ──────────────────────────────────────────────────────────


@dataclass
class PyramidKVBudgetMojoConfig:
    """Configuration for :class:`MojoPyramidKVBudget`.

    Attributes:
        alpha:      Decay strength in [0, 1].
        min_budget: Floor token budget per layer.
    """

    alpha: float = 0.5
    min_budget: int = 32


class MojoPyramidKVBudget:
    """Mojo-backed PyramidKV per-layer budget computation.

    Uses ``parallelize`` over layers for the linear-decay budget calculation.
    Falls back to NumPy when the Mojo runtime is absent.
    """

    def __init__(self, config: Optional[PyramidKVBudgetMojoConfig] = None) -> None:
        self._cfg = config or PyramidKVBudgetMojoConfig()

    def compute(
        self,
        base: float,
        n_layers: int,
        alpha: Optional[float] = None,
        min_budget: Optional[int] = None,
    ) -> np.ndarray:
        """Compute per-layer KV-cache token budgets.

        Args:
            base:       Budget for the shallowest layer.
            n_layers:   Total number of transformer layers.
            alpha:      Decay strength (overrides config).
            min_budget: Floor budget per layer (overrides config).

        Returns:
            ``(n_layers,)`` int32 monotonically non-increasing budgets.

        Raises:
            ValueError: If ``n_layers < 1`` or ``base < 0``.
        """
        if n_layers < 1:
            raise ValueError(f"n_layers must be ≥ 1, got {n_layers}")
        if base < 0:
            raise ValueError(f"base must be non-negative, got {base}")
        a = float(alpha) if alpha is not None else self._cfg.alpha
        mb = int(min_budget) if min_budget is not None else self._cfg.min_budget
        if _budget_kernel is not None:
            out = np.empty(n_layers, dtype=np.int32)
            _budget_kernel(out.ctypes.data, float(base), a, n_layers, mb)
            return out
        return _numpy_budget(float(base), a, int(n_layers), mb)

    def total(
        self,
        base: float,
        n_layers: int,
        alpha: Optional[float] = None,
        min_budget: Optional[int] = None,
    ) -> int:
        """Return sum of all per-layer budgets.

        Args:
            base:       Top-layer budget.
            n_layers:   Total layer count.
            alpha:      Decay strength (overrides config).
            min_budget: Floor value (overrides config).

        Returns:
            Total KV slots across all layers as a Python int.
        """
        return int(self.compute(base, n_layers, alpha, min_budget).sum())

    def backend(self) -> str:
        return "mojo" if _budget_kernel is not None else "numpy"
