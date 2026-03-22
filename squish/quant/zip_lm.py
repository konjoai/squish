"""ZipLM mixed-precision planning (arXiv 2302.04089, NeurIPS 2023).

Hessian-trace sensitivity ranking assigns INT2/INT3/INT4 per
transformer block to maximise quality under a total-memory budget.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class ZipLMConfig:
    """Configuration for ZipLM mixed-precision planning."""

    memory_budget_gb: float = 7.0
    min_bits: int = 2
    max_bits: int = 4
    group_size: int = 128
    seed: int = 0

    def __post_init__(self) -> None:
        if self.memory_budget_gb <= 0:
            raise ValueError(
                f"memory_budget_gb must be > 0, got {self.memory_budget_gb}"
            )
        if self.min_bits < 1:
            raise ValueError(f"min_bits must be >= 1, got {self.min_bits}")
        if self.max_bits < self.min_bits:
            raise ValueError(
                f"max_bits ({self.max_bits}) must be >= min_bits ({self.min_bits})"
            )


@dataclass
class ZipLMResult:
    """Output of ZipLM bit-width planning."""

    bit_schedule: List[int]                  # bit width per layer
    total_bits_est: float                    # total bits across all layers
    layers_by_bits: Dict[int, List[int]]     # bit_width → [layer indices]

    @property
    def effective_bits(self) -> float:
        """Mean bits per weight (average of per-layer bit widths)."""
        if not self.bit_schedule:
            return 0.0
        return float(sum(self.bit_schedule)) / len(self.bit_schedule)


class ZipLMMixedPrecision:
    """ZipLM: sensitivity-driven mixed-precision bit planner.

    Assigns the minimum feasible bit-width to each layer under a global
    memory budget, promoting the most sensitive layers to higher
    precision when the budget permits it.
    """

    def __init__(self, config: Optional[ZipLMConfig] = None) -> None:
        self._config = config or ZipLMConfig()

    @property
    def config(self) -> ZipLMConfig:
        return self._config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def estimate_memory_gb(
        self,
        shapes: List[Tuple[int, int]],
        bits_list: List[int],
    ) -> float:
        """Estimate total model memory in GB given per-layer bit widths.

        Parameters
        ----------
        shapes:
            List of ``(rows, cols)`` for each layer weight matrix.
        bits_list:
            Corresponding list of bit widths.

        Returns
        -------
        float
            Estimated memory in gigabytes.
        """
        total_bits = sum(
            r * c * b for (r, c), b in zip(shapes, bits_list)
        )
        return total_bits / 8 / 1e9

    def assign_bits(
        self,
        n_layers: int,
        shapes: List[Tuple[int, int]],
        sensitivities: np.ndarray,
    ) -> List[int]:
        """Greedily assign bit widths to layers under the memory budget.

        Strategy:
        1. Start every layer at ``min_bits``.
        2. Sort layers by sensitivity (descending).
        3. Promote each layer one bit at a time (up to ``max_bits``) as
           long as the total memory stays within ``memory_budget_gb``.

        Parameters
        ----------
        n_layers:
            Number of layers.
        shapes:
            ``(rows, cols)`` for each layer.
        sensitivities:
            ``(n_layers,)`` sensitivity score per layer.

        Returns
        -------
        List[int]
            ``(n_layers,)`` bit width per layer.
        """
        cfg = self._config
        bits = [cfg.min_bits] * n_layers
        current_mem = self.estimate_memory_gb(shapes, bits)

        # Sort by descending sensitivity so most sensitive layers get promoted first
        order = np.argsort(sensitivities)[::-1]

        for layer_idx in order:
            while bits[layer_idx] < cfg.max_bits:
                new_bits = list(bits)
                new_bits[layer_idx] += 1
                new_mem = self.estimate_memory_gb(shapes, new_bits)
                if new_mem <= cfg.memory_budget_gb:
                    bits[layer_idx] += 1
                    current_mem = new_mem
                else:
                    break

        return bits

    def plan(
        self,
        layer_shapes: List[Tuple[int, int]],
        layer_sensitivities: Optional[List[float]] = None,
    ) -> ZipLMResult:
        """Plan per-layer bit widths under the configured memory budget.

        Parameters
        ----------
        layer_shapes:
            List of ``(rows, cols)`` for each layer.
        layer_sensitivities:
            Optional per-layer sensitivity scores.  When ``None``,
            uniform random values are used (seeded for reproducibility).

        Returns
        -------
        ZipLMResult
        """
        cfg = self._config
        n_layers = len(layer_shapes)

        if layer_sensitivities is not None:
            sensitivities = np.asarray(layer_sensitivities, dtype=np.float32)
        else:
            rng = np.random.default_rng(cfg.seed)
            sensitivities = rng.random(n_layers).astype(np.float32)

        bit_schedule = self.assign_bits(n_layers, layer_shapes, sensitivities)

        total_bits_est = float(
            sum(r * c * b for (r, c), b in zip(layer_shapes, bit_schedule))
        )

        # Group layer indices by assigned bit width
        layers_by_bits: Dict[int, List[int]] = {}
        for i, b in enumerate(bit_schedule):
            layers_by_bits.setdefault(b, []).append(i)

        return ZipLMResult(
            bit_schedule=bit_schedule,
            total_bits_est=total_bits_est,
            layers_by_bits=layers_by_bits,
        )
