"""squish/lora/dora.py

DoRAAdapter — Weight-Decomposed Low-Rank Adaptation.

Reference
---------
Liu et al. "DoRA: Weight-Decomposed Low-Rank Adaptation."
ICML 2024 (arXiv 2402.09353).

Algorithm
---------
DoRA decomposes the weight matrix ``W₀`` into magnitude and direction:

    W₀ = m · (W₀ / ‖W₀‖_c)  where ‖·‖_c is column-wise L2 norm.

A standard LoRA adapter ``ΔW = B·A`` (with A/B low-rank) is applied to
the direction component only.  The adapted weight is:

    W' = m · (V + ΔW) / ‖V + ΔW‖_c

where ``V = W₀ / ‖W₀‖_c``.  This gives LoRA-equivalent parameter count
while maintaining gradient properties closer to full fine-tuning.

``merge_to_weight()`` absorbs the adapter back into a base weight matrix
for zero-overhead inference.

This module provides:

1. ``DoRAAdapter.adapted_weight()`` — compute the full adapted weight.
2. ``DoRAAdapter.forward(x)`` — apply the adapted linear transform.
3. ``DoRAAdapter.merge_to_weight()`` — return a merged weight with no adapter overhead.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

__all__ = [
    "DoRAConfig",
    "DoRAAdapter",
]


@dataclass
class DoRAConfig:
    """Configuration for :class:`DoRAAdapter`.

    Attributes:
        d_in: Input dimension.
        d_out: Output dimension.
        rank: LoRA rank (number of singular vectors).
        seed: RNG seed.
    """

    d_in: int = 4096
    d_out: int = 4096
    rank: int = 16
    seed: int = 0

    def __post_init__(self) -> None:
        if self.d_in < 1:
            raise ValueError(f"d_in must be >= 1; got {self.d_in}")
        if self.d_out < 1:
            raise ValueError(f"d_out must be >= 1; got {self.d_out}")
        if self.rank < 1:
            raise ValueError(f"rank must be >= 1; got {self.rank}")


class DoRAAdapter:
    """Weight-decomposed LoRA adapter.

    The base weight is stored as a fixed ``(d_in, d_out)`` matrix.  The
    learnable parameters are:

    * ``magnitude`` — ``(d_out,)`` column-wise scale.
    * ``lora_A`` — ``(d_in, rank)``.
    * ``lora_B`` — ``(rank, d_out)``.

    In this simulation, all parameters are initialised with small random
    values for testability.

    Example::

        cfg = DoRAConfig(d_in=32, d_out=32, rank=4)
        adapter = DoRAAdapter(cfg)

        x = np.random.randn(6, 32).astype(np.float32)
        y = adapter.forward(x)
        assert y.shape == (6, 32)
    """

    def __init__(self, config: Optional[DoRAConfig] = None) -> None:
        self._cfg = config or DoRAConfig()
        c = self._cfg
        rng = np.random.default_rng(c.seed)
        scale = np.sqrt(1.0 / c.d_in)

        # Frozen base weight (d_in, d_out)
        self._W0: np.ndarray = (
            rng.standard_normal((c.d_in, c.d_out)).astype(np.float32) * scale
        )
        # Column-wise L2 norms of the base weight: (d_out,)
        col_norms = np.linalg.norm(self._W0, axis=0, keepdims=False)
        col_norms = np.maximum(col_norms, 1e-8)

        # Learnable magnitude vector (d_out,)
        self._magnitude: np.ndarray = col_norms.copy()

        # Direction (unit-norm columns): stored as (d_in, d_out)
        self._V0: np.ndarray = self._W0 / col_norms[np.newaxis, :]

        # LoRA low-rank factors
        scale_a = np.sqrt(1.0 / c.d_in)
        self._lora_A: np.ndarray = (
            rng.standard_normal((c.d_in, c.rank)).astype(np.float32) * scale_a
        )
        # B initialised to near-zero so adapter starts as a small perturbation
        self._lora_B: np.ndarray = (
            rng.standard_normal((c.rank, c.d_out)).astype(np.float32) * 1e-4
        )

    @property
    def config(self) -> DoRAConfig:
        return self._cfg

    @property
    def magnitude(self) -> np.ndarray:
        """Column-wise magnitude ``(d_out,)``."""
        return self._magnitude

    @property
    def direction(self) -> np.ndarray:
        """Unit-column direction matrix ``(d_in, d_out)``."""
        return self._V0

    @property
    def lora_A(self) -> np.ndarray:
        """Low-rank A factor ``(d_in, rank)``."""
        return self._lora_A

    @property
    def lora_B(self) -> np.ndarray:
        """Low-rank B factor ``(rank, d_out)``."""
        return self._lora_B

    def adapted_weight(self) -> np.ndarray:
        """Compute the full adapted weight matrix ``(d_in, d_out)``.

        Returns:
            ``W' = m * (V + ΔW) / ‖V + ΔW‖_c``
        """
        delta_W = self._lora_A @ self._lora_B           # (d_in, d_out)
        V_hat = self._V0 + delta_W
        col_norms = np.linalg.norm(V_hat, axis=0, keepdims=False)
        col_norms = np.maximum(col_norms, 1e-8)
        W_prime = self._magnitude[np.newaxis, :] * (V_hat / col_norms[np.newaxis, :])
        return W_prime.astype(np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Apply the adapted linear layer.

        Args:
            x: ``(*, d_in)`` input tensor.

        Returns:
            ``(*, d_out)`` output.
        """
        W = self.adapted_weight()
        return np.asarray(x, dtype=np.float32) @ W

    def merge_to_weight(self) -> np.ndarray:
        """Return the fully merged weight for zero-overhead inference.

        This is identical to :meth:`adapted_weight`; the separate method
        exists to make the intent explicit at the call site.

        Returns:
            ``(d_in, d_out)`` merged weight.
        """
        return self.adapted_weight()
