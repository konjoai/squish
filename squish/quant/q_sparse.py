"""squish/quant/q_sparse.py

Q-Sparse — Top-K Activation Sparsifier.

Q-Sparse retains only the top-K% of activations by absolute magnitude and
zeros the rest before the linear projection.  During inference this reduces
the effective FLOPs proportionally.  Per-layer calibration finds the
empirical average sparsity across a recorded activation corpus.

The approach pairs naturally with weight quantization:
* Apply Q-Sparse to activations before a quantized or full-precision matmul.
* Combined with INT8 weights: significant end-to-end throughput gain with
  minimal accuracy loss (≤0.3 PPL on LLaMA-2-7B at 50% sparsity).

Reference
---------
Sun, M. et al. "Q-Sparse: All Large Language Models can be Fully Sparsely-
Activated." Microsoft Research. arXiv:2407.10969, Jul 2024.
"""

from __future__ import annotations

__all__ = ["QSparseConfig", "QSparsifier"]

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from numpy import ndarray


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class QSparseConfig:
    """Configuration for QSparsifier.

    Parameters
    ----------
    top_k_ratio:
        Fraction of activations to retain, in (0, 1].  0.5 means the top-50%
        by magnitude are kept; the rest are zeroed.
    seed:
        RNG seed (reserved for future stochastic sparsification modes).
    """

    top_k_ratio: float = 0.5
    seed: int = 0

    def __post_init__(self) -> None:
        if not 0.0 < self.top_k_ratio <= 1.0:
            raise ValueError("top_k_ratio must be in (0, 1]")


# ---------------------------------------------------------------------------
# Sparsifier
# ---------------------------------------------------------------------------

class QSparsifier:
    """Top-K activation sparsifier following the Q-Sparse recipe.

    Parameters
    ----------
    config:
        ``QSparseConfig`` instance.
    """

    def __init__(self, config: QSparseConfig | None = None) -> None:
        self.config = config or QSparseConfig()

    def sparsify(self, x: ndarray) -> Tuple[ndarray, ndarray]:
        """Zero all but the top-``top_k_ratio`` activations by magnitude.

        Parameters
        ----------
        x:
            Activation vector or matrix, any shape.  Sparsification is
            applied element-wise across the last dimension for matrices --
            each row is sparsified independently.

        Returns
        -------
        x_sparse:
            Sparsified copy of ``x``, same shape as ``x``.
        mask:
            Boolean mask, ``True`` where activations were *kept*,
            same shape as ``x``.
        """
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            return self._sparsify_row(x)

        # 2-D: sparsify each row independently
        x_sparse = np.zeros_like(x)
        mask = np.zeros(x.shape, dtype=bool)
        for i in range(x.shape[0]):
            row_sparse, row_mask = self._sparsify_row(x[i])
            x_sparse[i] = row_sparse
            mask[i] = row_mask
        return x_sparse, mask

    def _sparsify_row(self, row: ndarray) -> Tuple[ndarray, ndarray]:
        """Sparsify a 1-D row by top-K% magnitude."""
        n = row.size
        k = max(1, int(np.ceil(n * self.config.top_k_ratio)))
        abs_vals = np.abs(row)
        # Partial sort: indices of top-k elements
        threshold_idx = np.argpartition(abs_vals, -k)[-k:]
        mask = np.zeros(n, dtype=bool)
        mask[threshold_idx] = True
        sparse = np.where(mask, row, 0.0)
        return sparse.astype(np.float32), mask

    def sparse_matmul(self, x: ndarray, W: ndarray) -> ndarray:
        """Sparsify ``x`` then compute ``x_sparse @ W.T``.

        Parameters
        ----------
        x:
            Activation matrix, shape ``(batch, in_features)``.
        W:
            Weight matrix, shape ``(out_features, in_features)``.

        Returns
        -------
        Output activations, shape ``(batch, out_features)``.
        """
        x_sparse, _ = self.sparsify(np.asarray(x, dtype=np.float32))
        return x_sparse @ np.asarray(W, dtype=np.float32).T

    def flop_reduction(self) -> float:
        """Theoretical FLOP reduction factor from sparsification.

        Returns
        -------
        Fraction of FLOPs eliminated, equal to ``1 - top_k_ratio``.
        For ``top_k_ratio=0.5``, 50% of FLOPs are saved.
        """
        return 1.0 - self.config.top_k_ratio

    def calibrate_per_layer(self, activations_list: List[ndarray]) -> Dict[str, float]:
        """Compute empirical sparsity statistics across a list of activations.

        Parameters
        ----------
        activations_list:
            List of activation arrays; each array is a batch or single
            sample at a particular layer.

        Returns
        -------
        Dictionary with keys:
        * ``"avg_sparsity"`` — average fraction of zeroed elements after
          sparsification across all provided arrays.
        * ``"min_sparsity"`` — minimum observed sparsity.
        * ``"max_sparsity"`` — maximum observed sparsity.
        """
        if not activations_list:
            raise ValueError("activations_list must not be empty")

        sparsities: List[float] = []
        for act in activations_list:
            x_sparse, _ = self.sparsify(np.asarray(act, dtype=np.float32))
            n_total = x_sparse.size
            n_zero = int((x_sparse == 0.0).sum())
            sparsities.append(n_zero / n_total if n_total > 0 else 0.0)

        return {
            "avg_sparsity": float(np.mean(sparsities)),
            "min_sparsity": float(np.min(sparsities)),
            "max_sparsity": float(np.max(sparsities)),
        }
