"""squish/quant/wanda_pruner.py

WandaPruner — Activation×Magnitude unstructured weight pruning.

Reference
---------
Sun et al. "A Simple and Effective Pruning Approach for Large Language
Models." ICLR 2024. arXiv:2306.11695.

Algorithm
---------
For each weight matrix W (out_features × in_features) with corresponding
input activations X (n_samples × in_features):

  importance[i, j] = |W[i, j]| × ||X[:, j]||_2 / sqrt(n_samples)

Prune the fraction ``sparsity`` of weights with the lowest importance score
(set to zero). Optionally export in N:M structured-sparse format.

The method requires only one forward pass worth of activations and is
calibration-free (no gradient descent). It achieves 50% sparsity with < 2%
quality loss and is 5× faster than SparseGPT on the same hardware.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class WandaConfig:
    """Configuration for WandaPruner.

    Parameters
    ----------
    sparsity:
        Fraction of weights to prune globally (0.5 → 50% sparsity).
    structured_n:
        N in N:M structured sparsity (None = unstructured).
    structured_m:
        M in N:M structured sparsity (None = unstructured).
    n_calibration_samples:
        Number of calibration samples used to estimate activation norms.
    seed:
        RNG seed for synthetic calibration if no activations are provided.
    """

    sparsity: float = 0.5
    structured_n: Optional[int] = None
    structured_m: Optional[int] = None
    n_calibration_samples: int = 128
    seed: int = 0

    def __post_init__(self) -> None:
        if not 0.0 <= self.sparsity < 1.0:
            raise ValueError("sparsity must be in [0, 1)")
        if (self.structured_n is None) != (self.structured_m is None):
            raise ValueError("structured_n and structured_m must both be set or both be None")
        if self.structured_n is not None and self.structured_n >= self.structured_m:  # type: ignore[operator]
            raise ValueError("structured_n must be < structured_m")


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class WandaResult:
    """Result of a Wanda pruning operation.

    Parameters
    ----------
    W_pruned:
        Weight matrix with pruned entries zeroed out.
        Shape ``(out_features, in_features)``.
    mask:
        Boolean mask; ``True`` where weight is retained.
    importance:
        Per-element importance scores.
    sparsity_achieved:
        Actual fraction of zero weights after pruning.
    """

    W_pruned: np.ndarray
    mask: np.ndarray
    importance: np.ndarray
    sparsity_achieved: float

    def nnz(self) -> int:
        """Number of non-zero weights."""
        return int(self.mask.sum())

    def apply(self, x: np.ndarray) -> np.ndarray:
        """Apply sparse weight to an input vector/matrix.

        Parameters
        ----------
        x:
            Shape ``(in_features,)`` or ``(batch, in_features)``.

        Returns
        -------
        np.ndarray
            Shape ``(out_features,)`` or ``(batch, out_features)``.
        """
        return (x @ self.W_pruned.T).astype(np.float32)


# ---------------------------------------------------------------------------
# Pruner
# ---------------------------------------------------------------------------

class WandaPruner:
    """Wanda importance-guided weight pruner.

    Parameters
    ----------
    config:
        Wanda configuration.
    """

    def __init__(self, config: Optional[WandaConfig] = None) -> None:
        self._cfg = config or WandaConfig()
        self._rng = np.random.default_rng(self._cfg.seed)

    @property
    def config(self) -> WandaConfig:
        return self._cfg

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _activation_rms(self, X: np.ndarray) -> np.ndarray:
        """Compute per-column RMS of activation matrix X.

        Parameters
        ----------
        X:
            Shape ``(n_samples, in_features)``.

        Returns
        -------
        np.ndarray
            Shape ``(in_features,)``.
        """
        return np.sqrt((X ** 2).mean(axis=0)).astype(np.float32)

    def _compute_importance(
        self, W: np.ndarray, activation_rms: np.ndarray
    ) -> np.ndarray:
        """Importance = |W| × activation_rms (broadcast over rows)."""
        return np.abs(W) * activation_rms[np.newaxis, :]

    def _unstructured_mask(self, importance: np.ndarray) -> np.ndarray:
        """Global unstructured pruning mask — retain top (1-sparsity) fraction."""
        threshold = np.quantile(importance, self._cfg.sparsity)
        return importance > threshold

    def _nm_mask(self, importance: np.ndarray) -> np.ndarray:
        """N:M structured sparsity mask — applied column-block-wise."""
        n, m = self._cfg.structured_n, self._cfg.structured_m  # type: ignore[misc]
        out_f, in_f = importance.shape
        mask = np.zeros_like(importance, dtype=bool)
        for col_start in range(0, in_f, m):
            col_end = min(col_start + m, in_f)
            block = importance[:, col_start:col_end]
            # For each row, keep top-n within the block
            idx = np.argsort(block, axis=1)[:, -(col_end - col_start - n):]
            # Mark retained entries
            for row in range(out_f):
                mask[row, col_start:col_end][idx[row]] = True
        return mask

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def prune(
        self,
        W: np.ndarray,
        activations: Optional[np.ndarray] = None,
    ) -> WandaResult:
        """Prune weight matrix W using Wanda importance scores.

        Parameters
        ----------
        W:
            Weight matrix ``(out_features, in_features)``.
        activations:
            Calibration activations ``(n_samples, in_features)``.
            If None, synthetic Gaussian samples are used.

        Returns
        -------
        WandaResult
        """
        W = np.asarray(W, dtype=np.float32)
        out_f, in_f = W.shape

        if activations is None:
            activations = self._rng.standard_normal(
                (self._cfg.n_calibration_samples, in_f)
            ).astype(np.float32)

        X = np.asarray(activations, dtype=np.float32)
        if X.ndim == 3:
            X = X.reshape(-1, X.shape[-1])

        act_rms = self._activation_rms(X)
        importance = self._compute_importance(W, act_rms)

        if self._cfg.structured_n is not None:
            mask = self._nm_mask(importance)
        else:
            mask = self._unstructured_mask(importance)

        W_pruned = W * mask

        sparsity_achieved = 1.0 - float(mask.sum()) / float(mask.size)

        return WandaResult(
            W_pruned=W_pruned,
            mask=mask,
            importance=importance,
            sparsity_achieved=sparsity_achieved,
        )

    def prune_layer(
        self,
        W: np.ndarray,
        bias: Optional[np.ndarray] = None,
        activations: Optional[np.ndarray] = None,
    ) -> tuple[WandaResult, Optional[np.ndarray]]:
        """Prune a linear layer weight (and optionally its bias).

        Bias is returned unchanged; only W is pruned.
        """
        result = self.prune(W, activations)
        return result, bias
