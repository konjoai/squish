"""squish/quant/slice_gpt.py

SliceGPT — PCA orthogonal column/row slicing of transformer weight matrices.

Reference
---------
Ashkboos et al. "SliceGPT: Compress Large Language Models by Deleting Rows
and Columns." ICLR 2024. arXiv:2401.15024.

Algorithm
---------
For each weight matrix W (out_features × in_features):

1. Collect input activations X over a small calibration set.
2. Compute the principal components Q of X via SVD / eigendecomposition of
   X^T X.
3. Rotate the weight into the principal-component basis: W' = W Q.
4. Slice: keep only the top-d columns (d = round(in_features × (1 − sparsity))).
5. The sliced weight W'[:, :d] replaces the original in the model.

For a pair of adjacent linear layers (W1, W2) in the same block:
  - W1 is column-sliced (output dimension reduced).
  - W2 is row-sliced (input dimension reduced to match).

This is an orthogonal compression: no quality-degradation term from the
dropped directions beyond the removed variance, making it fully composable
with post-hoc quantization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class SliceGPTConfig:
    """Configuration for SliceGPTPruner.

    Parameters
    ----------
    sparsity:
        Fraction of dimensions to remove (0.25 → 25% parameter reduction).
    n_calibration_samples:
        Number of random calibration vectors used to estimate X^T X.
    seed:
        RNG seed for reproducible calibration.
    """

    sparsity: float = 0.25
    n_calibration_samples: int = 128
    seed: int = 42

    def __post_init__(self) -> None:
        if not 0.0 < self.sparsity < 1.0:
            raise ValueError("sparsity must be in (0, 1)")


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class SliceGPTResult:
    """Output of a SliceGPT slice operation.

    Parameters
    ----------
    W_sliced:
        Compressed weight matrix — shape ``(out_features, d)`` where
        ``d = round(in_features × (1 − sparsity))``.
    Q:
        Orthogonal rotation matrix — shape ``(in_features, in_features)``.
        The first ``d`` columns correspond to retained principal components.
    d:
        Number of retained dimensions.
    original_shape:
        ``(out_features, in_features)`` of the original weight.
    """

    W_sliced: np.ndarray
    Q: np.ndarray
    d: int
    original_shape: tuple[int, int]

    @property
    def compression_ratio(self) -> float:
        """Fraction of parameters retained."""
        out_f, in_f = self.original_shape
        return float(out_f * self.d) / float(out_f * in_f)

    def reconstruct(self) -> np.ndarray:
        """Reconstruct an approximation of the original weight.

        Returns
        -------
        np.ndarray
            Shape ``(out_features, in_features)``.
        """
        # W_sliced = W @ Q[:, :d]  →  W_approx = W_sliced @ Q[:, :d].T
        return self.W_sliced @ self.Q[:, : self.d].T


# ---------------------------------------------------------------------------
# Pruner
# ---------------------------------------------------------------------------

class SliceGPTPruner:
    """PCA-based orthogonal column pruner.

    Parameters
    ----------
    config:
        SliceGPT configuration.
    """

    def __init__(self, config: Optional[SliceGPTConfig] = None) -> None:
        self._cfg = config or SliceGPTConfig()
        self._rng = np.random.default_rng(self._cfg.seed)

    @property
    def config(self) -> SliceGPTConfig:
        return self._cfg

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_rotation(self, activations: np.ndarray) -> np.ndarray:
        """Compute the orthogonal rotation matrix Q from calibration activations.

        Parameters
        ----------
        activations:
            Shape ``(n_samples, in_features)``.  If 3-D
            ``(batch, seq, features)`` it is automatically flattened.

        Returns
        -------
        np.ndarray
            Orthogonal matrix Q of shape ``(in_features, in_features)``.
        """
        X = np.asarray(activations, dtype=np.float32)
        if X.ndim == 3:
            X = X.reshape(-1, X.shape[-1])
        # Covariance proxy: X^T X  (row-wise mean-centering is common but
        # the paper skips it for rotational simplicity)
        cov = X.T @ X  # (in_f, in_f)
        # Eigendecomposition; eigenvalues sorted ascending by np.linalg.eigh
        eigvals, eigvecs = np.linalg.eigh(cov.astype(np.float64))
        # Sort descending (highest variance first)
        idx = np.argsort(eigvals)[::-1]
        Q = eigvecs[:, idx].astype(np.float32)
        return Q

    def slice_weight(
        self,
        W: np.ndarray,
        Q: Optional[np.ndarray] = None,
        activations: Optional[np.ndarray] = None,
    ) -> SliceGPTResult:
        """Slice a weight matrix using the PCA rotation.

        Exactly one of ``Q`` or ``activations`` must be provided.

        Parameters
        ----------
        W:
            Weight matrix of shape ``(out_features, in_features)``.
        Q:
            Pre-computed rotation matrix ``(in_features, in_features)``.
        activations:
            Raw calibration activations; Q will be computed on-the-fly.

        Returns
        -------
        SliceGPTResult
        """
        W = np.asarray(W, dtype=np.float32)
        out_f, in_f = W.shape

        if Q is None and activations is None:
            raise ValueError("Provide either Q or activations.")
        if Q is None:
            Q = self.compute_rotation(activations)

        d = max(1, round(in_f * (1.0 - self._cfg.sparsity)))

        # Rotate and slice
        W_rotated = W @ Q              # (out_f, in_f)
        W_sliced = W_rotated[:, :d]   # (out_f, d)

        return SliceGPTResult(
            W_sliced=W_sliced,
            Q=Q,
            d=d,
            original_shape=(out_f, in_f),
        )

    def calibrate_and_slice(
        self, W: np.ndarray, in_features: Optional[int] = None
    ) -> SliceGPTResult:
        """Convenience: generate synthetic calibration activations and slice.

        Parameters
        ----------
        W:
            Weight matrix ``(out_features, in_features)``.
        in_features:
            Override the detected ``in_features`` dimension (rarely needed).

        Returns
        -------
        SliceGPTResult
        """
        W = np.asarray(W, dtype=np.float32)
        n_in = in_features or W.shape[1]
        X = self._rng.standard_normal(
            (self._cfg.n_calibration_samples, n_in)
        ).astype(np.float32)
        return self.slice_weight(W, activations=X)

    def slice_pair(
        self,
        W1: np.ndarray,
        W2: np.ndarray,
        activations: Optional[np.ndarray] = None,
    ) -> tuple[SliceGPTResult, np.ndarray]:
        """Slice a sequential (W1, W2) pair consistently.

        W1 is column-sliced; W2 is row-sliced to match.

        Parameters
        ----------
        W1:
            First layer weight ``(out1, in1)``.
        W2:
            Second layer weight ``(out2, in1)`` — its input dimension must
            equal W1's output dimension after rotation.
        activations:
            Calibration activations for W1's input; synthetic if None.

        Returns
        -------
        tuple[SliceGPTResult, np.ndarray]
            (result_for_W1, W2_row_sliced)
        """
        W1 = np.asarray(W1, dtype=np.float32)
        W2 = np.asarray(W2, dtype=np.float32)

        if activations is None:
            activations = self._rng.standard_normal(
                (self._cfg.n_calibration_samples, W1.shape[1])
            ).astype(np.float32)

        result1 = self.slice_weight(W1, activations=activations)
        d = result1.d

        # W2 is in the same rotated basis on its input side; slice rows
        W2_rotated = (result1.Q.T @ W2.T).T  # (out2, in1) → rotate columns
        W2_sliced = W2_rotated[:, :d]         # (out2, d)

        return result1, W2_sliced
