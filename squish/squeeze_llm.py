"""
squish/squeeze_llm.py

SqueezeLLM — Dense-and-Sparse Quantization for LLM Weight Compression.

Inspired by:
  "SqueezeLLM: Dense-and-Sparse Quantization"
  arXiv:2306.07629  (ICML 2024, UC Berkeley)

Background
----------
Weight quantization (INT4, INT3) suffers disproportionately from outlier
weights — a small fraction of weights with atypically large magnitude that
contribute most of the quantization error.

SqueezeLLM decomposes each weight matrix into two parts:

  1. **Dense bulk** (≈99.55% of weights): quantized to INT3 (or INT4) using
     non-uniform bin boundaries (k-means on the value distribution).

  2. **Sparse outliers** (≈0.45% of weights): stored in FP16 as a sparse
     COO (coordinate) representation — dictionary keyed by ``(row, col)``.

At inference time: reconstruct the dense matrix (via lookup table) and add
back the sparse outliers.  Result: quality between INT4 and FP16 at 3 bits
average bits-per-weight.

Outlier selection strategies
-----------------------------
* **Magnitude** (default, simple): top-``sparsity_ratio`` fraction of weights
  by absolute value.
* **Sensitivity** (optional): weights whose removal would cause the largest
  reconstruction error (approximated by Fisher information; requires
  calibration data — passed separately).

This module provides:
  ``SqueezeLLMConfig``     — all hyperparameters
  ``OutlierDetector``      — identify the sparse outlier set
  ``SqueezeLLMLayer``      — compressed layer (INT3 dense + sparse COO)
  ``SqueezeLLMQuantizer``  — compress / decompress weight matrices

Usage::

    from squish.squeeze_llm import SqueezeLLMConfig, SqueezeLLMQuantizer

    quant = SqueezeLLMQuantizer(SqueezeLLMConfig(quant_bits=3, sparsity_ratio=0.0045))

    layer = quant.compress(weight_matrix)           # float32 (out, in)
    print(f"sparse outliers: {layer.n_outliers}")

    approx = quant.decompress(layer)                # float32 (out, in)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "SqueezeLLMConfig",
    "OutlierDetector",
    "SqueezeLLMLayer",
    "SqueezeLLMQuantizer",
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SqueezeLLMConfig:
    """
    SqueezeLLM compression configuration.

    Parameters
    ----------
    quant_bits : int
        Bits used for the dense (bulk) quantization.  Supports 2, 3, 4.
        Default 3.
    sparsity_ratio : float
        Fraction of weights designated as outliers and stored in FP16.
        Default 0.0045 (0.45%) — the value from the original paper.
    group_size : int
        Number of weights per quantization group (share one set of bin
        boundaries).  Default 128.
    n_fit_iters : int
        k-means iterations for computing non-uniform bin boundaries.
        Default 20.
    seed : int
        Random seed.  Default 42.
    """
    quant_bits:     int   = 3
    sparsity_ratio: float = 0.0045
    group_size:     int   = 128
    n_fit_iters:    int   = 20
    seed:           int   = 42

    def __post_init__(self) -> None:
        if self.quant_bits not in (2, 3, 4):
            raise ValueError(
                f"quant_bits must be 2, 3, or 4, got {self.quant_bits}"
            )
        if not (0.0 <= self.sparsity_ratio < 1.0):
            raise ValueError(
                f"sparsity_ratio must be in [0, 1), got {self.sparsity_ratio}"
            )
        if self.group_size < 1:
            raise ValueError(f"group_size must be ≥ 1, got {self.group_size}")
        if self.n_fit_iters < 1:
            raise ValueError(f"n_fit_iters must be ≥ 1, got {self.n_fit_iters}")


# ---------------------------------------------------------------------------
# Outlier detector
# ---------------------------------------------------------------------------

class OutlierDetector:
    """
    Identify the sparse outlier set in a weight matrix.

    Uses a **magnitude** criterion: the ``sparsity_ratio`` fraction of weights
    with the largest absolute value are selected as outliers.

    Parameters
    ----------
    sparsity_ratio : float
    """

    def __init__(self, sparsity_ratio: float = 0.0045) -> None:
        if not (0.0 <= sparsity_ratio < 1.0):
            raise ValueError(
                f"sparsity_ratio must be in [0, 1), got {sparsity_ratio}"
            )
        self.sparsity_ratio = sparsity_ratio

    def identify(
        self,
        weight: np.ndarray,
    ) -> tuple[np.ndarray, dict[tuple[int, int], float]]:
        """
        Identify outlier weights.

        Parameters
        ----------
        weight : np.ndarray  float32  shape (out_features, in_features)

        Returns
        -------
        dense_weight : np.ndarray  float32  same shape — outliers zeroed out
        outliers     : dict  {(row, col): float32 value}
        """
        W = np.asarray(weight, dtype=np.float32)
        if self.sparsity_ratio == 0.0:
            return W.copy(), {}

        n_total   = W.size
        n_outlier = max(1, int(round(n_total * self.sparsity_ratio)))
        abs_flat  = np.abs(W).ravel()
        threshold_idx = np.argpartition(abs_flat, -n_outlier)[-n_outlier:]
        outlier_mask  = np.zeros(n_total, dtype=bool)
        outlier_mask[threshold_idx] = True
        outlier_mask  = outlier_mask.reshape(W.shape)

        dense = W.copy()
        dense[outlier_mask] = 0.0

        rows, cols = np.where(outlier_mask)
        outliers: dict[tuple[int, int], float] = {
            (int(r), int(c)): float(W[r, c])
            for r, c in zip(rows, cols, strict=False)
        }
        return dense, outliers


# ---------------------------------------------------------------------------
# Non-uniform (k-means) quantizer helper
# ---------------------------------------------------------------------------

def _nonuniform_quantize(
    values: np.ndarray,
    n_bins: int,
    n_iters: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Quantize 1-D float32 values using k-means to find non-uniform bin centres.

    Returns
    -------
    indices  : np.ndarray  uint8  shape (n,)      — bin index per value
    centres  : np.ndarray  float32 shape (n_bins,) — bin centre values
    """
    rng = np.random.default_rng(seed)
    flat = values.ravel().astype(np.float32)
    k = min(n_bins, flat.size)

    # k-means++ init (1-D specialised)
    init_idx = [int(rng.integers(len(flat)))]
    for _ in range(k - 1):
        dists = np.array([
            min((flat[j] - flat[i]) ** 2 for i in init_idx)
            for j in range(len(flat))
        ])
        probs = dists.astype(np.float64)
        total = probs.sum()
        if total <= 0:
            probs = np.ones(len(flat), dtype=np.float64) / len(flat)
        else:
            probs /= total
            probs /= probs.sum()  # ensure exact 1.0 after fp rounding
        init_idx.append(int(rng.choice(len(flat), p=probs)))
    centres = flat[init_idx]

    for _ in range(n_iters):
        dists2 = (flat[:, None] - centres[None, :]) ** 2  # (n, k)
        labels = np.argmin(dists2, axis=-1)
        for c in range(k):
            mask = labels == c
            if mask.any():
                centres[c] = flat[mask].mean()

    # Final assignment
    dists2  = (flat[:, None] - centres[None, :]) ** 2
    indices = np.argmin(dists2, axis=-1).astype(np.uint8)
    return indices, centres


# ---------------------------------------------------------------------------
# SqueezeLLMLayer
# ---------------------------------------------------------------------------

@dataclass
class SqueezeLLMLayer:
    """
    Compressed representation of a single weight matrix.

    Attributes
    ----------
    quant_indices : np.ndarray  uint8  shape (n_groups * group_size,) or (out, in)
        INT3/INT4 quantization indices for the dense part.
    bin_centres   : np.ndarray  float32  shape (n_groups, n_bins)
        Non-uniform bin centres (one set per group).
    outliers      : dict {(row, col): float32}
        Sparse FP16 outlier values.
    original_shape : tuple  (out_features, in_features)
    group_size    : int
    """
    quant_indices:  np.ndarray
    bin_centres:    np.ndarray                    # (n_groups, n_bins) float32
    outliers:       dict[tuple[int, int], float]
    original_shape: tuple[int, ...]
    group_size:     int

    # ------------------------------------------------------------------

    @property
    def n_outliers(self) -> int:
        return len(self.outliers)

    @property
    def sparsity(self) -> float:
        """Actual outlier fraction relative to total weight count."""
        n_total = 1
        for s in self.original_shape:
            n_total *= s
        return self.n_outliers / max(n_total, 1)

    @property
    def n_groups(self) -> int:
        return len(self.bin_centres)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Compute ``x @ W^T`` using the dense + sparse decomposition.

        Parameters
        ----------
        x : np.ndarray  float32  shape (batch, in_features)

        Returns
        -------
        np.ndarray  float32  shape (batch, out_features)
        """
        W = decompress_layer_sq(self)  # (out_features, in_features)
        return np.asarray(x, dtype=np.float32) @ W.T


def decompress_layer_sq(layer: SqueezeLLMLayer) -> np.ndarray:
    """
    Reconstruct the approximate weight matrix from a ``SqueezeLLMLayer``.

    Returns
    -------
    np.ndarray  float32  shape = layer.original_shape
    """
    flat_indices = layer.quant_indices.ravel()
    n_orig = 1
    for s in layer.original_shape:
        n_orig *= s
    gs   = layer.group_size
    n_grp = layer.n_groups

    # Lookup per group
    flat_out = np.empty(n_orig, dtype=np.float32)
    for g in range(n_grp):
        start = g * gs
        end   = min(start + gs, n_orig)
        idxs  = flat_indices[start:end].astype(np.int64)
        caps  = layer.bin_centres[g]
        flat_out[start:end] = caps[idxs]

    W = flat_out.reshape(layer.original_shape)

    # Add back sparse outliers
    for (r, c), val in layer.outliers.items():
        W[r, c] += val

    return W


# ---------------------------------------------------------------------------
# SqueezeLLMQuantizer
# ---------------------------------------------------------------------------

class SqueezeLLMQuantizer:
    """
    Compress and decompress weight matrices using SqueezeLLM.

    Parameters
    ----------
    config : SqueezeLLMConfig
    """

    def __init__(self, config: SqueezeLLMConfig | None = None) -> None:
        self.config   = config or SqueezeLLMConfig()
        self._detector = OutlierDetector(self.config.sparsity_ratio)

    # ------------------------------------------------------------------

    def compress(self, weight: np.ndarray) -> SqueezeLLMLayer:
        """
        Compress a weight matrix.

        Parameters
        ----------
        weight : np.ndarray  float32  shape (out_features, in_features)

        Returns
        -------
        SqueezeLLMLayer
        """
        cfg  = self.config
        W    = np.asarray(weight, dtype=np.float32)
        if W.ndim == 1:
            W = W.reshape(1, -1)

        # Step 1: extract outliers
        dense_W, outliers = self._detector.identify(W)

        # Step 2: group-wise non-uniform quantization of dense part
        flat   = dense_W.ravel()
        n_orig = flat.size
        gs     = cfg.group_size
        pad    = (gs - n_orig % gs) % gs
        if pad:
            flat_padded = np.pad(flat, (0, pad))
        else:
            flat_padded = flat

        n_bins  = 2 ** cfg.quant_bits
        n_groups = len(flat_padded) // gs

        all_indices: list = []
        all_centres: list = []
        for g in range(n_groups):
            chunk   = flat_padded[g * gs:(g + 1) * gs]
            idxs, centres = _nonuniform_quantize(
                chunk, n_bins, cfg.n_fit_iters, cfg.seed + g
            )
            all_indices.append(idxs)
            all_centres.append(centres)

        quant_indices = np.concatenate(all_indices, axis=0)[:n_orig]  # trim pad
        bin_centres   = np.stack(all_centres, axis=0)                 # (n_groups, n_bins)

        return SqueezeLLMLayer(
            quant_indices  = quant_indices,
            bin_centres    = bin_centres,
            outliers       = outliers,
            original_shape = W.shape,
            group_size     = gs,
        )

    def decompress(self, layer: SqueezeLLMLayer) -> np.ndarray:
        """
        Decompress a ``SqueezeLLMLayer`` back to a float32 weight matrix.

        Parameters
        ----------
        layer : SqueezeLLMLayer

        Returns
        -------
        np.ndarray  float32  shape = layer.original_shape
        """
        return decompress_layer_sq(layer)
