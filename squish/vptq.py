"""
squish/vptq.py

VPTQ — Vector Post-Training Quantization for LLM Weight Compression.

Inspired by:
  "VPTQ: Extreme Low-Bit Vector Post-Training Quantization for Large Language Models"
  arXiv:2409.17066 (NeurIPS 2025 Spotlight, Microsoft Research)

Background
----------
Classical scalar quantization (INT4, INT3) quantizes each weight independently.
Vector quantization (VQ) groups ``group_size`` contiguous weights into a vector
and replaces the whole vector with a single codebook index.

VPTQ extends basic VQ with:
  1. **Primary codebook**: large (e.g. 4096-entry, 12-bit code) for coarse
     representation of weight vectors.
  2. **Residual codebook**: small (e.g. 256-entry, 8-bit code) that quantizes
     the approximation error from the primary stage.
  3. **Column-wise scaling**: per-column L2 normalisation before VQ improves
     codebook utilisation.

Typical result at 2-bit average bits-per-weight:
  • Primary code: 12 bits / group_size, e.g. 12/8 = 1.5 bpw
  • Residual code:  8 bits / group_size, e.g.  8/8 = 1.0 bpw
  • Total overhead (scales): ~0.5 bpw
  → ~3 bpw total with quality approaching scalar INT4

This module provides:
  ``VPTQConfig``      — all hyperparameters
  ``VPTQCodebook``    — fit, encode, decode a set of vectors
  ``VPTQLayer``       — compressed weight layer (primary + residual indices)
  ``VPTQQuantizer``   — compress / decompress full weight matrices

Usage::

    from squish.vptq import VPTQConfig, VPTQQuantizer

    quant = VPTQQuantizer(VPTQConfig(n_codebook_entries=256, group_size=8))
    layer = quant.compress(weight_matrix)             # float32 (out, in)

    approx = quant.decompress(layer)                  # float32 (out, in)
    error  = np.mean((weight_matrix - approx) ** 2)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "VPTQConfig",
    "VPTQCodebook",
    "VPTQLayer",
    "VPTQQuantizer",
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class VPTQConfig:
    """
    VPTQ compression configuration.

    Parameters
    ----------
    n_codebook_entries : int
        Number of codewords in the primary codebook.
        Powers of 2 are most efficient (bits = log2(n)).
        Default 256 (8-bit codes).
    group_size : int
        Number of contiguous weights per vector (sub-vector length).
        Default 8.
    n_residual_entries : int
        Number of codewords in the residual codebook.
        0 = no residual stage.  Default 16.
    n_fit_iters : int
        k-means iterations for codebook fitting.  Default 20.
    seed : int
        Random seed for reproducible k-means initialisation.  Default 42.
    """
    n_codebook_entries: int = 256
    group_size:         int = 8
    n_residual_entries: int = 16
    n_fit_iters:        int = 20
    seed:               int = 42

    def __post_init__(self) -> None:
        if self.n_codebook_entries < 2:
            raise ValueError(
                f"n_codebook_entries must be ≥ 2, got {self.n_codebook_entries}"
            )
        if self.group_size < 1:
            raise ValueError(f"group_size must be ≥ 1, got {self.group_size}")
        if self.n_residual_entries < 0:
            raise ValueError(
                f"n_residual_entries must be ≥ 0, got {self.n_residual_entries}"
            )
        if self.n_fit_iters < 1:
            raise ValueError(f"n_fit_iters must be ≥ 1, got {self.n_fit_iters}")


# ---------------------------------------------------------------------------
# k-means helper
# ---------------------------------------------------------------------------

def _kmeans(
    X: np.ndarray,
    n_clusters: int,
    n_iters: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simple k-means with k-means++ initialisation.

    Parameters
    ----------
    X          : np.ndarray  float32  (n_samples, dim)
    n_clusters : int
    n_iters    : int
    seed       : int

    Returns
    -------
    centroids : np.ndarray  float32  (n_clusters, dim)
    labels    : np.ndarray  int64    (n_samples,)
    """
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    k = min(n_clusters, n)

    # k-means++ init
    centres = [X[rng.integers(n)]]
    for _ in range(k - 1):
        dists = np.array([
            min(np.sum((x - c) ** 2) for c in centres)
            for x in X
        ])
        probs = dists.astype(np.float64)
        total = probs.sum()
        if total <= 0:
            probs = np.ones(n, dtype=np.float64) / n
        else:
            probs /= total
            probs /= probs.sum()  # ensure exact 1.0 after fp rounding
        idx = rng.choice(n, p=probs)
        centres.append(X[idx])
    centroids = np.stack(centres, axis=0).astype(np.float32)

    labels = np.zeros(n, dtype=np.int64)
    for _ in range(n_iters):
        # Assignment
        diffs  = X[:, None, :] - centroids[None, :, :]   # (n, k, dim)
        dists2 = np.sum(diffs ** 2, axis=-1)              # (n, k)
        labels = np.argmin(dists2, axis=-1)

        # Update
        new_centroids = np.zeros_like(centroids)
        counts = np.bincount(labels, minlength=k)
        for c in range(k):
            if counts[c] > 0:
                new_centroids[c] = X[labels == c].mean(axis=0)
            else:
                new_centroids[c] = centroids[c]
        centroids = new_centroids

    return centroids, labels


# ---------------------------------------------------------------------------
# VPTQCodebook
# ---------------------------------------------------------------------------

class VPTQCodebook:
    """
    A single-stage VQ codebook for weight vectors of length ``group_size``.

    Parameters
    ----------
    group_size         : int
    n_codebook_entries : int
    n_fit_iters        : int
    seed               : int
    """

    def __init__(
        self,
        group_size:         int,
        n_codebook_entries: int,
        n_fit_iters:        int = 20,
        seed:               int = 42,
    ) -> None:
        self.group_size         = group_size
        self.n_codebook_entries = n_codebook_entries
        self.n_fit_iters        = n_fit_iters
        self.seed               = seed
        self._centroids: np.ndarray | None = None  # (n_entries, group_size)

    @property
    def is_fitted(self) -> bool:
        return self._centroids is not None

    # ------------------------------------------------------------------

    def fit(self, vectors: np.ndarray) -> None:
        """
        Fit the codebook to a set of weight vectors.

        Parameters
        ----------
        vectors : np.ndarray  float32  shape (n, group_size)
        """
        X = np.asarray(vectors, dtype=np.float32)
        centroids, _ = _kmeans(X, self.n_codebook_entries, self.n_fit_iters, self.seed)
        self._centroids = centroids

    def encode(self, vectors: np.ndarray) -> np.ndarray:
        """
        Assign each vector to its nearest centroid.

        Parameters
        ----------
        vectors : np.ndarray  float32  shape (n, group_size)

        Returns
        -------
        np.ndarray  int64  shape (n,)  — centroid indices
        """
        if self._centroids is None:
            raise RuntimeError("Call fit() before encode()")
        X = np.asarray(vectors, dtype=np.float32)
        diffs  = X[:, None, :] - self._centroids[None, :, :]  # (n, k, d)
        dists2 = np.sum(diffs ** 2, axis=-1)                   # (n, k)
        return np.argmin(dists2, axis=-1).astype(np.int64)

    def decode(self, indices: np.ndarray) -> np.ndarray:
        """
        Reconstruct vectors from centroid indices.

        Parameters
        ----------
        indices : np.ndarray  int64  shape (n,)

        Returns
        -------
        np.ndarray  float32  shape (n, group_size)
        """
        if self._centroids is None:
            raise RuntimeError("Call fit() before decode()")
        return self._centroids[indices].astype(np.float32)

    @property
    def centroids(self) -> np.ndarray:
        """The codebook centroids array (n_entries, group_size)."""
        if self._centroids is None:
            raise RuntimeError("Codebook not fitted")
        return self._centroids


# ---------------------------------------------------------------------------
# VPTQLayer
# ---------------------------------------------------------------------------

@dataclass
class VPTQLayer:
    """
    Compressed weight layer produced by ``VPTQQuantizer.compress``.

    Attributes
    ----------
    primary_indices  : np.ndarray  int64  shape (n_groups,)
        Primary codebook assignment for each weight group.
    residual_indices : np.ndarray  int64  shape (n_groups,) or None
        Residual codebook assignment (None if residual disabled).
    primary_cb       : VPTQCodebook
    residual_cb      : VPTQCodebook or None
    original_shape   : tuple
    col_scales       : np.ndarray  float32  shape (n_cols,) or None
        Per-column L2 scale used during compression (applied during decompress).
    """
    primary_indices:  np.ndarray
    residual_indices: np.ndarray | None
    primary_cb:       VPTQCodebook
    residual_cb:      VPTQCodebook | None
    original_shape:   tuple
    col_scales:       np.ndarray | None = None

    # ------------------------------------------------------------------

    @property
    def n_groups(self) -> int:
        return len(self.primary_indices)

    @property
    def compressed_bits(self) -> int:
        """Approximate total bit count of the compressed indices."""
        import math
        bits_primary  = math.ceil(math.log2(max(self.primary_cb.n_codebook_entries, 2)))
        bits_residual = 0
        if self.residual_cb is not None:
            bits_residual = math.ceil(math.log2(max(self.residual_cb.n_codebook_entries, 2)))
        return self.n_groups * (bits_primary + bits_residual)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Compute ``x @ W^T`` using the compressed weight approximation.

        Parameters
        ----------
        x : np.ndarray  float32  shape (batch, in_features)

        Returns
        -------
        np.ndarray  float32  shape (batch, out_features)
        """
        W = decompress_layer(self)  # (out_features, in_features)
        return np.asarray(x, dtype=np.float32) @ W.T


def decompress_layer(layer: VPTQLayer) -> np.ndarray:
    """
    Reconstruct the approximate weight matrix from a ``VPTQLayer``.

    Returns
    -------
    np.ndarray  float32  shape = layer.original_shape
    """
    # Primary reconstruction
    approx = layer.primary_cb.decode(layer.primary_indices)  # (n_groups, group_size)

    # Residual correction
    if layer.residual_cb is not None and layer.residual_indices is not None:
        residual = layer.residual_cb.decode(layer.residual_indices)
        approx = approx + residual

    # Flatten and reshape
    flat = approx.ravel()
    n_orig = 1
    for s in layer.original_shape:
        n_orig *= s
    # Trim / pad to original size
    if flat.size >= n_orig:
        flat = flat[:n_orig]
    else:
        flat = np.pad(flat, (0, n_orig - flat.size))

    W = flat.reshape(layer.original_shape).astype(np.float32)

    # Undo column scaling
    if layer.col_scales is not None:
        W = W * layer.col_scales[None, :]  # (out, in) * (in,)

    return W


# ---------------------------------------------------------------------------
# VPTQQuantizer
# ---------------------------------------------------------------------------

class VPTQQuantizer:
    """
    End-to-end VQ compressor/decompressor for weight matrices.

    Parameters
    ----------
    config : VPTQConfig
    """

    def __init__(self, config: VPTQConfig | None = None) -> None:
        self.config = config or VPTQConfig()

    # ------------------------------------------------------------------

    def _make_groups(
        self,
        weight: np.ndarray,
        gs: int,
    ) -> tuple[np.ndarray, int]:
        """
        Flatten a weight matrix and split into (group_size,) vectors.

        Pads with zeros if needed.

        Returns
        -------
        groups : np.ndarray  float32  shape (n_groups, gs)
        n_orig : int  original flat size (for trim on decompress)
        """
        flat   = weight.ravel().astype(np.float32)
        n_orig = flat.size
        pad    = (gs - n_orig % gs) % gs
        if pad:
            flat = np.pad(flat, (0, pad))
        return flat.reshape(-1, gs), n_orig

    def compress(self, weight: np.ndarray) -> VPTQLayer:
        """
        Compress a weight matrix (e.g. linear layer weights).

        Parameters
        ----------
        weight : np.ndarray  float32  shape (out_features, in_features)

        Returns
        -------
        VPTQLayer
        """
        cfg = self.config
        W   = np.asarray(weight, dtype=np.float32)

        # Column-wise L2 normalisation
        col_scales = np.linalg.norm(W, axis=0) + 1e-8  # (in_features,)
        W_norm = W / col_scales[None, :]

        gs     = cfg.group_size
        groups, _ = self._make_groups(W_norm, gs)

        # Primary codebook
        primary_cb = VPTQCodebook(
            gs, cfg.n_codebook_entries, cfg.n_fit_iters, cfg.seed
        )
        primary_cb.fit(groups)
        primary_idx = primary_cb.encode(groups)

        # Residual stage
        residual_cb:  VPTQCodebook | None  = None
        residual_idx: np.ndarray | None    = None
        if cfg.n_residual_entries > 0:
            residuals = groups - primary_cb.decode(primary_idx)  # (n_groups, gs)
            residual_cb = VPTQCodebook(
                gs, cfg.n_residual_entries, cfg.n_fit_iters, cfg.seed + 1
            )
            residual_cb.fit(residuals)
            residual_idx = residual_cb.encode(residuals)

        return VPTQLayer(
            primary_indices  = primary_idx,
            residual_indices = residual_idx,
            primary_cb       = primary_cb,
            residual_cb      = residual_cb,
            original_shape   = W_norm.shape,
            col_scales       = col_scales,
        )

    def decompress(self, layer: VPTQLayer) -> np.ndarray:
        """
        Decompress a ``VPTQLayer`` back to a float32 weight matrix.

        Parameters
        ----------
        layer : VPTQLayer

        Returns
        -------
        np.ndarray  float32  shape = layer.original_shape
        """
        return decompress_layer(layer)
