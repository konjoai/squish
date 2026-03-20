"""
squish/quant/flute_quant.py

FluteQuantizer: Flexible Lookup-Table Weight-Only Quantization.

Reference
---------
Guo et al. "FLUTE: Flexible Lookup Table for Efficient Weight-Only
Quantization." ICLR 2025.

Algorithm
---------
FLUTE replaces the standard INT4 weight-dequantisation → FP16 GEMM pipeline
with a lookup-table GEMM that avoids the dequantisation step entirely:

  1. During quantisation, each weight value is mapped to an index into a
     codebook of K = 2^bits candidate values (default 16 for INT4).
  2. The codebook is typically *non-uniform* (NF4-style) and is chosen to
     minimise quantisation error.
  3. At inference, the GEMM kernel looks up activation values for each weight
     index directly from the codebook, bypassing FP16 dequant.

This module implements:
  * ``FluteQuantizer.quantise(weight)`` — map FP32/FP16 weights to int indices
    and build a codebook via k-means (LBG algorithm).
  * ``FluteQuantizer.dequantise(codes, codebook)`` — expand codes back to FP32
    (used for verification / NumPy simulation path).
  * ``FluteQuantizer.lut_gemm(x, codes, codebook)`` — simulate the LUT-GEMM
    forward pass (gather + matmul in FP32).

Key properties
--------------
* ``bits`` — quantisation bits, 2–8 (default 4; 3 is also supported).
* ``group_size`` — number of weight columns sharing a codebook (default 128).
* ``kmeans_iters`` — Lloyd iterations for codebook construction (default 50).
* NumPy-only; no external dependencies.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np


@dataclass
class FluteConfig:
    """Configuration for FluteQuantizer."""

    bits: int = 4
    """Quantisation bits.  Supported: 2, 3, 4, 8."""

    group_size: int = 128
    """Columns per codebook group."""

    kmeans_iters: int = 50
    """Lloyd's k-means iterations for codebook generation."""

    symmetric: bool = False
    """If True use symmetric codebook (zero centred)."""

    def __post_init__(self) -> None:
        if self.bits not in (2, 3, 4, 8):
            raise ValueError(f"bits must be one of 2, 3, 4, 8; got {self.bits}")
        if self.group_size < 1:
            raise ValueError("group_size must be >= 1")
        if self.kmeans_iters < 1:
            raise ValueError("kmeans_iters must be >= 1")

    @property
    def codebook_size(self) -> int:
        """Number of codebook entries = 2^bits."""
        return 2 ** self.bits


@dataclass
class FluteStats:
    """Runtime counters for FluteQuantizer."""

    quantise_calls: int = 0
    dequantise_calls: int = 0
    lut_gemm_calls: int = 0


class FluteQuantizer:
    """LUT-based weight-only quantizer (NumPy simulation).

    Usage
    -----
    ::

        quant = FluteQuantizer()
        codes, codebook = quant.quantise(weight_matrix)
        output = quant.lut_gemm(x, codes, codebook)
    """

    def __init__(self, config: Optional[FluteConfig] = None) -> None:
        self.config = config or FluteConfig()
        self.stats = FluteStats()

    # ------------------------------------------------------------------
    # Codebook construction (Lloyd's k-means over 1-D values)
    # ------------------------------------------------------------------

    def _build_codebook(self, values: np.ndarray) -> np.ndarray:
        """Build a 1-D codebook from a flat array of weight values.

        Parameters
        ----------
        values:
            Flat float32 array of weight values.

        Returns
        -------
        codebook:
            Shape ``(k,)`` float32 centroid values.
        """
        k = self.config.codebook_size
        rng = np.random.default_rng(seed=0)
        indices = rng.choice(len(values), size=min(k, len(values)), replace=False)
        centroids = values[indices].astype(np.float32)
        # Pad if needed
        if len(centroids) < k:
            extra = np.linspace(values.min(), values.max(), k - len(centroids))
            centroids = np.concatenate([centroids, extra.astype(np.float32)])
        centroids.sort()

        for _ in range(self.config.kmeans_iters):
            # Assign each value to nearest centroid
            dists = np.abs(values[:, None] - centroids[None, :])  # (n, k)
            labels = dists.argmin(axis=1)
            # Update centroids
            new_centroids = np.empty(k, dtype=np.float32)
            for ci in range(k):
                members = values[labels == ci]
                new_centroids[ci] = members.mean() if len(members) > 0 else centroids[ci]
            if np.allclose(centroids, new_centroids, atol=1e-6):
                break
            centroids = new_centroids

        if self.config.symmetric:
            centroids = (centroids - centroids.mean())

        return centroids

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def quantise(
        self, weight: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Quantise a 2-D weight matrix using group-wise LUT coding.

        Parameters
        ----------
        weight:
            Shape ``(rows, cols)`` FP32 weight matrix.

        Returns
        -------
        codes:
            Shape ``(rows, cols)`` uint8 index matrix.
        codebook:
            Shape ``(n_groups, codebook_size)`` float32 centroids.
        """
        self.stats.quantise_calls += 1
        weight = weight.astype(np.float32)
        rows, cols = weight.shape
        gs = min(self.config.group_size, cols)
        n_groups = math.ceil(cols / gs)
        k = self.config.codebook_size

        codes = np.empty((rows, cols), dtype=np.uint8)
        codebook = np.empty((n_groups, k), dtype=np.float32)

        for g in range(n_groups):
            col_start = g * gs
            col_end = min(col_start + gs, cols)
            group = weight[:, col_start:col_end]  # (rows, gs_actual)
            flat = group.ravel()
            cb = self._build_codebook(flat)
            codebook[g] = cb

            # Encode: nearest centroid index
            dists = np.abs(flat[:, None] - cb[None, :])  # (n, k)
            group_codes = dists.argmin(axis=1).astype(np.uint8).reshape(rows, -1)
            codes[:, col_start:col_end] = group_codes

        return codes, codebook

    def dequantise(
        self, codes: np.ndarray, codebook: np.ndarray
    ) -> np.ndarray:
        """Expand LUT codes back to FP32 weights.

        Parameters
        ----------
        codes:
            Shape ``(rows, cols)`` uint8.
        codebook:
            Shape ``(n_groups, codebook_size)`` float32.

        Returns
        -------
        weight_approx:
            Shape ``(rows, cols)`` float32.
        """
        self.stats.dequantise_calls += 1
        rows, cols = codes.shape
        n_groups = codebook.shape[0]
        gs = math.ceil(cols / n_groups)
        out = np.empty((rows, cols), dtype=np.float32)

        for g in range(n_groups):
            col_start = g * gs
            col_end = min(col_start + gs, cols)
            group_codes = codes[:, col_start:col_end]
            cb = codebook[g]
            out[:, col_start:col_end] = cb[group_codes]

        return out

    def lut_gemm(
        self,
        x: np.ndarray,
        codes: np.ndarray,
        codebook: np.ndarray,
    ) -> np.ndarray:
        """Simulate LUT-GEMM: x @ W_reconstructed.

        Parameters
        ----------
        x:
            Shape ``(batch, in_features)`` float32 activation.
        codes:
            Shape ``(out_features, in_features)`` uint8 weight codes.
        codebook:
            Shape ``(n_groups, codebook_size)`` float32.

        Returns
        -------
        output:
            Shape ``(batch, out_features)`` float32.
        """
        self.stats.lut_gemm_calls += 1
        w = self.dequantise(codes, codebook)  # (out_features, in_features)
        return x.astype(np.float32) @ w.T
