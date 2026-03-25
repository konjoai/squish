"""
any4 — Learned 4-Bit LUT Quantization.

Inspired by: "any4: Learned 4-bit LUT Quantization" (Meta AI, NeurIPS 2025).
Key ideas:
  1.  Replace the fixed INT4/NF4/FP4 value sets with a *learned* codebook of
      16 floating-point codewords, optimised to minimise quantization error on
      calibration data.
  2.  Only a single well-chosen calibration example is required (Meta paper
      finding) — dramatically simpler than AWQ / GPTQ pipelines.
  3.  Lookup-table dequantization is GPU/ANE-efficient (single gather op).

This pure-NumPy implementation is framework-agnostic.  In production, the
tinygemm-style fast GEMM can be wired in via MLX custom ops.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class Any4Config:
    """Configuration for any4 learned 4-bit quantization."""
    codebook_size: int = 16          # 2^4 = 16 codewords
    group_size: int = 128            # quantize in groups of this many weights
    calibration_iters: int = 100     # Lloyd-style k-means iterations
    calibration_seed: int = 42       # RNG seed for centroid init
    symmetric: bool = False          # symmetric vs asymmetric group scaling

    def __post_init__(self) -> None:
        if self.codebook_size != 16:
            raise ValueError("any4 requires codebook_size == 16 (4 bits)")
        if self.group_size < 16:
            raise ValueError(f"group_size must be >= 16, got {self.group_size}")


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class Any4Quantized:
    """Packed any4 weight tensor."""
    codes: np.ndarray          # uint8, shape (n_groups, group_size//2) — packed nibbles
    codebook: np.ndarray       # float32 (16,) — learned lookup table
    scale: np.ndarray          # float32 (n_groups,) — per-group scale
    zero: np.ndarray           # float32 (n_groups,) — per-group zero-point
    original_shape: Tuple[int, ...]

    def nbytes(self) -> int:
        return int(self.codes.nbytes + self.codebook.nbytes + self.scale.nbytes + self.zero.nbytes)

    @property
    def compression_ratio(self) -> float:
        original_bytes = int(np.prod(self.original_shape)) * 4  # fp32
        return original_bytes / self.nbytes()


@dataclass
class Any4Stats:
    """Statistics for any4 operations."""
    calibration_calls: int = 0
    quantize_calls: int = 0
    dequantize_calls: int = 0
    total_weights_quantized: int = 0

    def __repr__(self) -> str:
        return (
            f"Any4Stats(calibrations={self.calibration_calls}, "
            f"quantize={self.quantize_calls}, "
            f"weights={self.total_weights_quantized:,})"
        )


# ---------------------------------------------------------------------------
# Quantizer
# ---------------------------------------------------------------------------

class Any4Quantizer:
    """Learns a 4-bit codebook from calibration data and quantizes weight tensors.

    Usage::

        quantizer = Any4Quantizer(Any4Config())
        quantizer.calibrate(weight_sample)   # single calibration example
        q = quantizer.quantize(weight)
        w_approx = quantizer.dequantize(q)
    """

    def __init__(self, config: Any4Config) -> None:
        self.config = config
        self.stats = Any4Stats()
        self._codebook: Optional[np.ndarray] = None  # (16,) float32
        self._calibrated = False

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def calibrate(self, weight_sample: np.ndarray) -> None:
        """Learn a 16-codeword codebook from weight values via k-means.

        Args:
            weight_sample: any-shape float32 array of representative weights.
                           The Meta paper shows one layer of one forward pass
                           is sufficient.
        """
        values = weight_sample.astype(np.float32).ravel()
        self._codebook = self._kmeans_codebook(values)
        self._calibrated = True
        self.stats.calibration_calls += 1

    def _kmeans_codebook(self, values: np.ndarray) -> np.ndarray:
        """1-D k-means to find 16 representative codewords."""
        cfg = self.config
        rng = np.random.default_rng(cfg.calibration_seed)
        k = cfg.codebook_size

        # Initialise centroids with evenly-spaced quantiles
        quantiles = np.linspace(0, 100, k)
        centroids = np.percentile(values, quantiles).astype(np.float32)

        for _ in range(cfg.calibration_iters):
            dists = np.abs(values[:, np.newaxis] - centroids[np.newaxis, :])
            assignments = np.argmin(dists, axis=1)
            new_centroids = np.array([
                values[assignments == j].mean() if (assignments == j).any() else centroids[j]
                for j in range(k)
            ], dtype=np.float32)
            if np.allclose(centroids, new_centroids, atol=1e-6):
                break
            centroids = new_centroids

        return np.sort(centroids)  # sorted so nibble index is monotonic

    # ------------------------------------------------------------------
    # Quantize / dequantize
    # ------------------------------------------------------------------

    def quantize(self, weight: np.ndarray) -> Any4Quantized:
        """Quantize a weight tensor to any4 format.

        Args:
            weight: (rows, cols) or any 2-D float tensor.

        Returns:
            Any4Quantized with packed nibble codes and per-group scale/zero.
        """
        if not self._calibrated:
            raise RuntimeError("Any4Quantizer must be calibrated before quantize()")
        if weight.ndim < 1:
            raise ValueError("weight must have at least 1 dimension")

        original_shape = weight.shape
        w = weight.astype(np.float32).ravel()
        n = len(w)
        gs = self.config.group_size

        # Pad to multiple of group_size
        pad = (gs - n % gs) % gs
        if pad:
            w = np.concatenate([w, np.zeros(pad, dtype=np.float32)])

        n_groups = len(w) // gs
        w_groups = w.reshape(n_groups, gs)

        # Per-group scale and zero
        g_min = w_groups.min(axis=1)           # (n_groups,)
        g_max = w_groups.max(axis=1)
        scale = (g_max - g_min).clip(min=1e-8)
        zero = g_min

        # Normalise each group to [0, 1], map to codebook
        w_norm = (w_groups - zero[:, np.newaxis]) / scale[:, np.newaxis]  # [0, 1]
        # Scale codebook to [0, 1]
        cb = self._codebook
        cb_norm = (cb - cb.min()) / (cb.max() - cb.min()).clip(min=1e-8)

        # Nearest-centroid assignment for each weight
        dists = np.abs(w_norm[:, :, np.newaxis] - cb_norm[np.newaxis, np.newaxis, :])
        codes_unpacked = np.argmin(dists, axis=2).astype(np.uint8)  # (n_groups, gs)

        # Pack pairs of nibbles into bytes
        codes = self._pack_nibbles(codes_unpacked)  # (n_groups, gs//2)

        self.stats.quantize_calls += 1
        self.stats.total_weights_quantized += n

        return Any4Quantized(
            codes=codes,
            codebook=cb.copy(),
            scale=scale.astype(np.float32),
            zero=zero.astype(np.float32),
            original_shape=original_shape,
        )

    def dequantize(self, q: Any4Quantized) -> np.ndarray:
        """Reconstruct fp32 weight tensor from any4 representation.

        Args:
            q: Any4Quantized produced by quantize().

        Returns:
            fp32 array with shape q.original_shape.
        """
        codes_unpacked = self._unpack_nibbles(q.codes)  # (n_groups, gs)
        n_groups, gs = codes_unpacked.shape

        cb = q.codebook
        cb_norm = (cb - cb.min()) / (cb.max() - cb.min()).clip(min=1e-8)

        # Lookup in normalised codebook → renormalise with group scale/zero
        values_norm = cb_norm[codes_unpacked]                          # (n_groups, gs)
        values = values_norm * q.scale[:, np.newaxis] + q.zero[:, np.newaxis]

        flat = values.ravel()
        n_orig = int(np.prod(q.original_shape))
        self.stats.dequantize_calls += 1
        return flat[:n_orig].reshape(q.original_shape).astype(np.float32)

    # ------------------------------------------------------------------
    # Nibble packing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _pack_nibbles(codes: np.ndarray) -> np.ndarray:
        """Pack (n_groups, gs) uint8 codes into (n_groups, gs//2) bytes."""
        n_groups, gs = codes.shape
        if gs % 2 != 0:
            # pad to even
            codes = np.concatenate([codes, np.zeros((n_groups, 1), dtype=np.uint8)], axis=1)
            gs += 1
        lo = codes[:, 0::2] & 0x0F   # even indices → low nibble
        hi = codes[:, 1::2] & 0x0F   # odd indices  → high nibble
        return (lo | (hi << 4)).astype(np.uint8)

    @staticmethod
    def _unpack_nibbles(packed: np.ndarray) -> np.ndarray:
        """Unpack (n_groups, gs//2) bytes into (n_groups, gs) uint8 codes."""
        lo = packed & 0x0F
        hi = (packed >> 4) & 0x0F
        # interleave: even=lo, odd=hi
        n_groups, half_gs = packed.shape
        out = np.empty((n_groups, half_gs * 2), dtype=np.uint8)
        out[:, 0::2] = lo
        out[:, 1::2] = hi
        return out

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def codebook(self) -> Optional[np.ndarray]:
        return self._codebook

    def __repr__(self) -> str:
        return (
            f"Any4Quantizer(group_size={self.config.group_size}, "
            f"calibrated={self._calibrated}, {self.stats})"
        )
