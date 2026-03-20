"""
INT3RuntimeLoader — MiLo INT3 Runtime Dequantization.

Context: Squish's MiLo INT3 compression is currently *storage-only*.
Compressed weights are saved to npy-dir format but there is no runtime
dequantization path — the loader falls back to BF16 for inference.

This module closes that gap: given the packed INT3 representation written
by squish's `quantize` command, it reconstructs fp32 tensors on demand,
enabling sub-4B models to run entirely from INT3 weights at ~4× memory
savings.

Format (same as squish/quantizer.py MiLo output):
  Each npy-dir layer has:
    <name>__q3   — (n_groups, group_size) uint8, 3-bit per weight (packed in 8-bit bytes)
    <name>__s3   — (n_groups,) float32 scale
    <name>__z3   — (n_groups,) float32 zero-point

Where n_groups = ceil(n_weights / group_size) and weights are packed as
3 bits per element in byte arrays (8 weights per 3 bytes — 24-bit aligned).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, Generator, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class INT3RuntimeConfig:
    """Configuration for INT3 runtime dequantization."""
    group_size: int = 64            # must match the compression group size
    tile_size: int = 256            # dequantize this many groups at a time
    dtype: np.dtype = field(default_factory=lambda: np.dtype("float32"))

    def __post_init__(self) -> None:
        if self.group_size < 8:
            raise ValueError(f"group_size must be >= 8, got {self.group_size}")
        if self.tile_size < 1:
            raise ValueError(f"tile_size must be >= 1, got {self.tile_size}")


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class INT3LayerWeights:
    """Loaded INT3 weights ready for runtime dequantization."""
    q_packed: np.ndarray            # (n_groups, group_size) uint8 — 3-bit values in low bits
    scales: np.ndarray              # (n_groups,) float32
    zeros: np.ndarray               # (n_groups,) float32
    original_shape: Tuple[int, ...]
    group_size: int

    @property
    def n_groups(self) -> int:
        return self.q_packed.shape[0]

    @property
    def compactness(self) -> float:
        """Approximate compression ratio vs fp32."""
        fp32_bytes = int(np.prod(self.original_shape)) * 4
        packed_bytes = self.q_packed.nbytes + self.scales.nbytes + self.zeros.nbytes
        return fp32_bytes / max(packed_bytes, 1)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

@dataclass
class INT3LoaderStats:
    layers_loaded: int = 0
    tensors_dequantized: int = 0
    total_weights_recovered: int = 0

    def __repr__(self) -> str:
        return (
            f"INT3LoaderStats(loaded={self.layers_loaded}, "
            f"dequant={self.tensors_dequantized}, "
            f"weights={self.total_weights_recovered:,})"
        )


class INT3RuntimeLoader:
    """Load and dequantize MiLo INT3 weights from a squish npy-dir.

    Usage::

        loader = INT3RuntimeLoader(INT3RuntimeConfig())
        # Option A — load from saved .npy arrays
        weights = loader.load_from_arrays(q_packed, scales, zeros, shape)
        tensor = loader.dequantize(weights)

        # Option B — stream in tiles (low peak memory)
        for tile in loader.dequantize_tiled(weights):
            ...  # process fp32 tile
    """

    def __init__(self, config: INT3RuntimeConfig) -> None:
        self.config = config
        self.stats = INT3LoaderStats()

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load_from_arrays(
        self,
        q_packed: np.ndarray,
        scales: np.ndarray,
        zeros: np.ndarray,
        original_shape: Tuple[int, ...],
    ) -> INT3LayerWeights:
        """Construct INT3LayerWeights from raw numpy arrays.

        Args:
            q_packed:       (n_groups, group_size) uint8 — 3-bit values (0–7).
            scales:         (n_groups,) float32.
            zeros:          (n_groups,) float32.
            original_shape: target weight shape after dequantization.

        Returns:
            INT3LayerWeights ready for dequantization.
        """
        if q_packed.ndim != 2:
            raise ValueError(f"q_packed must be 2-D, got shape {q_packed.shape}")
        if q_packed.shape[0] != scales.shape[0] or q_packed.shape[0] != zeros.shape[0]:
            raise ValueError(
                f"q_packed, scales, zeros must have same n_groups: "
                f"{q_packed.shape[0]}, {scales.shape[0]}, {zeros.shape[0]}"
            )
        self.stats.layers_loaded += 1
        return INT3LayerWeights(
            q_packed=q_packed.astype(np.uint8),
            scales=scales.astype(np.float32),
            zeros=zeros.astype(np.float32),
            original_shape=original_shape,
            group_size=q_packed.shape[1],
        )

    def load_layer(self, npy_dir: str, layer_name: str) -> INT3LayerWeights:
        """Load INT3 weights for one layer from a squish npy-dir.

        Expects files:  {layer_name}__q3.npy, {layer_name}__s3.npy,
                        {layer_name}__z3.npy, {layer_name}__shape.npy

        Args:
            npy_dir:    path to squish npy-dir.
            layer_name: parameter name prefix.

        Returns:
            INT3LayerWeights.
        """
        def _load(suffix: str) -> np.ndarray:
            path = os.path.join(npy_dir, f"{layer_name}{suffix}.npy")
            if not os.path.isfile(path):
                raise FileNotFoundError(f"INT3 file not found: {path}")
            return np.load(path)

        q_packed = _load("__q3")
        scales = _load("__s3")
        zeros = _load("__z3")
        shape_arr = _load("__shape")
        original_shape = tuple(int(d) for d in shape_arr)

        return self.load_from_arrays(q_packed, scales, zeros, original_shape)

    # ------------------------------------------------------------------
    # Dequantize
    # ------------------------------------------------------------------

    def dequantize(self, weights: INT3LayerWeights) -> np.ndarray:
        """Reconstruct full fp32 tensor from INT3LayerWeights.

        Args:
            weights: INT3LayerWeights from load_layer() or load_from_arrays().

        Returns:
            fp32 ndarray with shape weights.original_shape.
        """
        n_groups = weights.n_groups
        gs = weights.group_size

        # codes ∈ [0, 7] (3-bit unsigned); dequant formula:
        #   w = scale * (code - zero)
        codes = weights.q_packed.astype(np.float32)   # (n_groups, gs)
        reconstructed = (
            codes * weights.scales[:, np.newaxis]
            + weights.zeros[:, np.newaxis]
        ).ravel()

        n_orig = int(np.prod(weights.original_shape))
        result = reconstructed[:n_orig].reshape(weights.original_shape)

        self.stats.tensors_dequantized += 1
        self.stats.total_weights_recovered += n_orig
        return result.astype(self.config.dtype)

    def dequantize_tiled(
        self, weights: INT3LayerWeights
    ) -> Generator[np.ndarray, None, None]:
        """Stream dequantization in tile_size-group chunks.

        Reduces peak memory by only materialising one tile at a time.

        Args:
            weights: INT3LayerWeights.

        Yields:
            fp32 tiles, each of shape (tile_groups * group_size,).
        """
        tile = self.config.tile_size
        n_groups = weights.n_groups
        gs = weights.group_size

        for start in range(0, n_groups, tile):
            end = min(start + tile, n_groups)
            codes = weights.q_packed[start:end].astype(np.float32)
            scales = weights.scales[start:end]
            zeros = weights.zeros[start:end]
            chunk = (codes * scales[:, np.newaxis] + zeros[:, np.newaxis]).ravel()
            self.stats.tensors_dequantized += 1
            self.stats.total_weights_recovered += len(chunk)
            yield chunk.astype(self.config.dtype)

    def __repr__(self) -> str:
        return (
            f"INT3RuntimeLoader(gs={self.config.group_size}, "
            f"tile={self.config.tile_size}, {self.stats})"
        )
