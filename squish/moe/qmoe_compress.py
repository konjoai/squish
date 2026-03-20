"""squish/moe/qmoe_compress.py

QMoECompressor — Sub-1-bit codebook compression for MoE expert weights
(Frantar & Alistarh, NeurIPS 2023 / production 2025).

Reference
---------
"QMoE: Practical Sub-1-Bit Compression of Trillion-Parameter Models."
Frantar & Alistarh, NeurIPS 2023 (arXiv:2310.16795).

Algorithm
---------
QMoE compresses each MoE expert weight matrix using a codebook (dictionary
learning) approach:

1. For each expert weight matrix W of shape (out, in):
   a. Divide W into ``block_size``-element blocks.
   b. Run K-Means on blocks to build a codebook of ``n_codes`` centroids.
   c. Each block is assigned to its nearest centroid (1 index per block).
2. Compression ratio ≈ log2(n_codes) / (32 * block_size) bits per parameter.
   With n_codes=256 and block_size=64: 8 / (32*64) = 0.00390625 bits/param
   → effectively sub-1-bit.

Decompression:
* Look up centroid for each block index.
* Reconstruct the full weight matrix from centroids + indices.

Key properties
--------------
* NumPy-only simulation; no GPU dependency.
* ``n_codes`` — codebook size (default 256 → 8-bit indices).
* ``block_size`` — number of weight elements per codebook block.
* ``n_iter`` — K-Means iterations for codebook construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

__all__ = [
    "QMoEConfig",
    "QMoECompressedExpert",
    "QMoECompressor",
]

# ── Config ────────────────────────────────────────────────────────────────────


@dataclass
class QMoEConfig:
    """Configuration for :class:`QMoECompressor`.

    Attributes:
        n_codes: Codebook size — must be a power of 2 ≥ 2.
        block_size: Number of weight elements per codebook entry.
        n_iter: K-Means iterations for codebook construction.
        seed: Random seed for K-Means initialisation.
    """

    n_codes: int = 256
    block_size: int = 64
    n_iter: int = 20
    seed: int = 0

    def __post_init__(self) -> None:
        if self.n_codes < 2:
            raise ValueError(f"n_codes must be ≥ 2; got {self.n_codes}")
        if (self.n_codes & (self.n_codes - 1)) != 0:
            raise ValueError(f"n_codes must be a power of 2; got {self.n_codes}")
        if self.block_size < 1:
            raise ValueError(f"block_size must be ≥ 1; got {self.block_size}")
        if self.n_iter < 1:
            raise ValueError(f"n_iter must be ≥ 1; got {self.n_iter}")


# ── Compressed expert ─────────────────────────────────────────────────────────


class QMoECompressedExpert:
    """Codebook-compressed representation of one MoE expert weight matrix.

    Attributes:
        codebook: ``(n_codes, block_size)`` float32 centroid vectors.
        indices: ``(n_blocks,)`` uint16 codebook indices.
        original_shape: ``(out_dim, in_dim)`` original weight shape.
        pad: Number of padding elements appended before blocking.
    """

    def __init__(
        self,
        codebook: np.ndarray,
        indices: np.ndarray,
        original_shape: Tuple[int, int],
        pad: int,
    ) -> None:
        self.codebook = codebook
        self.indices = indices
        self.original_shape = original_shape
        self.pad = pad

    def bits_per_param(self) -> float:
        """Effective bits per parameter after compression."""
        n_params = self.original_shape[0] * self.original_shape[1]
        n_codes = self.codebook.shape[0]
        bits_per_block = np.log2(n_codes)
        block_size = self.codebook.shape[1]
        return bits_per_block / block_size


# ── Compressor ────────────────────────────────────────────────────────────────


class QMoECompressor:
    """Codebook-based compressor for MoE expert weights.

    Example::

        cfg = QMoEConfig(n_codes=16, block_size=8, n_iter=10)
        compressor = QMoECompressor(cfg)

        W = np.random.randn(64, 64).astype(np.float32)
        compressed = compressor.compress(expert_id=0, weight=W)
        W_hat = compressor.decompress(compressed)
        err = compressor.relative_error(W, W_hat)
    """

    def __init__(self, config: Optional[QMoEConfig] = None) -> None:
        self.config = config or QMoEConfig()
        self._experts: Dict[int, QMoECompressedExpert] = {}

    # ── Public API ────────────────────────────────────────────────────────────

    def compress(
        self,
        expert_id: int,
        weight: np.ndarray,
    ) -> QMoECompressedExpert:
        """Compress one expert weight matrix.

        Args:
            expert_id: Expert identifier (used for storage).
            weight: ``(out_dim, in_dim)`` float32 weight matrix.

        Returns:
            :class:`QMoECompressedExpert` with codebook and indices.
        """
        weight = np.asarray(weight, dtype=np.float32)
        if weight.ndim != 2:
            raise ValueError(f"weight must be 2-D (out, in); got {weight.shape}")
        original_shape = weight.shape
        flat = weight.ravel()

        bs = self.config.block_size
        pad = (bs - len(flat) % bs) % bs
        if pad:
            flat = np.pad(flat, (0, pad))

        n_blocks = len(flat) // bs
        blocks = flat.reshape(n_blocks, bs)  # (n_blocks, block_size)

        codebook, indices = self._kmeans(blocks)
        compressed = QMoECompressedExpert(
            codebook=codebook,
            indices=indices.astype(np.uint16),
            original_shape=original_shape,
            pad=pad,
        )
        self._experts[expert_id] = compressed
        return compressed

    def decompress(
        self,
        compressed: QMoECompressedExpert,
    ) -> np.ndarray:
        """Decompress an expert weight matrix.

        Args:
            compressed: :class:`QMoECompressedExpert` from :meth:`compress`.

        Returns:
            Reconstructed ``(out_dim, in_dim)`` float32 weight matrix.
        """
        blocks = compressed.codebook[compressed.indices.astype(np.int32)]  # (n_blocks, bs)
        flat = blocks.ravel()
        if compressed.pad:
            flat = flat[: -compressed.pad]
        return flat.reshape(compressed.original_shape).astype(np.float32)

    def relative_error(
        self,
        original: np.ndarray,
        reconstructed: np.ndarray,
    ) -> float:
        """Compute relative Frobenius reconstruction error.

        Returns:
            ||W - W_hat||_F / ||W||_F.
        """
        numerator = float(np.linalg.norm(original - reconstructed))
        denominator = float(np.linalg.norm(original)) + 1e-9
        return numerator / denominator

    def store(self, expert_id: int, compressed: QMoECompressedExpert) -> None:
        """Store a compressed expert by ID."""
        self._experts[expert_id] = compressed

    def load(self, expert_id: int) -> QMoECompressedExpert:
        """Load a compressed expert by ID.

        Raises:
            KeyError: If expert_id not stored.
        """
        if expert_id not in self._experts:
            raise KeyError(f"Expert {expert_id} not stored.")
        return self._experts[expert_id]

    def n_stored_experts(self) -> int:
        """Number of compressed experts stored."""
        return len(self._experts)

    # ── K-Means helper ────────────────────────────────────────────────────────

    def _kmeans(
        self,
        blocks: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run K-Means on ``blocks`` and return (codebook, indices).

        Args:
            blocks: ``(n_blocks, block_size)`` float32 array.

        Returns:
            Tuple ``(codebook, indices)`` where codebook is
            ``(n_codes, block_size)`` and indices is ``(n_blocks,)`` int32.
        """
        rng = np.random.default_rng(self.config.seed)
        n_blocks, bs = blocks.shape
        k = min(self.config.n_codes, n_blocks)

        # Initialise centroids via random subset
        idx = rng.choice(n_blocks, size=k, replace=False)
        centroids = blocks[idx].copy()

        for _ in range(self.config.n_iter):
            # Assign
            dists = np.sum(
                (blocks[:, None, :] - centroids[None, :, :]) ** 2, axis=-1
            )  # (n_blocks, k)
            assignments = dists.argmin(axis=-1)  # (n_blocks,)

            # Update
            new_centroids = np.zeros_like(centroids)
            counts = np.zeros(k, dtype=np.int32)
            for ci in range(k):
                mask = assignments == ci
                if mask.any():
                    new_centroids[ci] = blocks[mask].mean(axis=0)
                    counts[ci] = mask.sum()
                else:
                    # Re-init empty cluster
                    new_centroids[ci] = blocks[rng.integers(n_blocks)]
            centroids = new_centroids

        # Final assignment
        dists = np.sum(
            (blocks[:, None, :] - centroids[None, :, :]) ** 2, axis=-1
        )
        assignments = dists.argmin(axis=-1)

        # Pad codebook to n_codes if needed
        if k < self.config.n_codes:
            pad_rows = np.zeros(
                (self.config.n_codes - k, bs), dtype=np.float32
            )
            centroids = np.vstack([centroids, pad_rows])

        return centroids.astype(np.float32), assignments.astype(np.int32)

    def __repr__(self) -> str:
        cfg = self.config
        return (
            f"QMoECompressor(n_codes={cfg.n_codes}, block_size={cfg.block_size}, "
            f"n_experts_stored={self.n_stored_experts()})"
        )
