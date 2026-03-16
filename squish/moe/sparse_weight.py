# [Experimental] This module is part of Squish v6+ (Waves 19–26).
# Proof-of-concept quality: API and behaviour may change without notice.
# For stable core functionality see Waves 1–12 in MODULES.md.
"""SparseWeight — CSR-style 2:4 pruned weight matrix storage.

Stores the non-zero values from a 2:4 (N:M) structured-sparse weight matrix
in a compact Compressed Sparse Row-equivalent layout:

* **values** — float32 array of shape ``(rows, cols // M × N)`` containing
  the kept weights for each row.
* **col_indices** — int16 array of the same shape recording the absolute
  column index of each stored value within its row.

For the canonical 2:4 configuration (50% sparsity), the values array holds
exactly half as many elements as the original dense matrix and the col_indices
array uses 2 bytes each, yielding the following storage comparison:

    dense bytes  = rows × cols × 4
    sparse bytes = rows × (cols/2) × 4   (values, float32)
                 + rows × (cols/2) × 2   (col indices, int16)
                 = rows × cols × 3

This yields a compression_ratio of 4/3 ≈ 1.33× when both values and
indices are counted; or 2× if only the value storage is compared.

Usage::

    import numpy as np
    from squish.moe.sparse_weight import SparsityConfig, SparseWeightStore

    cfg   = SparsityConfig(N=2, M=4)
    store = SparseWeightStore(cfg)

    weights = np.random.randn(4096, 4096).astype(np.float32)
    store.compress(weights)

    dense = store.decompress()
    print(store.compression_ratio)            # ≈ 1.33
    print(store.stats.current_compression_ratio)
"""

from __future__ import annotations

__all__ = [
    "SparsityConfig",
    "SparseWeightStore",
    "SparseStats",
]

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class SparsityConfig:
    """Configuration for N:M sparse weight storage.

    Attributes:
        N: Number of non-zero values to keep per group.  Must satisfy
            ``1 <= N < M``.
        M: Group size (number of consecutive elements per group).  Must
            be >= 2.
    """

    N: int = 2
    M: int = 4

    def __post_init__(self) -> None:
        if self.M < 2:
            raise ValueError(
                f"M must be >= 2; got {self.M}"
            )
        if self.N < 1:
            raise ValueError(
                f"N must be >= 1; got {self.N}"
            )
        if self.N >= self.M:
            raise ValueError(
                f"N must be strictly less than M; got N={self.N}, M={self.M}"
            )


# ── Stats ─────────────────────────────────────────────────────────────────────

@dataclass
class SparseStats:
    """Runtime statistics for a :class:`SparseWeightStore` session.

    Attributes:
        total_compress_calls: Cumulative calls to
            :meth:`SparseWeightStore.compress`.
        total_decompress_calls: Cumulative calls to
            :meth:`SparseWeightStore.decompress`.
        _ratio_snapshot: Internal float capturing the compression ratio at
            the time this stats snapshot was created.  Exposed via the
            :attr:`current_compression_ratio` property.
    """

    total_compress_calls: int = 0
    total_decompress_calls: int = 0
    _ratio_snapshot: float = field(default=1.0, repr=False)

    @property
    def current_compression_ratio(self) -> float:
        """Compression ratio (dense bytes / sparse bytes) at snapshot time."""
        return self._ratio_snapshot


# ── Store ─────────────────────────────────────────────────────────────────────

class SparseWeightStore:
    """CSR-equivalent storage for 2:4 structured-sparse weight matrices.

    Compresses a 2-D float32 weight matrix into (values, col_indices) arrays
    and reconstructs the dense matrix on demand.

    Args:
        config: :class:`SparsityConfig` specifying N and M.
    """

    def __init__(self, config: SparsityConfig) -> None:
        self.config = config

        self._values: np.ndarray | None = None       # float32 (rows, nnz_per_row)
        self._col_indices: np.ndarray | None = None  # int16   (rows, nnz_per_row)
        self._rows: int = 0
        self._cols: int = 0

        self._n_compress:   int = 0
        self._n_decompress: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compress(self, dense_weights: np.ndarray) -> None:
        """Compress a dense weight matrix into CSR-like sparse storage.

        For each row and each group of *M* consecutive columns, the *N*
        elements with the largest magnitude are stored; all others are
        discarded.  The absolute column indices of the stored elements are
        saved as int16 for memory efficiency.

        Args:
            dense_weights: Float32 array of shape ``(rows, cols)`` where
                ``cols`` must be divisible by *M*.

        Raises:
            ValueError: If *dense_weights* is not 2-D, or if ``cols`` is
                not divisible by *M*.
        """
        cfg = self.config
        N, M = cfg.N, cfg.M

        dense_weights = np.asarray(dense_weights, dtype=np.float32)

        if dense_weights.ndim != 2:
            raise ValueError(
                f"dense_weights must be 2-D (rows, cols); "
                f"got {dense_weights.ndim}-D shape {dense_weights.shape}"
            )

        rows, cols = dense_weights.shape
        if cols % M != 0:
            raise ValueError(
                f"cols ({cols}) must be divisible by M={M}"
            )

        n_groups    = cols // M
        nnz_per_row = n_groups * N  # non-zeros per row

        # Reshape to (rows, n_groups, M) for vectorised group-wise selection.
        reshaped = dense_weights.reshape(rows, n_groups, M)

        # Indices of the top N by magnitude within each group.
        # top_idx shape: (rows, n_groups, N)
        top_idx = np.argsort(-np.abs(reshaped), axis=-1)[:, :, :N]

        # Sort indices within each group so reconstruction is deterministic
        # and consistent with sorted-index conventions in sparse libraries.
        top_idx = np.sort(top_idx, axis=-1)

        # Extract the kept values using advanced indexing.
        row_idx = np.arange(rows).reshape(rows, 1, 1)           # (rows, 1, 1)
        grp_idx = np.arange(n_groups).reshape(1, n_groups, 1)   # (1, n_groups, 1)
        vals    = reshaped[row_idx, grp_idx, top_idx]            # (rows, n_groups, N)

        # Absolute column indices: group_base + intra-group offset.
        grp_base = (
            np.arange(n_groups).reshape(1, n_groups, 1) * M
        )  # (1, n_groups, 1)
        abs_col  = (grp_base + top_idx).astype(np.int16)         # (rows, n_groups, N)

        # Flatten the (n_groups, N) axes into one axis of length nnz_per_row.
        self._values      = vals.reshape(rows, nnz_per_row).astype(np.float32)
        self._col_indices = abs_col.reshape(rows, nnz_per_row)
        self._rows        = rows
        self._cols        = cols

        self._n_compress += 1

    def decompress(self) -> np.ndarray:
        """Reconstruct the dense float32 weight matrix from sparse storage.

        Places stored values back at their recorded column positions; all
        other positions are zero.

        Returns:
            Float32 array of shape ``(rows, cols)`` matching the original
            dense matrix (up to values that were zeroed during compression).

        Raises:
            RuntimeError: If :meth:`compress` has not been called first.
        """
        if self._values is None or self._col_indices is None:
            raise RuntimeError(
                "compress() must be called before decompress()"
            )

        rows, cols = self._rows, self._cols
        dense = np.zeros((rows, cols), dtype=np.float32)

        # Vectorised scatter: place each stored value at its recorded column.
        row_idx = np.arange(rows).reshape(rows, 1)  # (rows, 1)
        dense[row_idx, self._col_indices.astype(np.int32)] = self._values

        self._n_decompress += 1
        return dense

    def memory_bytes(self) -> int:
        """Bytes consumed by the current sparse representation.

        Counts 4 bytes per float32 value and 2 bytes per int16 column index.

        Returns:
            Total sparse storage in bytes, or 0 if :meth:`compress` has not
            yet been called.
        """
        if self._values is None:
            return 0
        value_bytes = int(self._values.size) * 4     # float32
        index_bytes = int(self._col_indices.size) * 2  # int16
        return value_bytes + index_bytes

    def dense_memory_bytes(self) -> int:
        """Bytes that would be needed for the equivalent dense float32 matrix.

        Returns:
            ``rows × cols × 4``, or 0 if :meth:`compress` has not yet been
            called.
        """
        return self._rows * self._cols * 4

    @property
    def compression_ratio(self) -> float:
        """Ratio of dense storage bytes to sparse storage bytes.

        Values greater than 1 indicate compression.  Returns 1.0 when
        :meth:`compress` has not been called or sparse storage is empty.
        """
        sparse = self.memory_bytes()
        return float(self.dense_memory_bytes()) / max(1, sparse)

    @property
    def stats(self) -> SparseStats:
        """Live statistics snapshot including the current compression ratio."""
        return SparseStats(
            total_compress_calls=self._n_compress,
            total_decompress_calls=self._n_decompress,
            _ratio_snapshot=self.compression_ratio,
        )
