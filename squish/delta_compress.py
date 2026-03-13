# [Experimental] This module is part of Squish v6+ (Waves 19–26).
# Proof-of-concept quality: API and behaviour may change without notice.
# For stable core functionality see Waves 1–12 in MODULES.md.
"""DeltaCompress — SVD-based delta compression for fine-tuned model weights.

Fine-tuning a large model produces weight deltas ΔW = W_ft − W_base that are
often low-rank in practice.  Truncated SVD yields a rank-k approximation
ΔW ≈ U_k Σ_k V_k^T where k << min(rows, cols), achieving compression ratios
that scale directly with how low-rank the true delta is.

This approach is applied post-hoc to arbitrary weight deltas (no changes to
the fine-tuning procedure are required), making it compatible with any LoRA or
full fine-tune checkpoint.

Reference:
    Inspired by Hu et al., "LoRA: Low-Rank Adaptation of Large Language
    Models", ICLR 2022. https://arxiv.org/abs/2106.09685

Usage::

    import numpy as np
    from squish.delta_compress import DeltaConfig, DeltaCompressor, DeltaStats

    rng = np.random.default_rng(42)
    base      = rng.standard_normal((256, 512)).astype(np.float32)
    finetuned = base + 0.01 * rng.standard_normal((256, 512)).astype(np.float32)

    cfg        = DeltaConfig(rank=16)
    compressor = DeltaCompressor(cfg)

    U_k, S_k, Vt_k = compressor.compress(base, finetuned)
    delta_approx    = compressor.decompress(U_k, S_k, Vt_k)

    ratio = compressor.compression_ratio(256, 512, S_k.shape[0])
    print(f"Compression ratio: {ratio:.2f}x")
    print(compressor.stats)
"""

from __future__ import annotations

__all__ = ["DeltaConfig", "DeltaCompressor", "DeltaStats"]

from dataclasses import dataclass

import numpy as np


@dataclass
class DeltaConfig:
    """Configuration for SVD-based delta compression.

    Attributes:
        rank: Maximum number of singular values / vectors to retain.
            The effective rank ``k = min(rank, min(rows, cols))``.
    """

    rank: int = 16

    def __post_init__(self) -> None:
        if self.rank < 1:
            raise ValueError(f"rank must be >= 1; got {self.rank}")


@dataclass
class DeltaStats:
    """Running statistics for a :class:`DeltaCompressor` session.

    Attributes:
        total_compress_calls: Number of :meth:`DeltaCompressor.compress`
            invocations.
        total_singular_values_kept: Cumulative count of singular values
            retained across all compress calls.
    """

    total_compress_calls: int = 0
    total_singular_values_kept: int = 0


class DeltaCompressor:
    """Compresses fine-tune weight deltas via truncated SVD.

    Args:
        config: :class:`DeltaConfig` controlling the truncation rank.

    Example::

        cfg = DeltaConfig(rank=8)
        c   = DeltaCompressor(cfg)
        U_k, S_k, Vt_k = c.compress(base_w, ft_w)
        approx_delta = c.decompress(U_k, S_k, Vt_k)
    """

    def __init__(self, config: DeltaConfig) -> None:
        self._config = config
        self._stats = DeltaStats()

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def compress(
        self,
        base: np.ndarray,
        finetuned: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute a rank-k SVD approximation of the weight delta.

        Args:
            base: Base model weights, shape ``(rows, cols)`` float32.
            finetuned: Fine-tuned weights, identical shape, float32.

        Returns:
            Tuple ``(U_k, S_k, Vt_k)`` where:

            * ``U_k``  — left singular vectors, shape ``(rows, k)``, float32.
            * ``S_k``  — singular values, shape ``(k,)``, float32.
            * ``Vt_k`` — right singular vectors, shape ``(k, cols)``, float32.

        Raises:
            ValueError: If ``base`` and ``finetuned`` have different shapes or
                are not 2-D.
        """
        base = np.asarray(base, dtype=np.float32)
        finetuned = np.asarray(finetuned, dtype=np.float32)

        if base.ndim != 2:
            raise ValueError(
                f"base must be 2-D; got shape {base.shape}"
            )
        if base.shape != finetuned.shape:
            raise ValueError(
                f"base and finetuned must have the same shape; "
                f"got {base.shape} vs {finetuned.shape}"
            )

        delta: np.ndarray = finetuned - base

        # Full thin SVD: U (rows, k), S (k,), Vt (k, cols)
        U, S, Vt = np.linalg.svd(delta, full_matrices=False)

        # Effective rank is bounded by the number of non-zero singular values
        k = min(self._config.rank, int(S.shape[0]))

        U_k  = U[:, :k].astype(np.float32)
        S_k  = S[:k].astype(np.float32)
        Vt_k = Vt[:k, :].astype(np.float32)

        self._stats.total_compress_calls += 1
        self._stats.total_singular_values_kept += k

        return U_k, S_k, Vt_k

    def decompress(
        self,
        U_k: np.ndarray,
        S_k: np.ndarray,
        Vt_k: np.ndarray,
    ) -> np.ndarray:
        """Reconstruct the delta from its truncated SVD factors.

        Args:
            U_k:  Left singular vectors, shape ``(rows, k)``.
            S_k:  Singular values, shape ``(k,)``.
            Vt_k: Right singular vectors, shape ``(k, cols)``.

        Returns:
            Approximate delta ``(U_k * S_k) @ Vt_k``, shape ``(rows, cols)``,
            dtype float32.
        """
        U_k  = np.asarray(U_k,  dtype=np.float32)
        S_k  = np.asarray(S_k,  dtype=np.float32)
        Vt_k = np.asarray(Vt_k, dtype=np.float32)

        # (rows, k) * (k,) → (rows, k); then @ (k, cols) → (rows, cols)
        return ((U_k * S_k) @ Vt_k).astype(np.float32)

    @staticmethod
    def compression_ratio(rows: int, cols: int, k: int) -> float:
        """Ratio of original parameter count to compressed parameter count.

        The compressed representation stores ``rows*k + k + k*cols`` values
        (U_k, S_k, Vt_k) vs the original ``rows*cols``.

        Args:
            rows: Number of rows in the weight matrix.
            cols: Number of columns in the weight matrix.
            k:    Effective rank used for compression.

        Returns:
            Compression ratio (>1 means the compressed form is smaller).
        """
        if k <= 0:
            raise ValueError(f"k must be >= 1; got {k}")
        original   = rows * cols
        compressed = rows * k + k + k * cols
        return original / compressed

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def stats(self) -> DeltaStats:
        """Running compression statistics."""
        return self._stats
