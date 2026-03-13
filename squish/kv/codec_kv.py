# [Experimental] This module is part of Squish v6+ (Waves 19–26).
# Proof-of-concept quality: API and behaviour may change without notice.
# For stable core functionality see Waves 1–12 in MODULES.md.
"""CodecKV — Learned KV codec for 2–4× KV compression via latent codes.

Rather than quantising KV cache entries with fixed-precision arithmetic, a
codebook of typical KV patterns is fitted via Lloyd's k-means algorithm.
Each new KV entry is encoded as the index of its nearest codebook centroid,
achieving better reconstruction quality than uniform quantisation at the same
index bit-width.

Compression ratio for ``head_dim=64``, ``n_codebook=256`` (8-bit indices)::

    32 * 64 / log2(256)  =  2048 / 8  =  256×

In practice the ratio is computed over a single vector: ``32 * head_dim``
original bits vs. ``log2(n_codebook)`` encoded bits.

Reference:
    Wan et al., "TiC-KV: KV Cache Compression via Tiered Codebooks",
    arXiv 2024.

Usage::

    import numpy as np
    from squish.kv.codec_kv import KVCodec, CodecConfig

    cfg   = CodecConfig(n_codebook=256, head_dim=64, n_heads=8)
    codec = KVCodec(cfg)

    rng = np.random.default_rng(0)
    keys_sample   = rng.standard_normal((500, 64)).astype(np.float32)
    values_sample = rng.standard_normal((500, 64)).astype(np.float32)
    codec.fit(keys_sample, values_sample)

    keys   = rng.standard_normal((8, 32, 64)).astype(np.float32)
    idx_k  = codec.encode_keys(keys)          # (8, 32) int32
    k_hat  = codec.decode_keys(idx_k[0], 0)  # (32, 64)
    print(f"Compression ratio: {codec.compression_ratio:.1f}×")
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

__all__ = ["CodecConfig", "KVCodec", "CodecStats"]

_MAX_LLOYD_ITERS: int = 20


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class CodecConfig:
    """Configuration for the learned KV codec.

    Attributes:
        n_codebook: Number of codebook entries.  Must be >= 2.
        head_dim: Dimension of each key/value head vector.
        n_heads: Number of attention heads.
        n_fit_samples: Maximum number of vectors used per Lloyd's k-means
            iteration.  If the supplied data exceeds this count, a random
            subsample is taken.
    """

    n_codebook: int = 256
    head_dim: int = 64
    n_heads: int = 8
    n_fit_samples: int = 1000

    def __post_init__(self) -> None:
        if self.n_codebook < 2:
            raise ValueError(f"n_codebook must be >= 2; got {self.n_codebook}")
        if self.head_dim < 1:
            raise ValueError(f"head_dim must be >= 1; got {self.head_dim}")
        if self.n_heads < 1:
            raise ValueError(f"n_heads must be >= 1; got {self.n_heads}")
        if self.n_fit_samples < self.n_codebook:
            raise ValueError(
                f"n_fit_samples ({self.n_fit_samples}) must be >= "
                f"n_codebook ({self.n_codebook})"
            )


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


@dataclass
class CodecStats:
    """Accumulated statistics for :class:`KVCodec`.

    Attributes:
        n_fit_calls: Number of times :meth:`KVCodec.fit` has been called.
        n_encode_calls: Total encode calls (keys and values combined).
        total_encoded_tokens: Total per-head token vectors encoded across all
            calls (``n_heads * seq_len`` per call).
    """

    n_fit_calls: int = 0
    n_encode_calls: int = 0
    total_encoded_tokens: int = 0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _pairwise_sq_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute pairwise squared Euclidean distances.

    Args:
        a: Float32 array of shape ``(n, d)``.
        b: Float32 array of shape ``(k, d)``.

    Returns:
        Float32 array of shape ``(n, k)`` where ``result[i, j]`` is
        ``||a[i] - b[j]||²``.
    """
    sq_a = np.sum(a ** 2, axis=1, keepdims=True)   # (n, 1)
    sq_b = np.sum(b ** 2, axis=1, keepdims=True)   # (k, 1)
    return sq_a + sq_b.T - 2.0 * (a @ b.T)        # (n, k)


def _lloyd_kmeans(
    data: np.ndarray,
    n_clusters: int,
    rng: np.random.Generator,
    max_iters: int = _MAX_LLOYD_ITERS,
) -> np.ndarray:
    """Fit k-means centroids via Lloyd's algorithm with k-means++ seeding.

    Args:
        data: Float32 array of shape ``(n_samples, dim)``.
        n_clusters: Number of cluster centroids to fit.
        rng: NumPy random generator used for centroid initialisation and
            empty-cluster reseeding.
        max_iters: Maximum number of Lloyd iterations.  Stops earlier if
            centroids converge (absolute tolerance ``1e-6``).

    Returns:
        Centroid matrix, shape ``(n_clusters, dim)`` float32.
    """
    n_samples, dim = data.shape
    if n_samples < n_clusters:
        raise ValueError(
            f"Need at least {n_clusters} samples to fit {n_clusters} centroids; "
            f"got {n_samples}"
        )

    # k-means++ initialisation: choose centroids with probability proportional
    # to squared distance from the nearest already-chosen centroid.
    first_idx = int(rng.integers(0, n_samples))
    centroids: list[np.ndarray] = [data[first_idx].copy()]

    for _ in range(1, n_clusters):
        centroid_mat = np.array(centroids, dtype=np.float32)  # (i, dim)
        dists_sq = _pairwise_sq_dist(data, centroid_mat)       # (n, i)
        min_dists = dists_sq.min(axis=1)                       # (n,)
        min_dists = np.maximum(min_dists, 0.0)                # guard against fp rounding
        probs = min_dists / (min_dists.sum() + 1e-30)
        chosen = int(rng.choice(n_samples, p=probs))
        centroids.append(data[chosen].copy())

    centroids_arr = np.array(centroids, dtype=np.float32)  # (k, dim)

    for _ in range(max_iters):
        # Assignment step: find nearest centroid for each sample
        dists_sq = _pairwise_sq_dist(data, centroids_arr)  # (n, k)
        assignments = np.argmin(dists_sq, axis=1)          # (n,)

        # Update step: recompute centroids as cluster means
        new_centroids = np.zeros_like(centroids_arr)
        counts = np.bincount(assignments, minlength=n_clusters)
        np.add.at(new_centroids, assignments, data)

        # Normalise non-empty clusters; reinitialise empty ones
        for j in range(n_clusters):
            if counts[j] > 0:
                new_centroids[j] /= counts[j]
            else:
                # Reinitialise dead centroid to a random data point
                new_centroids[j] = data[int(rng.integers(0, n_samples))].copy()

        if np.allclose(new_centroids, centroids_arr, atol=1e-6):
            break
        centroids_arr = new_centroids.astype(np.float32)

    return centroids_arr


# ---------------------------------------------------------------------------
# Codec
# ---------------------------------------------------------------------------


class KVCodec:
    """Learned key/value codec backed by a fitted codebook.

    Call :meth:`fit` once with representative KV samples, then use
    :meth:`encode_keys` / :meth:`decode_keys` (and the corresponding value
    methods) during inference.

    The key and value codebooks are fitted independently, allowing the codec
    to capture the different distributional properties of keys vs. values.

    Args:
        config: :class:`CodecConfig` instance.
        rng: Optional NumPy random generator for reproducible fitting.
            Defaults to ``np.random.default_rng(42)``.
    """

    def __init__(
        self,
        config: CodecConfig,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.config = config
        self._rng = rng if rng is not None else np.random.default_rng(42)
        self._key_codebook: Optional[np.ndarray] = None  # (n_codebook, head_dim)
        self._val_codebook: Optional[np.ndarray] = None  # (n_codebook, head_dim)
        self._stats = CodecStats()

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, keys: np.ndarray, values: np.ndarray) -> None:
        """Fit the key and value codebooks using Lloyd's k-means algorithm.

        Args:
            keys: Representative key vectors, shape ``(n, head_dim)`` float32.
                Must have at least ``config.n_codebook`` rows.
            values: Representative value vectors, shape ``(n, head_dim)``
                float32.  Must have the same first dimension as ``keys``.

        Raises:
            ValueError: If shapes are incorrect or there are too few samples.
        """
        cfg = self.config
        if keys.ndim != 2 or keys.shape[1] != cfg.head_dim:
            raise ValueError(
                f"keys must have shape (n, {cfg.head_dim}); got {keys.shape}"
            )
        if values.ndim != 2 or values.shape[1] != cfg.head_dim:
            raise ValueError(
                f"values must have shape (n, {cfg.head_dim}); got {values.shape}"
            )
        if keys.shape[0] != values.shape[0]:
            raise ValueError(
                f"keys and values must have the same number of rows; "
                f"got {keys.shape[0]} vs {values.shape[0]}"
            )

        n = keys.shape[0]
        if n < cfg.n_codebook:
            raise ValueError(
                f"Need at least {cfg.n_codebook} samples to fit codebook; "
                f"got {n}"
            )

        # Subsample if more data than requested
        if n > cfg.n_fit_samples:
            idx = self._rng.choice(n, size=cfg.n_fit_samples, replace=False)
            keys = keys[idx]
            values = values[idx]

        self._key_codebook = _lloyd_kmeans(
            keys.astype(np.float32), cfg.n_codebook, self._rng
        )
        self._val_codebook = _lloyd_kmeans(
            values.astype(np.float32), cfg.n_codebook, self._rng
        )
        self._stats.n_fit_calls += 1

    # ------------------------------------------------------------------
    # Internal encode helper
    # ------------------------------------------------------------------

    def _encode(self, vectors: np.ndarray, codebook: np.ndarray) -> np.ndarray:
        """Encode a batch of vectors to nearest codebook indices.

        Args:
            vectors: Float32 array of shape ``(n, head_dim)``.
            codebook: Float32 array of shape ``(n_codebook, head_dim)``.

        Returns:
            Int32 index array of shape ``(n,)``.
        """
        dists_sq = _pairwise_sq_dist(vectors, codebook)  # (n, k)
        return np.argmin(dists_sq, axis=1).astype(np.int32)

    # ------------------------------------------------------------------
    # Keys
    # ------------------------------------------------------------------

    def encode_keys(self, keys: np.ndarray) -> np.ndarray:
        """Encode key vectors to codebook indices.

        Args:
            keys: Float32 array of shape ``(n_heads, seq_len, head_dim)``.

        Returns:
            Int32 index array of shape ``(n_heads, seq_len)``.

        Raises:
            RuntimeError: If the codec has not been fitted yet.
            ValueError: If ``keys`` has an unexpected shape.
        """
        if not self.is_fitted:
            raise RuntimeError("Codec is not fitted; call fit() first.")
        cfg = self.config
        if (
            keys.ndim != 3
            or keys.shape[0] != cfg.n_heads
            or keys.shape[2] != cfg.head_dim
        ):
            raise ValueError(
                f"keys must have shape ({cfg.n_heads}, seq_len, {cfg.head_dim}); "
                f"got {keys.shape}"
            )

        n_heads, seq_len, _ = keys.shape
        indices = np.zeros((n_heads, seq_len), dtype=np.int32)
        assert self._key_codebook is not None
        for h in range(n_heads):
            indices[h] = self._encode(keys[h].astype(np.float32), self._key_codebook)

        self._stats.n_encode_calls += 1
        self._stats.total_encoded_tokens += n_heads * seq_len
        return indices

    def decode_keys(self, indices: np.ndarray, head_idx: int) -> np.ndarray:
        """Reconstruct key vectors from codebook indices for a single head.

        Args:
            indices: Int32 array of shape ``(seq_len,)``.
            head_idx: Index of the attention head.  Currently unused
                structurally (codebook is shared across heads after fit),
                but validated for API correctness.

        Returns:
            Reconstructed key vectors, shape ``(seq_len, head_dim)`` float32.

        Raises:
            RuntimeError: If the codec has not been fitted yet.
            ValueError: If ``indices`` is not 1-D or ``head_idx`` is invalid.
        """
        if not self.is_fitted:
            raise RuntimeError("Codec is not fitted; call fit() first.")
        if indices.ndim != 1:
            raise ValueError(f"indices must be 1-D; got shape {indices.shape}")
        if head_idx < 0 or head_idx >= self.config.n_heads:
            raise ValueError(
                f"head_idx must be in [0, {self.config.n_heads}); got {head_idx}"
            )
        assert self._key_codebook is not None
        return self._key_codebook[indices].astype(np.float32)

    # ------------------------------------------------------------------
    # Values
    # ------------------------------------------------------------------

    def encode_values(self, values: np.ndarray) -> np.ndarray:
        """Encode value vectors to codebook indices.

        Args:
            values: Float32 array of shape ``(n_heads, seq_len, head_dim)``.

        Returns:
            Int32 index array of shape ``(n_heads, seq_len)``.

        Raises:
            RuntimeError: If the codec has not been fitted yet.
            ValueError: If ``values`` has an unexpected shape.
        """
        if not self.is_fitted:
            raise RuntimeError("Codec is not fitted; call fit() first.")
        cfg = self.config
        if (
            values.ndim != 3
            or values.shape[0] != cfg.n_heads
            or values.shape[2] != cfg.head_dim
        ):
            raise ValueError(
                f"values must have shape ({cfg.n_heads}, seq_len, {cfg.head_dim}); "
                f"got {values.shape}"
            )

        n_heads, seq_len, _ = values.shape
        indices = np.zeros((n_heads, seq_len), dtype=np.int32)
        assert self._val_codebook is not None
        for h in range(n_heads):
            indices[h] = self._encode(values[h].astype(np.float32), self._val_codebook)

        self._stats.n_encode_calls += 1
        self._stats.total_encoded_tokens += n_heads * seq_len
        return indices

    def decode_values(self, indices: np.ndarray, head_idx: int) -> np.ndarray:
        """Reconstruct value vectors from codebook indices for a single head.

        Args:
            indices: Int32 array of shape ``(seq_len,)``.
            head_idx: Index of the attention head.

        Returns:
            Reconstructed value vectors, shape ``(seq_len, head_dim)`` float32.

        Raises:
            RuntimeError: If the codec has not been fitted yet.
            ValueError: If ``indices`` is not 1-D or ``head_idx`` is invalid.
        """
        if not self.is_fitted:
            raise RuntimeError("Codec is not fitted; call fit() first.")
        if indices.ndim != 1:
            raise ValueError(f"indices must be 1-D; got shape {indices.shape}")
        if head_idx < 0 or head_idx >= self.config.n_heads:
            raise ValueError(
                f"head_idx must be in [0, {self.config.n_heads}); got {head_idx}"
            )
        assert self._val_codebook is not None
        return self._val_codebook[indices].astype(np.float32)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_fitted(self) -> bool:
        """``True`` if :meth:`fit` has been called at least once."""
        return self._key_codebook is not None and self._val_codebook is not None

    @property
    def compression_ratio(self) -> float:
        """Compression ratio: ``32 * head_dim / log2(n_codebook)``.

        Represents how many original float32 bits are represented by a single
        codebook index bit.  E.g. ``head_dim=64``, ``n_codebook=256`` →
        ``256×`` compression.
        """
        cfg = self.config
        bits_original = 32.0 * cfg.head_dim
        bits_encoded = math.log2(cfg.n_codebook)
        return bits_original / bits_encoded

    @property
    def stats(self) -> CodecStats:
        """Current accumulated :class:`CodecStats`."""
        return self._stats
