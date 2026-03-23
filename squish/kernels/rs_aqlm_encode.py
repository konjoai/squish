"""squish/kernels/rs_aqlm_encode.py — Rust-backed AQLM multi-codebook encoding.

Wraps ``squish_quant_rs.aqlm_encode_f32`` and ``squish_quant_rs.aqlm_kmeans_f32``
with NumPy fallbacks.

AQLM (Additive Quantisation of Language Models) compresses weight matrices by
greedily finding the nearest entry in each of multiple codebooks and subtracting
the matched codeword from a running residual (codebook peeling).  The codebooks
are learned via K-means++ initialised Lloyd clustering on the weight sub-vectors.

Reference: Egiazarian et al., "Extreme Compression of Large Language Models via
Additive Quantization," arXiv 2401.06118, 2024.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

__all__ = [
    "AQLMEncodeConfig",
    "RustAQLMEncode",
]

try:
    import squish_quant as _sq
    _HAS_ENCODE = hasattr(_sq, "aqlm_encode_f32")
    _HAS_KMEANS = hasattr(_sq, "aqlm_kmeans_f32")
    _HAS_RUST = _HAS_ENCODE and _HAS_KMEANS
except ImportError:
    _sq = None  # type: ignore[assignment]
    _HAS_ENCODE = _HAS_KMEANS = _HAS_RUST = False


# ── NumPy fallbacks ───────────────────────────────────────────────────────────


def _numpy_kmeans(
    vecs: np.ndarray,
    k: int,
    n_iter: int,
    seed: int,
) -> np.ndarray:
    """K-means++ init + Lloyd for codebook construction.

    Args:
        vecs:   ``(N, gs)`` float32 vectors.
        k:      Number of centroids.
        n_iter: Lloyd iterations.
        seed:   Random seed.

    Returns:
        ``(k, gs)`` float32 centroids.
    """
    rng = np.random.default_rng(seed)
    n, gs = vecs.shape
    k = min(k, n)
    # K-means++ init
    idx = [int(rng.integers(n))]
    for _ in range(1, k):
        diffs = vecs[:, None, :] - vecs[idx][None, :, :]  # (N, len(idx), gs)
        dists = (diffs ** 2).sum(axis=-1).min(axis=1)     # (N,)
        probs = dists / dists.sum()
        idx.append(int(rng.choice(n, p=probs)))
    centroids = vecs[idx].copy().astype(np.float32)
    for _ in range(n_iter):
        diffs = vecs[:, None, :] - centroids[None, :, :]
        assignments = (diffs ** 2).sum(axis=-1).argmin(axis=1)
        for c in range(k):
            mask = assignments == c
            centroids[c] = vecs[mask].mean(axis=0) if mask.any() else vecs[rng.integers(n)]
    return centroids.astype(np.float32)


def _numpy_encode(
    residuals: np.ndarray,
    codebook: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Greedy nearest-codebook-entry lookup + residual subtract.

    Args:
        residuals: ``(out, n_groups, gs)`` float32.
        codebook:  ``(CB, gs)`` float32.

    Returns:
        Tuple of indices ``(out, n_groups)`` uint16,
        updated residuals ``(out, n_groups, gs)`` float32.
    """
    res = residuals.copy().astype(np.float32)
    out_f, n_g, gs = res.shape
    indices = np.zeros((out_f, n_g), dtype=np.uint16)
    for i in range(out_f):
        for g in range(n_g):
            diff = res[i, g, :][None, :] - codebook  # (CB, gs)
            dists = (diff ** 2).sum(axis=-1)
            best = int(dists.argmin())
            indices[i, g] = best
            res[i, g, :] -= codebook[best]
    return indices, res


# ── Config / wrapper ──────────────────────────────────────────────────────────


@dataclass
class AQLMEncodeConfig:
    """Configuration for :class:`RustAQLMEncode`.

    Attributes:
        n_iter: K-means Lloyd iterations for codebook training.
        seed:   Random seed.
    """

    n_iter: int = 50
    seed: int = 42


class RustAQLMEncode:
    """Rust-accelerated AQLM multi-codebook encoding.

    Learns a ``(CB, gs)`` codebook from weight sub-vectors via K-means++
    Lloyd clustering, then encodes weight matrices by greedy codebook peeling
    (nearest-entry lookup + residual subtract per group).  Falls back to NumPy
    when ``squish_quant_rs`` is unavailable.

    Example::

        enc = RustAQLMEncode()
        codebook = enc.fit_codebook(vecs, k=256)
        indices, residuals = enc.encode(residuals=w, codebook=codebook)
    """

    def __init__(self, config: Optional[AQLMEncodeConfig] = None) -> None:
        self._cfg = config or AQLMEncodeConfig()

    def fit_codebook(
        self,
        vecs: np.ndarray,
        k: int,
        n_iter: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """Train a K-means codebook on sub-vectors.

        Args:
            vecs:   ``(N, gs)`` float32 sub-vectors.
            k:      Number of codebook entries.
            n_iter: Lloyd iterations (overrides config).
            seed:   Random seed (overrides config).

        Returns:
            ``(k, gs)`` float32 codebook centroids.

        Raises:
            ValueError: If ``vecs`` is not 2-D or ``k < 1``.
        """
        v = np.ascontiguousarray(vecs, dtype=np.float32)
        if v.ndim != 2:
            raise ValueError(f"vecs must be 2-D (N, gs), got {v.shape}")
        if k < 1:
            raise ValueError(f"k must be ≥ 1, got {k}")
        ni = int(n_iter) if n_iter is not None else self._cfg.n_iter
        sd = int(seed) if seed is not None else self._cfg.seed
        if _HAS_KMEANS:
            return np.asarray(_sq.aqlm_kmeans_f32(v, min(k, len(v)), ni, sd), dtype=np.float32)
        return _numpy_kmeans(v, k, ni, sd)

    def encode(
        self,
        residuals: np.ndarray,
        codebook: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Encode residuals using greedy codebook peeling.

        Args:
            residuals: ``(out, n_groups, gs)`` float32 residual tensor.
            codebook:  ``(CB, gs)`` float32 codebook.

        Returns:
            Tuple of:
            - ``indices``: ``(out, n_groups)`` uint16 codebook indices.
            - ``updated_residuals``: ``(out, n_groups, gs)`` float32.

        Raises:
            ValueError: If array shapes are incompatible.
        """
        r = np.ascontiguousarray(residuals, dtype=np.float32)
        cb = np.ascontiguousarray(codebook, dtype=np.float32)
        if r.ndim != 3:
            raise ValueError(f"residuals must be 3-D (out, n_groups, gs), got {r.shape}")
        if cb.ndim != 2:
            raise ValueError(f"codebook must be 2-D (CB, gs), got {cb.shape}")
        if r.shape[2] != cb.shape[1]:
            raise ValueError(f"group size mismatch: residuals gs={r.shape[2]}, codebook gs={cb.shape[1]}")
        if _HAS_ENCODE:
            idx, upd = _sq.aqlm_encode_f32(r, cb)
            return np.asarray(idx, dtype=np.uint16), np.asarray(upd, dtype=np.float32)
        return _numpy_encode(r, cb)

    def decode(
        self,
        indices: np.ndarray,
        codebook: np.ndarray,
    ) -> np.ndarray:
        """Reconstruct weight tensor from indices and codebook.

        Args:
            indices:  ``(out, n_groups)`` integer codebook indices.
            codebook: ``(CB, gs)`` float32 codebook.

        Returns:
            ``(out, n_groups, gs)`` float32 reconstructed tensor.
        """
        idx = np.asarray(indices)
        cb = np.asarray(codebook, dtype=np.float32)
        return cb[idx.astype(np.int64)]  # (out, n_groups, gs)

    def backend(self) -> str:
        return "rust" if _HAS_RUST else "numpy"
