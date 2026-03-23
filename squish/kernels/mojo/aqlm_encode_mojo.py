"""squish/kernels/mojo/aqlm_encode_mojo.py — Mojo-backed AQLM multi-codebook encoding.

Wraps the ``aqlm_encode_kernel`` Mojo stub via MojoBridge with a NumPy fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from squish.kernels.mojo.mojo_bridge import MojoBridge

__all__ = [
    "AQLMEncodeMojoConfig",
    "MojoAQLMEncode",
]

_bridge = MojoBridge()
_encode_kernel = _bridge.load_kernel("aqlm_encode_kernel")
_kmeans_kernel = _bridge.load_kernel("aqlm_kmeans_kernel")


# ── NumPy fallbacks ───────────────────────────────────────────────────────────


def _numpy_kmeans(vecs: np.ndarray, k: int, n_iter: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n, gs = vecs.shape
    k = min(k, n)
    idx = rng.choice(n, size=k, replace=False)
    centroids = vecs[idx].copy().astype(np.float32)
    for _ in range(n_iter):
        diffs = vecs[:, None, :] - centroids[None, :, :]
        assignments = (diffs ** 2).sum(axis=-1).argmin(axis=1)
        for c in range(k):
            mask = assignments == c
            centroids[c] = vecs[mask].mean(axis=0) if mask.any() else vecs[rng.integers(n)]
    return centroids.astype(np.float32)


def _numpy_encode(
    residuals: np.ndarray, codebook: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    res = residuals.copy().astype(np.float32)
    out_f, n_g, gs = res.shape
    indices = np.zeros((out_f, n_g), dtype=np.uint16)
    for i in range(out_f):
        for g in range(n_g):
            diff = res[i, g, :][None, :] - codebook
            best = int((diff ** 2).sum(axis=-1).argmin())
            indices[i, g] = best
            res[i, g, :] -= codebook[best]
    return indices, res


# ── Config / wrapper ──────────────────────────────────────────────────────────


@dataclass
class AQLMEncodeMojoConfig:
    """Configuration for :class:`MojoAQLMEncode`.

    Attributes:
        n_iter: K-means Lloyd iterations.
        seed:   Random seed.
    """

    n_iter: int = 50
    seed: int = 42


class MojoAQLMEncode:
    """Mojo-backed AQLM multi-codebook encoder.

    Falls back to NumPy when the Mojo runtime is absent.

    Example::

        enc = MojoAQLMEncode()
        codebook = enc.fit_codebook(vecs, k=256)
        indices, residuals = enc.encode(residuals=w, codebook=codebook)
    """

    def __init__(self, config: Optional[AQLMEncodeMojoConfig] = None) -> None:
        self._cfg = config or AQLMEncodeMojoConfig()

    def fit_codebook(
        self,
        vecs: np.ndarray,
        k: int,
        n_iter: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """Train a K-means codebook on sub-vectors.

        Args:
            vecs:   ``(N, gs)`` float32 vectors.
            k:      Number of codebook entries.
            n_iter: Lloyd iterations (overrides config).
            seed:   Random seed (overrides config).

        Returns:
            ``(k, gs)`` float32 codebook centroids.
        """
        v = np.ascontiguousarray(vecs, dtype=np.float32)
        ni = int(n_iter) if n_iter is not None else self._cfg.n_iter
        sd = int(seed) if seed is not None else self._cfg.seed
        # Mojo kernel path (stub — not yet compiled)
        if _kmeans_kernel is not None:
            out = np.zeros((min(k, len(v)), v.shape[1]), dtype=np.float32)
            _kmeans_kernel(
                v.ctypes.data, out.ctypes.data, len(v), v.shape[1], min(k, len(v)), ni, sd
            )
            return out
        return _numpy_kmeans(v, k, ni, sd)

    def encode(
        self,
        residuals: np.ndarray,
        codebook: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Encode via greedy codebook peeling.

        Args:
            residuals: ``(out, n_groups, gs)`` float32.
            codebook:  ``(CB, gs)`` float32.

        Returns:
            Tuple of indices ``(out, n_groups)`` uint16,
            updated residuals ``(out, n_groups, gs)`` float32.
        """
        r = np.ascontiguousarray(residuals, dtype=np.float32)
        cb = np.ascontiguousarray(codebook, dtype=np.float32)
        if _encode_kernel is not None:
            out_f, n_g, gs = r.shape
            indices = np.zeros((out_f, n_g), dtype=np.uint16)
            upd = r.copy()
            _encode_kernel(
                r.ctypes.data, cb.ctypes.data, indices.ctypes.data, upd.ctypes.data,
                out_f, n_g, gs, cb.shape[0],
            )
            return indices, upd
        return _numpy_encode(r, cb)

    def decode(self, indices: np.ndarray, codebook: np.ndarray) -> np.ndarray:
        """Reconstruct weight tensor from indices and codebook.

        Args:
            indices:  ``(out, n_groups)`` integer indices.
            codebook: ``(CB, gs)`` float32.

        Returns:
            ``(out, n_groups, gs)`` float32.
        """
        return np.asarray(codebook, dtype=np.float32)[np.asarray(indices).astype(np.int64)]

    def backend(self) -> str:
        return "mojo" if (_encode_kernel is not None and _kmeans_kernel is not None) else "numpy"
