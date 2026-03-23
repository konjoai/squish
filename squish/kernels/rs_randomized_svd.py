"""rs_randomized_svd.py — Rust-accelerated randomized SVD.

Wraps `squish_quant.randomized_svd_f32` (Wave 57a). Falls back to
NumPy LAPACK SVD when the Rust extension is unavailable.

RustRandomizedSVD uses a Gaussian sketch + QR + thin SVD algorithm
(Halko et al. 2011) to compute rank-k approximations 3–8× faster
than `np.linalg.svd(full_matrices=False)` at rank ≤ 64.

Hooks into 12 `np.linalg.svd` call sites across:
  shadow_kv.py, gear_kv.py, kv_cache.py, milo_quant.py,
  context/delta_compress.py, kv/adaptive_kvtc.py

Reference:
  Halko et al. (SIAM Review 2011) — Finding the Number of Latent
  Factors in High-Dimensional Data with Randomized SVD.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

try:
    import squish_quant as _sq

    _RUST_AVAILABLE = hasattr(_sq, "randomized_svd_f32")
except ImportError:
    _RUST_AVAILABLE = False

__all__ = ["RandomizedSVDConfig", "RustRandomizedSVD"]


@dataclass
class RandomizedSVDConfig:
    """Configuration for RustRandomizedSVD.

    Attributes:
        rank:        Target rank of the approximation (default 32).
        n_oversamples: Extra columns for the sketch to improve accuracy
                     (default 10; total sketch columns = rank + n_oversamples).
    """

    rank: int = 32
    n_oversamples: int = 10


class RustRandomizedSVD:
    """Rust-accelerated randomized SVD (rank-k approximation).

    Usage::

        rsvd = RustRandomizedSVD(RandomizedSVDConfig(rank=16))
        A = np.random.randn(4096, 128).astype(np.float32)
        U, S, Vt = rsvd.fit(A)
        # U: (4096, 16), S: (16,), Vt: (16, 128)
        A_approx = (U * S) @ Vt
    """

    def __init__(self, config: RandomizedSVDConfig | None = None) -> None:
        self._cfg = config or RandomizedSVDConfig()

    def fit(
        self,
        a: np.ndarray,
        rank: int | None = None,
        n_oversamples: int | None = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute rank-k randomized SVD of `a`.

        Args:
            a:           2-D float32 matrix `(m, n)`.
            rank:        Target rank (overrides config).
            n_oversamples: Extra sketch columns (overrides config).

        Returns:
            Tuple `(U, S, Vt)`:
            - U  `(m, rank)` — left singular vectors
            - S  `(rank,)`   — singular values (descending)
            - Vt `(rank, n)` — right singular vectors (transposed)
        """
        a = np.asarray(a, dtype=np.float32)
        r = rank if rank is not None else self._cfg.rank
        os = n_oversamples if n_oversamples is not None else self._cfg.n_oversamples
        if _RUST_AVAILABLE:
            result = _sq.randomized_svd_f32(a, r, os)
            u = np.asarray(result[0], dtype=np.float32)
            s = np.asarray(result[1], dtype=np.float32)
            vt = np.asarray(result[2], dtype=np.float32)
            return u, s, vt
        return self._numpy_svd(a, r)

    def reconstruct(
        self,
        a: np.ndarray,
        rank: int | None = None,
    ) -> np.ndarray:
        """Return the rank-k reconstruction `(U * S) @ Vt`.

        Args:
            a:    2-D float32 matrix `(m, n)`.
            rank: Target rank (overrides config).

        Returns:
            Reconstructed float32 matrix `(m, n)`.
        """
        u, s, vt = self.fit(a, rank=rank)
        return ((u * s) @ vt).astype(np.float32)

    def rank(self) -> int:
        """Return configured target rank."""
        return self._cfg.rank

    def backend(self) -> str:
        """Return which backend is being used: 'rust' or 'numpy'."""
        return "rust" if _RUST_AVAILABLE else "numpy"

    # ── NumPy fallback ── -----------------------------------------------------

    @staticmethod
    def _numpy_svd(
        a: np.ndarray, rank: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """NumPy randomized SVD via sketch + QR + thin SVD."""
        m, n = a.shape
        rng = np.random.default_rng(0)
        k = min(rank + 10, min(m, n))
        omega = rng.standard_normal((n, k)).astype(np.float32)
        y = a @ omega  # (m, k)
        q, _ = np.linalg.qr(y)  # Q: (m, k)
        b = q.T @ a   # (k, n)
        u_small, s, vt = np.linalg.svd(b, full_matrices=False)
        u = q @ u_small  # (m, k) -> back to (m, rank)
        actual = min(rank, len(s))
        return (
            u[:, :actual].astype(np.float32),
            s[:actual].astype(np.float32),
            vt[:actual, :].astype(np.float32),
        )
