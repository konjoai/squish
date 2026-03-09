"""
squish/paris_kv.py

ParisKV — Drift-Robust Online KV-Cache Quantization via Centroid Refresh.

Inspired by:
  "ParisKV: Drift-Free Product Quantization for KV Caches"
  (Technical report, 2025)

Problem
-------
Vector quantization codebooks (e.g. CommVQ) are trained offline on calibration
data, but the actual KV distribution shifts (drifts) during inference—
especially across conversation turns and domain changes.  Stale centroids raise
quantization error, degrading attention-pattern fidelity.

ParisKV Solution
----------------
Run a lightweight *online* centroid update alongside the normal decode loop:

  1. For each new KV batch, compute per-centroid assignment counts and
     accumulative residuals from the current codebook.
  2. Apply an exponential-moving-average (EMA) centroid refresh:
         c_k ← (1 - lr) * c_k  +  lr * mean({x : argmin_j ||x - c_j|| == k})
  3. Refresh only centroids that received ≥ ``min_count`` assignments (protects
     rarely-used centroids from random noise).
  4. The drift score (mean quantization error delta) is tracked so callers can
     trigger a full re-fit when needed.

This module provides:
  * ``ParisKVConfig``
  * ``ParisKVCodebook`` — online-refineable codebook (extends CommVQ logic)
  * ``ema_update_centroids()`` — pure function for the EMA step

Usage::

    from squish.paris_kv import ParisKVCodebook, ParisKVConfig

    codebook = ParisKVCodebook(dim=128, n_codes=16, config=ParisKVConfig())
    codebook.fit(calibration_vecs)                # initial offline fit

    for kv_batch in stream:
        indices = codebook.encode(kv_batch)
        codebook.online_update(kv_batch)          # EMA centroid refresh
        approx  = codebook.decode(indices)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "ParisKVConfig",
    "ParisKVCodebook",
    "ema_update_centroids",
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ParisKVConfig:
    """
    Configuration for ParisKV online centroid refresh.

    Parameters
    ----------
    learning_rate : float
        EMA step size for centroid refresh.  0 = no online update (frozen
        codebook); 1 = full replacement each step.
        Typical range: 0.01 – 0.1.
    min_count : int
        Minimum number of vectors assigned to a centroid in the current batch
        for the centroid to be updated.  Prevents noise-driven drift of
        rarely-used codes.
    drift_window : int
        Number of recent per-step mean-squared-errors to retain for drift
        scoring.
    refine_iters : int
        Number of k-means iterations in the initial offline ``fit()`` pass.
    """
    learning_rate: float = 0.05
    min_count:     int   = 2
    drift_window:  int   = 50
    refine_iters:  int   = 20

    def __post_init__(self) -> None:
        if not 0.0 <= self.learning_rate <= 1.0:
            raise ValueError("learning_rate must be in [0, 1]")
        if self.min_count < 1:
            raise ValueError("min_count must be ≥ 1")
        if self.drift_window < 1:
            raise ValueError("drift_window must be ≥ 1")
        if self.refine_iters < 1:
            raise ValueError("refine_iters must be ≥ 1")


# ---------------------------------------------------------------------------
# EMA update (pure function)
# ---------------------------------------------------------------------------

def ema_update_centroids(
    centroids:     np.ndarray,
    vectors:       np.ndarray,
    assignments:   np.ndarray,
    learning_rate: float = 0.05,
    min_count:     int   = 2,
) -> np.ndarray:
    """
    Apply one EMA centroid refresh step.

    Parameters
    ----------
    centroids    : (K, D) float32 — current codebook centroids
    vectors      : (N, D) float32 — new (dequantized or raw) KV vectors
    assignments  : (N,)   int     — centroid index for each vector
    learning_rate : float — EMA weight for incoming mean
    min_count    : int   — skip update for centroids with fewer assignments

    Returns
    -------
    updated_centroids : (K, D) float32 (new array; originals not mutated)
    """
    centroids  = np.asarray(centroids, dtype=np.float32)
    vectors    = np.asarray(vectors,   dtype=np.float32)
    K          = centroids.shape[0]
    new_c      = centroids.copy()

    for k_idx in range(K):
        mask  = assignments == k_idx
        count = int(mask.sum())
        if count < min_count:
            continue
        mean_vec     = vectors[mask].mean(axis=0)
        new_c[k_idx] = (1.0 - learning_rate) * new_c[k_idx] + learning_rate * mean_vec

    return new_c


# ---------------------------------------------------------------------------
# ParisKV Codebook
# ---------------------------------------------------------------------------

class ParisKVCodebook:
    """
    Vector-quantization codebook with online EMA centroid refresh.

    Parameters
    ----------
    dim     : int — vector dimension
    n_codes : int — number of centroid codes (K)
    config  : ParisKVConfig
    """

    def __init__(
        self,
        dim:     int,
        n_codes: int  = 16,
        config:  ParisKVConfig | None = None,
    ) -> None:
        if dim < 1:
            raise ValueError("dim must be ≥ 1")
        if n_codes < 1:
            raise ValueError("n_codes must be ≥ 1")
        self._dim      = dim
        self._n_codes  = n_codes
        self._cfg      = config or ParisKVConfig()
        self._centroids: np.ndarray | None = None   # (K, D) float32
        self._is_fitted = False

        # Drift tracking
        self._drift_history: list[float] = []

    # ── Fitting ───────────────────────────────────────────────────────────────

    def fit(
        self,
        vectors: np.ndarray,
        seed:    int = 42,
    ) -> ParisKVCodebook:
        """
        Fit the initial codebook via k-means.

        Parameters
        ----------
        vectors : (N, D) float32 — calibration vectors (N ≥ K)
        seed    : int — RNG seed

        Returns
        -------
        self (for method chaining)
        """
        vecs = np.asarray(vectors, dtype=np.float32)
        if vecs.ndim != 2 or vecs.shape[1] != self._dim:
            raise ValueError(
                f"Expected (N, {self._dim}) array, got {vecs.shape}"
            )
        K   = self._n_codes
        rng = np.random.default_rng(seed)

        # k-means++ init
        centroids = _kmeans_plus_plus(vecs, K, rng)

        for _ in range(self._cfg.refine_iters):
            dists       = _pairwise_sq_dist(vecs, centroids)    # (N, K)
            assignments = dists.argmin(axis=1)                  # (N,)
            new_c       = np.zeros_like(centroids)
            for k_idx in range(K):
                mask = assignments == k_idx
                if mask.sum() > 0:
                    new_c[k_idx] = vecs[mask].mean(axis=0)
                else:
                    new_c[k_idx] = centroids[k_idx]
            if np.allclose(centroids, new_c, atol=1e-6):
                break
            centroids = new_c

        self._centroids = centroids
        self._is_fitted = True
        return self

    # ── Encode / Decode ───────────────────────────────────────────────────────

    def encode(self, vectors: np.ndarray) -> np.ndarray:
        """
        Quantize vectors to centroid indices.

        Parameters
        ----------
        vectors : (N, D) float32

        Returns
        -------
        indices : (N,) uint16
        """
        self._require_fitted()
        vecs  = np.asarray(vectors, dtype=np.float32)
        dists = _pairwise_sq_dist(vecs, self._centroids)
        return dists.argmin(axis=1).astype(np.uint16)

    def decode(self, indices: np.ndarray) -> np.ndarray:
        """
        Dequantize indices back to float32 centroids.

        Parameters
        ----------
        indices : (N,) int or uint16

        Returns
        -------
        vectors : (N, D) float32
        """
        self._require_fitted()
        idx = np.asarray(indices, dtype=np.int64).flatten()
        return self._centroids[idx].astype(np.float32)

    # ── Online Update ─────────────────────────────────────────────────────────

    def online_update(
        self,
        vectors: np.ndarray,
    ) -> float:
        """
        Update centroids in-place via EMA using the incoming vector batch.

        Parameters
        ----------
        vectors : (N, D) float32 — new KV vectors (raw, float)

        Returns
        -------
        float — mean quantization error *before* the update (for drift tracking)
        """
        self._require_fitted()
        vecs        = np.asarray(vectors, dtype=np.float32)
        assignments = self.encode(vecs).astype(np.int64)

        # Compute pre-update error
        reconstructed = self.decode(assignments)
        pre_error     = float(np.mean((vecs - reconstructed) ** 2))

        # EMA update
        self._centroids = ema_update_centroids(
            centroids     = self._centroids,
            vectors       = vecs,
            assignments   = assignments,
            learning_rate = self._cfg.learning_rate,
            min_count     = self._cfg.min_count,
        )

        # Track drift
        self._drift_history.append(pre_error)
        if len(self._drift_history) > self._cfg.drift_window:
            self._drift_history.pop(0)

        return pre_error

    # ── Drift Metrics ─────────────────────────────────────────────────────────

    @property
    def drift_score(self) -> float:
        """
        Mean quantization error over the recent drift window.
        Returns 0.0 if no history yet.
        """
        if not self._drift_history:
            return 0.0
        return float(np.mean(self._drift_history))

    @property
    def quantization_error(self) -> float:
        """Alias for the most recent per-batch error (last drift_history entry)."""
        if not self._drift_history:
            return 0.0
        return self._drift_history[-1]

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    @property
    def centroids(self) -> np.ndarray | None:
        """(K, D) float32 centroid array, or None if not fitted."""
        return self._centroids

    @property
    def n_codes(self) -> int:
        return self._n_codes

    @property
    def dim(self) -> int:
        return self._dim

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _require_fitted(self) -> None:
        if not self._is_fitted or self._centroids is None:
            raise RuntimeError("Codebook is not fitted yet. Call .fit() first.")


# ---------------------------------------------------------------------------
# K-means helpers
# ---------------------------------------------------------------------------

def _kmeans_plus_plus(
    vecs: np.ndarray,
    K:    int,
    rng:  np.random.Generator,
) -> np.ndarray:
    """K-means++ initialisation.  Returns (K, D) centroid array."""
    N = vecs.shape[0]
    chosen = [int(rng.integers(0, N))]
    for _ in range(1, K):
        dists = np.array([
            min(float(np.sum((vecs[i] - vecs[c]) ** 2)) for c in chosen)
            for i in range(N)
        ])
        probs  = dists / (dists.sum() + 1e-12)
        chosen.append(int(rng.choice(N, p=probs)))
    return vecs[chosen].astype(np.float32)


def _pairwise_sq_dist(
    a: np.ndarray,
    b: np.ndarray,
) -> np.ndarray:
    """
    Compute squared Euclidean distances between rows of *a* and *b*.

    ||a - b||^2  =  ||a||^2 - 2*a·b + ||b||^2

    Returns (N, M) float32 array.
    """
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    a2 = np.sum(a ** 2, axis=1, keepdims=True)    # (N, 1)
    b2 = np.sum(b ** 2, axis=1, keepdims=True)    # (M, 1)
    ab = a @ b.T                                   # (N, M)
    return np.maximum(a2 - 2 * ab + b2.T, 0.0).astype(np.float32)
