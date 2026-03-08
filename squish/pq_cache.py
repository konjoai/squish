"""
squish/pq_cache.py

PQCache — Product-Quantization KV Cache with Asymmetric Distance-Based Retrieval.

Inspired by:
  "PQCache: Product Quantization-based KVCache for Long Context LLM Inference"
  arXiv:2407.12820 (Jul 2024)

Problem
-------
Long-context inference requires O(N) KV memory.  Eviction is lossy; full
retention is expensive.  For *retrieval heads* (those attending non-locally),
we need a compact representation that supports **approximate nearest-neighbour
(ANN)** lookup.

PQCache Approach
----------------
Apply **Product Quantization (PQ)** to the key cache:
  * Split the head dimension D into M sub-vectors of size D/M.
  * For each sub-space maintain a small codebook of K codes.
  * Each key is stored as M × 1-byte (or 2-byte) indices → D/M compression.
  * At query time, compute Asymmetric Distance Computation (ADC):
        d(q, k_pq) ≈ Σ_m ||q_m - c_m[idx_m]||²
    where ``c_m`` is the m-th sub-space centroid lookup table.
  * Top-K keys are retrieved; their values are looked up from a compressed
    value store (INT8 quantized).

Memory reduction: 32× vs fp32 with 8 sub-spaces × 256 codes.

This module provides:
  * ``PQCacheConfig``
  * ``PQCodebook``  — per-subspace codebook
  * ``PQKeyIndex``  — PQ-compressed key index with ADC search
  * ``PQValueStore`` — INT8-quantised value store

Usage::

    from squish.pq_cache import PQCacheConfig, PQKeyIndex, PQValueStore

    cfg     = PQCacheConfig(n_subvectors=8, n_codes=256)
    key_idx = PQKeyIndex(dim=128, config=cfg)
    val_st  = PQValueStore()

    for step in decode:
        key_idx.add(key_vec, seq_pos)
        val_st.add(seq_pos, val_vec)

    top_keys, top_vals = retrieve(query_vec, key_idx, val_st, top_k=64)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

__all__ = [
    "PQCacheConfig",
    "PQCodebook",
    "PQKeyIndex",
    "PQValueStore",
    "retrieve",
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class PQCacheConfig:
    """
    Configuration for PQCache.

    Parameters
    ----------
    n_subvectors : int
        Number of sub-spaces (M).  Must divide the key dimension D evenly.
    n_codes : int
        Codebook size per sub-space (K).  Typically 256 for 1-byte indices.
    train_iters : int
        K-means iterations for codebook fitting.
    seed : int
        RNG seed for reproducible fitting.
    """
    n_subvectors: int = 8
    n_codes:      int = 256
    train_iters:  int = 20
    seed:         int = 42

    def __post_init__(self) -> None:
        if self.n_subvectors < 1:
            raise ValueError("n_subvectors must be ≥ 1")
        if self.n_codes < 2:
            raise ValueError("n_codes must be ≥ 2")
        if self.train_iters < 1:
            raise ValueError("train_iters must be ≥ 1")


# ---------------------------------------------------------------------------
# Per-subspace codebook
# ---------------------------------------------------------------------------

class PQCodebook:
    """
    Single sub-space vector-quantization codebook.

    Parameters
    ----------
    sub_dim  : int — dimension of each sub-vector (D / M)
    n_codes  : int — number of centroids (K)
    n_iters  : int — k-means iterations for fitting
    seed     : int — RNG seed
    """

    def __init__(
        self,
        sub_dim: int,
        n_codes: int  = 256,
        n_iters: int  = 20,
        seed:    int  = 42,
    ) -> None:
        self._sub_dim  = sub_dim
        self._n_codes  = n_codes
        self._n_iters  = n_iters
        self._seed     = seed
        self._centroids: Optional[np.ndarray] = None

    def fit(self, sub_vecs: np.ndarray) -> None:
        """
        Fit centroids on *sub_vecs* via k-means.

        Parameters
        ----------
        sub_vecs : (N, sub_dim) float32
        """
        vecs = np.asarray(sub_vecs, dtype=np.float32)
        K    = min(self._n_codes, len(vecs))
        rng  = np.random.default_rng(self._seed)

        # K-means++ init
        chosen = [int(rng.integers(0, len(vecs)))]
        for _ in range(1, K):
            dists = np.array([
                min(float(np.sum((vecs[i] - vecs[c]) ** 2)) for c in chosen)
                for i in range(len(vecs))
            ])
            probs  = dists / (dists.sum() + 1e-12)
            chosen.append(int(rng.choice(len(vecs), p=probs)))
        centroids = vecs[chosen].astype(np.float32)

        for _ in range(self._n_iters):
            dists       = self._sq_dist(vecs, centroids)
            assignments = dists.argmin(axis=1)
            new_c       = np.zeros_like(centroids)
            for k in range(K):
                mask = assignments == k
                new_c[k] = vecs[mask].mean(axis=0) if mask.any() else centroids[k]
            if np.allclose(centroids, new_c, atol=1e-6):
                break
            centroids = new_c

        self._centroids = centroids
        self._n_codes   = K   # actual number (may be < requested if N < K)

    def encode(self, sub_vecs: np.ndarray) -> np.ndarray:
        """(N, sub_dim) → (N,) uint16 indices."""
        if self._centroids is None:
            raise RuntimeError("Not fitted — call .fit() first")
        vecs  = np.asarray(sub_vecs, dtype=np.float32)
        dists = self._sq_dist(vecs, self._centroids)
        return dists.argmin(axis=1).astype(np.uint16)

    def decode(self, indices: np.ndarray) -> np.ndarray:
        """(N,) indices → (N, sub_dim) float32 centroids."""
        if self._centroids is None:
            raise RuntimeError("Not fitted — call .fit() first")
        return self._centroids[np.asarray(indices, dtype=np.int64)].astype(np.float32)

    def lookup_table(self, query_sub: np.ndarray) -> np.ndarray:
        """
        Pre-compute a distance lookup table for ADC.

        Parameters
        ----------
        query_sub : (sub_dim,) float32 — query sub-vector

        Returns
        -------
        lut : (K,) float32 — squared distances to each centroid
        """
        if self._centroids is None:
            raise RuntimeError("Not fitted — call .fit() first")
        q   = np.asarray(query_sub, dtype=np.float32).reshape(1, -1)
        lut = self._sq_dist(q, self._centroids).flatten()
        return lut

    @property
    def is_fitted(self) -> bool:
        return self._centroids is not None

    # ── Static ────────────────────────────────────────────────────────────────

    @staticmethod
    def _sq_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        a = a.astype(np.float64)
        b = b.astype(np.float64)
        a2 = (a ** 2).sum(axis=1, keepdims=True)
        b2 = (b ** 2).sum(axis=1, keepdims=True)
        return np.maximum(a2 - 2 * (a @ b.T) + b2.T, 0.0).astype(np.float32)


# ---------------------------------------------------------------------------
# PQ Key Index
# ---------------------------------------------------------------------------

class PQKeyIndex:
    """
    Product-quantized index for key vectors supporting ADC-based top-K search.

    Parameters
    ----------
    dim    : int — key vector dimension (must be divisible by n_subvectors)
    config : PQCacheConfig
    """

    def __init__(self, dim: int, config: Optional[PQCacheConfig] = None) -> None:
        cfg = config or PQCacheConfig()
        if dim % cfg.n_subvectors != 0:
            raise ValueError(
                f"dim ({dim}) must be divisible by n_subvectors ({cfg.n_subvectors})"
            )
        self._dim        = dim
        self._cfg        = cfg
        self._sub_dim    = dim // cfg.n_subvectors
        self._codebooks: List[PQCodebook] = [
            PQCodebook(self._sub_dim, cfg.n_codes, cfg.train_iters, cfg.seed + i)
            for i in range(cfg.n_subvectors)
        ]
        self._is_fitted  = False
        # code storage: list of (M,) uint16 arrays, one per token in cache
        self._codes:    List[np.ndarray] = []
        self._seq_pos:  List[int]        = []   # token positions (for retrieval)

    # ── Fitting ───────────────────────────────────────────────────────────────

    def fit(self, keys: np.ndarray) -> None:
        """
        Fit all sub-space codebooks on calibration key vectors.

        Parameters
        ----------
        keys : (N, D) float32
        """
        K = np.asarray(keys, dtype=np.float32)
        for m, cb in enumerate(self._codebooks):
            start = m * self._sub_dim
            end   = start + self._sub_dim
            cb.fit(K[:, start:end])
        self._is_fitted = True

    # ── Add / Search ─────────────────────────────────────────────────────────

    def add(self, key_vec: np.ndarray, seq_pos: int) -> None:
        """
        Compress and store one key vector.

        Parameters
        ----------
        key_vec : (D,) float32
        seq_pos : int — token position (returned by search for value lookup)
        """
        if not self._is_fitted:
            raise RuntimeError("Index not fitted — call .fit() first")
        k   = np.asarray(key_vec, dtype=np.float32).flatten()
        code = np.zeros(len(self._codebooks), dtype=np.uint16)
        for m, cb in enumerate(self._codebooks):
            start = m * self._sub_dim
            end   = start + self._sub_dim
            code[m] = int(cb.encode(k[start:end].reshape(1, -1))[0])
        self._codes.append(code)
        self._seq_pos.append(seq_pos)

    def search(
        self,
        query: np.ndarray,
        top_k: int = 64,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retrieve top-K approximately nearest keys via ADC.

        Parameters
        ----------
        query : (D,) float32
        top_k : int

        Returns
        -------
        positions : (top_k,) int — token positions in the KV cache
        distances : (top_k,) float32 — approximate squared distances
        """
        if not self._codes:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float32)

        q = np.asarray(query, dtype=np.float32).flatten()
        # Build LUTs for each sub-space
        luts = [
            cb.lookup_table(q[m * self._sub_dim: (m + 1) * self._sub_dim])
            for m, cb in enumerate(self._codebooks)
        ]

        # ADC: accumulate distances from LUTs
        n   = len(self._codes)
        adc = np.zeros(n, dtype=np.float32)
        for m, lut in enumerate(luts):
            code_col = np.array([self._codes[i][m] for i in range(n)], dtype=np.int64)
            adc     += lut[code_col]

        k_out = min(top_k, n)
        idxs  = np.argpartition(adc, k_out - 1)[:k_out]
        idxs  = idxs[np.argsort(adc[idxs])]

        positions = np.array([self._seq_pos[i] for i in idxs], dtype=np.int64)
        distances = adc[idxs]
        return positions, distances

    def __len__(self) -> int:
        return len(self._codes)

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted


# ---------------------------------------------------------------------------
# PQ Value Store (INT8 quantized)
# ---------------------------------------------------------------------------

class PQValueStore:
    """
    Stores value vectors in INT8 format with per-vector scale factors.

    Memory: (D + 2) bytes per value vector vs. D × 4 bytes for float32.
    Compression factor: ~3.7×.
    """

    def __init__(self) -> None:
        self._values: Dict[int, Tuple[np.ndarray, float, float]] = {}
        # seq_pos → (int8_vec, scale, zero_point)

    def add(self, seq_pos: int, value: np.ndarray) -> None:
        """
        Quantize and store a value vector.

        Parameters
        ----------
        seq_pos : int — token position key (must match PQKeyIndex.add)
        value   : (D,) float32
        """
        v = np.asarray(value, dtype=np.float32)
        v_min, v_max = float(v.min()), float(v.max())
        if v_min == v_max:
            self._values[seq_pos] = (np.zeros_like(v, dtype=np.int8), 1.0, v_min)
            return
        scale    = (v_max - v_min) / 254.0   # 8-bit symmetric around 0 (-127..127)
        zero_pt  = v_min
        q        = np.round((v - zero_pt) / scale).clip(-127, 127).astype(np.int8)
        self._values[seq_pos] = (q, scale, zero_pt)

    def get(self, seq_pos: int) -> Optional[np.ndarray]:
        """
        Dequantize and return the value vector for *seq_pos*.

        Returns None if not found.
        """
        entry = self._values.get(seq_pos)
        if entry is None:
            return None
        q, scale, zero_pt = entry
        return q.astype(np.float32) * scale + zero_pt

    def get_batch(self, positions: np.ndarray) -> np.ndarray:
        """
        Retrieve multiple value vectors by position array.

        Parameters
        ----------
        positions : (K,) int

        Returns
        -------
        values : (K, D) float32  — missing positions are zero-filled
        """
        results = []
        for pos in positions:
            v = self.get(int(pos))
            if v is not None:
                results.append(v)
        if not results:
            return np.empty((0, 0), dtype=np.float32)
        return np.stack(results, axis=0)

    def __len__(self) -> int:
        return len(self._values)


# ---------------------------------------------------------------------------
# Convenience: retrieve keys + values from a PQKeyIndex + PQValueStore
# ---------------------------------------------------------------------------

def retrieve(
    query:     np.ndarray,
    key_index: PQKeyIndex,
    val_store: PQValueStore,
    top_k:     int = 64,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Retrieve top-K (key, value) pairs for a query via PQ ADC.

    Parameters
    ----------
    query     : (D,) float32 — attention query vector
    key_index : fitted PQKeyIndex
    val_store : PQValueStore with matching positions
    top_k     : int

    Returns
    -------
    positions : (K,) int64
    values    : (K, D_v) float32
    """
    positions, _ = key_index.search(query, top_k=top_k)
    values       = val_store.get_batch(positions)
    return positions, values
