"""
squish/shadow_kv.py

ShadowKV — Low-Rank Pre-RoPE Key Cache with CPU Value Shadow.

Inspired by:
  "ShadowKV: KV Cache in Shadows for High-Throughput Long-Context LLM Inference"
  arXiv:2410.21465 (Oct 2024)

Problem
-------
Long-context inference (≥32K tokens) blows up KV memory: a 7B model at 128K
tokens may need 100+ GB just for the KV cache — more than any consumer GPU
has.  Standard eviction is lossy; CPU offload of ALL KV pairs is slow due to
PCIe bandwidth.

ShadowKV Strategy
-----------------
1. **Low-rank key compression (GPU)** — before applying RoPE, compute an SVD
   of the key matrix over a calibration window:
       K ≈ K_proj @ V^T     where V ∈ R^{head_dim × rank}
   Store only the low-rank projections (``rank ≪ head_dim``) on GPU.
   RoPE is applied afterwards, so the compression is rotation-equivariant.

2. **Value shadow (CPU)** — full-precision value tensors are stored in a CPU
   dict keyed by ``(layer_id, token_pos)`` — the "shadow".

3. **Landmark-based sparse retrieval** — at each decode step:
   a. Select K *landmark* positions whose keys best cover the current query
      (via cosine similarity on the low-rank projections).
   b. Prefetch those K value vectors from CPU.
   c. Reconstruct the full keys from their low-rank representations.
   d. Run sparse attention over the K landmark K/V pairs.

   Result: each decode step reads only K ≪ N values from CPU → bandwidth cut
   by N/K.  Typical ratio ≈ 128× for N=128K, K=1K.

This module provides:
  ``ShadowKVConfig``      — all hyperparameters
  ``LowRankKeyCache``     — per-layer SVD-compressed key store (GPU-side)
  ``LandmarkSelector``    — choose top-k token positions for sparse attention
  ``ShadowKVCache``       — unified manager: store + recall for all layers

Usage::

    from squish.kv.shadow_kv import ShadowKVCache, ShadowKVConfig

    cfg   = ShadowKVConfig(svd_rank=32, n_landmarks=128)
    cache = ShadowKVCache(n_layers=32, n_heads=32, head_dim=128, config=cfg)

    # During prefill:
    for layer_id, (keys, values) in enumerate(kv_pairs):
        cache.store(layer_id, keys, values)

    # During decode:
    for layer_id, query in enumerate(queries):
        k_sparse, v_sparse = cache.recall(layer_id, query, top_k=cfg.n_landmarks)
        # Run attention over k_sparse, v_sparse  (shape: top_k × head_dim)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "ShadowKVConfig",
    "LowRankKeyCache",
    "LandmarkSelector",
    "ShadowKVCache",
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ShadowKVConfig:
    """
    Configuration for ShadowKV.

    Parameters
    ----------
    svd_rank : int
        Rank of the per-head SVD key projection.
        Smaller → more compression; larger → better approximation.
        Typical: 32–64 for head_dim=128.
    n_landmarks : int
        Number of landmark (top-k) token positions to load from CPU per
        decode step.  Typical: 128–1024.
    min_calibration_tokens : int
        Minimum number of tokens required before fitting the SVD projection.
        Until this threshold is reached, keys are stored in raw float16.
    """
    svd_rank:                int = 32
    n_landmarks:             int = 128
    min_calibration_tokens:  int = 64

    def __post_init__(self) -> None:
        if self.svd_rank < 1:
            raise ValueError(f"svd_rank must be ≥ 1, got {self.svd_rank}")
        if self.n_landmarks < 1:
            raise ValueError(f"n_landmarks must be ≥ 1, got {self.n_landmarks}")
        if self.min_calibration_tokens < 1:
            raise ValueError(
                f"min_calibration_tokens must be ≥ 1, got {self.min_calibration_tokens}"
            )


# ---------------------------------------------------------------------------
# Low-rank key cache (GPU-side)
# ---------------------------------------------------------------------------

class LowRankKeyCache:
    """
    Stores SVD-compressed key projections for a single transformer layer.

    After ``fit_svd`` is called with a calibration batch, incoming keys are
    projected to ``shape (n_tokens, n_heads, rank)`` instead of
    ``(n_tokens, n_heads, head_dim)``.

    Projection basis ``V`` (shape ``(n_heads, rank, head_dim)``) is frozen
    after calibration (mirrors the approach in ``kv_cache.py::KVLayerCache``).

    Parameters
    ----------
    n_heads  : int
    head_dim : int
    rank     : int
    """

    def __init__(self, n_heads: int, head_dim: int, rank: int) -> None:
        self.n_heads  = n_heads
        self.head_dim = head_dim
        self.rank     = rank
        self._V: np.ndarray | None = None  # (n_heads, rank, head_dim) float32
        # Dense storage: list of (n_heads, rank) rows — one per stored token
        self._projections: list[np.ndarray] = []

    # ------------------------------------------------------------------

    @property
    def is_fitted(self) -> bool:
        """True if the SVD basis has been computed."""
        return self._V is not None

    @property
    def n_stored(self) -> int:
        """Number of token positions currently stored."""
        return len(self._projections)

    # ------------------------------------------------------------------

    def fit_svd(self, keys: np.ndarray) -> None:
        """
        Compute and store the per-head SVD projection basis.

        Parameters
        ----------
        keys : np.ndarray  shape (n_tokens, n_heads, head_dim)  — float16 or float32
            Calibration keys (ideally from the prefill pass).
        """
        keys_f = np.asarray(keys, dtype=np.float32)
        n_tokens, n_heads, head_dim = keys_f.shape
        rank = min(self.rank, head_dim, n_tokens)
        V_list = []
        for h in range(n_heads):
            K_h = keys_f[:, h, :]          # (n_tokens, head_dim)
            # Thin SVD: K = U S V^T  → keep top-rank rows of V^T
            _, _, Vt = np.linalg.svd(K_h, full_matrices=False)
            V_list.append(Vt[:rank, :])    # (rank, head_dim)
        self._V = np.stack(V_list, axis=0).astype(np.float32)  # (n_heads, rank, head_dim)

    def project(self, key: np.ndarray) -> np.ndarray:
        """
        Project a single-token key (n_heads, head_dim) to low-rank space.

        Parameters
        ----------
        key : np.ndarray  shape (n_heads, head_dim)

        Returns
        -------
        np.ndarray  shape (n_heads, rank)
        """
        if self._V is None:
            raise RuntimeError("Call fit_svd() before project()")
        k = np.asarray(key, dtype=np.float32)  # (n_heads, head_dim)
        # batch matmul: (n_heads, rank, head_dim) @ (n_heads, head_dim, 1)
        proj = np.einsum("hrd,hd->hr", self._V, k)  # (n_heads, rank)
        return proj.astype(np.float16)

    def reconstruct(self, proj: np.ndarray) -> np.ndarray:
        """
        Reconstruct an approximate key from its low-rank projection.

        Parameters
        ----------
        proj : np.ndarray  shape (n_heads, rank)

        Returns
        -------
        np.ndarray  shape (n_heads, head_dim)
        """
        if self._V is None:
            raise RuntimeError("Call fit_svd() before reconstruct()")
        p = np.asarray(proj, dtype=np.float32)   # (n_heads, rank)
        # (n_heads, head_dim) = (n_heads, rank) @ (n_heads, rank, head_dim)
        full = np.einsum("hr,hrd->hd", p, self._V)  # (n_heads, head_dim)
        return full.astype(np.float16)

    def add(self, key: np.ndarray) -> int:
        """
        Add a key (raw or projected) and return its position index.

        If the SVD basis is fitted, the key is projected first.
        Otherwise it is stored in its original form (zero-padded / truncated
        to (n_heads, rank) for uniform storage).

        Parameters
        ----------
        key : np.ndarray  shape (n_heads, head_dim) or (n_heads, rank)

        Returns
        -------
        int — position index (0-based)
        """
        k = np.asarray(key, dtype=np.float32)
        if self._V is not None and k.shape[-1] == self.head_dim:
            stored = self.project(k)
        else:
            # Store raw (truncated or pre-projected)
            stored = k[:, :self.rank].astype(np.float16)
        self._projections.append(stored)
        return len(self._projections) - 1

    def get_all_projections(self) -> np.ndarray:
        """
        Return all stored projections as a dense array.

        Returns
        -------
        np.ndarray  shape (n_stored, n_heads, rank)
        """
        if not self._projections:
            return np.empty((0, self.n_heads, self.rank), dtype=np.float16)
        return np.stack(self._projections, axis=0)


# ---------------------------------------------------------------------------
# Landmark selector
# ---------------------------------------------------------------------------

class LandmarkSelector:
    """
    Selects the K most-relevant token positions for a given query using
    cosine similarity on the low-rank key projections.

    Parameters
    ----------
    n_landmarks : int
        Maximum number of positions to select.
    """

    def __init__(self, n_landmarks: int = 128) -> None:
        if n_landmarks < 1:
            raise ValueError(f"n_landmarks must be ≥ 1, got {n_landmarks}")
        self.n_landmarks = n_landmarks

    def select(
        self,
        query: np.ndarray,
        key_projections: np.ndarray,
    ) -> np.ndarray:
        """
        Select the top-K token positions by average cosine similarity over heads.

        Parameters
        ----------
        query : np.ndarray
            Shape (n_heads, rank) — low-rank projected query.
        key_projections : np.ndarray
            Shape (n_tokens, n_heads, rank) — all stored key projections.

        Returns
        -------
        np.ndarray  shape (min(K, n_tokens),)  dtype int64
            Indices of selected positions, sorted by descending score.
        """
        n_tokens = key_projections.shape[0]
        if n_tokens == 0:
            return np.empty(0, dtype=np.int64)

        q = np.asarray(query, dtype=np.float32)        # (n_heads, rank)
        K = np.asarray(key_projections, dtype=np.float32)  # (n_tokens, n_heads, rank)

        # Per-head dot products: (n_tokens, n_heads)
        dots = np.einsum("tnr,hr->tn", K, q)  # (n_tokens, n_heads)

        # L2 norms for cosine normalisation
        q_norm = np.linalg.norm(q, axis=-1, keepdims=True) + 1e-8  # (n_heads, 1)
        k_norm = np.linalg.norm(K, axis=-1) + 1e-8                  # (n_tokens, n_heads)
        # dots / (q_norm.T * k_norm)  — broadcast carefully
        scores = dots / (q_norm.T * k_norm)  # (n_tokens, n_heads)
        mean_scores = scores.mean(axis=-1)    # (n_tokens,)

        k = min(self.n_landmarks, n_tokens)
        top_k = np.argpartition(mean_scores, -k)[-k:]
        # Sort by descending score
        top_k = top_k[np.argsort(mean_scores[top_k])[::-1]]
        return top_k.astype(np.int64)


# ---------------------------------------------------------------------------
# ShadowKVCache — unified manager
# ---------------------------------------------------------------------------

class ShadowKVCache:
    """
    Manages low-rank GPU key cache + CPU value shadow for all transformer layers.

    Parameters
    ----------
    n_layers : int
    n_heads  : int
    head_dim : int
    config   : ShadowKVConfig
    """

    def __init__(
        self,
        n_layers: int,
        n_heads:  int,
        head_dim: int,
        config:   ShadowKVConfig | None = None,
    ) -> None:
        cfg = config or ShadowKVConfig()
        self.n_layers  = n_layers
        self.n_heads   = n_heads
        self.head_dim  = head_dim
        self.config    = cfg

        # Per-layer key cache (GPU-side, low-rank projections)
        self._key_caches: list[LowRankKeyCache] = [
            LowRankKeyCache(n_heads, head_dim, cfg.svd_rank)
            for _ in range(n_layers)
        ]
        # CPU value shadow: (layer_id, token_pos) → (n_heads, head_dim) float16
        self._value_shadow: dict[tuple[int, int], np.ndarray] = {}
        # Per-layer token count
        self._n_tokens: list[int] = [0] * n_layers
        # Landmark selector (shared across layers)
        self._selector = LandmarkSelector(cfg.n_landmarks)

    # ------------------------------------------------------------------

    def store(
        self,
        layer_id: int,
        keys:     np.ndarray,
        values:   np.ndarray,
    ) -> None:
        """
        Store keys and values for a batch of tokens in one layer.

        Keys are projected to low-rank space (after SVD is fitted).
        Values are stored on CPU in the shadow dict.

        Parameters
        ----------
        layer_id : int
        keys   : np.ndarray  shape (n_tokens, n_heads, head_dim)
        values : np.ndarray  shape (n_tokens, n_heads, head_dim)
        """
        kc = self._key_caches[layer_id]
        keys_f = np.asarray(keys, dtype=np.float16)
        vals_f = np.asarray(values, dtype=np.float16)
        n_tokens = keys_f.shape[0]

        # Fit SVD on the first calibration batch
        if not kc.is_fitted and n_tokens >= self.config.min_calibration_tokens:
            kc.fit_svd(keys_f)

        # Store each token
        base = self._n_tokens[layer_id]
        for i in range(n_tokens):
            pos = base + i
            kc.add(keys_f[i])  # project + store
            self._value_shadow[(layer_id, pos)] = vals_f[i]  # CPU

        self._n_tokens[layer_id] += n_tokens

    def recall(
        self,
        layer_id: int,
        query:    np.ndarray,
        top_k:    int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Recall the top-K key/value pairs for a query at one layer.

        Parameters
        ----------
        layer_id : int
        query    : np.ndarray  shape (n_heads, head_dim)
        top_k    : int or None — defaults to config.n_landmarks

        Returns
        -------
        keys   : np.ndarray  shape (K, n_heads, head_dim) — reconstructed
        values : np.ndarray  shape (K, n_heads, head_dim) — fetched from CPU
        """
        k = top_k if top_k is not None else self.config.n_landmarks
        kc = self._key_caches[layer_id]
        all_proj = kc.get_all_projections()  # (n_stored, n_heads, rank)

        if all_proj.shape[0] == 0:
            return (
                np.empty((0, self.n_heads, self.head_dim), dtype=np.float16),
                np.empty((0, self.n_heads, self.head_dim), dtype=np.float16),
            )

        # Project query to low-rank space (if SVD is fitted)
        if kc.is_fitted:
            q_proj = kc.project(np.asarray(query, dtype=np.float32))  # (n_heads, rank)
        else:
            q_proj = np.asarray(query, dtype=np.float32)[:, :self.config.svd_rank].astype(np.float16)

        positions = self._selector.select(q_proj, all_proj)
        # Fetch and reconstruct
        keys_out   = []
        values_out = []
        for pos in positions[:k]:
            proj = all_proj[int(pos)]               # (n_heads, rank)
            if kc.is_fitted:
                key_full = kc.reconstruct(proj)     # (n_heads, head_dim)
            else:
                # SVD not yet fitted — return zero-padded projection
                key_full = np.zeros((self.n_heads, self.head_dim), dtype=np.float16)
                rk = proj.shape[-1]
                key_full[:, :rk] = proj
            val_full = self._value_shadow.get(
                (layer_id, int(pos)),
                np.zeros((self.n_heads, self.head_dim), dtype=np.float16),
            )
            keys_out.append(key_full)
            values_out.append(val_full)

        if not keys_out:
            return (
                np.empty((0, self.n_heads, self.head_dim), dtype=np.float16),
                np.empty((0, self.n_heads, self.head_dim), dtype=np.float16),
            )

        return (
            np.stack(keys_out,   axis=0),
            np.stack(values_out, axis=0),
        )

    # ------------------------------------------------------------------

    def n_stored(self, layer_id: int) -> int:
        """Return the number of token positions stored for a layer."""
        return self._n_tokens[layer_id]

    def clear(self, layer_id: int | None = None) -> None:
        """
        Clear cached data.

        Parameters
        ----------
        layer_id : int or None — if None, clears all layers.
        """
        if layer_id is None:
            self._value_shadow.clear()
            self._n_tokens = [0] * self.n_layers
            self._key_caches = [
                LowRankKeyCache(self.n_heads, self.head_dim, self.config.svd_rank)
                for _ in range(self.n_layers)
            ]
        else:
            keys_to_remove = [
                k for k in self._value_shadow if k[0] == layer_id
            ]
            for k in keys_to_remove:
                del self._value_shadow[k]
            self._n_tokens[layer_id] = 0
            self._key_caches[layer_id] = LowRankKeyCache(
                self.n_heads, self.head_dim, self.config.svd_rank
            )
