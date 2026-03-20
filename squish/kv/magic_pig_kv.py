"""squish/kv/magic_pig_kv.py

MagicPIG KV — LSH-Based Top-K KV Sampling for Approximate Attention.

Reference
---------
"MagicPIG: LLM Serving using Sampling-based KV Cache Compression."
NeurIPS 2024 (arXiv:2410.16179).

Algorithm
---------
At million-token scale, exact top-K KV retrieval is too slow.  MagicPIG
approximates the inner-product search using **Locality-Sensitive Hashing
(LSH)** with multiple random sign-projection hash tables:

1. Keys are hashed into ``n_tables`` × ``n_bits``-width hash buckets.
2. For each query, the same hash is computed to get candidate buckets.
3. All candidates across tables are merged and exact attention is run only
   over this reduced candidate set (typically a few hundred out of millions).

This module provides a NumPy-based simulation using random sign-projection
hashing (SimHash), which is equivalent to the theoretical analysis in the
paper.

Key properties
--------------
* NumPy-only CPU simulation; hashing is approximate inner-product retrieval.
* ``n_tables`` — number of independent hash tables.
* ``n_bits`` — bits per hash bucket (determines bucket width 2^n_bits).
* ``min_candidates`` — at least this many candidates are always examined.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

__all__ = [
    "MagicPIGConfig",
    "MagicPIGKV",
]

# ── Config ────────────────────────────────────────────────────────────────────


@dataclass
class MagicPIGConfig:
    """Configuration for :class:`MagicPIGKV`.

    Attributes:
        n_tables: Number of independent LSH tables.
        n_bits: Number of hash bits per table (bucket count = 2^n_bits).
        min_candidates: Minimum candidates retrieved regardless of hash hits.
        n_heads: Attention heads per layer.
        head_dim: Dimension per head.
        seed: RNG seed for hash projections.
    """

    n_tables: int = 4
    n_bits: int = 8
    min_candidates: int = 64
    n_heads: int = 8
    head_dim: int = 64
    seed: int = 0

    def __post_init__(self) -> None:
        if self.n_tables < 1:
            raise ValueError(f"n_tables must be ≥ 1; got {self.n_tables}")
        if self.n_bits < 1:
            raise ValueError(f"n_bits must be ≥ 1; got {self.n_bits}")
        if self.min_candidates < 1:
            raise ValueError(
                f"min_candidates must be ≥ 1; got {self.min_candidates}"
            )
        if self.n_heads < 1:
            raise ValueError(f"n_heads must be ≥ 1; got {self.n_heads}")
        if self.head_dim < 1:
            raise ValueError(f"head_dim must be ≥ 1; got {self.head_dim}")


# ── Core class ─────────────────────────────────────────────────────────────────


class MagicPIGKV:
    """LSH-based approximate attention with KV candidate retrieval.

    Example::

        cfg  = MagicPIGConfig(n_tables=4, n_bits=4, n_heads=2, head_dim=8)
        pig  = MagicPIGKV(cfg)

        K = np.random.randn(2, 512, 8).astype(np.float32)
        V = np.random.randn(2, 512, 8).astype(np.float32)
        pig.build_index(K)

        Q = np.random.randn(2, 1, 8).astype(np.float32)
        out = pig.attend(Q, K, V)  # (2, 1, 8)
    """

    def __init__(self, config: Optional[MagicPIGConfig] = None) -> None:
        self.config = config or MagicPIGConfig()
        cfg = self.config
        rng = np.random.default_rng(cfg.seed)
        # Random sign projections: (n_tables, n_bits, head_dim)
        self._projections: np.ndarray = rng.standard_normal(
            (cfg.n_tables, cfg.n_bits, cfg.head_dim)
        ).astype(np.float32)
        # Indexed key hashes per head: built at build_index time
        self._key_hashes: Optional[np.ndarray] = None  # (H, n_tables, S)
        self._n_indexed: int = 0

    # ── Index building ────────────────────────────────────────────────────────

    def build_index(self, K: np.ndarray) -> None:
        """Hash all key vectors and store the hash codes.

        Args:
            K: ``(n_heads, S, head_dim)`` key tensor.
        """
        K = np.asarray(K, dtype=np.float32)
        H, S, d = K.shape
        n_t = self.config.n_tables
        n_b = self.config.n_bits

        # (H, n_tables, S): sign projections → 0/1 codes → integer bucket
        hashes = np.zeros((H, n_t, S), dtype=np.int32)
        for t in range(n_t):
            proj = self._projections[t]  # (n_bits, d)
            # (H, S, n_bits) via broadcasting
            signs = (K @ proj.T) >= 0   # (H, S, n_bits)
            # Pack bits into integer bucket id
            bits = (1 << np.arange(n_b, dtype=np.int32))[np.newaxis, np.newaxis, :]
            hashes[:, t, :] = (signs.astype(np.int32) * bits).sum(axis=-1)

        self._key_hashes = hashes
        self._n_indexed = S

    def _retrieve_candidates(self, q: np.ndarray, h: int) -> np.ndarray:
        """Retrieve candidate KV indices for a single-head query vector.

        Args:
            q: ``(head_dim,)`` query vector.
            h: Head index.

        Returns:
            Sorted unique candidate indices.
        """
        n_t = self.config.n_tables
        n_b = self.config.n_bits
        S = self._n_indexed

        q_hashes = np.zeros(n_t, dtype=np.int32)
        for t in range(n_t):
            proj = self._projections[t]  # (n_bits, d)
            signs = (proj @ q) >= 0  # (n_bits,)
            bits = (1 << np.arange(n_b, dtype=np.int32))
            q_hashes[t] = int((signs.astype(np.int32) * bits).sum())

        # Collect matching positions
        candidates = set()
        for t in range(n_t):
            matches = np.where(self._key_hashes[h, t] == q_hashes[t])[0]
            candidates.update(matches.tolist())

        result = np.array(sorted(candidates), dtype=np.int32)

        # Always include at least min_candidates (random fill if too few)
        if len(result) < self.config.min_candidates:
            remaining = np.setdiff1d(np.arange(S, dtype=np.int32), result)
            n_extra = min(self.config.min_candidates - len(result), len(remaining))
            if n_extra > 0:
                extra = remaining[:n_extra]
                result = np.sort(np.concatenate([result, extra]))

        return result

    # ── Attend ────────────────────────────────────────────────────────────────

    def attend(
        self,
        Q: np.ndarray,
        K: np.ndarray,
        V: np.ndarray,
    ) -> np.ndarray:
        """Approximate attention using LSH candidate retrieval.

        Args:
            Q: ``(n_heads, T_q, head_dim)``.
            K: ``(n_heads, S, head_dim)``.
            V: ``(n_heads, S, head_dim)``.

        Returns:
            ``(n_heads, T_q, head_dim)`` context vectors.

        Note: If the index has not been built yet, falls back to exact attention.
        """
        Q = np.asarray(Q, dtype=np.float32)
        K = np.asarray(K, dtype=np.float32)
        V = np.asarray(V, dtype=np.float32)
        H, Tq, d = Q.shape
        S = K.shape[1]
        scale = 1.0 / np.sqrt(d)
        out = np.zeros((H, Tq, d), dtype=np.float32)

        if self._key_hashes is None:
            # Fallback: exact attention
            scores = np.einsum("htd,hsd->hts", Q, K) * scale
            e = np.exp(scores - scores.max(axis=-1, keepdims=True))
            attn = e / (e.sum(axis=-1, keepdims=True) + 1e-9)
            return (attn @ V).astype(np.float32)

        for h in range(H):
            for t in range(Tq):
                cands = self._retrieve_candidates(Q[h, t], h)
                cands = cands[cands < S]
                k_c = K[h][cands]  # (C, d)
                v_c = V[h][cands]  # (C, d)
                scores = (Q[h, t] @ k_c.T) * scale  # (C,)
                e = np.exp(scores - scores.max())
                attn = e / (e.sum() + 1e-9)
                out[h, t] = attn @ v_c

        return out

    def __repr__(self) -> str:
        return (
            f"MagicPIGKV(n_tables={self.config.n_tables}, "
            f"n_bits={self.config.n_bits}, "
            f"indexed={self._n_indexed} keys)"
        )
