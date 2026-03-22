"""squish/kv/magic_pig_v2.py

MagicPIGv2 — LSH KV Retrieval with Adaptive Per-Layer Probe Budget.

Reference
---------
Chen et al. "MagicPIG: LLM Serving using Sampling-based KV Cache
Compression." NeurIPS 2024 workshop (full: arXiv:2410.16179).
Extended variant with adaptive probe count and beam-search draft integration.

Algorithm
---------
MagicPIGv2 extends Wave-40 MagicPIG with:

1. **Adaptive probe count** — each layer starts with a default probe
   budget; if the retrieval quality (estimated by attention-weight entropy)
   falls below a target, the budget is doubled up to a cap.
2. **Beam-search draft integration** — the LSH candidate set is exposed
   so downstream speculative decoders can re-use it as a draft token pool.
3. **Per-layer independence** — each layer maintains its own hash tables
   and probe budget, allowing fine-grained control.

Key properties
--------------
* NumPy-only.
* ``n_tables`` — number of LSH hash tables.
* ``n_bits`` — hash bits per table.
* ``min_probes`` — minimum candidate tokens retrieved.
* ``max_probes`` — adaptive cap on candidate count.
* ``target_entropy_ratio`` — entropy target for adaptive probe expansion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

__all__ = [
    "MagicPIGv2Config",
    "MagicPIGv2",
]


@dataclass
class MagicPIGv2Config:
    """Configuration for :class:`MagicPIGv2`.

    Attributes:
        n_tables: Number of independent LSH tables.
        n_bits: Bits per hash bucket.
        min_probes: Minimum candidates to retrieve.
        max_probes: Maximum candidates (adaptive cap).
        n_heads: Number of attention heads.
        head_dim: Dimension per head.
        target_entropy_ratio: If entropy / log(n_candidates) < this, expand
            probe budget.
        seed: RNG seed for projection matrices.
    """

    n_tables: int = 8
    hash_dim: int = 64
    min_probes: int = 64
    max_probes: int = 512
    n_heads: int = 8
    head_dim: int = 64
    budget: int = 128
    target_entropy_ratio: float = 0.7
    seed: int = 0


class MagicPIGv2:
    """LSH-based KV retrieval with adaptive per-layer probe budget.

    Parameters
    ----------
    config:
        MagicPIGv2 configuration.
    """

    def __init__(self, config: Optional[MagicPIGv2Config] = None) -> None:
        self._cfg = config or MagicPIGv2Config()
        self._rng_seed = self._cfg.seed
        # Projections are lazy-initialised on first use (depend on actual vector dim)
        self._projections: Optional[np.ndarray] = None
        self._input_dim: int = 0
        # Per-layer probe budget (starts at budget or min_probes)
        self._current_probes: int = max(self._cfg.budget, self._cfg.min_probes)
        self._probe_expansions: int = 0
        self._total_queries: int = 0

    @property
    def config(self) -> MagicPIGv2Config:
        return self._cfg

    @property
    def current_probes(self) -> int:
        return self._current_probes

    @property
    def probe_expansion_count(self) -> int:
        return self._probe_expansions

    def _ensure_projections(self, dim: int) -> None:
        """Lazily initialise projection matrices when input dimension is known."""
        if self._projections is None or self._input_dim != dim:
            rng = np.random.default_rng(self._rng_seed)
            self._projections = rng.standard_normal(
                (self._cfg.n_tables, self._cfg.hash_dim, dim)
            ).astype(np.float32)
            self._input_dim = dim

    def _hash(self, vectors: np.ndarray) -> np.ndarray:
        """Compute multi-table SimHash for a batch of vectors.

        Parameters
        ----------
        vectors:
            Shape ``(n, head_dim)``.

        Returns
        -------
        np.ndarray
            Shape ``(n_tables, n)`` integer hash codes.
        """
        v = np.asarray(vectors, dtype=np.float32)
        if v.ndim == 1:
            v = v[None, :]
        self._ensure_projections(v.shape[1])
        # projections: (n_tables, hash_dim, input_dim)
        # v: (n, input_dim) → projections @ v.T → (n_tables, hash_dim, n)
        signs = (np.einsum("tbd,nd->tbn", self._projections, v) > 0).astype(np.int32)
        # Encode hash_dim binary values as integer
        bits = 2 ** np.arange(self._cfg.hash_dim, dtype=np.int32)
        codes = (signs * bits[:, None]).sum(axis=1)  # (n_tables, n)
        return codes

    def retrieve_candidates(
        self,
        query: np.ndarray,
        keys: np.ndarray,
    ) -> np.ndarray:
        """Retrieve candidate key indices via LSH.

        Parameters
        ----------
        query:
            Shape ``(head_dim,)`` — single-head query vector.
        keys:
            Shape ``(seq_len, head_dim)`` — key vectors for one head.

        Returns
        -------
        np.ndarray
            Candidate indices, shape ``(n_candidates,)`` where
            n_candidates <= current_probes.
        """
        q = np.asarray(query, dtype=np.float32).reshape(1, -1)
        K = np.asarray(keys, dtype=np.float32)
        seq_len = len(K)

        q_codes = self._hash(q)  # (n_tables, 1)
        k_codes = self._hash(K)  # (n_tables, seq_len)

        # Collect matches: index l where any table matches
        match_scores = (q_codes == k_codes).sum(axis=0)  # (seq_len,) — match count
        top_k = min(self._current_probes, seq_len)
        candidates = np.argpartition(match_scores, -top_k)[-top_k:]
        return candidates

    def attend(
        self,
        query: np.ndarray,
        keys: np.ndarray,
        values: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Approximate attention via LSH candidate retrieval.

        Supports both single-head (1D/2D query) and multi-head (3D) modes.

        Parameters
        ----------
        query:
            Shape ``(head_dim,)`` for single-head or ``(n_heads, head_dim)``
            for multi-head.
        keys:
            Shape ``(seq_len, head_dim)`` for single-head or
            ``(n_heads, seq_len, head_dim)`` for multi-head.
        values:
            Same shape as keys.

        Returns
        -------
        Tuple of (output, candidate_indices).
        """
        q = np.asarray(query, dtype=np.float32)
        K = np.asarray(keys, dtype=np.float32)
        V = np.asarray(values, dtype=np.float32)

        if q.ndim == 1:
            # Single-head mode: q:(d,), keys:(seq,d), values:(seq,d)
            cands = self.retrieve_candidates(q, K)
            K_c = K[cands, :]
            V_c = V[cands, :]
            scale = float(q.shape[0] ** -0.5)
            scores = (K_c @ q) * scale
            scores -= scores.max()
            w = np.exp(scores)
            w /= w.sum() + 1e-9
            out = (V_c.T @ w).astype(np.float32)
            self._total_queries += 1
            return out, cands

        # Multi-head mode: q:(H,d), keys:(H,seq,d), values:(H,seq,d)
        scale = float(K.shape[-1] ** -0.5)
        n_heads = q.shape[0]
        outputs = []
        all_candidates = []
        for h in range(n_heads):
            cands = self.retrieve_candidates(q[h], K[h])
            all_candidates.append(cands)
            K_c = K[h, cands, :]
            V_c = V[h, cands, :]
            scores = (K_c @ q[h]) * scale
            scores -= scores.max()
            w = np.exp(scores)
            w /= w.sum()
            outputs.append(V_c.T @ w)

            if len(w) > 1:
                ent = float(-(w * np.log(w.clip(min=1e-10))).sum())
                max_ent = np.log(len(w))
                if max_ent > 0 and ent / max_ent < self._cfg.target_entropy_ratio:
                    new_probes = min(self._cfg.max_probes, self._current_probes * 2)
                    if new_probes > self._current_probes:
                        self._current_probes = new_probes
                        self._probe_expansions += 1

        self._total_queries += n_heads
        return np.stack(outputs, axis=0).astype(np.float32), all_candidates

    def reset_probe_budget(self) -> None:
        """Reset adaptive probe budget to minimum."""
        self._current_probes = self._cfg.min_probes

    def reset_stats(self) -> None:
        self._probe_expansions = 0
        self._total_queries = 0
