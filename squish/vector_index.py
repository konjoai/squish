"""
squish/vector_index.py

Thin wrapper around hnswlib for retrieval-attention (Phase 2).

Provides an ANNS (approximate nearest-neighbour search) index over token
key vectors stored in the disk KV tier.  Only head 0 is indexed as a
representative; retrieval recalls token positions for all heads.

Install the optional dependency with::

    pip install squish[retrieval]   # adds hnswlib

or manually::

    pip install hnswlib
"""
import numpy as np

try:
    import hnswlib as _hnswlib  # type: ignore[import]
    _HNSWLIB_AVAILABLE = True
except ImportError:
    _HNSWLIB_AVAILABLE = False


class HNSWIndex:
    """
    Approximate nearest-neighbour index over float32 vectors.

    Uses inner-product (cosine-equivalent for normalised vectors) space.

    Parameters
    ----------
    dim             : vector dimension
    max_elements    : pre-allocated capacity (can be resized later)
    M               : HNSW connectivity parameter (higher = better recall, more RAM)
    ef_construction : build-time search size (higher = better quality, slower build)
    """

    def __init__(
        self,
        dim: int,
        max_elements: int = 500_000,
        M: int = 16,
        ef_construction: int = 200,
    ):
        if not _HNSWLIB_AVAILABLE:
            raise ImportError(
                "hnswlib is required for retrieval attention.\n"
                "Install with: pip install hnswlib\n"
                "Or: pip install squish[retrieval]"
            )
        self._dim = dim
        self._index = _hnswlib.Index(space="ip", dim=dim)
        self._index.init_index(
            max_elements=max_elements,
            M=M,
            ef_construction=ef_construction,
            random_seed=42,
        )
        self._index.set_ef(max(64, M * 4))   # query-time search width
        self._count = 0

    # ── Public API ─────────────────────────────────────────────────────────────

    def add(self, vectors: np.ndarray, ids: np.ndarray) -> None:
        """
        Add a batch of vectors with associated token-position IDs.

        Parameters
        ----------
        vectors : (n, dim) float32
        ids     : (n,) int64 — token position indices in the disk KV tier
        """
        if len(vectors) == 0:
            return
        vecs = np.asarray(vectors, dtype=np.float32)
        if not vecs.flags["C_CONTIGUOUS"]:
            vecs = np.ascontiguousarray(vecs)
        self._index.add_items(vecs, ids.astype(np.int64))
        self._count += len(vectors)

    def search(self, query: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Return the ``top_k`` most similar token positions.

        Parameters
        ----------
        query : (dim,) or (1, dim) float32
        top_k : int

        Returns
        -------
        ids       : (k,) int64 — token positions (k ≤ top_k)
        distances : (k,) float32 — inner-product scores (higher = more similar)
        """
        if self._count == 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float32)
        q = np.asarray(query, dtype=np.float32).reshape(1, -1)
        if not q.flags["C_CONTIGUOUS"]:
            q = np.ascontiguousarray(q)
        k = min(top_k, self._count)
        ids, dists = self._index.knn_query(q, k=k)
        return ids[0].astype(np.int64), dists[0].astype(np.float32)

    def save(self, path) -> None:
        """Persist the index to ``path`` (hnswlib binary format)."""
        self._index.save_index(str(path))

    def load(self, path) -> None:
        """Load a previously saved index from ``path``."""
        self._index.load_index(
            str(path),
            max_elements=self._index.get_max_elements(),
        )

    @property
    def count(self) -> int:
        """Number of vectors currently indexed."""
        return self._count

    @property
    def dim(self) -> int:
        """Vector dimension."""
        return self._dim


# ---------------------------------------------------------------------------
# MRL (Matryoshka Representation Learning) Index
# ---------------------------------------------------------------------------

class MRLIndex:
    """
    Two-stage approximate nearest-neighbour search using Matryoshka embeddings.

    Matryoshka Representation Learning (MRL) trains embedding models so that
    *any prefix* of the full embedding vector is a valid, progressively more
    accurate, lower-dimensional representation.

    MRLIndex exploits this property for **efficient two-stage retrieval**:

    Stage 1 — Coarse search (``coarse_dim`` dimensions):
        Use only the first ``coarse_dim`` dimensions of each embedding.
        Retrieve ``coarse_k`` candidates cheaply.

    Stage 2 — Re-rank (``full_dim`` dimensions):
        Score the coarse candidates using the full embedding vector.
        Return the top-``k`` results.

    Memory: the coarse index stores only ``coarse_dim / full_dim`` of the
    full embedding data, and the coarse search scans only that reduced space.

    Parameters
    ----------
    full_dim   : int — dimension of the full embedding vector
    coarse_dim : int — dimension used for Stage 1 (must be ≤ full_dim)
    coarse_k   : int — number of candidates to retrieve in Stage 1
                       (default = 5 × top_k at search time)
    normalize  : bool — if True, L2-normalise vectors before storing/querying
    """

    def __init__(
        self,
        full_dim:   int,
        coarse_dim: int,
        coarse_k:   int = 0,
        normalize:  bool = True,
    ) -> None:
        if full_dim < 1:
            raise ValueError("full_dim must be ≥ 1")
        if coarse_dim < 1 or coarse_dim > full_dim:
            raise ValueError("coarse_dim must be in [1, full_dim]")
        self._full_dim    = full_dim
        self._coarse_dim  = coarse_dim
        self._coarse_k    = coarse_k   # 0 → auto (5 × top_k)
        self._normalize   = normalize

        # Storage: numpy arrays grown on demand for the pure-numpy fallback
        self._full_vecs:   list[np.ndarray] = []   # full-dim vectors
        self._ids:         list[int]        = []
        self._count:       int              = 0

    # ── Public API ─────────────────────────────────────────────────────────────

    def add(self, vectors: np.ndarray, ids: np.ndarray) -> None:
        """
        Add a batch of full-dimension embedding vectors.

        Parameters
        ----------
        vectors : (n, full_dim) float32
        ids     : (n,) int64 — identifies stored items (e.g. token positions)
        """
        vecs = np.asarray(vectors, dtype=np.float32)
        ids_ = np.asarray(ids, dtype=np.int64).flatten()
        if vecs.ndim == 1:
            vecs = vecs.reshape(1, -1)
        if vecs.shape[1] != self._full_dim:
            raise ValueError(
                f"Expected vectors of dim {self._full_dim}, got {vecs.shape[1]}"
            )
        if self._normalize:
            norms = np.linalg.norm(vecs, axis=1, keepdims=True).clip(1e-12)
            vecs  = vecs / norms

        for v, item_id in zip(vecs, ids_):
            self._full_vecs.append(v)
            self._ids.append(int(item_id))
        self._count += len(ids_)

    def search(self, query: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Two-stage MRL nearest-neighbour search.

        Parameters
        ----------
        query  : (full_dim,) float32
        top_k  : int — final number of results to return

        Returns
        -------
        ids       : (k,) int64
        distances : (k,) float32 — full-dim dot-product scores (higher = closer)
        """
        if self._count == 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float32)

        q = np.asarray(query, dtype=np.float32).flatten()
        if q.shape[0] != self._full_dim:
            raise ValueError(
                f"Query dim {q.shape[0]} does not match index dim {self._full_dim}"
            )
        if self._normalize:
            q = q / (np.linalg.norm(q) + 1e-12)

        # ── Stage 1: coarse retrieval ─────────────────────────────────────────
        ck = self._coarse_k if self._coarse_k > 0 else max(top_k * 5, 20)
        ck = min(ck, self._count)

        q_coarse  = q[: self._coarse_dim]
        all_coarse = np.stack(
            [v[: self._coarse_dim] for v in self._full_vecs]
        )                                              # (N, coarse_dim)
        coarse_scores = all_coarse @ q_coarse          # (N,) dot products

        cand_idx  = np.argpartition(coarse_scores, -ck)[-ck:]

        # ── Stage 2: full-dim re-rank ─────────────────────────────────────────
        cand_vecs = np.stack([self._full_vecs[i] for i in cand_idx])
        full_scores = cand_vecs @ q                    # (ck,)
        k_out     = min(top_k, len(cand_idx))
        top_local = np.argpartition(full_scores, -k_out)[-k_out:]
        top_local = top_local[np.argsort(full_scores[top_local])[::-1]]

        result_ids   = np.array([self._ids[cand_idx[i]] for i in top_local], dtype=np.int64)
        result_dists = full_scores[top_local].astype(np.float32)
        return result_ids, result_dists

    @property
    def count(self) -> int:
        """Number of vectors in the index."""
        return self._count

    @property
    def full_dim(self) -> int:
        return self._full_dim

    @property
    def coarse_dim(self) -> int:
        return self._coarse_dim
