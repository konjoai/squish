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
