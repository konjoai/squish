"""StreamRAG — Mid-generation document injection for streaming RAG.

Documents can be injected into the RAG buffer at any generation step without
restarting the decode loop.  Injected documents are stored as token-ID
sequences ready for KV prefill and ranked by cosine similarity to a running
query embedding.

When the buffer is full the least-relevant document is evicted to make room
for the new arrival, so the injector always maintains the most useful document
set relative to the current query.

Typical usage::

    from squish.stream_rag import StreamRAGConfig, StreamRAGInjector
    import numpy as np

    cfg      = StreamRAGConfig(max_docs=4, embed_dim=256)
    injector = StreamRAGInjector(cfg)

    injector.inject(
        doc_id    = "doc-001",
        token_ids = np.array([1, 2, 3, 4], dtype=np.int64),
        embedding = np.random.randn(256).astype(np.float32),
    )

    query = np.random.randn(256).astype(np.float32)
    docs  = injector.retrieve(query, top_k=1)
    print(docs[0].relevance)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "StreamRAGConfig",
    "RAGDocument",
    "StreamRAGInjector",
    "StreamRAGStats",
]

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class StreamRAGConfig:
    """Configuration for :class:`StreamRAGInjector`.

    Parameters
    ----------
    max_docs : int
        Maximum number of documents buffered simultaneously.
    max_doc_tokens : int
        Maximum token-ID sequence length per document.
    embed_dim : int
        Expected dimensionality of document and query embeddings.
    top_k_retrieve : int
        Default number of documents returned by :meth:`~StreamRAGInjector.retrieve`
        when *top_k* is not explicitly specified.
    """

    max_docs: int = 8
    max_doc_tokens: int = 512
    embed_dim: int = 256
    top_k_retrieve: int = 3

    def __post_init__(self) -> None:
        if self.max_docs < 1:
            raise ValueError("max_docs must be >= 1")
        if self.max_doc_tokens < 1:
            raise ValueError("max_doc_tokens must be >= 1")
        if self.embed_dim < 1:
            raise ValueError("embed_dim must be >= 1")
        if self.top_k_retrieve < 1:
            raise ValueError("top_k_retrieve must be >= 1")


# ---------------------------------------------------------------------------
# RAGDocument
# ---------------------------------------------------------------------------


@dataclass
class RAGDocument:
    """A single document in the StreamRAG buffer.

    Attributes
    ----------
    doc_id : str
        Unique identifier for this document.
    token_ids : np.ndarray
        1-D int64 array of pre-tokenised token IDs.
    embedding : np.ndarray
        1-D float32 embedding vector for relevance ranking.
    relevance : float
        Most recently computed cosine-similarity relevance score.
        Initialised to ``0.0`` at injection time.
    """

    doc_id: str
    token_ids: np.ndarray
    embedding: np.ndarray
    relevance: float = 0.0


# ---------------------------------------------------------------------------
# Stats dataclass
# ---------------------------------------------------------------------------


@dataclass
class StreamRAGStats:
    """Aggregate statistics for :class:`StreamRAGInjector`.

    Attributes
    ----------
    total_injections : int
        Total number of successful :meth:`~StreamRAGInjector.inject` calls.
    total_retrievals : int
        Total number of :meth:`~StreamRAGInjector.retrieve` calls.
    total_evictions : int
        Total number of least-relevant document evictions.
    """

    total_injections: int = 0
    total_retrievals: int = 0
    total_evictions: int = 0

    # Mutable ref to the live document list — set by the injector.
    _docs_ref: list[RAGDocument] = field(default_factory=list, repr=False, compare=False)

    @property
    def avg_relevance(self) -> float:
        """Mean relevance score across all currently buffered documents.

        Returns ``0.0`` when the buffer is empty.
        """
        if not self._docs_ref:
            return 0.0
        return float(np.mean([d.relevance for d in self._docs_ref]))


# ---------------------------------------------------------------------------
# StreamRAGInjector
# ---------------------------------------------------------------------------


class StreamRAGInjector:
    """Mid-generation document injector with cosine-similarity ranking.

    Documents are held in a list.  On each :meth:`retrieve` call their
    ``relevance`` fields are updated with the current query's cosine
    similarity.  When the buffer is full, :meth:`inject` evicts the document
    with the *lowest* current relevance before inserting the new one.

    Parameters
    ----------
    config : StreamRAGConfig
        Injector configuration.
    """

    def __init__(self, config: StreamRAGConfig) -> None:
        self._cfg = config
        self._docs: list[RAGDocument] = []
        self._stats = StreamRAGStats(_docs_ref=self._docs)

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def inject(
        self,
        doc_id: str,
        token_ids: np.ndarray,
        embedding: np.ndarray,
    ) -> None:
        """Add a document to the buffer, evicting the least-relevant doc if full.

        Parameters
        ----------
        doc_id : str
            Unique identifier for the document.
        token_ids : np.ndarray
            1-D int64 token-ID sequence.  Length must be <= ``max_doc_tokens``.
        embedding : np.ndarray
            1-D float32 embedding, must have shape ``(embed_dim,)``.

        Raises
        ------
        ValueError
            If *token_ids* is too long or *embedding* has the wrong shape.
        """
        token_ids = np.asarray(token_ids, dtype=np.int64)
        embedding = np.asarray(embedding, dtype=np.float32)

        if token_ids.ndim != 1:
            raise ValueError(
                f"token_ids must be 1-D; got ndim={token_ids.ndim}"
            )
        if len(token_ids) > self._cfg.max_doc_tokens:
            raise ValueError(
                f"token_ids length {len(token_ids)} exceeds max_doc_tokens "
                f"({self._cfg.max_doc_tokens})"
            )
        if embedding.shape != (self._cfg.embed_dim,):
            raise ValueError(
                f"embedding shape must be ({self._cfg.embed_dim},); got {embedding.shape}"
            )

        if len(self._docs) >= self._cfg.max_docs:
            # Evict the document with the lowest relevance score.
            lru_idx = int(np.argmin([d.relevance for d in self._docs]))
            self._docs.pop(lru_idx)
            self._stats.total_evictions += 1

        doc = RAGDocument(
            doc_id=doc_id,
            token_ids=token_ids,
            embedding=embedding,
            relevance=0.0,
        )
        self._docs.append(doc)
        self._stats.total_injections += 1

    def retrieve(
        self,
        query_embedding: np.ndarray,
        top_k: int | None = None,
    ) -> list[RAGDocument]:
        """Return the top-*k* most relevant documents for *query_embedding*.

        Computes cosine similarity between *query_embedding* and every stored
        document's embedding, updates each document's ``relevance`` field, and
        returns the top documents in descending order of relevance.

        Parameters
        ----------
        query_embedding : np.ndarray
            1-D float32 query vector, shape ``(embed_dim,)``.
        top_k : int or None
            Number of documents to return.  Falls back to
            ``config.top_k_retrieve`` when ``None``.

        Returns
        -------
        list[RAGDocument]
            Up to *top_k* documents sorted by ``relevance`` descending.

        Raises
        ------
        ValueError
            If *query_embedding* has the wrong shape.
        """
        query_embedding = np.asarray(query_embedding, dtype=np.float32)
        if query_embedding.shape != (self._cfg.embed_dim,):
            raise ValueError(
                f"query_embedding shape must be ({self._cfg.embed_dim},); "
                f"got {query_embedding.shape}"
            )

        k = min(
            top_k if top_k is not None else self._cfg.top_k_retrieve,
            len(self._docs),
        )
        self._stats.total_retrievals += 1

        if not self._docs:
            return []

        q_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)

        for doc in self._docs:
            d_norm = doc.embedding / (np.linalg.norm(doc.embedding) + 1e-10)
            doc.relevance = float(np.dot(q_norm, d_norm))

        ranked = sorted(self._docs, key=lambda d: d.relevance, reverse=True)
        return ranked[:k]

    # ------------------------------------------------------------------ #
    # Properties                                                           #
    # ------------------------------------------------------------------ #

    @property
    def n_docs(self) -> int:
        """Number of documents currently buffered."""
        return len(self._docs)

    @property
    def stats(self) -> StreamRAGStats:
        """Current aggregate statistics."""
        return self._stats

    def __repr__(self) -> str:
        return (
            f"StreamRAGInjector(n_docs={self.n_docs}/{self._cfg.max_docs}, "
            f"injections={self._stats.total_injections})"
        )
