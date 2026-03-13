"""
squish/cocktail_kv.py

Cocktail — Chunk-Level Query-Similarity-Adaptive KV Quantization.

Based on:
  "Cocktail: Query-Similarity-Based Adaptive KV Cache Quantization for
  Long-Context LLM Inference"
  arXiv:2503.23294 (IEEE DATE 2025)

Problem
-------
For long-document queries (e.g. git diff analysis, code review), most of the
input context is irrelevant to the specific query.  Prior KV quantization
methods (KVTuner, DiffKV) calibrate per-layer properties — they never consider
whether an *input chunk* is query-relevant.  Aggressive quantization of
query-relevant chunks causes large accuracy loss; conservative quantization of
irrelevant chunks wastes memory.

Cocktail Solution
-----------------
1. Partition the input sequence into fixed-size chunks (default 32 tokens).
2. Compute a query-similarity score for each chunk:
   ``sim(chunk) = cos_sim(query_embedding, chunk_mean_embedding)``
3. Assign precision per chunk based on similarity rank:
   - High similarity → FP16 (query-relevant, protect)
   - Medium similarity → INT4
   - Low similarity → INT2

Hardware efficiency: reorder KV chunks by assigned precision before storage so
that all INT2 chunks are contiguous, all INT4 are contiguous, and all FP16 are
contiguous.  This enables coalesced memory access across each precision tier.

Squish Interaction
------------------
- **GemFilter synergy**: GemFilter reduces the input to top-K query-relevant
  tokens before prefill.  Cocktail assigns KV precision to input chunks based
  on query relevance *after* GemFilter selection.  Together: filter irrelevant
  tokens (GemFilter) → compress remaining irrelevant ones aggressively (Cocktail).
- **PM-KVQ synergy**: PM-KVQ handles temporal precision scheduling (when
  to compress during generation); Cocktail handles spatial-chunk precision
  (which input chunks to compress).  Orthogonal axes.

Provides
--------
  CocktailConfig      — configuration parameters
  ChunkSimilarityScorer — query-similarity score per chunk
  CocktailKVQuantizer — assign precision + quantize chunks
  CocktailKVStore     — store and retrieve chunks with precision maps
  CocktailStats       — tracking statistics
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "CocktailConfig",
    "ChunkSimilarityScorer",
    "CocktailKVQuantizer",
    "CocktailKVStore",
    "CocktailStats",
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class CocktailConfig:
    """Configuration for Cocktail query-similarity-based KV quantization.

    Parameters
    ----------
    chunk_size:
        Number of tokens per input chunk (default 32).
    fp16_fraction:
        Fraction of chunks assigned FP16 (default 0.15, top similarity).
    int2_fraction:
        Fraction of chunks assigned INT2 (default 0.50, bottom similarity).
    group_size:
        INT4/INT2 quantization group size (default 64).
    similarity_metric:
        How to compute chunk-query similarity.  ``"cosine"`` (default) or
        ``"dot"`` (faster, no normalisation).
    reorder_by_precision:
        If True, reorder chunks inside the KV store so that all INT2 are
        contiguous, then INT4, then FP16.  Enables coalesced memory reads.
    """

    chunk_size:            int   = 32
    fp16_fraction:         float = 0.15
    int2_fraction:         float = 0.50
    group_size:            int   = 64
    similarity_metric:     str   = "cosine"
    reorder_by_precision:  bool  = True

    def __post_init__(self) -> None:
        if self.chunk_size < 1:
            raise ValueError("chunk_size must be >= 1")
        if not 0.0 <= self.fp16_fraction <= 1.0:
            raise ValueError("fp16_fraction must be in [0, 1]")
        if not 0.0 <= self.int2_fraction <= 1.0:
            raise ValueError("int2_fraction must be in [0, 1]")
        if self.fp16_fraction + self.int2_fraction > 1.0:
            raise ValueError("fp16_fraction + int2_fraction must be <= 1.0")
        if self.similarity_metric not in ("cosine", "dot"):
            raise ValueError("similarity_metric must be 'cosine' or 'dot'")


# ---------------------------------------------------------------------------
# Chunk similarity scorer
# ---------------------------------------------------------------------------

class ChunkSimilarityScorer:
    """Compute query-similarity scores for input sequence chunks.

    Parameters
    ----------
    config:
        Cocktail configuration.
    """

    def __init__(self, config: CocktailConfig | None = None) -> None:
        self._cfg = config or CocktailConfig()

    def _embed(self, vectors: np.ndarray) -> np.ndarray:
        """Mean-pool a 2D array into a 1D embedding vector."""
        return vectors.mean(axis=0)

    def _cosine(self, a: np.ndarray, b: np.ndarray) -> float:
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na < 1e-8 or nb < 1e-8:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    def score_chunks(
        self,
        query_embedding: np.ndarray,
        token_embeddings: np.ndarray,
    ) -> np.ndarray:
        """Compute similarity scores for each chunk of ``token_embeddings``.

        Parameters
        ----------
        query_embedding:
            Float32 1D array of shape ``(embed_dim,)`` — the query embedding
            (e.g. mean-pooled query-token embeddings or a single query vector).
        token_embeddings:
            Float32 2D array of shape ``(seq_len, embed_dim)`` — input token
            representations (e.g. residual-stream vectors after embedding layer).

        Returns
        -------
        np.ndarray of shape ``(n_chunks,)`` with similarity scores in [0, 1].
        """
        q   = query_embedding.astype(np.float32)
        emb = token_embeddings.astype(np.float32)
        n   = emb.shape[0]
        cs  = self._cfg.chunk_size
        n_chunks = max(1, (n + cs - 1) // cs)

        scores = np.empty(n_chunks, dtype=np.float32)
        for i in range(n_chunks):
            chunk_emb = self._embed(emb[i * cs: (i + 1) * cs])
            if self._cfg.similarity_metric == "cosine":
                scores[i] = self._cosine(q, chunk_emb)
            else:
                scores[i] = float(np.dot(q, chunk_emb))

        # Normalise to [0, 1]
        s_min, s_max = scores.min(), scores.max()
        if s_max > s_min:
            scores = (scores - s_min) / (s_max - s_min)
        else:
            scores[:] = 0.5
        return scores

    def assign_bits(self, similarity_scores: np.ndarray) -> np.ndarray:
        """Assign bit-widths per chunk based on similarity rank.

        Parameters
        ----------
        similarity_scores:
            Float array of shape ``(n_chunks,)`` from ``score_chunks()``.

        Returns
        -------
        np.ndarray of shape ``(n_chunks,)`` with values in ``{2, 4, 16}``.
        """
        n       = len(similarity_scores)
        cfg     = self._cfg
        n_fp16  = max(1, int(np.round(cfg.fp16_fraction * n)))
        n_int2  = max(1, int(np.round(cfg.int2_fraction * n)))

        rank = np.argsort(-similarity_scores)  # descending similarity
        bits = np.full(n, 4, dtype=np.int32)
        bits[rank[:n_fp16]]           = 16
        bits[rank[n - n_int2:]]       = 2
        return bits


# ---------------------------------------------------------------------------
# Cocktail KV quantizer
# ---------------------------------------------------------------------------

class CocktailKVQuantizer:
    """Quantize / dequantize KV chunks at per-chunk assigned bit-widths.

    Parameters
    ----------
    config:
        Cocktail configuration.
    """

    def __init__(self, config: CocktailConfig | None = None) -> None:
        self._cfg = config or CocktailConfig()

    def quantize_chunk(
        self,
        kv_chunk: np.ndarray,
        bits: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Quantize one KV chunk.

        Parameters
        ----------
        kv_chunk:
            Float32 array of shape ``(chunk_size, head_dim)`` or any 2D.
        bits:
            Target bit-width (2, 4, or 16).

        Returns
        -------
        q_chunk : np.ndarray — quantized (int8 or float16)
        scales  : np.ndarray — per-group float32 scales (scalar [1] for FP16)
        """
        if bits == 16:
            return kv_chunk.astype(np.float16), np.ones(1, dtype=np.float32)

        flat     = kv_chunk.astype(np.float32).flatten()
        gs       = self._cfg.group_size
        n        = len(flat)
        levels   = (1 << bits) - 1
        n_groups = max(1, (n + gs - 1) // gs)
        padded   = np.zeros(n_groups * gs, dtype=np.float32)
        padded[:n] = flat
        groups   = padded.reshape(n_groups, gs)
        vmax     = np.abs(groups).max(axis=1, keepdims=True).clip(min=1e-7)
        scale    = vmax / max(levels // 2, 1)
        q        = np.round(groups / scale).clip(
            -(levels // 2), levels // 2
        ).astype(np.int8)
        return q.flatten()[:n].reshape(kv_chunk.shape), scale.flatten()

    def dequantize_chunk(
        self,
        q_chunk: np.ndarray,
        scales: np.ndarray,
        bits: int,
        original_shape: tuple,
    ) -> np.ndarray:
        """Reconstruct a float32 KV chunk.

        Parameters
        ----------
        q_chunk       : quantized array (int8 or float16)
        scales        : scales from ``quantize_chunk``
        bits          : bit-width used during quantization
        original_shape: target output shape

        Returns
        -------
        np.ndarray of dtype float32.
        """
        if bits == 16:
            return q_chunk.astype(np.float32).reshape(original_shape)

        flat     = q_chunk.astype(np.float32).flatten()
        gs       = self._cfg.group_size
        n        = len(flat)
        n_groups = len(scales)
        padded   = np.zeros(n_groups * gs, dtype=np.float32)
        padded[:n] = flat
        groups   = padded.reshape(n_groups, gs)
        dq       = (groups * scales.reshape(-1, 1)).flatten()[:n]
        return dq.reshape(original_shape)


# ---------------------------------------------------------------------------
# Cocktail KV Store
# ---------------------------------------------------------------------------

class CocktailKVStore:
    """Store and retrieve KV cache with Cocktail mixed-precision chunks.

    Chunks are optionally reordered by precision tier for coalesced access.

    Parameters
    ----------
    config:
        Cocktail configuration.
    """

    def __init__(self, config: CocktailConfig | None = None) -> None:
        self._cfg        = config or CocktailConfig()
        self._quantizer  = CocktailKVQuantizer(self._cfg)
        self._scorer     = ChunkSimilarityScorer(self._cfg)
        self._stats      = CocktailStats()
        self._chunks:    list[tuple[np.ndarray, np.ndarray, int]] = []
        self._chunk_bits: list[int] = []
        self._chunk_orig_shapes: list[tuple] = []

    @property
    def stats(self) -> CocktailStats:
        return self._stats

    def store(
        self,
        kv_matrix: np.ndarray,
        query_embedding: np.ndarray,
        token_embeddings: np.ndarray,
    ) -> None:
        """Quantize and store a full KV matrix with Cocktail assignment.

        Parameters
        ----------
        kv_matrix:
            Float32 array of shape ``(seq_len, head_dim)`` — the full KV
            sequence for one head.
        query_embedding:
            Float32 1D array — query embedding for similarity scoring.
        token_embeddings:
            Float32 array of shape ``(seq_len, embed_dim)`` — input token
            embeddings (used for chunk similarity scoring).
        """
        self._chunks.clear()
        self._chunk_bits.clear()
        self._chunk_orig_shapes.clear()

        n   = kv_matrix.shape[0]  # noqa: F841
        cs  = self._cfg.chunk_size

        # Score and assign bits
        sim_scores = self._scorer.score_chunks(query_embedding, token_embeddings)
        chunk_bits = self._scorer.assign_bits(sim_scores)

        self._stats.total_chunks += len(chunk_bits)
        self._stats.fp16_chunks  += int((chunk_bits == 16).sum())
        self._stats.int4_chunks  += int((chunk_bits == 4).sum())
        self._stats.int2_chunks  += int((chunk_bits == 2).sum())

        for i, bits in enumerate(chunk_bits):
            chunk = kv_matrix[i * cs: (i + 1) * cs]
            orig_shape = chunk.shape
            q, sc = self._quantizer.quantize_chunk(chunk, int(bits))
            self._chunks.append((q, sc, int(bits)))
            self._chunk_bits.append(int(bits))
            self._chunk_orig_shapes.append(orig_shape)

        # Optionally reorder for coalesced access: INT2 → INT4 → FP16
        if self._cfg.reorder_by_precision:
            order = sorted(
                range(len(self._chunks)),
                key=lambda x: self._chunk_bits[x]
            )
            self._chunks           = [self._chunks[j] for j in order]
            self._chunk_bits       = [self._chunk_bits[j] for j in order]
            self._chunk_orig_shapes = [self._chunk_orig_shapes[j] for j in order]

    def retrieve(self) -> np.ndarray:
        """Reconstruct the full KV matrix from stored chunks.

        Returns
        -------
        np.ndarray of dtype float32.
        """
        if not self._chunks:
            return np.empty((0,), dtype=np.float32)

        rows = []
        for (q, sc, bits), orig_shape in zip(self._chunks, self._chunk_orig_shapes, strict=False):
            dq = self._quantizer.dequantize_chunk(q, sc, bits, orig_shape)
            rows.append(dq)
        return np.concatenate(rows, axis=0)

    def reset(self) -> None:
        """Clear stored chunks."""
        self._chunks.clear()
        self._chunk_bits.clear()
        self._chunk_orig_shapes.clear()


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

@dataclass
class CocktailStats:
    """Track chunk-level precision distribution.

    Attributes
    ----------
    total_chunks    : total chunks processed
    fp16_chunks     : chunks assigned FP16 (query-relevant)
    int4_chunks     : chunks assigned INT4
    int2_chunks     : chunks assigned INT2 (irrelevant)
    """

    total_chunks: int = 0
    fp16_chunks:  int = 0
    int4_chunks:  int = 0
    int2_chunks:  int = 0

    @property
    def avg_bits(self) -> float:
        if self.total_chunks == 0:
            return 0.0
        return (
            16 * self.fp16_chunks + 4 * self.int4_chunks + 2 * self.int2_chunks
        ) / self.total_chunks

    @property
    def compression_ratio(self) -> float:
        """Bits saved relative to uniform FP16."""
        if self.total_chunks == 0:
            return 1.0
        return self.avg_bits / 16.0

    def reset(self) -> None:
        self.total_chunks = 0
        self.fp16_chunks  = 0
        self.int4_chunks  = 0
        self.int2_chunks  = 0
