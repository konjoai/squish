# [Experimental] This module is part of Squish v6+ (Waves 19–26).
# Proof-of-concept quality: API and behaviour may change without notice.
# For stable core functionality see Waves 1–12 in MODULES.md.
"""LongContextChunk — Semantic-boundary chunking for 1M+ token contexts.

Splits a very long token-embedding sequence into non-overlapping chunks at
detected topic boundaries rather than at fixed window positions.  Boundary
detection uses a cosine-distance signal between consecutive windows of
``min_chunk_size`` tokens, z-score thresholded at ``boundary_sensitivity``.
When no statistical boundary is found within ``max_chunk_size`` tokens, the
algorithm falls back to a forced split at the hard size limit, guaranteeing
no chunk exceeds ``max_chunk_size``.

Algorithm
---------
1. Divide the sequence into non-overlapping windows of ``min_chunk_size``
   tokens each; compute the mean embedding per window.
2. Compute the cosine distance between each consecutive pair of window means.
3. Z-score normalise the distances.  Candidate boundaries are windows where
   ``z > boundary_sensitivity``.
4. Map candidate window indices back to token positions and enforce:
   - Minimum gap of ``min_chunk_size`` between successive chunk starts.
   - Hard upper limit of ``max_chunk_size`` tokens per chunk (forced split).
5. Always return contiguous, non-overlapping ``(start, end)`` pairs that
   cover the entire sequence from ``0`` to ``seq_len``.

Typical usage::

    from squish.streaming.long_context_chunk import ChunkConfig, LongContextChunker
    import numpy as np

    cfg     = ChunkConfig(max_chunk_size=512, min_chunk_size=64,
                          boundary_sensitivity=2.0, embed_dim=128)
    chunker = LongContextChunker(cfg)

    embeddings = np.random.randn(2048, 128).astype(np.float32)
    chunks = chunker.chunk(embeddings)
    # e.g. [(0, 512), (512, 1024), (1024, 1536), (1536, 2048)]
    print(chunker.stats)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "ChunkConfig",
    "LongContextChunker",
    "ChunkStats",
]

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class ChunkConfig:
    """Configuration for :class:`LongContextChunker`.

    Parameters
    ----------
    max_chunk_size : int
        Hard upper bound on chunk length in tokens.  No chunk will exceed this
        value; the algorithm forces a split at this boundary if necessary.
    min_chunk_size : int
        Minimum chunk length in tokens.  Also the window size used for the
        boundary detection signal.  Must be >= 1 and < ``max_chunk_size``.
    boundary_sensitivity : float
        Z-score threshold above which a cosine-distance value is declared a
        semantic boundary.  Higher values produce fewer, larger chunks.
    embed_dim : int
        Embedding dimensionality of the input token features.
    """

    max_chunk_size: int = 512
    min_chunk_size: int = 64
    boundary_sensitivity: float = 2.0
    embed_dim: int = 128

    def __post_init__(self) -> None:
        if self.min_chunk_size < 1:
            raise ValueError("min_chunk_size must be >= 1")
        if self.max_chunk_size < 1:
            raise ValueError("max_chunk_size must be >= 1")
        if self.min_chunk_size >= self.max_chunk_size:
            raise ValueError(
                "min_chunk_size must be strictly less than max_chunk_size; "
                f"got min={self.min_chunk_size}, max={self.max_chunk_size}"
            )
        if self.boundary_sensitivity <= 0.0:
            raise ValueError("boundary_sensitivity must be > 0")
        if self.embed_dim < 1:
            raise ValueError("embed_dim must be >= 1")


# ---------------------------------------------------------------------------
# Stats dataclass
# ---------------------------------------------------------------------------


@dataclass
class ChunkStats:
    """Aggregate statistics for :class:`LongContextChunker`.

    Attributes
    ----------
    total_chunk_calls : int
        Total number of :meth:`~LongContextChunker.chunk` calls.
    total_tokens_chunked : int
        Total number of token positions processed.
    total_chunks_produced : int
        Total number of ``(start, end)`` chunk pairs returned.
    """

    total_chunk_calls: int = 0
    total_tokens_chunked: int = 0
    total_chunks_produced: int = 0

    @property
    def avg_chunk_size(self) -> float:
        """Mean chunk size in tokens across all :meth:`~LongContextChunker.chunk` calls."""
        return self.total_tokens_chunked / max(1, self.total_chunks_produced)


# ---------------------------------------------------------------------------
# LongContextChunker
# ---------------------------------------------------------------------------


class LongContextChunker:
    """Semantic-boundary token-sequence chunker.

    Parameters
    ----------
    config : ChunkConfig
        Chunker configuration.
    """

    def __init__(self, config: ChunkConfig) -> None:
        self._cfg = config
        self._stats = ChunkStats()

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def chunk(
        self,
        token_embeddings: np.ndarray,
    ) -> list[tuple[int, int]]:
        """Split *token_embeddings* into semantically coherent chunks.

        Parameters
        ----------
        token_embeddings : np.ndarray
            Token embedding matrix, shape ``(seq_len, embed_dim)``.

        Returns
        -------
        list[tuple[int, int]]
            List of ``(start, end)`` index pairs (``end`` is exclusive), in
            ascending order, non-overlapping, and covering the full sequence.
            Guaranteed: ``chunks[0][0] == 0`` and ``chunks[-1][1] == seq_len``.

        Raises
        ------
        ValueError
            If *token_embeddings* is not 2-D or has the wrong ``embed_dim``.
        """
        token_embeddings = np.asarray(token_embeddings, dtype=np.float32)
        if token_embeddings.ndim != 2:
            raise ValueError(
                f"token_embeddings must be 2-D (seq_len, embed_dim); "
                f"got ndim={token_embeddings.ndim}"
            )
        seq_len, embed_dim = token_embeddings.shape
        if embed_dim != self._cfg.embed_dim:
            raise ValueError(
                f"token_embeddings embed_dim must be {self._cfg.embed_dim}; "
                f"got {embed_dim}"
            )

        chunks = self._compute_chunks(token_embeddings, seq_len)
        self._stats.total_chunk_calls += 1
        self._stats.total_tokens_chunked += seq_len
        self._stats.total_chunks_produced += len(chunks)
        return chunks

    # ------------------------------------------------------------------ #
    # Properties                                                           #
    # ------------------------------------------------------------------ #

    @property
    def stats(self) -> ChunkStats:
        """Current aggregate statistics."""
        return self._stats

    def __repr__(self) -> str:
        return (
            f"LongContextChunker(max_chunk={self._cfg.max_chunk_size}, "
            f"min_chunk={self._cfg.min_chunk_size}, "
            f"sensitivity={self._cfg.boundary_sensitivity})"
        )

    # ------------------------------------------------------------------ #
    # Internal implementation                                              #
    # ------------------------------------------------------------------ #

    def _compute_chunks(
        self,
        token_embeddings: np.ndarray,
        seq_len: int,
    ) -> list[tuple[int, int]]:
        """Core chunking logic; seq_len is already validated."""
        min_w = self._cfg.min_chunk_size
        max_w = self._cfg.max_chunk_size

        # Degenerate: sequence fits in a single chunk.
        if seq_len <= max_w:
            return [(0, seq_len)]

        # Step 1: compute window mean embeddings.
        n_complete = seq_len // min_w          # number of complete windows
        if n_complete < 2:
            # Cannot compute even one distance; fall back to fixed splitting.
            return self._fixed_split(seq_len)

        window_means: list[np.ndarray] = []
        for w in range(n_complete):
            start = w * min_w
            end   = start + min_w
            window_means.append(token_embeddings[start:end].mean(axis=0))

        means_arr = np.stack(window_means, axis=0)  # (n_complete, embed_dim)

        # Step 2: cosine distances between consecutive windows.
        norms = np.linalg.norm(means_arr, axis=1, keepdims=True)
        normed = means_arr / (norms + 1e-10)
        # dot product of consecutive pairs → cosine similarity
        cos_sims = (normed[:-1] * normed[1:]).sum(axis=1)   # (n_complete-1,)
        cos_dists = 1.0 - cos_sims                           # (n_complete-1,)

        # Step 3: z-score normalise.
        d_mean = cos_dists.mean()
        d_std  = cos_dists.std()
        if d_std < 1e-10:
            # Uniform distances → no statistical boundaries; fixed split.
            return self._fixed_split(seq_len)

        z_scores = (cos_dists - d_mean) / d_std  # (n_complete-1,)

        # Step 4: candidate boundary token positions.
        # Window gap i corresponds to boundary BEFORE window i+1,
        # i.e. at token position (i+1) * min_w.
        boundary_candidates: list[int] = []
        for i, z in enumerate(z_scores):
            if z > self._cfg.boundary_sensitivity:
                boundary_candidates.append((i + 1) * min_w)

        # Step 5: enforce min/max chunk constraints and build chunk list.
        return self._enforce_constraints(boundary_candidates, seq_len)

    def _enforce_constraints(
        self,
        boundary_candidates: list[int],
        seq_len: int,
    ) -> list[tuple[int, int]]:
        """Turn raw boundary positions into valid (start, end) chunks.

        Rules applied in order:
        1. Skip candidates that would create a chunk shorter than ``min_chunk_size``
           from the preceding chunk start.
        2. Insert forced boundaries when the next candidate or sequence end
           would exceed ``max_chunk_size`` from the current chunk start.
        3. Always end the last chunk at ``seq_len``.
        """
        min_w = self._cfg.min_chunk_size
        max_w = self._cfg.max_chunk_size

        # Filter: deduplicate and sort (should already be sorted).
        candidates = sorted(set(boundary_candidates))

        chunks: list[tuple[int, int]] = []
        chunk_start = 0

        cand_iter = iter(candidates)
        next_cand: int | None = next(cand_iter, None)

        while chunk_start < seq_len:
            remaining = seq_len - chunk_start

            if remaining <= max_w:
                # Either we've consumed all candidates or the rest fits in
                # one chunk — but we still split at semantic boundaries if
                # they exist within this remaining window.
                # Collect valid candidates within [chunk_start+min_w, seq_len).
                sub_boundaries: list[int] = []
                cand = next_cand
                while cand is not None and cand < seq_len:
                    if cand - chunk_start >= min_w and cand < seq_len:
                        sub_boundaries.append(cand)
                    cand = next(cand_iter, None)
                next_cand = None  # exhausted

                # Emit chunks from sub-boundaries within max_w constraint.
                for b in sub_boundaries:
                    if b - chunk_start < min_w:
                        continue  # too small; skip
                    if b - chunk_start > max_w:
                        # Force-split before this boundary.
                        forced_end = chunk_start + max_w
                        chunks.append((chunk_start, forced_end))
                        chunk_start = forced_end
                    chunks.append((chunk_start, b))
                    chunk_start = b
                    if chunk_start >= seq_len:
                        break

                if chunk_start < seq_len:
                    chunks.append((chunk_start, seq_len))
                break

            # We must split before chunk_start + max_w.
            # Advance past candidates that are too close.
            while next_cand is not None and next_cand - chunk_start < min_w:
                next_cand = next(cand_iter, None)

            if next_cand is not None and next_cand - chunk_start <= max_w:
                # Use the semantic boundary.
                chunks.append((chunk_start, next_cand))
                chunk_start = next_cand
                next_cand = next(cand_iter, None)
            else:
                # No valid candidate within window; force-split at max_w.
                forced_end = chunk_start + max_w
                chunks.append((chunk_start, forced_end))
                chunk_start = forced_end

        return chunks

    def _fixed_split(self, seq_len: int) -> list[tuple[int, int]]:
        """Fall back: split into fixed-size ``max_chunk_size`` windows."""
        max_w = self._cfg.max_chunk_size
        chunks: list[tuple[int, int]] = []
        start = 0
        while start < seq_len:
            end = min(start + max_w, seq_len)
            chunks.append((start, end))
            start = end
        return chunks
