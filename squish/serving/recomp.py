"""RECOMP: Retrieval-Augmented LM context compression (arXiv 2310.04408, EMNLP 2023).

Extractive compressor: sentence-level SBERT cosine scoring retains top-k
sentences most relevant to the query.
Abstractive compressor: simulates T5-small summarisation by selecting and
joining the top-scoring sentences (production: replace with real T5 call).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional

import numpy as np

__all__ = [
    "RECOMPConfig",
    "RECOMPResult",
    "RECOMPCompressor",
]

_MODES = {"extractive", "abstractive"}


@dataclass
class RECOMPConfig:
    """Configuration for :class:`RECOMPCompressor`.

    Attributes:
        mode: Default compression mode — ``'extractive'`` or ``'abstractive'``.
        top_k: Maximum number of sentences to retain per document.
        max_length: Maximum character length of the compressed output string.
        seed: RNG seed for the deterministic SBERT simulation.
    """

    mode: str = "extractive"
    top_k: int = 3
    max_length: int = 512
    seed: int = 0

    def __post_init__(self) -> None:
        if self.mode not in _MODES:
            raise ValueError(
                f"mode must be one of {_MODES}, got {self.mode!r}"
            )
        if self.top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {self.top_k}")
        if self.max_length < 1:
            raise ValueError(f"max_length must be >= 1, got {self.max_length}")


@dataclass
class RECOMPResult:
    """Output of :class:`RECOMPCompressor`.

    Attributes:
        compressed_context: Compressed context string.
        source_count: Total sentences across all input documents.
        retained_count: Number of sentences retained.
        mode: Mode used (``'extractive'`` or ``'abstractive'``).
    """

    compressed_context: str
    source_count: int
    retained_count: int
    mode: str


class RECOMPCompressor:
    """RECOMP RAG context compressor.

    Extractive mode uses cosine-similarity sentence scoring (SBERT proxy:
    character-overlap bag-of-words similarity) to rank sentences against the
    query and keep the top-k most relevant ones.

    Abstractive mode further joins the retained sentences into a compact
    summary paragraph (production: replace inner join with T5-small generation).
    """

    def __init__(self, config: Optional[RECOMPConfig] = None) -> None:
        self._config = config or RECOMPConfig()

    @property
    def config(self) -> RECOMPConfig:
        return self._config

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        """Split text into sentences on full-stop / newline boundaries."""
        import re
        parts = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
        return [p.strip() for p in parts if p.strip()]

    @staticmethod
    def _bow_vector(text: str) -> np.ndarray:
        """Character-level bigram bag-of-words vector (fast SBERT proxy)."""
        text_lower = text.lower()
        bigrams: dict[str, int] = {}
        for i in range(len(text_lower) - 1):
            bg = text_lower[i : i + 2]
            bigrams[bg] = bigrams.get(bg, 0) + 1
        if not bigrams:
            return np.zeros(1, dtype=np.float32)
        keys = sorted(bigrams)
        return np.array([bigrams[k] for k in keys], dtype=np.float32)

    def _cosine_sim(self, a: str, b: str) -> float:
        """Cosine similarity of two texts via bigram BoW vectors."""
        va = self._bow_vector(a)
        vb = self._bow_vector(b)
        # Align dimensions
        n = max(len(va), len(vb))
        va = np.pad(va, (0, n - len(va)))
        vb = np.pad(vb, (0, n - len(vb)))
        norm_a = float(np.linalg.norm(va))
        norm_b = float(np.linalg.norm(vb))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(va, vb) / (norm_a * norm_b))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compress(
        self,
        documents: List[str],
        query: str,
        mode: Optional[str] = None,
    ) -> RECOMPResult:
        """Compress a list of retrieved documents relative to the query.

        Parameters
        ----------
        documents:
            Retrieved context passages (list of strings).
        query:
            The user query used to score relevance.
        mode:
            Override the config mode for this call.

        Returns
        -------
        RECOMPResult
        """
        cfg = self._config
        used_mode = mode if mode is not None else cfg.mode
        if used_mode not in _MODES:
            raise ValueError(f"mode must be one of {_MODES}, got {used_mode!r}")

        # Gather all sentences
        all_sentences: List[str] = []
        for doc in documents:
            all_sentences.extend(self._split_sentences(doc))

        source_count = len(all_sentences)

        if source_count == 0:
            return RECOMPResult(
                compressed_context="",
                source_count=0,
                retained_count=0,
                mode=used_mode,
            )

        # Score each sentence against the query
        scores = np.array(
            [self._cosine_sim(s, query) for s in all_sentences],
            dtype=np.float32,
        )

        # Select top-k unique sentences
        n_keep = min(cfg.top_k, source_count)
        top_indices = np.argsort(scores)[-n_keep:][::-1]
        # Preserve document order for readability
        top_indices_sorted = np.sort(top_indices)
        selected = [all_sentences[i] for i in top_indices_sorted]

        if used_mode == "extractive":
            compressed = " ".join(selected)
        else:
            # Abstractive: join selected sentences as a condensed paragraph
            # (production: replace with T5 generation)
            compressed = " ".join(selected)
            # Trim to max_length
            if len(compressed) > cfg.max_length:
                compressed = compressed[: cfg.max_length].rsplit(" ", 1)[0] + "…"

        if len(compressed) > cfg.max_length:
            compressed = compressed[: cfg.max_length].rsplit(" ", 1)[0] + "…"

        return RECOMPResult(
            compressed_context=compressed,
            source_count=source_count,
            retained_count=len(selected),
            mode=used_mode,
        )
