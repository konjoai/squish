"""LLMLingua-2: token-level prompt compression (arXiv 2403.12968, EMNLP 2024).

Data-distillation-trained binary classifier keeps/drops tokens to compress
prompts by 4–20× in ~15 ms, retaining 95%+ downstream quality on RAG and
summarisation benchmarks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

__all__ = [
    "LLMLingua2Config",
    "LLMLingua2Result",
    "LLMLingua2Compressor",
]


@dataclass
class LLMLingua2Config:
    """Configuration for :class:`LLMLingua2Compressor`.

    Attributes:
        target_ratio: Fraction of tokens to retain (0, 1).  A value of 0.3
            means keep 30% of the input tokens.
        min_tokens: Minimum retained tokens regardless of ratio.
        force_tokens: Whitelist of token strings that are *always* kept
            (e.g. ``["<|system|>", "<|user|>"]``).
        seed: RNG seed used for the deterministic classifier simulation.
    """

    target_ratio: float = 0.3
    min_tokens: int = 10
    force_tokens: List[str] = field(default_factory=list)
    seed: int = 0

    def __post_init__(self) -> None:
        if not (0 < self.target_ratio < 1):
            raise ValueError(
                f"target_ratio must be in (0, 1), got {self.target_ratio}"
            )
        if self.min_tokens < 1:
            raise ValueError(f"min_tokens must be >= 1, got {self.min_tokens}")


@dataclass
class LLMLingua2Result:
    """Output of :class:`LLMLingua2Compressor`.

    Attributes:
        compressed_tokens: Retained token strings.
        original_count: Number of tokens in the input.
        compressed_count: Number of tokens retained.
        ratio_achieved: ``compressed_count / original_count``.
        token_mask: Boolean mask over input tokens — ``True`` means kept.
    """

    compressed_tokens: List[str]
    original_count: int
    compressed_count: int
    ratio_achieved: float
    token_mask: np.ndarray  # shape (original_count,), dtype bool


class LLMLingua2Compressor:
    """LLMLingua-2 token-level prompt compressor.

    Simulates the binary keep/drop classifier via importance scoring
    (character-length proxy for lexical information content) so that the
    module is dependency-free and fully deterministic given ``seed``.

    In production the ``_score_tokens`` method would be replaced by a
    small fine-tuned encoder (e.g. XLM-RoBERTa-base sized) that outputs
    a keep-probability per token.
    """

    def __init__(self, config: Optional[LLMLingua2Config] = None) -> None:
        self._config = config or LLMLingua2Config()

    @property
    def config(self) -> LLMLingua2Config:
        return self._config

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _tokenize(self, text: str) -> List[str]:
        """Whitespace-split tokenisation (production: use a real tokenizer)."""
        return text.split()

    def _score_tokens(self, tokens: List[str]) -> np.ndarray:
        """Assign an importance score in [0, 1] to each token.

        Heuristic: longer, less-common tokens carry more information.
        Score = normalised character length + small noise for tie-breaking.
        """
        rng = np.random.default_rng(self._config.seed)
        lengths = np.array([len(t) for t in tokens], dtype=np.float32)
        if lengths.max() > 0:
            lengths /= lengths.max()
        noise = rng.uniform(0, 0.05, size=len(tokens)).astype(np.float32)
        return np.clip(lengths + noise, 0.0, 1.0)

    def _force_mask(self, tokens: List[str]) -> np.ndarray:
        """Return True for tokens that must always be kept."""
        forced = set(self._config.force_tokens)
        return np.array([t in forced for t in tokens], dtype=bool)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compress_tokens(
        self,
        tokens: List[str],
        target_ratio: Optional[float] = None,
    ) -> LLMLingua2Result:
        """Compress a pre-tokenized list using the binary keep/drop classifier.

        Parameters
        ----------
        tokens:
            Input token strings.
        target_ratio:
            Override config ``target_ratio`` for this call.

        Returns
        -------
        LLMLingua2Result
        """
        cfg = self._config
        ratio = target_ratio if target_ratio is not None else cfg.target_ratio
        if not (0 < ratio < 1):
            raise ValueError(f"target_ratio must be in (0, 1), got {ratio}")

        n = len(tokens)
        if n == 0:
            return LLMLingua2Result(
                compressed_tokens=[],
                original_count=0,
                compressed_count=0,
                ratio_achieved=0.0,
                token_mask=np.zeros(0, dtype=bool),
            )

        scores = self._score_tokens(tokens)
        forced = self._force_mask(tokens)

        # Number of tokens to keep
        n_keep = max(cfg.min_tokens, int(np.ceil(n * ratio)))
        n_keep = min(n_keep, n)

        # Always keep forced tokens; fill remaining budget by top score
        mask = forced.copy()
        n_forced = int(forced.sum())
        n_remaining = max(0, n_keep - n_forced)

        if n_remaining > 0:
            # Zero out forced positions so they don't compete
            candidate_scores = scores.copy()
            candidate_scores[forced] = -1.0
            top_indices = np.argsort(candidate_scores)[-n_remaining:]
            mask[top_indices] = True

        compressed = [t for t, keep in zip(tokens, mask) if keep]
        n_comp = len(compressed)

        return LLMLingua2Result(
            compressed_tokens=compressed,
            original_count=n,
            compressed_count=n_comp,
            ratio_achieved=n_comp / n if n > 0 else 0.0,
            token_mask=mask,
        )

    def compress(
        self,
        prompt: str,
        target_ratio: Optional[float] = None,
    ) -> LLMLingua2Result:
        """Compress a raw string prompt.

        Parameters
        ----------
        prompt:
            The raw prompt text to compress.
        target_ratio:
            Override config ``target_ratio`` for this call.

        Returns
        -------
        LLMLingua2Result
        """
        tokens = self._tokenize(prompt)
        return self.compress_tokens(tokens, target_ratio=target_ratio)
