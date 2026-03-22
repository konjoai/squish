"""Selective Context: self-information token pruning (arXiv 2304.01210, EACL 2024).

Per-token log-probability pruning: tokens whose log P under the LM exceeds
the information threshold τ (i.e., are too predictable) are discarded.
No secondary model is required — the logits already computed during prefill
are reused for free.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

__all__ = [
    "SelectiveContextConfig",
    "SelectiveContextResult",
    "SelectiveContextCompressor",
]


@dataclass
class SelectiveContextConfig:
    """Configuration for :class:`SelectiveContextCompressor`.

    Attributes:
        threshold: Self-information threshold τ.  Tokens with
            ``-log_prob < threshold`` (low information content) are pruned.
            Higher values prune more aggressively.
        min_tokens: Minimum number of tokens to retain regardless of threshold.
        seed: RNG seed (used only when synthetic log-probs are generated for
            testing).
    """

    threshold: float = 0.5
    min_tokens: int = 5
    seed: int = 0

    def __post_init__(self) -> None:
        if not (0 < self.threshold < 1):
            raise ValueError(
                f"threshold must be in (0, 1), got {self.threshold}"
            )
        if self.min_tokens < 1:
            raise ValueError(f"min_tokens must be >= 1, got {self.min_tokens}")


@dataclass
class SelectiveContextResult:
    """Output of :class:`SelectiveContextCompressor`.

    Attributes:
        compressed_tokens: Retained tokens.
        original_count: Number of input tokens.
        retained_count: Number of retained tokens.
        mask: Boolean mask — True if the token is retained.
    """

    compressed_tokens: List[str]
    original_count: int
    retained_count: int
    mask: np.ndarray  # shape (original_count,), dtype bool


class SelectiveContextCompressor:
    """Self-information token pruner (Selective Context).

    Tokens with low self-information (``-log_prob < threshold``) are
    considered redundant and discarded.  The threshold τ can be tuned per
    call; a value of 0.5 retains roughly 50% of a typical prompt at near-zero
    quality cost.

    When ``log_probs`` is ``None`` a synthetic score proportional to token
    character length is used, simulating the information content that a real
    language model would assign.
    """

    def __init__(self, config: Optional[SelectiveContextConfig] = None) -> None:
        self._config = config or SelectiveContextConfig()

    @property
    def config(self) -> SelectiveContextConfig:
        return self._config

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _synthetic_log_probs(self, tokens: List[str]) -> np.ndarray:
        """Generate deterministic synthetic log-probs for testing.

        Shorter tokens get log-probs closer to 0 (less informative).
        """
        rng = np.random.default_rng(self._config.seed)
        lengths = np.array([len(t) for t in tokens], dtype=np.float32)
        # Map to [-5, 0] range; longer = more negative (less probable = more informative)
        if lengths.max() > 0:
            lp = -5.0 * lengths / lengths.max()
        else:
            lp = np.zeros(len(tokens), dtype=np.float32)
        noise = rng.uniform(-0.1, 0.1, size=len(tokens)).astype(np.float32)
        return np.clip(lp + noise, -10.0, 0.0)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compress(
        self,
        tokens: List[str],
        log_probs: Optional[np.ndarray] = None,
        threshold: Optional[float] = None,
    ) -> SelectiveContextResult:
        """Prune low-self-information tokens.

        Parameters
        ----------
        tokens:
            Input token strings.
        log_probs:
            Optional ``(n_tokens,)`` float32 array of per-token log-probabilities
            from the language model.  When ``None`` synthetic scores are used.
        threshold:
            Override the config threshold for this call.

        Returns
        -------
        SelectiveContextResult
        """
        cfg = self._config
        tau = threshold if threshold is not None else cfg.threshold
        if not (0 < tau < 1):
            raise ValueError(f"threshold must be in (0, 1), got {tau}")

        n = len(tokens)
        if n == 0:
            return SelectiveContextResult(
                compressed_tokens=[],
                original_count=0,
                retained_count=0,
                mask=np.zeros(0, dtype=bool),
            )

        if log_probs is not None:
            lp = np.asarray(log_probs, dtype=np.float32)
            if lp.shape[0] != n:
                raise ValueError(
                    f"log_probs length {lp.shape[0]} != tokens length {n}"
                )
        else:
            lp = self._synthetic_log_probs(tokens)

        # Self-information: non-negative, higher = more informative
        self_info = -lp  # shape (n,)

        # Keep tokens whose self-information >= tau
        mask = self_info >= tau

        # Enforce minimum retention
        n_kept = int(mask.sum())
        if n_kept < cfg.min_tokens:
            # Promote the highest-self-info tokens up to min_tokens
            deficit = cfg.min_tokens - n_kept
            candidates = np.where(~mask)[0]
            if len(candidates) > 0:
                top = np.argsort(self_info[candidates])[-deficit:]
                mask[candidates[top]] = True

        compressed = [t for t, keep in zip(tokens, mask) if keep]
        retained = int(mask.sum())

        return SelectiveContextResult(
            compressed_tokens=compressed,
            original_count=n,
            retained_count=retained,
            mask=mask,
        )

    def compress_text(
        self,
        text: str,
        threshold: Optional[float] = None,
    ) -> SelectiveContextResult:
        """Convenience wrapper: whitespace-tokenize, compress, return result.

        Parameters
        ----------
        text:
            Raw string to compress.
        threshold:
            Override config threshold for this call.

        Returns
        -------
        SelectiveContextResult
        """
        tokens = text.split()
        return self.compress(tokens, threshold=threshold)
