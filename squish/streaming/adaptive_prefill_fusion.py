"""
squish/streaming/adaptive_prefill_fusion.py

AdaptivePrefillFusion — Unified ToMe + LayerSkip + ChunkedPrefill controller.

Key insight
-----------
Steps 1A (chunked prefill), 1D (token merging), and 1E (layer skip) each
activate independently via separate flags.  In practice, the best combination
depends on the *complexity* of the incoming prompt:

  * **High-entropy prompts** (creative, diverse content): chunked prefill +
    dense attention → faithful representation, avoid ToMe quality loss.
  * **Medium-entropy prompts** (chat, QA, summarisation): ToMe layers 4–11 +
    chunked prefill → balanced speedup at minimal quality cost.
  * **Low-entropy prompts** (repetitive, code templates, boilerplate): aggressive
    ToMe + layer-skip at tail layers + n-gram speculation → maximum speed.

``PrefillFusionController`` estimates prompt complexity once at request start
using a lightweight token-frequency entropy probe (no model call required),
then returns a ``PrefillPlan`` that the caller can use to configure the
prefill dispatch.

Decisions made once per request; no per-token overhead.

Usage::

    from squish.streaming.adaptive_prefill_fusion import (
        PrefillFusionConfig,
        PrefillFusionController,
        PrefillComplexity,
    )

    cfg = PrefillFusionConfig(
        chunk_size=512,
        tome_r=16,
        tome_start_layer=4,
        tome_end_layer=11,
        exit_layer=24,
        early_exit_threshold=0.95,
    )
    controller = PrefillFusionController(cfg)
    plan = controller.plan(token_ids)

    if plan.use_chunk_prefill:
        print(f"chunked prefill — chunk_size={plan.chunk_size}")
    if plan.use_tome:
        print(f"token merging   — layers {plan.tome_start}–{plan.tome_end}")
    if plan.use_layer_skip:
        print(f"layer skip      — exit_layer={plan.exit_layer}")
"""

from __future__ import annotations

__all__ = [
    "PrefillComplexity",
    "PrefillFusionConfig",
    "PrefillPlan",
    "PrefillFusionController",
]

import math
from dataclasses import dataclass
from enum import Enum
from typing import List

import numpy as np


# ---------------------------------------------------------------------------
# Complexity enum
# ---------------------------------------------------------------------------

class PrefillComplexity(Enum):
    """Estimated complexity of an incoming prompt."""
    HIGH   = "high"    # high entropy → dense attention needed
    MEDIUM = "medium"  # average entropy → ToMe + chunked
    LOW    = "low"     # low entropy  → aggressive ToMe + layer-skip + ngram


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class PrefillFusionConfig:
    """Configuration for AdaptivePrefillFusion.

    Parameters
    ----------
    chunk_size : int
        Chunk size for chunked prefill (ignored for SHORT sequences).
    chunk_threshold : int
        Minimum prompt length before chunked prefill is considered.
    tome_r : int
        Token-merge pairs per layer (passed to ``TokenMergingConfig``).
    tome_start_layer : int
        First transformer layer to apply token merging.
    tome_end_layer : int
        Last transformer layer (inclusive) to apply token merging.
    tome_seq_threshold : int
        Minimum prompt length before ToMe is applied.
    exit_layer : int
        Layer at which the early-exit draft runs.
    early_exit_threshold : float
        Confidence threshold for layer-skip activation.
    high_entropy_cutoff : float
        Normalised per-token entropy above this → HIGH complexity.
    low_entropy_cutoff : float
        Normalised per-token entropy below this → LOW complexity.
    vocab_probe_size : int
        Vocabulary size used for entropy normalisation heuristic.
    """

    chunk_size:          int   = 512
    chunk_threshold:     int   = 512
    tome_r:              int   = 16
    tome_start_layer:    int   = 4
    tome_end_layer:      int   = 11
    tome_seq_threshold:  int   = 64
    exit_layer:          int   = 24
    early_exit_threshold: float = 0.95
    high_entropy_cutoff: float = 0.60
    low_entropy_cutoff:  float = 0.35
    vocab_probe_size:    int   = 50_000

    def __post_init__(self) -> None:
        if self.chunk_size < 1:
            raise ValueError(f"chunk_size must be ≥ 1; got {self.chunk_size}")
        if self.tome_start_layer > self.tome_end_layer:
            raise ValueError(
                f"tome_start_layer ({self.tome_start_layer}) must be "
                f"≤ tome_end_layer ({self.tome_end_layer})"
            )
        if not (0.0 < self.early_exit_threshold <= 1.0):
            raise ValueError(
                f"early_exit_threshold must be in (0,1]; "
                f"got {self.early_exit_threshold}"
            )
        if not (0.0 <= self.low_entropy_cutoff < self.high_entropy_cutoff <= 1.0):
            raise ValueError(
                "low_entropy_cutoff must be < high_entropy_cutoff and both in [0,1]; "
                f"got low={self.low_entropy_cutoff}, high={self.high_entropy_cutoff}"
            )


# ---------------------------------------------------------------------------
# Plan dataclass
# ---------------------------------------------------------------------------

@dataclass
class PrefillPlan:
    """Dispatch plan returned by :class:`PrefillFusionController`.

    Attributes
    ----------
    complexity : PrefillComplexity
        Estimated prompt complexity.
    entropy : float
        Normalised entropy estimate (0 = fully deterministic, 1 = uniform).
    use_chunk_prefill : bool
        Whether to use chunked prefill for this request.
    chunk_size : int
        Chunk size to use (only meaningful when *use_chunk_prefill* is True).
    use_tome : bool
        Whether to apply token merging during prefill.
    tome_start : int
        First layer index for token merging.
    tome_end : int
        Last layer index for token merging.
    tome_r : int
        Token-merge pairs per layer.
    use_layer_skip : bool
        Whether to enable adaptive layer-skip during decode.
    exit_layer : int
        Early-exit layer for layer-skip mode.
    """

    complexity:       PrefillComplexity
    entropy:          float
    use_chunk_prefill: bool
    chunk_size:       int
    use_tome:         bool
    tome_start:       int
    tome_end:         int
    tome_r:           int
    use_layer_skip:   bool
    exit_layer:       int


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------

class PrefillFusionController:
    """Estimates prompt complexity and returns an optimal :class:`PrefillPlan`.

    Complexity is estimated from the token-frequency entropy of the prompt
    alone — no model forward pass is needed.

    Parameters
    ----------
    config : PrefillFusionConfig
    """

    def __init__(self, config: PrefillFusionConfig) -> None:
        self._cfg = config

    def estimate_entropy(self, token_ids: List[int]) -> float:
        """Return normalised token-frequency entropy in ``[0, 1]``.

        This is a cheap proxy for prompt diversity: high entropy means a
        diverse, high-complexity sequence; low entropy means repetition.

        Parameters
        ----------
        token_ids : list[int]
            The prompt token IDs.  Must be non-empty.

        Returns
        -------
        float
            Normalised entropy in ``[0, 1]``.
        """
        if not token_ids:
            return 0.5  # neutral estimate for empty input

        counts: dict[int, int] = {}
        for t in token_ids:
            counts[t] = counts.get(t, 0) + 1

        n = len(token_ids)
        h = 0.0
        for c in counts.values():
            p = c / n
            h -= p * math.log(p + 1e-12)

        # Normalise by log(vocab_probe_size) as the theoretical maximum
        h_max = math.log(max(self._cfg.vocab_probe_size, len(counts) + 1))
        return float(np.clip(h / h_max, 0.0, 1.0))

    def classify(self, entropy: float) -> PrefillComplexity:
        """Map a normalised entropy value to a :class:`PrefillComplexity`."""
        if entropy >= self._cfg.high_entropy_cutoff:
            return PrefillComplexity.HIGH
        if entropy <= self._cfg.low_entropy_cutoff:
            return PrefillComplexity.LOW
        return PrefillComplexity.MEDIUM

    def plan(self, token_ids: List[int]) -> PrefillPlan:
        """Compute a :class:`PrefillPlan` for the given token sequence.

        Parameters
        ----------
        token_ids : list[int]
            Full prompt token IDs (before any compression).

        Returns
        -------
        PrefillPlan
        """
        cfg = self._cfg
        n   = len(token_ids)
        entropy    = self.estimate_entropy(token_ids)
        complexity = self.classify(entropy)

        # Chunked prefill: always beneficial for long prompts
        use_chunk = n >= cfg.chunk_threshold

        if complexity == PrefillComplexity.HIGH:
            # High entropy: dense attention, optionally chunked — no ToMe
            use_tome       = False
            use_layer_skip = False
        elif complexity == PrefillComplexity.MEDIUM:
            # Medium entropy: ToMe on mid layers + chunked prefill
            use_tome       = n >= cfg.tome_seq_threshold
            use_layer_skip = False
        else:
            # Low entropy: aggressive ToMe + layer-skip
            use_tome       = n >= cfg.tome_seq_threshold
            use_layer_skip = True

        return PrefillPlan(
            complexity       = complexity,
            entropy          = entropy,
            use_chunk_prefill = use_chunk,
            chunk_size       = cfg.chunk_size,
            use_tome         = use_tome,
            tome_start       = cfg.tome_start_layer,
            tome_end         = cfg.tome_end_layer,
            tome_r           = cfg.tome_r,
            use_layer_skip   = use_layer_skip,
            exit_layer       = cfg.exit_layer,
        )
