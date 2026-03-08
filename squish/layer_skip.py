"""
squish/layer_skip.py

LayerSkip — early exit and self-speculative decoding for transformer models.

Based on:
  "LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding"
  — Elhoushi et al., Meta AI 2024  (arXiv:2404.16710)

Key insight
-----------
Transformer hidden states converge toward the final output distribution
earlier in the stack for "easy" tokens.  Training with layer-wise exit
losses allows a single model to produce usable token predictions well
before its final layer.

Two inference modes
-------------------
1. **Early-exit inference** — every forward pass stops at layer ``exit_layer``
   when the output confidence exceeds ``confidence_threshold``.  Saves
   (total_layers - exit_layer) / total_layers × compute per easy token.

2. **Self-speculative decoding** — the early-exit path acts as the *draft*
   model; the full model acts as the *verifier*.  No separate checkpoint is
   needed.  Draft tokens generated at exit_layer speed; verify batched over
   all draft tokens in one full forward pass.

This module provides:
  * ``EarlyExitConfig`` — thresholds and mode.
  * ``ConfidenceEstimator`` — computes various confidence signals from logits.
  * ``EarlyExitDecoder`` — drives an exit/speculative decode loop.
  * ``EarlyExitStats`` — lightweight counters returned after generation.

Usage::

    from squish.layer_skip import EarlyExitConfig, EarlyExitDecoder

    cfg = EarlyExitConfig(
        num_layers=32, exit_layer=16,
        confidence_threshold=0.9, mode="self_speculative",
        gamma=4,
    )
    dec = EarlyExitDecoder(
        full_forward=lambda ids, limit: model(ids, layer_limit=limit),
        config=cfg,
    )
    tok_ids, stats = dec.generate(prompt_ids, max_new_tokens=128)
    print(f"acceptance rate: {stats.acceptance_rate:.2%}")
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np

__all__ = [
    "EarlyExitConfig",
    "ConfidenceEstimator",
    "EarlyExitDecoder",
    "EarlyExitStats",
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class EarlyExitConfig:
    """Configuration for LayerSkip inference.

    Parameters
    ----------
    num_layers : int
        Total transformer layers in the model (e.g. 32).
    exit_layer : int
        Layer at which the early exit / draft is taken.  Must be in
        ``[1, num_layers-1]``.
    confidence_threshold : float
        Minimum confidence for accepting an early exit in ``"early_exit"``
        mode, or for immediately accepting a draft token in
        ``"self_speculative"`` mode.
    mode : str
        ``"early_exit"`` — always exit at ``exit_layer`` if confident;
        otherwise run full forward and exit at ``num_layers``.
        ``"self_speculative"`` — generate ``gamma`` draft tokens with early
        exit, then verify with one full-model forward.
    gamma : int
        Number of draft tokens per speculative step (only used in
        ``"self_speculative"`` mode).
    confidence_metric : str
        Which signal to compute: ``"max_prob"`` (default), ``"margin"``,
        or ``"neg_entropy"`` (normalised to [0,1]).
    """

    num_layers:           int   = 32
    exit_layer:           int   = 16
    confidence_threshold: float = 0.85
    mode:                 str   = "early_exit"
    gamma:                int   = 4
    confidence_metric:    str   = "max_prob"

    def __post_init__(self) -> None:
        if self.num_layers < 2:
            raise ValueError("num_layers must be ≥ 2")
        if not (1 <= self.exit_layer < self.num_layers):
            raise ValueError(
                f"exit_layer must be in [1, num_layers-1]; "
                f"got exit_layer={self.exit_layer}, num_layers={self.num_layers}"
            )
        if not 0.0 < self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be in (0, 1]")
        if self.mode not in ("early_exit", "self_speculative"):
            raise ValueError("mode must be 'early_exit' or 'self_speculative'")
        if self.gamma < 1:
            raise ValueError("gamma must be ≥ 1")
        if self.confidence_metric not in ("max_prob", "margin", "neg_entropy"):
            raise ValueError(
                "confidence_metric must be 'max_prob', 'margin', or 'neg_entropy'"
            )


# ---------------------------------------------------------------------------
# Confidence estimator
# ---------------------------------------------------------------------------

class ConfidenceEstimator:
    """Compute scalar confidence scores from a token-prediction distribution.

    Parameters
    ----------
    metric : str
        One of ``"max_prob"``, ``"margin"``, ``"neg_entropy"``.
    """

    def __init__(self, metric: str = "max_prob") -> None:
        if metric not in ("max_prob", "margin", "neg_entropy"):
            raise ValueError(
                "metric must be 'max_prob', 'margin', or 'neg_entropy'"
            )
        self._metric = metric

    def estimate(self, logits: np.ndarray) -> float:
        """Return a confidence score in [0, 1].

        Parameters
        ----------
        logits : (vocab_size,) float — raw model output for the last position.
        """
        logits = np.asarray(logits, dtype=np.float64)
        if logits.ndim != 1 or logits.size == 0:
            raise ValueError("logits must be a non-empty 1-D array")
        # Numerically stable softmax
        shifted = logits - logits.max()
        exp     = np.exp(shifted)
        probs   = exp / exp.sum()

        if self._metric == "max_prob":
            return float(probs.max())

        if self._metric == "margin":
            # Difference between top-1 and top-2 probability
            if len(probs) < 2:
                return float(probs.max())
            top2 = np.partition(probs, -2)[-2:]
            top2 = np.sort(top2)[::-1]
            return float(top2[0] - top2[1])

        # neg_entropy: normalise entropy to [0,1] so that 1 = fully determined
        vocab = len(probs)
        eps   = 1e-12
        h     = -float((probs * np.log(probs + eps)).sum())
        h_max = math.log(vocab) if vocab > 1 else 1.0
        return float(max(0.0, 1.0 - h / h_max))

    def top_token(self, logits: np.ndarray) -> int:
        """Return the argmax token id."""
        return int(np.argmax(np.asarray(logits)))


# ---------------------------------------------------------------------------
# Stats dataclass
# ---------------------------------------------------------------------------

@dataclass
class EarlyExitStats:
    """Generation statistics returned by :class:`EarlyExitDecoder`.

    Attributes
    ----------
    total_tokens_generated : int
    early_exits : int
        Tokens that were resolved at ``exit_layer`` (skipped the full model).
    full_forwards : int
        Tokens that required a full forward pass.
    accepted_draft : int   (self_speculative only)
        Draft tokens accepted by the verifier.
    rejected_draft : int   (self_speculative only)
        Draft tokens rejected.
    """

    total_tokens_generated: int = 0
    early_exits:            int = 0
    full_forwards:          int = 0
    accepted_draft:         int = 0
    rejected_draft:         int = 0

    @property
    def acceptance_rate(self) -> float:
        """Fraction of draft tokens accepted in self-speculative mode."""
        total = self.accepted_draft + self.rejected_draft
        return self.accepted_draft / total if total > 0 else 0.0

    @property
    def early_exit_rate(self) -> float:
        """Fraction of tokens resolved via early exit."""
        total = self.total_tokens_generated
        return self.early_exits / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

class EarlyExitDecoder:
    """Drive a layer-skip inference loop using a callable forward function.

    The decoder accepts a ``full_forward`` callable that must support an
    optional ``layer_limit`` kwarg.  When ``layer_limit=k`` is passed, the
    forward returns logits from layer ``k`` instead of the final layer.

    Parameters
    ----------
    full_forward : callable
        Signature: ``full_forward(token_ids: List[int], layer_limit: Optional[int] = None)
        -> np.ndarray`` of shape ``(vocab_size,)``.
    config : EarlyExitConfig
    """

    def __init__(
        self,
        full_forward: Callable[[List[int], Optional[int]], np.ndarray],
        config: EarlyExitConfig,
    ) -> None:
        self._fwd   = full_forward
        self._cfg   = config
        self._conf  = ConfidenceEstimator(config.confidence_metric)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        input_ids: List[int],
        max_new_tokens: int = 64,
    ) -> Tuple[List[int], EarlyExitStats]:
        """Generate tokens using the configured strategy.

        Parameters
        ----------
        input_ids : list[int]
            Prompt token ids.
        max_new_tokens : int
            Maximum new tokens to produce.

        Returns
        -------
        (generated_ids, stats) where ``generated_ids`` includes the prompt.
        """
        if self._cfg.mode == "early_exit":
            return self._generate_early_exit(list(input_ids), max_new_tokens)
        return self._generate_self_speculative(list(input_ids), max_new_tokens)

    # ------------------------------------------------------------------
    # Early-exit mode
    # ------------------------------------------------------------------

    def _generate_early_exit(
        self,
        ids: List[int],
        max_new: int,
    ) -> Tuple[List[int], EarlyExitStats]:
        stats = EarlyExitStats()
        for _ in range(max_new):
            # Try early-exit first
            early_logits = self._fwd(ids, self._cfg.exit_layer)
            conf = self._conf.estimate(early_logits)
            if conf >= self._cfg.confidence_threshold:
                tok = self._conf.top_token(early_logits)
                stats.early_exits += 1
            else:
                full_logits = self._fwd(ids, None)
                tok = self._conf.top_token(full_logits)
                stats.full_forwards += 1
            ids.append(tok)
            stats.total_tokens_generated += 1
        return ids, stats

    # ------------------------------------------------------------------
    # Self-speculative mode
    # ------------------------------------------------------------------

    def _generate_self_speculative(
        self,
        ids: List[int],
        max_new: int,
    ) -> Tuple[List[int], EarlyExitStats]:
        stats = EarlyExitStats()
        generated = 0

        while generated < max_new:
            gamma = min(self._cfg.gamma, max_new - generated)

            # --- Draft phase (early-exit forward) ---
            draft_ids: List[int] = []
            draft_logits: List[np.ndarray] = []
            ctx = list(ids)
            for _ in range(gamma):
                logits = self._fwd(ctx, self._cfg.exit_layer)
                tok    = self._conf.top_token(logits)
                draft_logits.append(logits)
                draft_ids.append(tok)
                ctx.append(tok)

            # --- Verify phase (full forward over draft sequence) ---
            # For simplicity: verify each draft token sequentially using full model.
            ctx_verify = list(ids)
            accepted: List[int] = []
            for di, d_tok in enumerate(draft_ids):
                verify_logits = self._fwd(ctx_verify, None)
                verify_tok    = self._conf.top_token(verify_logits)
                if verify_tok == d_tok:
                    accepted.append(d_tok)
                    ctx_verify.append(d_tok)
                    stats.accepted_draft += 1
                else:
                    # Reject — append verifier's token instead
                    accepted.append(verify_tok)
                    ctx_verify.append(verify_tok)
                    stats.rejected_draft += 1
                    break
                stats.full_forwards += 1

            ids.extend(accepted)
            generated += len(accepted)
            stats.total_tokens_generated += len(accepted)
            stats.early_exits += len(draft_ids)   # each draft used early-exit pass

        return ids, stats
