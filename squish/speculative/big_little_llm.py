"""squish/speculative/big_little_llm.py

BigLittleLLM — Confidence-Based Token Routing to Small/Large Model.

Reference
---------
Kim et al. "Big-Little Decoder: A Novel Approach for Inference
Acceleration in LLMs." EMNLP 2023 / production 2024 (arXiv:2302.07863).

Algorithm
---------
BigLittleLLM routes each token to either a smaller (faster) model or the
larger (more accurate) model based on confidence:

1. Run the small model and compute the top-1 probability p_max.
2. If p_max >= confidence_threshold, accept the small model's token
   directly (no large model needed).
3. Otherwise, forward the context to the large model for a verified token.
4. Optionally, use the large model output as a correction signal and
   update the routing threshold dynamically.

This achieves ~40% oracle token savings on typical workloads.

Key properties
--------------
* NumPy-only.
* ``confidence_threshold`` — router threshold (default 0.9).
* ``dynamic_threshold`` — if True, adapt threshold to hit target savings.
* ``target_small_fraction`` — target fraction of tokens from small model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

__all__ = [
    "BigLittleLLMConfig",
    "RoutingDecision",
    "BigLittleLLM",
]


@dataclass
class BigLittleLLMConfig:
    """Configuration for :class:`BigLittleLLM`.

    Attributes:
        confidence_threshold: Confidence above which the small model is used.
        dynamic_threshold: Enable adaptive threshold to hit target usage.
        target_small_fraction: Target fraction of tokens served by small model.
        vocab_size: Vocabulary size.
        temperature: Sampling temperature.
    """

    confidence_threshold: float = 0.90
    dynamic_threshold: bool = True
    target_small_fraction: float = 0.40
    vocab_size: int = 32000
    temperature: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 < self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be in (0, 1]")


@dataclass
class RoutingDecision:
    """Result of a routing step.

    Attributes:
        used_small: True if the small model was used.
        token: Selected token ID.
        confidence: Top-1 probability from the small model.
    """

    used_small: bool
    token: int
    confidence: float


class BigLittleLLM:
    """Big-Little LLM token router.

    Parameters
    ----------
    config:
        BigLittleLLM configuration.
    seed:
        RNG seed.
    """

    def __init__(self, config: Optional[BigLittleLLMConfig] = None, seed: int = 0) -> None:
        self._cfg = config or BigLittleLLMConfig()
        self._rng = np.random.default_rng(seed)
        self._threshold: float = self._cfg.confidence_threshold
        self._small_count: int = 0
        self._large_count: int = 0

    @property
    def config(self) -> BigLittleLLMConfig:
        return self._cfg

    @property
    def small_fraction(self) -> float:
        total = self._small_count + self._large_count
        return self._small_count / total if total > 0 else 0.0

    @property
    def effective_threshold(self) -> float:
        return self._threshold

    def route(
        self,
        small_logits: np.ndarray,
        large_logits: Optional[np.ndarray] = None,
    ) -> RoutingDecision:
        """Route a single token to small or large model.

        Parameters
        ----------
        small_logits:
            Logits from the small model, shape ``(vocab_size,)``.
        large_logits:
            Logits from the large model.  If None, the large model is
            assumed not yet run (it will be indicated by the routing decision).

        Returns
        -------
        RoutingDecision
        """
        sl = np.asarray(small_logits, dtype=np.float64)
        if self._cfg.temperature != 1.0:
            sl /= max(self._cfg.temperature, 1e-6)
        sl -= sl.max()
        small_probs = np.exp(sl)
        small_probs /= small_probs.sum()
        confidence = float(small_probs.max())
        best_small = int(small_probs.argmax())

        use_small = confidence >= self._threshold

        if use_small:
            self._small_count += 1
            token = best_small
        else:
            self._large_count += 1
            if large_logits is not None:
                ll = np.asarray(large_logits, dtype=np.float64)
                if self._cfg.temperature != 1.0:
                    ll /= max(self._cfg.temperature, 1e-6)
                ll -= ll.max()
                large_probs = np.exp(ll)
                large_probs /= large_probs.sum()
                token = int(large_probs.argmax())
            else:
                # Large model not available; fall back to small model
                token = best_small

        # Adapt threshold
        if self._cfg.dynamic_threshold:
            self._adapt_threshold()

        return RoutingDecision(used_small=use_small, token=token, confidence=confidence)

    def _adapt_threshold(self) -> None:
        """Nudge threshold to drive small_fraction toward target."""
        if self._small_count + self._large_count < 10:
            return
        if self.small_fraction < self._cfg.target_small_fraction:
            self._threshold = max(0.0, self._threshold - 0.005)
        else:
            self._threshold = min(1.0, self._threshold + 0.005)

    def reset_stats(self) -> None:
        self._small_count = 0
        self._large_count = 0

    def reset_threshold(self) -> None:
        self._threshold = self._cfg.confidence_threshold
