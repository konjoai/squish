"""squish/sampling/minp_sampler.py

MinPSampler — Minimum-probability threshold sampling.

Reference
---------
Nguyen et al. "Sampling with a Minimum Probability Threshold."
arXiv:2407.01082, 2024.

Algorithm
---------
Given logits l over the vocabulary:

1. Compute probabilities p = softmax(l).
2. Determine the maximum probability: p_max = max(p).
3. Set the dynamic probability floor: threshold = min_p_factor × p_max.
4. Mask out all tokens with p < threshold.
5. Renormalize and sample from the surviving tokens.

Unlike top-p (nucleus sampling), the floor adapts to the sharpness of the
distribution: when the model is confident (p_max → 1), the threshold is
high and only near-certain tokens pass. When the model is uncertain
(p_max low), the threshold is low, preserving diversity.

On creative generation benchmarks, min-p outperforms top-p at matching
temperature, sampling accuracy, and output diversity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class MinPConfig:
    """Configuration for MinPSampler.

    Parameters
    ----------
    min_p_factor:
        Floor multiplier applied to p_max (default 0.1).
        Typical range: 0.05–0.2.
    temperature:
        Temperature applied before softmax (1.0 = no change).
    top_k:
        Optional top-k pre-filter (0 = disabled).
    seed:
        RNG seed for reproducibility.
    """

    min_p_factor: float = 0.1
    temperature: float = 1.0
    top_k: int = 0
    seed: int = 0

    def __post_init__(self) -> None:
        if not 0.0 <= self.min_p_factor < 1.0:
            raise ValueError("min_p_factor must be in [0, 1)")
        if self.temperature <= 0:
            raise ValueError("temperature must be > 0")
        if self.top_k < 0:
            raise ValueError("top_k must be >= 0 (0 = disabled)")


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class MinPResult:
    """Output of a MinP sampling step.

    Parameters
    ----------
    token_id:
        Sampled token index.
    probability:
        Probability assigned to the sampled token (after filtering).
    n_candidates:
        Number of tokens that passed the min-p filter.
    threshold:
        The min-p threshold used (= min_p_factor × p_max).
    """

    token_id: int
    probability: float
    n_candidates: int
    threshold: float


# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------

class MinPSampler:
    """Min-p probability-floor sampler.

    Parameters
    ----------
    config:
        MinP configuration.
    """

    def __init__(self, config: Optional[MinPConfig] = None) -> None:
        self._cfg = config or MinPConfig()
        self._rng = np.random.default_rng(self._cfg.seed)

    @property
    def config(self) -> MinPConfig:
        return self._cfg

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        logits = logits - logits.max()
        exp = np.exp(logits)
        return exp / exp.sum()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def sample(self, logits: np.ndarray) -> MinPResult:
        """Sample a token using min-p filtering.

        Parameters
        ----------
        logits:
            Shape ``(vocab_size,)``.

        Returns
        -------
        MinPResult
        """
        logits = np.asarray(logits, dtype=np.float32).ravel()

        # Temperature scaling
        if self._cfg.temperature != 1.0:
            logits = logits / self._cfg.temperature

        # Optional top-k pre-filter
        if self._cfg.top_k > 0:
            k = min(self._cfg.top_k, len(logits))
            top_k_threshold = np.partition(logits, -k)[-k]
            logits = np.where(logits >= top_k_threshold, logits, -np.inf)

        probs = self._softmax(logits)

        # Min-p filter
        p_max = float(probs.max())
        threshold = self._cfg.min_p_factor * p_max
        mask = probs >= threshold

        filtered_probs = probs * mask
        # n_candidates counts tokens that actually have positive probability
        # (tokens zeroed out by top-k have prob=0.0 and should not be counted)
        n_candidates = int((filtered_probs > 0).sum())
        if n_candidates == 0:
            # Fallback: keep only the argmax token
            filtered_probs = np.zeros_like(probs)
            filtered_probs[int(probs.argmax())] = 1.0
            n_candidates = 1
        else:
            filtered_probs /= filtered_probs.sum()

        token_id = int(self._rng.choice(len(logits), p=filtered_probs))

        return MinPResult(
            token_id=token_id,
            probability=float(filtered_probs[token_id]),
            n_candidates=n_candidates,
            threshold=threshold,
        )

    def sample_batch(self, logits: np.ndarray) -> np.ndarray:
        """Sample a token for each row in a batch of logits.

        Parameters
        ----------
        logits:
            Shape ``(batch, vocab_size)``.

        Returns
        -------
        np.ndarray
            Shape ``(batch,)`` — sampled token ids (int32).
        """
        return np.array(
            [self.sample(logits[i]).token_id for i in range(len(logits))],
            dtype=np.int32,
        )

    def filter_logits(self, logits: np.ndarray) -> np.ndarray:
        """Apply min-p filtering and return filtered log-probs.

        Tokens below the threshold receive ``-inf``; the rest are
        log-softmax normalized.

        Parameters
        ----------
        logits:
            Shape ``(vocab_size,)``.

        Returns
        -------
        np.ndarray
            Shape ``(vocab_size,)`` — filtered log-probabilities.
        """
        logits = np.asarray(logits, dtype=np.float32).ravel()
        probs = self._softmax(logits / self._cfg.temperature)
        p_max = float(probs.max())
        threshold = self._cfg.min_p_factor * p_max
        mask = probs >= threshold
        filtered = np.where(mask, probs, 0.0)
        total = filtered.sum()
        if total == 0:
            filtered[int(probs.argmax())] = 1.0
            total = 1.0
        filtered /= total
        log_probs = np.where(mask, np.log(np.maximum(filtered, 1e-38)), -np.inf)
        return log_probs.astype(np.float32)
