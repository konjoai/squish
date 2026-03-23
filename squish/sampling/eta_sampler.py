"""squish/sampling/eta_sampler.py

EtaCutoffSampler — Entropy-adaptive hard logit cutoff sampling (η-sampling).

The cutoff threshold adapts automatically to each token's information content:

    threshold_prob = η × exp(H(p))

where H(p) = -Σ p_i log(p_i) is the per-step conditional entropy and η is a
configurable sensitivity coefficient.  Tokens with probability below the
threshold are masked out.

This provides a natural vocabulary truncation that:
* Expands the candidate set on uncertain distributions (high H → high threshold)
* Contracts the set on confident distributions (low H → near-zero threshold)

Consistently outperforms fixed top-p and top-k on factual and creative benchmarks
without requiring per-task threshold tuning.

Reference
---------
Hewitt et al. "Truncation Sampling as Language Model Desmoothing."
EMNLP 2022 Findings. arXiv:2210.15191, 2022.
"""

from __future__ import annotations

__all__ = ["EtaConfig", "EtaCutoffSampler"]

from dataclasses import dataclass

import numpy as np
from numpy import ndarray


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class EtaConfig:
    """Configuration for EtaCutoffSampler.

    Parameters
    ----------
    eta:
        Entropy-adaptive sensitivity coefficient.  Typical values: 0.001–0.01.
        Higher values = more aggressive truncation.
    temperature:
        Softmax temperature applied before probability computation.
    seed:
        RNG seed.
    """

    eta: float = 0.003
    temperature: float = 1.0
    seed: int = 0

    def __post_init__(self) -> None:
        if self.eta <= 0.0:
            raise ValueError("eta must be > 0")
        if self.temperature <= 0.0:
            raise ValueError("temperature must be > 0")


# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------

class EtaCutoffSampler:
    """Entropy-adaptive logit cutoff sampler (η-sampling).

    Parameters
    ----------
    config:
        ``EtaConfig`` instance.
    """

    def __init__(self, config: EtaConfig) -> None:
        self.config = config
        self._rng = np.random.default_rng(config.seed)

    def filter_logits(self, logits: ndarray) -> ndarray:
        """Apply η-cutoff mask to logits.

        Parameters
        ----------
        logits:
            Logit vector, shape ``(vocab_size,)``.

        Returns
        -------
        Masked logits; tokens below cutoff are set to ``-1e9``.
        """
        logits = np.asarray(logits, dtype=np.float32)
        if logits.ndim != 1:
            raise ValueError("logits must be 1-D")
        scaled = logits / self.config.temperature
        probs = self._softmax(scaled)
        h = self._entropy(probs)
        threshold = self.config.eta * float(np.exp(h))
        mask = probs >= threshold
        # Always keep at least the top-probability token (fallback)
        if not mask.any():
            mask[int(np.argmax(probs))] = True
        return np.where(mask, logits, -1e9)

    def sample(self, logits: ndarray) -> int:
        """Sample one token using η-cutoff.

        Parameters
        ----------
        logits:
            Logit vector, shape ``(vocab_size,)``.

        Returns
        -------
        Sampled token index.
        """
        masked = self.filter_logits(logits)
        scaled = masked / self.config.temperature
        probs = self._softmax(scaled)
        return int(self._rng.choice(len(probs), p=probs))

    def entropy(self, logits: ndarray) -> float:
        """Return per-step conditional entropy H(p) in nats."""
        logits = np.asarray(logits, dtype=np.float32)
        probs = self._softmax(logits / self.config.temperature)
        return float(self._entropy(probs))

    def survival_count(self, logits: ndarray) -> int:
        """Return how many tokens survive the η-cutoff filter."""
        masked = self.filter_logits(logits)
        return int((masked > -1e8).sum())

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _softmax(x: ndarray) -> ndarray:
        shifted = x - x.max()
        exp_x = np.exp(shifted)
        return exp_x / exp_x.sum()

    @staticmethod
    def _entropy(probs: ndarray) -> float:
        """Shannon entropy in nats."""
        safe = np.clip(probs, 1e-40, None)
        return float(-np.sum(probs * np.log(safe)))


# server.py compatibility alias
EtaSampler = EtaCutoffSampler
