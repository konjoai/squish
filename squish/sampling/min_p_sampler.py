"""squish/sampling/min_p_sampler.py

MinPSampler — Dynamic minimum-probability threshold sampling.

Sets a per-step adaptive probability floor equal to ``p_min × p_max`` where
``p_max`` is the peak softmax probability at the current decoding step.  This
contrasts with top-p (nucleus) which uses a fixed cumulative probability mass:
Min-P adapts automatically to the sharpness of each distribution — when the
model is confident (high p_max) the floor rises, eliminating noise; when the
model is uncertain (low p_max) the floor lowers, preserving diversity.

Adopted as the default alternative to top-p in llama.cpp and Ollama in 2024–25.

Reference
---------
Nguyen et al. "Calibrating the Confidence of Large Language Models by
Eliciting Fidelity (Min-P sampling)." arXiv:2407.01082, 2024.
"""

from __future__ import annotations

__all__ = ["MinPConfig", "MinPSampler"]

from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy import ndarray


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class MinPConfig:
    """Configuration for MinPSampler.

    Parameters
    ----------
    p_min:
        Minimum probability factor.  The dynamic floor is set to
        ``p_min × p_max``.  Typical values: 0.05–0.15.
    temperature:
        Softmax temperature applied before probability computation.
        Set to 1.0 to disable.
    seed:
        RNG seed for reproducible sampling.
    """

    p_min: float = 0.05
    temperature: float = 1.0
    seed: int = 0

    def __post_init__(self) -> None:
        if not (0.0 < self.p_min < 1.0):
            raise ValueError("p_min must be in (0, 1)")
        if self.temperature <= 0.0:
            raise ValueError("temperature must be > 0")

    @property
    def min_p_factor(self) -> float:  # server.py compatibility alias
        return self.p_min


# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------

class MinPSampler:
    """Min-P dynamic probability-floor sampler.

    Parameters
    ----------
    config:
        ``MinPConfig`` instance.
    """

    def __init__(self, config: MinPConfig) -> None:
        self.config = config
        self._rng = np.random.default_rng(config.seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def filter_logits(self, logits: ndarray) -> ndarray:
        """Apply the Min-P logit mask and return masked logits.

        Tokens whose softmax probability falls below ``p_min × p_max``
        are set to ``-inf``.

        Parameters
        ----------
        logits:
            Raw logit vector, shape ``(vocab_size,)``.

        Returns
        -------
        Masked logits, shape ``(vocab_size,)``.
        """
        logits = np.asarray(logits, dtype=np.float32)
        if logits.ndim != 1:
            raise ValueError("logits must be 1-D")
        scaled = logits / self.config.temperature
        probs = self._softmax(scaled)
        p_max = float(probs.max())
        floor = self.config.p_min * p_max
        masked = np.where(probs >= floor, logits, -1e9)
        return masked

    def sample(self, logits: ndarray) -> int:
        """Sample one token index from Min-P filtered logits.

        Parameters
        ----------
        logits:
            Raw logit vector, shape ``(vocab_size,)``.

        Returns
        -------
        Sampled token index.
        """
        masked = self.filter_logits(logits)
        scaled = masked / self.config.temperature
        probs = self._softmax(scaled)
        return int(self._rng.choice(len(probs), p=probs))

    def top_token(self, logits: ndarray) -> int:
        """Return the argmax of Min-P filtered logits (deterministic)."""
        masked = self.filter_logits(logits)
        return int(np.argmax(masked))

    def survival_count(self, logits: ndarray) -> int:
        """Return how many tokens survive the Min-P filter."""
        logits = np.asarray(logits, dtype=np.float32)
        if logits.ndim != 1:
            raise ValueError("logits must be 1-D")
        probs = self._softmax(logits / self.config.temperature)
        p_max = float(probs.max())
        return int((probs >= self.config.p_min * p_max).sum())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _softmax(x: ndarray) -> ndarray:
        shifted = x - x.max()
        exp_x = np.exp(shifted)
        return exp_x / exp_x.sum()
