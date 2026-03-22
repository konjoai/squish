"""squish/sampling/cfg_sampler.py

CFGLogitsSampler — Classifier-Free Guidance for text generation.

Repurposes diffusion's CFG technique for LLM logit-space steering:

    logits_cfg = logits_uncond + w × (logits_cond − logits_uncond)

where ``logits_cond`` comes from the full prompt and ``logits_uncond`` comes
from a null / unconditional prefix.  Guidance scale ``w`` > 1 amplifies the
conditioning signal; ``w`` = 1 is equivalent to standard sampling.

Enables per-request style, formality, and sentiment steering without any
fine-tuning.  Typical overhead: 1.5–2× because both the conditioned and
unconditioned logits must be available each step (can share KV-cache for
the null prefix).

Reference
---------
Sanchez et al. "Stay on-topic with Classifier-Free Guidance."
arXiv:2306.17806, 2023.
"""

from __future__ import annotations

__all__ = ["CFGConfig", "CFGLogitsSampler"]

from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy import ndarray


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class CFGConfig:
    """Configuration for CFGLogitsSampler.

    Parameters
    ----------
    guidance_scale:
        Guidance weight ``w``.  1.0 = no guidance; > 1.0 = stronger
        conditioning; < 1.0 = steers away from conditioning.
    temperature:
        Softmax temperature applied to the merged logits.
    seed:
        RNG seed.
    """

    guidance_scale: float = 1.5
    temperature: float = 1.0
    seed: int = 0

    def __post_init__(self) -> None:
        if self.guidance_scale < 0.0:
            raise ValueError("guidance_scale must be >= 0")
        if self.temperature <= 0.0:
            raise ValueError("temperature must be > 0")


# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------

class CFGLogitsSampler:
    """Classifier-Free Guidance logit merger and sampler.

    Parameters
    ----------
    config:
        ``CFGConfig`` instance.
    """

    def __init__(self, config: CFGConfig) -> None:
        self.config = config
        self._rng = np.random.default_rng(config.seed)

    def merge_logits(
        self, logits_cond: ndarray, logits_uncond: ndarray
    ) -> ndarray:
        """Compute merged CFG logits.

        Parameters
        ----------
        logits_cond:
            Logits from the conditioned (full prompt) forward pass,
            shape ``(vocab_size,)``.
        logits_uncond:
            Logits from the unconditioned (null prefix) forward pass,
            shape ``(vocab_size,)``.

        Returns
        -------
        merged logits, shape ``(vocab_size,)``.
        """
        logits_cond = np.asarray(logits_cond, dtype=np.float32)
        logits_uncond = np.asarray(logits_uncond, dtype=np.float32)
        if logits_cond.shape != logits_uncond.shape:
            raise ValueError("logits_cond and logits_uncond must have the same shape")
        if logits_cond.ndim != 1:
            raise ValueError("logits must be 1-D")
        w = self.config.guidance_scale
        return logits_uncond + w * (logits_cond - logits_uncond)

    def sample(
        self, logits_cond: ndarray, logits_uncond: ndarray
    ) -> int:
        """Merge logits and sample one token.

        Parameters
        ----------
        logits_cond:
            Conditioned logits, shape ``(vocab_size,)``.
        logits_uncond:
            Unconditioned logits, shape ``(vocab_size,)``.

        Returns
        -------
        Sampled token index.
        """
        merged = self.merge_logits(logits_cond, logits_uncond)
        probs = self._softmax(merged / self.config.temperature)
        return int(self._rng.choice(len(probs), p=probs))

    def top_token(
        self, logits_cond: ndarray, logits_uncond: ndarray
    ) -> int:
        """Return the argmax of merged CFG logits."""
        merged = self.merge_logits(logits_cond, logits_uncond)
        return int(np.argmax(merged))

    def guidance_delta(
        self, logits_cond: ndarray, logits_uncond: ndarray
    ) -> ndarray:
        """Return the pure guidance delta = w × (cond − uncond)."""
        logits_cond = np.asarray(logits_cond, dtype=np.float32)
        logits_uncond = np.asarray(logits_uncond, dtype=np.float32)
        return self.config.guidance_scale * (logits_cond - logits_uncond)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _softmax(x: ndarray) -> ndarray:
        shifted = x - x.max()
        exp_x = np.exp(shifted)
        return exp_x / exp_x.sum()
