"""contrastive_decoding.py — Contrastive Decoding (CD) Sampler

Sharpens the generation quality by contrasting an expert logit distribution
against an amateur logit distribution:

    cd_logits = logits_expert - alpha * logits_amateur

The amateur represents a weaker, over-smoothed version of the expert:
  - A smaller sibling model (if available)
  - Early-exit logits from the same model (e.g., half-depth)
  - A high-temperature softmax (soft amateur)
  - A Dirichlet smoothed prior over the vocabulary

Additionally implements the Adaptive Plausibility Constraint (APC) from the
original paper: amplification only applies to tokens in the "plausible set"
where p_expert(t) ≥ beta * max(p_expert).  Tokens outside the plausible set
are masked to -inf.

Based on:
  - "Contrastive Decoding: Open-ended Text Generation as Optimization"
     Li et al., 2022 (ACL 2023)
  - "Contrastive Decoding Improves Reasoning in Large Language Models"
     O'Brien & Lewis, 2023

Usage:
    cdec = ContrastiveDecoder(alpha=0.5, beta=0.1)
    # At each decode step, supply both expert and amateur logits:
    cd_logits = cdec.contrast(logits_expert, logits_amateur)
    next_token = np.argmax(cd_logits)

    # Or sample from the CD distribution:
    next_token = cdec.sample(logits_expert, logits_amateur, temperature=0.7)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class ContrastiveDecoderConfig:
    """Configuration for ContrastiveDecoder.

    Args:
        alpha:       Contrastive weighting: cd = expert - alpha * amateur.
                     0.0 = no contrastive effect (pure expert).
                     0.5 = strong contrastive effect (recommended).
        beta:        Plausibility threshold for APC:
                     Only tokens with p_expert ≥ beta * max(p_expert) are
                     considered. 0.0 disables APC.
        amateur_mode: How to derive amateur logits when not externally supplied.
                     'high_temp' — softmax at high temperature (default, T=10)
                     'uniform' — uniform distribution over vocab
                     'entropy' — proportional to entropy-weighted expert logits
        amateur_temperature: Temperature used when amateur_mode='high_temp'.
        normalize:   If True, re-normalize cd_logits via log-softmax after APC.
    """
    alpha: float = 0.5
    beta: float = 0.1
    amateur_mode: str = "high_temp"        # 'high_temp' | 'uniform' | 'entropy'
    amateur_temperature: float = 10.0
    normalize: bool = True


@dataclass
class ContrastiveDecoderStats:
    """Runtime statistics for a ContrastiveDecoder."""
    total_calls: int = 0
    masked_fraction_sum: float = 0.0

    @property
    def mean_masked_fraction(self) -> float:
        if self.total_calls == 0:
            return 0.0
        return self.masked_fraction_sum / self.total_calls


class ContrastiveDecoder:
    """Expert/amateur contrastive logit adjustment with APC.

    Can be used with externally supplied amateur logits (from a small model or
    early-exit) or with self-derived amateurlogits via ``amateur_mode``.
    """

    def __init__(self, config: Optional[ContrastiveDecoderConfig] = None) -> None:
        self.config = config or ContrastiveDecoderConfig()
        if not 0.0 <= self.config.alpha <= 2.0:
            raise ValueError("alpha must be in [0, 2]")
        if not 0.0 <= self.config.beta <= 1.0:
            raise ValueError("beta must be in [0, 1]")
        if self.config.amateur_mode not in ("high_temp", "uniform", "entropy"):
            raise ValueError(
                "amateur_mode must be 'high_temp', 'uniform', or 'entropy'"
            )
        self.stats = ContrastiveDecoderStats()

    # ------------------------------------------------------------------
    # Core contrast
    # ------------------------------------------------------------------

    def contrast(
        self,
        logits_expert: np.ndarray,          # (vocab_size,) raw logits
        logits_amateur: Optional[np.ndarray] = None,  # (vocab_size,) or None
    ) -> np.ndarray:
        """Compute CD-adjusted logits.

        If logits_amateur is None, self-derives amateur logits based on
        the config's amateur_mode.

        Returns:
            np.ndarray of shape (vocab_size,) — CD-adjusted log-likelihoods.
        """
        if logits_expert.ndim != 1:
            raise ValueError(
                f"logits_expert must be 1-D, got shape {logits_expert.shape}"
            )
        cfg = self.config
        vocab_size = logits_expert.shape[0]

        expert = logits_expert.astype(np.float64)

        if logits_amateur is None:
            amateur = self._derive_amateur(expert)
        else:
            if logits_amateur.shape != logits_expert.shape:
                raise ValueError(
                    "logits_amateur must have the same shape as logits_expert"
                )
            amateur = logits_amateur.astype(np.float64)

        # Contrastive combination
        cd = expert - cfg.alpha * amateur  # (vocab_size,)

        # Adaptive Plausibility Constraint (APC)
        if cfg.beta > 0.0:
            p_expert = self._softmax(expert)
            threshold = cfg.beta * p_expert.max()
            mask = p_expert >= threshold
            n_masked = int((~mask).sum())
            cd = np.where(mask, cd, -np.inf)
            self.stats.masked_fraction_sum += n_masked / vocab_size
        else:
            self.stats.masked_fraction_sum += 0.0

        # Optionally normalize to log-probabilities
        if cfg.normalize:
            valid = cd > -1e9
            if valid.any():
                cd_valid = cd[valid]
                # log-softmax
                cd_valid -= cd_valid.max()
                cd_valid -= math.log(np.exp(cd_valid).sum() + 1e-12)
                cd[valid] = cd_valid

        self.stats.total_calls += 1
        return cd.astype(np.float32)

    def sample(
        self,
        logits_expert: np.ndarray,
        logits_amateur: Optional[np.ndarray] = None,
        temperature: float = 1.0,
    ) -> int:
        """Sample a token from the CD-adjusted distribution.

        Args:
            logits_expert:  Expert model raw logits (vocab_size,).
            logits_amateur: Amateur logits or None (auto-derived).
            temperature:    Softmax temperature (applied after CD).

        Returns:
            Sampled token index.
        """
        if temperature <= 0.0:
            raise ValueError("temperature must be > 0")

        cd = self.contrast(logits_expert, logits_amateur)
        # Apply temperature scaling
        cd_scaled = cd.astype(np.float64) / temperature
        probs = self._softmax(cd_scaled)
        # Ensure valid probability distribution
        probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        total = probs.sum()
        if total <= 0:
            return int(np.argmax(logits_expert))
        probs /= total
        return int(np.random.choice(len(probs), p=probs))

    # ------------------------------------------------------------------
    # Amateur derivation
    # ------------------------------------------------------------------

    def _derive_amateur(self, logits: np.ndarray) -> np.ndarray:
        """Derive amateur logits from expert logits based on amateur_mode."""
        cfg = self.config
        if cfg.amateur_mode == "high_temp":
            # Rescale logits to high temperature → over-smooth distribution
            return logits / cfg.amateur_temperature
        elif cfg.amateur_mode == "uniform":
            # Uniform logits (equal probability mass on all tokens)
            return np.zeros_like(logits)
        elif cfg.amateur_mode == "entropy":
            # Entropy-weighted: tokens near the uniform prior are amplified
            # amateur = logits * (1 - entropy_weight)
            p = self._softmax(logits)
            entropy_weight = float(-np.sum(p * np.log(p + 1e-12)) / math.log(len(logits)))
            return logits * (1.0 - entropy_weight)
        else:
            return np.zeros_like(logits)

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
        """Numerically stable softmax."""
        shifted = logits - logits.max(axis=axis, keepdims=True)
        exp_l = np.exp(shifted)
        return exp_l / (exp_l.sum(axis=axis, keepdims=True) + 1e-12)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def reset_stats(self) -> None:
        self.stats = ContrastiveDecoderStats()

    @property
    def effective_amplification(self) -> float:
        """Theoretical amplification factor at alpha=cfg.alpha."""
        return 1.0 / max(1.0 - self.config.alpha, 1e-3)

    def __repr__(self) -> str:
        cfg = self.config
        return (
            f"ContrastiveDecoder(alpha={cfg.alpha}, beta={cfg.beta}, "
            f"mode={cfg.amateur_mode}, calls={self.stats.total_calls})"
        )
