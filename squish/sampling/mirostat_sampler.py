"""squish/sampling/mirostat_sampler.py

MirostatSampler — Online perplexity-controlled text decoding (Mirostat-v2).

Maintains a running estimate of the model's surprise (cross-entropy) and
adjusts a temperature-like scaling factor each step via a PID-style update so
that the per-step perplexity tracks a user-specified target τ.  This eliminates
both catastrophic repetition (low τ) and incoherent topic-drift (high τ).

Mirostat-v2 estimates the expected surprise using the empirical token
probabilities rather than relying on vocabulary Zipf-law parameters.

Reference
---------
Basu et al. "Mirostat: A Neural Text Decoding Algorithm that Directly Controls
Perplexity." ICLR 2021.  Also: Mirostat-v2 implementation in llama.cpp (2023).
"""

from __future__ import annotations

__all__ = ["MirostatConfig", "MirostatState", "MirostatSampler"]

from dataclasses import dataclass

import numpy as np
from numpy import ndarray


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class MirostatConfig:
    """Configuration for MirostatSampler.

    Parameters
    ----------
    tau:
        Target cross-entropy (perplexity target in nats).  τ = 3–5 is
        typically coherent; τ = 7–9 increases creativity.
    eta:
        Learning rate for the surprise estimate update.
    seed:
        RNG seed.
    """

    tau: float = 5.0
    eta: float = 0.1
    seed: int = 0

    def __post_init__(self) -> None:
        if self.tau <= 0.0:
            raise ValueError("tau must be > 0")
        if self.eta <= 0.0:
            raise ValueError("eta must be > 0")


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

@dataclass
class MirostatState:
    """Mutable state for MirostatSampler.

    Attributes
    ----------
    mu:
        Current estimate of the maximum acceptable surprise level.
        Initialised to ``2 × tau``.
    n_tokens:
        Number of tokens sampled so far.
    last_surprise:
        Surprise of the last sampled token (nats).
    """

    mu: float
    n_tokens: int = 0
    last_surprise: float = 0.0


# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------

class MirostatSampler:
    """Perplexity-controlled sampler (Mirostat-v2).

    Parameters
    ----------
    config:
        ``MirostatConfig`` instance.
    """

    def __init__(self, config: MirostatConfig) -> None:
        self.config = config
        self._rng = np.random.default_rng(config.seed)

    def new_state(self) -> MirostatState:
        """Create a fresh state with μ = 2τ."""
        return MirostatState(mu=2.0 * self.config.tau)

    def sample(
        self, logits: ndarray, state: MirostatState
    ) -> tuple[int, MirostatState]:
        """Sample one token and update the perplexity controller.

        Parameters
        ----------
        logits:
            Raw logit vector, shape ``(vocab_size,)``.
        state:
            Current ``MirostatState``.

        Returns
        -------
        token:
            Sampled token index.
        state:
            Updated state.
        """
        logits = np.asarray(logits, dtype=np.float32)
        if logits.ndim != 1:
            raise ValueError("logits must be 1-D")

        # Truncation: keep only tokens with surprise ≤ mu
        probs_full = self._softmax(logits)
        surprise_full = -np.log(np.clip(probs_full, 1e-40, None))  # (V,)

        # Mirostat-v2: truncate at mu
        mask = surprise_full <= state.mu
        if not mask.any():
            # No token survives — fall back to argmax
            mask = np.zeros_like(mask)
            mask[int(np.argmax(logits))] = True

        valid_logits = np.where(mask, logits, -1e9)
        probs = self._softmax(valid_logits)
        token = int(self._rng.choice(len(probs), p=probs))

        # Update mu: error = surprise(sampled) − tau; mu ← mu − eta × error
        sampled_surprise = float(surprise_full[token])
        error = sampled_surprise - self.config.tau
        new_mu = state.mu - self.config.eta * error
        new_mu = max(0.1, new_mu)  # keep mu positive

        new_state = MirostatState(
            mu=new_mu,
            n_tokens=state.n_tokens + 1,
            last_surprise=sampled_surprise,
        )
        return token, new_state

    def reset(self) -> MirostatState:
        """Return a fresh state (alias for ``new_state``)."""
        return self.new_state()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _softmax(x: ndarray) -> ndarray:
        shifted = x - x.max()
        exp_x = np.exp(shifted)
        return exp_x / exp_x.sum()
