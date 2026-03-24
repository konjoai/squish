# [Experimental] This module is part of Squish v42+ (Wave 68).
# Proof-of-concept quality: API and behaviour may change without notice.
# For stable core functionality see Waves 1–12 in MODULES.md.
"""squish/speculative/eagle_head.py — Production EAGLE Draft Head Inference.

Wave 68: accepts the last hidden state from the main model's forward pass and
produces ``n_draft`` draft token candidates with log-probability scores.
Integrates with :class:`~squish.speculative.draft_multiplexer.DraftMultiplexer`
as the highest-priority strategy.

The module exposes two entry points:

1. **High-level**: :class:`EAGLEHeadRunner` — loads weights once, provides
   :meth:`~EAGLEHeadRunner.generate_drafts` for per-step draft generation.
2. **Low-level**: :func:`eagle_decode_step` — stateless single-step helper for
   direct integration into custom inference loops.

Fallback policy
───────────────
The runner tracks a rolling 64-token acceptance rate.  If the rate drops below
the threshold stored in the loaded heads's ``acceptance_fallback_threshold``
config field (default 0.55), :meth:`~EAGLEHeadRunner.should_fallback` returns
``True`` and the caller should switch to n-gram or another strategy until
the acceptance rate recovers.

Usage::

    from squish.speculative.eagle_head import EAGLEHeadRunner, EAGLERunnerConfig
    from squish.compress.distill_eagle import load_eagle_head

    weights = load_eagle_head("~/.squish/models/qwen3-8b.squizd-eagle")
    config  = EAGLERunnerConfig(n_draft=5)
    runner  = EAGLEHeadRunner(weights, config)

    # In the main inference loop:
    drafts = runner.generate_drafts(hidden_state_p50, hidden_state_p75)
    # drafts: list of (token_id, log_prob) tuples, length = n_draft

    # After verifying drafts:
    runner.record_acceptance(n_accepted=3, n_proposed=5)

    if runner.should_fallback():
        # Switch to n-gram strategy
        pass
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from squish.compress.distill_eagle import (
    EAGLEHeadWeights,
    _eagle_forward,
    _FALLBACK_ACCEPTANCE_THRESHOLD,
)

__all__ = [
    "EAGLERunnerConfig",
    "DraftToken",
    "EAGLEHeadRunner",
    "eagle_decode_step",
    "ROLLING_WINDOW_SIZE",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ROLLING_WINDOW_SIZE: int = 64   # tokens in the rolling acceptance window


# ---------------------------------------------------------------------------
# Config and result types
# ---------------------------------------------------------------------------

@dataclass
class EAGLERunnerConfig:
    """Runtime configuration for :class:`EAGLEHeadRunner`.

    Attributes:
        n_draft: Number of draft tokens to generate per step (default 5).
        temperature: Sampling temperature applied to draft log-probabilities.
            ``1.0`` = no change; ``< 1.0`` = sharper; ``> 1.0`` = softer.
        top_k: Sample from the top-k highest log-probability tokens.
            ``0`` disables top-k filtering.
    """

    n_draft: int = 5
    temperature: float = 1.0
    top_k: int = 50

    def __post_init__(self) -> None:
        if self.n_draft < 1:
            raise ValueError("n_draft must be >= 1")
        if self.temperature <= 0.0:
            raise ValueError("temperature must be > 0")
        if self.top_k < 0:
            raise ValueError("top_k must be >= 0")


@dataclass
class DraftToken:
    """A single draft token produced by the EAGLE head.

    Attributes:
        token_id: Vocabulary index of the draft token.
        log_prob: Log-probability assigned by the draft head.
    """

    token_id: int
    log_prob: float


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------

def _sample_top_k(
    log_probs: np.ndarray,
    n: int,
    *,
    temperature: float = 1.0,
    top_k: int = 0,
    rng: Optional[np.random.Generator] = None,
) -> List[DraftToken]:
    """Sample *n* tokens from *log_probs* using top-k temperature sampling.

    Args:
        log_probs: Log-probability distribution, shape ``(vocab_size,)``.
        n: Number of tokens to sample.
        temperature: Sampling temperature.
        top_k: Keep only the top-k candidates; 0 disables top-k.
        rng: NumPy random generator (creates a default one if ``None``).

    Returns:
        List of :class:`DraftToken` instances, sorted descending by log_prob.
    """
    if rng is None:
        rng = np.random.default_rng()

    scaled = log_probs / temperature
    # Top-k masking
    if top_k > 0 and top_k < len(scaled):
        threshold_idx = np.argpartition(scaled, -top_k)[-top_k:]
        mask = np.full(len(scaled), -np.inf)
        mask[threshold_idx] = scaled[threshold_idx]
        scaled = mask

    # Convert to probabilities and sample
    scaled = scaled - np.max(scaled)
    probs = np.exp(scaled)
    probs = probs / (np.sum(probs) + 1e-8)

    # Draw without replacement up to n distinct tokens.
    # k is bounded by the number of tokens with non-zero probability:
    # after top-k masking, only top_k (or fewer) entries are non-zero.
    vocab_size = len(probs)
    n_nonzero = int(np.count_nonzero(probs))
    k = min(n, n_nonzero) if n_nonzero > 0 else 1
    token_ids = rng.choice(vocab_size, size=k, replace=False, p=probs)
    return sorted(
        [DraftToken(token_id=int(tid), log_prob=float(log_probs[tid]))
         for tid in token_ids],
        key=lambda t: t.log_prob,
        reverse=True,
    )


# ---------------------------------------------------------------------------
# Stateless single-step helper
# ---------------------------------------------------------------------------

def eagle_decode_step(
    hidden_p50: np.ndarray,
    hidden_p75: np.ndarray,
    weights: EAGLEHeadWeights,
    *,
    n_draft: int = 5,
    temperature: float = 1.0,
    top_k: int = 50,
    rng: Optional[np.random.Generator] = None,
) -> List[DraftToken]:
    """Single-step EAGLE draft generation (stateless).

    Args:
        hidden_p50: Hidden state from the 50th-percentile source layer,
            shape ``(d_model,)`` or ``(1, d_model)``.
        hidden_p75: Hidden state from the 75th-percentile source layer,
            shape ``(d_model,)`` or ``(1, d_model)``.
        weights: Loaded :class:`~squish.compress.distill_eagle.EAGLEHeadWeights`.
        n_draft: Number of draft tokens to return.
        temperature: Sampling temperature.
        top_k: Top-k candidates to sample from.
        rng: NumPy random generator.

    Returns:
        List of :class:`DraftToken` (descending log_prob order).
    """
    h50 = np.asarray(hidden_p50, dtype=np.float32).reshape(-1)
    h75 = np.asarray(hidden_p75, dtype=np.float32).reshape(-1)
    h_in = np.concatenate([h50, h75])[np.newaxis, :]  # (1, 2*d_model)

    log_probs = _eagle_forward(h_in, weights)  # (vocab_size,)
    return _sample_top_k(log_probs, n_draft, temperature=temperature,
                         top_k=top_k, rng=rng)


# ---------------------------------------------------------------------------
# Stateful runner
# ---------------------------------------------------------------------------

class EAGLEHeadRunner:
    """Stateful EAGLE draft head runner with rolling acceptance tracking.

    Maintains a circular buffer of the last :data:`ROLLING_WINDOW_SIZE`
    acceptance events and falls back to a simpler strategy when the rolling
    acceptance rate drops below ``weights.config.acceptance_fallback_threshold``.

    Args:
        weights: Loaded :class:`~squish.compress.distill_eagle.EAGLEHeadWeights`.
        config: :class:`EAGLERunnerConfig`.
        rng_seed: Optional seed for reproducible sampling.
    """

    def __init__(
        self,
        weights: EAGLEHeadWeights,
        config: Optional[EAGLERunnerConfig] = None,
        *,
        rng_seed: Optional[int] = None,
    ) -> None:
        self.weights = weights
        self.config = config or EAGLERunnerConfig()
        self._rng = np.random.default_rng(rng_seed)
        self._fallback_threshold: float = (
            weights.config.acceptance_fallback_threshold
        )

        # Rolling acceptance window:  1 = accepted, 0 = rejected
        self._window: list[int] = []
        self._total_proposed: int = 0
        self._total_accepted: int = 0

    # ------------------------------------------------------------------
    # Draft generation
    # ------------------------------------------------------------------

    def generate_drafts(
        self,
        hidden_p50: np.ndarray,
        hidden_p75: np.ndarray,
        *,
        n_draft: Optional[int] = None,
    ) -> List[DraftToken]:
        """Generate draft token candidates for the current decoding step.

        Args:
            hidden_p50: Hidden state from source layer at 50th-percentile
                depth, shape ``(d_model,)``.
            hidden_p75: Hidden state from source layer at 75th-percentile
                depth, shape ``(d_model,)``.
            n_draft: Override number of draft tokens (defaults to
                ``config.n_draft``).

        Returns:
            Sorted list of :class:`DraftToken` (best first).
        """
        n = n_draft if n_draft is not None else self.config.n_draft
        return eagle_decode_step(
            hidden_p50,
            hidden_p75,
            self.weights,
            n_draft=n,
            temperature=self.config.temperature,
            top_k=self.config.top_k,
            rng=self._rng,
        )

    # ------------------------------------------------------------------
    # Acceptance tracking
    # ------------------------------------------------------------------

    def record_acceptance(self, n_accepted: int, n_proposed: int) -> None:
        """Update the rolling acceptance window.

        Args:
            n_accepted: Number of draft tokens accepted by the verifier.
            n_proposed: Number of draft tokens that were proposed.
        """
        if n_proposed < 0:
            raise ValueError("n_proposed must be >= 0")
        if n_accepted < 0:
            raise ValueError("n_accepted must be >= 0")
        if n_accepted > n_proposed:
            raise ValueError("n_accepted must be <= n_proposed")

        for _ in range(n_accepted):
            self._window.append(1)
        for _ in range(n_proposed - n_accepted):
            self._window.append(0)

        # Keep window at most ROLLING_WINDOW_SIZE
        if len(self._window) > ROLLING_WINDOW_SIZE:
            self._window = self._window[-ROLLING_WINDOW_SIZE:]

        self._total_proposed += n_proposed
        self._total_accepted += n_accepted

    @property
    def rolling_acceptance_rate(self) -> float:
        """Acceptance rate over the last :data:`ROLLING_WINDOW_SIZE` tokens.

        Returns ``1.0`` if no tokens have been recorded yet (optimistic default
        to allow warm-up).
        """
        if not self._window:
            return 1.0
        return float(sum(self._window)) / len(self._window)

    @property
    def lifetime_acceptance_rate(self) -> float:
        """Acceptance rate over all tokens seen since construction."""
        if self._total_proposed == 0:
            return 1.0
        return self._total_accepted / self._total_proposed

    def should_fallback(self) -> bool:
        """Return ``True`` if the rolling acceptance rate is below threshold.

        The fallback is only triggered after at least :data:`ROLLING_WINDOW_SIZE`
        / 4 tokens have been recorded (avoids premature fallback during warm-up).
        """
        if len(self._window) < ROLLING_WINDOW_SIZE // 4:
            return False
        return self.rolling_acceptance_rate < self._fallback_threshold

    def reset_stats(self) -> None:
        """Reset the rolling window and lifetime statistics."""
        self._window = []
        self._total_proposed = 0
        self._total_accepted = 0
