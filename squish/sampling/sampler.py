"""
squish/sampling/sampler.py

Phase 6 — Structured Sampling + LLM-42 Determinism

Motivation
──────────
Sampling logic in squish is currently scattered across:
  • speculative/speculative.py  (_sample, _softmax_np, _top_p_filter, _greedy)
  • server.py                   (temperature / top_p passed per-call)
  • grammar/                    (constrained token selection)

This module consolidates all per-request sampling state into one
``StructuredSampler`` — a single object that:

  1. Holds all sampling hyperparameters (temperature, top-k, top-p, seed).
  2. Maintains per-request state: seeded RNG + repetition-penalty window.
  3. Provides a deterministic path: same (prompt, seed) → same tokens.
     The default seed (42) is the "LLM-42" naming convention.

Design
──────
• Pure numpy — no MLX dependency; fully unit-testable.
• ``reset(seed=...)`` clears per-request state at the start of each new
  generation without re-constructing config objects.
• ``sample(logits)`` is the single public entry point.  It applies in order:
    1. Repetition penalty
    2. Temperature scaling
    3. Top-k truncation  (0 = off)
    4. Top-p (nucleus)   (1.0 = off)
    5. Seeded multinomial sample  (temperature=0 → greedy argmax)
• ``update(token_id)`` records the emitted token for repetition tracking.

LLM-42 naming
─────────────
The default seed is 42 (the "answer to life, the universe, and everything").
When ``seed`` is not None, generation of any prompt is fully deterministic:
    sampler = StructuredSampler(SamplerConfig(seed=42))
    sampler.reset()
    tok1 = sampler.sample(logits)    # reproducible
    sampler.reset()
    tok2 = sampler.sample(logits)    # identical to tok1
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SamplerConfig:
    """
    Hyperparameters for :class:`StructuredSampler`.

    Parameters
    ----------
    temperature         : Logit scaling factor.  0.0 → greedy argmax.
    top_p               : Nucleus filtering threshold ∈ (0, 1].  1.0 = off.
    top_k               : Keep only top-k logits before sampling.  0 = off.
    seed                : Fixed RNG seed for deterministic output.
                          None = non-deterministic (system randomness).
                          42 = *LLM-42* determinism default.
    rep_penalty         : Repetition penalty factor ≥ 1.0.
                          1.0 = off; > 1.0 reduces probability of recent tokens.
    rep_penalty_window  : Number of recent tokens tracked for repetition penalty.
    """
    temperature:         float    = 1.0
    top_p:               float    = 1.0
    top_k:               int      = 0
    seed:                int | None = None
    rep_penalty:         float    = 1.0
    rep_penalty_window:  int      = 64

    def __post_init__(self) -> None:
        if self.temperature < 0.0:
            raise ValueError(f"temperature must be ≥ 0, got {self.temperature}")
        if not (0.0 < self.top_p <= 1.0):
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}")
        if self.top_k < 0:
            raise ValueError(f"top_k must be ≥ 0, got {self.top_k}")
        if self.rep_penalty < 1.0:
            raise ValueError(f"rep_penalty must be ≥ 1.0, got {self.rep_penalty}")
        if self.rep_penalty_window < 0:
            raise ValueError(
                f"rep_penalty_window must be ≥ 0, got {self.rep_penalty_window}"
            )


# ---------------------------------------------------------------------------
# Sentinel for optional seed override
# ---------------------------------------------------------------------------

class _SentinelType:
    """Internal singleton used as the default for reset(seed=...)."""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance


_SENTINEL = _SentinelType()


# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------

class StructuredSampler:
    """
    Per-request sampling engine.

    Instantiate once with a :class:`SamplerConfig`, then call :meth:`reset`
    at the start of each generation request to re-seed the RNG and clear the
    repetition window.

    Parameters
    ----------
    config : :class:`SamplerConfig`
    """

    __slots__ = (
        "config",
        "_rng",           # np.random.Generator — seeded per request
        "_rep_window",    # deque-like list of recent token IDs
        "_step",          # decode step counter (for diagnostics)
    )

    def __init__(self, config: SamplerConfig) -> None:
        self.config      = config
        self._rng        = self._build_rng(config.seed)
        self._rep_window: list[int] = []
        self._step       = 0

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def reset(self, seed: int | None = _SENTINEL) -> None:  # type: ignore[assignment]
        """
        Reset per-request state.

        Parameters
        ----------
        seed : If provided, overrides ``config.seed`` for this request.
               Pass ``None`` explicitly for non-deterministic.
               Omit (default sentinel) to use ``config.seed``.
        """
        effective_seed = self.config.seed if seed is _SENTINEL else seed
        self._rng        = self._build_rng(effective_seed)
        self._rep_window = []
        self._step       = 0

    # ── Main pipeline ─────────────────────────────────────────────────────────

    def sample(self, logits: np.ndarray) -> int:
        """
        Sample one token from *logits*.

        Applies in order:
          1. Repetition penalty (if ``rep_penalty > 1.0``)
          2. Temperature scaling
          3. Top-k truncation
          4. Top-p (nucleus) filtering
          5. Seeded or random multinomial sample

        Parameters
        ----------
        logits : (vocab_size,) float32 raw logits from lm_head

        Returns
        -------
        token_id : int
        """
        logits = np.asarray(logits, dtype=np.float32)
        cfg    = self.config

        # ── 1. Repetition penalty ─────────────────────────────────────────────
        if cfg.rep_penalty > 1.0 and self._rep_window:
            logits = _apply_rep_penalty(logits, self._rep_window, cfg.rep_penalty)

        # ── 2. Temperature ────────────────────────────────────────────────────
        if cfg.temperature == 0.0:
            return int(np.argmax(logits))

        scaled = logits / np.float32(cfg.temperature)

        # ── 3. Top-k ──────────────────────────────────────────────────────────
        if cfg.top_k > 0:
            scaled = _apply_top_k(scaled, cfg.top_k)

        # ── 4. Top-p ──────────────────────────────────────────────────────────
        probs = _softmax_f32(scaled)
        if cfg.top_p < 1.0:
            probs = _apply_top_p(probs, cfg.top_p)

        # ── 5. Sample ─────────────────────────────────────────────────────────
        self._step += 1
        return int(self._rng.choice(len(probs), p=probs))

    def update(self, token_id: int) -> None:
        """
        Record the most-recently emitted token for repetition tracking.

        Must be called *after* :meth:`sample` for each token that is
        appended to the sequence.
        """
        self._rep_window.append(token_id)
        limit = self.config.rep_penalty_window
        if limit > 0 and len(self._rep_window) > limit:
            self._rep_window = self._rep_window[-limit:]

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def step(self) -> int:
        """Number of tokens sampled since the last ``reset()``."""
        return self._step

    @property
    def rep_window(self) -> list[int]:
        """Copy of the current repetition penalty window."""
        return list(self._rep_window)

    # ── Internals ─────────────────────────────────────────────────────────────

    @staticmethod
    def _build_rng(seed: int | None) -> np.random.Generator:
        if seed is None:
            return np.random.default_rng()
        return np.random.Generator(np.random.PCG64(seed))


# ---------------------------------------------------------------------------
# Kernel helpers (pure numpy, exposed for reuse by kernel registry)
# ---------------------------------------------------------------------------

def _softmax_f32(logits: np.ndarray) -> np.ndarray:
    """
    Numerically-stable softmax → float32 probability distribution.

    Parameters
    ----------
    logits : (vocab_size,) — may be -inf in masked positions

    Returns
    -------
    probs : (vocab_size,) float32, sums to 1.0 (or 0.0 on all-inf input)
    """
    x     = np.asarray(logits, dtype=np.float32)
    # Shift by max for numerical stability; -inf positions stay 0 after exp
    x_max = np.max(x, where=np.isfinite(x), initial=-1e38)
    e     = np.exp(x - x_max)
    e     = np.where(np.isfinite(x), e, 0.0)
    s     = e.sum()
    if s == 0.0:
        # Fallback: uniform over finite positions
        finite = np.isfinite(x)
        if finite.any():
            p = finite.astype(np.float32)
            return (p / p.sum()).astype(np.float32)
        return np.ones(len(x), dtype=np.float32) / len(x)
    return (e / s).astype(np.float32)


def _apply_top_k(logits: np.ndarray, k: int) -> np.ndarray:
    """
    Zero-out (→ -inf) all logits except the top-k.

    Parameters
    ----------
    logits : (vocab_size,) raw logits
    k      : number of logits to retain (capped at vocab_size)

    Returns
    -------
    masked : (vocab_size,) with all but top-k set to -inf
    """
    k       = min(k, len(logits))
    topk    = np.partition(logits, -k)[-k]   # k-th largest value
    masked  = np.where(logits >= topk, logits, np.float32(-np.inf))
    return masked.astype(np.float32)


def _apply_top_p(probs: np.ndarray, top_p: float) -> np.ndarray:
    """
    Nucleus (top-p) filtering: zero out tokens outside the top-p mass.

    Parameters
    ----------
    probs : (vocab_size,) probability distribution (sums to 1)
    top_p : cumulative probability threshold ∈ (0, 1]

    Returns
    -------
    filtered : (vocab_size,) re-normalised probabilities
    """
    sorted_idx  = np.argsort(-probs)            # descending order
    sorted_p    = probs[sorted_idx]
    cumsum      = np.cumsum(sorted_p)

    # Find the first index where cumsum >= top_p; include that token
    cutoff      = int(np.searchsorted(cumsum, top_p, side="right")) + 1
    cutoff      = max(1, cutoff)                # always keep ≥ 1 token

    kept_idx    = sorted_idx[:cutoff]
    mask        = np.zeros_like(probs)
    mask[kept_idx] = probs[kept_idx]

    s = mask.sum()
    if s == 0.0:
        return probs
    return (mask / s).astype(np.float32)


def _apply_rep_penalty(
    logits:   np.ndarray,
    window:   Sequence[int],
    penalty:  float,
) -> np.ndarray:
    """
    Repetition penalty: divide logits of recent tokens by *penalty*
    (if logit > 0) or multiply by *penalty* (if logit < 0), following
    the convention from Keskar et al. (2019).

    Parameters
    ----------
    logits  : (vocab_size,) raw logits
    window  : sequence of recent token IDs to penalise
    penalty : factor ≥ 1.0

    Returns
    -------
    penalised : (vocab_size,) float32
    """
    out = logits.copy()
    for tok in set(window):
        if 0 <= tok < len(out):
            out[tok] = out[tok] / penalty if out[tok] > 0 else out[tok] * penalty
    return out.astype(np.float32)
