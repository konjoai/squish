"""
squish/speculative/jacobi_decode.py

JacobiDecoder — Parallel Fixed-Point Speculative Decoding (CLLMs).

Based on:
  "Consistency Large Language Models: A Family of Efficient Parallel Decoders"
  Kou et al., ICML 2024  —  arXiv:2403.00835

Background
----------
Standard autoregressive (AR) decoding generates one token per forward pass.
Speculative decoding requires a separate draft model.  Jacobi decoding
eliminates both constraints by treating token generation as a *fixed-point
iteration* problem:

  Given context c = [t_0 … t_{n-1}], we want t_n, t_{n+1}, …, t_{n+N}.

  Jacobi iteration:
    1. Initialize a guess: y^(0) = [g_0, g_1, …, g_{N-1}]  (any initial tokens)
    2. Each iteration: run ONE forward pass over [c | y^(k)] → logits
       Accept y_i if argmax(logits[i]) == y^(k)_i  ("fixed point")
       For non-fixed positions: update y^(k+1)_i = argmax(logits[i])
    3. Stop if all N positions are fixed OR max_iters reached

  The Gauss-Seidel variant propagates each accepted position immediately
  within the same iteration, accelerating convergence.

Key properties
--------------
- Zero draft model.  Only one forward pass shape per Jacobi step.
- Every fixed point is *exactly* correct (greedy).
- ~2–3.4× speedup on coherent / repetitive text (CLLMs, ICML 2024).
- Falls back gracefully to AR (n_tokens=1) for adversarial input.

This module provides:
  ``JacobiConfig``   — configuration
  ``JacobiStats``    — per-instance inference statistics
  ``JacobiDecoder``  — stateful decoder

Usage::

    from squish.speculative.jacobi_decode import JacobiConfig, JacobiDecoder

    cfg = JacobiConfig(n_tokens=4, max_iter=8)

    def logits_fn(token_ids):
        # token_ids: List[int] → np.ndarray (len, vocab)
        ...

    decoder = JacobiDecoder(cfg)
    accepted, n_iter = decoder.decode_step(logits_fn, context_ids=[1, 2, 3])
    # accepted: List[int] of at least 1 token
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np

__all__ = [
    "JacobiConfig",
    "JacobiStats",
    "JacobiDecoder",
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class JacobiConfig:
    """Configuration for Jacobi / Gauss-Seidel parallel decoding.

    Attributes:
        n_tokens:    Number of parallel positions to fill per decode step.
        max_iter:    Maximum Jacobi iterations per step.
        variant:     ``"jacobi"`` (update all at end of pass) or
                     ``"gauss_seidel"`` (update immediately in-pass).
        temperature: Sampling temperature (1.0 = greedy argmax via temp=0).
        seed:        Random seed for stochastic sampling.
        init:        Initialization strategy for the N positions:
                     ``"uniform"`` (all same token) or ``"random"``.
    """

    n_tokens: int = 4
    max_iter: int = 8
    variant: str = "jacobi"
    temperature: float = 0.0
    seed: int = 42
    init: str = "uniform"

    def __post_init__(self) -> None:
        if self.n_tokens < 1:
            raise ValueError(f"n_tokens must be >= 1, got {self.n_tokens}")
        if self.max_iter < 1:
            raise ValueError(f"max_iter must be >= 1, got {self.max_iter}")
        if self.variant not in ("jacobi", "gauss_seidel"):
            raise ValueError(
                f"variant must be 'jacobi' or 'gauss_seidel', got '{self.variant}'"
            )
        if self.temperature < 0:
            raise ValueError(f"temperature must be >= 0, got {self.temperature}")
        if self.init not in ("uniform", "random"):
            raise ValueError(f"init must be 'uniform' or 'random', got '{self.init}'")


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


@dataclass
class JacobiStats:
    """Lifetime statistics for a JacobiDecoder instance.

    Attributes:
        total_decode_steps:     Number of calls to ``decode_step``.
        total_tokens_generated: Total tokens accepted across all steps.
        total_iterations:       Sum of Jacobi iterations across all steps.
        total_fixed_points:     Total fixed-point positions accepted.
    """

    total_decode_steps: int = 0
    total_tokens_generated: int = 0
    total_iterations: int = 0
    total_fixed_points: int = 0

    @property
    def mean_tokens_per_step(self) -> float:
        """Average number of tokens accepted per decode step."""
        if self.total_decode_steps == 0:
            return 0.0
        return self.total_tokens_generated / self.total_decode_steps

    @property
    def mean_iterations_per_step(self) -> float:
        """Average Jacobi iterations per decode step."""
        if self.total_decode_steps == 0:
            return 0.0
        return self.total_iterations / self.total_decode_steps

    @property
    def fixed_point_rate(self) -> float:
        """Fraction of parallel positions that converge to fixed point."""
        total_positions = self.total_decode_steps * 1  # denominator approximation
        if total_positions == 0 or self.total_tokens_generated == 0:
            return 0.0
        return self.total_fixed_points / max(1, self.total_tokens_generated)

    def __repr__(self) -> str:
        return (
            f"JacobiStats(steps={self.total_decode_steps}, "
            f"tokens={self.total_tokens_generated}, "
            f"mean_tok/step={self.mean_tokens_per_step:.2f}, "
            f"mean_iter/step={self.mean_iterations_per_step:.2f})"
        )


# ---------------------------------------------------------------------------
# Jacobi Decoder
# ---------------------------------------------------------------------------


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable row-wise softmax."""
    x = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def _sample_token(logits: np.ndarray, temperature: float, rng: np.random.Generator) -> int:
    """Sample a single token from 1-D logits."""
    if temperature <= 1e-9:
        return int(np.argmax(logits))
    probs = _softmax(logits / temperature)
    return int(rng.choice(len(probs), p=probs))


class JacobiDecoder:
    """Jacobi / Gauss-Seidel parallel speculative decoder.

    Parameters
    ----------
    config:
        Decoder configuration.

    The ``logits_fn`` passed to ``decode_step`` must accept a ``List[int]``
    of token ids (context + draft positions) and return a ``np.ndarray``
    of shape ``(len(token_ids), vocab_size)`` — the next-token logit for
    each *input* position.
    """

    def __init__(self, config: Optional[JacobiConfig] = None) -> None:
        self._cfg = config or JacobiConfig()
        self._rng = np.random.default_rng(self._cfg.seed)
        self.stats = JacobiStats()

    # ------------------------------------------------------------------
    # Core decode step
    # ------------------------------------------------------------------

    def decode_step(
        self,
        logits_fn: Callable[[List[int]], np.ndarray],
        context_ids: List[int],
        vocab_size: Optional[int] = None,
    ) -> Tuple[List[int], int]:
        """Execute one parallel Jacobi decode step.

        Parameters
        ----------
        logits_fn:
            Callable that takes ``List[int]`` token ids and returns
            ``np.ndarray`` of shape ``(len(ids), vocab_size)``.
        context_ids:
            Current context (prompt + decoded so far).
        vocab_size:
            Optional vocab hint; only used for random initialization.

        Returns
        -------
        accepted_tokens : List[int]
            Accepted token ids (length ≥ 1).
        n_iterations : int
            Number of Jacobi iterations performed.
        """
        cfg = self._cfg
        n = cfg.n_tokens

        # --- Initialize parallel positions --------------------------------
        if cfg.init == "uniform" and context_ids:
            # Start all N positions with the last context token
            guesses: List[int] = [context_ids[-1]] * n
        elif cfg.init == "random" and vocab_size is not None:
            guesses = list(self._rng.integers(0, vocab_size, size=n))
        else:
            # Fallback: tiny deterministic init (EOS-safe: token 1)
            guesses = [1] * n

        # --- Jacobi iterations -------------------------------------------
        n_iter = 0
        for iteration in range(cfg.max_iter):
            n_iter += 1
            full_sequence = list(context_ids) + guesses

            # Single forward pass
            all_logits = logits_fn(full_sequence)  # (len, vocab)
            # We care about positions context_len .. context_len+n-1
            ctx_len = len(context_ids)
            pos_logits = all_logits[ctx_len: ctx_len + n]  # (n, vocab)

            new_guesses = [
                _sample_token(pos_logits[i], cfg.temperature, self._rng)
                for i in range(n)
            ]

            if cfg.variant == "gauss_seidel":
                # Update positions immediately — propagate within pass
                for i in range(n):
                    guesses[i] = new_guesses[i]
                # Recompute for convergence check
                converged = self._check_convergence(
                    logits_fn, context_ids, guesses, cfg.temperature
                )
                if converged is not None:
                    # All N positions are fixed
                    self._update_stats(n, n_iter)
                    return guesses, n_iter
            else:
                # Jacobi: check fixed points vs current guesses
                fixed = [new_guesses[i] == guesses[i] for i in range(n)]
                if all(fixed):
                    # Full convergence
                    self._update_stats(sum(fixed), n_iter)
                    return guesses, n_iter
                # Partial update
                guesses = new_guesses

        # --- Accept prefix of fixed positions on timeout ------------------
        # Run one final verification pass
        full_sequence = list(context_ids) + guesses
        all_logits = logits_fn(full_sequence)
        ctx_len = len(context_ids)
        pos_logits = all_logits[ctx_len: ctx_len + n]

        accepted: List[int] = []
        for i in range(n):
            pred = _sample_token(pos_logits[i], cfg.temperature, self._rng)
            if pred == guesses[i]:
                accepted.append(guesses[i])
            else:
                # Accept correction token and stop
                accepted.append(pred)
                break

        if not accepted:
            # Fallback: at least return the single AR token
            first_logits = all_logits[ctx_len]
            accepted = [_sample_token(first_logits, cfg.temperature, self._rng)]

        self._update_stats(len(accepted), n_iter)
        return accepted, n_iter

    def _check_convergence(
        self,
        logits_fn: Callable[[List[int]], np.ndarray],
        context_ids: List[int],
        guesses: List[int],
        temperature: float,
    ) -> Optional[List[int]]:
        """Return guesses if all positions are fixed points, else None."""
        full = list(context_ids) + guesses
        all_logits = logits_fn(full)
        ctx_len = len(context_ids)
        n = len(guesses)
        pos_logits = all_logits[ctx_len: ctx_len + n]
        for i in range(n):
            pred = _sample_token(pos_logits[i], temperature, self._rng)
            if pred != guesses[i]:
                return None
        return guesses

    def _update_stats(self, n_accepted: int, n_iter: int) -> None:
        self.stats.total_decode_steps += 1
        self.stats.total_tokens_generated += n_accepted
        self.stats.total_iterations += n_iter
        self.stats.total_fixed_points += n_accepted

    def reset_stats(self) -> None:
        """Clear accumulated statistics."""
        self.stats = JacobiStats()

    def __repr__(self) -> str:
        return (
            f"JacobiDecoder(n_tokens={self._cfg.n_tokens}, "
            f"variant='{self._cfg.variant}', "
            f"max_iter={self._cfg.max_iter}, "
            f"{self.stats})"
        )
