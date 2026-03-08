"""
squish/meta_reasoner.py

Meta-Reasoner — Dynamic Thinking Budget Control for Qwen3 and similar
chain-of-thought (CoT) models.

Inspired by:
  "Meta-Reasoner: Dynamic Guidance for Inference-Time Reasoning" (Aug 2025)
  arXiv:2502.19918

Problem: Qwen3's thinking mode (/think tokens) can consume many unnecessary
tokens for simple tasks.  With a static budget, simple queries use too many
thinking tokens; complex queries may be cut off too early.

Solution: Monitor *thinking token entropy* in real-time during the <think>
phase.  When entropy consistently falls below a threshold (reasoning has
converged to a stable distribution → answer is clear), force the </think>
closing token to terminate the thinking phase early.  When entropy is
consistently high (still exploring), extend the budget.

Signal definition:
    H(logits_t) = -Σ p_i log p_i    (standard Shannon entropy over vocab)

Convergence heuristic:
    If H < entropy_threshold for ``patience`` consecutive steps:
    → reasoning has converged; emit </think>
    If H > entropy_high_threshold for ``patience`` steps:
    → model is still exploring; grant budget extension up to max_think_tokens

Integration in server.py decode loop::

    from squish.meta_reasoner import MetaReasoner, MetaReasonerConfig

    cfg     = MetaReasonerConfig(think_end_token_id=151668)  # Qwen3 </think> id
    monitor = MetaReasoner(cfg)

    for step in decode_loop:
        logits = model_forward(...)
        if monitor.in_thinking_phase:
            if monitor.step(logits_np, token_id):
                # Force </think> next — set logits to spike on think_end_token_id
                logits = monitor.force_think_end(logits)
        monitor.advance(token_id)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

__all__ = [
    "MetaReasonerConfig",
    "MetaReasoner",
]

# Qwen3 special token IDs (defaults; can be overridden via config)
_QWEN3_THINK_START_DEFAULT = 151667   # <think>
_QWEN3_THINK_END_DEFAULT   = 151668   # </think>


@dataclass
class MetaReasonerConfig:
    """
    Configuration for the MetaReasoner thinking-budget controller.

    Parameters
    ----------
    think_start_token_id : int
        Token ID for the thinking start marker (e.g. ``<think>``).
    think_end_token_id : int
        Token ID for the thinking end marker (e.g. ``</think>``).
    entropy_threshold : float
        Entropy below this value signals that reasoning has converged.
        Lower values = more tolerant (only stop when very confident).
        Typical range: 0.5 – 3.0.  Default 1.5.
    entropy_high_threshold : float
        Entropy above this signals active exploration.
        Used to detect if budget should be extended.  Default 4.0.
    patience : int
        Number of consecutive converged/exploring steps before acting.
        Prevents reacting to single spikes.
    min_think_tokens : int
        Minimum number of tokens to generate in thinking phase before
        considering early termination.
    max_think_tokens : int
        Hard cap on thinking tokens even if reasoning does not converge.
    """
    think_start_token_id: int   = _QWEN3_THINK_START_DEFAULT
    think_end_token_id:   int   = _QWEN3_THINK_END_DEFAULT
    entropy_threshold:    float = 1.5
    entropy_high_threshold: float = 4.0
    patience:             int   = 3
    min_think_tokens:     int   = 5
    max_think_tokens:     int   = 512

    def __post_init__(self) -> None:
        if self.entropy_threshold <= 0:
            raise ValueError("entropy_threshold must be > 0")
        if self.entropy_high_threshold <= self.entropy_threshold:
            raise ValueError(
                "entropy_high_threshold must be > entropy_threshold"
            )
        if self.patience < 1:
            raise ValueError("patience must be ≥ 1")
        if self.min_think_tokens < 0:
            raise ValueError("min_think_tokens must be ≥ 0")
        if self.max_think_tokens < self.min_think_tokens:
            raise ValueError("max_think_tokens must be ≥ min_think_tokens")


class MetaReasoner:
    """
    Runtime thinking-budget monitor for CoT (chain-of-thought) models.

    One instance per request.  Tracks whether the model is inside a thinking
    block and monitors token entropy to decide when to force early termination.

    Methods
    -------
    advance(token_id) : update internal state after each sampled token
    step(logits_np)   : check logits and return True if </think> should be forced
    force_think_end(logits_np) : spike the </think> logit to guarantee selection
    reset()           : clear state for a new request
    """

    def __init__(self, config: MetaReasonerConfig) -> None:
        self._cfg              = config
        self._in_thinking      = False   # True while inside <think>...</think>
        self._think_tokens     = 0       # tokens generated inside <think>
        self._consecutive_low  = 0       # steps with entropy < threshold
        self._consecutive_high = 0       # steps with entropy > high threshold
        self._forced           = False   # already forced </think> this session

    # ── Public API ─────────────────────────────────────────────────────────────

    def advance(self, token_id: int) -> None:
        """
        Update phase tracking after sampling *token_id*.

        Call this AFTER sampling (after :meth:`step`).
        """
        if token_id == self._cfg.think_start_token_id:
            self._in_thinking  = True
            self._think_tokens = 0
            self._consecutive_low  = 0
            self._consecutive_high = 0
            self._forced = False
        elif token_id == self._cfg.think_end_token_id:
            self._in_thinking = False
        elif self._in_thinking:
            self._think_tokens += 1

    def step(self, logits_np: np.ndarray) -> bool:
        """
        Inspect current-step logits and decide whether to force ``</think>``.

        Parameters
        ----------
        logits_np : (vocab_size,) float32 — raw logits from the model

        Returns
        -------
        bool : True when the caller should force the </think> token.
        """
        if not self._in_thinking or self._forced:
            return False

        # Hard cap
        if self._think_tokens >= self._cfg.max_think_tokens:
            self._forced = True
            return True

        # Minimum thinking tokens before we even consider stopping
        if self._think_tokens < self._cfg.min_think_tokens:
            return False

        H = self.compute_entropy(logits_np)

        if H < self._cfg.entropy_threshold:
            self._consecutive_low  += 1
            self._consecutive_high  = 0
        elif H > self._cfg.entropy_high_threshold:
            self._consecutive_high += 1
            self._consecutive_low   = 0
        else:
            self._consecutive_low  = 0
            self._consecutive_high = 0

        if self._consecutive_low >= self._cfg.patience:
            self._forced = True
            return True

        return False

    def force_think_end(self, logits_np: np.ndarray) -> np.ndarray:
        """
        Modify *logits_np* in-place to guarantee the </think> token is sampled.

        Sets the </think> token logit to ``logits_np.max() + 100``.

        Parameters
        ----------
        logits_np : (vocab_size,) float32

        Returns
        -------
        Same array (modified in-place and returned for convenience).
        """
        out = np.asarray(logits_np, dtype=np.float32)
        out[self._cfg.think_end_token_id] = float(out.max()) + 100.0
        return out

    # ── Metrics ───────────────────────────────────────────────────────────────

    @staticmethod
    def compute_entropy(logits_np: np.ndarray) -> float:
        """
        Compute Shannon entropy of the softmax distribution over *logits_np*.

        Parameters
        ----------
        logits_np : (vocab_size,) float32

        Returns
        -------
        float — entropy in nats
        """
        logits = np.asarray(logits_np, dtype=np.float64)
        logits -= logits.max()
        exp_l  = np.exp(logits)
        probs  = exp_l / (exp_l.sum() + 1e-12)
        # Mask zero-probability entries to avoid log(0)
        nz     = probs > 1e-12
        return float(-np.sum(probs[nz] * np.log(probs[nz])))

    @staticmethod
    def think_end_probability(logits_np: np.ndarray, think_end_id: int) -> float:
        """
        Return the softmax probability assigned to the </think> token.

        Parameters
        ----------
        logits_np    : (vocab_size,) float32
        think_end_id : token ID for </think>

        Returns
        -------
        float in [0, 1]
        """
        logits = np.asarray(logits_np, dtype=np.float64)
        logits -= logits.max()
        exp_l  = np.exp(logits)
        probs  = exp_l / (exp_l.sum() + 1e-12)
        if think_end_id < 0 or think_end_id >= len(probs):
            return 0.0
        return float(probs[think_end_id])

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def in_thinking_phase(self) -> bool:
        """True while the model is generating inside a ``<think>`` block."""
        return self._in_thinking

    @property
    def think_tokens_generated(self) -> int:
        """Number of tokens generated inside the current thinking block."""
        return self._think_tokens

    @property
    def consecutive_converged_steps(self) -> int:
        """Steps with entropy below threshold (convergence signal)."""
        return self._consecutive_low

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Reset all state for a new request."""
        self._in_thinking      = False
        self._think_tokens     = 0
        self._consecutive_low  = 0
        self._consecutive_high = 0
        self._forced           = False
