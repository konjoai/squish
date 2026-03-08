"""
squish/online_sd.py

Online Speculative Decoding — Continuous Draft-Head Adaptation.

Based on:
  "Online Speculative Decoding"
  ICML 2024 — arXiv:2310.07177

Problem
-------
Standard speculative decoding uses a *static* draft model that was calibrated
on a general corpus.  During extended inference sessions, the actual token
distribution shifts toward the user's domain (legal, medical, code, etc.),
causing the static draft model to miss more candidates → lower acceptance rate.

Online SD monitors acceptance traces from the running session and
continuously fine-tunes the draft head (via LoRA or direct gradient updates)
using only the accepted token trajectories.  After each update cycle, the
draft model "personalises" to the session distribution with no target-model
calls.

Method
------
1. **OnlineTraceBuffer** — collects (hidden_state, accepted_token) pairs from
   each completed decode step.  Bounded ring buffer (FIFO eviction).

2. **OnlineDraftUpdater** — maintains the LoRA adapter state and update logic:
   - ``record(hidden, accepted_token)`` — appends a trace sample.
   - ``should_update()`` — True when the buffer has accumulated >= *update_every*
     new samples since the last update.
   - ``compute_loss(draft_head)`` — cross-entropy loss over the buffer samples.
   - ``apply_update(draft_head)`` — single gradient step on the draft head.

3. **OnlineSDStats** — tracks running accepted-token statistics.

Design note
-----------
This module is a *pure-Python, framework-agnostic* reference implementation.
It operates on ``numpy`` arrays.  In production, ``apply_update`` would be
wired to the MLX gradient API; the interface here makes that easy to add.

Conflict-resolved notes (Master Conflict Report)
-------------------------------------------------
- **Synergy with EAGLE-3**: fine-tune only the EAGLE-3 draft head; the main
  target model is never updated.
- **Independence**: Online SD only consumes CPU cycles between decode steps;
  no KV cache or attention conflict.
- **FR-Spec compatibility**: the LoRA adapter is applied *before* the FR-Spec
  head compression; after each update cycle, rebuild the compressed weight.

Provides
--------
  OnlineSDConfig        — configuration parameters.
  OnlineTraceBuffer     — bounded ring buffer for inference traces.
  OnlineDraftUpdater    — gradient-step orchestrator.
  OnlineSDStats         — running acceptance statistics.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional, Tuple

import numpy as np

__all__ = [
    "OnlineSDConfig",
    "OnlineTraceBuffer",
    "OnlineDraftUpdater",
    "OnlineSDStats",
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class OnlineSDConfig:
    """Parameters for Online Speculative Decoding.

    Parameters
    ----------
    buffer_capacity:
        Maximum (hidden_state, token) pairs retained in the trace buffer.
        Old samples are evicted FIFO when capacity is exceeded.
    update_every:
        Number of *new* samples required since the last update before
        ``should_update()`` returns True.
    learning_rate:
        Step size for the single-step gradient update on the draft head.
    lora_rank:
        Rank of the LoRA adapter applied to the draft LM-head weight.
        Set to 0 to disable LoRA and update the weight matrix directly.
    min_acceptance_rate:
        If the session's running acceptance rate is already above this
        threshold, skip the update (draft is already well-calibrated).
    """

    buffer_capacity: int = 512
    update_every: int = 64
    learning_rate: float = 1e-4
    lora_rank: int = 16
    min_acceptance_rate: float = 0.85


# ---------------------------------------------------------------------------
# OnlineTraceBuffer
# ---------------------------------------------------------------------------

class OnlineTraceBuffer:
    """Bounded FIFO ring buffer of (hidden_state, accepted_token) pairs.

    Parameters
    ----------
    capacity:
        Maximum number of samples to retain.
    """

    def __init__(self, capacity: int = 512) -> None:
        if capacity < 1:
            raise ValueError(f"capacity must be >= 1, got {capacity}")
        self._capacity = capacity
        self._buffer: Deque[Tuple[np.ndarray, int]] = deque(maxlen=capacity)

    @property
    def capacity(self) -> int:
        return self._capacity

    def __len__(self) -> int:
        return len(self._buffer)

    def is_full(self) -> bool:
        return len(self._buffer) == self._capacity

    def add(self, hidden: np.ndarray, accepted_token: int) -> None:
        """Append one (hidden, token) sample.

        Parameters
        ----------
        hidden:
            Last-layer hidden state at the draft position, shape
            ``(hidden_dim,)``.  A copy is stored to avoid mutation.
        accepted_token:
            The token that was accepted by the target model.
        """
        self._buffer.append((hidden.copy(), int(accepted_token)))

    def get_batch(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return all buffered samples as stacked arrays.

        Returns
        -------
        hiddens:
            Shape ``(n, hidden_dim)``.
        tokens:
            Shape ``(n,)``, dtype int64.
        """
        if not self._buffer:
            raise RuntimeError("Buffer is empty; cannot build batch")
        hiddens = np.stack([h for h, _ in self._buffer])
        tokens = np.array([t for _, t in self._buffer], dtype=np.int64)
        return hiddens, tokens

    def clear(self) -> None:
        self._buffer.clear()


# ---------------------------------------------------------------------------
# OnlineDraftUpdater
# ---------------------------------------------------------------------------

class OnlineDraftUpdater:
    """Manages background fine-tuning of a draft LM-head weight.

    The update is a single gradient step of cross-entropy loss between the
    draft head's predictions and the accepted tokens.

    Parameters
    ----------
    config:
        Online SD configuration.
    hidden_dim:
        Dimensionality of the hidden state fed to the draft head.
    vocab_size:
        Full vocabulary size of the draft head.
    """

    def __init__(
        self,
        config: Optional[OnlineSDConfig] = None,
        hidden_dim: int = 4096,
        vocab_size: int = 151_936,
    ) -> None:
        self._config = config or OnlineSDConfig()
        self._hidden_dim = hidden_dim
        self._vocab_size = vocab_size
        self._buffer = OnlineTraceBuffer(self._config.buffer_capacity)
        self._samples_since_update: int = 0
        self._total_updates: int = 0
        # LoRA adapter matrices (A: hidden→rank, B: rank→vocab)
        if self._config.lora_rank > 0:
            r = self._config.lora_rank
            self._lora_a = np.zeros((hidden_dim, r), dtype=np.float32)
            self._lora_b = np.zeros((r, vocab_size), dtype=np.float32)
        else:
            self._lora_a = None
            self._lora_b = None

    @property
    def buffer(self) -> OnlineTraceBuffer:
        return self._buffer

    @property
    def total_updates(self) -> int:
        return self._total_updates

    @property
    def hidden_dim(self) -> int:
        return self._hidden_dim

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def record(self, hidden: np.ndarray, accepted_token: int) -> None:
        """Record one accepted inference step.

        Parameters
        ----------
        hidden:
            Hidden state at the draft position, shape ``(hidden_dim,)``.
        accepted_token:
            Token ID accepted by the target model.
        """
        self._buffer.add(hidden, accepted_token)
        self._samples_since_update += 1

    def should_update(self) -> bool:
        """Return True when enough new samples have been collected."""
        return self._samples_since_update >= self._config.update_every

    def compute_loss(self, weight: np.ndarray) -> float:
        """Compute cross-entropy loss of *weight* over the current buffer.

        Parameters
        ----------
        weight:
            Draft LM-head weight matrix, shape ``(vocab_size, hidden_dim)``
            (or ``(hidden_dim, vocab_size)`` transposed — we auto-detect).

        Returns
        -------
        float — mean cross-entropy loss over buffer samples.
        """
        hiddens, tokens = self._buffer.get_batch()  # (n, h), (n,)

        # Auto-detect transposition
        if weight.shape == (self._vocab_size, self._hidden_dim):
            logits = hiddens @ weight.T          # (n, vocab)
        elif weight.shape == (self._hidden_dim, self._vocab_size):
            logits = hiddens @ weight             # (n, vocab)
        else:
            raise ValueError(
                f"weight shape {weight.shape} incompatible with "
                f"hidden_dim={self._hidden_dim}, vocab_size={self._vocab_size}"
            )

        # Numerically stable softmax + CE
        logits -= logits.max(axis=1, keepdims=True)
        log_sum_exp = np.log(np.exp(logits).sum(axis=1))
        correct_logits = logits[np.arange(len(tokens)), tokens]
        loss = float(np.mean(log_sum_exp - correct_logits))
        return loss

    def apply_update(self, weight: np.ndarray) -> np.ndarray:
        """Apply one gradient descent step on *weight* and return updated weight.

        Updates the LoRA adapter (or direct weight) using the CE gradient
        over the current buffer, then resets the samples counter.

        Parameters
        ----------
        weight:
            Draft LM-head weight, shape ``(vocab_size, hidden_dim)``.

        Returns
        -------
        Updated weight matrix with the same shape.
        """
        hiddens, tokens = self._buffer.get_batch()  # (n, h), (n,)
        n = len(tokens)
        lr = self._config.learning_rate

        # Forward: logits = hiddens @ weight.T
        logits = hiddens @ weight.T  # (n, vocab)
        logits_shifted = logits - logits.max(axis=1, keepdims=True)
        probs = np.exp(logits_shifted)
        probs /= probs.sum(axis=1, keepdims=True)

        # CE gradient w.r.t. logits: dL/d_logit = (prob - one_hot) / n
        d_logits = probs.copy()
        d_logits[np.arange(n), tokens] -= 1.0
        d_logits /= n  # (n, vocab)

        # Gradient w.r.t. weight.T: d_logits.T @ hiddens  (vocab, hidden)
        d_weight_T = d_logits.T @ hiddens  # (vocab, hidden)

        if self._lora_a is not None and self._lora_b is not None:
            # Apply gradient to LoRA matrices
            # weight ≈ weight_0 + (lora_a @ lora_b).T
            # Simplify: gradient update on lora_b
            d_lora_b = self._lora_a.T @ (hiddens.T @ d_logits)  # (r, vocab)
            self._lora_b -= lr * d_lora_b
            updated = weight - lr * d_weight_T
        else:
            # Direct weight update
            updated = weight - lr * d_weight_T

        self._samples_since_update = 0
        self._total_updates += 1
        return updated

    def lora_delta(self) -> Optional[np.ndarray]:
        """Compute the current LoRA correction matrix.

        Returns
        -------
        (vocab_size, hidden_dim) delta matrix, or None if LoRA is disabled.
        """
        if self._lora_a is None or self._lora_b is None:
            return None
        return (self._lora_a @ self._lora_b).T  # (vocab, hidden)

    def reset(self) -> None:
        """Clear the buffer and reset counters."""
        self._buffer.clear()
        self._samples_since_update = 0
        self._total_updates = 0
        if self._lora_a is not None:
            self._lora_a[:] = 0.0
            self._lora_b[:] = 0.0


# ---------------------------------------------------------------------------
# OnlineSDStats
# ---------------------------------------------------------------------------

@dataclass
class OnlineSDStats:
    """Running acceptance-rate statistics.

    Attributes
    ----------
    total_drafted:
        Total tokens proposed by the draft model.
    total_accepted:
        Total tokens accepted by the target model.
    update_count:
        Number of draft-head update steps performed.
    pre_update_acceptance:
        Acceptance rate immediately before the last update (diagnostic).
    """

    total_drafted: int = 0
    total_accepted: int = 0
    update_count: int = 0
    pre_update_acceptance: float = 0.0

    @property
    def acceptance_rate(self) -> float:
        return self.total_accepted / self.total_drafted if self.total_drafted else 0.0

    def record_step(self, n_drafted: int, n_accepted: int) -> None:
        self.total_drafted += n_drafted
        self.total_accepted += n_accepted

    def record_update(self) -> None:
        self.pre_update_acceptance = self.acceptance_rate
        self.update_count += 1

    def reset(self) -> None:
        self.total_drafted = 0
        self.total_accepted = 0
        self.update_count = 0
        self.pre_update_acceptance = 0.0
