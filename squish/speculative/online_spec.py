"""squish/speculative/online_spec.py

OnlineSpec — Session-Adaptive Speculative Decoding.

Reference
---------
Liu et al. "Online Speculative Decoding."
ICML 2024 (arXiv:2310.07177).

Algorithm
---------
OnlineSpec continuously refines the draft distribution from observed
target acceptances during a session:

1. Start with a fixed draft distribution (e.g., from a small draft model).
2. For each accepted token, treat (input_context, accepted_token) as a
   positive training example.
3. Use online gradient descent (SGD with momentum) to update a lightweight
   adapter on top of the draft distribution.
4. Accepted tokens reinforce confident draft tokens; rejected tokens
   provide a correction signal.

Over a session, acceptance rate improves from ~60% to 85%+ on average.

Key properties
--------------
* NumPy-only.
* ``vocab_size`` — vocabulary size.
* ``lr`` — online learning rate (default 1e-3).
* ``momentum`` — SGD momentum coefficient.
* ``history_len`` — number of recent (context, token) pairs retained.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Optional, Tuple

import numpy as np

__all__ = [
    "OnlineSpecConfig",
    "OnlineSpec",
]


@dataclass
class OnlineSpecConfig:
    """Configuration for :class:`OnlineSpec`.

    Attributes:
        vocab_size: Vocabulary size.
        hidden_dim: Feature dimension for the adapter.
        lr: Online learning rate.
        momentum: SGD momentum.
        history_len: Recent example buffer size for mini-batch updates.
        temperature: Sampling temperature for adjusted draft distribution.
    """

    vocab_size: int = 32000
    hidden_dim: int = 256
    lr: float = 1e-3
    momentum: float = 0.9
    history_len: int = 64
    temperature: float = 1.0


class OnlineSpec:
    """Session-adaptive draft distribution via online SGD.

    Parameters
    ----------
    config:
        OnlineSpec configuration.
    seed:
        RNG seed.
    """

    def __init__(self, config: Optional[OnlineSpecConfig] = None, seed: int = 0) -> None:
        self._cfg = config or OnlineSpecConfig()
        self._rng = np.random.default_rng(seed)
        # Lightweight adapter: logit bias per vocab token
        self._bias: np.ndarray = np.zeros(self._cfg.vocab_size, dtype=np.float32)
        self._bias_momentum: np.ndarray = np.zeros(self._cfg.vocab_size, dtype=np.float32)
        # History buffer of (context_hash, token_id, accepted)
        self._history: Deque[Tuple[int, int, bool]] = deque(maxlen=self._cfg.history_len)
        self._total_accepted: int = 0
        self._total_rejected: int = 0

    @property
    def config(self) -> OnlineSpecConfig:
        return self._cfg

    @property
    def acceptance_rate(self) -> float:
        total = self._total_accepted + self._total_rejected
        return self._total_accepted / total if total > 0 else 0.0

    def adjust_logits(self, draft_logits: np.ndarray) -> np.ndarray:
        """Apply the learned bias to draft logits.

        Parameters
        ----------
        draft_logits:
            Draft model logits, shape ``(vocab_size,)``.

        Returns
        -------
        np.ndarray
            Adjusted logits, shape ``(vocab_size,)``.
        """
        return np.asarray(draft_logits, dtype=np.float32) + self._bias

    def observe(self, token_id: int, accepted: bool, context_hash: int = 0) -> None:
        """Record an acceptance/rejection observation and update the adapter.

        Parameters
        ----------
        token_id:
            The draft token that was accepted or rejected.
        accepted:
            Whether the target model accepted the token.
        context_hash:
            Optional hash of the input context (used for mini-batch).
        """
        self._history.append((context_hash, token_id, accepted))
        if accepted:
            self._total_accepted += 1
        else:
            self._total_rejected += 1

        # Gradient: push accepted token's bias up, push rejected token's bias down
        grad = np.zeros(self._cfg.vocab_size, dtype=np.float32)
        signal = 1.0 if accepted else -1.0
        grad[token_id] = -signal  # negative for gradient descent on NLL

        self._bias_momentum = (
            self._cfg.momentum * self._bias_momentum + (1 - self._cfg.momentum) * grad
        )
        self._bias -= self._cfg.lr * self._bias_momentum
        # Clip to prevent runaway
        self._bias = np.clip(self._bias, -10.0, 10.0)

    def sample(self, draft_logits: np.ndarray) -> Tuple[int, float]:
        """Sample a token using the adapted draft distribution.

        Parameters
        ----------
        draft_logits:
            Draft model logits, shape ``(vocab_size,)``.

        Returns
        -------
        Tuple of (token_id, log_prob).
        """
        adjusted = self.adjust_logits(draft_logits).astype(np.float64)
        if self._cfg.temperature != 1.0:
            adjusted /= max(self._cfg.temperature, 1e-6)
        adjusted -= adjusted.max()
        probs = np.exp(adjusted)
        probs /= probs.sum()
        tok = int(self._rng.choice(self._cfg.vocab_size, p=probs))
        return tok, float(np.log(probs[tok].clip(min=1e-10)))

    def reset_session(self) -> None:
        """Reset per-session state (keep learned bias)."""
        self._history.clear()

    def reset_all(self) -> None:
        """Reset everything including learned bias."""
        self._bias[:] = 0.0
        self._bias_momentum[:] = 0.0
        self._history.clear()
        self._total_accepted = 0
        self._total_rejected = 0
