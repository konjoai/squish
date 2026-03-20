"""
squish/speculative/mtp_head.py

MultiTokenPredictor — Auxiliary N-Head Multi-Token Prediction.

Based on:
  "Better & Faster Large Language Models via Multi-token Prediction"
  Gloeckle et al., Meta FAIR — ICML 2024  —  arXiv:2404.19737

  Also used in:
  DeepSeek-V3 Technical Report (Dec 2024) — MTP with n=1 additional head
  achieving 1.5–1.7× throughput boost at inference time.

Background
----------
Standard LLMs have a single language-model head: given hidden state h_t,
they predict token t+1.  Multi-Token Prediction (MTP) adds N *auxiliary*
heads that independently predict tokens t+2, t+3, …, t+N+1 from the same
hidden state h_t.

At **inference** time (no training changes to the base model required when
heads are installed as add-ons):
  1. Run one forward pass → hidden states H.
  2. Head-0 (standard LM head) → token t+1.
  3. Head-k (k=1…N) → token t+k+1 (in parallel).
  4. Accept each auxiliary prediction if it matches a greedy continuation.

This provides a "speculative" boost:
  - Worst case: only head-0 accepted (same as AR).
  - Best case: all N+1 tokens accepted (no additional forward pass).

Implementation notes
--------------------
This module provides the *inference-side* multi-token prediction layer.
It does not implement training.  The auxiliary heads are modelled as simple
linear projections (vocab_size × emb_dim weight matrices), dimensionally
compatible with weight-tying to the base embedding.

Classes
-------
``MTPHeadConfig``    — configuration
``MTPHeadLayer``     — single linear head (one prediction depth)
``MTPHeadStats``     — per-instance statistics
``MultiTokenPredictor`` — N parallel independent heads

Usage::

    from squish.speculative.mtp_head import MTPHeadConfig, MultiTokenPredictor

    cfg = MTPHeadConfig(n_heads=4, vocab_size=32000, emb_dim=256)
    mtp = MultiTokenPredictor(cfg)

    # hidden_state: (emb_dim,) float32
    tokens, probs = mtp.sample_tokens(hidden_state)
    # tokens: list of n_heads ints (predicted tokens at positions +1..+n_heads)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

__all__ = [
    "MTPHeadConfig",
    "MTPHeadLayer",
    "MTPHeadStats",
    "MultiTokenPredictor",
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class MTPHeadConfig:
    """Configuration for multi-token prediction heads.

    Attributes:
        n_heads:     Number of auxiliary heads (predicts t+1 … t+n_heads).
        vocab_size:  Vocabulary size (output dim of each head).
        emb_dim:     Input embedding dimension.
        temperature: Sampling temperature (0 = greedy argmax).
        seed:        RNG seed.
        tie_weights: If True, share a single weight matrix across all heads
                     (weight-tying; reduces parameters but less expressive).
    """

    n_heads: int = 4
    vocab_size: int = 32_000
    emb_dim: int = 256
    temperature: float = 0.0
    seed: int = 42
    tie_weights: bool = False

    def __post_init__(self) -> None:
        if self.n_heads < 1:
            raise ValueError(f"n_heads must be >= 1, got {self.n_heads}")
        if self.vocab_size < 2:
            raise ValueError(f"vocab_size must be >= 2, got {self.vocab_size}")
        if self.emb_dim < 1:
            raise ValueError(f"emb_dim must be >= 1, got {self.emb_dim}")
        if self.temperature < 0:
            raise ValueError(f"temperature must be >= 0, got {self.temperature}")


# ---------------------------------------------------------------------------
# Single head layer
# ---------------------------------------------------------------------------


class MTPHeadLayer:
    """Single auxiliary prediction head: linear projection emb_dim → vocab_size.

    Parameters
    ----------
    emb_dim:    Input dimension.
    vocab_size: Output dimension.
    seed:       Weight initialization seed.
    """

    def __init__(self, emb_dim: int, vocab_size: int, seed: int = 42) -> None:
        rng = np.random.default_rng(seed)
        scale = (2.0 / (emb_dim + vocab_size)) ** 0.5
        self.weight: np.ndarray = rng.normal(0, scale, (vocab_size, emb_dim)).astype(np.float32)
        self._emb_dim = emb_dim
        self._vocab_size = vocab_size

    def forward(self, hidden: np.ndarray) -> np.ndarray:
        """Project hidden state to vocabulary logits.

        Parameters
        ----------
        hidden: (emb_dim,) or (batch, emb_dim) float32

        Returns
        -------
        logits: (vocab_size,) or (batch, vocab_size) float32
        """
        return hidden @ self.weight.T

    def set_weight(self, weight: np.ndarray) -> None:
        """Replace head weight (e.g. from a trained checkpoint)."""
        if weight.shape != self.weight.shape:
            raise ValueError(
                f"Weight shape mismatch: expected {self.weight.shape}, got {weight.shape}"
            )
        self.weight = weight.astype(np.float32)

    @property
    def nbytes(self) -> int:
        return self.weight.nbytes

    def __repr__(self) -> str:
        return f"MTPHeadLayer(emb_dim={self._emb_dim}, vocab={self._vocab_size})"


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


@dataclass
class MTPHeadStats:
    """Lifetime statistics for a MultiTokenPredictor.

    Attributes:
        total_forward_calls:  Number of calls to ``sample_tokens``.
        total_tokens_accepted: Total tokens accepted across all calls.
        head_accept_counts:   Per-head acceptance counts (list of n_heads ints).
    """

    total_forward_calls: int = 0
    total_tokens_accepted: int = 0
    head_accept_counts: List[int] = field(default_factory=list)

    def acceptance_rate_per_head(self) -> List[float]:
        """Per-head acceptance rate (fraction of calls where head k was accepted)."""
        if self.total_forward_calls == 0:
            return [0.0] * len(self.head_accept_counts)
        return [c / self.total_forward_calls for c in self.head_accept_counts]

    @property
    def mean_tokens_per_call(self) -> float:
        if self.total_forward_calls == 0:
            return 0.0
        return self.total_tokens_accepted / self.total_forward_calls

    def __repr__(self) -> str:
        rates = self.acceptance_rate_per_head()
        rate_str = ", ".join(f"{r:.2f}" for r in rates)
        return (
            f"MTPHeadStats(calls={self.total_forward_calls}, "
            f"tokens={self.total_tokens_accepted}, "
            f"head_rates=[{rate_str}])"
        )


# ---------------------------------------------------------------------------
# Multi-Token Predictor
# ---------------------------------------------------------------------------


def _softmax(logits: np.ndarray) -> np.ndarray:
    x = logits - logits.max()
    e = np.exp(x)
    return e / e.sum()


class MultiTokenPredictor:
    """N-head parallel multi-token prediction for fast inference.

    Each of the N heads independently predicts one future token position
    (t+1, t+2, … t+N) from the *same* hidden state.  At inference time,
    all N predictions happen in one Python call (no extra forward passes),
    yielding multi-token proposals that can be verified greedily.

    Parameters
    ----------
    config:
        Predictor configuration.
    """

    def __init__(self, config: Optional[MTPHeadConfig] = None) -> None:
        self._cfg = config or MTPHeadConfig()
        cfg = self._cfg
        self._rng = np.random.default_rng(cfg.seed)

        if cfg.tie_weights:
            # Single shared weight for all heads
            shared = MTPHeadLayer(cfg.emb_dim, cfg.vocab_size, seed=cfg.seed)
            self._heads: List[MTPHeadLayer] = [shared] * cfg.n_heads
        else:
            self._heads = [
                MTPHeadLayer(cfg.emb_dim, cfg.vocab_size, seed=cfg.seed + i)
                for i in range(cfg.n_heads)
            ]

        self.stats = MTPHeadStats(head_accept_counts=[0] * cfg.n_heads)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, hidden_state: np.ndarray) -> List[np.ndarray]:
        """Compute logits from all N heads.

        Parameters
        ----------
        hidden_state: (emb_dim,) float32

        Returns
        -------
        List of (vocab_size,) logit arrays, one per head.
        """
        h = np.asarray(hidden_state, dtype=np.float32).ravel()
        return [head.forward(h) for head in self._heads]

    def sample_tokens(
        self, hidden_state: np.ndarray
    ) -> Tuple[List[int], List[np.ndarray]]:
        """Sample one token per head.

        Parameters
        ----------
        hidden_state: (emb_dim,) float32

        Returns
        -------
        tokens : List[int] — one sampled token per head.
        probs  : List[np.ndarray] — (vocab_size,) probability distributions.
        """
        cfg = self._cfg
        logits_list = self.forward(hidden_state)
        tokens: List[int] = []
        probs_list: List[np.ndarray] = []
        for logits in logits_list:
            probs = _softmax(logits)
            if cfg.temperature <= 1e-9:
                tok = int(np.argmax(logits))
            else:
                tok = int(self._rng.choice(cfg.vocab_size, p=probs))
            tokens.append(tok)
            probs_list.append(probs)
        self.stats.total_forward_calls += 1
        return tokens, probs_list

    def verify_against_target(
        self,
        hidden_state: np.ndarray,
        target_logits: List[np.ndarray],
    ) -> Tuple[List[int], List[bool]]:
        """Greedy-verify MTP proposals against target model logits.

        Parameters
        ----------
        hidden_state:
            (emb_dim,) hidden state.
        target_logits:
            List of (vocab_size,) arrays from the target model for each
            speculative position.

        Returns
        -------
        tokens   : Accepted token ids (prefix up to first mismatch).
        accepted : Per-head boolean mask.
        """
        tokens, _ = self.sample_tokens(hidden_state)
        accepted: List[bool] = []
        result_tokens: List[int] = []
        for i, (tok, t_logits) in enumerate(zip(tokens, target_logits)):
            t_tok = int(np.argmax(t_logits))
            if tok == t_tok:
                accepted.append(True)
                result_tokens.append(tok)
                self.stats.head_accept_counts[i] += 1
            else:
                accepted.append(False)
                result_tokens.append(t_tok)  # correction
                break
        self.stats.total_tokens_accepted += len(result_tokens)
        return result_tokens, accepted

    def acceptance_rate(self, head_idx: int = 0) -> float:
        """Fraction of calls where head ``head_idx`` was accepted."""
        calls = self.stats.total_forward_calls
        if calls == 0:
            return 0.0
        return self.stats.head_accept_counts[head_idx] / calls

    def set_head_weights(self, idx: int, weight: np.ndarray) -> None:
        """Load a trained head weight matrix for head ``idx``."""
        self._heads[idx].set_weight(weight)

    def total_parameters(self) -> int:
        """Total number of parameters in all head layers."""
        if self._cfg.tie_weights:
            return self._heads[0].weight.size
        return sum(h.weight.size for h in self._heads)

    def __repr__(self) -> str:
        return (
            f"MultiTokenPredictor(n_heads={self._cfg.n_heads}, "
            f"vocab={self._cfg.vocab_size}, emb_dim={self._cfg.emb_dim}, "
            f"{self.stats})"
        )
