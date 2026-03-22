"""squish/sampling/contrastive_search.py

ContrastiveSearch — Repetition-free token generation via contrastive decoding.

Reference
---------
Su & Collier. "A Contrastive Framework for Neural Text Generation."
NeurIPS 2022. arXiv:2202.06417. (Widely used in production, 2024.)

Algorithm
---------
At each decoding step, given vocabulary logits l and the sequence of recent
context token embeddings E_ctx ∈ R^{context_len × d_embed}:

1. Select the top-k candidates from l.
2. For each candidate c:
   a. model_score(c) = p(c)  — softmax probability.
   b. degeneration_penalty(c) = max_{e_ctx ∈ E_ctx} cosine_sim(e_c, e_ctx)
      where e_c is the embedding of token c.
3. contrastive_score(c) = α × model_score(c) − (1 − α) × degeneration_penalty(c).
4. Select argmax of contrastive_score as the next token.

The parameter α balances fluency (model probability) vs non-repetition
(penalizing tokens whose embeddings are similar to recent context).

This is a *single-model* approach distinct from contrastive decoding
(two separate models: an expert + amateur).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class ContrastiveSearchConfig:
    """Configuration for ContrastiveSearch.

    Parameters
    ----------
    top_k:
        Number of top candidates to consider at each step.
    alpha:
        Weight for model probability vs degeneration penalty.
        alpha=1.0 → pure greedy; alpha=0.0 → purely anti-repetitive.
        Typical value: 0.6.
    context_window:
        Number of recent token embeddings to use for degeneration penalty.
    embed_dim:
        Dimensionality of token embeddings.
    vocab_size:
        Vocabulary size (for generating random embeddings in tests).
    seed:
        RNG seed for embedding table initialization (testing/simulation).
    """

    top_k: int = 5
    alpha: float = 0.6
    context_window: int = 16
    embed_dim: int = 4096
    vocab_size: int = 32000
    seed: int = 0

    def __post_init__(self) -> None:
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        if self.top_k < 1:
            raise ValueError("top_k must be >= 1")


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class ContrastiveResult:
    """Output of a contrastive search step.

    Parameters
    ----------
    token_id:
        Selected token index.
    model_score:
        Softmax probability of the selected token.
    degeneration_penalty:
        Max cosine similarity of the token embedding with recent context.
    contrastive_score:
        Final combined score.
    candidates:
        Top-k candidate token ids considered.
    """

    token_id: int
    model_score: float
    degeneration_penalty: float
    contrastive_score: float
    candidates: np.ndarray


# ---------------------------------------------------------------------------
# Searcher
# ---------------------------------------------------------------------------

class ContrastiveSearch:
    """Single-model contrastive search decoder.

    Parameters
    ----------
    config:
        ContrastiveSearch configuration.
    embeddings:
        Token embedding table ``(vocab_size, embed_dim)``.  If None, a
        random table is created for testing purposes.
    """

    def __init__(
        self,
        config: Optional[ContrastiveSearchConfig] = None,
        embeddings: Optional[np.ndarray] = None,
    ) -> None:
        self._cfg = config or ContrastiveSearchConfig()
        rng = np.random.default_rng(self._cfg.seed)
        if embeddings is not None:
            self._embeddings = np.asarray(embeddings, dtype=np.float32)
        else:
            self._embeddings = rng.standard_normal(
                (self._cfg.vocab_size, self._cfg.embed_dim)
            ).astype(np.float32)
        # L2-normalize the embedding table for efficient cosine similarity
        norms = np.linalg.norm(self._embeddings, axis=1, keepdims=True) + 1e-8
        self._embeddings_norm = self._embeddings / norms

        # Context buffer: recent token embeddings
        self._context: list[np.ndarray] = []

    @property
    def config(self) -> ContrastiveSearchConfig:
        return self._cfg

    @property
    def context_len(self) -> int:
        """Number of token embeddings currently in the context buffer."""
        return len(self._context)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        logits = logits - logits.max()
        exp = np.exp(logits)
        return exp / (exp.sum() + 1e-38)

    def _degeneration_penalty(self, token_id: int) -> float:
        """Max cosine similarity between token_id's embedding and context."""
        if not self._context:
            return 0.0
        e_c = self._embeddings_norm[token_id]  # (embed_dim,)
        # Stack recent context
        ctx = np.stack(self._context[-self._cfg.context_window:])  # (L, d)
        norms = np.linalg.norm(ctx, axis=1, keepdims=True) + 1e-8
        ctx_norm = ctx / norms
        sims = ctx_norm @ e_c  # (L,)
        return float(sims.max())

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(self, logits: np.ndarray) -> ContrastiveResult:
        """Select the next token using contrastive search.

        Parameters
        ----------
        logits:
            Raw logits ``(vocab_size,)``.

        Returns
        -------
        ContrastiveResult
        """
        logits = np.asarray(logits, dtype=np.float32).ravel()
        probs = self._softmax(logits)

        # Top-k candidate selection
        k = min(self._cfg.top_k, len(probs))
        top_k_idx = np.argpartition(probs, -k)[-k:]
        top_k_probs = probs[top_k_idx]

        # Compute contrastive score for each candidate
        scores = np.empty(k, dtype=np.float32)
        penalties = np.empty(k, dtype=np.float32)
        for i, tok in enumerate(top_k_idx):
            pen = self._degeneration_penalty(int(tok))
            penalties[i] = pen
            scores[i] = (
                self._cfg.alpha * float(top_k_probs[i])
                - (1.0 - self._cfg.alpha) * pen
            )

        best_i = int(np.argmax(scores))
        best_tok = int(top_k_idx[best_i])

        # Update context buffer
        self._context.append(self._embeddings[best_tok].copy())
        if len(self._context) > self._cfg.context_window:
            self._context.pop(0)

        return ContrastiveResult(
            token_id=best_tok,
            model_score=float(top_k_probs[best_i]),
            degeneration_penalty=float(penalties[best_i]),
            contrastive_score=float(scores[best_i]),
            candidates=top_k_idx.astype(np.int32),
        )

    def reset_context(self) -> None:
        """Clear the context buffer (start of a new sequence)."""
        self._context.clear()

    def generate(self, initial_logits_seq: list[np.ndarray]) -> list[int]:
        """Generate a sequence of token ids from a list of logit tensors.

        Parameters
        ----------
        initial_logits_seq:
            List of ``(vocab_size,)`` logit arrays, one per step.

        Returns
        -------
        list[int]
            Sampled token ids.
        """
        self.reset_context()
        return [self.step(logits).token_id for logits in initial_logits_seq]
