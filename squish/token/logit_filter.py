"""logit_filter.py — Approximate Top-K Vocab Filtering Before LM Head

At each decode step, instead of computing the full LM head matrix multiply:
    logits = hidden_state @ vocab_embed.T   (d_model × vocab_size → expensive)

LogitFilter uses a two-stage approximation to reduce cost:

Stage 1 — Sketch projection:
    sketch_scores = hidden_state @ sketch.T    (d_model × sketch_dim)
    The sketch is a precomputed random projection of the vocabulary embedding
    matrix (sketch = vocab_embed @ R, R ∈ R^{d_model × sketch_dim}).

Stage 2 — Approximate logit estimation:
    approx_logits = sketch_scores @ sketch_approx_T
    Tokens with low approx_logit are unlikely to be in the top-k.

Stage 3 — Candidate selection:
    top_k_indices = top-k(approx_logits)

Stage 4 — Exact LM head for candidates only:
    exact_logits[top_k_indices] = hidden_state @ vocab_embed[top_k_indices].T

Cost reduction: vocab_size → top_k exact dot products + sketch_dim approx.
At top_k=1024, vocab=32k, sketch_dim=256: ~30× fewer FLOPs for the LM head.

The sketch is built once from the embedding matrix and reused across all decode
steps (and all requests once the model is loaded).

Usage:
    filt = LogitFilter.from_embedding_matrix(vocab_embed_np, top_k=1024)
    # At each decode step:
    exact_logits = filt.filter_and_score(hidden_state_1d, vocab_embed_np)
    # exact_logits is a (vocab_size,) array: exact for top_k, -inf elsewhere
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class LogitFilterConfig:
    """Configuration for LogitFilter.

    Args:
        top_k:       Number of vocab tokens to score exactly.
        sketch_dim:  Dimensionality of the random projection sketch.
                     Higher → more accurate filtering, more cost.
        seed:        RNG seed for reproducible sketch construction.
        always_include_last_token: Always include the most recent input token
                     as a candidate (prevents filtering away the continuation).
    """
    top_k: int = 1024
    sketch_dim: int = 256
    seed: int = 42
    always_include_last_token: bool = True


@dataclass
class LogitFilterStats:
    """Accumulated statistics."""
    filter_calls: int = 0
    total_vocab_size: int = 0
    total_candidates_exact: int = 0

    @property
    def mean_compression_ratio(self) -> float:
        if self.total_vocab_size == 0:
            return 1.0
        return self.total_candidates_exact / self.total_vocab_size


class LogitFilter:
    """Two-stage approximate vocab filter for fast LM head scoring.

    Build via ``LogitFilter.from_embedding_matrix()`` or
    ``LogitFilter(config, sketch)`` if you supply the sketch directly.
    """

    def __init__(
        self,
        config: LogitFilterConfig,
        sketch: np.ndarray,                     # (vocab_size, sketch_dim)
        projection_T: Optional[np.ndarray] = None,  # (sketch_dim, d_model)
    ) -> None:
        self.config = config
        self._sketch = sketch.astype(np.float32)
        self._projection_T: Optional[np.ndarray] = (
            projection_T.astype(np.float32) if projection_T is not None else None
        )
        self.stats = LogitFilterStats()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_embedding_matrix(
        cls,
        vocab_embed: np.ndarray,   # (vocab_size, d_model)
        config: Optional[LogitFilterConfig] = None,
    ) -> "LogitFilter":
        """Build a LogitFilter from a vocabulary embedding matrix.

        The sketch is computed once: sketch = vocab_embed @ R where R is a
        random Gaussian projection matrix (d_model → sketch_dim).
        """
        cfg = config or LogitFilterConfig()
        vocab_size, d_model = vocab_embed.shape
        if cfg.top_k > vocab_size:
            raise ValueError(
                f"top_k ({cfg.top_k}) must be <= vocab_size ({vocab_size})"
            )
        rng = np.random.default_rng(cfg.seed)
        # Random orthogonal-ish projection (sign matrix for speed)
        # Use normalized random Gaussian: sketch = vocab_embed @ R / sqrt(sketch_dim)
        R = rng.standard_normal((d_model, cfg.sketch_dim)).astype(np.float32)
        R /= math.sqrt(cfg.sketch_dim)
        sketch = (vocab_embed.astype(np.float32) @ R)  # (vocab_size, sketch_dim)
        return cls(config=cfg, sketch=sketch, projection_T=R.T)  # R.T: (sketch_dim, d_model)

    @classmethod
    def from_embedding_norms(
        cls,
        vocab_size: int,
        d_model: int,
        config: Optional[LogitFilterConfig] = None,
    ) -> "LogitFilter":
        """Build a synthetic LogitFilter (for testing without real embeddings).

        Creates a random stand-in sketch of the right shape.
        """
        cfg = config or LogitFilterConfig()
        rng = np.random.default_rng(cfg.seed)
        sketch = rng.standard_normal((vocab_size, cfg.sketch_dim)).astype(
            np.float32
        )
        return cls(config=cfg, sketch=sketch)

    # ------------------------------------------------------------------
    # Core filtering
    # ------------------------------------------------------------------

    def select_candidates(
        self,
        hidden_state: np.ndarray,  # (d_model,) — last hidden state
        last_token_id: Optional[int] = None,
    ) -> np.ndarray:
        """Return the indices of the top-k candidate tokens.

        Uses the sketch for fast approximate scoring.

        Args:
            hidden_state:  1-D float array of shape (d_model,).
            last_token_id: If set and always_include_last_token is True,
                           the last input token ID is always included.

        Returns:
            np.ndarray of shape (k,) with sorted candidate token indices.
        """
        if hidden_state.ndim != 1:
            raise ValueError(
                f"hidden_state must be 1-D, got shape {hidden_state.shape}"
            )
        cfg = self.config
        vocab_size = self._sketch.shape[0]
        effective_k = min(cfg.top_k, vocab_size)

        h = hidden_state.astype(np.float32)
        # Project hidden state to sketch dimension if projection is available
        if self._projection_T is not None:
            h = self._projection_T @ h  # (sketch_dim,)
        # Approximate scores via sketch: (vocab_size,)
        approx_scores = self._sketch @ h  # (vocab_size,)

        # Select top-k candidate indices
        if effective_k >= vocab_size:
            candidates = np.arange(vocab_size)
        else:
            candidates = np.argpartition(approx_scores, -effective_k)[-effective_k:]

        # Optionally ensure last_token_id is included
        if (
            cfg.always_include_last_token
            and last_token_id is not None
            and 0 <= last_token_id < vocab_size
        ):
            candidates = np.union1d(candidates, [last_token_id])

        return np.sort(candidates)

    def filter_and_score(
        self,
        hidden_state: np.ndarray,       # (d_model,)
        vocab_embed: np.ndarray,        # (vocab_size, d_model)
        last_token_id: Optional[int] = None,
    ) -> np.ndarray:
        """Return full vocab logit array with exact scores for top-k candidates.

        Non-candidate positions are set to -inf (effectively masked out).

        Args:
            hidden_state:  1-D float array of shape (d_model,).
            vocab_embed:   Full embedding matrix (vocab_size, d_model).
            last_token_id: If set, always include this token as a candidate.

        Returns:
            np.ndarray of shape (vocab_size,): exact logits for top-k,
            -np.inf elsewhere.
        """
        vocab_size = vocab_embed.shape[0]
        candidates = self.select_candidates(hidden_state, last_token_id)

        logits = np.full(vocab_size, -np.inf, dtype=np.float32)
        # Exact dot products for candidate tokens only
        h = hidden_state.astype(np.float32)
        logits[candidates] = vocab_embed[candidates].astype(np.float32) @ h

        self.stats.filter_calls += 1
        self.stats.total_vocab_size += vocab_size
        self.stats.total_candidates_exact += len(candidates)
        return logits

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def update_sketch(self, vocab_embed: np.ndarray) -> None:
        """Recompute the sketch from an updated embedding matrix (e.g., after
        fine-tuning).  Thread-unsafe; call only when the model is not serving.
        """
        cfg = self.config
        d_model = vocab_embed.shape[1]
        rng = np.random.default_rng(cfg.seed)
        R = rng.standard_normal((d_model, cfg.sketch_dim)).astype(np.float32)
        R /= math.sqrt(cfg.sketch_dim)
        self._sketch = (vocab_embed.astype(np.float32) @ R)
        self._projection_T = R.T.astype(np.float32)

    def reset_stats(self) -> None:
        self.stats = LogitFilterStats()

    @property
    def vocab_size(self) -> int:
        return self._sketch.shape[0]

    @property
    def sketch_dim(self) -> int:
        return self._sketch.shape[1]

    def __repr__(self) -> str:
        cfg = self.config
        return (
            f"LogitFilter(vocab={self.vocab_size}, sketch_dim={self.sketch_dim}, "
            f"top_k={cfg.top_k}, "
            f"compression={self.stats.mean_compression_ratio:.3f})"
        )
