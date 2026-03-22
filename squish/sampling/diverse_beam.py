"""squish/sampling/diverse_beam.py

DiverseBeamSampler — Diverse Beam Search decoder.

Divides the beam into ``n_groups`` groups.  Within each group, standard
beam search is applied.  Between groups, an inter-group diversity penalty
discourages token overlap, encouraging semantically distinct hypotheses.

The diversity penalty for group g at step t:

    score(y_t | g) = log_prob(y_t) − λ × count(y_t ∈ beam_groups < g at step t)

This produces G × B/G beams that cover diverse hypotheses, valuable for:
* Code-generation candidate re-ranking
* RAG multi-answer hypothesis sets
* Multi-hypothesis planning outputs

Reference
---------
Vijayakumar et al. "Diverse Beam Search: Decoding Diverse Solutions from
Neural Sequence Models." AAAI 2018. arXiv:1610.02424, 2016.
"""

from __future__ import annotations

__all__ = ["DiverseBeamConfig", "DiverseBeamState", "DiverseBeamSampler"]

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
from numpy import ndarray


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class DiverseBeamConfig:
    """Configuration for DiverseBeamSampler.

    Parameters
    ----------
    beam_size:
        Total number of beams.  Must be divisible by ``n_groups``.
    n_groups:
        Number of diversity groups.
    diversity_strength:
        Penalty coefficient λ applied per token already chosen by
        a preceding group at the same step.
    vocab_size:
        Vocabulary size (needed to initialise beam state).
    max_length:
        Maximum sequence length to generate.
    seed:
        RNG seed (used only for tie-breaking).
    """

    beam_size: int = 4
    n_groups: int = 2
    diversity_strength: float = 0.5
    vocab_size: int = 32000
    max_length: int = 64
    seed: int = 0

    def __post_init__(self) -> None:
        if self.beam_size < 1:
            raise ValueError("beam_size must be >= 1")
        if self.n_groups < 1:
            raise ValueError("n_groups must be >= 1")
        if self.beam_size % self.n_groups != 0:
            raise ValueError("beam_size must be divisible by n_groups")
        if self.vocab_size < 1:
            raise ValueError("vocab_size must be >= 1")
        if self.max_length < 1:
            raise ValueError("max_length must be >= 1")


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

@dataclass
class DiverseBeamState:
    """Mutable state for DiverseBeamSampler.

    Attributes
    ----------
    beam_sequences:
        Current hypotheses per group, shape
        ``(n_groups, beams_per_group, current_length)``.
    beam_scores:
        Cumulative log-probability per hypothesis,
        shape ``(n_groups, beams_per_group)``.
    step:
        Current decoding step.
    is_done:
        Per-hypothesis completion flag,
        shape ``(n_groups, beams_per_group)``.
    """

    beam_sequences: List[List[List[int]]]
    beam_scores: ndarray
    step: int = 0
    is_done: ndarray = field(default_factory=lambda: np.zeros((1, 1), dtype=bool))


# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------

class DiverseBeamSampler:
    """Diverse Beam Search with inter-group diversity penalty.

    Parameters
    ----------
    config:
        ``DiverseBeamConfig`` instance.
    """

    def __init__(self, config: DiverseBeamConfig) -> None:
        self.config = config
        self._beams_per_group = config.beam_size // config.n_groups

    def new_state(self, initial_token: int = 0) -> DiverseBeamState:
        """Create a fresh beam state initialised with ``initial_token``.

        Parameters
        ----------
        initial_token:
            First token in all beam sequences (e.g., BOS token id).
        """
        G = self.config.n_groups
        B = self._beams_per_group
        # All beams start with the same initial token, score 0
        seqs = [[[initial_token] for _ in range(B)] for _ in range(G)]
        scores = np.zeros((G, B), dtype=np.float32)
        is_done = np.zeros((G, B), dtype=bool)
        return DiverseBeamState(
            beam_sequences=seqs, beam_scores=scores,
            step=0, is_done=is_done,
        )

    def step_logits(
        self, group_logits: ndarray, state: DiverseBeamState
    ) -> DiverseBeamState:
        """Perform one diverse beam search step.

        Parameters
        ----------
        group_logits:
            Log-probability (or logit) matrix for each active beam across
            all groups, shape ``(n_groups, beams_per_group, vocab_size)``.
        state:
            Current ``DiverseBeamState``.

        Returns
        -------
        Updated state.
        """
        group_logits = np.asarray(group_logits, dtype=np.float32)
        G = self.config.n_groups
        B = self._beams_per_group
        V = self.config.vocab_size

        if group_logits.shape != (G, B, V):
            raise ValueError(
                f"group_logits must be ({G}, {B}, {V}), got {group_logits.shape}"
            )

        # Log-softmax for numerical stability
        log_probs = group_logits - np.log(
            np.exp(group_logits - group_logits.max(axis=-1, keepdims=True)).sum(
                axis=-1, keepdims=True
            )
        ) - group_logits.max(axis=-1, keepdims=True)

        new_seqs = [list(g) for g in state.beam_sequences]
        new_seqs_copy = [[list(b) for b in g] for g in state.beam_sequences]
        new_scores = state.beam_scores.copy()
        new_done = state.is_done.copy()

        # Track which tokens have been selected by earlier groups this step
        chosen_tokens: List[int] = []

        for g in range(G):
            for b in range(B):
                if new_done[g, b]:
                    continue
                # Base scores = current beam score + log_prob for each token
                token_scores = new_scores[g, b] + log_probs[g, b]  # (V,)
                # Diversity penalty: subtract λ for each token already chosen
                for tok in chosen_tokens:
                    if 0 <= tok < V:
                        token_scores[tok] -= self.config.diversity_strength

                # Greedy-within-beam: pick top token
                best_tok = int(np.argmax(token_scores))
                new_seqs_copy[g][b] = new_seqs_copy[g][b] + [best_tok]
                new_scores[g, b] = float(token_scores[best_tok])
                chosen_tokens.append(best_tok)

        return DiverseBeamState(
            beam_sequences=new_seqs_copy,
            beam_scores=new_scores,
            step=state.step + 1,
            is_done=new_done,
        )

    def get_sequences(
        self, state: DiverseBeamState
    ) -> List[Tuple[List[int], float]]:
        """Return all beam sequences with their scores, sorted by score (desc).

        Returns
        -------
        List of ``(sequence, score)`` tuples from all groups, sorted descending
        by cumulative log-probability score.
        """
        results: List[Tuple[List[int], float]] = []
        for g in range(self.config.n_groups):
            for b in range(self._beams_per_group):
                seq = list(state.beam_sequences[g][b])
                score = float(state.beam_scores[g, b])
                results.append((seq, score))
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def best_sequence(self, state: DiverseBeamState) -> List[int]:
        """Return the highest-scoring beam sequence."""
        return self.get_sequences(state)[0][0]
