"""squish/speculative/eagle2_spec.py

EAGLE-2 — Context-Aware Dynamic Draft Tree for Speculative Decoding
(Li et al., ICML 2025 / arXiv:2406.16858).

Reference
---------
"EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees."
Li et al., ICML 2025 (arXiv:2406.16858).

Algorithm
---------
EAGLE-2 extends EAGLE with a context-aware acceptance-probability estimator:

1. A secondary scoring network estimates P(accept | token_id, context).
2. Draft tree nodes with estimated acceptance < threshold are pruned before
   verification by the full model.
3. The remaining tree is verified in a single forward pass of the target model.
4. Accepted tokens follow the highest-acceptance path; rejected subtrees are
   discarded and a residual-distribution fallback token is sampled.

This simulation:
* Uses a simple dot-product scoring function as the acceptance estimator.
* Builds a fixed-width/depth tree expanded greedily from draft probabilities.
* Prunes nodes whose estimated acceptance falls below ``prune_threshold``.
* Performs acceptance-rejection sampling along the surviving tree.

Key properties
--------------
* NumPy-only simulation; no GPU dependency.
* ``draft_length`` — maximum draft tokens per step (tree nodes).
* ``beam_width``   — branching factor at each depth.
* ``max_depth``    — maximum tree depth.
* ``prune_threshold`` — minimum estimated acceptance to retain a node.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np

__all__ = [
    "EAGLE2Config",
    "EAGLE2DraftResult",
    "EAGLE2Spec",
]

# ── Config ────────────────────────────────────────────────────────────────────


@dataclass
class EAGLE2Config:
    """Configuration for :class:`EAGLE2Spec`.

    Attributes:
        draft_length: Maximum draft tokens (tree budget).
        beam_width: Number of children per tree node.
        max_depth: Maximum tree depth.
        prune_threshold: Estimated acceptance score below which nodes are pruned.
        temperature: Sampling temperature for draft distribution.
        seed: Random seed for reproducibility.
    """

    draft_length: int = 16
    beam_width: int = 4
    max_depth: int = 6
    prune_threshold: float = 0.05
    temperature: float = 1.0
    seed: int = 0

    def __post_init__(self) -> None:
        if self.draft_length < 1:
            raise ValueError(f"draft_length must be ≥ 1; got {self.draft_length}")
        if self.beam_width < 1:
            raise ValueError(f"beam_width must be ≥ 1; got {self.beam_width}")
        if self.max_depth < 1:
            raise ValueError(f"max_depth must be ≥ 1; got {self.max_depth}")
        if not (0.0 <= self.prune_threshold < 1.0):
            raise ValueError(
                f"prune_threshold must be in [0, 1); got {self.prune_threshold}"
            )
        if self.temperature <= 0.0:
            raise ValueError(f"temperature must be > 0; got {self.temperature}")


# ── Result ────────────────────────────────────────────────────────────────────


@dataclass
class EAGLE2DraftResult:
    """Outcome of one :meth:`EAGLE2Spec.step` call.

    Attributes:
        accepted_tokens: Accepted token IDs (list of ints).
        n_accepted: Number of accepted tokens.
        n_drafted: Number of tree nodes evaluated.
        n_pruned: Number of tree nodes pruned before verification.
        acceptance_rate: n_accepted / max(n_drafted, 1).
    """

    accepted_tokens: List[int]
    n_accepted: int
    n_drafted: int
    n_pruned: int
    acceptance_rate: float


# ── Spec drafter ──────────────────────────────────────────────────────────────


class EAGLE2Spec:
    """EAGLE-2 context-aware dynamic draft tree speculative decoder.

    Example::

        cfg = EAGLE2Config(draft_length=12, beam_width=3, max_depth=4)
        spec = EAGLE2Spec(cfg)

        def draft_fn(token, context):
            logits = model.draft_forward(token, context)
            return softmax(logits)

        def score_fn(token, context):
            # returns scalar estimated acceptance probability in [0, 1]
            return model.score(token, context)

        def target_fn(token, context):
            logits = model.target_forward(token, context)
            return softmax(logits)

        result = spec.step([1, 2, 3], draft_fn, score_fn, target_fn)
    """

    def __init__(self, config: Optional[EAGLE2Config] = None, seed: Optional[int] = None) -> None:
        self.config = config or EAGLE2Config()
        _seed = seed if seed is not None else self.config.seed
        self._rng = np.random.default_rng(_seed)
        self._total_accepted: float = 0.0
        self._total_drafted: int = 0
        self._n_steps: int = 0

    # ── Public API ────────────────────────────────────────────────────────────

    def step(
        self,
        context_ids: List[int],
        draft_fn: Callable[[int, List[int]], np.ndarray],
        score_fn: Callable[[int, List[int]], float],
        target_fn: Callable[[int, List[int]], np.ndarray],
    ) -> EAGLE2DraftResult:
        """Run one EAGLE-2 speculative decoding step.

        Args:
            context_ids: Current context token IDs.
            draft_fn: ``(last_token, context) -> probs (vocab_size,)``
            score_fn: ``(candidate_token, context) -> acceptance_score in [0,1]``
            target_fn: ``(last_token, context) -> probs (vocab_size,)``

        Returns:
            :class:`EAGLE2DraftResult` with accepted tokens and statistics.
        """
        ctx = list(context_ids)
        last = ctx[-1] if ctx else 0

        # ── Phase 1: Build draft tree via BFS ─────────────────────────────────
        # Each node: (token_id, context, depth, draft_prob, parent_idx)
        nodes: List[Tuple[int, List[int], int, float, int]] = [(last, ctx, 0, 1.0, -1)]
        frontier: List[int] = [0]
        n_pruned = 0
        n_drafted = 0

        while frontier and n_drafted < self.config.draft_length:
            next_frontier: List[int] = []
            for parent_idx in frontier:
                if n_drafted >= self.config.draft_length:
                    break
                parent_token, parent_ctx, depth, _, _ = nodes[parent_idx]
                if depth >= self.config.max_depth:
                    continue
                probs = np.asarray(draft_fn(parent_token, parent_ctx), dtype=np.float32)
                probs = self._softmax(probs / self.config.temperature)
                top_k = min(self.config.beam_width, len(probs))
                top_tokens = np.argsort(probs)[::-1][:top_k]

                for tid in top_tokens:
                    if n_drafted >= self.config.draft_length:
                        break
                    score = float(score_fn(int(tid), parent_ctx))
                    # Prune low-acceptance nodes
                    if score < self.config.prune_threshold:
                        n_pruned += 1
                        continue
                    new_ctx = parent_ctx + [int(tid)]
                    node_idx = len(nodes)
                    nodes.append((int(tid), new_ctx, depth + 1, probs[tid], parent_idx))
                    next_frontier.append(node_idx)
                    n_drafted += 1
            frontier = next_frontier

        # ── Phase 2: Acceptance-rejection walk along tree ─────────────────────
        # Walk from root following the highest draft probability
        accepted: List[int] = []
        cur_ctx = ctx
        cur_token = last
        cur_depth = 0
        cur_parent = 0

        while True:
            # Find children of current node
            children = [
                i for i, n in enumerate(nodes)
                if i > 0 and n[4] == cur_parent and n[2] == cur_depth + 1
            ]
            if not children:
                # Bonus token from target
                target_probs = np.asarray(target_fn(cur_token, cur_ctx), dtype=np.float32)
                target_probs = self._softmax(target_probs / self.config.temperature)
                bonus = int(self._rng.choice(len(target_probs), p=target_probs))
                accepted.append(bonus)
                break

            # Pick highest-probability child
            best_idx = max(children, key=lambda i: nodes[i][3])
            draft_token, new_ctx, new_depth, draft_prob, _ = nodes[best_idx]

            # Target verification
            target_probs = np.asarray(target_fn(cur_token, cur_ctx), dtype=np.float32)
            target_probs = self._softmax(target_probs / self.config.temperature)
            draft_probs_vec = np.asarray(draft_fn(cur_token, cur_ctx), dtype=np.float32)
            draft_probs_vec = self._softmax(draft_probs_vec / self.config.temperature)

            t_p = float(target_probs[draft_token])
            d_p = float(draft_probs_vec[draft_token]) + 1e-9
            accept_prob = min(1.0, t_p / d_p)

            if self._rng.random() < accept_prob:
                accepted.append(draft_token)
                cur_token = draft_token
                cur_ctx = new_ctx
                cur_depth = new_depth
                cur_parent = best_idx
            else:
                # Residual resampling
                residual = np.maximum(target_probs - draft_probs_vec, 0.0)
                s = residual.sum()
                if s > 1e-9:
                    residual /= s
                    fallback = int(self._rng.choice(len(residual), p=residual))
                else:
                    fallback = int(self._rng.choice(len(target_probs), p=target_probs))
                accepted.append(fallback)
                break

        n_acc = len(accepted)
        self._total_accepted += n_acc
        self._total_drafted += max(n_drafted, 1)
        self._n_steps += 1

        return EAGLE2DraftResult(
            accepted_tokens=accepted,
            n_accepted=n_acc,
            n_drafted=n_drafted,
            n_pruned=n_pruned,
            acceptance_rate=n_acc / max(n_drafted, 1),
        )

    @property
    def mean_acceptance_rate(self) -> float:
        """Mean acceptance rate (tokens / drafted) over all steps."""
        return self._total_accepted / max(self._total_drafted, 1)

    def reset_stats(self) -> None:
        """Reset accumulated acceptance statistics."""
        self._total_accepted = 0.0
        self._total_drafted = 0
        self._n_steps = 0

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        e = np.exp(x - x.max())
        return e / (e.sum() + 1e-9)

    def __repr__(self) -> str:
        cfg = self.config
        return (
            f"EAGLE2Spec(draft_length={cfg.draft_length}, "
            f"beam_width={cfg.beam_width}, "
            f"prune_threshold={cfg.prune_threshold}, "
            f"mean_acceptance_rate={self.mean_acceptance_rate:.3f})"
        )
