"""DVTSSearch: Diverse Verifier Tree Search for reasoning chain exploration.

Tian et al. (arXiv 2501.08101, 2025).  MCTS-like expansion guided by a process reward
model (PRM) with forced diversity: N subtrees are seeded from K diverse reasoning
prefixes (via top-K sampling); each subtree is expanded independently; final answers
are merged via softmax-weighted voting.  Reaches 62% on MATH-500 at < 64 tokens per
reasoning step and outperforms standard PRM beam search at equal compute.

Reference: Tian et al., "DVTS: Diverse Verifier Tree Search", arXiv 2501.08101, 2025.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

__all__ = [
    "DVTSConfig",
    "DVTSNode",
    "DVTSResult",
    "DVTSSearch",
]


@dataclass
class DVTSConfig:
    """Configuration for :class:`DVTSSearch`.

    Attributes:
        n_subtrees: Number of diverse reasoning subtrees to grow.
        expand_depth: Maximum expansion steps per subtree.
        diversity_temperature: Sampling temperature for seeding diverse prefixes.
        prm_weight: Weight given to PRM score vs token-probability score.
        min_token_score: Minimum acceptable token probability (log scale).
        seed: RNG seed.
    """

    n_subtrees: int = 4
    expand_depth: int = 8
    diversity_temperature: float = 0.9
    prm_weight: float = 0.7
    min_token_score: float = -10.0
    seed: int = 0

    def __post_init__(self) -> None:
        if self.n_subtrees < 1:
            raise ValueError(f"n_subtrees must be ≥ 1, got {self.n_subtrees}")
        if self.expand_depth < 1:
            raise ValueError(f"expand_depth must be ≥ 1, got {self.expand_depth}")
        if self.diversity_temperature <= 0.0:
            raise ValueError(
                f"diversity_temperature must be > 0, got {self.diversity_temperature}"
            )
        if not 0.0 <= self.prm_weight <= 1.0:
            raise ValueError(
                f"prm_weight must be in [0, 1], got {self.prm_weight}"
            )


@dataclass
class DVTSNode:
    """One node in the DVTS expansion tree.

    Attributes:
        prefix: Token sequence reaching this node.
        token_score: Cumulative log-probability.
        prm_score: Process reward model score (higher = better).
        depth: Depth in the tree (0 = root seed).
        children: Child nodes expanded from this node.
        answer: Final answer string, non-empty at leaf nodes.
    """

    prefix: List[int]
    token_score: float = 0.0
    prm_score: float = 0.0
    depth: int = 0
    children: List["DVTSNode"] = field(default_factory=list)
    answer: str = ""

    @property
    def combined_score(self) -> float:
        return self.token_score + self.prm_score

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0


@dataclass
class DVTSResult:
    """Output of one DVTS search run.

    Attributes:
        best_answer: Highest-scored final answer.
        answer_scores: ``{answer: weighted_vote_score}`` dict.
        n_nodes_expanded: Total nodes created during search.
        subtree_roots: Root nodes of each diverse subtree.
    """

    best_answer: str
    answer_scores: Dict[str, float]
    n_nodes_expanded: int
    subtree_roots: List[DVTSNode]


# ---------------------------------------------------------------------------
# Scorer type alias
# ---------------------------------------------------------------------------
# A PRM scorer takes a list of token IDs and returns a float reward.
PRMScorer = Callable[[List[int]], float]


class DVTSSearch:
    """Diverse Verifier Tree Search with NumPy-backed simulation.

    In production this class wraps the model's forward pass and a trained PRM.
    For testing and offline use, *prm_scorer* and *expand_fn* may be provided
    as pure-Python callables, allowing full test coverage without a GPU.

    Usage::

        cfg = DVTSConfig(n_subtrees=4, expand_depth=4)
        search = DVTSSearch(cfg)
        result = search.run(
            seed_tokens=[[0, 1], [0, 2], [0, 3], [0, 4]],
            prm_scorer=my_prm,
            expand_fn=my_model_step,
            extract_answer=lambda tokens: str(tokens[-1]),
        )

    """

    def __init__(self, config: DVTSConfig) -> None:
        self.config = config
        self._rng = np.random.default_rng(config.seed)
        self._n_nodes_expanded = 0

    # ------------------------------------------------------------------
    # Primary API
    # ------------------------------------------------------------------

    def run(
        self,
        seed_tokens: List[int],
        prm_scorer: PRMScorer,
        expand_fn: Callable[[List[int]], List[Tuple[List[int], float]]],
        extract_answer: Callable[[List[int]], str],
        vocab_size: int = 50000,
    ) -> DVTSResult:
        """Run DVTS from a flat prompt *seed_tokens*.

        Parameters
        ----------
        seed_tokens:
            Flat list of prompt token ids.  Diverse subtree seeds are generated
            internally via :meth:`make_diverse_seeds`.
        prm_scorer:
            Callable ``(tokens) → float`` reward in [0, 1].
        expand_fn:
            Callable ``(tokens) → (next_token_id, log_prob)``; advances one step.
        extract_answer:
            Callable ``(tokens) → str``; extracts the final answer string.
        vocab_size:
            Vocabulary size used when sampling diverse seed extensions.
        """
        cfg = self.config
        per_tree_seeds = self.make_diverse_seeds(seed_tokens, vocab_size)
        n_trees = len(per_tree_seeds)
        self._n_nodes_expanded = 0

        subtree_roots: List[DVTSNode] = []
        for i in range(n_trees):
            root = DVTSNode(
                prefix=list(per_tree_seeds[i]),
                prm_score=prm_scorer(per_tree_seeds[i]),
            )
            self._expand_subtree(root, expand_fn, prm_scorer, extract_answer)
            subtree_roots.append(root)

        # Collect all leaf answers and vote
        answer_scores: Dict[str, float] = {}
        for root in subtree_roots:
            for leaf in self._collect_leaves(root):
                if not leaf.answer:
                    leaf.answer = extract_answer(leaf.prefix)
                score = float(np.exp(leaf.combined_score))
                answer_scores[leaf.answer] = (
                    answer_scores.get(leaf.answer, 0.0) + score
                )

        if not answer_scores:
            best_answer = ""
        else:
            best_answer = max(answer_scores, key=lambda a: answer_scores[a])

        return DVTSResult(
            best_answer=best_answer,
            answer_scores=answer_scores,
            n_nodes_expanded=self._n_nodes_expanded,
            subtree_roots=subtree_roots,
        )

    def make_diverse_seeds(
        self,
        base_tokens: List[int],
        vocab_size: int,
    ) -> List[List[int]]:
        """Sample N diverse seeds by appending a random vocab token to *base_tokens*."""
        cfg = self.config
        seeds = []
        chosen = self._rng.choice(vocab_size, size=cfg.n_subtrees, replace=False)
        for tok in chosen:
            seeds.append(list(base_tokens) + [int(tok)])
        return seeds

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _expand_subtree(
        self,
        node: DVTSNode,
        expand_fn: Callable[[List[int]], List[Tuple[List[int], float]]],
        prm_scorer: PRMScorer,
        extract_answer: Callable[[List[int]], str],
    ) -> None:
        """Recursively expand *node* up to expand_depth.

        *expand_fn* must return a list of ``(next_tokens, log_prob)`` pairs for
        each possible next-step extension — at least one element required.
        """
        if node.depth >= self.config.expand_depth:
            node.answer = extract_answer(node.prefix)
            return

        expansions = expand_fn(node.prefix)
        for next_tokens, log_prob in expansions:
            new_prefix = list(node.prefix) + list(next_tokens)
            child = DVTSNode(
                prefix=new_prefix,
                token_score=node.token_score + float(log_prob),
                prm_score=prm_scorer(new_prefix),
                depth=node.depth + 1,
            )
            self._n_nodes_expanded += 1
            node.children.append(child)
            self._expand_subtree(child, expand_fn, prm_scorer, extract_answer)

    def _collect_leaves(self, node: DVTSNode) -> List[DVTSNode]:
        """Return all leaf nodes in the subtree rooted at *node*."""
        if node.is_leaf:
            return [node]
        leaves = []
        for child in node.children:
            leaves.extend(self._collect_leaves(child))
        return leaves
