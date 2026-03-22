"""squish/speculative/prefix_tree_decode.py

PrefixTreeDecode — Static Prefix-Tree Parallel Draft Decoding.

Reference
---------
Miao et al. "SpecInfer: Accelerating Large Language Model Serving with
Tree-based Speculative Inference and Verification." ASPLOS 2024
(arXiv:2305.09781).

Algorithm
---------
A static *prefix tree* (trie) is built offline from a high-frequency corpus
of token sequences (e.g., common phrases, code patterns).  During decoding:

1. Walk the tree from the current context to find all candidate extensions.
2. Return up to ``max_candidates`` multi-token draft paths.
3. The target model verifies the paths in parallel (tree-attention).

Key properties
--------------
* ``build_from_corpus()`` populates the trie from token-sequence lists.
* ``lookup(context)`` returns candidate token sequences as a sorted list of
  (frequency, token_list) tuples.
* ``decode_step(logits, candidates)`` scores candidates against target logits
  and selects the best accepted prefix.
* NumPy-only; no ML dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "PrefixTreeConfig",
    "PrefixTreeNode",
    "PrefixTreeDecode",
]


@dataclass
class PrefixTreeConfig:
    """Configuration for :class:`PrefixTreeDecode`.

    Attributes:
        max_depth: Maximum trie depth to index.
        max_candidates: Maximum number of candidate paths returned per step.
        min_frequency: Minimum corpus frequency to include a node.
        accept_threshold: Minimum acceptance probability for a draft token.
    """

    max_depth: int = 8
    max_candidates: int = 16
    min_frequency: int = 2
    accept_threshold: float = 0.1


class PrefixTreeNode:
    """Node in the prefix tree.

    Attributes:
        token_id: Token stored at this node (or -1 for root).
        frequency: Number of times this path appears in the corpus.
        children: Mapping from next-token-id to child node.
    """

    __slots__ = ("token_id", "frequency", "children")

    def __init__(self, token_id: int = -1) -> None:
        self.token_id: int = token_id
        self.frequency: int = 0
        self.children: Dict[int, "PrefixTreeNode"] = {}


class PrefixTreeDecode:
    """Prefix-tree based speculative draft generator.

    Parameters
    ----------
    config:
        PrefixTreeDecode configuration.
    """

    def __init__(self, config: Optional[PrefixTreeConfig] = None) -> None:
        self._cfg = config or PrefixTreeConfig()
        self._root = PrefixTreeNode()

    @property
    def config(self) -> PrefixTreeConfig:
        return self._cfg

    # ------------------------------------------------------------------
    # Corpus indexing
    # ------------------------------------------------------------------

    def build_from_corpus(self, token_sequences: Sequence[Sequence[int]]) -> None:
        """Populate the prefix tree from a list of token sequences.

        Parameters
        ----------
        token_sequences:
            Each inner sequence is an ordered list of token IDs representing
            a common phrase or code snippet.
        """
        for seq in token_sequences:
            node = self._root
            for depth, tok in enumerate(seq):
                if depth >= self._cfg.max_depth:
                    break
                if tok not in node.children:
                    node.children[tok] = PrefixTreeNode(tok)
                node = node.children[tok]
                node.frequency += 1

    def reset(self) -> None:
        """Clear the prefix tree."""
        self._root = PrefixTreeNode()

    # ------------------------------------------------------------------
    # Candidate lookup
    # ------------------------------------------------------------------

    def lookup(
        self, context_tokens: Sequence[int]
    ) -> List[Tuple[int, List[int]]]:
        """Return candidate token extensions for the given context.

        Walks the trie using the *suffix* of ``context_tokens``, then does
        a BFS/DFS to enumerate candidate paths from that position.

        Returns
        -------
        List of (frequency, token_list) sorted by descending frequency,
        capped at ``config.max_candidates``.
        """
        node = self._walk(context_tokens)
        if node is None:
            return []
        candidates: List[Tuple[int, List[int]]] = []
        self._enumerate(node, [], candidates)
        # Sort by frequency descending
        candidates.sort(key=lambda x: -x[0])
        return candidates[: self._cfg.max_candidates]

    def _walk(self, context_tokens: Sequence[int]) -> Optional[PrefixTreeNode]:
        """Return the deepest node reachable from context suffix."""
        node = self._root
        for tok in context_tokens:
            if tok in node.children:
                node = node.children[tok]
            else:
                # Try starting over from root with this token
                if tok in self._root.children:
                    node = self._root.children[tok]
                else:
                    node = self._root
        return node if node is not self._root else None

    def _enumerate(
        self,
        node: PrefixTreeNode,
        path: List[int],
        results: List[Tuple[int, List[int]]],
    ) -> None:
        """DFS to collect all paths with frequency >= min_frequency."""
        if len(results) >= self._cfg.max_candidates * 4:
            return
        for child_tok, child in node.children.items():
            if child.frequency < self._cfg.min_frequency:
                continue
            new_path = path + [child_tok]
            results.append((child.frequency, new_path))
            self._enumerate(child, new_path, results)

    # ------------------------------------------------------------------
    # Decode step
    # ------------------------------------------------------------------

    def decode_step(
        self,
        target_logits: np.ndarray,
        candidates: List[Tuple[int, List[int]]],
    ) -> Tuple[int, List[int]]:
        """Score candidates against target logits and return best accepted prefix.

        Parameters
        ----------
        target_logits:
            Target model logits for next token, shape ``(vocab_size,)``.
        candidates:
            List of (frequency, token_list) from :meth:`lookup`.

        Returns
        -------
        (accepted_length, accepted_tokens)
        """
        if not candidates:
            # Fall back to greedy target token
            greedy = int(np.argmax(target_logits))
            return 1, [greedy]

        log_probs = target_logits - _logsumexp(target_logits)
        best_len = 0
        best_path: List[int] = []

        for _freq, token_list in candidates:
            accepted: List[int] = []
            for tok in token_list:
                prob = float(np.exp(log_probs[tok]))
                if prob >= self._cfg.accept_threshold:
                    accepted.append(tok)
                else:
                    break
            if len(accepted) > best_len:
                best_len = len(accepted)
                best_path = accepted

        if not best_path:
            greedy = int(np.argmax(target_logits))
            return 1, [greedy]

        return best_len, best_path


def _logsumexp(x: np.ndarray) -> float:
    m = x.max()
    return float(m + np.log(np.sum(np.exp(x - m))))
