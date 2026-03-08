"""
squish/rasd.py

RASD — Retrieval-Augmented Speculative Decoding.

Based on:
  "RASD: Retrieval Augmented Speculative Decoding"
  arXiv:2504.00942

Problem
-------
Standard speculative decoding drafters (N-gram, EAGLE-3) either memorise the
prompt or use a trained draft model.  Both approaches miss a large reservoir
of high-quality token continuations that already exist in the model's
knowledge corpus (e.g., Squish's Cortex RAG graph).

RASD mines Squish's Cortex knowledge graph for token sequences whose prefix
matches the current decoding context.  Retrieved sequences are combined with
the generative draft tree via *longest-prefix merge* (LPM), producing a
richer candidate tree that the target model verifies in a single forward pass.

Method
------
1. **CorpusIndex** — maintains a prefix-indexed store of token sequences.
   - ``add_sequence(tokens)`` indexes every prefix of the sequence.
   - ``search(prefix, top_k)`` returns the ``top_k`` sequences whose stored
     prefix best matches the query, ranked by prefix length.

2. **DraftTree** — a minimal n-ary tree over candidate token sequences.
   - Leaf nodes hold draft token IDs.
   - ``add_path(tokens)`` merges a token sequence into the tree.
   - ``all_paths()`` yields all root-to-leaf paths.

3. **RASDBatcher** — orchestrates retrieval + merge:
   - ``prune_tree(tree, draft_probs, beam_width)`` prunes the generative draft
      tree to ``beam_width`` branches using per-token draft probabilities.
   - ``fuse_trees(gen_tree, ret_tree)`` merges generation and retrieval trees
     via shared-prefix deduplication.
   - ``build_draft(context_tokens, corpus, gen_tree, beam_width)`` full pipeline.

Conflict-resolved notes (Master Conflict Report)
-------------------------------------------------
- **Cascade order**: CopySpec → N-gram → RASD → EAGLE-3.  RASD only runs when
  CopySpec and N-gram both fail to produce qualifying drafts.
- **Synergy with EAGLE-3**: RASD's retrieved tree augments (not replaces)
  EAGLE-3's generative tree.  LPM fusion keeps the combined tree bounded.
- **Independence**: RASD reads Cortex but does not write to it; no conflict
  with KV management, quantization, or attention kernels.

Provides
--------
  RASDConfig       — retrieval and tree-pruning parameters.
  CorpusIndex      — prefix-indexed token sequence store.
  DraftTree        — n-ary tree for multi-path draft candidates.
  RASDBatcher      — orchestrates retrieval + tree fusion.
  RASDStats        — counters for corpus hit rate and tree utilisation.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

__all__ = [
    "RASDConfig",
    "CorpusIndex",
    "DraftTree",
    "RASDBatcher",
    "RASDStats",
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class RASDConfig:
    """Parameters controlling RASD retrieval and fusion.

    Parameters
    ----------
    beam_width:
        Maximum number of generative draft branches to retain after pruning.
    max_retrieval_candidates:
        Maximum number of corpus sequences returned per search.
    min_prefix_len:
        Minimum prefix length required for a corpus hit to be used.
    max_tree_depth:
        Maximum depth of the merged draft tree (limits verification cost).
    max_corpus_sequences:
        Maximum number of sequences stored in the corpus index.
    """

    beam_width: int = 4
    max_retrieval_candidates: int = 8
    min_prefix_len: int = 2
    max_tree_depth: int = 6
    max_corpus_sequences: int = 50_000


# ---------------------------------------------------------------------------
# CorpusIndex
# ---------------------------------------------------------------------------

class CorpusIndex:
    """Prefix-indexed store of token sequences.

    Sequences are indexed by every prefix of length >= ``min_prefix_len``.
    Lookup returns sequences whose stored prefix best extends the query.

    Parameters
    ----------
    min_prefix_len:
        Minimum prefix length indexed; shorter sequences are ignored.
    max_sequences:
        Total capacity.  Old sequences are discarded (FIFO) when exceeded.
    """

    def __init__(
        self,
        min_prefix_len: int = 2,
        max_sequences: int = 50_000,
    ) -> None:
        self._min_prefix_len = min_prefix_len
        self._max_sequences = max_sequences
        # prefix_tup → list of full sequences
        self._index: Dict[tuple, List[List[int]]] = defaultdict(list)
        self._insertion_order: List[tuple] = []  # tracks first-insertion prefix keys
        self._total: int = 0

    @property
    def n_sequences(self) -> int:
        """Total sequences currently stored."""
        return self._total

    def add_sequence(self, tokens: Sequence[int]) -> None:
        """Index a token sequence by each of its prefixes.

        Parameters
        ----------
        tokens:
            A token sequence from the corpus (e.g. a retrieved document chunk).
            Must contain at least ``min_prefix_len`` tokens to be indexed.
        """
        tokens = list(tokens)
        if len(tokens) < self._min_prefix_len:
            return

        if self._total >= self._max_sequences:
            self._evict_oldest()

        # Index every valid prefix
        for prefix_len in range(self._min_prefix_len, len(tokens) + 1):
            key = tuple(tokens[:prefix_len])
            if key not in self._index or not self._index[key]:
                self._insertion_order.append(key)
            self._index[key].append(tokens)

        self._total += 1

    def _evict_oldest(self) -> None:
        """Remove the oldest index key to bound memory usage."""
        while self._insertion_order:
            key = self._insertion_order.pop(0)
            if key in self._index and self._index[key]:
                self._index[key].pop(0)
                if not self._index[key]:
                    del self._index[key]
                self._total = max(0, self._total - 1)
                return

    def search(
        self, prefix: Sequence[int], top_k: int = 8
    ) -> List[List[int]]:
        """Return up to ``top_k`` sequences whose stored prefix extends *prefix*.

        Searches from longest possible prefix down to ``min_prefix_len``,
        returning the first hit level's sequences.

        Parameters
        ----------
        prefix:
            Context tokens used as the lookup key.
        top_k:
            Maximum number of sequences to return.

        Returns
        -------
        List of token sequences (each a ``list[int]``).  Empty if no hit.
        """
        prefix = list(prefix)
        for end in range(len(prefix), self._min_prefix_len - 1, -1):
            key = tuple(prefix[:end])
            if key in self._index and self._index[key]:
                return self._index[key][:top_k]
        return []

    def clear(self) -> None:
        """Remove all indexed sequences."""
        self._index.clear()
        self._insertion_order.clear()
        self._total = 0


# ---------------------------------------------------------------------------
# DraftTree
# ---------------------------------------------------------------------------

class _DraftNode:
    """A single node in the draft tree."""

    __slots__ = ("token_id", "children", "prob")

    def __init__(self, token_id: int, prob: float = 1.0) -> None:
        self.token_id: int = token_id
        self.children: Dict[int, "_DraftNode"] = {}
        self.prob: float = prob


class DraftTree:
    """N-ary tree of draft token candidates.

    The root is a virtual node (token_id = -1) representing the current
    decoding position.  Each path from root to leaf is a draft sequence.

    Parameters
    ----------
    max_depth:
        Maximum number of token hops from root to leaf.
    """

    def __init__(self, max_depth: int = 6) -> None:
        self._root = _DraftNode(-1)
        self._max_depth = max_depth

    @property
    def root(self) -> _DraftNode:
        return self._root

    def add_path(
        self,
        tokens: Sequence[int],
        probs: Optional[Sequence[float]] = None,
    ) -> None:
        """Insert a token path into the tree.

        Parameters
        ----------
        tokens:
            Ordered sequence of draft token IDs (root → leaf direction).
        probs:
            Optional per-token probabilities.  Defaults to 1.0 for each.
        """
        if not tokens:
            return
        n = min(len(tokens), self._max_depth)
        node = self._root
        for i in range(n):
            tid = tokens[i]
            prob = probs[i] if probs else 1.0
            if tid not in node.children:
                node.children[tid] = _DraftNode(tid, prob)
            node = node.children[tid]

    def all_paths(self) -> List[List[int]]:
        """Return all root-to-leaf paths as lists of token IDs."""
        paths: List[List[int]] = []
        self._dfs(self._root, [], paths)
        return paths

    def _dfs(
        self,
        node: _DraftNode,
        path: List[int],
        out: List[List[int]],
    ) -> None:
        if not node.children:
            if path:
                out.append(list(path))
            return
        for child in node.children.values():
            path.append(child.token_id)
            self._dfs(child, path, out)
            path.pop()

    def n_nodes(self) -> int:
        """Count total nodes (excluding virtual root)."""
        return self._count(self._root)

    def _count(self, node: _DraftNode) -> int:
        return sum(1 + self._count(c) for c in node.children.values())

    def is_empty(self) -> bool:
        return len(self._root.children) == 0


# ---------------------------------------------------------------------------
# RASDBatcher
# ---------------------------------------------------------------------------

class RASDBatcher:
    """Orchestrates RASD retrieval and draft-tree fusion.

    Parameters
    ----------
    config:
        RASD configuration parameters.
    """

    def __init__(self, config: Optional[RASDConfig] = None) -> None:
        self._config = config or RASDConfig()

    def prune_tree(
        self,
        tree: DraftTree,
        draft_probs: Optional[Dict[int, float]] = None,
        beam_width: Optional[int] = None,
    ) -> DraftTree:
        """Prune *tree* to at most *beam_width* branches.

        Children at each node are ranked by ``draft_probs`` (if provided) or
        by their original insertion order, keeping only the top ``beam_width``
        at each level.

        Parameters
        ----------
        tree:
            The generative draft tree (e.g. from EAGLE-3).
        draft_probs:
            Optional token → probability mapping from the draft model.
        beam_width:
            Override for ``config.beam_width``.

        Returns
        -------
        A new pruned ``DraftTree``.
        """
        bw = beam_width if beam_width is not None else self._config.beam_width
        result = DraftTree(max_depth=self._config.max_tree_depth)
        self._prune_node(tree.root, result.root, draft_probs or {}, bw, depth=0)
        return result

    def _prune_node(
        self,
        src: _DraftNode,
        dst: _DraftNode,
        probs: Dict[int, float],
        beam_width: int,
        depth: int,
    ) -> None:
        if depth >= self._config.max_tree_depth:
            return
        children = list(src.children.values())
        # Sort by prob descending (higher prob first)
        children.sort(key=lambda c: probs.get(c.token_id, c.prob), reverse=True)
        for child in children[:beam_width]:
            new_node = _DraftNode(child.token_id, child.prob)
            dst.children[child.token_id] = new_node
            self._prune_node(child, new_node, probs, beam_width, depth + 1)

    def fuse_trees(self, gen_tree: DraftTree, ret_tree: DraftTree) -> DraftTree:
        """Merge generation and retrieval trees via shared-prefix deduplication.

        Paths in *ret_tree* are added to *gen_tree* (mutating a copy) only if
        they extend the tree without duplicating existing nodes.

        Parameters
        ----------
        gen_tree:
            Generative draft tree (EAGLE-3 or pruned result).
        ret_tree:
            Retrieval draft tree (built from corpus candidates).

        Returns
        -------
        A new fused ``DraftTree``.
        """
        fused = DraftTree(max_depth=self._config.max_tree_depth)
        # First copy gen_tree paths
        for path in gen_tree.all_paths():
            fused.add_path(path)
        # Then add retrieval paths (shared prefixes automatically merged)
        for path in ret_tree.all_paths():
            fused.add_path(path)
        return fused

    def build_retrieval_tree(
        self,
        context_tokens: Sequence[int],
        corpus: CorpusIndex,
    ) -> DraftTree:
        """Retrieve candidate sequences and build a retrieval ``DraftTree``.

        Parameters
        ----------
        context_tokens:
            Current decoding context.  The last ``min_prefix_len`` .. ``N``
            tokens are used as the search prefix.
        corpus:
            The indexed corpus to search.

        Returns
        -------
        A ``DraftTree`` containing all retrieved candidates (possibly empty).
        """
        cfg = self._config
        prefix = list(context_tokens)[-cfg.min_prefix_len * 4:]  # generous window
        candidates = corpus.search(prefix, top_k=cfg.max_retrieval_candidates)
        tree = DraftTree(max_depth=cfg.max_tree_depth)
        plen = len(prefix)
        for seq in candidates:
            # Only the continuation past the prefix is the draft
            if len(seq) <= plen:
                continue
            tree.add_path(seq[plen:])
        return tree


# ---------------------------------------------------------------------------
# RASDStats
# ---------------------------------------------------------------------------

@dataclass
class RASDStats:
    """Counters for RASD retrieval quality and tree utilisation.

    Attributes
    ----------
    retrieval_attempts:
        Total corpus search calls.
    retrieval_hits:
        Searches that returned at least one candidate.
    total_retrieved_tokens:
        Sum of tokens across all retrieved candidate sequences.
    fused_tree_nodes:
        Total nodes added to fused trees across all decode steps.
    pruned_nodes:
        Nodes removed from generative trees by ``prune_tree``.
    """

    retrieval_attempts: int = 0
    retrieval_hits: int = 0
    total_retrieved_tokens: int = 0
    fused_tree_nodes: int = 0
    pruned_nodes: int = 0

    @property
    def retrieval_hit_rate(self) -> float:
        return (
            self.retrieval_hits / self.retrieval_attempts
            if self.retrieval_attempts else 0.0
        )

    @property
    def avg_retrieved_tokens(self) -> float:
        return (
            self.total_retrieved_tokens / self.retrieval_hits
            if self.retrieval_hits else 0.0
        )

    def record_retrieval(self, n_candidates: int, n_tokens: int) -> None:
        self.retrieval_attempts += 1
        if n_candidates > 0:
            self.retrieval_hits += 1
            self.total_retrieved_tokens += n_tokens

    def record_fusion(self, fused_nodes: int, pruned: int) -> None:
        self.fused_tree_nodes += fused_nodes
        self.pruned_nodes += pruned

    def reset(self) -> None:
        self.retrieval_attempts = 0
        self.retrieval_hits = 0
        self.total_retrieved_tokens = 0
        self.fused_tree_nodes = 0
        self.pruned_nodes = 0
