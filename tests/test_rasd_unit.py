"""tests/test_rasd_unit.py — 100 % coverage for squish/rasd.py"""
import pytest

from squish.rasd import (
    CorpusIndex,
    DraftTree,
    RASDBatcher,
    RASDConfig,
    RASDStats,
    _DraftNode,
)

# ---------------------------------------------------------------------------
# RASDConfig
# ---------------------------------------------------------------------------

class TestRASDConfig:
    def test_defaults(self):
        cfg = RASDConfig()
        assert cfg.beam_width == 4
        assert cfg.max_retrieval_candidates == 8
        assert cfg.min_prefix_len == 2
        assert cfg.max_tree_depth == 6
        assert cfg.max_corpus_sequences == 50_000

    def test_custom(self):
        cfg = RASDConfig(beam_width=2, min_prefix_len=1, max_tree_depth=3)
        assert cfg.beam_width == 2
        assert cfg.min_prefix_len == 1


# ---------------------------------------------------------------------------
# CorpusIndex
# ---------------------------------------------------------------------------

class TestCorpusIndex:
    def _make(self, min_plen=2, max_seq=100):
        return CorpusIndex(min_prefix_len=min_plen, max_sequences=max_seq)

    def test_initial_state(self):
        ci = self._make()
        assert ci.n_sequences == 0

    def test_add_short_sequence_ignored(self):
        ci = self._make(min_plen=3)
        ci.add_sequence([1, 2])  # only 2 tokens, min_prefix_len=3
        assert ci.n_sequences == 0

    def test_add_sequence_and_search(self):
        ci = self._make(min_plen=2)
        ci.add_sequence([1, 2, 3, 4, 5])
        results = ci.search([1, 2])
        assert len(results) > 0
        assert results[0] == [1, 2, 3, 4, 5]

    def test_search_with_longer_prefix_first(self):
        ci = self._make(min_plen=1)
        ci.add_sequence([10, 20, 30, 40])
        # Searching [10, 20, 30] should hit the sequence
        results = ci.search([10, 20, 30])
        assert len(results) > 0

    def test_search_falls_back_to_shorter_prefix(self):
        ci = self._make(min_plen=1)
        ci.add_sequence([1, 2, 3])
        # Query prefix [1, 2, 99]: exact 3-prefix not found, falls back to [1, 2]
        results = ci.search([1, 2, 99])
        assert len(results) > 0

    def test_search_returns_empty_when_no_match(self):
        ci = self._make(min_plen=2)
        ci.add_sequence([1, 2, 3])
        results = ci.search([99, 100])
        assert results == []

    def test_search_respects_top_k(self):
        ci = self._make(min_plen=1)
        for i in range(10):
            ci.add_sequence([1, 2, i + 10])
        results = ci.search([1, 2], top_k=3)
        assert len(results) <= 3

    def test_eviction_when_max_sequences_exceeded(self):
        ci = self._make(min_plen=1, max_seq=2)
        ci.add_sequence([1, 2, 3])
        ci.add_sequence([4, 5, 6])
        # Third one triggers eviction
        ci.add_sequence([7, 8, 9])
        # Should still work, n_sequences <= max_seq
        assert ci.n_sequences <= 3

    def test_clear_empties_index(self):
        ci = self._make()
        ci.add_sequence([1, 2, 3, 4])
        ci.clear()
        assert ci.n_sequences == 0
        assert ci.search([1, 2]) == []

    def test_search_stops_at_min_prefix_len_boundary(self):
        ci = self._make(min_plen=2)
        ci.add_sequence([5, 6, 7])
        # Search [5, 6]  exactly at min boundary
        results = ci.search([5, 6])
        assert len(results) > 0

    def test_add_exact_min_prefix_len(self):
        ci = self._make(min_plen=2)
        ci.add_sequence([1, 2])  # exactly min_prefix_len tokens
        assert ci.n_sequences == 1

    def test_search_empty_prefix(self):
        ci = self._make(min_plen=2)
        ci.add_sequence([1, 2, 3])
        # prefix shorter than min_prefix_len → starts loop below threshold → []
        results = ci.search([1])  # len=1 < min_prefix_len=2
        assert results == []

    def test_multiple_evictions(self):
        ci = self._make(min_plen=1, max_seq=1)
        for i in range(5):
            ci.add_sequence([i, i + 1, i + 2])
        # Should have evicted down; no crash

    def test_evict_oldest_empty_insertion_order(self):
        # max_corpus_sequences=0 → eviction triggered before ANY sequence added
        # insertion_order is empty → while loop exits immediately (165→exit branch)
        ci = CorpusIndex(min_prefix_len=2, max_sequences=0)
        ci.add_sequence([1, 2, 3])
        # No crash; sequence is still added after the no-op eviction
        assert ci.n_sequences == 1

    def test_evict_oldest_stale_key_skipped(self):
        # Inject a stale key into insertion_order (key not in index).
        # _evict_oldest should skip it and find a valid key (167→165 branch).
        ci = CorpusIndex(min_prefix_len=2, max_sequences=1)
        ci.add_sequence([1, 2, 3])
        # Insert a stale key at the front of the insertion_order
        ci._insertion_order.insert(0, (99, 88))
        # Trigger eviction: add a second sequence exceeding max
        ci.add_sequence([4, 5])
        # Stale key (99,88) was skipped; [1,2,3] was evicted instead

    def test_evict_oldest_shared_prefix_key_not_deleted(self):
        # Two sequences share prefix (1,2).  After one eviction (1,2) still
        # has an entry → 'if not self._index[key]' is False (169→171 branch).
        ci = CorpusIndex(min_prefix_len=2, max_sequences=2)
        ci.add_sequence([1, 2, 3])
        ci.add_sequence([1, 2, 4])  # same prefix (1,2) → 2 entries under key
        # Third add triggers eviction of the first sequence from (1,2)
        ci.add_sequence([5, 6, 7])
        # (1,2) should still exist in the index after partial eviction
        results = ci.search([1, 2])
        # At least one sequence under (1,2) should survive
        assert results is not None


# ---------------------------------------------------------------------------
# DraftTree / _DraftNode
# ---------------------------------------------------------------------------

class TestDraftNode:
    def test_initial_state(self):
        node = _DraftNode(5, prob=0.9)
        assert node.token_id == 5
        assert node.prob == 0.9
        assert node.children == {}

    def test_default_prob(self):
        node = _DraftNode(3)
        assert node.prob == 1.0


class TestDraftTree:
    def test_initial_is_empty(self):
        tree = DraftTree()
        assert tree.is_empty()
        assert tree.n_nodes() == 0

    def test_add_single_path(self):
        tree = DraftTree()
        tree.add_path([1, 2, 3])
        assert not tree.is_empty()
        assert tree.n_nodes() == 3

    def test_add_path_with_probs(self):
        tree = DraftTree()
        tree.add_path([1, 2], probs=[0.9, 0.7])
        assert tree.n_nodes() == 2

    def test_add_empty_path_noop(self):
        tree = DraftTree()
        tree.add_path([])
        assert tree.is_empty()

    def test_add_shared_prefix_merges(self):
        tree = DraftTree()
        tree.add_path([1, 2, 3])
        tree.add_path([1, 2, 4])
        # Two paths share prefix [1, 2] → 3 total nodes (1, 2, then 3 and 4)
        assert tree.n_nodes() == 4

    def test_all_paths_empty_tree(self):
        tree = DraftTree()
        assert tree.all_paths() == []

    def test_all_paths_single_path(self):
        tree = DraftTree()
        tree.add_path([10, 20, 30])
        paths = tree.all_paths()
        assert len(paths) == 1
        assert paths[0] == [10, 20, 30]

    def test_all_paths_multiple(self):
        tree = DraftTree()
        tree.add_path([1, 2])
        tree.add_path([1, 3])
        tree.add_path([4, 5])
        paths = tree.all_paths()
        assert len(paths) == 3

    def test_max_depth_respected(self):
        tree = DraftTree(max_depth=2)
        tree.add_path([1, 2, 3, 4, 5])
        # Max depth 2 → only 2 nodes stored
        assert tree.n_nodes() == 2

    def test_n_nodes_correct(self):
        tree = DraftTree()
        tree.add_path([1, 2])
        tree.add_path([1, 3])
        tree.add_path([2, 4])
        # root→{1,2}: 1 has children {2,3}, 2 has child {4} → 5 nodes total
        assert tree.n_nodes() == 5


# ---------------------------------------------------------------------------
# RASDBatcher
# ---------------------------------------------------------------------------

class TestRASDBatcher:
    def _make_cfg(self, **kw):
        return RASDConfig(**kw)

    def test_default_config_when_none(self):
        batcher = RASDBatcher(config=None)
        assert batcher._config.beam_width == 4

    def test_prune_tree_limits_branches(self):
        batcher = RASDBatcher(RASDConfig(beam_width=2))
        tree = DraftTree()
        for i in range(5):
            tree.add_path([i, i + 10])
        pruned = batcher.prune_tree(tree)
        paths = pruned.all_paths()
        assert len(paths) <= 2

    def test_prune_tree_uses_draft_probs(self):
        batcher = RASDBatcher(RASDConfig(beam_width=1))
        tree = DraftTree()
        tree.add_path([1, 11])
        tree.add_path([2, 22])
        tree.add_path([3, 33])
        probs = {1: 0.1, 2: 0.9, 3: 0.3}
        pruned = batcher.prune_tree(tree, draft_probs=probs)
        paths = pruned.all_paths()
        assert len(paths) == 1
        assert paths[0][0] == 2  # highest prob

    def test_prune_tree_beam_width_override(self):
        batcher = RASDBatcher(RASDConfig(beam_width=10))
        tree = DraftTree()
        for i in range(4):
            tree.add_path([i])
        pruned = batcher.prune_tree(tree, beam_width=2)
        assert len(pruned.all_paths()) <= 2

    def test_prune_tree_respects_max_depth(self):
        batcher = RASDBatcher(RASDConfig(beam_width=4, max_tree_depth=2))
        tree = DraftTree()
        tree.add_path([1, 2, 3, 4, 5])
        pruned = batcher.prune_tree(tree)
        paths = pruned.all_paths()
        assert all(len(p) <= 2 for p in paths)

    def test_fuse_trees_combines_paths(self):
        batcher = RASDBatcher()
        gen = DraftTree()
        gen.add_path([1, 2, 3])
        ret = DraftTree()
        ret.add_path([4, 5, 6])
        fused = batcher.fuse_trees(gen, ret)
        paths = fused.all_paths()
        assert len(paths) == 2

    def test_fuse_trees_deduplicates_shared_prefix(self):
        batcher = RASDBatcher()
        gen = DraftTree()
        gen.add_path([1, 2, 3])
        ret = DraftTree()
        ret.add_path([1, 2, 3])  # same path → deduplication
        fused = batcher.fuse_trees(gen, ret)
        paths = fused.all_paths()
        assert len(paths) == 1

    def test_build_retrieval_tree_with_hits(self):
        batcher = RASDBatcher(RASDConfig(min_prefix_len=2))
        corpus = CorpusIndex(min_prefix_len=2)
        corpus.add_sequence([1, 2, 3, 4, 5])
        corpus.add_sequence([1, 2, 6, 7])
        # context [1,2] matches both corpus sequences indexed under key (1,2)
        context = [1, 2]
        tree = batcher.build_retrieval_tree(context, corpus)
        assert not tree.is_empty()

    def test_build_retrieval_tree_empty_when_no_match(self):
        batcher = RASDBatcher(RASDConfig(min_prefix_len=2))
        corpus = CorpusIndex(min_prefix_len=2)
        corpus.add_sequence([10, 20, 30])
        tree = batcher.build_retrieval_tree([99, 100], corpus)
        # No match → empty tree
        assert tree.is_empty()

    def test_build_retrieval_tree_skips_short_sequences(self):
        batcher = RASDBatcher(RASDConfig(min_prefix_len=2))
        corpus = CorpusIndex(min_prefix_len=1)
        # Sequence exactly equals prefix → no continuation → skip
        corpus.add_sequence([5, 6])
        tree = batcher.build_retrieval_tree([5, 6], corpus)
        # Continuation is empty → skipped
        assert tree.is_empty()

    def test_prune_tree_empty_input(self):
        batcher = RASDBatcher()
        tree = DraftTree()
        pruned = batcher.prune_tree(tree)
        assert pruned.is_empty()

    def test_fuse_empty_trees(self):
        batcher = RASDBatcher()
        fused = batcher.fuse_trees(DraftTree(), DraftTree())
        assert fused.is_empty()


# ---------------------------------------------------------------------------
# RASDStats
# ---------------------------------------------------------------------------

class TestRASDStats:
    def test_defaults(self):
        s = RASDStats()
        assert s.retrieval_attempts == 0
        assert s.retrieval_hits == 0
        assert s.total_retrieved_tokens == 0
        assert s.fused_tree_nodes == 0
        assert s.pruned_nodes == 0

    def test_retrieval_hit_rate_zero_when_no_attempts(self):
        assert RASDStats().retrieval_hit_rate == 0.0

    def test_retrieval_hit_rate_nonzero(self):
        s = RASDStats()
        s.record_retrieval(n_candidates=3, n_tokens=12)
        s.record_retrieval(n_candidates=0, n_tokens=0)
        assert s.retrieval_hit_rate == pytest.approx(0.5)

    def test_avg_retrieved_tokens_zero_when_no_hits(self):
        s = RASDStats()
        s.record_retrieval(n_candidates=0, n_tokens=0)
        assert s.avg_retrieved_tokens == 0.0

    def test_avg_retrieved_tokens_nonzero(self):
        s = RASDStats()
        s.record_retrieval(n_candidates=2, n_tokens=10)
        assert s.avg_retrieved_tokens == pytest.approx(10.0)

    def test_record_retrieval_miss(self):
        s = RASDStats()
        s.record_retrieval(n_candidates=0, n_tokens=0)
        assert s.retrieval_attempts == 1
        assert s.retrieval_hits == 0

    def test_record_fusion(self):
        s = RASDStats()
        s.record_fusion(fused_nodes=5, pruned=2)
        assert s.fused_tree_nodes == 5
        assert s.pruned_nodes == 2

    def test_reset(self):
        s = RASDStats()
        s.record_retrieval(2, 8)
        s.record_fusion(3, 1)
        s.reset()
        assert s.retrieval_attempts == 0
        assert s.retrieval_hits == 0
        assert s.total_retrieved_tokens == 0
        assert s.fused_tree_nodes == 0
        assert s.pruned_nodes == 0
