"""
tests/test_medusa_spec_unit.py

Unit tests for the Medusa speculative-decoding classes appended to
squish/speculative.py — 100% coverage of:
  MedusaConfig, MedusaHead, MedusaTreeDraft, MedusaGenerator.
"""

import numpy as np
import pytest

from squish.speculative import (
    MedusaConfig,
    MedusaHead,
    MedusaTreeDraft,
    MedusaGenerator,
)


# ---------------------------------------------------------------------------
# MedusaConfig
# ---------------------------------------------------------------------------

class TestMedusaConfig:
    def test_defaults(self):
        cfg = MedusaConfig()
        assert cfg.num_heads >= 1
        assert cfg.top_k >= 1
        assert cfg.hidden_dim >= 1
        assert cfg.vocab_size >= 1
        assert 0.0 <= cfg.acceptance_threshold <= 1.0

    def test_custom(self):
        cfg = MedusaConfig(num_heads=5, top_k=3, hidden_dim=256, vocab_size=1000)
        assert cfg.num_heads == 5
        assert cfg.top_k == 3

    @pytest.mark.parametrize("kwargs, exc", [
        ({"num_heads": 0},              "num_heads"),
        ({"top_k": 0},                  "top_k"),
        ({"hidden_dim": 0},             "hidden_dim"),
        ({"vocab_size": 0},             "vocab_size"),
        ({"acceptance_threshold": -0.1}, "acceptance_threshold"),
        ({"acceptance_threshold": 1.1},  "acceptance_threshold"),
    ])
    def test_validation(self, kwargs, exc):
        with pytest.raises(ValueError, match=exc):
            MedusaConfig(**kwargs)


# ---------------------------------------------------------------------------
# MedusaHead
# ---------------------------------------------------------------------------

class TestMedusaHead:
    def test_random_init_logits_shape(self):
        rng = np.random.default_rng(0)
        head = MedusaHead(hidden_dim=16, vocab_size=50, rng=rng)
        hidden = np.ones(16)
        logits = head.logits(hidden)
        assert logits.shape == (50,)

    def test_zero_init(self):
        head = MedusaHead(hidden_dim=8, vocab_size=20)  # rng=None → zeros
        hidden = np.ones(8)
        logits = head.logits(hidden)
        assert logits.shape == (20,)
        assert np.allclose(logits, 0.0)

    def test_top_k_tokens_count(self):
        rng = np.random.default_rng(1)
        head = MedusaHead(hidden_dim=32, vocab_size=100, rng=rng)
        hidden = rng.standard_normal(32)
        toks = head.top_k_tokens(hidden, k=5)
        assert toks.shape == (5,)
        assert len(set(toks)) == 5   # all distinct

    def test_top_k_tokens_k_equals_vocab(self):
        rng = np.random.default_rng(2)
        head = MedusaHead(hidden_dim=4, vocab_size=10, rng=rng)
        hidden = rng.standard_normal(4)
        toks = head.top_k_tokens(hidden, k=10)
        assert len(toks) == 10

    def test_wrong_shape_raises(self):
        head = MedusaHead(hidden_dim=8, vocab_size=20)
        with pytest.raises(ValueError, match="hidden"):
            head.logits(np.ones(5))   # wrong hidden_dim

    def test_invalid_dims(self):
        with pytest.raises(ValueError, match="hidden_dim"):
            MedusaHead(hidden_dim=0, vocab_size=10)

    def test_weight_bias_accessible(self):
        rng = np.random.default_rng(3)
        head = MedusaHead(hidden_dim=4, vocab_size=8, rng=rng)
        assert head.weight.shape == (8, 4)
        assert head.bias.shape   == (8,)


# ---------------------------------------------------------------------------
# MedusaTreeDraft
# ---------------------------------------------------------------------------

class TestMedusaTreeDraft:
    def _make_heads(self, num_heads, hidden_dim, vocab_size):
        rng = np.random.default_rng(42)
        return [MedusaHead(hidden_dim, vocab_size, rng=rng) for _ in range(num_heads)]

    def test_wrong_head_count_raises(self):
        cfg   = MedusaConfig(num_heads=3, top_k=2, hidden_dim=8, vocab_size=20)
        heads = self._make_heads(2, 8, 20)
        with pytest.raises(ValueError, match="heads"):
            MedusaTreeDraft(cfg, heads)

    def test_draft_returns_candidates(self):
        cfg   = MedusaConfig(num_heads=2, top_k=3, hidden_dim=8, vocab_size=50)
        heads = self._make_heads(2, 8, 50)
        tree  = MedusaTreeDraft(cfg, heads)
        hidden = np.ones(8)
        cands = tree.draft(hidden, max_candidates=9)
        assert len(cands) >= 1
        assert len(cands) <= 9
        for c in cands:
            assert len(c) == 2   # num_heads == 2

    def test_draft_max_candidates_limit(self):
        cfg   = MedusaConfig(num_heads=3, top_k=4, hidden_dim=16, vocab_size=100)
        heads = self._make_heads(3, 16, 100)
        tree  = MedusaTreeDraft(cfg, heads)
        hidden = np.random.default_rng(5).standard_normal(16)
        cands = tree.draft(hidden, max_candidates=4)
        assert len(cands) <= 4

    def test_single_head_single_top_k(self):
        cfg   = MedusaConfig(num_heads=1, top_k=1, hidden_dim=4, vocab_size=10)
        heads = self._make_heads(1, 4, 10)
        tree  = MedusaTreeDraft(cfg, heads)
        hidden = np.zeros(4)
        cands = tree.draft(hidden)
        assert len(cands) == 1
        assert len(cands[0]) == 1

    def test_large_tree_pruned(self):
        cfg   = MedusaConfig(num_heads=4, top_k=10, hidden_dim=8, vocab_size=200)
        heads = self._make_heads(4, 8, 200)
        tree  = MedusaTreeDraft(cfg, heads)
        hidden = np.ones(8)
        cands = tree.draft(hidden, max_candidates=16)
        assert len(cands) <= 16


# ---------------------------------------------------------------------------
# MedusaGenerator
# ---------------------------------------------------------------------------

class TestMedusaGenerator:
    def _setup(self, vocab=50, hidden_dim=16, num_heads=2, top_k=3):
        cfg   = MedusaConfig(num_heads=num_heads, top_k=top_k,
                             hidden_dim=hidden_dim, vocab_size=vocab)
        rng   = np.random.default_rng(7)
        heads = [MedusaHead(hidden_dim, vocab, rng=rng) for _ in range(num_heads)]
        target_tok = 5

        def hidden_fwd(ids):
            hidden = rng.standard_normal(hidden_dim)
            logits = np.full(vocab, -10.0)
            logits[target_tok] = 10.0
            return hidden, logits

        def verify_fwd(ids):
            logits = np.full(vocab, -10.0)
            logits[target_tok] = 10.0
            return logits

        return cfg, heads, hidden_fwd, verify_fwd, target_tok

    def test_generate_correct_length(self):
        cfg, heads, hfwd, vfwd, _ = self._setup()
        gen = MedusaGenerator(hfwd, vfwd, cfg, heads)
        ids = gen.generate([1, 2, 3], max_new_tokens=8)
        assert len(ids) == 11   # 3 prompt + 8 new

    def test_acceptance_rate_initial_zero(self):
        cfg, heads, hfwd, vfwd, _ = self._setup()
        gen = MedusaGenerator(hfwd, vfwd, cfg, heads)
        assert gen.acceptance_rate == 0.0

    def test_acceptance_rate_after_generation(self):
        cfg, heads, hfwd, vfwd, target_tok = self._setup()
        gen = MedusaGenerator(hfwd, vfwd, cfg, heads)
        gen.generate([0], max_new_tokens=10)
        # Some tokens accepted, some rejected — depends on head weights
        rate = gen.acceptance_rate
        assert 0.0 <= rate <= 1.0

    def test_generate_empty_candidates_fallback(self):
        """When candidates is empty, must fall back to base logit greedy."""
        vocab, hidden_dim = 20, 8
        cfg = MedusaConfig(num_heads=1, top_k=1, hidden_dim=hidden_dim, vocab_size=vocab)

        class ZeroHead(MedusaHead):
            def top_k_tokens(self, hidden, k):
                return np.array([], dtype=np.int32)  # empty!

        heads = [ZeroHead(hidden_dim, vocab)]
        base_tok = 3

        def hfwd(ids):
            logits = np.full(vocab, -10.0)
            logits[base_tok] = 10.0
            return np.zeros(hidden_dim), logits

        def vfwd(ids):
            logits = np.full(vocab, -10.0)
            logits[base_tok] = 10.0
            return logits

        gen = MedusaGenerator(hfwd, vfwd, cfg, heads)
        ids = gen.generate([0], max_new_tokens=3)
        assert len(ids) == 4
        assert all(t == base_tok for t in ids[1:])

    def test_rejection_triggers_verifier_token(self):
        """Draft and verify disagree → rejected, verifier's token used."""
        vocab, hidden_dim = 30, 8
        draft_tok  = 7
        verify_tok = 15
        cfg   = MedusaConfig(num_heads=1, top_k=1, hidden_dim=hidden_dim, vocab_size=vocab)
        rng   = np.random.default_rng(42)

        class FixedHead(MedusaHead):
            def top_k_tokens(self, hidden, k):
                return np.array([draft_tok], dtype=np.int32)

        heads = [FixedHead(hidden_dim, vocab)]

        def hfwd(ids):
            logits = np.full(vocab, -10.0)
            logits[draft_tok] = 10.0
            return rng.standard_normal(hidden_dim), logits

        def vfwd(ids):
            logits = np.full(vocab, -10.0)
            logits[verify_tok] = 10.0   # always disagrees
            return logits

        gen = MedusaGenerator(hfwd, vfwd, cfg, heads)
        ids = gen.generate([0], max_new_tokens=4)
        # verify_tok should appear in output
        assert verify_tok in ids
