"""tests/test_c2t_unit.py — 100 % coverage for squish/c2t.py"""
import numpy as np
import pytest

from squish.token.c2t import (
    AdaptiveTreeBuilder,
    C2TClassifier,
    C2TConfig,
    C2TFeatures,
    C2TStats,
    C2TTrainer,
)

RNG = np.random.default_rng(7)


# ---------------------------------------------------------------------------
# C2TConfig
# ---------------------------------------------------------------------------

class TestC2TConfig:
    def test_defaults(self):
        cfg = C2TConfig()
        assert cfg.tree_depth == 5
        assert cfg.wide_branches == 2
        assert cfg.narrow_branches == 1
        assert cfg.classify_threshold == pytest.approx(0.5)
        assert cfg.learning_rate == pytest.approx(1e-3)
        assert cfg.feature_dim == 3

    def test_custom(self):
        cfg = C2TConfig(tree_depth=3, wide_branches=3, narrow_branches=1)
        assert cfg.tree_depth == 3
        assert cfg.wide_branches == 3


# ---------------------------------------------------------------------------
# C2TFeatures
# ---------------------------------------------------------------------------

class TestC2TFeatures:
    def test_output_shape(self):
        logits = RNG.standard_normal(100).astype(np.float32)
        feats = C2TFeatures.compute(logits)
        assert feats.shape == (3,)
        assert feats.dtype == np.float32

    def test_top1_is_max(self):
        logits = np.array([1.0, 5.0, 3.0, 2.0], dtype=np.float32)
        feats = C2TFeatures.compute(logits)
        assert feats[0] == pytest.approx(5.0)

    def test_gap_is_top1_minus_top2(self):
        logits = np.array([1.0, 5.0, 4.0, 0.0], dtype=np.float32)
        feats = C2TFeatures.compute(logits)
        assert feats[1] == pytest.approx(5.0 - 4.0)

    def test_entropy_positive(self):
        logits = RNG.standard_normal(50).astype(np.float32)
        feats = C2TFeatures.compute(logits)
        assert feats[2] > 0.0

    def test_uniform_logits_high_entropy(self):
        logits = np.zeros(100, dtype=np.float32)
        feats = C2TFeatures.compute(logits)
        # entropy should be near log(100)
        assert feats[2] > 4.0

    def test_spike_logits_low_entropy(self):
        # Nearly one-hot → should have near-zero entropy
        logits = np.full(100, -100.0, dtype=np.float32)
        logits[0] = 100.0
        feats = C2TFeatures.compute(logits)
        assert feats[2] < 0.01

    def test_invalid_ndim_raises(self):
        with pytest.raises(ValueError, match="1-D"):
            C2TFeatures.compute(np.ones((3, 4), dtype=np.float32))


# ---------------------------------------------------------------------------
# C2TClassifier
# ---------------------------------------------------------------------------

class TestC2TClassifier:
    def test_initial_properties(self):
        clf = C2TClassifier(feature_dim=3, threshold=0.5)
        assert clf.threshold == pytest.approx(0.5)
        assert clf.weights.shape == (3,)
        assert isinstance(clf.bias, float)

    def test_score_returns_probability(self):
        clf = C2TClassifier()
        feats = np.zeros(3, dtype=np.float32)
        s = clf.score(feats)
        assert 0.0 <= s <= 1.0

    def test_score_invalid_feature_dim(self):
        clf = C2TClassifier(feature_dim=3)
        with pytest.raises(ValueError, match="feature"):
            clf.score(np.zeros(4, dtype=np.float32))

    def test_score_invalid_ndim(self):
        clf = C2TClassifier(feature_dim=3)
        with pytest.raises(ValueError):
            clf.score(np.zeros((2, 3), dtype=np.float32))

    def test_classify_returns_0_or_1(self):
        clf = C2TClassifier()
        feats = RNG.standard_normal(3).astype(np.float32)
        result = clf.classify(feats)
        assert result in (0, 1)

    def test_classify_wide_when_score_above_threshold(self):
        # Force score > threshold by pushing weights
        clf = C2TClassifier(feature_dim=1, threshold=0.5)
        clf._w = np.array([100.0], dtype=np.float32)  # very large weight
        feats = np.array([1.0], dtype=np.float32)
        assert clf.classify(feats) == 1

    def test_classify_narrow_when_score_below_threshold(self):
        clf = C2TClassifier(feature_dim=1, threshold=0.5)
        clf._w = np.array([-100.0], dtype=np.float32)  # very negative
        feats = np.array([1.0], dtype=np.float32)
        assert clf.classify(feats) == 0

    def test_update_changes_weights(self):
        clf = C2TClassifier()
        feats = np.ones(3, dtype=np.float32)
        w_before = clf.weights.copy()
        clf.update(feats, label=1, lr=1.0)
        assert not np.allclose(clf.weights, w_before)

    def test_reset_zeroes_weights(self):
        clf = C2TClassifier()
        clf._w[:] = 99.0
        clf.reset()
        np.testing.assert_array_equal(clf.weights, np.zeros(3))
        assert clf.bias == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# AdaptiveTreeBuilder
# ---------------------------------------------------------------------------

class TestAdaptiveTreeBuilder:
    VOCAB = 20

    def _dummy_draft_fn(self, hidden: np.ndarray):
        """Returns random logits and same hidden (stateless for tests)."""
        logits = RNG.standard_normal(self.VOCAB).astype(np.float32)
        return logits, hidden

    def test_default_construction(self):
        builder = AdaptiveTreeBuilder()
        assert builder.classifier is not None

    def test_custom_classifier(self):
        clf = C2TClassifier()
        builder = AdaptiveTreeBuilder(classifier=clf)
        assert builder.classifier is clf

    def test_classify_position_returns_valid_width(self):
        cfg = C2TConfig(wide_branches=2, narrow_branches=1)
        builder = AdaptiveTreeBuilder(config=cfg)
        logits = np.zeros(self.VOCAB, dtype=np.float32)
        w = builder.classify_position(logits)
        assert w in (cfg.wide_branches, cfg.narrow_branches)

    def test_build_returns_list_of_paths(self):
        cfg = C2TConfig(tree_depth=2, wide_branches=2, narrow_branches=1)
        builder = AdaptiveTreeBuilder(config=cfg)
        root = np.zeros(8, dtype=np.float32)
        paths = builder.build(self._dummy_draft_fn, root)
        assert isinstance(paths, list)
        assert len(paths) > 0
        assert all(isinstance(p, list) for p in paths)
        assert all(isinstance(t, int) for p in paths for t in p)

    def test_build_respects_depth(self):
        cfg = C2TConfig(tree_depth=3, wide_branches=1, narrow_branches=1)
        builder = AdaptiveTreeBuilder(config=cfg)
        root = np.zeros(8, dtype=np.float32)
        paths = builder.build(self._dummy_draft_fn, root)
        assert all(len(p) == cfg.tree_depth for p in paths)

    def test_build_with_forced_narrow(self):
        cfg = C2TConfig(tree_depth=2, wide_branches=3, narrow_branches=1)
        builder = AdaptiveTreeBuilder(config=cfg)
        root = np.zeros(8, dtype=np.float32)
        # Force all positions narrow
        forced = [True, True]
        paths = builder.build(self._dummy_draft_fn, root, forced_narrow=forced)
        # With narrow=1 at every level, only 1 path
        assert len(paths) == 1

    def test_build_wide_produces_multiple_branches(self):
        # Force classifier to always return wide by using large positive weight
        cfg = C2TConfig(tree_depth=1, wide_branches=3, narrow_branches=1)
        clf = C2TClassifier(feature_dim=3, threshold=0.0)  # threshold=0 → always wide
        builder = AdaptiveTreeBuilder(config=cfg, classifier=clf)
        root = np.zeros(8, dtype=np.float32)
        paths = builder.build(self._dummy_draft_fn, root)
        assert len(paths) == 3

    def test_build_zero_depth_returns_empty(self):
        # tree_depth=0 → no BFS iterations → queue contains only (root, [])
        # 'if prefix:' is False for empty prefix → no paths appended
        cfg = C2TConfig(tree_depth=0, wide_branches=2, narrow_branches=1)
        builder = AdaptiveTreeBuilder(config=cfg)
        root = np.zeros(8, dtype=np.float32)
        paths = builder.build(self._dummy_draft_fn, root)
        assert paths == []


# ---------------------------------------------------------------------------
# C2TTrainer
# ---------------------------------------------------------------------------

class TestC2TTrainer:
    def _make_trainer(self):
        clf = C2TClassifier()
        cfg = C2TConfig(learning_rate=0.01)
        return C2TTrainer(classifier=clf, config=cfg), clf

    def test_initial_updates(self):
        trainer, _ = self._make_trainer()
        assert trainer.n_updates == 0

    def test_default_config_when_none(self):
        clf = C2TClassifier()
        trainer = C2TTrainer(classifier=clf, config=None)
        assert trainer.n_updates == 0

    def test_update_increments_counter(self):
        trainer, _ = self._make_trainer()
        feats = np.ones(3, dtype=np.float32)
        trainer.update(feats, branching_helped=True)
        assert trainer.n_updates == 1

    def test_update_label_1_shifts_score_up(self):
        trainer, clf = self._make_trainer()
        feats = np.ones(3, dtype=np.float32)
        # Multiple updates with label=1 → score should increase (toward 1)
        for _ in range(50):
            trainer.update(feats, branching_helped=True)
        assert clf.score(feats) > 0.5

    def test_update_label_0_shifts_score_down(self):
        trainer, clf = self._make_trainer()
        feats = np.ones(3, dtype=np.float32)
        # Start with moderately positive weights (score ≈ 0.95)
        clf._w[:] = 1.0
        # Now update with label=0 repeatedly → should decrease well below 0.9
        for _ in range(50):
            trainer.update(feats, branching_helped=False)
        assert clf.score(feats) < 0.9  # score should have dropped

    def test_reset_clears_counter(self):
        trainer, _ = self._make_trainer()
        feats = np.ones(3, dtype=np.float32)
        trainer.update(feats, branching_helped=True)
        trainer.reset()
        assert trainer.n_updates == 0


# ---------------------------------------------------------------------------
# C2TStats
# ---------------------------------------------------------------------------

class TestC2TStats:
    def test_defaults(self):
        s = C2TStats()
        assert s.wide_decisions == 0
        assert s.narrow_decisions == 0
        assert s.wide_helped == 0

    def test_wide_fraction_zero_when_no_decisions(self):
        assert C2TStats().wide_fraction == 0.0

    def test_wide_fraction_all_wide(self):
        s = C2TStats()
        s.record_wide(helped=True)
        s.record_wide(helped=False)
        assert s.wide_fraction == 1.0

    def test_wide_fraction_mixed(self):
        s = C2TStats()
        s.record_wide(helped=True)
        s.record_narrow()
        assert s.wide_fraction == pytest.approx(0.5)

    def test_wide_help_rate_zero_when_no_wides(self):
        assert C2TStats().wide_help_rate == 0.0

    def test_wide_help_rate_all_helped(self):
        s = C2TStats()
        s.record_wide(helped=True)
        s.record_wide(helped=True)
        assert s.wide_help_rate == 1.0

    def test_wide_help_rate_partial(self):
        s = C2TStats()
        s.record_wide(helped=True)
        s.record_wide(helped=False)
        assert s.wide_help_rate == pytest.approx(0.5)

    def test_record_wide_not_helped(self):
        s = C2TStats()
        s.record_wide(helped=False)
        assert s.wide_decisions == 1
        assert s.wide_helped == 0

    def test_record_narrow(self):
        s = C2TStats()
        s.record_narrow()
        assert s.narrow_decisions == 1

    def test_reset(self):
        s = C2TStats()
        s.record_wide(helped=True)
        s.record_narrow()
        s.reset()
        assert s.wide_decisions == 0
        assert s.narrow_decisions == 0
        assert s.wide_helped == 0
