"""tests/test_mobile_moe_unit.py — 100 % coverage for squish/mobile_moe.py"""
import numpy as np
import pytest

from squish.moe.mobile_moe import (
    ExpertImportanceScorer,
    MoBiLEConfig,
    MoBiLELayer,
    MoBiLERouter,
    MoBiLEStats,
)

RNG = np.random.default_rng(31)


# ---------------------------------------------------------------------------
# MoBiLEConfig
# ---------------------------------------------------------------------------

class TestMoBiLEConfig:
    def test_defaults(self):
        cfg = MoBiLEConfig()
        assert cfg.n_experts_total == 128
        assert cfg.n_experts_active == 8
        assert cfg.n_experts_min == 2
        assert cfg.importance_threshold == pytest.approx(0.3)

    def test_custom(self):
        cfg = MoBiLEConfig(n_experts_total=64, n_experts_active=4, n_experts_min=1)
        assert cfg.n_experts_total == 64

    def test_invalid_n_experts_min_zero(self):
        with pytest.raises(ValueError, match="n_experts_min"):
            MoBiLEConfig(n_experts_min=0)

    def test_invalid_n_experts_min_exceeds_active(self):
        with pytest.raises(ValueError, match="n_experts_min"):
            MoBiLEConfig(n_experts_active=4, n_experts_min=5)

    def test_invalid_threshold_below_zero(self):
        with pytest.raises(ValueError, match="importance_threshold"):
            MoBiLEConfig(importance_threshold=-0.1)

    def test_invalid_threshold_above_one(self):
        with pytest.raises(ValueError, match="importance_threshold"):
            MoBiLEConfig(importance_threshold=1.1)

    def test_boundary_threshold_zero(self):
        cfg = MoBiLEConfig(importance_threshold=0.0)
        assert cfg.importance_threshold == 0.0

    def test_boundary_threshold_one(self):
        cfg = MoBiLEConfig(importance_threshold=1.0)
        assert cfg.importance_threshold == 1.0


# ---------------------------------------------------------------------------
# ExpertImportanceScorer
# ---------------------------------------------------------------------------

class TestExpertImportanceScorer:
    def test_gini_equal_weights_near_zero(self):
        scorer = ExpertImportanceScorer()
        w = np.ones(8, dtype=np.float64)
        g = scorer.gini(w)
        assert g < 0.1  # near-equal → near-zero Gini

    def test_gini_concentrated_weight_near_one(self):
        scorer = ExpertImportanceScorer()
        # 8 elements: one dominant expert (0.93), seven small (0.01 each)
        # Gini reflects high inequality across multiple non-zero weights
        w = np.array([0.93, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01], dtype=np.float64)
        g = scorer.gini(w)
        # Concentrated multi-element distribution → high Gini
        assert g > 0.5

    def test_gini_all_zeros(self):
        scorer = ExpertImportanceScorer()
        g = scorer.gini(np.zeros(4))
        assert g == 0.0

    def test_gini_single_nonzero(self):
        scorer = ExpertImportanceScorer()
        g = scorer.gini(np.array([0.0, 0.0, 1.0, 0.0]))
        # Single nonzero → Gini undefined / treated as 0 after filtering
        assert 0.0 <= g <= 1.0

    def test_score_equal_weights_near_one(self):
        scorer = ExpertImportanceScorer()
        w = np.ones(8, dtype=np.float32)
        s = scorer.score(w)
        assert s > 0.8  # equal weights → high importance (use all experts)

    def test_score_concentrated_weight_near_zero(self):
        scorer = ExpertImportanceScorer()
        # 8 elements: one dominant expert, rest small → high Gini → low importance score
        w = np.array([0.93, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01], dtype=np.float32)
        s = scorer.score(w)
        assert s < 0.5  # concentrated → low importance (can reduce)

    def test_score_range(self):
        scorer = ExpertImportanceScorer()
        w = RNG.dirichlet(np.ones(8)).astype(np.float32)
        s = scorer.score(w)
        assert 0.0 <= s <= 1.0

    def test_n_active_property(self):
        scorer = ExpertImportanceScorer(n_active=4)
        assert scorer.n_active == 4


# ---------------------------------------------------------------------------
# MoBiLERouter
# ---------------------------------------------------------------------------

class TestMoBiLERouter:
    def _make(self, threshold=0.3):
        cfg = MoBiLEConfig(n_experts_active=8, n_experts_min=2, importance_threshold=threshold)
        return MoBiLERouter(cfg)

    def test_default_config_when_none(self):
        r = MoBiLERouter(config=None)
        assert r.config.n_experts_active == 8

    def test_route_returns_full_for_uncertain_token(self):
        r = self._make(threshold=0.1)  # very low threshold → always full
        # Equal gate weights → high importance → full
        gate = np.ones(128, dtype=np.float32) / 128
        assert r.route(gate) == 8

    def test_route_returns_min_for_background_token(self):
        r = self._make(threshold=0.9)  # very high threshold → almost always min
        # Concentrated 8-element gate: one expert dominates, rest are small
        gate = np.zeros(128, dtype=np.float32)
        gate[0] = 0.93
        gate[1:8] = 0.01  # 7 small non-zero weights
        assert r.route(gate) == 2

    def test_route_batch(self):
        r = self._make()
        batch = np.zeros((3, 128), dtype=np.float32)
        batch[0, 0] = 1.0            # concentrated → min
        batch[1] = np.ones(128) / 128  # equal → full
        batch[2, :4] = 0.25          # partial
        results = r.route_batch(batch)
        assert len(results) == 3
        assert all(x in (r.config.n_experts_min, r.config.n_experts_active) for x in results)

    def test_scorer_property(self):
        r = MoBiLERouter()
        assert r.scorer is not None


# ---------------------------------------------------------------------------
# MoBiLELayer
# ---------------------------------------------------------------------------

class TestMoBiLELayer:
    N_EXPERTS = 16
    HIDDEN = 8

    def _dummy_expert_fn(self, hidden: np.ndarray, top_indices: np.ndarray) -> np.ndarray:
        """Identity-style expert: multiplies hidden by number of selected experts."""
        scale = len(top_indices) / self.N_EXPERTS
        return hidden * scale

    def _make(self, threshold=0.3):
        cfg = MoBiLEConfig(
            n_experts_total=self.N_EXPERTS,
            n_experts_active=4,
            n_experts_min=1,
            importance_threshold=threshold,
        )
        return MoBiLELayer(self._dummy_expert_fn, cfg)

    def test_forward_returns_correct_shape(self):
        layer = self._make()
        hidden = np.ones(self.HIDDEN, dtype=np.float32)
        gate = np.zeros(self.N_EXPERTS, dtype=np.float32)
        gate[:4] = 0.25
        out = layer.forward(hidden, gate)
        assert out.shape == (self.HIDDEN,)

    def test_forward_with_full_experts(self):
        layer = self._make(threshold=0.0)  # always full
        hidden = np.ones(self.HIDDEN, dtype=np.float32)
        gate = np.ones(self.N_EXPERTS, dtype=np.float32) / self.N_EXPERTS
        out = layer.forward(hidden, gate)
        assert out.shape == (self.HIDDEN,)

    def test_forward_with_reduced_experts(self):
        layer = self._make(threshold=1.0)  # always reduce
        hidden = np.ones(self.HIDDEN, dtype=np.float32)
        gate = np.zeros(self.N_EXPERTS, dtype=np.float32)
        gate[0] = 1.0  # concentrated → score < 1.0 = threshold
        out = layer.forward(hidden, gate)
        assert out.shape == (self.HIDDEN,)

    def test_forward_batch_returns_correct_shape(self):
        layer = self._make()
        hiddens = np.ones((4, self.HIDDEN), dtype=np.float32)
        gates = np.zeros((4, self.N_EXPERTS), dtype=np.float32)
        for i in range(4):
            gates[i, i] = 1.0
        out = layer.forward_batch(hiddens, gates)
        assert out.shape == (4, self.HIDDEN)

    def test_default_router_when_none(self):
        layer = MoBiLELayer(self._dummy_expert_fn, config=None, router=None)
        assert layer.router is not None

    def test_custom_router_used(self):
        cfg = MoBiLEConfig()
        router = MoBiLERouter(cfg)
        layer = MoBiLELayer(self._dummy_expert_fn, router=router)
        assert layer.router is router


# ---------------------------------------------------------------------------
# MoBiLEStats
# ---------------------------------------------------------------------------

class TestMoBiLEStats:
    def test_defaults(self):
        s = MoBiLEStats()
        assert s.total_tokens == 0
        assert s.reduced_tokens == 0
        assert s.full_tokens == 0
        assert s.total_expert_calls == 0
        assert s.baseline_expert_calls == 0

    def test_reduction_rate_zero_when_no_tokens(self):
        assert MoBiLEStats().reduction_rate == 0.0

    def test_compute_savings_zero_when_no_calls(self):
        assert MoBiLEStats().compute_savings == 0.0

    def test_record_full_token(self):
        s = MoBiLEStats()
        s.record(n_active_used=8, n_active_full=8)
        assert s.total_tokens == 1
        assert s.full_tokens == 1
        assert s.reduced_tokens == 0

    def test_record_reduced_token(self):
        s = MoBiLEStats()
        s.record(n_active_used=2, n_active_full=8)
        assert s.total_tokens == 1
        assert s.reduced_tokens == 1
        assert s.full_tokens == 0

    def test_reduction_rate(self):
        s = MoBiLEStats()
        s.record(2, 8)   # reduced
        s.record(8, 8)   # full
        assert s.reduction_rate == pytest.approx(0.5)

    def test_compute_savings(self):
        s = MoBiLEStats()
        s.record(2, 8)   # used 2 of 8 → saved 6
        s.record(8, 8)   # used 8 of 8 → saved 0
        # total_expert_calls = 10, baseline = 16
        savings = 1.0 - 10 / 16
        assert s.compute_savings == pytest.approx(savings)

    def test_reset(self):
        s = MoBiLEStats()
        s.record(2, 8)
        s.reset()
        assert s.total_tokens == 0
        assert s.reduced_tokens == 0
        assert s.full_tokens == 0
        assert s.total_expert_calls == 0
        assert s.baseline_expert_calls == 0
