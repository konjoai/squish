"""
tests/test_h2o_kvcache_unit.py

Unit tests for the H2O (Heavy-Hitter Oracle) classes appended to
squish/kv_cache.py — 100% coverage of H2OConfig and H2OEvictionPolicy.
"""

import numpy as np
import pytest

from squish.kv_cache import H2OConfig, H2OEvictionPolicy

# ---------------------------------------------------------------------------
# H2OConfig
# ---------------------------------------------------------------------------

class TestH2OConfig:
    def test_defaults(self):
        cfg = H2OConfig()
        assert 0 < cfg.heavy_ratio < 1
        assert cfg.recent_window >= 0
        assert cfg.max_seq_len >= 0

    def test_custom(self):
        cfg = H2OConfig(heavy_ratio=0.2, recent_window=64, max_seq_len=512)
        assert cfg.heavy_ratio == 0.2
        assert cfg.recent_window == 64
        assert cfg.max_seq_len == 512

    @pytest.mark.parametrize("kwargs, exc", [
        ({"heavy_ratio": 0.0},    "heavy_ratio"),
        ({"heavy_ratio": 1.0},    "heavy_ratio"),
        ({"recent_window": -1},   "recent_window"),
        ({"max_seq_len": -1},     "max_seq_len"),
    ])
    def test_validation(self, kwargs, exc):
        with pytest.raises(ValueError, match=exc):
            H2OConfig(**kwargs)


# ---------------------------------------------------------------------------
# H2OEvictionPolicy
# ---------------------------------------------------------------------------

class TestH2OEvictionPolicy:
    def _pol(self, **kw):
        cfg_kw = {"heavy_ratio": 0.3, "recent_window": 2, "max_seq_len": 0}
        cfg_kw.update(kw)
        return H2OEvictionPolicy(H2OConfig(**cfg_kw))

    def test_add_token_returns_sequential_positions(self):
        pol = self._pol()
        for i in range(5):
            pos = pol.add_token()
            assert pos == i
        assert pol.num_cached == 5

    def test_positions_property(self):
        pol = self._pol()
        pol.add_token()
        pol.add_token()
        assert pol.positions == [0, 1]

    def test_record_attention_accumulates(self):
        pol = self._pol()
        pol.add_token(0.0)  # pos 0
        pol.add_token(0.0)  # pos 1
        pol.record_attention(np.array([0.3, 0.7]))
        pol.record_attention(np.array([0.2, 0.8]))
        # pos 0 should have score ~ 0.5, pos 1 ~ 1.5
        hh = pol.top_heavy_hitters(2)
        top_pos = hh[0][0]
        assert top_pos == 1

    def test_evict_to_budget_keeps_recent(self):
        pol = self._pol(heavy_ratio=0.1, recent_window=3)
        for _ in range(10):
            pol.add_token()
        pol.evict_to_budget(5)
        assert pol.num_cached == 5
        # Last 3 positions must be kept (recency window)
        kept = set(pol.positions)
        for p in [7, 8, 9]:
            assert p in kept

    def test_evict_to_budget_no_op_when_under(self):
        pol = self._pol()
        for _ in range(4):
            pol.add_token()
        evicted = pol.evict_to_budget(10)
        assert evicted == []
        assert pol.num_cached == 4

    def test_evict_to_budget_invalid(self):
        pol = self._pol()
        with pytest.raises(ValueError, match="budget"):
            pol.evict_to_budget(0)

    def test_evict_respects_high_scorers(self):
        pol = self._pol(heavy_ratio=0.5, recent_window=0)
        for _i in range(6):
            pol.add_token()
        # Give positions 0 and 1 very high scores
        pol.record_attention(np.array([10.0, 10.0, 0.0, 0.0, 0.0, 0.0]))
        pol.evict_to_budget(3)
        assert pol.num_cached == 3
        kept = set(pol.positions)
        # positions 0 and 1 should be kept (heavy hitters)
        assert 0 in kept
        assert 1 in kept

    def test_top_heavy_hitters_order(self):
        pol = self._pol()
        for _ in range(4):
            pol.add_token()
        pol.record_attention(np.array([0.1, 0.4, 0.3, 0.2]))
        hh = pol.top_heavy_hitters(2)
        assert hh[0][1] >= hh[1][1]  # descending
        assert hh[0][0] == 1          # pos 1 has highest score

    def test_top_heavy_hitters_empty(self):
        pol = self._pol()
        assert pol.top_heavy_hitters(5) == []

    def test_auto_eviction_on_max_seq_len(self):
        pol = H2OEvictionPolicy(H2OConfig(heavy_ratio=0.3, recent_window=2, max_seq_len=5))
        for _ in range(8):
            pol.add_token()
        assert pol.num_cached <= 5

    def test_record_attention_partial_row(self):
        """record_attention with shorter row than cached positions (safe truncation)."""
        pol = self._pol()
        for _ in range(5):
            pol.add_token()
        # Only 3 scores; should not crash
        pol.record_attention(np.array([0.2, 0.5, 0.3]))
        assert pol.num_cached == 5  # no eviction triggered

    def test_add_token_with_init_score(self):
        pol = self._pol()
        pol.add_token(init_score=5.0)
        pol.add_token(init_score=0.0)
        hh = pol.top_heavy_hitters(1)
        assert hh[0][0] == 0   # pos 0 has higher init score
