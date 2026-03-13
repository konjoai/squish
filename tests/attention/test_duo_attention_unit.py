"""
tests/test_duo_attention_unit.py

Unit tests for squish/duo_attention.py — 100% coverage.
"""

import numpy as np
import pytest

from squish.attention.duo_attention import (
    DuoAttentionConfig,
    DuoKVManager,
    HeadCalibration,
    HeadClassifier,
    StreamingKVWindow,
)

# ---------------------------------------------------------------------------
# DuoAttentionConfig
# ---------------------------------------------------------------------------

class TestDuoAttentionConfig:
    def test_defaults(self):
        cfg = DuoAttentionConfig()
        assert cfg.num_layers == 32
        assert cfg.num_heads  == 32
        assert cfg.head_dim   == 128
        assert cfg.sink_tokens == 4
        assert cfg.local_window == 256
        assert 0 < cfg.retrieval_threshold < 1
        assert 0 <= cfg.min_retrieval_fraction <= 1

    def test_custom_values(self):
        cfg = DuoAttentionConfig(num_layers=4, num_heads=8, head_dim=64)
        assert cfg.num_layers == 4
        assert cfg.num_heads  == 8
        assert cfg.head_dim   == 64

    @pytest.mark.parametrize("kwargs, exc", [
        ({"num_layers": 0},              "num_layers"),
        ({"num_heads": 0},               "num_heads"),
        ({"head_dim": 0},                "head_dim"),
        ({"sink_tokens": -1},            "sink_tokens"),
        ({"local_window": 0},            "local_window"),
        ({"retrieval_threshold": 0.0},   "retrieval_threshold"),
        ({"retrieval_threshold": 1.0},   "retrieval_threshold"),
        ({"min_retrieval_fraction": -0.1}, "min_retrieval_fraction"),
        ({"min_retrieval_fraction": 1.1},  "min_retrieval_fraction"),
    ])
    def test_validation_errors(self, kwargs, exc):
        with pytest.raises(ValueError, match=exc):
            DuoAttentionConfig(**kwargs)


# ---------------------------------------------------------------------------
# HeadCalibration
# ---------------------------------------------------------------------------

class TestHeadCalibration:
    def _make_cfg(self, **kw):
        defaults = dict(num_layers=2, num_heads=4, head_dim=8, sink_tokens=2,
                        local_window=4, retrieval_threshold=0.1,
                        min_retrieval_fraction=0.25)
        defaults.update(kw)
        return DuoAttentionConfig(**defaults)

    def test_record_and_classify_basic(self):
        cfg = self._make_cfg()
        cal = HeadCalibration(cfg)
        np.random.default_rng(0)
        # 4 heads, 3 queries, 10 keys
        # Heads 0,1: all attention in out-of-window zone → high score → retrieval
        # Heads 2,3: all attention in sink/window → low score → streaming
        n_heads, q_len, k_len = 4, 3, 10
        attn = np.zeros((n_heads, q_len, k_len))
        attn[0, :, 3:7] = 1.0 / 4   # out-of-window (sink=2, window=4 → in_window=[0,1,6,7,8,9])
        attn[1, :, 4:6] = 0.5        # out-of-window
        attn[2, :, 0:2] = 0.5        # sink → in-window
        attn[3, :, 8:10] = 0.5       # tail window → in-window
        cal.record(0, attn)
        scores = cal.mean_scores()
        assert (0, 0) in scores
        assert scores[(0, 0)] > scores[(0, 2)]

    def test_classify_returns_all_heads(self):
        cfg = self._make_cfg()
        cal = HeadCalibration(cfg)
        attn = np.ones((4, 1, 8)) / 8
        cal.record(0, attn)
        cal.record(1, attn)
        labels = cal.classify()
        assert len(labels) == cfg.num_layers * cfg.num_heads
        assert all(v in ("retrieval", "streaming") for v in labels.values())

    def test_min_retrieval_fraction_enforced(self):
        cfg = self._make_cfg(min_retrieval_fraction=0.5)
        cal = HeadCalibration(cfg)
        # All heads attend only in-window → all scores low
        attn = np.zeros((4, 1, 10))
        attn[:, :, 8:10] = 0.5     # in local window
        cal.record(0, attn)
        cal.record(1, attn)
        labels = cal.classify()
        n_ret = sum(1 for v in labels.values() if v == "retrieval")
        total = cfg.num_layers * cfg.num_heads
        assert n_ret >= int(total * 0.5)

    def test_record_wrong_shape_raises(self):
        cfg = self._make_cfg()
        cal = HeadCalibration(cfg)
        with pytest.raises(ValueError, match="n_heads"):
            cal.record(0, np.ones((2, 3)))   # 2-D, wrong

    def test_no_data_classify_defaults_streaming(self):
        cfg = self._make_cfg(min_retrieval_fraction=0.0)
        cal = HeadCalibration(cfg)
        labels = cal.classify()
        # With no data min_retrieval_fraction=0 and threshold not matched
        assert all(v == "streaming" for v in labels.values())

    def test_mean_scores_empty(self):
        cfg = self._make_cfg()
        cal = HeadCalibration(cfg)
        assert cal.mean_scores() == {}


# ---------------------------------------------------------------------------
# HeadClassifier
# ---------------------------------------------------------------------------

class TestHeadClassifier:
    def test_record_and_classify(self):
        cfg = DuoAttentionConfig(num_layers=1, num_heads=2, head_dim=4,
                                  sink_tokens=1, local_window=2,
                                  retrieval_threshold=0.1, min_retrieval_fraction=0.0)
        clf = HeadClassifier(cfg)
        # head 0: high out-of-window → retrieval
        attn = np.zeros((2, 1, 6))
        attn[0, :, 2:4] = 0.5   # out of sink(1) + window(last 2 = [4,5])
        clf.record(0, attn)
        labels = clf.classify()
        assert labels[(0, 0)] == "retrieval"

    def test_retrieval_fraction(self):
        cfg = DuoAttentionConfig(num_layers=2, num_heads=4, head_dim=8,
                                  sink_tokens=2, local_window=4,
                                  retrieval_threshold=0.05, min_retrieval_fraction=0.25)
        clf = HeadClassifier(cfg)
        attn = np.ones((4, 1, 10)) / 10
        clf.record(0, attn)
        clf.record(1, attn)
        frac = clf.retrieval_fraction()
        assert 0.0 <= frac <= 1.0


# ---------------------------------------------------------------------------
# StreamingKVWindow
# ---------------------------------------------------------------------------

class TestStreamingKVWindow:
    def test_push_sink_positions(self):
        w = StreamingKVWindow(sink_tokens=2, window=4, head_dim=3)
        w.push(0, np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0]))
        w.push(1, np.array([7.0, 8.0, 9.0]), np.array([10., 11., 12.]))
        keys, vals = w.get_kv()
        assert keys.shape == (2, 3)
        assert len(w) == 2

    def test_push_recent_tokens_rotate(self):
        w = StreamingKVWindow(sink_tokens=0, window=3, head_dim=2)
        for i in range(5):
            w.push(i, np.array([float(i), 0.]), np.zeros(2))
        # Window=3: only last 3 recent
        keys, _ = w.get_kv()
        assert keys.shape[0] == 3
        assert len(w) == 3

    def test_sink_plus_window(self):
        w = StreamingKVWindow(sink_tokens=2, window=3, head_dim=4)
        for i in range(7):
            w.push(i, np.ones(4) * i, np.zeros(4))
        keys, _ = w.get_kv()
        # 2 sinks + 3 recent = 5
        assert keys.shape[0] == 5

    def test_get_kv_empty(self):
        w = StreamingKVWindow(sink_tokens=2, window=4, head_dim=8)
        keys, vals = w.get_kv()
        assert keys.shape == (0, 8)
        assert vals.shape == (0, 8)

    @pytest.mark.parametrize("kwargs, exc", [
        ({"sink_tokens": -1, "window": 4, "head_dim": 8}, "sink_tokens"),
        ({"sink_tokens": 0,  "window": 0, "head_dim": 8}, "window"),
        ({"sink_tokens": 0,  "window": 4, "head_dim": 0}, "head_dim"),
    ])
    def test_invalid_params(self, kwargs, exc):
        with pytest.raises(ValueError, match=exc):
            StreamingKVWindow(**kwargs)

    def test_single_token_window(self):
        w = StreamingKVWindow(sink_tokens=0, window=1, head_dim=2)
        w.push(0, np.array([1., 2.]), np.zeros(2))
        w.push(1, np.array([3., 4.]), np.zeros(2))
        keys, _ = w.get_kv()
        assert keys.shape[0] == 1

    def test_len_with_sinks_and_recent(self):
        w = StreamingKVWindow(sink_tokens=3, window=2, head_dim=5)
        for i in range(6):
            w.push(i, np.zeros(5), np.zeros(5))
        # 3 sinks + min(window=2, recent pushed=3) = 3+2=5
        assert len(w) == 5


# ---------------------------------------------------------------------------
# DuoKVManager
# ---------------------------------------------------------------------------

class TestDuoKVManager:
    def _make_cfg(self):
        return DuoAttentionConfig(num_layers=2, num_heads=2, head_dim=4,
                                   sink_tokens=1, local_window=2,
                                   retrieval_threshold=0.1,
                                   min_retrieval_fraction=0.0)

    def test_default_all_retrieval(self):
        cfg = self._make_cfg()
        mgr = DuoKVManager(cfg)  # no labels → all retrieval
        k = np.array([1., 2., 3., 4.])
        v = np.array([5., 6., 7., 8.])
        mgr.store_kv(0, 0, 0, k, v)
        mgr.store_kv(0, 0, 1, k * 2, v * 2)
        keys, vals = mgr.load_kv(0, 0)
        assert keys.shape == (2, 4)
        assert vals.shape == (2, 4)

    def test_streaming_head_uses_window(self):
        cfg = self._make_cfg()
        labels = {(0, 1): "streaming", (0, 0): "retrieval",
                  (1, 0): "retrieval", (1, 1): "streaming"}
        mgr = DuoKVManager(cfg, labels)
        k = np.ones(4)
        v = np.ones(4)
        for i in range(10):
            mgr.store_kv(0, 1, i, k * i, v * i)
        keys, vals = mgr.load_kv(0, 1)
        # sink=1 + window=2 = 3 max
        assert keys.shape[0] <= 3

    def test_load_kv_empty_retrieval(self):
        cfg = self._make_cfg()
        mgr = DuoKVManager(cfg)
        keys, vals = mgr.load_kv(0, 0)
        assert keys.shape == (0, 4)
        assert vals.shape == (0, 4)

    def test_load_kv_empty_streaming(self):
        cfg = self._make_cfg()
        labels = {(0, 0): "streaming", (0, 1): "streaming",
                  (1, 0): "streaming", (1, 1): "streaming"}
        mgr = DuoKVManager(cfg, labels)
        keys, vals = mgr.load_kv(0, 0)
        assert keys.shape == (0, 4)

    def test_cache_size_tokens(self):
        cfg = self._make_cfg()
        labels = {(0, 0): "retrieval", (0, 1): "streaming",
                  (1, 0): "retrieval", (1, 1): "streaming"}
        mgr = DuoKVManager(cfg, labels)
        k = np.ones(4)
        v = np.ones(4)
        for i in range(5):
            mgr.store_kv(0, 0, i, k, v)   # retrieval
        for i in range(5):
            mgr.store_kv(0, 1, i, k, v)   # streaming
        sizes = mgr.cache_size_tokens()
        assert sizes["retrieval"] == 5
        assert sizes["streaming"] >= 1

    def test_clear(self):
        cfg = self._make_cfg()
        mgr = DuoKVManager(cfg)
        k   = np.ones(4)
        mgr.store_kv(0, 0, 0, k, k)
        mgr.clear()
        keys, _ = mgr.load_kv(0, 0)
        assert keys.shape[0] == 0
