"""Regression: SinkStats.util_fraction must divide by window size, not itself.

The formula was n_tokens_seen / max(1, n_tokens_seen), which is 1.0 for any
nonzero count regardless of window size — so a barely-used cache reported 100%
utilization.
"""
from __future__ import annotations

from squish.streaming.streaming_sink import SinkConfig, SinkKVCache, SinkStats


def test_partial_window_reports_real_fraction():
    assert SinkStats(n_tokens_seen=5, window_size=256).util_fraction == 5 / 256


def test_zero_tokens_is_zero():
    assert SinkStats(n_tokens_seen=0, window_size=256).util_fraction == 0.0


def test_overfull_is_clamped_to_one():
    assert SinkStats(n_tokens_seen=512, window_size=256).util_fraction == 1.0


def test_get_stats_populates_window_size():
    cache = SinkKVCache(SinkConfig(n_sink_tokens=4, window_size=128), n_heads=2, head_dim=8)
    stats = cache.get_stats()
    assert stats.window_size == 128
    # A fresh cache has seen no tokens → 0 utilization (not 1.0).
    assert stats.util_fraction == 0.0
