"""Behavioral coverage for ``squish.experimental.layer_overlap_loader`` — the
threading-based prefetch-ahead layer loader. Workers are joined before assertions
so the suite is deterministic (no sleeps, no flakiness on CI).
"""

from __future__ import annotations

import threading

import pytest

from squish.experimental.layer_overlap_loader import (
    LayerHandle,
    LayerOverlapConfig,
    LayerOverlapLoader,
    LayerOverlapStats,
)


def _good(idx):
    return {"w": idx}


def _drain(loader):
    """Block until every scheduled worker has finished."""
    for t in list(loader._threads):
        t.join(timeout=2.0)


# ── LayerOverlapConfig ───────────────────────────────────────────────────────


def test_config_defaults():
    c = LayerOverlapConfig()
    assert c.prefetch_count == 2 and c.load_timeout_s == 5.0


@pytest.mark.parametrize(
    "kw,msg",
    [
        ({"prefetch_count": 0}, "prefetch_count must be"),
        ({"load_timeout_s": 0.0}, "load_timeout_s must be"),
    ],
)
def test_config_validation(kw, msg):
    with pytest.raises(ValueError, match=msg):
        LayerOverlapConfig(**kw)


# ── LayerHandle ──────────────────────────────────────────────────────────────


def test_layer_handle_wait_and_repr():
    h = LayerHandle(3)
    assert h.wait(timeout=0.01) is False and "ready=False" in repr(h)
    h.ready.set()
    assert h.wait(timeout=0.01) is True and "ready=True" in repr(h)


# ── LayerOverlapStats ────────────────────────────────────────────────────────


def test_stats_zero_state():
    s = LayerOverlapStats()
    assert s.hit_rate == 0.0 and s.mean_load_ms == 0.0


def test_stats_nonzero():
    s = LayerOverlapStats(
        prefetch_hits=3, prefetch_misses=1, total_layers_loaded=4, total_load_ms=8.0
    )
    assert s.hit_rate == 0.75 and s.mean_load_ms == 2.0
    assert "hit_rate=75.00%" in repr(s)


# ── Loader: lifecycle + happy path ───────────────────────────────────────────


def test_loader_default_config():
    assert LayerOverlapLoader()._cfg.prefetch_count == 2


def test_sequential_iteration_hits_and_evicts():
    loader = LayerOverlapLoader(LayerOverlapConfig(prefetch_count=2))
    loader.start(n_layers=6, load_fn=_good)
    for i in range(6):
        _drain(loader)
        w = loader.get_layer(i)  # ready → hit
        assert w["w"] == i
        loader.prefetch_next(i)  # reschedules / out-of-range near the end
    # Early layers evicted (k < layer_idx - 1) → cache stays small.
    assert loader.cached_layer_count <= 3
    assert loader.stats.prefetch_hits >= 1
    assert loader.stats.mean_load_ms >= 0.0 and loader.stats.hit_rate > 0.0
    assert "n_layers=6" in repr(loader)
    loader.stop()
    assert loader.cached_layer_count == 0


def test_get_layer_scheduling_miss_then_synchronous_load():
    loader = LayerOverlapLoader(LayerOverlapConfig(prefetch_count=1))
    loader.start(n_layers=8, load_fn=_good)
    _drain(loader)
    # Layer 5 was never prefetched → handle is None → miss + synchronous schedule.
    w = loader.get_layer(5)
    assert w["w"] == 5 and loader.stats.prefetch_misses >= 1
    loader.stop()


def test_get_layer_waits_when_not_ready():
    gate = threading.Event()

    def _blocked(idx):
        gate.wait(2.0)
        return {"w": idx}

    loader = LayerOverlapLoader(LayerOverlapConfig(prefetch_count=1, load_timeout_s=0.05))
    loader.start(n_layers=4, load_fn=_blocked)
    # Worker is blocked on the gate → not ready → get_layer waits out the timeout.
    w = loader.get_layer(0)
    assert w is None and loader.stats.prefetch_misses >= 1
    gate.set()
    loader.stop()


# ── Loader: error + guard branches ───────────────────────────────────────────


def test_get_layer_before_start_raises():
    with pytest.raises(RuntimeError, match="start\\(\\) must be called"):
        LayerOverlapLoader().get_layer(0)


def test_get_layer_out_of_range_raises():
    loader = LayerOverlapLoader()
    loader.start(n_layers=2, load_fn=_good)
    with pytest.raises(ValueError, match="out of range"):
        loader.get_layer(9)
    loader.stop()


def test_get_layer_propagates_worker_error():
    def _bad(idx):
        raise ValueError("boom")

    loader = LayerOverlapLoader(LayerOverlapConfig(prefetch_count=1))
    loader.start(n_layers=2, load_fn=_bad)
    _drain(loader)
    with pytest.raises(RuntimeError, match="load failed"):
        loader.get_layer(0)
    assert loader.stats.total_layers_loaded >= 1
    loader.stop()


def test_schedule_ignores_out_of_range_index():
    loader = LayerOverlapLoader()
    loader.start(n_layers=2, load_fn=_good)
    loader._schedule(-1)  # negative → no-op
    loader._schedule(99)  # beyond n_layers → no-op
    _drain(loader)
    assert -1 not in loader._cache and 99 not in loader._cache
    loader.stop()
