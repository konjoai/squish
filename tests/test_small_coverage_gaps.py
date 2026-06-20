"""Close small (1-6 line) coverage gaps across several pure-Python/numpy modules:
nf4_quant + head_importance validation guards, hqq n_levels, streaming_sink
config property, router truncate + default-singleton, and the SplitInfo
convenience properties. All host-agnostic (no MLX/Metal).
"""

from __future__ import annotations

import numpy as np
import pytest


# ── nf4_quant validation ─────────────────────────────────────────────────────


def test_nf4_validate_rejects_bad_group_size():
    from squish.quant.nf4_quant import _validate_2d

    with pytest.raises(ValueError, match="group_size must be"):
        _validate_2d(np.zeros((4, 8), np.float32), group_size=0)


# ── hqq n_levels ─────────────────────────────────────────────────────────────


def test_hqq_config_n_levels():
    from squish.quant.hqq import HQQConfig

    assert HQQConfig(bits=4.0).n_levels == 16


# ── streaming_sink config property ───────────────────────────────────────────


def test_sink_cache_exposes_config():
    from squish.streaming.streaming_sink import SinkConfig, SinkKVCache

    cfg = SinkConfig(n_sink_tokens=2, window_size=8)
    cache = SinkKVCache(cfg, n_heads=2, head_dim=4)
    assert cache.config is cfg


# ── router truncate + default singleton ──────────────────────────────────────


def test_router_truncate_long_text():
    from squish.serving.router import _truncate

    assert _truncate("abcdefgh", max_len=5) == "abcd…"
    assert _truncate("short", max_len=50) == "short"


def test_get_default_router_is_cached_singleton():
    from squish.serving import router as r

    r._default_router_singleton = None  # reset for a deterministic first call
    first = r.get_default_router()
    second = r.get_default_router()  # second call hits the cached branch
    assert first is second


# ── head_importance validation guards ────────────────────────────────────────


def test_head_importance_validation_raises():
    from squish.kv.head_importance import _validate_layer_samples

    with pytest.raises(ValueError, match="at least one layer"):
        _validate_layer_samples([], None)
    with pytest.raises(ValueError, match="no sample tokens"):
        _validate_layer_samples([[]], None)
    with pytest.raises(ValueError, match="must be 2-D"):
        _validate_layer_samples([[np.zeros(3, np.float32)]], None)


# ── SplitInfo convenience properties ─────────────────────────────────────────


def test_split_info_properties_and_str():
    from squish.io.split_loader import SplitInfo

    info = SplitInfo(
        gpu_layers=[0, 1, 2],
        cpu_layers=[3],
        gpu_bytes=2 * 1024**3,
        cpu_bytes=1024**3,
        metal_limit=4 * 1024**3,
    )
    assert info.gpu_gb == 2.0 and info.cpu_gb == 1.0
    assert info.gpu_count == 3 and info.cpu_count == 1
    assert "SplitInfo(" in str(info)
