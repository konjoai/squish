"""Coverage for ``squish.context.lazy_llm`` — LazyLLM dynamic token pruning.
The mlx ops are backed by a numpy-shimmed fake ``mlx.core`` (both the ``mlx``
package and ``mlx.core`` are injected — ``import mlx.core as mx`` binds via the
package attribute), so every path runs host-agnostically on macOS + Linux CI.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from squish.context import lazy_llm as ll
from squish.context.lazy_llm import (
    LazyLLMConfig,
    _build_keep_mask,
    _get_layers,
    _PruneState,
    patch_model_lazy_llm,
    unpatch_model_lazy_llm,
)


@pytest.fixture(autouse=True)
def _fake_mlx(monkeypatch):
    """Inject a numpy-backed fake mlx.core (+ mlx package)."""
    core = types.ModuleType("mlx.core")
    core.float32 = np.float32
    core.array = lambda data, dtype=None: np.asarray(
        data, dtype=dtype if dtype is not None else None
    )
    core.sqrt = np.sqrt
    core.sum = np.sum
    core.eval = lambda *a, **k: None
    pkg = types.ModuleType("mlx")
    pkg.core = core
    monkeypatch.setitem(sys.modules, "mlx", pkg)
    monkeypatch.setitem(sys.modules, "mlx.core", core)
    return core


# ── LazyLLMConfig ────────────────────────────────────────────────────────────


def test_config_defaults():
    c = LazyLLMConfig()
    assert c.keep_ratio == 0.70 and c.start_layer == 2 and c.revive_window == 4


@pytest.mark.parametrize(
    "kw,msg",
    [
        ({"keep_ratio": 0.0}, "keep_ratio"),
        ({"keep_ratio": 1.5}, "keep_ratio"),
        ({"start_layer": -1}, "start_layer"),
        ({"revive_window": -1}, "revive_window"),
    ],
)
def test_config_validation(kw, msg):
    with pytest.raises(ValueError, match=msg):
        LazyLLMConfig(**kw)


def test_prune_state_starts_inactive():
    assert _PruneState().active_mask is None


# ── _importance_scores / _apply_mask_to_hidden ───────────────────────────────


def test_importance_scores_l2_norm():
    hidden = np.ones((1, 3, 4), dtype=np.float32)  # each token norm = 2.0
    scores = ll._importance_scores(hidden)
    assert scores.shape == (3,)
    assert np.allclose(scores, 2.0)


def test_apply_mask_zeros_pruned_positions():
    hidden = np.ones((1, 3, 2), dtype=np.float32)
    mask = np.array([True, False, True])
    out = ll._apply_mask_to_hidden(hidden, mask)
    assert out[0, 1].tolist() == [0.0, 0.0] and out[0, 0].tolist() == [1.0, 1.0]


# ── _build_keep_mask ─────────────────────────────────────────────────────────


def test_build_keep_mask_ranks_and_revives():
    scores = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7], dtype=np.float32)
    mask = _build_keep_mask(scores, keep_ratio=0.5, revive_window=2)
    assert mask[-1] and mask[-2]  # revive window kept
    assert mask[1]  # highest non-revive score kept


def test_build_keep_mask_no_revive_window():
    scores = np.array([0.1, 0.9, 0.2, 0.8], dtype=np.float32)
    mask = _build_keep_mask(scores, keep_ratio=0.5, revive_window=0)
    assert mask.sum() >= 1


def test_build_keep_mask_revive_covers_budget():
    # revive_window >= T → n_rank_keep == 0 and n_non_revive == 0 (ranking skipped)
    scores = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    mask = _build_keep_mask(scores, keep_ratio=0.5, revive_window=4)
    assert mask.all()


# ── _LazyLLMLayerWrapper ─────────────────────────────────────────────────────


def _wrapper(orig, idx=3, cfg=None, state=None):
    return ll._LazyLLMLayerWrapper(orig, idx, cfg or LazyLLMConfig(), state or _PruneState())


def test_wrapper_passthrough_on_decode():
    sentinel = np.ones((1, 1, 4), dtype=np.float32)  # T==1 → pass-through
    w = _wrapper(lambda x, *a, **k: ("decoded", x))
    out = w(sentinel)
    assert out[0] == "decoded" and out[1] is sentinel


def test_wrapper_prefill_updates_mask_verbose(capsys):
    state = _PruneState()
    cfg = LazyLLMConfig(start_layer=0, keep_ratio=0.5, revive_window=1, verbose=True)
    w = _wrapper(lambda x, *a, **k: x, idx=0, cfg=cfg, state=state)
    out = w(np.ones((1, 6, 4), dtype=np.float32))
    assert out.shape == (1, 6, 4)
    assert state.active_mask is not None
    assert "[lazy_llm]" in capsys.readouterr().out


def test_wrapper_applies_existing_mask_and_tuple_return():
    state = _PruneState()
    state.active_mask = np.array([True, False, True, True, True, True])
    cfg = LazyLLMConfig(start_layer=0, verbose=False)
    # layer returns a tuple (hidden, kv) → wrapper extracts hidden for scoring
    w = _wrapper(lambda x, *a, **k: (x, "kv"), idx=1, cfg=cfg, state=state)
    out = w(np.ones((1, 6, 4), dtype=np.float32))
    assert isinstance(out, tuple) and out[1] == "kv"


def test_wrapper_prefill_below_start_layer_skips_scoring():
    # idx < start_layer → mask update skipped (226 false branch)
    state = _PruneState()
    cfg = LazyLLMConfig(start_layer=5)
    w = _wrapper(lambda x, *a, **k: x, idx=0, cfg=cfg, state=state)
    w(np.ones((1, 6, 4), dtype=np.float32))
    assert state.active_mask is None


def test_wrapper_skips_scoring_when_output_collapses_to_one_token():
    # x is prefill (T=6) but the layer returns T==1 → hidden.shape[1] > 1 is false.
    state = _PruneState()
    cfg = LazyLLMConfig(start_layer=0)
    w = _wrapper(lambda x, *a, **k: np.ones((1, 1, 4), np.float32), idx=0, cfg=cfg, state=state)
    out = w(np.ones((1, 6, 4), dtype=np.float32))
    assert out.shape == (1, 1, 4) and state.active_mask is None


def test_wrapper_getattr_delegates():
    orig = types.SimpleNamespace(weight="W")
    w = _wrapper(orig)
    assert w.weight == "W"


# ── _get_layers / patch / unpatch ────────────────────────────────────────────


def test_get_layers_variants():
    nested = types.SimpleNamespace(model=types.SimpleNamespace(layers=[1, 2]))
    assert _get_layers(nested) == [1, 2]
    direct = types.SimpleNamespace(layers=[3])
    assert _get_layers(direct) == [3]
    assert _get_layers(types.SimpleNamespace()) is None


def test_patch_incompatible_model_returns_none(caplog):
    assert patch_model_lazy_llm(types.SimpleNamespace()) is None


def test_patch_and_unpatch_nested_layout():
    inner = types.SimpleNamespace(layers=[object(), object(), object(), object()])
    model = types.SimpleNamespace(model=inner)
    state = patch_model_lazy_llm(model, LazyLLMConfig(start_layer=2))
    assert state is not None
    # layers 0,1 untouched; 2,3 wrapped
    assert isinstance(model.model.layers[2], ll._LazyLLMLayerWrapper)
    assert not isinstance(model.model.layers[0], ll._LazyLLMLayerWrapper)
    unpatch_model_lazy_llm(model)
    assert all(not isinstance(x, ll._LazyLLMLayerWrapper) for x in model.model.layers)


def test_patch_direct_layout_default_config():
    model = types.SimpleNamespace(layers=[object(), object(), object()])
    state = patch_model_lazy_llm(model)  # config=None → default (start_layer=2)
    assert state is not None
    assert isinstance(model.layers[2], ll._LazyLLMLayerWrapper)
    unpatch_model_lazy_llm(model)
    assert all(not isinstance(x, ll._LazyLLMLayerWrapper) for x in model.layers)


def test_unpatch_unpatched_model_is_noop():
    unpatch_model_lazy_llm(types.SimpleNamespace())  # must not raise


def test_unpatch_when_layers_attr_absent():
    # Patched marker present but neither model.layers nor .layers → just deletes
    # the stashed attrs (the elif-false branch).
    model = types.SimpleNamespace()
    model._lazy_llm_state = _PruneState()
    model._lazy_llm_orig = [object()]
    unpatch_model_lazy_llm(model)
    assert not hasattr(model, "_lazy_llm_orig") and not hasattr(model, "_lazy_llm_state")
