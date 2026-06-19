"""Behavioral coverage for ``squish.integrations.hf`` — the HuggingFace
DynamicCache bridge (SquishCache), the squish_compress decorator / _patch_model
monkey-patch, and the _to_numpy / _restore_type tensor converters.

SquishCache is driven with numpy K/V tensors (int8 mode is numpy-native, no MLX);
torch/mlx converter branches use injected fakes / importorskip — host-agnostic.
"""
from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from squish.integrations import hf
from squish.integrations.hf import (
    SquishCache,
    _restore_type,
    _to_numpy,
    squish_compress,
)


def _kv(n_heads=2, seq=3, head_dim=8):
    k = np.ones((1, n_heads, seq, head_dim), dtype=np.float32)
    v = np.full((1, n_heads, seq, head_dim), 2.0, dtype=np.float32)
    return k, v


# ── SquishCache construction ────────────────────────────────────────────────


def test_invalid_quantization_raises():
    with pytest.raises(ValueError, match="quantization must be one of"):
        SquishCache(quantization="int7")


def test_update_single_layer_numpy_roundtrip():
    c = SquishCache(quantization="int8", rotate=False, window=4)
    k, v = _kv()
    fk, fv = c.update(k, v, layer_idx=0)
    assert fk.shape == (1, 2, 3, 8) and fv.shape == (1, 2, 3, 8)
    assert c.get_seq_length(0) == 3


def test_update_rejects_batch_gt_one():
    c = SquishCache(quantization="int8", rotate=False)
    k = np.ones((2, 2, 3, 8), dtype=np.float32)
    with pytest.raises(ValueError, match="batch_size=1"):
        c.update(k, k, layer_idx=0)


def test_update_extends_layers_on_higher_index():
    c = SquishCache(quantization="int8", rotate=False, window=4)
    k, v = _kv()
    c.update(k, v, layer_idx=0)            # builds cache with 1 layer
    c.update(k, v, layer_idx=3)            # 3 >= len → extend to 4 layers
    assert c.get_seq_length(3) == 3
    assert len(c._squish_cache._layers) == 4


def test_update_rotate_uses_hadamard_cache():
    c = SquishCache(quantization="int8", rotate=True, window=4, seed=7)
    k, v = _kv()
    fk, _ = c.update(k, v, layer_idx=0)
    assert fk.shape == (1, 2, 3, 8)
    assert type(c._squish_cache).__name__ == "HadamardKVCache"


# ── HF protocol methods ─────────────────────────────────────────────────────


def test_get_seq_length_before_first_update_is_zero():
    assert SquishCache(rotate=False).get_seq_length(0) == 0


def test_get_seq_length_unknown_layer_is_zero():
    c = SquishCache(quantization="int8", rotate=False, window=4)
    k, v = _kv()
    c.update(k, v, layer_idx=0)
    assert c.get_seq_length(99) == 0  # layer beyond what's built


def test_get_usable_length_mirrors_seq_length():
    c = SquishCache(quantization="int8", rotate=False, window=4)
    k, v = _kv()
    c.update(k, v, layer_idx=0)
    assert c.get_usable_length(10, layer_idx=0) == 3


def test_reset_clears_cache():
    c = SquishCache(quantization="int8", rotate=False, window=4)
    k, v = _kv()
    c.update(k, v, layer_idx=0)
    c.reset()
    assert c.get_seq_length(0) == 0


def test_metrics_none_before_update_then_present():
    c = SquishCache(quantization="int8", rotate=False, window=4)
    assert c.metrics() is None
    k, v = _kv()
    c.update(k, v, layer_idx=0)
    assert c.metrics() is not None  # CompressionResult after first token


# ── squish_compress / _patch_model ──────────────────────────────────────────


class _FakeModel:
    _cache_class = None

    def forward(self, **kwargs):
        return kwargs.get("past_key_values")


def test_squish_compress_patches_model_and_injects_cache():
    @squish_compress(quantization="int8", rotate=False)
    def load():
        return _FakeModel()

    model = load()
    assert model._cache_class is SquishCache
    # use_cache + no past_key_values → a fresh SquishCache is injected.
    injected = model.forward(use_cache=True, past_key_values=None)
    assert isinstance(injected, SquishCache)
    # use_cache=False → no injection.
    assert model.forward(use_cache=False, past_key_values=None) is None


def test_patch_model_without_cache_class_attr():
    class _NoCacheClass:
        def forward(self, **kwargs):
            return kwargs.get("past_key_values")

    m = hf._patch_model(_NoCacheClass(), quantization="int8", rotate=False,
                        window=4, sink_token_count=2, precision_map=None, seed=1)
    # No _cache_class attr → that branch skipped; forward still patched + injects.
    out = m.forward(use_cache=True, past_key_values=None)
    assert isinstance(out, SquishCache)


# ── _to_numpy ───────────────────────────────────────────────────────────────


def test_to_numpy_ndarray_passthrough():
    out = _to_numpy(np.ones((2, 2), dtype=np.float32))
    assert out.dtype == np.float16


def test_to_numpy_via_numpy_method():
    class _HasNumpy:
        def numpy(self):
            return np.ones((2, 2), dtype=np.float32)

    assert _to_numpy(_HasNumpy()).dtype == np.float16


def test_to_numpy_via_array_protocol():
    class _HasArray:
        def __array__(self, dtype=None):
            return np.ones((2, 2), dtype=np.float32)

    assert _to_numpy(_HasArray()).dtype == np.float16


def test_to_numpy_rejects_unconvertible():
    with pytest.raises(TypeError, match="Cannot convert"):
        _to_numpy(object())


# ── _restore_type ───────────────────────────────────────────────────────────


def test_restore_type_numpy_matches_ref_dtype():
    ref = np.ones((2, 2), dtype=np.float32)
    out = _restore_type(np.ones((2, 2), dtype=np.float16), ref)
    assert out.dtype == np.float32


def test_restore_type_torch_branch(monkeypatch):
    fake_torch = types.ModuleType("torch")

    class _T:
        def to(self, *a):
            return self

    fake_torch.from_numpy = lambda arr: _T()
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    class Tensor:  # __name__ == "Tensor"
        from_numpy = True
        dtype = "float16"
        device = "cpu"

    out = _restore_type(np.ones((2, 2), np.float16), Tensor())
    assert isinstance(out, _T)


def test_restore_type_mlx_branch_without_mlx(monkeypatch):
    monkeypatch.setitem(sys.modules, "mlx.core", None)  # import fails

    class _MlxRef:
        __module__ = "mlx.core"

    arr = np.ones((2, 2), np.float16)
    # import mlx.core fails → falls through and returns the numpy array (364-366).
    assert _restore_type(arr, _MlxRef()) is arr


def test_restore_type_mlx_branch_with_mlx():
    mx = pytest.importorskip("mlx.core")

    class _MlxRef:
        __module__ = "mlx.core"

    out = _restore_type(np.ones((2, 2), np.float16), _MlxRef())
    assert isinstance(out, mx.array)


def test_restore_type_unknown_ref_returns_array():
    arr = np.ones((2, 2), np.float16)
    assert _restore_type(arr, "some string ref") is arr  # passthrough (366)


# ── Extra branches ──────────────────────────────────────────────────────────


def test_update_existing_layer_skips_extend():
    c = SquishCache(quantization="int8", rotate=False, window=4)
    k, v = _kv()
    c.update(k, v, layer_idx=0)
    c.update(k, v, layer_idx=0)  # layer already exists → skip build/extend (178→192)
    assert c.get_seq_length(0) == 6


def test_update_empty_token_batch_returns_inputs():
    c = SquishCache(quantization="int8", rotate=False, window=4)
    k = np.ones((1, 2, 0, 8), dtype=np.float32)  # T_new == 0 → nothing appended
    v = np.ones((1, 2, 0, 8), dtype=np.float32)
    fk, fv = c.update(k, v, layer_idx=0)
    # get_full_kv is None → the new (empty) states are returned unchanged (199-201).
    assert fk is k and fv is v


def test_reset_before_any_update_is_noop():
    SquishCache(rotate=False).reset()  # _squish_cache is None → 225→exit
