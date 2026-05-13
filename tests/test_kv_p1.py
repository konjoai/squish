"""tests/test_kv_p1.py — P1 sprint: attention sink, precision_map,
CompressionResult, HF SquishCache.

Four test classes, one per feature.
"""
from __future__ import annotations

import numpy as np
import pytest

from squish.kv.kv_cache import (
    CompressionResult,
    HadamardKVCache,
    KVLayerCache,
    QuantizedKVCache,
    _parse_precision_map,
    make_kv_cache,
)
from squish.integrations.hf import SquishCache, squish_compress


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(2026)


def _kv(n_heads: int = 4, head_dim: int = 64, dtype=np.float16) -> tuple:
    k = RNG.standard_normal((n_heads, head_dim)).astype(dtype)
    v = RNG.standard_normal((n_heads, head_dim)).astype(dtype)
    return k, v


def _push(cache: QuantizedKVCache, layer: int, n: int,
          n_heads: int = 4, head_dim: int = 64) -> None:
    for _ in range(n):
        k, v = _kv(n_heads, head_dim)
        cache.update(layer, k, v)


# ---------------------------------------------------------------------------
# 1. Attention sink preservation
# ---------------------------------------------------------------------------


class TestAttentionSink:
    def test_sink_tokens_bypass_window(self):
        """First sink_token_count tokens go directly to keys_sink."""
        cache = QuantizedKVCache(n_layers=1, window=4, mode="int8",
                                 sink_token_count=3)
        _push(cache, 0, 10)
        layer = cache._layers[0]
        assert len(layer.keys_sink) == 3
        assert len(layer.values_sink) == 3

    def test_sink_tokens_are_float16(self):
        """Sink tokens stay at FP16 — never quantized."""
        cache = QuantizedKVCache(n_layers=1, window=2, mode="int2",
                                 sink_token_count=4)
        _push(cache, 0, 10)
        layer = cache._layers[0]
        assert all(a.dtype == np.float16 for a in layer.keys_sink)

    def test_n_tokens_includes_sink(self):
        """n_tokens = n_sink + n_compressed + n_recent."""
        cache = QuantizedKVCache(n_layers=1, window=4, mode="int8",
                                 sink_token_count=3)
        _push(cache, 0, 10)
        assert cache._layers[0].n_tokens == 10

    def test_get_full_kv_sink_first(self):
        """get_full_kv() prepends sink tokens before compressed old tier."""
        cache = QuantizedKVCache(n_layers=1, window=4, mode="int8",
                                 sink_token_count=2)
        _push(cache, 0, 8)
        layer = cache._layers[0]
        k_full, v_full = layer.get_full_kv()
        assert k_full is not None
        assert k_full.shape[1] == 8        # all 8 tokens reconstructed
        # First 2 (sink) must match the original sink tokens exactly.
        for i in range(2):
            np.testing.assert_array_equal(
                k_full[:, i, :], layer.keys_sink[i]
            )

    def test_sink_count_zero_no_change(self):
        """sink_token_count=0 (default) produces identical behaviour to old code."""
        c0 = QuantizedKVCache(n_layers=1, window=4, mode="int8",
                              sink_token_count=0)
        c1 = QuantizedKVCache(n_layers=1, window=4, mode="int8")
        for _ in range(8):
            k, v = _kv()
            c0.update(0, k, v)
            c1.update(0, k, v)
        k0, _ = c0._layers[0].get_full_kv()
        k1, _ = c1._layers[0].get_full_kv()
        assert k0.shape == k1.shape

    def test_sink_larger_than_window(self):
        """sink_token_count > window is valid; sink fills first, then window."""
        cache = QuantizedKVCache(n_layers=1, window=2, mode="int8",
                                 sink_token_count=8)
        _push(cache, 0, 12)
        layer = cache._layers[0]
        assert len(layer.keys_sink) == 8
        assert layer.n_tokens == 12

    def test_sink_preserved_after_reset(self):
        """reset() clears the sink buffer but preserves sink_count config."""
        cache = QuantizedKVCache(n_layers=1, window=4, mode="int8",
                                 sink_token_count=3)
        _push(cache, 0, 10)
        cache._layers[0].reset()
        assert len(cache._layers[0].keys_sink) == 0
        assert cache._layers[0]._sink_count == 3   # config preserved

    def test_hadamard_kv_cache_inherits_sink(self):
        """HadamardKVCache forwards sink_token_count to its layers."""
        cache = HadamardKVCache(n_layers=2, window=4, mode="int4",
                                sink_token_count=2)
        _push(cache, 0, 8)
        _push(cache, 1, 8)
        assert cache._layers[0]._sink_count == 2
        assert len(cache._layers[0].keys_sink) == 2

    def test_make_kv_cache_sink_kwarg(self):
        """make_kv_cache() accepts sink_token_count via **extra."""
        cache = make_kv_cache(n_layers=4, planned_context=4096,
                              sink_token_count=4)
        assert cache._layers[0]._sink_count == 4

    def test_negative_sink_count_raises(self):
        with pytest.raises(ValueError, match="≥ 0"):
            QuantizedKVCache(n_layers=1, window=4, mode="int8",
                             sink_token_count=-1)


# ---------------------------------------------------------------------------
# 2. Mixed-precision per-layer API (precision_map)
# ---------------------------------------------------------------------------


class TestPrecisionMap:
    def test_parse_range_spec(self):
        modes = _parse_precision_map({"0-3": "fp16", "4-7": "int4"}, 8)
        assert modes[:4]  == ["fp16"] * 4
        assert modes[4:]  == ["int4"] * 4

    def test_parse_single_index(self):
        modes = _parse_precision_map({"5": "int2"}, 8)
        assert modes[5] == "int2"
        assert all(m is None for i, m in enumerate(modes) if i != 5)

    def test_parse_out_of_range_ignored(self):
        """Layer indices outside [0, n_layers) are silently ignored."""
        modes = _parse_precision_map({"0-3": "int2", "100": "fp16"}, 4)
        assert len(modes) == 4

    def test_parse_rejects_unknown_mode(self):
        with pytest.raises(ValueError, match="not valid"):
            _parse_precision_map({"0": "int3"}, 4)

    def test_parse_rejects_inverted_range(self):
        with pytest.raises(ValueError, match="lo > hi"):
            _parse_precision_map({"5-2": "int4"}, 8)

    def test_fp16_layers_do_not_quantize(self):
        """Layers with kv_mode='fp16' accumulate evicted tokens at FP16."""
        cache = QuantizedKVCache(n_layers=2, window=2, mode="int8",
                                 precision_map={"0": "fp16"})
        _push(cache, 0, 8)
        layer = cache._layers[0]
        assert layer._kv_mode == "fp16"
        # No quantized buffer — tokens went to fp16 accumulator
        assert layer.keys_old_q is None
        assert layer._fp16_old_k is not None

    def test_fp16_layer_full_kv_correct_shape(self):
        """get_full_kv() reconstructs fp16-mode layers at correct shape."""
        cache = QuantizedKVCache(n_layers=1, window=2, mode="int8",
                                 precision_map={"0": "fp16"})
        _push(cache, 0, 8)
        k, v = cache._layers[0].get_full_kv()
        assert k.shape[1] == 8
        assert k.dtype == np.float16

    def test_mixed_precision_per_layer(self):
        """Different layers quantize at different precisions."""
        cache = QuantizedKVCache(
            n_layers=4, window=2, mode="int8",
            precision_map={"0-1": "int2", "2-3": "int4"},
        )
        _push(cache, 0, 8); _push(cache, 1, 8)
        _push(cache, 2, 8); _push(cache, 3, 8)
        assert cache._layers[0]._kv_mode == "int2"
        assert cache._layers[1]._kv_mode == "int2"
        assert cache._layers[2]._kv_mode == "int4"
        assert cache._layers[3]._kv_mode == "int4"
        # INT2 packs head_dim/4 per row
        assert cache._layers[0].keys_old_q.shape[2] == 64 // 4
        assert cache._layers[2].keys_old_q.shape[2] == 64 // 2

    def test_uncovered_layers_use_cache_mode(self):
        """Layers not in precision_map fall back to the cache-level mode."""
        cache = QuantizedKVCache(n_layers=4, window=2, mode="int4",
                                 precision_map={"0": "fp16"})
        assert cache._layers[0]._kv_mode == "fp16"
        assert cache._layers[1]._kv_mode == "int4"
        assert cache._layers[3]._kv_mode == "int4"

    def test_precision_map_with_hadamard(self):
        """HadamardKVCache passes precision_map through to its layers."""
        cache = HadamardKVCache(n_layers=4, window=2, mode="int8",
                                precision_map={"0-1": "int2"})
        assert cache._layers[0]._kv_mode == "int2"
        assert cache._layers[2]._kv_mode == "int8"


# ---------------------------------------------------------------------------
# 3. CompressionResult metrics
# ---------------------------------------------------------------------------


class TestCompressionResult:
    def test_returns_compression_result_instance(self):
        cache = QuantizedKVCache(n_layers=2, window=4, mode="int8")
        _push(cache, 0, 10)
        _push(cache, 1, 10)
        m = cache.metrics()
        assert isinstance(m, CompressionResult)

    def test_tokens_compressed_count(self):
        """tokens_compressed equals tokens that entered the quantized old tier."""
        cache = QuantizedKVCache(n_layers=1, window=4, mode="int8")
        _push(cache, 0, 10)
        m = cache.metrics()
        # 10 tokens, window=4 → 6 evicted to old tier
        assert m.tokens_compressed == 6

    def test_tokens_fp16_count(self):
        """tokens_fp16 = recent_window + sink."""
        cache = QuantizedKVCache(n_layers=1, window=4, mode="int8",
                                 sink_token_count=2)
        _push(cache, 0, 10)
        m = cache.metrics()
        # sink=2, recent=4 → 6 FP16 tokens
        assert m.tokens_fp16 == 6

    def test_bits_used_scales_with_mode(self):
        """INT2 uses fewer bits than INT8 for the same token count."""
        c8 = QuantizedKVCache(n_layers=1, window=2, mode="int8")
        c2 = QuantizedKVCache(n_layers=1, window=2, mode="int2")
        _push(c8, 0, 8)
        _push(c2, 0, 8)
        assert c8.metrics().bits_used > c2.metrics().bits_used > 0

    def test_memory_saved_positive_for_int2(self):
        """INT2 saves more bytes than FP16 baseline."""
        cache = QuantizedKVCache(n_layers=1, window=2, mode="int2")
        _push(cache, 0, 20)
        m = cache.metrics()
        assert m.memory_saved_bytes > 0

    def test_no_savings_for_fp16_layers(self):
        """Layers with kv_mode='fp16' contribute 0 bits to the compressed tier."""
        cache = QuantizedKVCache(n_layers=1, window=2,
                                 precision_map={"0": "fp16"})
        _push(cache, 0, 8)
        m = cache.metrics()
        assert m.tokens_compressed == 0
        assert m.bits_used == 0

    def test_layer_breakdown_length(self):
        """layers list has one entry per layer."""
        cache = QuantizedKVCache(n_layers=3, window=4, mode="int8")
        m = cache.metrics()
        assert len(m.layers) == 3

    def test_layer_breakdown_values(self):
        """Per-layer breakdown (compressed, fp16, bits) sums to totals."""
        cache = QuantizedKVCache(n_layers=2, window=2, mode="int8")
        _push(cache, 0, 6)
        _push(cache, 1, 6)
        m = cache.metrics()
        total_comp = sum(row[0] for row in m.layers)
        total_fp16 = sum(row[1] for row in m.layers)
        total_bits = sum(row[2] for row in m.layers)
        assert total_comp == m.tokens_compressed
        assert total_fp16 == m.tokens_fp16
        assert total_bits == m.bits_used

    def test_reset_clears_counters(self):
        """reset() zeros the observability counters."""
        cache = QuantizedKVCache(n_layers=1, window=2, mode="int8")
        _push(cache, 0, 8)
        cache.reset()
        m = cache.metrics()
        assert m.tokens_compressed == 0
        assert m.tokens_fp16 == 0

    def test_compression_result_str(self):
        m = CompressionResult(tokens_compressed=100, tokens_fp16=20,
                              bits_used=6400, memory_saved_bytes=1024)
        s = str(m)
        assert "100" in s
        assert "MB" in s


# ---------------------------------------------------------------------------
# 4. HuggingFace SquishCache
# ---------------------------------------------------------------------------


class TestSquishCache:
    def _make_kv(self, batch=1, n_heads=2, seq=4, head_dim=64):
        k = RNG.standard_normal((batch, n_heads, seq, head_dim)).astype(np.float16)
        v = RNG.standard_normal((batch, n_heads, seq, head_dim)).astype(np.float16)
        return k, v

    def test_update_returns_correct_shape(self):
        """update() returns (batch, n_heads, T_total, head_dim) numpy arrays."""
        sc = SquishCache(quantization="int8", sink_token_count=0)
        k, v = self._make_kv()
        k2, v2 = sc.update(k, v, layer_idx=0)
        assert k2.shape == k.shape
        assert v2.shape == v.shape

    def test_seq_length_grows(self):
        """get_seq_length() returns cumulative token count."""
        sc = SquishCache(quantization="int8")
        k, v = self._make_kv(seq=4)
        sc.update(k, v, layer_idx=0)
        sc.update(k, v, layer_idx=0)
        assert sc.get_seq_length(0) == 8

    def test_accumulated_output_shape(self):
        """After 3 updates of 4 tokens each, output has 12 tokens."""
        sc = SquishCache(quantization="int8")
        k, v = self._make_kv(seq=4)
        for _ in range(3):
            k_out, v_out = sc.update(k, v, layer_idx=0)
        assert k_out.shape[2] == 12

    def test_sink_preserved_in_squish_cache(self):
        """SquishCache respects sink_token_count."""
        sc = SquishCache(quantization="int8", window=2, sink_token_count=3)
        k, v = self._make_kv(seq=8)
        sc.update(k, v, layer_idx=0)
        assert sc._squish_cache._layers[0]._sink_count == 3

    def test_metrics_returns_compression_result(self):
        sc = SquishCache(quantization="int4")
        k, v = self._make_kv(seq=4)
        sc.update(k, v, layer_idx=0)
        m = sc.metrics()
        assert isinstance(m, CompressionResult)

    def test_metrics_none_before_first_update(self):
        sc = SquishCache()
        assert sc.metrics() is None

    def test_reset_clears_cache(self):
        sc = SquishCache(quantization="int8")
        k, v = self._make_kv(seq=4)
        sc.update(k, v, layer_idx=0)
        sc.reset()
        assert sc.get_seq_length(0) == 0

    def test_invalid_quantization_raises(self):
        with pytest.raises(ValueError, match="quantization"):
            SquishCache(quantization="int3")

    def test_multi_layer_update(self):
        """Each layer_idx maps to a separate slot in the inner cache."""
        sc = SquishCache(quantization="int8")
        k, v = self._make_kv(seq=4)
        sc.update(k, v, layer_idx=0)
        sc.update(k, v, layer_idx=1)
        assert sc.get_seq_length(0) == 4
        assert sc.get_seq_length(1) == 4

    def test_batch_size_not_one_raises(self):
        sc = SquishCache(quantization="int8")
        k = RNG.standard_normal((2, 2, 4, 64)).astype(np.float16)
        v = k.copy()
        with pytest.raises(ValueError, match="batch_size=1"):
            sc.update(k, v, layer_idx=0)

    def test_get_usable_length(self):
        sc = SquishCache(quantization="int8")
        k, v = self._make_kv(seq=4)
        sc.update(k, v, layer_idx=0)
        assert sc.get_usable_length(4, layer_idx=0) == 4

    def test_squish_compress_decorator(self):
        """@squish_compress patches the returned object's forward method."""
        class FakeModel:
            def forward(self, x, past_key_values=None, use_cache=True):
                return past_key_values

        @squish_compress(quantization="int8")
        def factory():
            return FakeModel()

        model = factory()
        result = model.forward("dummy")
        assert isinstance(result, SquishCache)
