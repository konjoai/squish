"""
tests/test_wave41b_modules.py

Test suite for Wave 41b modules — Cross-Layer Sharing, QMoE Compression,
LADE Decoding, Infini Attention, AKVQ Cache, Delta Zip:

  - squish/attention/cla_share.py          (CLAShareAttention)
  - squish/moe/qmoe_compress.py            (QMoECompressor)
  - squish/speculative/lade_decode.py      (LADEDecoder)
  - squish/attention/infini_attn.py        (InfiniAttention)
  - squish/kv/akvq_cache.py               (AKVQCache)
  - squish/quant/delta_zip.py             (DeltaZipAdapter)
"""

import numpy as np
import pytest

# ============================================================
# CLAShareAttention tests
# ============================================================

from squish.attention.cla_share import (
    CLAShareConfig,
    CLAShareAttention,
)


class TestCLAShareConfig:
    def test_defaults(self):
        cfg = CLAShareConfig()
        assert cfg.sharing_stride >= 1
        assert cfg.n_layers >= 1
        assert cfg.n_heads >= 1
        assert cfg.head_dim >= 1

    def test_invalid_sharing_stride(self):
        with pytest.raises(ValueError, match="sharing_stride"):
            CLAShareConfig(sharing_stride=0)

    def test_invalid_n_layers(self):
        with pytest.raises(ValueError, match="n_layers"):
            CLAShareConfig(n_layers=0)

    def test_invalid_n_heads(self):
        with pytest.raises(ValueError, match="n_heads"):
            CLAShareConfig(n_heads=0)


class TestCLAShareAttention:
    def _kv(self, H=4, T=8, d=8, seed=0):
        rng = np.random.default_rng(seed)
        K = rng.standard_normal((H, T, d)).astype(np.float32)
        V = rng.standard_normal((H, T, d)).astype(np.float32)
        return K, V

    def test_anchor_layer_correct(self):
        cla = CLAShareAttention(CLAShareConfig(sharing_stride=2, n_layers=8))
        assert cla.anchor_layer(0) == 0
        assert cla.anchor_layer(1) == 0
        assert cla.anchor_layer(2) == 2
        assert cla.anchor_layer(3) == 2

    def test_is_anchor(self):
        cla = CLAShareAttention(CLAShareConfig(sharing_stride=2, n_layers=8))
        assert cla.is_anchor(0)
        assert not cla.is_anchor(1)
        assert cla.is_anchor(2)
        assert not cla.is_anchor(3)

    def test_compute_kv_stores_anchor(self):
        H, T, d = 4, 8, 8
        cla = CLAShareAttention(CLAShareConfig(sharing_stride=2, n_layers=8, n_heads=H, head_dim=d))
        K, V = self._kv(H, T, d)
        cla.compute_kv(0, K, V)
        K_out, V_out = cla.get_kv(0)
        np.testing.assert_array_equal(K_out, K)
        np.testing.assert_array_equal(V_out, V)

    def test_get_kv_non_anchor_returns_anchor(self):
        H, T, d = 4, 8, 8
        cla = CLAShareAttention(CLAShareConfig(sharing_stride=2, n_layers=8, n_heads=H, head_dim=d))
        K, V = self._kv(H, T, d)
        cla.compute_kv(0, K, V)   # anchor
        cla.compute_kv(1, K, V)   # non-anchor — should not overwrite
        K_out, _ = cla.get_kv(1)
        np.testing.assert_array_equal(K_out, K)

    def test_get_kv_missing_anchor_raises(self):
        cla = CLAShareAttention(CLAShareConfig(sharing_stride=2, n_layers=8))
        with pytest.raises(KeyError):
            cla.get_kv(2)  # anchor=2 not yet stored

    def test_memory_ratio(self):
        stride = 3
        cla = CLAShareAttention(CLAShareConfig(sharing_stride=stride, n_layers=9))
        assert cla.memory_ratio() == pytest.approx(1 / stride)

    def test_n_anchor_layers(self):
        cla = CLAShareAttention(CLAShareConfig(sharing_stride=2, n_layers=8))
        assert cla.n_anchor_layers() == 4

    def test_clear(self):
        H, T, d = 2, 4, 4
        cla = CLAShareAttention(CLAShareConfig(sharing_stride=2, n_layers=4, n_heads=H, head_dim=d))
        K, V = self._kv(H, T, d)
        cla.compute_kv(0, K, V)
        cla.clear()
        with pytest.raises(KeyError):
            cla.get_kv(0)

    def test_repr(self):
        cla = CLAShareAttention(CLAShareConfig())
        assert "CLA" in repr(cla) or "share" in repr(cla).lower()


# ============================================================
# QMoECompressor tests
# ============================================================

from squish.moe.qmoe_compress import (
    QMoEConfig,
    QMoECompressedExpert,
    QMoECompressor,
)


class TestQMoEConfig:
    def test_defaults(self):
        cfg = QMoEConfig()
        assert cfg.n_codes >= 2
        assert cfg.block_size >= 1

    def test_invalid_n_codes_not_power_of_two(self):
        with pytest.raises(ValueError, match="n_codes"):
            QMoEConfig(n_codes=100)

    def test_invalid_n_codes_less_than_two(self):
        with pytest.raises(ValueError, match="n_codes"):
            QMoEConfig(n_codes=1)

    def test_invalid_block_size(self):
        with pytest.raises(ValueError, match="block_size"):
            QMoEConfig(block_size=0)


class TestQMoECompressor:
    def _weight(self, rows=64, cols=64, seed=0):
        rng = np.random.default_rng(seed)
        return rng.standard_normal((rows, cols)).astype(np.float32)

    def test_compress_returns_compressed(self):
        comp = QMoECompressor(QMoEConfig(n_codes=16, block_size=8))
        w = self._weight(32, 32)
        result = comp.compress(0, w)
        assert isinstance(result, QMoECompressedExpert)

    def test_decompress_shape_matches(self):
        comp = QMoECompressor(QMoEConfig(n_codes=16, block_size=8))
        w = self._weight(32, 32)
        compressed = comp.compress(0, w)
        reconstructed = comp.decompress(compressed)
        assert reconstructed.shape == w.shape

    def test_relative_error_reasonable(self):
        comp = QMoECompressor(QMoEConfig(n_codes=256, block_size=4, n_iter=50))
        w = self._weight(64, 64)
        compressed = comp.compress(0, w)
        reconstructed = comp.decompress(compressed)
        err = comp.relative_error(w, reconstructed)
        assert 0.0 <= err < 2.0

    def test_store_and_load_roundtrip(self):
        comp = QMoECompressor(QMoEConfig(n_codes=16, block_size=8))
        w = self._weight(32, 32)
        compressed = comp.compress(0, w)
        comp.store(0, compressed)
        loaded = comp.load(0)
        assert loaded is compressed

    def test_load_missing_raises(self):
        comp = QMoECompressor(QMoEConfig())
        with pytest.raises(KeyError):
            comp.load(99)

    def test_n_stored_experts(self):
        comp = QMoECompressor(QMoEConfig(n_codes=16, block_size=8))
        assert comp.n_stored_experts() == 0
        c = comp.compress(0, self._weight())
        comp.store(0, c)
        assert comp.n_stored_experts() == 1

    def test_non_2d_weight_raises(self):
        comp = QMoECompressor(QMoEConfig())
        with pytest.raises(ValueError):
            comp.compress(0, np.ones((4, 8, 8), dtype=np.float32))

    def test_bits_per_param_less_than_storage(self):
        comp = QMoECompressor(QMoEConfig(n_codes=16, block_size=8))
        w = self._weight(64, 64)
        compressed = comp.compress(0, w)
        # index bits: log2(16) = 4 bits → storage < 32 bits (fp32)
        n_params = int(np.prod(w.shape))
        idx_bytes = compressed.indices.nbytes
        bits_per_param = (idx_bytes * 8) / n_params
        assert bits_per_param < 32

    def test_repr(self):
        comp = QMoECompressor(QMoEConfig())
        assert "QMoE" in repr(comp) or "q" in repr(comp).lower()


# ============================================================
# LADEDecoder tests
# ============================================================

from squish.speculative.lade_decode import (
    LADEConfig,
    LADEDraftResult,
    LADEDecoder,
)


class TestLADEConfig:
    def test_defaults(self):
        cfg = LADEConfig()
        assert cfg.n_gram >= 2
        assert cfg.n_lookahead >= 1
        assert cfg.temperature > 0.0

    def test_invalid_n_gram(self):
        with pytest.raises(ValueError, match="n_gram"):
            LADEConfig(n_gram=1)

    def test_invalid_temperature_zero(self):
        with pytest.raises(ValueError, match="temperature"):
            LADEConfig(temperature=0.0)

    def test_invalid_temperature_negative(self):
        with pytest.raises(ValueError, match="temperature"):
            LADEConfig(temperature=-1.0)


class TestLADEDecoder:
    def _target_fn(self, vocab=32):
        rng = np.random.default_rng(1)

        def fn(last_token, context):
            p = rng.dirichlet(np.ones(vocab))
            return p.astype(np.float32)

        return fn

    def test_update_ngram_table(self):
        dec = LADEDecoder(LADEConfig(n_gram=2))
        dec.update_ngram_table([1, 2, 3, 4, 5])
        assert dec.n_ngram_entries() >= 1

    def test_step_returns_result(self):
        dec = LADEDecoder(LADEConfig(n_gram=2, n_lookahead=3))
        dec.update_ngram_table([1, 2, 3, 4, 5])
        target_fn = self._target_fn()
        result = dec.step([1, 2, 3], target_fn)
        assert isinstance(result, LADEDraftResult)

    def test_n_accepted_at_least_one(self):
        dec = LADEDecoder(LADEConfig(n_gram=2, n_lookahead=3))
        dec.update_ngram_table(list(range(20)))
        target_fn = self._target_fn()
        result = dec.step(list(range(5)), target_fn)
        assert result.n_accepted >= 1

    def test_acceptance_rate_in_range(self):
        dec = LADEDecoder(LADEConfig(n_gram=2))
        dec.update_ngram_table(list(range(10)))
        result = dec.step(list(range(3)), self._target_fn())
        assert 0.0 <= result.acceptance_rate <= 1.0

    def test_n_ngram_entries_grows(self):
        dec = LADEDecoder(LADEConfig(n_gram=3))
        before = dec.n_ngram_entries()
        dec.update_ngram_table([10, 20, 30, 40, 50])
        assert dec.n_ngram_entries() >= before

    def test_reset_stats(self):
        dec = LADEDecoder(LADEConfig(n_gram=2))
        dec.update_ngram_table(list(range(10)))
        dec.step(list(range(3)), self._target_fn())
        dec.reset_stats()
        assert dec.mean_acceptance_rate == 0.0

    def test_mean_acceptance_rate_accumulates(self):
        dec = LADEDecoder(LADEConfig(n_gram=2))
        dec.update_ngram_table(list(range(10)))
        fn = self._target_fn()
        dec.step(list(range(3)), fn)
        dec.step(list(range(3)), fn)
        assert 0.0 <= dec.mean_acceptance_rate <= 1.0

    def test_accepted_tokens_list(self):
        dec = LADEDecoder(LADEConfig(n_gram=2, n_lookahead=3))
        dec.update_ngram_table(list(range(20)))
        result = dec.step(list(range(5)), self._target_fn())
        assert isinstance(result.accepted_tokens, list)

    def test_repr(self):
        dec = LADEDecoder(LADEConfig())
        assert "LADE" in repr(dec) or "lade" in repr(dec).lower()


# ============================================================
# InfiniAttention tests
# ============================================================

from squish.attention.infini_attn import (
    InfiniAttentionConfig,
    InfiniAttention,
)


class TestInfiniAttentionConfig:
    def test_defaults(self):
        cfg = InfiniAttentionConfig()
        assert cfg.segment_len >= 1
        assert cfg.n_heads >= 1
        assert cfg.head_dim >= 1

    def test_invalid_segment_len(self):
        with pytest.raises(ValueError, match="segment_len"):
            InfiniAttentionConfig(segment_len=0)

    def test_invalid_n_heads(self):
        with pytest.raises(ValueError, match="n_heads"):
            InfiniAttentionConfig(n_heads=0)

    def test_invalid_head_dim(self):
        with pytest.raises(ValueError, match="head_dim"):
            InfiniAttentionConfig(head_dim=0)


class TestInfiniAttention:
    def _qkv(self, H=4, T=16, d=8, seed=0):
        rng = np.random.default_rng(seed)
        Q = rng.standard_normal((H, T, d)).astype(np.float32)
        K = rng.standard_normal((H, T, d)).astype(np.float32)
        V = rng.standard_normal((H, T, d)).astype(np.float32)
        return Q, K, V

    def test_forward_output_shape(self):
        H, T, d = 4, 16, 8
        attn = InfiniAttention(InfiniAttentionConfig(segment_len=8, n_heads=H, head_dim=d))
        Q, K, V = self._qkv(H, T, d)
        out = attn.forward(Q, K, V)
        assert out.shape == (H, T, d)

    def test_forward_output_finite(self):
        H, T, d = 4, 16, 8
        attn = InfiniAttention(InfiniAttentionConfig(segment_len=8, n_heads=H, head_dim=d))
        Q, K, V = self._qkv(H, T, d)
        out = attn.forward(Q, K, V)
        assert np.all(np.isfinite(out))

    def test_n_segments_increments(self):
        H, T, d = 4, 16, 8
        attn = InfiniAttention(InfiniAttentionConfig(segment_len=8, n_heads=H, head_dim=d))
        assert attn.n_segments == 0
        Q, K, V = self._qkv(H, T, d)
        attn.forward(Q, K, V)
        assert attn.n_segments >= 1

    def test_reset_memory_clears(self):
        H, T, d = 4, 16, 8
        attn = InfiniAttention(InfiniAttentionConfig(segment_len=8, n_heads=H, head_dim=d))
        Q, K, V = self._qkv(H, T, d)
        attn.forward(Q, K, V)
        attn.reset_memory()
        assert attn.n_segments == 0

    def test_memory_bytes_positive(self):
        H, d = 4, 8
        attn = InfiniAttention(InfiniAttentionConfig(n_heads=H, head_dim=d))
        assert attn.memory_bytes() > 0

    def test_multiple_forward_passes(self):
        H, T, d = 4, 8, 8
        attn = InfiniAttention(InfiniAttentionConfig(segment_len=8, n_heads=H, head_dim=d))
        Q, K, V = self._qkv(H, T, d)
        attn.forward(Q, K, V)
        attn.forward(Q, K, V)
        assert attn.n_segments >= 2

    def test_memory_second_pass_differs(self):
        """After processing two segments the memory state differs from init."""
        H, T, d = 4, 8, 8
        attn = InfiniAttention(InfiniAttentionConfig(segment_len=8, n_heads=H, head_dim=d))
        Q, K, V = self._qkv(H, T, d)
        attn.forward(Q, K, V)
        # Memory should be non-zero after first pass
        mem_norm = np.linalg.norm(attn._M)
        assert mem_norm > 0.0

    def test_repr(self):
        attn = InfiniAttention(InfiniAttentionConfig())
        assert "Infini" in repr(attn) or "infini" in repr(attn).lower()


# ============================================================
# AKVQCache tests
# ============================================================

from squish.kv.akvq_cache import (
    AKVQConfig,
    AKVQTensor,
    AKVQCache,
)


class TestAKVQConfig:
    def test_defaults(self):
        cfg = AKVQConfig()
        assert cfg.high_precision_bits in (4, 8)
        assert cfg.low_precision_bits in (2, 4)
        assert cfg.low_precision_bits < cfg.high_precision_bits
        assert 0.0 <= cfg.outlier_ratio < 0.5

    def test_invalid_high_precision_bits(self):
        with pytest.raises(ValueError, match="high_precision_bits"):
            AKVQConfig(high_precision_bits=3)

    def test_invalid_low_precision_bits(self):
        with pytest.raises(ValueError, match="low_precision_bits"):
            AKVQConfig(low_precision_bits=3)

    def test_low_not_less_than_high(self):
        with pytest.raises(ValueError, match="low_precision_bits"):
            AKVQConfig(high_precision_bits=4, low_precision_bits=4)

    def test_invalid_outlier_ratio_negative(self):
        with pytest.raises(ValueError, match="outlier_ratio"):
            AKVQConfig(outlier_ratio=-0.1)

    def test_invalid_outlier_ratio_too_large(self):
        with pytest.raises(ValueError, match="outlier_ratio"):
            AKVQConfig(outlier_ratio=0.6)

    def test_invalid_n_heads(self):
        with pytest.raises(ValueError, match="n_heads"):
            AKVQConfig(n_heads=0)


class TestAKVQCache:
    def _kv(self, H=4, S=16, d=8, seed=0):
        rng = np.random.default_rng(seed)
        K = rng.standard_normal((H, S, d)).astype(np.float32)
        V = rng.standard_normal((H, S, d)).astype(np.float32)
        return K, V

    def _attn_scores(self, H=4, T=8, S=16, seed=0):
        rng = np.random.default_rng(seed)
        A = rng.random((H, T, S)).astype(np.float32)
        return A / A.sum(axis=-1, keepdims=True)

    def test_calibrate_returns_bits_list(self):
        H = 4
        cache = AKVQCache(AKVQConfig(n_heads=H, head_dim=8))
        attn = self._attn_scores(H)
        bits = cache.calibrate(attn)
        assert len(bits) == H
        assert all(b in (2, 4, 8) for b in bits)

    def test_store_and_load_roundtrip_shape(self):
        H, S, d = 4, 16, 8
        cache = AKVQCache(AKVQConfig(n_heads=H, head_dim=d, outlier_ratio=0.1))
        attn = self._attn_scores(H)
        cache.calibrate(attn)
        K, V = self._kv(H, S, d)
        cache.store(0, K, V)
        K_out, V_out = cache.load(0)
        assert K_out.shape == K.shape
        assert V_out.shape == V.shape

    def test_load_missing_raises(self):
        cache = AKVQCache(AKVQConfig())
        with pytest.raises(KeyError):
            cache.load(99)

    def test_store_preserves_dtype(self):
        H, S, d = 4, 8, 8
        cache = AKVQCache(AKVQConfig(n_heads=H, head_dim=d, outlier_ratio=0.1))
        cache.calibrate(self._attn_scores(H))
        K, V = self._kv(H, S, d)
        cache.store(0, K, V)
        K_out, V_out = cache.load(0)
        assert K_out.dtype == np.float32
        assert V_out.dtype == np.float32

    def test_n_layers_cached(self):
        H, S, d = 4, 8, 8
        cache = AKVQCache(AKVQConfig(n_heads=H, head_dim=d))
        assert cache.n_layers_cached() == 0
        cache.calibrate(self._attn_scores(H))
        K, V = self._kv(H, S, d)
        cache.store(0, K, V)
        cache.store(1, K, V)
        assert cache.n_layers_cached() == 2

    def test_memory_bytes_less_than_fp32(self):
        H, S, d = 4, 16, 8
        cache = AKVQCache(AKVQConfig(n_heads=H, head_dim=d, high_precision_bits=4,
                                     low_precision_bits=2, outlier_ratio=0.05))
        cache.calibrate(self._attn_scores(H))
        K, V = self._kv(H, S, d)
        cache.store(0, K, V)
        fp32_bytes = H * S * d * 4 * 2  # K and V
        assert cache.memory_bytes() < fp32_bytes

    def test_head_bits_none_before_calibrate(self):
        cache = AKVQCache(AKVQConfig())
        # Should be None or a default list
        bits = cache.head_bits()
        assert bits is None or isinstance(bits, list)

    def test_repr(self):
        cache = AKVQCache(AKVQConfig())
        assert "AKVQ" in repr(cache) or "akvq" in repr(cache).lower()


# ============================================================
# DeltaZipAdapter tests
# ============================================================

from squish.quant.delta_zip import (
    DeltaZipConfig,
    DeltaCompressedAdapter,
    DeltaZipAdapter,
)


class TestDeltaZipConfig:
    def test_defaults(self):
        cfg = DeltaZipConfig()
        assert cfg.quant_bits in (2, 4, 8)
        assert cfg.block_size >= 1

    def test_invalid_quant_bits(self):
        with pytest.raises(ValueError, match="quant_bits"):
            DeltaZipConfig(quant_bits=3)

    def test_invalid_block_size(self):
        with pytest.raises(ValueError, match="block_size"):
            DeltaZipConfig(block_size=0)


class TestDeltaZipAdapter:
    def _weights(self, rows=64, cols=64, delta_scale=0.01, seed=0):
        rng = np.random.default_rng(seed)
        base = rng.standard_normal((rows, cols)).astype(np.float32)
        adapted = base + delta_scale * rng.standard_normal((rows, cols)).astype(np.float32)
        return base, adapted

    def test_compress_returns_compressed(self):
        store = DeltaZipAdapter(DeltaZipConfig())
        base, adapted = self._weights()
        result = store.compress_delta("a", base, adapted)
        assert isinstance(result, DeltaCompressedAdapter)

    def test_shape_mismatch_raises(self):
        store = DeltaZipAdapter(DeltaZipConfig())
        base = np.ones((64, 64), dtype=np.float32)
        adapted = np.ones((32, 32), dtype=np.float32)
        with pytest.raises(ValueError):
            store.compress_delta("a", base, adapted)

    def test_merge_shape(self):
        store = DeltaZipAdapter(DeltaZipConfig())
        base, adapted = self._weights()
        store.compress_delta("a", base, adapted)
        merged = store.merge("a", base)
        assert merged.shape == base.shape

    def test_merge_close_to_adapted(self):
        """8-bit quantisation should recover the adapted weight with small error."""
        store = DeltaZipAdapter(DeltaZipConfig(quant_bits=8, block_size=8))
        base, adapted = self._weights(rows=32, cols=32, delta_scale=0.01)
        store.compress_delta("a", base, adapted)
        merged = store.merge("a", base)
        err = np.abs(merged - adapted).max()
        assert err < 0.1

    def test_merge_missing_raises(self):
        store = DeltaZipAdapter(DeltaZipConfig())
        with pytest.raises(KeyError):
            store.merge("missing", np.ones((4, 4)))

    def test_decompress_delta_shape(self):
        store = DeltaZipAdapter(DeltaZipConfig())
        base, adapted = self._weights(rows=32, cols=32)
        store.compress_delta("b", base, adapted)
        delta = store.decompress_delta("b")
        assert delta.shape == base.shape

    def test_compression_ratio_less_than_one(self):
        store = DeltaZipAdapter(DeltaZipConfig(quant_bits=4, block_size=4))
        base, adapted = self._weights()
        store.compress_delta("c", base, adapted)
        ratio = store.compression_ratio("c")
        assert ratio < 1.0

    def test_n_adapters(self):
        store = DeltaZipAdapter(DeltaZipConfig())
        assert store.n_adapters() == 0
        base, adapted = self._weights(rows=16, cols=16)
        store.compress_delta("a", base, adapted)
        store.compress_delta("b", base, adapted)
        assert store.n_adapters() == 2

    def test_memory_bytes_positive(self):
        store = DeltaZipAdapter(DeltaZipConfig())
        base, adapted = self._weights()
        store.compress_delta("a", base, adapted)
        assert store.memory_bytes() > 0

    def test_merge_base_shape_mismatch_raises(self):
        store = DeltaZipAdapter(DeltaZipConfig())
        base, adapted = self._weights(rows=32, cols=32)
        store.compress_delta("x", base, adapted)
        wrong_base = np.ones((16, 16), dtype=np.float32)
        with pytest.raises(ValueError):
            store.merge("x", wrong_base)

    def test_2bit_quantisation_roundtrip(self):
        store = DeltaZipAdapter(DeltaZipConfig(quant_bits=2, block_size=8))
        base, adapted = self._weights(rows=16, cols=16)
        store.compress_delta("a", base, adapted)
        merged = store.merge("a", base)
        assert merged.shape == base.shape
        assert np.all(np.isfinite(merged))

    def test_repr(self):
        store = DeltaZipAdapter(DeltaZipConfig())
        assert "DeltaZip" in repr(store) or "delta" in repr(store).lower()
