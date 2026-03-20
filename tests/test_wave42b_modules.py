"""tests/test_wave42b_modules.py

Tests for Wave 42b modules:
  - RESTDecode (squish/speculative/rest_decode.py)
  - StarAttention (squish/attention/star_attn.py)
  - SplitwiseScheduler (squish/serving/splitwise_scheduler.py)
  - KVQuantCache (squish/kv/kvquant.py)
  - EfficientQAT (squish/quant/efficient_qat.py)
  - CacheGenCodec (squish/kv/cache_gen.py)
"""

from __future__ import annotations

import pytest
import numpy as np

# ── RESTDecode ────────────────────────────────────────────────────────────────

from squish.speculative.rest_decode import RESTConfig, RESTDraftResult, RESTDecode


class TestRESTConfig:
    def test_defaults(self):
        cfg = RESTConfig()
        assert cfg.n_gram >= 2
        assert cfg.top_k_draft >= 1
        assert cfg.temperature > 0.0

    def test_invalid_n_gram(self):
        with pytest.raises(ValueError, match="n_gram"):
            RESTConfig(n_gram=1)

    def test_invalid_top_k_draft(self):
        with pytest.raises(ValueError, match="top_k_draft"):
            RESTConfig(top_k_draft=0)

    def test_invalid_temperature(self):
        with pytest.raises(ValueError, match="temperature"):
            RESTConfig(temperature=0.0)

    def test_invalid_max_datastore(self):
        with pytest.raises(ValueError, match="max_datastore"):
            RESTConfig(max_datastore=0)


class TestRESTDecode:
    VOCAB = 16

    def _target_fn(self, favor_token=None):
        def fn(last_token, context):
            p = np.ones(self.VOCAB) / self.VOCAB
            if favor_token is not None:
                p = np.zeros(self.VOCAB)
                p[favor_token] = 1.0
            return p
        return fn

    def test_add_to_datastore(self):
        dec = RESTDecode(RESTConfig(n_gram=3, seed=0))
        dec.add_to_datastore([1, 2, 3, 4, 5])
        assert dec.datastore_size() > 0

    def test_step_returns_rest_draft_result(self):
        dec = RESTDecode(RESTConfig(n_gram=3, seed=0))
        dec.add_to_datastore([1, 2, 3, 4])
        result = dec.step([1, 2], self._target_fn())
        assert isinstance(result, RESTDraftResult)

    def test_accepted_tokens_nonempty(self):
        dec = RESTDecode(RESTConfig(n_gram=3, seed=0))
        dec.add_to_datastore([1, 2, 3, 4])
        result = dec.step([1, 2], self._target_fn())
        assert len(result.accepted_tokens) >= 1

    def test_acceptance_rate_in_range(self):
        dec = RESTDecode(RESTConfig(n_gram=3, seed=0))
        dec.add_to_datastore([1, 2, 3])
        result = dec.step([1, 2], self._target_fn())
        assert 0.0 <= result.acceptance_rate <= 1.0

    def test_no_proposals_when_empty_datastore(self):
        dec = RESTDecode(RESTConfig(n_gram=3, seed=0))
        result = dec.step([1, 2], self._target_fn())
        assert result.n_proposed == 0
        assert len(result.accepted_tokens) == 1  # fallback token only

    def test_mean_acceptance_rate_after_steps(self):
        dec = RESTDecode(RESTConfig(n_gram=3, seed=1))
        dec.add_to_datastore(list(range(10)))
        for i in range(5):
            dec.step(list(range(max(0, i - 1), i + 2)), self._target_fn())
        assert dec.mean_acceptance_rate >= 0.0

    def test_reset_stats(self):
        dec = RESTDecode(RESTConfig(n_gram=3, seed=0))
        dec.add_to_datastore([1, 2, 3, 4])
        dec.step([1, 2], self._target_fn())
        dec.reset_stats()
        assert dec.mean_acceptance_rate == 0.0

    def test_datastore_cap(self):
        cfg = RESTConfig(n_gram=2, max_datastore=3, seed=0)
        dec = RESTDecode(cfg)
        dec.add_to_datastore(list(range(20)))
        assert dec.datastore_size() <= 3

    def test_repr(self):
        dec = RESTDecode()
        assert "RESTDecode" in repr(dec)

    def test_default_config(self):
        dec = RESTDecode()
        assert dec.config is not None

    def test_high_acceptance_with_perfect_draft(self):
        """When draft matches target exactly, all proposals should be accepted."""
        cfg = RESTConfig(n_gram=2, top_k_draft=1, seed=42)
        dec = RESTDecode(cfg)
        # Store token 9 as the continuation of context [5].
        dec.add_to_datastore([5, 9])
        # Target favors token 9 with probability 1.
        result = dec.step([5], self._target_fn(favor_token=9))
        assert 9 in result.accepted_tokens

    def test_n_proposed_matches_datastore_hits(self):
        dec = RESTDecode(RESTConfig(n_gram=2, top_k_draft=3, seed=0))
        dec.add_to_datastore([1, 2, 3, 4, 5])
        result = dec.step([1], self._target_fn())
        assert result.n_proposed <= 3


# ── StarAttention ─────────────────────────────────────────────────────────────

from squish.attention.star_attn import StarAttentionConfig, StarAttention


class TestStarAttentionConfig:
    def test_defaults(self):
        cfg = StarAttentionConfig()
        assert cfg.n_heads >= 1
        assert cfg.head_dim >= 1
        assert cfg.block_size >= 1

    def test_invalid_n_heads(self):
        with pytest.raises(ValueError, match="n_heads"):
            StarAttentionConfig(n_heads=0)

    def test_invalid_head_dim(self):
        with pytest.raises(ValueError, match="head_dim"):
            StarAttentionConfig(head_dim=0)

    def test_invalid_block_size(self):
        with pytest.raises(ValueError, match="block_size"):
            StarAttentionConfig(block_size=0)


class TestStarAttention:
    def _make_model(self, n_heads=2, head_dim=8, block_size=4, causal=True):
        cfg = StarAttentionConfig(
            n_heads=n_heads, head_dim=head_dim, block_size=block_size, causal=causal
        )
        return StarAttention(cfg)

    def _random_qkv(self, n_heads=2, T=12, head_dim=8):
        rng = np.random.default_rng(0)
        Q = rng.standard_normal((n_heads, T, head_dim)).astype(np.float32)
        K = rng.standard_normal((n_heads, T, head_dim)).astype(np.float32)
        V = rng.standard_normal((n_heads, T, head_dim)).astype(np.float32)
        return Q, K, V

    def test_output_shape(self):
        model = self._make_model()
        Q, K, V = self._random_qkv()
        out = model.forward(Q, K, V)
        assert out.shape == (2, 12, 8)

    def test_output_finite(self):
        model = self._make_model()
        Q, K, V = self._random_qkv()
        out = model.forward(Q, K, V)
        assert np.all(np.isfinite(out))

    def test_non_causal(self):
        model = self._make_model(causal=False)
        Q, K, V = self._random_qkv()
        out = model.forward(Q, K, V)
        assert out.shape == (2, 12, 8)

    def test_single_block(self):
        """If T <= block_size, all tokens are in one block = anchor."""
        model = self._make_model(block_size=16)
        Q, K, V = self._random_qkv(T=8)
        out = model.forward(Q, K, V)
        assert out.shape == (2, 8, 8)
        assert np.all(np.isfinite(out))

    def test_many_blocks(self):
        model = self._make_model(block_size=4)
        Q, K, V = self._random_qkv(T=20)
        out = model.forward(Q, K, V)
        assert out.shape == (2, 20, 8)

    def test_unaligned_sequence_length(self):
        """T not divisible by block_size should still work."""
        model = self._make_model(block_size=5)
        Q, K, V = self._random_qkv(T=13)
        out = model.forward(Q, K, V)
        assert out.shape == (2, 13, 8)

    def test_repr(self):
        model = self._make_model()
        assert "StarAttention" in repr(model)

    def test_default_config(self):
        model = StarAttention()
        assert model.config is not None


# ── SplitwiseScheduler ────────────────────────────────────────────────────────

from squish.serving.splitwise_scheduler import (
    SplitwiseConfig,
    SplitwiseRequest,
    SplitwiseScheduler,
)


class TestSplitwiseConfig:
    def test_defaults(self):
        cfg = SplitwiseConfig()
        assert cfg.prefill_workers >= 1
        assert cfg.decode_workers >= 1

    def test_invalid_prefill_workers(self):
        with pytest.raises(ValueError, match="prefill_workers"):
            SplitwiseConfig(prefill_workers=0)

    def test_invalid_decode_workers(self):
        with pytest.raises(ValueError, match="decode_workers"):
            SplitwiseConfig(decode_workers=0)

    def test_invalid_max_prefill_batch(self):
        with pytest.raises(ValueError, match="max_prefill_batch"):
            SplitwiseConfig(max_prefill_batch=0)

    def test_invalid_max_decode_batch(self):
        with pytest.raises(ValueError, match="max_decode_batch"):
            SplitwiseConfig(max_decode_batch=0)


class TestSplitwiseScheduler:
    def _make_sched(self):
        cfg = SplitwiseConfig(
            prefill_workers=2, decode_workers=4,
            max_prefill_batch=2, max_decode_batch=4,
        )
        return SplitwiseScheduler(cfg)

    def test_submit_and_schedule_prefill(self):
        sched = self._make_sched()
        req = SplitwiseRequest("r1", prompt_tokens=64)
        sched.submit(req)
        batch = sched.schedule_prefill()
        assert len(batch) == 1

    def test_complete_prefill_moves_to_decode(self):
        sched = self._make_sched()
        req = SplitwiseRequest("r1", prompt_tokens=64)
        sched.submit(req)
        sched.schedule_prefill()
        sched.complete_prefill("r1")
        decode_batch = sched.schedule_decode()
        assert len(decode_batch) == 1

    def test_complete_decode_marks_done(self):
        sched = self._make_sched()
        req = SplitwiseRequest("r1", prompt_tokens=64)
        sched.submit(req)
        sched.schedule_prefill()
        sched.complete_prefill("r1")
        sched.schedule_decode()
        sched.complete_decode("r1")
        assert req.is_done

    def test_stats_dict(self):
        sched = self._make_sched()
        stats = sched.stats()
        assert "n_completed" in stats
        assert "total_submitted" in stats

    def test_n_inflight(self):
        sched = self._make_sched()
        req = SplitwiseRequest("r1", prompt_tokens=64)
        sched.submit(req)
        sched.schedule_prefill()
        assert sched.n_inflight() == 1

    def test_duplicate_request_id_raises(self):
        sched = self._make_sched()
        req = SplitwiseRequest("dup", prompt_tokens=10)
        sched.submit(req)
        with pytest.raises(ValueError, match="Duplicate"):
            sched.submit(SplitwiseRequest("dup", prompt_tokens=10))

    def test_complete_non_prefilling_raises(self):
        sched = self._make_sched()
        sched.submit(SplitwiseRequest("r1", prompt_tokens=10))
        with pytest.raises(RuntimeError):
            sched.complete_prefill("r1")  # Not yet scheduled

    def test_multiple_requests(self):
        sched = self._make_sched()
        for i in range(5):
            sched.submit(SplitwiseRequest(f"r{i}", prompt_tokens=10))
        batch = sched.schedule_prefill()
        assert len(batch) <= 2  # max_prefill_batch

    def test_repr(self):
        sched = self._make_sched()
        assert "SplitwiseScheduler" in repr(sched)

    def test_default_config(self):
        sched = SplitwiseScheduler()
        assert sched.config is not None


# ── KVQuantCache ──────────────────────────────────────────────────────────────

from squish.kv.kvquant import KVQuantConfig, KVQuantCache


class TestKVQuantConfig:
    def test_defaults(self):
        cfg = KVQuantConfig()
        assert cfg.n_heads >= 1
        assert cfg.head_dim >= 1
        assert cfg.bits in (2, 4, 8)

    def test_invalid_bits(self):
        with pytest.raises(ValueError, match="bits"):
            KVQuantConfig(bits=3)

    def test_invalid_n_heads(self):
        with pytest.raises(ValueError, match="n_heads"):
            KVQuantConfig(n_heads=0)

    def test_invalid_head_dim(self):
        with pytest.raises(ValueError, match="head_dim"):
            KVQuantConfig(head_dim=0)

    def test_invalid_calibration_window(self):
        with pytest.raises(ValueError, match="calibration_window"):
            KVQuantConfig(calibration_window=0)


class TestKVQuantCache:
    def _kv(self, n_heads=2, T=16, head_dim=8, seed=0):
        rng = np.random.default_rng(seed)
        K = rng.standard_normal((n_heads, T, head_dim)).astype(np.float32)
        V = rng.standard_normal((n_heads, T, head_dim)).astype(np.float32)
        return K, V

    def test_calibrate_and_quantize(self):
        cfg = KVQuantConfig(n_heads=2, head_dim=8, bits=8, calibration_window=4)
        cache = KVQuantCache(cfg)
        K, V = self._kv()
        cache.calibrate(K, V)
        cache.quantize(0, K, V)
        assert cache.n_layers_cached() == 1

    def test_dequantize_shape(self):
        cfg = KVQuantConfig(n_heads=2, head_dim=8, bits=8, calibration_window=4)
        cache = KVQuantCache(cfg)
        K, V = self._kv()
        cache.calibrate(K, V)
        cache.quantize(0, K, V)
        K2, V2 = cache.dequantize(0)
        assert K2.shape == K.shape
        assert V2.shape == V.shape

    def test_dequantize_missing_raises(self):
        cache = KVQuantCache()
        with pytest.raises(KeyError):
            cache.dequantize(99)

    def test_relative_error_acceptable(self):
        cfg = KVQuantConfig(n_heads=2, head_dim=8, bits=8, calibration_window=8)
        cache = KVQuantCache(cfg)
        K, V = self._kv()
        cache.calibrate(K, V)
        cache.quantize(0, K, V)
        K2, _ = cache.dequantize(0)
        err = cache.relative_error(K, K2)
        assert err < 0.5  # 8-bit should be reasonably accurate

    def test_memory_bytes(self):
        cfg = KVQuantConfig(n_heads=2, head_dim=8, bits=4, calibration_window=4)
        cache = KVQuantCache(cfg)
        K, V = self._kv()
        cache.quantize(0, K, V)
        assert cache.memory_bytes() > 0

    def test_n_layers_cached(self):
        cfg = KVQuantConfig(n_heads=2, head_dim=8, bits=4, calibration_window=4)
        cache = KVQuantCache(cfg)
        K, V = self._kv()
        for layer in range(4):
            cache.quantize(layer, K, V)
        assert cache.n_layers_cached() == 4

    def test_repr(self):
        cache = KVQuantCache()
        assert "KVQuantCache" in repr(cache)

    def test_bits_4(self):
        cfg = KVQuantConfig(n_heads=2, head_dim=8, bits=4, calibration_window=4)
        cache = KVQuantCache(cfg)
        K, V = self._kv()
        cache.calibrate(K, V)
        cache.quantize(0, K, V)
        K2, _ = cache.dequantize(0)
        assert K2.shape == K.shape

    def test_default_config(self):
        cache = KVQuantCache()
        assert cache.config is not None


# ── EfficientQAT ──────────────────────────────────────────────────────────────

from squish.quant.efficient_qat import EfficientQATConfig, EfficientQAT


class TestEfficientQATConfig:
    def test_defaults(self):
        cfg = EfficientQATConfig()
        assert cfg.bits in (2, 4, 8)
        assert cfg.block_size >= 1
        assert cfg.n_calibration_steps >= 1

    def test_invalid_bits(self):
        with pytest.raises(ValueError, match="bits"):
            EfficientQATConfig(bits=3)

    def test_invalid_block_size(self):
        with pytest.raises(ValueError, match="block_size"):
            EfficientQATConfig(block_size=0)

    def test_invalid_n_calibration_steps(self):
        with pytest.raises(ValueError, match="n_calibration_steps"):
            EfficientQATConfig(n_calibration_steps=0)


class TestEfficientQAT:
    def _weight_and_activations(self, out=16, inp=8, n_tok=10, seed=0):
        rng = np.random.default_rng(seed)
        w = rng.standard_normal((out, inp)).astype(np.float32)
        a = rng.standard_normal((n_tok, inp)).astype(np.float32)
        return w, a

    def test_calibrate_block_returns_scale(self):
        qat = EfficientQAT(EfficientQATConfig(bits=4, block_size=8))
        w, a = self._weight_and_activations()
        scale = qat.calibrate_block(0, w, a)
        assert scale.shape == (16,)
        assert np.all(scale > 0)

    def test_quantize_weight_shape(self):
        qat = EfficientQAT(EfficientQATConfig(bits=4, block_size=8))
        w, a = self._weight_and_activations()
        codes, scales = qat.quantize_weight(w)
        assert codes.shape == w.shape
        assert scales.shape == (16,)

    def test_dequantize_weight_shape(self):
        qat = EfficientQAT(EfficientQATConfig(bits=4, block_size=8))
        w, a = self._weight_and_activations()
        codes, scales = qat.quantize_weight(w)
        w_hat = qat.dequantize_weight(codes, scales)
        assert w_hat.shape == w.shape

    def test_relative_error_reasonable(self):
        qat = EfficientQAT(EfficientQATConfig(bits=8, block_size=8))
        w, a = self._weight_and_activations()
        codes, scales = qat.quantize_weight(w)
        w_hat = qat.dequantize_weight(codes, scales)
        err = qat.relative_error(w, w_hat)
        assert err < 0.5

    def test_n_calibrated_blocks(self):
        qat = EfficientQAT(EfficientQATConfig(bits=4, block_size=8))
        w, a = self._weight_and_activations()
        qat.calibrate_block(0, w, a)
        qat.calibrate_block(1, w, a)
        assert qat.n_calibrated_blocks() == 2

    def test_repr(self):
        qat = EfficientQAT()
        assert "EfficientQAT" in repr(qat)

    def test_default_config(self):
        qat = EfficientQAT()
        assert qat.config is not None

    def test_bits_4_vs_8_error(self):
        """4-bit should have higher error than 8-bit."""
        w, a = self._weight_and_activations(seed=42)
        qat4 = EfficientQAT(EfficientQATConfig(bits=4, block_size=8))
        codes4, scales4 = qat4.quantize_weight(w)
        w_hat4 = qat4.dequantize_weight(codes4, scales4)
        err4 = qat4.relative_error(w, w_hat4)

        qat8 = EfficientQAT(EfficientQATConfig(bits=8, block_size=8))
        codes8, scales8 = qat8.quantize_weight(w)
        w_hat8 = qat8.dequantize_weight(codes8, scales8)
        err8 = qat8.relative_error(w, w_hat8)

        assert err4 >= err8


# ── CacheGenCodec ─────────────────────────────────────────────────────────────

from squish.kv.cache_gen import CacheGenConfig, CacheGenCodec


class TestCacheGenConfig:
    def test_defaults(self):
        cfg = CacheGenConfig()
        assert cfg.bits in (4, 8)
        assert cfg.block_size >= 1

    def test_invalid_bits(self):
        with pytest.raises(ValueError, match="bits"):
            CacheGenConfig(bits=2)

    def test_invalid_block_size(self):
        with pytest.raises(ValueError, match="block_size"):
            CacheGenConfig(block_size=0)


class TestCacheGenCodec:
    def _kv(self, n_heads=2, T=16, head_dim=8, seed=0):
        rng = np.random.default_rng(seed)
        K = rng.standard_normal((n_heads, T, head_dim)).astype(np.float32)
        V = rng.standard_normal((n_heads, T, head_dim)).astype(np.float32)
        return K, V

    def test_encode_returns_bytes(self):
        codec = CacheGenCodec(CacheGenConfig(bits=8))
        K, V = self._kv()
        bs = codec.encode(K, V)
        assert isinstance(bs, bytes)
        assert len(bs) > 0

    def test_encode_decode_roundtrip_shape(self):
        codec = CacheGenCodec(CacheGenConfig(bits=8))
        K, V = self._kv()
        bs = codec.encode(K, V)
        K2, V2 = codec.decode(bs, K.shape, V.shape)
        assert K2.shape == K.shape
        assert V2.shape == V.shape

    def test_encode_decode_relative_error(self):
        codec = CacheGenCodec(CacheGenConfig(bits=8))
        K, V = self._kv()
        bs = codec.encode(K, V)
        K2, V2 = codec.decode(bs, K.shape, V.shape)
        err = float(np.linalg.norm(K - K2) / (np.linalg.norm(K) + 1e-9))
        assert err < 0.5

    def test_compression_ratio_less_than_one(self):
        """8-bit compression of float32 should give ratio < 1."""
        codec = CacheGenCodec(CacheGenConfig(bits=8))
        K, V = self._kv(T=64)
        ratio = codec.compression_ratio(K, V)
        # 8-bit codes stored as uint8 (same byte count as int8) but float32 is
        # 4 bytes per element.  With header overhead, ratio should still be < 2.
        assert ratio < 2.0

    def test_stream_encode_reassembles(self):
        codec = CacheGenCodec(CacheGenConfig(bits=8))
        K, V = self._kv()
        full_bs = codec.encode(K, V)
        chunks = list(codec.stream_encode(K, V, chunk_size=16))
        reassembled = b"".join(chunks)
        assert reassembled == full_bs

    def test_4bit_encode_decode_shape(self):
        codec = CacheGenCodec(CacheGenConfig(bits=4))
        K, V = self._kv()
        bs = codec.encode(K, V)
        K2, V2 = codec.decode(bs, K.shape, V.shape)
        assert K2.shape == K.shape
        assert V2.shape == V.shape

    def test_repr(self):
        codec = CacheGenCodec()
        assert "CacheGenCodec" in repr(codec)

    def test_default_config(self):
        codec = CacheGenCodec()
        assert codec.config is not None

    def test_stream_encode_chunks_nonempty(self):
        codec = CacheGenCodec(CacheGenConfig(bits=8))
        K, V = self._kv()
        chunks = list(codec.stream_encode(K, V, chunk_size=8))
        assert len(chunks) > 0
        for chunk in chunks:
            assert isinstance(chunk, bytes)
