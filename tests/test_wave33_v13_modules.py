"""tests/test_wave33_v13_modules.py — Tests for Wave 33 v13 (new modules).

Covers v13 Wave 33 additions (JacobiDecoder, MultiTokenPredictor, FP6Quantizer,
DraftTokenRecycler, LayerDeduplicator, TokenPipeline).
"""

import math
import pytest
import numpy as np

# ---------------------------------------------------------------------------
# JacobiDecoder
# ---------------------------------------------------------------------------

from squish.speculative.jacobi_decode import (
    JacobiConfig,
    JacobiDecoder,
    JacobiStats,
)


class TestJacobiConfigV13:
    def test_defaults(self):
        cfg = JacobiConfig()
        assert cfg.n_tokens == 4
        assert cfg.max_iter == 8
        assert cfg.variant == "jacobi"

    def test_invalid_n_tokens(self):
        with pytest.raises(ValueError, match="n_tokens"):
            JacobiConfig(n_tokens=0)

    def test_invalid_max_iter(self):
        with pytest.raises(ValueError, match="max_iter"):
            JacobiConfig(max_iter=0)

    def test_invalid_variant(self):
        with pytest.raises(ValueError):
            JacobiConfig(variant="invalid_variant")

    def test_invalid_temperature(self):
        with pytest.raises(ValueError, match="temperature"):
            JacobiConfig(temperature=-0.1)

    def test_invalid_init(self):
        with pytest.raises(ValueError):
            JacobiConfig(init="bad_init")

    def test_gauss_seidel_variant_valid(self):
        cfg = JacobiConfig(variant="gauss_seidel")
        assert cfg.variant == "gauss_seidel"


class TestJacobiStatsV13:
    def test_zero_state(self):
        stats = JacobiStats()
        assert stats.total_decode_steps == 0
        assert stats.mean_tokens_per_step == 0.0
        assert stats.fixed_point_rate == 0.0

    def test_mean_properties(self):
        stats = JacobiStats(
            total_decode_steps=4,
            total_tokens_generated=12,
            total_iterations=8,
            total_fixed_points=6,
        )
        assert stats.mean_tokens_per_step == 3.0
        assert stats.mean_iterations_per_step == 2.0
        assert stats.fixed_point_rate == pytest.approx(6 / 12, abs=1e-6)


class TestJacobiDecoderV13:
    def _make_consistent_logits_fn(self, vocab_size=32):
        """Logits function: next_token = (tok + 1) % vocab."""
        def fn(ids):
            logits = np.zeros((len(ids), vocab_size), dtype=np.float32)
            for i, tok in enumerate(ids):
                logits[i, (tok + 1) % vocab_size] = 10.0
            return logits
        return fn

    def test_returns_list_of_ints(self):
        decoder = JacobiDecoder(JacobiConfig(n_tokens=2, max_iter=4, seed=0))
        fn = self._make_consistent_logits_fn()
        tokens, n_iter = decoder.decode_step(fn, [1, 2, 3], vocab_size=32)
        assert isinstance(tokens, list)
        assert all(isinstance(t, int) for t in tokens)
        assert len(tokens) >= 1

    def test_tokens_in_vocab_range(self):
        decoder = JacobiDecoder(JacobiConfig(n_tokens=3, max_iter=5, seed=0))
        fn = self._make_consistent_logits_fn(vocab_size=50)
        tokens, _ = decoder.decode_step(fn, [1, 2], vocab_size=50)
        assert all(0 <= t < 50 for t in tokens)

    def test_n_iter_positive(self):
        decoder = JacobiDecoder(JacobiConfig(n_tokens=2, max_iter=3, seed=0))
        fn = self._make_consistent_logits_fn()
        _, n_iter = decoder.decode_step(fn, [5], vocab_size=32)
        assert n_iter >= 1

    def test_stats_updated_on_step(self):
        decoder = JacobiDecoder(JacobiConfig(n_tokens=2, max_iter=3, seed=0))
        fn = self._make_consistent_logits_fn()
        decoder.decode_step(fn, [1, 2], vocab_size=32)
        assert decoder.stats.total_decode_steps == 1
        assert decoder.stats.total_tokens_generated >= 1

    def test_reset_stats_clears_counts(self):
        decoder = JacobiDecoder(JacobiConfig(n_tokens=2, max_iter=3, seed=0))
        fn = self._make_consistent_logits_fn()
        decoder.decode_step(fn, [1, 2], vocab_size=32)
        decoder.reset_stats()
        assert decoder.stats.total_decode_steps == 0
        assert decoder.stats.total_tokens_generated == 0

    def test_gauss_seidel_runs(self):
        decoder = JacobiDecoder(
            JacobiConfig(n_tokens=2, max_iter=4, variant="gauss_seidel", seed=1)
        )
        fn = self._make_consistent_logits_fn()
        tokens, n_iter = decoder.decode_step(fn, [1, 2], vocab_size=32)
        assert len(tokens) >= 1

    def test_fallback_one_token_minimum(self):
        decoder = JacobiDecoder(JacobiConfig(n_tokens=4, max_iter=1, seed=99))
        rng = np.random.default_rng(7)
        def random_fn(ids):
            return rng.random((len(ids), 32)).astype(np.float32)
        tokens, _ = decoder.decode_step(random_fn, [10, 20, 30], vocab_size=32)
        assert len(tokens) >= 1

    def test_repr_contains_class_name(self):
        decoder = JacobiDecoder(JacobiConfig(n_tokens=2))
        assert "JacobiDecoder" in repr(decoder)


# ---------------------------------------------------------------------------
# MultiTokenPredictor
# ---------------------------------------------------------------------------

from squish.speculative.mtp_head import (
    MTPHeadConfig,
    MTPHeadLayer,
    MTPHeadStats,
    MultiTokenPredictor,
)


class TestMTPHeadConfigV13:
    def test_defaults(self):
        cfg = MTPHeadConfig()
        assert cfg.n_heads == 4
        assert cfg.vocab_size == 32_000

    def test_invalid_n_heads(self):
        with pytest.raises(ValueError, match="n_heads"):
            MTPHeadConfig(n_heads=0)

    def test_invalid_vocab_size(self):
        with pytest.raises(ValueError, match="vocab_size"):
            MTPHeadConfig(vocab_size=1)

    def test_invalid_emb_dim(self):
        with pytest.raises(ValueError, match="emb_dim"):
            MTPHeadConfig(emb_dim=0)

    def test_invalid_temperature(self):
        with pytest.raises(ValueError, match="temperature"):
            MTPHeadConfig(temperature=-0.5)


class TestMTPHeadLayerV13:
    def test_forward_shape(self):
        layer = MTPHeadLayer(emb_dim=32, vocab_size=100, seed=0)
        h = np.random.randn(32).astype(np.float32)
        out = layer.forward(h)
        assert out.shape == (100,)

    def test_set_weight(self):
        layer = MTPHeadLayer(emb_dim=32, vocab_size=100)
        new_w = np.zeros((100, 32), dtype=np.float32)
        layer.set_weight(new_w)
        np.testing.assert_array_equal(layer.weight, new_w)

    def test_set_weight_wrong_shape_raises(self):
        layer = MTPHeadLayer(emb_dim=32, vocab_size=100)
        with pytest.raises(ValueError):
            layer.set_weight(np.zeros((50, 32), dtype=np.float32))

    def test_nbytes_correct(self):
        layer = MTPHeadLayer(emb_dim=32, vocab_size=100)
        assert layer.nbytes == 32 * 100 * 4


class TestMultiTokenPredictorV13:
    def _small_cfg(self, n_heads=3, vs=64, emb=32):
        return MTPHeadConfig(n_heads=n_heads, vocab_size=vs, emb_dim=emb)

    def test_sample_tokens_count(self):
        mtp = MultiTokenPredictor(self._small_cfg(n_heads=3))
        h = np.random.randn(32).astype(np.float32)
        tokens, probs = mtp.sample_tokens(h)
        assert len(tokens) == 3
        assert len(probs) == 3

    def test_tokens_in_vocab_range(self):
        mtp = MultiTokenPredictor(self._small_cfg(n_heads=2, vs=50))
        h = np.random.randn(32).astype(np.float32)
        tokens, _ = mtp.sample_tokens(h)
        assert all(0 <= t < 50 for t in tokens)

    def test_greedy_is_deterministic(self):
        mtp = MultiTokenPredictor(self._small_cfg())
        h = np.random.randn(32).astype(np.float32)
        t1, _ = mtp.sample_tokens(h)
        t2, _ = mtp.sample_tokens(h)
        assert t1 == t2

    def test_forward_calls_stat(self):
        mtp = MultiTokenPredictor(self._small_cfg(n_heads=2))
        h = np.random.randn(32).astype(np.float32)
        mtp.sample_tokens(h)
        assert mtp.stats.total_forward_calls == 1

    def test_verify_all_accepted(self):
        cfg = self._small_cfg(n_heads=2, vs=64, emb=32)
        mtp = MultiTokenPredictor(cfg)
        # Use all-ones hidden so head.forward(h) = sum(weight_row) per token.
        # With weight[0,:]=1, weight[others,:]=0, token-0 logit = emb_dim >> 0.
        h = np.ones(32, dtype=np.float32)
        for head in mtp._heads:
            head.weight[:] = 0.0
            head.weight[0, :] = 1.0  # logit[0] = 32 > logit[others] = 0
        target_logits = [
            np.array([100.0] + [0.0] * 63, dtype=np.float32),
            np.array([100.0] + [0.0] * 63, dtype=np.float32),
        ]
        tokens, accepted = mtp.verify_against_target(h, target_logits)
        assert all(accepted)

    def test_total_parameters(self):
        cfg = self._small_cfg(n_heads=2, vs=50, emb=20)
        mtp = MultiTokenPredictor(cfg)
        assert mtp.total_parameters() == 2 * 50 * 20

    def test_tied_weights_param_count(self):
        cfg = MTPHeadConfig(n_heads=3, vocab_size=50, emb_dim=20, tie_weights=True)
        mtp = MultiTokenPredictor(cfg)
        assert mtp.total_parameters() == 50 * 20

    def test_repr_contains_class_name(self):
        mtp = MultiTokenPredictor(self._small_cfg())
        assert "MultiTokenPredictor" in repr(mtp)


# ---------------------------------------------------------------------------
# FP6Quantizer
# ---------------------------------------------------------------------------

from squish.quant.fp6_quant import (
    FP6Config,
    FP6Quantized,
    FP6Stats,
    FP6Quantizer,
)


class TestFP6ConfigV13:
    def test_defaults(self):
        cfg = FP6Config()
        assert cfg.fmt == "e3m2"
        assert cfg.group_size == 64

    def test_invalid_fmt(self):
        with pytest.raises(ValueError, match="fmt"):
            FP6Config(fmt="e4m1")

    def test_invalid_group_size(self):
        with pytest.raises(ValueError, match="group_size"):
            FP6Config(group_size=0)

    def test_e3m2_exp_man_bits(self):
        cfg = FP6Config(fmt="e3m2")
        assert cfg.exp_bits == 3
        assert cfg.man_bits == 2

    def test_e2m3_exp_man_bits(self):
        cfg = FP6Config(fmt="e2m3")
        assert cfg.exp_bits == 2
        assert cfg.man_bits == 3

    def test_fp6_max_positive(self):
        for fmt in ["e3m2", "e2m3"]:
            cfg = FP6Config(fmt=fmt)
            assert cfg.fp6_max > 0


class TestFP6QuantizerV13:
    def test_quantize_returns_packed(self):
        q = FP6Quantizer(FP6Config(group_size=16))
        w = np.random.randn(8, 16).astype(np.float32)
        packed = q.quantize(w)
        assert isinstance(packed, FP6Quantized)
        assert packed.original_shape == (8, 16)
        assert packed.fmt == "e3m2"

    def test_compression_ratio_below_1(self):
        q = FP6Quantizer(FP6Config(group_size=16))
        w = np.random.randn(32, 32).astype(np.float32)
        packed = q.quantize(w)
        assert packed.compression_ratio < 1.0

    def test_roundtrip_shape_preserved(self):
        q = FP6Quantizer(FP6Config(fmt="e3m2", group_size=16))
        w = np.random.randn(4, 16).astype(np.float32)
        packed = q.quantize(w)
        restored = q.dequantize(packed)
        assert restored.shape == w.shape

    def test_roundtrip_reasonable_error(self):
        q = FP6Quantizer(FP6Config(fmt="e3m2", group_size=16))
        w = np.random.randn(4, 16).astype(np.float32)
        packed = q.quantize(w)
        restored = q.dequantize(packed)
        # FP6 is lossy; require relative error < 50%
        rel_err = np.abs(w - restored).mean() / (np.abs(w).mean() + 1e-7)
        assert rel_err < 0.5

    def test_zero_weight_roundtrip(self):
        q = FP6Quantizer(FP6Config(group_size=8))
        w = np.zeros((4, 8), dtype=np.float32)
        packed = q.quantize(w)
        restored = q.dequantize(packed)
        np.testing.assert_allclose(restored, 0.0, atol=1e-5)

    def test_e2m3_roundtrip_shape(self):
        q = FP6Quantizer(FP6Config(fmt="e2m3", group_size=8))
        w = np.random.uniform(-1, 1, (4, 8)).astype(np.float32)
        packed = q.quantize(w)
        restored = q.dequantize(packed)
        assert restored.shape == w.shape

    def test_stats_incremented(self):
        q = FP6Quantizer(FP6Config(group_size=8))
        w = np.ones((4, 8), dtype=np.float32)
        packed = q.quantize(w)
        q.dequantize(packed)
        assert q.stats.quantize_calls == 1
        assert q.stats.dequant_calls == 1

    def test_theoretical_compression_ratio(self):
        q = FP6Quantizer()
        assert abs(q.compression_ratio - 6.0 / 32.0) < 1e-9

    def test_repr_contains_class_name(self):
        q = FP6Quantizer()
        assert "FP6Quantizer" in repr(q)


# ---------------------------------------------------------------------------
# DraftTokenRecycler
# ---------------------------------------------------------------------------

from squish.speculative.token_recycler import (
    RecycleConfig,
    RecycleEntry,
    RecycleStats,
    DraftTokenRecycler,
)


class TestRecycleConfigV13:
    def test_defaults(self):
        cfg = RecycleConfig()
        assert cfg.buffer_size == 16
        assert cfg.strategy == "correction"

    def test_invalid_buffer_size(self):
        with pytest.raises(ValueError, match="buffer_size"):
            RecycleConfig(buffer_size=0)

    def test_invalid_strategy(self):
        with pytest.raises(ValueError, match="strategy"):
            RecycleConfig(strategy="unknown")

    def test_invalid_min_context(self):
        with pytest.raises(ValueError, match="min_context"):
            RecycleConfig(min_context=-5)


class TestDraftTokenRecyclerV13:
    def test_record_enables_hit(self):
        r = DraftTokenRecycler(RecycleConfig(buffer_size=4, min_context=2))
        ctx = [1, 2, 3, 4]
        r.record_step(ctx, [5, 6, 7], [True, True, False], correction_token=99)
        seeds = r.get_seed_tokens(ctx, n_tokens=2)
        assert seeds is not None
        assert seeds[0] == 99

    def test_miss_on_different_context(self):
        r = DraftTokenRecycler(RecycleConfig(min_context=1))
        r.record_step([1, 2], [5], [True], 10)
        result = r.get_seed_tokens([3, 4], n_tokens=1)
        assert result is None

    def test_miss_if_context_too_short(self):
        r = DraftTokenRecycler(RecycleConfig(min_context=5))
        r.record_step([1, 2], [5], [True], 10)
        result = r.get_seed_tokens([1, 2], n_tokens=1)
        assert result is None

    def test_buffer_capacity_respected(self):
        r = DraftTokenRecycler(RecycleConfig(buffer_size=2, min_context=0))
        for i in range(10):
            r.record_step([i], [i + 1], [True], i + 2)
        assert r.buffer_size <= 2

    def test_invalidate_clears_all(self):
        r = DraftTokenRecycler(RecycleConfig(min_context=0))
        r.record_step([1], [2], [True], 3)
        r.invalidate()
        assert r.get_seed_tokens([1], n_tokens=1) is None

    def test_prefix_strategy_returns_prefix_plus_correction(self):
        r = DraftTokenRecycler(RecycleConfig(strategy="prefix", min_context=1))
        ctx = [10, 20]
        r.record_step(ctx, [30, 40, 50], [True, True, False], correction_token=99)
        seeds = r.get_seed_tokens(ctx, n_tokens=5)
        assert seeds is not None
        assert 99 in seeds  # correction must be present

    def test_hit_rate_computed_correctly(self):
        r = DraftTokenRecycler(RecycleConfig(min_context=0))
        ctx = [1, 2]
        r.record_step(ctx, [3], [True], 4)
        r.get_seed_tokens(ctx)      # hit
        r.get_seed_tokens([9, 10])  # miss
        assert r.cache_hit_rate == pytest.approx(0.5)

    def test_repr_contains_class_name(self):
        r = DraftTokenRecycler()
        assert "DraftTokenRecycler" in repr(r)


# ---------------------------------------------------------------------------
# LayerDeduplicator
# ---------------------------------------------------------------------------

from squish.quant.layer_dedup import (
    LayerDedupConfig,
    LayerSimilarity,
    DedupEntry,
    LayerDedupStats,
    LayerDeduplicator,
)


class TestLayerDedupConfigV13:
    def test_defaults(self):
        cfg = LayerDedupConfig()
        assert cfg.similarity_threshold == 0.99
        assert cfg.delta_bits == 8

    def test_invalid_threshold_zero(self):
        with pytest.raises(ValueError):
            LayerDedupConfig(similarity_threshold=0.0)

    def test_invalid_threshold_over_1(self):
        with pytest.raises(ValueError):
            LayerDedupConfig(similarity_threshold=1.1)

    def test_invalid_delta_bits(self):
        with pytest.raises(ValueError, match="delta_bits"):
            LayerDedupConfig(delta_bits=4)

    def test_invalid_min_rows(self):
        with pytest.raises(ValueError, match="min_rows"):
            LayerDedupConfig(min_rows=0)


class TestLayerDeduplicatorV13:
    def _identical_pair(self, shape=(32, 32)):
        w = np.random.randn(*shape).astype(np.float32)
        return {"layer_0.weight": w.copy(), "layer_1.weight": w.copy()}

    def test_analyze_identical_similarity_close_to_1(self):
        d = LayerDeduplicator(LayerDedupConfig())
        weights = self._identical_pair()
        sims = d.analyze(weights)
        assert len(sims) == 1
        assert sims[0].row_similarity > 0.99

    def test_analyze_different_shapes_skipped(self):
        d = LayerDeduplicator()
        weights = {
            "a": np.random.randn(16, 16).astype(np.float32),
            "b": np.random.randn(32, 32).astype(np.float32),
        }
        sims = d.analyze(weights)
        assert len(sims) == 0

    def test_deduplicate_produces_dedup_entry(self):
        d = LayerDeduplicator(LayerDedupConfig(similarity_threshold=0.9, min_rows=2))
        store = d.deduplicate(self._identical_pair())
        has_dedup = any(isinstance(v, DedupEntry) for v in store.values())
        assert has_dedup

    def test_reconstruct_shape_matches(self):
        d = LayerDeduplicator(LayerDedupConfig(similarity_threshold=0.5, min_rows=2))
        weights = self._identical_pair()
        store = d.deduplicate(weights)
        for key in weights:
            rec = d.reconstruct(store, key)
            assert rec.shape == weights[key].shape

    def test_no_dedup_below_threshold(self):
        d = LayerDeduplicator(LayerDedupConfig(similarity_threshold=0.9999, min_rows=2))
        w1 = np.eye(32, dtype=np.float32)
        w2 = np.eye(32, dtype=np.float32)[::-1].copy()
        store = d.deduplicate({"a": w1, "b": w2})
        assert all(isinstance(v, np.ndarray) for v in store.values())

    def test_disk_reduction_ratio_range(self):
        d = LayerDeduplicator(LayerDedupConfig(similarity_threshold=0.9, min_rows=2))
        d.deduplicate(self._identical_pair())
        assert 0.0 <= d.disk_reduction_ratio <= 1.0

    def test_pairs_analyzed_incremented(self):
        d = LayerDeduplicator(LayerDedupConfig(similarity_threshold=0.9, min_rows=2))
        d.deduplicate(self._identical_pair())
        assert d.stats.pairs_analyzed >= 1

    def test_missing_key_raises(self):
        d = LayerDeduplicator()
        with pytest.raises(KeyError):
            d.reconstruct({}, "missing_key")

    def test_repr_contains_class_name(self):
        d = LayerDeduplicator()
        assert "LayerDeduplicator" in repr(d)


# ---------------------------------------------------------------------------
# TokenPipeline
# ---------------------------------------------------------------------------

from squish.kernels.token_pipeline import (
    PipelineConfig,
    PipelineStage,
    PipelineStats,
    TokenPipeline,
)


class TestPipelineConfigV13:
    def test_defaults(self):
        cfg = PipelineConfig()
        assert cfg.ring_size == 16
        assert cfg.max_batch_tokens == 128

    def test_invalid_ring_size(self):
        with pytest.raises(ValueError, match="ring_size"):
            PipelineConfig(ring_size=0)

    def test_invalid_max_batch(self):
        with pytest.raises(ValueError, match="max_batch_tokens"):
            PipelineConfig(max_batch_tokens=0)


class TestTokenPipelineV13:
    def test_single_stage_identity(self):
        pipe = TokenPipeline().add_stage("id", lambda x: x)
        assert pipe.process(42) == 42

    def test_chained_stages(self):
        pipe = (
            TokenPipeline()
            .add_stage("double", lambda x: x * 2)
            .add_stage("add10", lambda x: x + 10)
        )
        assert pipe.process(5) == 20

    def test_no_stages_passthrough(self):
        pipe = TokenPipeline()
        assert pipe.process(99) == 99

    def test_process_batch_results(self):
        pipe = TokenPipeline().add_stage("sq", lambda x: x ** 2)
        results = pipe.process_batch([1, 2, 3, 4])
        assert results == [1, 4, 9, 16]

    def test_batch_limit_respected(self):
        pipe = TokenPipeline(PipelineConfig(max_batch_tokens=3))
        pipe.add_stage("id", lambda x: x)
        results = pipe.process_batch([1, 2, 3, 4, 5])
        assert len(results) == 3

    def test_stats_tokens_processed(self):
        pipe = TokenPipeline().add_stage("id", lambda x: x)
        pipe.process(1)
        pipe.process(2)
        assert pipe.stats.tokens_processed == 2

    def test_stats_mean_latency_nonneg(self):
        pipe = TokenPipeline().add_stage("id", lambda x: x)
        for _ in range(5):
            pipe.process(0)
        assert pipe.stats.mean_latency_us >= 0.0

    def test_reset_stats(self):
        pipe = TokenPipeline().add_stage("id", lambda x: x)
        pipe.process(1)
        pipe.reset_stats()
        assert pipe.stats.tokens_processed == 0

    def test_drain_returns_buffered(self):
        pipe = TokenPipeline().add_stage("triple", lambda x: x * 3)
        pipe.process(2)
        pipe.process(3)
        items = pipe.drain()
        assert len(items) >= 1

    def test_non_callable_stage_raises(self):
        pipe = TokenPipeline()
        with pytest.raises(TypeError):
            pipe.add_stage("bad", "not_callable")  # type: ignore

    def test_n_stages(self):
        pipe = TokenPipeline()
        pipe.add_stage("a", lambda x: x)
        pipe.add_stage("b", lambda x: x)
        assert pipe.n_stages == 2

    def test_repr_contains_class_name(self):
        pipe = TokenPipeline().add_stage("test", lambda x: x)
        assert "TokenPipeline" in repr(pipe)
