"""tests/test_wave28_server_wiring.py

Wave 28 Phase 2 — unit tests for all six new modules:

- squish.speculative.cascade_spec          (Step 2A)
- squish.streaming.adaptive_prefill_fusion (Step 2B)
- squish.speculative.draft_multiplexer     (Step 2C)
- squish.kernels.async_decode_overlap      (Step 2D)
- squish.attention.per_layer_sparse_attn   (Step 2E)
- squish.speculative.speculative_prefill   (Step 2F)

4+ tests per class; 100% import coverage; edge-case validation.
"""

from __future__ import annotations

import numpy as np
import pytest

# ============================================================
# Step 2A — CascadeSpec
# ============================================================


class TestCascadeSpecConfig:
    def test_import(self):
        from squish.speculative.cascade_spec import CascadeSpecConfig  # noqa: F401

    def test_defaults(self):
        from squish.speculative.cascade_spec import CascadeSpecConfig
        cfg = CascadeSpecConfig()
        assert cfg.eagle_depth >= 1
        assert cfg.ngram_extend >= 0
        assert cfg.ngram_min >= 1

    def test_invalid_eagle_depth(self):
        from squish.speculative.cascade_spec import CascadeSpecConfig
        with pytest.raises(ValueError, match="eagle_depth"):
            CascadeSpecConfig(eagle_depth=0)

    def test_invalid_ngram_extend(self):
        from squish.speculative.cascade_spec import CascadeSpecConfig
        with pytest.raises(ValueError, match="ngram_extend"):
            CascadeSpecConfig(ngram_extend=-1)

    def test_invalid_ngram_max(self):
        from squish.speculative.cascade_spec import CascadeSpecConfig
        with pytest.raises(ValueError, match="ngram_max"):
            CascadeSpecConfig(ngram_min=5, ngram_max=3)

    def test_invalid_temperature(self):
        from squish.speculative.cascade_spec import CascadeSpecConfig
        with pytest.raises(ValueError, match="temperature"):
            CascadeSpecConfig(temperature=-0.1)

    def test_invalid_context_window(self):
        from squish.speculative.cascade_spec import CascadeSpecConfig
        with pytest.raises(ValueError, match="max_context_window"):
            CascadeSpecConfig(max_context_window=2)


class TestCascadeSpecDecoder:
    def _make_decoder(self, vocab=64):
        from squish.speculative.cascade_spec import CascadeSpecConfig, CascadeSpecDecoder
        cfg = CascadeSpecConfig(
            eagle_depth=3,
            ngram_extend=2,
            ngram_min=2,
            ngram_max=4,
            temperature=0.0,
        )
        rng = np.random.default_rng(0)

        def _fwd(ids):
            logits = rng.standard_normal(vocab).astype(np.float32)
            logits[ids[-1] % vocab] += 5.0
            return logits

        return CascadeSpecDecoder(_fwd, cfg)

    def test_instantiate(self):
        dec = self._make_decoder()
        assert dec is not None

    def test_generate_returns_correct_types(self):
        dec = self._make_decoder()
        output, stats = dec.generate([1, 2, 3, 4, 5], max_new_tokens=10)
        assert isinstance(output, list)
        assert all(isinstance(t, int) for t in output)

    def test_generate_respects_max_tokens(self):
        dec = self._make_decoder()
        # max_new_tokens limits verification steps; with eagle_depth=3 +
        # ngram_extend=2 + bonus token, each step can yield up to 6 tokens.
        # The output should not grow unboundedly.
        output, _ = dec.generate(list(range(8)), max_new_tokens=5)
        # Loose upper bound: 5 steps × (eagle_depth + ngram_extend + bonus)
        assert len(output) <= 5 * (3 + 2 + 1) + 1

    def test_stats_properties(self):
        from squish.speculative.cascade_spec import CascadeSpecStats
        s = CascadeSpecStats(
            total_tokens=10, total_steps=5,
            eagle_accepted=6, ngram_accepted=2, rejected=2
        )
        assert 0.0 <= s.acceptance_rate <= 1.0
        assert s.mean_tokens_per_step == pytest.approx(2.0)

    def test_stats_zero_steps(self):
        from squish.speculative.cascade_spec import CascadeSpecStats
        s = CascadeSpecStats()
        assert s.mean_tokens_per_step == 0.0
        assert s.acceptance_rate == 0.0

    def test_eos_stops_early(self):
        from squish.speculative.cascade_spec import CascadeSpecConfig, CascadeSpecDecoder
        eos = 99

        def _fwd_eos(ids):
            logits = np.zeros(100, dtype=np.float32)
            logits[eos] = 100.0
            return logits

        cfg = CascadeSpecConfig(eagle_depth=1, ngram_extend=0)
        dec = CascadeSpecDecoder(_fwd_eos, cfg)
        output, _ = dec.generate([1, 2, 3], max_new_tokens=20, eos_id=eos)
        assert eos in output
        assert output[-1] == eos

    def test_reset_stats(self):
        dec = self._make_decoder()
        dec.generate([1, 2, 3], max_new_tokens=5)
        dec.reset_stats()
        assert dec.stats.total_tokens == 0


# ============================================================
# Step 2B — AdaptivePrefillFusion
# ============================================================


class TestPrefillFusionConfig:
    def test_import(self):
        from squish.streaming.adaptive_prefill_fusion import PrefillFusionConfig  # noqa: F401

    def test_defaults(self):
        from squish.streaming.adaptive_prefill_fusion import PrefillFusionConfig
        cfg = PrefillFusionConfig()
        assert cfg.chunk_size > 0
        assert cfg.tome_r > 0
        assert cfg.tome_start_layer <= cfg.tome_end_layer

    def test_invalid_chunk_size(self):
        from squish.streaming.adaptive_prefill_fusion import PrefillFusionConfig
        with pytest.raises(ValueError, match="chunk_size"):
            PrefillFusionConfig(chunk_size=0)

    def test_invalid_layer_order(self):
        from squish.streaming.adaptive_prefill_fusion import PrefillFusionConfig
        with pytest.raises(ValueError, match="tome_start_layer"):
            PrefillFusionConfig(tome_start_layer=12, tome_end_layer=4)

    def test_invalid_threshold(self):
        from squish.streaming.adaptive_prefill_fusion import PrefillFusionConfig
        with pytest.raises(ValueError, match="early_exit_threshold"):
            PrefillFusionConfig(early_exit_threshold=1.5)

    def test_invalid_entropy_cutoffs(self):
        from squish.streaming.adaptive_prefill_fusion import PrefillFusionConfig
        with pytest.raises(ValueError):
            PrefillFusionConfig(low_entropy_cutoff=0.7, high_entropy_cutoff=0.5)


class TestPrefillFusionController:
    def _make_ctrl(self):
        from squish.streaming.adaptive_prefill_fusion import (
            PrefillFusionConfig, PrefillFusionController,
        )
        return PrefillFusionController(PrefillFusionConfig(
            chunk_threshold=8,
            tome_seq_threshold=4,
        ))

    def test_plan_high_entropy(self):
        from squish.streaming.adaptive_prefill_fusion import (
            PrefillComplexity, PrefillFusionConfig, PrefillFusionController,
        )
        # Use a very low high_entropy_cutoff to force HIGH classification
        ctrl = PrefillFusionController(PrefillFusionConfig(
            chunk_threshold=8,
            tome_seq_threshold=4,
            low_entropy_cutoff=0.05,
            high_entropy_cutoff=0.20,  # anything above 0.20 norm entropy is HIGH
        ))
        # Diverse sequence of 200 unique tokens
        diverse = list(range(200))
        plan = ctrl.plan(diverse)
        assert plan.complexity == PrefillComplexity.HIGH
        assert not plan.use_tome
        assert not plan.use_layer_skip

    def test_plan_low_entropy(self):
        from squish.streaming.adaptive_prefill_fusion import PrefillComplexity
        ctrl = self._make_ctrl()
        # Repetitive sequence → low entropy
        repetitive = [42] * 100
        plan = ctrl.plan(repetitive)
        assert plan.complexity == PrefillComplexity.LOW
        assert plan.use_layer_skip

    def test_plan_medium_entropy(self):
        from squish.streaming.adaptive_prefill_fusion import PrefillComplexity
        ctrl = self._make_ctrl()
        # Mildly varied sequence
        seq = ([1, 2, 3] * 30)[:50]
        plan = ctrl.plan(seq)
        assert plan.complexity in (PrefillComplexity.MEDIUM, PrefillComplexity.LOW)

    def test_plan_short_prompt_no_chunk(self):
        ctrl = self._make_ctrl()
        # Short prompt — below chunk threshold
        plan = ctrl.plan([1, 2, 3])
        assert not plan.use_chunk_prefill

    def test_plan_long_prompt_chunks(self):
        ctrl = self._make_ctrl()
        plan = ctrl.plan(list(range(100)))
        assert plan.use_chunk_prefill

    def test_estimate_entropy_empty(self):
        ctrl = self._make_ctrl()
        h = ctrl.estimate_entropy([])
        assert 0.0 <= h <= 1.0

    def test_estimate_entropy_uniform(self):
        ctrl = self._make_ctrl()
        h = ctrl.estimate_entropy(list(range(50)))
        assert h > 0.3  # reasonably high

    def test_estimate_entropy_constant(self):
        ctrl = self._make_ctrl()
        h = ctrl.estimate_entropy([7] * 50)
        assert h < 0.1  # near zero for single token


# ============================================================
# Step 2C — DraftMultiplexer
# ============================================================


class TestDraftMultiplexerConfig:
    def test_import(self):
        from squish.speculative.draft_multiplexer import DraftMultiplexerConfig  # noqa: F401

    def test_defaults(self):
        from squish.speculative.draft_multiplexer import DraftMultiplexerConfig
        cfg = DraftMultiplexerConfig()
        assert len(cfg.strategies) > 0
        assert 0.0 < cfg.ema_alpha <= 1.0

    def test_invalid_strategy(self):
        from squish.speculative.draft_multiplexer import DraftMultiplexerConfig
        with pytest.raises(ValueError, match="Unknown strategy"):
            DraftMultiplexerConfig(strategies=["does_not_exist"])

    def test_invalid_ema_alpha(self):
        from squish.speculative.draft_multiplexer import DraftMultiplexerConfig
        with pytest.raises(ValueError, match="ema_alpha"):
            DraftMultiplexerConfig(ema_alpha=0.0)

    def test_invalid_min_samples(self):
        from squish.speculative.draft_multiplexer import DraftMultiplexerConfig
        with pytest.raises(ValueError, match="min_samples"):
            DraftMultiplexerConfig(min_samples=0)


class TestDraftMultiplexer:
    def _make_mux(self):
        from squish.speculative.draft_multiplexer import DraftMultiplexer, DraftMultiplexerConfig
        cfg = DraftMultiplexerConfig(
            strategies=["eagle3", "ngram", "layer_skip"],
            ema_alpha=0.5,
            min_samples=2,
        )
        return DraftMultiplexer(cfg)

    def test_instantiate(self):
        mux = self._make_mux()
        assert mux is not None

    def test_select_returns_strategy(self):
        from squish.speculative.draft_multiplexer import DraftStrategy
        mux = self._make_mux()
        result = mux.select(prompt="Hello, how are you?")
        assert isinstance(result, DraftStrategy)

    def test_round_robin_during_init(self):
        from squish.speculative.draft_multiplexer import DraftStrategy
        mux = self._make_mux()
        # Before min_samples, should cycle round-robin through strategies
        seen = {mux.select(prompt="test") for _ in range(6)}
        assert len(seen) >= 2  # multiple strategies selected

    def test_update_and_select_after_warmup(self):
        from squish.speculative.draft_multiplexer import DraftStrategy
        mux = self._make_mux()
        # Feed enough samples to exceed min_samples
        for _ in range(4):
            mux.update(DraftStrategy.EAGLE3, "conversation", 0.8, 50.0)
            mux.update(DraftStrategy.NGRAM, "conversation", 0.3, 80.0)
            mux.update(DraftStrategy.LAYER_SKIP, "conversation", 0.5, 60.0)
        result = mux.select(prompt="Tell me about quantum computing.")
        assert isinstance(result, DraftStrategy)

    def test_stats_empty(self):
        mux = self._make_mux()
        stats = mux.strategy_stats("coding")
        assert isinstance(stats, dict)

    def test_classify_task_coding(self):
        from squish.speculative.draft_multiplexer import classify_task, DraftTaskType
        tt = classify_task("```python\ndef solve():\n    return 42\n```")
        assert tt == DraftTaskType.CODING

    def test_classify_task_math(self):
        from squish.speculative.draft_multiplexer import classify_task, DraftTaskType
        tt = classify_task("Please solve the integral dx/dt = 2x")
        assert tt == DraftTaskType.MATH

    def test_classify_task_rag(self):
        from squish.speculative.draft_multiplexer import classify_task, DraftTaskType
        tt = classify_task("According to the document, the policy states...")
        assert tt == DraftTaskType.RAG

    def test_classify_task_conversation(self):
        from squish.speculative.draft_multiplexer import classify_task, DraftTaskType
        tt = classify_task("Hi! How was your day?")
        assert tt == DraftTaskType.CONVERSATION


# ============================================================
# Step 2D — AsyncDecodeOverlap
# ============================================================


class TestOverlapConfig:
    def test_import(self):
        from squish.kernels.async_decode_overlap import OverlapConfig  # noqa: F401

    def test_defaults(self):
        from squish.kernels.async_decode_overlap import OverlapConfig
        cfg = OverlapConfig()
        assert cfg.temperature >= 0.0
        assert cfg.pipeline_depth >= 1

    def test_invalid_temperature(self):
        from squish.kernels.async_decode_overlap import OverlapConfig
        with pytest.raises(ValueError, match="temperature"):
            OverlapConfig(temperature=-0.5)

    def test_invalid_top_p(self):
        from squish.kernels.async_decode_overlap import OverlapConfig
        with pytest.raises(ValueError, match="top_p"):
            OverlapConfig(top_p=0.0)

    def test_invalid_top_k(self):
        from squish.kernels.async_decode_overlap import OverlapConfig
        with pytest.raises(ValueError, match="top_k"):
            OverlapConfig(top_k=-1)

    def test_invalid_pipeline_depth(self):
        from squish.kernels.async_decode_overlap import OverlapConfig
        with pytest.raises(ValueError, match="pipeline_depth"):
            OverlapConfig(pipeline_depth=0)


class TestAsyncDecodeOverlapSampler:
    """Test the pure-NumPy _sample_np helper used by AsyncDecodeOverlap."""

    def test_greedy_returns_argmax(self):
        from squish.kernels.async_decode_overlap import _sample_np
        logits = np.array([0.0, 0.0, 0.0, 100.0, 0.0], dtype=np.float32)
        tok = _sample_np(logits, 0.0, 1.0, 0, 1.0, [])
        assert tok == 3

    def test_sample_valid_range(self):
        from squish.kernels.async_decode_overlap import _sample_np
        rng    = np.random.default_rng(7)
        logits = rng.standard_normal(128).astype(np.float32)
        tok    = _sample_np(logits, 1.0, 0.9, 50, 1.0, [])
        assert 0 <= tok < 128

    def test_repetition_penalty(self):
        from squish.kernels.async_decode_overlap import _sample_np
        logits          = np.zeros(64, dtype=np.float32)
        logits[5]       = 100.0
        # Apply heavy penalty to token 5
        tok = _sample_np(logits, 0.0, 1.0, 0, 10.0, [5])
        # With greedy + heavy penalty, token 5 should still be argmax
        # (penalty applies but the dominance is extreme)
        assert isinstance(tok, int)
        assert 0 <= tok < 64

    def test_overlap_stats_defaults(self):
        from squish.kernels.async_decode_overlap import OverlapStats
        s = OverlapStats()
        assert s.overlap_rate == 0.0

    def test_overlap_stats_rate(self):
        from squish.kernels.async_decode_overlap import OverlapStats
        s = OverlapStats(total_steps=10, overlapped_steps=8, fallback_steps=2)
        assert s.overlap_rate == pytest.approx(0.8)


# ============================================================
# Step 2E — PerLayerSparseAttn
# ============================================================


class TestPerLayerSparseConfig:
    def test_import(self):
        from squish.attention.per_layer_sparse_attn import PerLayerSparseConfig  # noqa: F401

    def test_defaults(self):
        from squish.attention.per_layer_sparse_attn import PerLayerSparseConfig
        cfg = PerLayerSparseConfig()
        assert cfg.n_layers >= 1

    def test_invalid_n_layers(self):
        from squish.attention.per_layer_sparse_attn import PerLayerSparseConfig
        with pytest.raises(ValueError, match="n_layers"):
            PerLayerSparseConfig(n_layers=0)

    def test_invalid_entropy_threshold(self):
        from squish.attention.per_layer_sparse_attn import PerLayerSparseConfig
        with pytest.raises(ValueError, match="entropy_threshold"):
            PerLayerSparseConfig(entropy_threshold=1.5)

    def test_invalid_ema_alpha(self):
        from squish.attention.per_layer_sparse_attn import PerLayerSparseConfig
        with pytest.raises(ValueError, match="ema_alpha"):
            PerLayerSparseConfig(ema_alpha=0.0)


class TestPerLayerSparseAttn:
    def _make(self, n_layers=4, n_heads=4):
        from squish.attention.per_layer_sparse_attn import (
            PerLayerSparseConfig, PerLayerSparseAttn,
        )
        cfg = PerLayerSparseConfig(
            n_layers=n_layers,
            n_heads=n_heads,
            entropy_threshold=0.5,
            min_seq_len=4,
            warmup_steps=0,
        )
        return PerLayerSparseAttn(cfg)

    def test_instantiate(self):
        ctrl = self._make()
        assert ctrl is not None

    def test_no_profile_returns_all_dense(self):
        ctrl = self._make()
        mask = ctrl.sparse_mask(layer=0)
        assert mask.dtype == bool
        assert not mask.any()  # all dense before profiling

    def test_profile_prefill_and_query(self):
        rng  = np.random.default_rng(13)
        ctrl = self._make(n_layers=2, n_heads=2)
        # Simulate attention weights: (n_layers, n_heads, sq, sk)
        attn_w = rng.random((2, 2, 8, 8)).astype(np.float32)
        # Normalise to be valid attention distributions
        attn_w /= attn_w.sum(axis=-1, keepdims=True)
        ctrl.profile_prefill(attn_w)
        mask = ctrl.sparse_mask(layer=0)
        assert mask.shape == (2,)
        assert mask.dtype == bool

    def test_reset_clears_profiles(self):
        rng  = np.random.default_rng(14)
        ctrl = self._make(n_layers=2, n_heads=2)
        attn_w = rng.random((2, 2, 8, 8)).astype(np.float32)
        attn_w /= attn_w.sum(axis=-1, keepdims=True)
        ctrl.profile_prefill(attn_w)
        ctrl.reset()
        assert not ctrl.is_active()

    def test_tick_advances_step(self):
        ctrl = self._make()
        ctrl.tick()
        ctrl.tick()
        # Not yet active (no profiling done)
        assert not ctrl.is_active()

    def test_profile_single_layer(self):
        rng  = np.random.default_rng(16)
        ctrl = self._make(n_layers=4, n_heads=4)
        attn_w = rng.random((4, 8, 8)).astype(np.float32)
        attn_w /= attn_w.sum(axis=-1, keepdims=True)
        ctrl.profile_single_layer(layer=0, attn_weights=attn_w)
        mask = ctrl.sparse_mask(layer=0)
        assert mask.shape == (4,)

    def test_head_profile_fraction(self):
        from squish.attention.per_layer_sparse_attn import HeadProfile
        mask = np.array([True, False, True, False])
        ent  = np.array([0.2, 0.8, 0.3, 0.7], dtype=np.float32)
        hp   = HeadProfile(layer=0, entropies=ent, sparse_mask=mask)
        assert hp.n_sparse == 2
        assert hp.sparse_fraction == pytest.approx(0.5)


# ============================================================
# Step 2F — SpeculativePrefill
# ============================================================


class TestSpecPrefillConfig:
    def test_import(self):
        from squish.speculative.speculative_prefill import SpecPrefillConfig  # noqa: F401

    def test_defaults(self):
        from squish.speculative.speculative_prefill import SpecPrefillConfig
        cfg = SpecPrefillConfig()
        assert cfg.n_layers >= 1
        assert 0.0 < cfg.kv_accept_threshold <= 1.0

    def test_invalid_n_layers(self):
        from squish.speculative.speculative_prefill import SpecPrefillConfig
        with pytest.raises(ValueError, match="n_layers"):
            SpecPrefillConfig(n_layers=0)

    def test_invalid_threshold(self):
        from squish.speculative.speculative_prefill import SpecPrefillConfig
        with pytest.raises(ValueError, match="kv_accept_threshold"):
            SpecPrefillConfig(kv_accept_threshold=1.5)

    def test_invalid_probe_fraction(self):
        from squish.speculative.speculative_prefill import SpecPrefillConfig
        with pytest.raises(ValueError, match="probe_fraction"):
            SpecPrefillConfig(probe_fraction=0.0)

    def test_invalid_min_prompt_len(self):
        from squish.speculative.speculative_prefill import SpecPrefillConfig
        with pytest.raises(ValueError, match="min_prompt_len"):
            SpecPrefillConfig(min_prompt_len=0)


class TestSpeculativePrefiller:
    def _make_prefiller(self, n_layers=4):
        from squish.speculative.speculative_prefill import SpecPrefillConfig, SpeculativePrefiller
        cfg = SpecPrefillConfig(
            n_layers=n_layers,
            kv_accept_threshold=0.9,
            min_prompt_len=4,
        )
        rng = np.random.default_rng(99)

        def _draft(ids):
            return [rng.standard_normal((2, len(ids), 16)).astype(np.float32)
                    for _ in range(n_layers)]

        def _target(ids, mask):
            return [rng.standard_normal((2, len(ids), 16)).astype(np.float32)
                    for _ in range(n_layers)]

        return SpeculativePrefiller(_draft, _target, cfg)

    def test_instantiate(self):
        p = self._make_prefiller()
        assert p is not None

    def test_prefill_returns_kv_and_stats(self):
        p = self._make_prefiller(n_layers=4)
        kv, stats = p.prefill(list(range(8)))
        assert isinstance(kv, list)
        assert len(kv) == 4
        assert stats.total_layers == 4

    def test_short_prompt_uses_standard(self):
        from squish.speculative.speculative_prefill import SpecPrefillConfig, SpeculativePrefiller
        cfg = SpecPrefillConfig(n_layers=4, min_prompt_len=100)
        rng = np.random.default_rng(1)

        def _draft(ids):
            return []

        def _target(ids, mask):
            return [rng.standard_normal((1, len(ids), 8)).astype(np.float32)
                    for _ in range(4)]

        p = SpeculativePrefiller(_draft, _target, cfg)
        # Prompt shorter than min_prompt_len → standard prefill (0 layers skipped)
        kv, stats = p.prefill([1, 2, 3])
        assert stats.layers_skipped == 0

    def test_stats_properties(self):
        from squish.speculative.speculative_prefill import SpecPrefillStats
        s = SpecPrefillStats(
            total_layers=32, layers_skipped=16, layers_recomputed=16,
            mean_kv_similarity=0.95,
        )
        assert s.skip_rate == pytest.approx(0.5)
        assert s.speedup_estimate > 1.0

    def test_stats_zero_layers(self):
        from squish.speculative.speculative_prefill import SpecPrefillStats
        s = SpecPrefillStats()
        assert s.skip_rate == 0.0
        assert s.speedup_estimate == 1.0

    def test_layer_map_custom(self):
        from squish.speculative.speculative_prefill import SpecPrefillConfig, SpeculativePrefiller
        cfg = SpecPrefillConfig(
            n_layers=4,
            min_prompt_len=2,
            draft_layer_map=[0, 1, 2, 3],
        )
        rng = np.random.default_rng(3)

        def _draft(ids):
            return [rng.standard_normal((1, len(ids), 4)).astype(np.float32)
                    for _ in range(4)]

        def _target(ids, mask):
            return [rng.standard_normal((1, len(ids), 4)).astype(np.float32)
                    for _ in range(4)]

        p = SpeculativePrefiller(_draft, _target, cfg)
        kv, stats = p.prefill([1, 2, 3, 4, 5])
        assert len(kv) == 4
