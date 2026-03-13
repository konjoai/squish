"""
tests/test_branch_coverage.py

Targeted branch-coverage tests for non-raise, non-return-0.0 gaps
in Wave 19 / 21 / 24 / 25 modules. Each test covers exactly one
uncovered conditional branch that is not already excluded by the
coverage.py ``exclude_lines`` patterns.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

RNG = np.random.default_rng(0xABCD_1234)


# ---------------------------------------------------------------------------
# gqa.py
# ---------------------------------------------------------------------------


class TestGQABranches:
    def test_config_explicit_softmax_scale_false_branch(self):
        """GQAConfig with explicit softmax_scale skips the None-assignment branch."""
        from squish.gqa import GQAConfig

        cfg = GQAConfig(n_q_heads=4, n_kv_heads=2, head_dim=16, softmax_scale=0.25)
        # softmax_scale should remain at the explicit value (not overwritten).
        assert cfg.softmax_scale == pytest.approx(0.25)

    def test_grouped_query_attention_gqa_expansion(self):
        """grouped_query_attention with n_kv_heads < n_q_heads triggers GQA expansion."""
        from squish.gqa import GQAConfig, GQACache, grouped_query_attention

        cfg = GQAConfig(n_q_heads=4, n_kv_heads=2, head_dim=8, max_seq_len=16)
        cache = GQACache(cfg)
        for _ in range(5):
            k = RNG.standard_normal((cfg.n_kv_heads, cfg.head_dim)).astype(np.float32)
            v = RNG.standard_normal((cfg.n_kv_heads, cfg.head_dim)).astype(np.float32)
            cache.append(k, v)

        # q must be 3D: (n_q_heads, seq_q, head_dim)
        q = RNG.standard_normal((cfg.n_q_heads, 1, cfg.head_dim)).astype(np.float32)
        k, v = cache.get_kv()  # (n_kv_heads, seq_len, head_dim)
        out = grouped_query_attention(q, k, v, cfg)
        assert out.shape == (cfg.n_q_heads, 1, cfg.head_dim)


# ---------------------------------------------------------------------------
# sliding_window_attn.py
# ---------------------------------------------------------------------------


class TestSlidingWindowAttnBranches:
    def test_attention_empty_cache_returns_zeros(self):
        """sliding_window_attention with an empty cache returns a zeros tensor."""
        from squish.sliding_window_attn import SWAConfig, SlidingWindowKVCache
        from squish.sliding_window_attn import sliding_window_attention

        cfg = SWAConfig(window_size=8, n_heads=4, head_dim=16)
        cache = SlidingWindowKVCache(cfg)  # no appends → window_used == 0

        q = RNG.standard_normal((cfg.n_heads, cfg.head_dim)).astype(np.float32)
        out = sliding_window_attention(q, cache, cfg)

        assert out.shape == (cfg.n_heads, cfg.head_dim)
        np.testing.assert_array_equal(out, np.zeros_like(out))

    def test_attention_gqa_expansion(self):
        """sliding_window_attention with kv_n_heads < n_heads triggers GQA expansion."""
        from squish.sliding_window_attn import SWAConfig, SlidingWindowKVCache
        from squish.sliding_window_attn import sliding_window_attention

        cfg = SWAConfig(window_size=8, n_heads=4, head_dim=16, kv_n_heads=2)
        assert cfg.kv_n_heads == 2  # explicit, not defaulted

        cache = SlidingWindowKVCache(cfg)
        for _ in range(5):
            k = RNG.standard_normal((cfg.kv_n_heads, cfg.head_dim)).astype(np.float32)
            v = RNG.standard_normal((cfg.kv_n_heads, cfg.head_dim)).astype(np.float32)
            cache.append(k, v)

        q = RNG.standard_normal((cfg.n_heads, cfg.head_dim)).astype(np.float32)
        out = sliding_window_attention(q, cache, cfg)
        assert out.shape == (cfg.n_heads, cfg.head_dim)


# ---------------------------------------------------------------------------
# rope_scaling.py
# ---------------------------------------------------------------------------


class TestRopeScalingBranches:
    def test_longrope_scaler_head_dim_2_half_equals_1(self):
        """LongRoPEScaler.get_freqs: head_dim=2 → half=1 → else branch (not half>1)."""
        from squish.rope_scaling import RoPEConfig, create_rope_scaler

        # head_dim=2 is the smallest even value; half = head_dim // 2 = 1
        # This forces the else-branch of `if half > 1:` in get_freqs.
        cfg = RoPEConfig(
            head_dim=2,
            base_theta=10000.0,
            original_max_len=2048,
            target_max_len=4096,
            method="longrope",
            scale_factor=2.0,
        )
        scaler = create_rope_scaler(cfg)

        # get_freqs is called internally by apply(); 4 positions, head_dim=2
        pos_ids = np.array([0, 1, 2, 3], dtype=np.int32)
        x = np.ones((4, 1, 2), dtype=np.float32)  # (seq_len, n_heads, head_dim)
        out = scaler.apply(x, pos_ids)
        assert out.shape == (4, 1, 2)

    def test_rope_config_explicit_scale_factor_false_branch(self):
        """RoPEConfig with explicit scale_factor skips the None-assignment branch."""
        from squish.rope_scaling import RoPEConfig

        cfg = RoPEConfig(
            head_dim=64,
            base_theta=10000.0,
            original_max_len=2048,
            target_max_len=4096,
            method="ntk",
            scale_factor=4.0,
        )
        assert cfg.scale_factor == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# fused_rmsnorm.py
# ---------------------------------------------------------------------------


class TestFusedRMSNormBranches:
    def test_rms_norm_explicit_weight(self):
        """FusedRMSNorm with an explicit weight uses it (weight is not None branch)."""
        from squish.fused_rmsnorm import FusedNormConfig, FusedRMSNorm

        dim = 16
        cfg = FusedNormConfig(hidden_dim=dim, eps=1e-6, elementwise_scale=True)
        weight = np.ones(dim, dtype=np.float32) * 2.0
        norm = FusedRMSNorm(cfg, weight=weight)

        assert np.allclose(norm.weight, 2.0)

    def test_rms_norm_elementwise_scale_false(self):
        """FusedRMSNorm.forward with elementwise_scale=False skips the scale branch."""
        from squish.fused_rmsnorm import FusedNormConfig, FusedRMSNorm

        dim = 16
        cfg = FusedNormConfig(hidden_dim=dim, eps=1e-6, elementwise_scale=False)
        norm = FusedRMSNorm(cfg)

        x = RNG.standard_normal((4, dim)).astype(np.float32)
        out, _ = norm.forward(x)
        assert out.shape == (4, dim)

    def test_layer_norm_explicit_weight_and_bias(self):
        """FusedLayerNorm with explicit weight and bias uses both (not-None branches)."""
        from squish.fused_rmsnorm import FusedNormConfig, FusedLayerNorm

        dim = 16
        cfg = FusedNormConfig(hidden_dim=dim, eps=1e-5, elementwise_scale=True)
        weight = np.ones(dim, dtype=np.float32) * 0.5
        bias = np.zeros(dim, dtype=np.float32) + 0.1
        norm = FusedLayerNorm(cfg, weight=weight, bias=bias)

        assert np.allclose(norm.weight, 0.5)
        assert np.allclose(norm.bias, 0.1)

    def test_layer_norm_elementwise_scale_false(self):
        """FusedLayerNorm.forward with elementwise_scale=False skips the scale branch."""
        from squish.fused_rmsnorm import FusedNormConfig, FusedLayerNorm

        dim = 16
        cfg = FusedNormConfig(hidden_dim=dim, eps=1e-5, elementwise_scale=False)
        norm = FusedLayerNorm(cfg)

        x = RNG.standard_normal((4, dim)).astype(np.float32)
        out, _ = norm.forward(x)
        assert out.shape == (4, dim)


# ---------------------------------------------------------------------------
# paged_kv.py
# ---------------------------------------------------------------------------


class TestPagedKVBranches:
    def test_block_table_free_nonexistent_seq_noop(self):
        """BlockTable.free with an unknown seq_id is a no-op (early return branch)."""
        from squish.paged_kv import BlockTable

        bt = BlockTable(n_blocks=8, block_size=4)
        # free a seq_id that was never allocated → should be a no-op
        bt.free(999)  # no exception, no change
        assert bt.n_free_blocks == 8

    def test_paged_kv_cache_free_nonexistent_seq_noop(self):
        """PagedKVCache.free with an unknown seq_id is a no-op (early return branch)."""
        from squish.paged_kv import PagedKVConfig, PagedKVCache

        cfg = PagedKVConfig(block_size=4, n_blocks=8, n_heads=2, head_dim=16)
        cache = PagedKVCache(cfg)
        cache.free(seq_id=42)  # never allocated → no-op
        assert cache.n_sequences == 0


# ---------------------------------------------------------------------------
# lora_inference.py
# ---------------------------------------------------------------------------


class TestLoRAInferenceBranches:
    def test_config_explicit_target_modules_false_branch(self):
        """LoRAConfig with explicit target_modules skips the None-assignment branch."""
        from squish.lora_inference import LoRAConfig

        cfg = LoRAConfig(rank=8, alpha=16.0, target_modules=("q_proj", "k_proj"))
        assert cfg.target_modules == ("q_proj", "k_proj")

    def test_apply_unregistered_module_returns_base(self):
        """LoRAInferenceAdapter.apply with an unregistered module returns base unchanged."""
        from squish.lora_inference import LoRAConfig, LoRAInferenceAdapter

        cfg = LoRAConfig(rank=4, alpha=8.0)
        adapter = LoRAInferenceAdapter(cfg)

        x = RNG.standard_normal((2, 32)).astype(np.float32)
        base = RNG.standard_normal((2, 32)).astype(np.float32)

        # "mlp" was never registered → should return base_output unchanged
        out = adapter.apply("mlp", x, base)
        np.testing.assert_array_equal(out, base)


# ---------------------------------------------------------------------------
# flash_decode.py
# ---------------------------------------------------------------------------


class TestFlashDecodeBranches:
    def test_config_explicit_softmax_scale_false_branch(self):
        """FlashDecodeConfig with explicit softmax_scale skips the None-assignment."""
        from squish.flash_decode import FlashDecodeConfig

        cfg = FlashDecodeConfig(n_heads=4, head_dim=64, n_splits=4, softmax_scale=0.1)
        assert cfg.softmax_scale == pytest.approx(0.1)

    def test_config_explicit_kv_n_heads_false_branch(self):
        """FlashDecodeConfig with explicit kv_n_heads skips the None-assignment."""
        from squish.flash_decode import FlashDecodeConfig

        cfg = FlashDecodeConfig(n_heads=8, head_dim=64, n_splits=4, kv_n_heads=2)
        assert cfg.kv_n_heads == 2

    def test_decode_break_when_split_start_exceeds_seq_len(self):
        """FlashDecodeAttention.decode triggers break when start >= seq_len."""
        from squish.flash_decode import FlashDecodeConfig, FlashDecodeAttention

        # With seq_len=5 and n_splits=4, split_len=ceil(5/4)=2.
        # Splits: start=0 (ok), start=2 (ok), start=4 (ok), start=6 >= 5 → break.
        n_heads, head_dim, n_splits = 4, 32, 4
        seq_len = 5

        cfg = FlashDecodeConfig(n_heads=n_heads, head_dim=head_dim, n_splits=n_splits)
        attn = FlashDecodeAttention(cfg)

        q = RNG.standard_normal((n_heads, head_dim)).astype(np.float32)
        k = RNG.standard_normal((n_heads, seq_len, head_dim)).astype(np.float32)
        v = RNG.standard_normal((n_heads, seq_len, head_dim)).astype(np.float32)

        out = attn.decode(q, k, v)
        assert out.shape == (n_heads, head_dim)


# ---------------------------------------------------------------------------
# flash_prefill.py
# ---------------------------------------------------------------------------


class TestFlashPrefillBranches:
    def test_config_explicit_softmax_scale_false_branch(self):
        """PrefillConfig with explicit softmax_scale skips the None-assignment."""
        from squish.flash_prefill import PrefillConfig

        cfg = PrefillConfig(n_heads=4, head_dim=64, softmax_scale=0.2)
        assert cfg.softmax_scale == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# prefix_pool.py
# ---------------------------------------------------------------------------


class TestPrefixPoolBranches:
    def test_config_explicit_kv_n_heads_elif_branch(self):
        """PrefixPoolConfig with explicit valid kv_n_heads takes the elif-else path."""
        from squish.prefix_pool import PrefixPoolConfig

        # kv_n_heads is not None (skip if-branch) and > 0 (skip elif raise-branch).
        cfg = PrefixPoolConfig(
            max_entries=32, n_heads=4, head_dim=8, kv_n_heads=2
        )
        assert cfg.kv_n_heads == 2

    def test_prefix_entry_explicit_last_used_skips_assignment(self):
        """PrefixEntry with explicit non-zero last_used skips the assignment branch."""
        from squish.prefix_pool import PrefixEntry
        import numpy as np

        keys = np.zeros((1, 4, 8), dtype=np.float32)
        vals = np.zeros((1, 4, 8), dtype=np.float32)
        entry = PrefixEntry(
            prefix_hash="deadbeef",
            keys=keys,
            values=vals,
            last_used=1234567890.0,
        )
        # last_used was not 0.0, so __post_init__ should not overwrite it.
        assert entry.last_used == pytest.approx(1234567890.0)


# ---------------------------------------------------------------------------
# kv_defrag.py
# ---------------------------------------------------------------------------


class TestKVDefragBranches:
    def test_defrag_empty_pool_n_alloc_zero_branch(self):
        """KVDefragmenter.defrag() with no allocations takes the n_alloc==0 path."""
        from squish.kv_defrag import KVDefragmenter

        d = KVDefragmenter(page_size=4, n_heads=2, head_dim=4)
        # No allocations → n_alloc == 0 → early return with zeroed DefragStats.
        stats = d.defrag()
        assert stats.n_pages_before == stats.n_pages_after
        assert stats.bytes_freed == 0


# ---------------------------------------------------------------------------
# sparse_weight.py
# ---------------------------------------------------------------------------


class TestSparseWeightBranches:
    def test_memory_bytes_before_compress_returns_zero(self):
        """SparseWeightStore.memory_bytes() before compress() returns 0."""
        from squish.sparse_weight import SparsityConfig, SparseWeightStore

        cfg = SparsityConfig(N=2, M=4)
        store = SparseWeightStore(cfg)
        # _values is still None → returns integer 0
        assert store.memory_bytes() == 0


# ---------------------------------------------------------------------------
# mix_kvq.py — additional guard branches
# ---------------------------------------------------------------------------


class TestMixKVQBranches:
    def test_channel_scorer_difficulty_empty(self):
        """ChannelScorer.difficulty before any record → returns zeros (not filled)."""
        from squish.mix_kvq import MixKVQConfig, ChannelScorer

        n_ch = 8
        scorer = ChannelScorer(n_channels=n_ch, config=MixKVQConfig())
        # _pos == 0 and not _filled → returns np.zeros
        scores = scorer.difficulty()
        assert scores.shape == (n_ch,)
        np.testing.assert_array_equal(scores, np.zeros(n_ch, dtype=np.float32))

    def test_channel_scorer_assign_bits_empty_key_matrix(self):
        """ChannelScorer.assign_bits with 0-row key matrix → returns zeros."""
        from squish.mix_kvq import MixKVQConfig, ChannelScorer

        n_ch = 8
        cfg_obj = MixKVQConfig()
        scorer = ChannelScorer(n_channels=n_ch, config=cfg_obj)
        query = RNG.standard_normal(n_ch).astype(np.float32)
        empty_keys = np.zeros((0, n_ch), dtype=np.float32)
        result = scorer.assign_bits(query, empty_keys)
        assert result.shape == (n_ch,)


# ---------------------------------------------------------------------------
# parallel_sampler.py — n==1 diversity branch
# ---------------------------------------------------------------------------


class TestParallelSamplerBranches:
    def test_diversity_score_single_sample(self):
        """ParallelSampler._diversity_score with a single sample → returns zeros."""
        from squish.parallel_sampler import ParallelSampler, DiversityConfig

        cfg = DiversityConfig(n_samples=1, temperature=1.0, seed=0)
        ps = ParallelSampler(cfg)
        vocab = 32
        logits = RNG.standard_normal(vocab).astype(np.float32)
        result = ps.sample(logits)
        # With n=1 the diversity metric returns 0 (single sample → no spread).
        assert result.diversity_score == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# mx_quant.py — zero-tile guard branch
# ---------------------------------------------------------------------------


class TestMxQuantBranches:
    def test_quantize_all_zero_tile(self):
        """MXQuantizer._encode_tile with all-zero tile hits the amax==0 guard."""
        from squish.mx_quant import MXConfig, MXQuantizer

        cfg = MXConfig(tile_size=4)  # tile_size must be a power of 2
        quantizer = MXQuantizer(cfg)
        zero_tile = np.zeros(4, dtype=np.float32)
        packed, e8m0 = quantizer._encode_tile(zero_tile)
        assert e8m0 == 0
        np.testing.assert_array_equal(packed, np.zeros(4, dtype=np.uint8))


# ---------------------------------------------------------------------------
# quant_spec.py — uniform-range guard branch
# ---------------------------------------------------------------------------


class TestQuantSpecBranches:
    def test_quantize_uniform_range(self):
        """DraftQuantizer.quantize with t_min==t_max → all-zero codes + scale=1."""
        from squish.quant_spec import DraftQuantizer

        quantizer = DraftQuantizer(bits=8)
        uniform = np.ones(8, dtype=np.float32) * 3.14
        codes, scale, zero = quantizer.quantize(uniform)
        # t_min == t_max → scale should be 1.0 and codes all zero
        assert scale == pytest.approx(1.0)
        np.testing.assert_array_equal(codes, np.zeros(8, dtype=np.int8))


# ---------------------------------------------------------------------------
# lora_compose.py — no-adapters guard branch
# ---------------------------------------------------------------------------


class TestLoRAComposeBranches:
    def test_compose_weights_none_guard(self):
        """LoRAComposer.forward with no adapters loaded returns zeros early."""
        from squish.lora_compose import LoRAComposer

        composer = LoRAComposer(hidden_dim=8)
        x = RNG.standard_normal((2, 8)).astype(np.float32)
        # No adapters registered → if not self._adapters: → returns zeros
        out = composer.forward(x)
        assert out.shape == (2, 8)
        np.testing.assert_array_equal(out, np.zeros((2, 8), dtype=np.float32))


# ---------------------------------------------------------------------------
# shadow_kv.py — empty projection guard
# ---------------------------------------------------------------------------


class TestShadowKVBranches:
    def test_recall_empty_sequence_guard(self):
        """ShadowKVCache.recall with no stored data returns empty arrays."""
        from squish.shadow_kv import ShadowKVCache, ShadowKVConfig

        cfg = ShadowKVConfig(svd_rank=4, n_landmarks=8)
        cache = ShadowKVCache(n_layers=2, n_heads=2, head_dim=8, config=cfg)
        # Never stored anything for layer 0 → all_proj.shape[0] == 0 branch
        query = np.zeros((2, 8), dtype=np.float32)
        keys, vals = cache.recall(layer_id=0, query=query)
        assert keys.shape[0] == 0  # no tokens → empty return


# ---------------------------------------------------------------------------
# ada_serve.py — goodput_rate, estimated_goodput_improvement, mean_gamma branches
# ---------------------------------------------------------------------------


class TestAdaServeBranches:
    def test_goodput_rate_zero_tokens_returns_one(self):
        """AdaServeStats.goodput_rate with zero tokens returns 1.0 (not excluded)."""
        from squish.ada_serve import AdaServeStats

        stats = AdaServeStats()
        # total_tokens_generated == 0 → return 1.0 (NOT 0.0, so NOT excluded)
        assert stats.goodput_rate == pytest.approx(1.0)

    def test_goodput_rate_false_branch_with_tokens(self):
        """AdaServeStats.goodput_rate False branch: total_tokens_generated > 0."""
        from squish.ada_serve import AdaServeStats

        stats = AdaServeStats()
        # slo_met=True → goodput_tokens incremented (True branch of record_request)
        stats.record_request(gamma_used=4, tokens_generated=10, slo_met=True)
        # total_tokens_generated == 10 > 0 → takes the False branch
        assert stats.goodput_rate == pytest.approx(1.0)

    def test_record_request_slo_not_met_increments_violations(self):
        """AdaServeStats.record_request with slo_met=False covers the else branch."""
        from squish.ada_serve import AdaServeStats

        stats = AdaServeStats()
        # slo_met=False → total_slo_violations incremented (False branch)
        stats.record_request(gamma_used=4, tokens_generated=5, slo_met=False)
        assert stats.total_slo_violations == 1
        # slo_violation_rate False branch: total_requests > 0
        assert stats.slo_violation_rate == pytest.approx(1.0)

    def test_estimated_goodput_improvement_empty_histogram(self):
        """AdaServeStats.estimated_goodput_improvement_vs_fixed with empty histogram."""
        from squish.ada_serve import AdaServeStats

        stats = AdaServeStats()
        # gamma_histogram is empty → return 1.0 (NOT 0.0, so NOT excluded)
        assert stats.estimated_goodput_improvement_vs_fixed == pytest.approx(1.0)

    def test_estimated_goodput_improvement_multiple_gammas(self):
        """AdaServeStats.estimated_goodput_improvement_vs_fixed False branch."""
        from squish.ada_serve import AdaServeStats

        stats = AdaServeStats()
        stats.record_request(gamma_used=2, tokens_generated=5, slo_met=True)
        stats.record_request(gamma_used=6, tokens_generated=5, slo_met=True)
        # len(gamma_histogram) >= 2 → takes the False branch → returns > 1.0
        assert stats.estimated_goodput_improvement_vs_fixed > 1.0

    def test_mean_gamma_non_empty_histogram(self):
        """AdaServeStats.mean_gamma with non-empty histogram covers the False branch."""
        from squish.ada_serve import AdaServeStats

        stats = AdaServeStats()
        stats.record_request(gamma_used=4, tokens_generated=10, slo_met=True)
        # gamma_histogram not empty → False branch of `if not self.gamma_histogram:`
        assert stats.mean_gamma == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# distil_spec.py — kl_history guard branches
# ---------------------------------------------------------------------------


class TestDistilSpecBranches:
    def test_stats_with_empty_history(self):
        """DistilSpecCalibrator.stats() with no recorded steps returns DistilStats(n_steps=0)."""
        from squish.distil_spec import DistilConfig, DistilSpecCalibrator

        cfg = DistilConfig()
        spec = DistilSpecCalibrator(cfg)
        # No steps recorded → if not self._kl_history: return DistilStats(n_steps=0)
        # This branch is NOT excluded (returns a DistilStats object, not 0.0).
        s = spec.stats()
        assert s.n_steps == 0

    def test_mean_kl_with_history(self):
        """DistilSpecCalibrator.mean_kl with non-empty history covers the False branch."""
        from squish.distil_spec import DistilConfig, DistilSpecCalibrator

        cfg = DistilConfig()
        spec = DistilSpecCalibrator(cfg)
        logits = RNG.standard_normal(64).astype(np.float32)
        # Record a step so _kl_history is non-empty
        spec.record_step(logits, logits)  # same logits → KL near 0
        # mean_kl False branch: not self._kl_history is False → returns float(np.mean(...))
        kl = spec.mean_kl
        assert isinstance(kl, float)
        assert kl >= 0.0


# ---------------------------------------------------------------------------
# pipeline_bubble.py — empty schedule guard
# ---------------------------------------------------------------------------


class TestPipelineBubbleBranches:
    def test_n_slots_empty_schedule_returns_zero(self):
        """StageSchedule.n_slots with empty schedule returns 0 (int, not excluded)."""
        from squish.pipeline_bubble import StageSchedule

        sched = StageSchedule(schedule=[])
        # n_slots: if not self.schedule: return 0 (int, NOT excluded by return 0.0 rule)
        assert sched.n_slots == 0

    def test_n_slots_nonempty_schedule_false_branch(self):
        """StageSchedule.n_slots with a non-empty schedule takes the False branch."""
        from squish.pipeline_bubble import BubbleEliminator, StageConfig

        cfg = StageConfig(n_stages=2, n_microbatches=4, stage_latency_ms=1.0)
        elim = BubbleEliminator(cfg)
        sched = elim.build_schedule()
        # n_slots False branch: max(len(row) for row in self.schedule)
        assert sched.n_slots > 0


# ---------------------------------------------------------------------------
# budget_spec.py — remaining <= 0 guard
# ---------------------------------------------------------------------------


class TestBudgetSpecBranches:
    def test_budget_next_draft_exhausted_returns_zero(self):
        """BudgetSpecDecoder.effective_draft_len with exhausted budget returns 0."""
        from squish.budget_spec import BudgetConfig, BudgetSpecDecoder

        cfg = BudgetConfig(total_budget=4, n_draft=2)
        ctrl = BudgetSpecDecoder(cfg)
        # Exhaust the budget by accepting all 4 tokens at once
        ctrl.step(n_accepted=4)
        # remaining = total_budget - tokens_generated = 4 - 4 = 0 → return 0 (int)
        count = ctrl.effective_draft_len()
        assert count == 0


# ---------------------------------------------------------------------------
# sparse_verify.py — n_draft <= 1 guard
# ---------------------------------------------------------------------------


class TestSparseVerifyBranches:
    def test_estimated_reuse_single_draft(self):
        """SparseVerifyPass._simulate_reuse with n_draft<=1 returns 0."""
        from squish.sparse_verify import SparseVerifyConfig, SparseVerifyPass

        cfg = SparseVerifyConfig()
        svp = SparseVerifyPass(
            verify_fn=lambda ctx, draft: (draft, []),
            config=cfg,
        )
        # n_draft=1 → the `if n_draft <= 1: return 0` guard
        reuse = svp._simulate_reuse(n_draft=1, ctx_len=10)
        assert reuse == 0
