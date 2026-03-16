"""Unit tests achieving 100% coverage for 6 experimental modules:
    ternary_quant, retention_attn, cot_compress, act_sparsity,
    delta_compress, flash_mla.
"""

import numpy as np
import pytest

from squish.attention.flash_mla import FlashMLACache, MLAConfig
from squish.attention.retention_attn import (
    RetentionConfig,
    RetentionKernel,
    RetentionState,
    RetentionStats,
)
from squish.context.cot_compress import CoTCompressor, CoTConfig, CoTStats
from squish.context.delta_compress import DeltaCompressor, DeltaConfig, DeltaStats
from squish.quant.ternary_quant import TernaryConfig, TernaryQuantizer, TernaryStats
from squish.token.act_sparsity import (
    ActSparsityPredictor,
    ActSparsityStats,
    SparseFFNGate,
    SparsityConfig,
)

# ---------------------------------------------------------------------------
# TernaryQuant
# ---------------------------------------------------------------------------


class TestTernaryQuantCoverage:
    def test_config_zero_threshold_raises(self) -> None:
        with pytest.raises(ValueError, match="zero_threshold must be strictly positive"):
            TernaryConfig(zero_threshold=0.0)

    def test_config_negative_threshold_raises(self) -> None:
        with pytest.raises(ValueError, match="zero_threshold must be strictly positive"):
            TernaryConfig(zero_threshold=-1.0)

    def test_stats_sparsity_zero_case(self) -> None:
        # total_weights_quantized == 0 → denominator uses max(1, 0) = 1 → 0.0
        stats = TernaryStats()
        assert stats.sparsity == 0.0

    def test_quantize_produces_ternary_values(self) -> None:
        cfg = TernaryConfig(zero_threshold=0.5)
        qtz = TernaryQuantizer(cfg)
        weights = np.array([0.0, 0.0, 1.0, -1.0], dtype=np.float32)
        ternary, scale = qtz.quantize(weights)
        assert ternary.dtype == np.int8
        assert set(ternary.flatten().tolist()).issubset({-1, 0, 1})
        assert scale > 0.0

    def test_stats_sparsity_after_quantize(self) -> None:
        cfg = TernaryConfig(zero_threshold=0.5)
        qtz = TernaryQuantizer(cfg)
        weights = np.array([0.0, 0.0, 1.0, -1.0], dtype=np.float32)
        qtz.quantize(weights)
        assert 0.0 <= qtz.stats.sparsity <= 1.0
        assert qtz.stats.total_quantize_calls == 1

    def test_dequantize_returns_float32(self) -> None:
        cfg = TernaryConfig(zero_threshold=0.5)
        qtz = TernaryQuantizer(cfg)
        weights = np.array([1.0, -1.0, 0.5, -0.5], dtype=np.float32)
        ternary, scale = qtz.quantize(weights)
        recon = qtz.dequantize(ternary, scale)
        assert recon.shape == weights.shape
        assert recon.dtype == np.float32


# ---------------------------------------------------------------------------
# RetentionAttn
# ---------------------------------------------------------------------------


class TestRetentionAttnCoverage:
    def test_config_hidden_dim_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="hidden_dim must be >= 1"):
            RetentionConfig(hidden_dim=0)

    def test_config_n_heads_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="n_heads must be >= 1"):
            RetentionConfig(n_heads=0)

    def test_config_not_divisible_raises(self) -> None:
        with pytest.raises(ValueError, match="must be divisible by"):
            RetentionConfig(hidden_dim=9, n_heads=4)

    def test_config_gamma_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="gamma must be in"):
            RetentionConfig(hidden_dim=8, n_heads=4, gamma=0.0)

    def test_config_head_dim_property(self) -> None:
        cfg = RetentionConfig(hidden_dim=8, n_heads=4, gamma=0.9)
        assert cfg.head_dim == 2

    def test_step_normal_path(self) -> None:
        cfg = RetentionConfig(hidden_dim=8, n_heads=4, gamma=0.9)
        kernel = RetentionKernel(cfg)
        state = kernel.init_state()
        assert kernel.stats.total_states_init == 1
        rng = np.random.default_rng(1)
        q = rng.standard_normal((4, 2)).astype(np.float32)
        k = rng.standard_normal((4, 2)).astype(np.float32)
        v = rng.standard_normal((4, 2)).astype(np.float32)
        output, new_state = kernel.step(q, k, v, state)
        assert output.shape == (4, 2)
        assert output.dtype == np.float32
        assert new_state.step == 1
        assert kernel.stats.total_steps == 1

    def test_step_wrong_q_shape_raises(self) -> None:
        cfg = RetentionConfig(hidden_dim=8, n_heads=4, gamma=0.9)
        kernel = RetentionKernel(cfg)
        state = kernel.init_state()
        q = np.zeros((4, 3), dtype=np.float32)  # wrong head_dim (should be 2)
        k = np.zeros((4, 2), dtype=np.float32)
        v = np.zeros((4, 2), dtype=np.float32)
        with pytest.raises(ValueError, match="Expected q shape"):
            kernel.step(q, k, v, state)

    def test_step_wrong_state_shape_raises(self) -> None:
        cfg = RetentionConfig(hidden_dim=8, n_heads=4, gamma=0.9)
        kernel = RetentionKernel(cfg)
        q = np.zeros((4, 2), dtype=np.float32)
        k = np.zeros((4, 2), dtype=np.float32)
        v = np.zeros((4, 2), dtype=np.float32)
        bad_state = RetentionState(S=np.zeros((4, 2, 3), dtype=np.float32))
        with pytest.raises(ValueError, match="state.S must have shape"):
            kernel.step(q, k, v, bad_state)

    def test_reset_stats_clears_counters(self) -> None:
        cfg = RetentionConfig(hidden_dim=8, n_heads=4, gamma=0.9)
        kernel = RetentionKernel(cfg)
        state = kernel.init_state()
        q = np.zeros((4, 2), dtype=np.float32)
        k = np.zeros((4, 2), dtype=np.float32)
        v = np.zeros((4, 2), dtype=np.float32)
        kernel.step(q, k, v, state)
        assert kernel.stats.total_steps == 1
        kernel.reset_stats()
        assert kernel.stats.total_steps == 0
        assert kernel.stats.total_states_init == 0


# ---------------------------------------------------------------------------
# CoTCompress
# ---------------------------------------------------------------------------


class TestCoTCompressCoverage:
    def test_config_compress_ratio_one_raises(self) -> None:
        with pytest.raises(ValueError, match="compress_ratio must be in"):
            CoTConfig(compress_ratio=1.0)

    def test_config_min_tokens_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="min_tokens must be >= 1"):
            CoTConfig(min_tokens=0)

    def test_config_smoothing_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="smoothing must be > 0"):
            CoTConfig(smoothing=0.0)

    def test_compress_non_1d_token_ids_raises(self) -> None:
        cfg = CoTConfig()
        compressor = CoTCompressor(cfg)
        tokens = np.zeros((5, 3), dtype=np.int64)
        with pytest.raises(ValueError, match="token_ids must be 1-D"):
            compressor.compress(tokens)

    def test_compress_early_return_when_input_shorter_than_min_tokens(self) -> None:
        # 5 tokens < min_tokens=16 → n_keep gets clamped to n=5 → early return
        cfg = CoTConfig(compress_ratio=0.4, min_tokens=16)
        compressor = CoTCompressor(cfg)
        tokens = np.array([1, 2, 3, 4, 5], dtype=np.int64)
        result = compressor.compress(tokens)
        assert len(result) == 5
        assert compressor.stats.total_compress_calls == 1
        assert compressor.stats.total_tokens_out == 5

    def test_compress_normal_path_reduces_tokens(self) -> None:
        # compress_ratio=0.4 with min_tokens=1: n_keep = round(0.6 * 100) = 60
        cfg = CoTConfig(compress_ratio=0.4, min_tokens=1)
        compressor = CoTCompressor(cfg)
        rng = np.random.default_rng(0)
        tokens = rng.integers(0, 50_000, size=100, dtype=np.int64)
        result = compressor.compress(tokens)
        assert len(result) == 60

    def test_stats_avg_compression_ratio_zero_case(self) -> None:
        # When no tokens processed: 1.0 - 0/(0+1e-9) = 1.0
        stats = CoTStats()
        assert stats.avg_compression_ratio == pytest.approx(1.0, abs=1e-3)

    def test_stats_avg_compression_ratio_after_call(self) -> None:
        cfg = CoTConfig(compress_ratio=0.4, min_tokens=1)
        compressor = CoTCompressor(cfg)
        rng = np.random.default_rng(0)
        tokens = rng.integers(0, 50_000, size=100, dtype=np.int64)
        compressor.compress(tokens)
        # Keeps 60/100 tokens → compression ratio = 1 - 60/100 = 0.4
        assert compressor.stats.avg_compression_ratio == pytest.approx(0.4, abs=0.01)


# ---------------------------------------------------------------------------
# ActSparsity
# ---------------------------------------------------------------------------


class TestActSparsityCoverage:
    def test_config_hidden_dim_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="hidden_dim must be a positive integer"):
            SparsityConfig(hidden_dim=0)

    def test_config_n_layers_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="n_layers must be a positive integer"):
            SparsityConfig(n_layers=0)

    def test_config_threshold_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="threshold must be non-negative"):
            SparsityConfig(threshold=-0.1)

    def test_config_min_sparsity_too_large_raises(self) -> None:
        with pytest.raises(ValueError, match="min_sparsity must be in"):
            SparsityConfig(min_sparsity=1.5)

    def test_config_calibration_steps_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="calibration_steps must be a positive integer"):
            SparsityConfig(calibration_steps=0)

    def test_record_layer_idx_negative_raises(self) -> None:
        config = SparsityConfig()
        predictor = ActSparsityPredictor(config)
        acts = np.zeros((10, 4096), dtype=np.float32)
        with pytest.raises(ValueError, match="layer_idx must be in"):
            predictor.record(-1, acts)

    def test_record_3d_activations_raises(self) -> None:
        config = SparsityConfig()
        predictor = ActSparsityPredictor(config)
        acts = np.zeros((2, 10, 4096), dtype=np.float32)
        with pytest.raises(ValueError, match="activations must be 2-D"):
            predictor.record(0, acts)

    def test_should_skip_returns_true_for_all_zero_activations(self) -> None:
        # All-zero activations → 100% near-zero → sparsity > 50% → should_skip True
        config = SparsityConfig()
        predictor = ActSparsityPredictor(config)
        acts = np.zeros((10, 4096), dtype=np.float32)
        predictor.record(0, acts)
        assert predictor.should_skip(0) is True

    def test_should_skip_returns_false_for_large_activations(self) -> None:
        config = SparsityConfig()
        predictor = ActSparsityPredictor(config)
        acts = np.ones((10, 4096), dtype=np.float32) * 100.0
        predictor.record(0, acts)
        assert predictor.should_skip(0) is False

    def test_get_sparsity_returns_min_sparsity_when_no_data(self) -> None:
        # total == 0 for unrecorded layer → returns min_sparsity floor
        config = SparsityConfig(min_sparsity=0.1)
        predictor = ActSparsityPredictor(config)
        assert predictor.get_sparsity(0) == pytest.approx(0.1)

    def test_calibrate_returns_sparsity_map(self) -> None:
        config = SparsityConfig()
        predictor = ActSparsityPredictor(config)
        acts = np.zeros((10, 4096), dtype=np.float32)
        predictor.record(0, acts)
        sparsity_map = predictor.calibrate()
        assert 0 in sparsity_map
        assert sparsity_map[0] == pytest.approx(1.0)

    def test_reset_clears_all_data(self) -> None:
        config = SparsityConfig()
        predictor = ActSparsityPredictor(config)
        acts = np.zeros((10, 4096), dtype=np.float32)
        predictor.record(0, acts)
        predictor.reset()
        assert predictor.calibrate() == {}

    def test_sparse_ffn_gate_invalid_layer_idx_raises(self) -> None:
        config = SparsityConfig()
        with pytest.raises(ValueError, match="layer_idx must be in"):
            SparseFFNGate(config, layer_idx=config.n_layers)

    def test_sparse_ffn_gate_compression_ratio_before_apply_returns_one(self) -> None:
        config = SparsityConfig()
        gate = SparseFFNGate(config, layer_idx=0)
        assert gate.compression_ratio() == 1.0

    def test_sparse_ffn_gate_apply_and_compression_ratio(self) -> None:
        config = SparsityConfig(threshold=0.5)
        gate = SparseFFNGate(config, layer_idx=0)
        acts = np.array([[0.1, 0.9, 0.0, 1.0]], dtype=np.float32)
        masked = gate.apply(acts)
        assert masked.shape == acts.shape
        ratio = gate.compression_ratio()
        assert 0.0 <= ratio <= 1.0

    def test_act_sparsity_stats_sparsity_rate_zero_case(self) -> None:
        stats = ActSparsityStats()
        assert stats.sparsity_rate == 0.0

    def test_act_sparsity_stats_sparsity_rate_nonzero(self) -> None:
        stats = ActSparsityStats(total_activations_seen=100, total_zeros=40)
        assert stats.sparsity_rate == pytest.approx(0.4)

    def test_act_sparsity_stats_skip_rate_zero_case(self) -> None:
        stats = ActSparsityStats()
        assert stats.skip_rate == 0.0

    def test_act_sparsity_stats_skip_rate_nonzero(self) -> None:
        stats = ActSparsityStats(
            total_activations_seen=5,
            total_zeros=2,
            total_skipped_layers=5,
        )
        # skip_rate = 5 / (5 + 5) = 0.5
        assert stats.skip_rate == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# DeltaCompress
# ---------------------------------------------------------------------------


class TestDeltaCompressCoverage:
    def test_config_rank_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="rank must be >= 1"):
            DeltaConfig(rank=0)

    def test_compress_1d_base_raises(self) -> None:
        cfg = DeltaConfig(rank=4)
        compressor = DeltaCompressor(cfg)
        base = np.ones(5, dtype=np.float32)
        finetuned = np.ones(5, dtype=np.float32)
        with pytest.raises(ValueError, match="base must be 2-D"):
            compressor.compress(base, finetuned)

    def test_compress_mismatched_shapes_raises(self) -> None:
        cfg = DeltaConfig(rank=4)
        compressor = DeltaCompressor(cfg)
        base = np.ones((3, 4), dtype=np.float32)
        finetuned = np.ones((3, 5), dtype=np.float32)
        with pytest.raises(ValueError, match="same shape"):
            compressor.compress(base, finetuned)

    def test_compression_ratio_k_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="k must be >= 1"):
            DeltaCompressor.compression_ratio(rows=10, cols=10, k=0)

    def test_compress_and_decompress_normal(self) -> None:
        rng = np.random.default_rng(42)
        base = rng.standard_normal((16, 32)).astype(np.float32)
        finetuned = base + 0.01 * rng.standard_normal((16, 32)).astype(np.float32)
        cfg = DeltaConfig(rank=4)
        compressor = DeltaCompressor(cfg)
        U_k, S_k, Vt_k = compressor.compress(base, finetuned)
        assert U_k.shape[0] == 16
        assert S_k.ndim == 1
        delta_approx = compressor.decompress(U_k, S_k, Vt_k)
        assert delta_approx.shape == (16, 32)
        assert delta_approx.dtype == np.float32
        assert compressor.stats.total_compress_calls == 1
        assert compressor.stats.total_singular_values_kept >= 1

    def test_compression_ratio_normal(self) -> None:
        ratio = DeltaCompressor.compression_ratio(rows=256, cols=512, k=16)
        assert ratio > 1.0


# ---------------------------------------------------------------------------
# FlashMLA
# ---------------------------------------------------------------------------


class TestFlashMLACoverage:
    def test_config_n_heads_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="n_heads must be >= 1"):
            MLAConfig(n_heads=0, head_dim=64, latent_dim=64)

    def test_config_head_dim_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="head_dim must be >= 1"):
            MLAConfig(n_heads=1, head_dim=0, latent_dim=64)

    def test_config_latent_dim_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="latent_dim must be >= 1"):
            MLAConfig(n_heads=1, head_dim=64, latent_dim=0)

    def test_config_rope_dim_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="rope_dim must be >= 0"):
            MLAConfig(n_heads=1, head_dim=64, latent_dim=64, rope_dim=-1)

    def test_cache_max_seq_len_zero_raises(self) -> None:
        cfg = MLAConfig(n_heads=2, head_dim=4, latent_dim=8)
        with pytest.raises(ValueError, match="max_seq_len must be >= 1"):
            FlashMLACache(cfg, max_seq_len=0)

    def test_append_wrong_shape_raises(self) -> None:
        cfg = MLAConfig(n_heads=2, head_dim=4, latent_dim=8)
        cache = FlashMLACache(cfg, max_seq_len=4)
        wrong = np.zeros(7, dtype=np.float32)  # latent_dim=8, so (7,) is wrong
        with pytest.raises(ValueError, match="x must have shape"):
            cache.append(wrong)

    def test_append_overflow_raises(self) -> None:
        cfg = MLAConfig(n_heads=2, head_dim=4, latent_dim=8)
        cache = FlashMLACache(cfg, max_seq_len=2)
        latent = np.zeros(8, dtype=np.float32)
        cache.append(latent)
        cache.append(latent)
        with pytest.raises(OverflowError, match="FlashMLACache is full"):
            cache.append(latent)

    def test_attend_wrong_q_shape_raises(self) -> None:
        cfg = MLAConfig(n_heads=2, head_dim=4, latent_dim=8)
        cache = FlashMLACache(cfg, max_seq_len=4)
        # q shape check fires before empty-cache check
        q = np.zeros((3, 4), dtype=np.float32)   # wrong n_heads (should be 2)
        W_uk = np.zeros((8, 8), dtype=np.float32)
        W_uv = np.zeros((8, 8), dtype=np.float32)
        with pytest.raises(ValueError, match="q must have shape"):
            cache.attend(q, W_uk, W_uv)

    def test_attend_wrong_w_uk_shape_raises(self) -> None:
        cfg = MLAConfig(n_heads=2, head_dim=4, latent_dim=8)
        cache = FlashMLACache(cfg, max_seq_len=4)
        q = np.zeros((2, 4), dtype=np.float32)
        W_uk = np.zeros((8, 7), dtype=np.float32)  # wrong: should be (8, 8)
        W_uv = np.zeros((8, 8), dtype=np.float32)
        with pytest.raises(ValueError, match="W_uk must have shape"):
            cache.attend(q, W_uk, W_uv)

    def test_attend_wrong_w_uv_shape_raises(self) -> None:
        cfg = MLAConfig(n_heads=2, head_dim=4, latent_dim=8)
        cache = FlashMLACache(cfg, max_seq_len=4)
        q = np.zeros((2, 4), dtype=np.float32)
        W_uk = np.zeros((8, 8), dtype=np.float32)
        W_uv = np.zeros((8, 7), dtype=np.float32)  # wrong: should be (8, 8)
        with pytest.raises(ValueError, match="W_uv must have shape"):
            cache.attend(q, W_uk, W_uv)

    def test_attend_empty_cache_raises(self) -> None:
        cfg = MLAConfig(n_heads=2, head_dim=4, latent_dim=8)
        cache = FlashMLACache(cfg, max_seq_len=4)
        q = np.zeros((2, 4), dtype=np.float32)
        W_uk = np.zeros((8, 8), dtype=np.float32)
        W_uv = np.zeros((8, 8), dtype=np.float32)
        with pytest.raises(ValueError, match="Cache is empty"):
            cache.attend(q, W_uk, W_uv)

    def test_attend_normal_path(self) -> None:
        cfg = MLAConfig(n_heads=2, head_dim=4, latent_dim=8)
        cache = FlashMLACache(cfg, max_seq_len=4)
        rng = np.random.default_rng(0)
        latent = rng.standard_normal(8).astype(np.float32)
        cache.append(latent)
        assert cache.seq_len == 1
        q = rng.standard_normal((2, 4)).astype(np.float32)
        W_uk = rng.standard_normal((8, 8)).astype(np.float32)
        W_uv = rng.standard_normal((8, 8)).astype(np.float32)
        out = cache.attend(q, W_uk, W_uv)
        assert out.shape == (2, 4)
        assert out.dtype == np.float32

    def test_compression_ratio_property(self) -> None:
        cfg = MLAConfig(n_heads=8, head_dim=64, latent_dim=64)
        cache = FlashMLACache(cfg, max_seq_len=512)
        assert cache.compression_ratio == pytest.approx(8.0)

    def test_reset_resets_seq_len_and_attend_raises_empty(self) -> None:
        cfg = MLAConfig(n_heads=2, head_dim=4, latent_dim=8)
        cache = FlashMLACache(cfg, max_seq_len=4)
        latent = np.zeros(8, dtype=np.float32)
        cache.append(latent)
        assert cache.seq_len == 1
        cache.reset()
        assert cache.seq_len == 0
        q = np.zeros((2, 4), dtype=np.float32)
        W_uk = np.zeros((8, 8), dtype=np.float32)
        W_uv = np.zeros((8, 8), dtype=np.float32)
        with pytest.raises(ValueError, match="Cache is empty"):
            cache.attend(q, W_uk, W_uv)
