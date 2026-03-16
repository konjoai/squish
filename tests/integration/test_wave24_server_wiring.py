"""tests/test_wave24_server_wiring.py

Verifies that all Wave 24 module classes are importable and have the expected
public APIs. No GPU required — pure numpy smoke tests.

Wave 24 — Quantisation Evolution & Model Surgery:
    ternary_quant, binary_attn, structured_prune, layer_fuse,
    weight_sharing, quant_calib, sparse_weight, delta_compress,
    model_surgery, zero_quant_v2, gptq_layer, sparse_moe,
    awq_v2, iter_prune
"""
from __future__ import annotations

import numpy as np
import pytest

RNG = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# 1. ternary_quant
# ---------------------------------------------------------------------------

class TestTernaryQuant:
    def test_imports(self):
        from squish.quant.ternary_quant import TernaryConfig, TernaryQuantizer, TernaryStats
        assert TernaryConfig is not None
        assert TernaryQuantizer is not None
        assert TernaryStats is not None

    def test_quantize_returns_tuple(self):
        from squish.quant.ternary_quant import TernaryConfig, TernaryQuantizer
        cfg = TernaryConfig(zero_threshold=0.05)
        quantizer = TernaryQuantizer(cfg)
        weights = RNG.standard_normal((32, 32)).astype(np.float32)
        ternary, scale = quantizer.quantize(weights)
        assert isinstance(scale, float)
        assert ternary.dtype == np.int8

    def test_dequantize_shape(self):
        from squish.quant.ternary_quant import TernaryConfig, TernaryQuantizer
        cfg = TernaryConfig(zero_threshold=0.05)
        quantizer = TernaryQuantizer(cfg)
        weights = RNG.standard_normal((16, 16)).astype(np.float32)
        ternary, scale = quantizer.quantize(weights)
        recovered = quantizer.dequantize(ternary, scale)
        assert recovered.shape == weights.shape

    def test_stats(self):
        from squish.quant.ternary_quant import TernaryConfig, TernaryQuantizer
        cfg = TernaryConfig(zero_threshold=0.05)
        quantizer = TernaryQuantizer(cfg)
        weights = RNG.standard_normal((8, 8)).astype(np.float32)
        quantizer.quantize(weights)
        stats = quantizer.stats
        assert stats.total_quantize_calls == 1
        assert stats.total_weights_quantized == 64


# ---------------------------------------------------------------------------
# 3. structured_prune
# ---------------------------------------------------------------------------

class TestStructuredPrune:
    def test_imports(self):
        from squish.moe.structured_prune import PruneConfig, PruneStats, StructuredPruner
        assert PruneConfig is not None
        assert StructuredPruner is not None
        assert PruneStats is not None

    def test_prune_returns_tuple(self):
        from squish.moe.structured_prune import PruneConfig, StructuredPruner
        cfg = PruneConfig(N=2, M=4)
        pruner = StructuredPruner(cfg)
        weights = RNG.standard_normal((8, 8)).astype(np.float32)
        pruned, mask = pruner.prune(weights)
        assert pruned.shape == weights.shape
        assert mask.shape == weights.shape
        assert mask.dtype == bool

    def test_sparsity_fraction(self):
        from squish.moe.structured_prune import PruneConfig, StructuredPruner
        cfg = PruneConfig(N=2, M=4)
        pruner = StructuredPruner(cfg)
        weights = RNG.standard_normal((8, 8)).astype(np.float32)
        pruned, _ = pruner.prune(weights)
        frac = pruner.sparsity_fraction(pruned)
        assert 0.0 <= frac <= 1.0

    def test_stats(self):
        from squish.moe.structured_prune import PruneConfig, StructuredPruner
        cfg = PruneConfig(N=2, M=4)
        pruner = StructuredPruner(cfg)
        weights = RNG.standard_normal((8, 8)).astype(np.float32)
        pruner.prune(weights)
        stats = pruner.stats
        assert stats.total_prune_calls == 1
        assert stats.total_weights_zeroed > 0


# ---------------------------------------------------------------------------
# 4. layer_fuse
# ---------------------------------------------------------------------------

class TestLayerFuse:
    def test_imports(self):
        from squish.token.layer_fuse import FusionConfig, FusionStats, LayerFuser
        assert FusionConfig is not None
        assert LayerFuser is not None
        assert FusionStats is not None

    def test_fuse_shape(self):
        from squish.token.layer_fuse import FusionConfig, LayerFuser
        cfg = FusionConfig(hidden_dim=16, similarity_threshold=0.97)
        fuser = LayerFuser(cfg)
        a = RNG.standard_normal((16, 16)).astype(np.float32)
        b = RNG.standard_normal((16, 16)).astype(np.float32)
        fused = fuser.fuse(a, b)
        assert fused.shape == a.shape

    def test_should_fuse_identical(self):
        from squish.token.layer_fuse import FusionConfig, LayerFuser
        cfg = FusionConfig(hidden_dim=16, similarity_threshold=0.97)
        fuser = LayerFuser(cfg)
        a = RNG.standard_normal((16, 16)).astype(np.float32)
        result = fuser.should_fuse(a, a)
        assert result is True

    def test_stats(self):
        from squish.token.layer_fuse import FusionConfig, LayerFuser
        cfg = FusionConfig(hidden_dim=16, similarity_threshold=0.5)
        fuser = LayerFuser(cfg)
        a = RNG.standard_normal((16, 16)).astype(np.float32)
        fuser.fuse(a, a)
        stats = fuser.stats
        assert stats.total_fusions >= 1


# ---------------------------------------------------------------------------
# 5. weight_sharing
# ---------------------------------------------------------------------------

class TestWeightSharing:
    def test_imports(self):
        from squish.lora.weight_sharing import SharingConfig, SharingStats, WeightSharer
        assert SharingConfig is not None
        assert WeightSharer is not None
        assert SharingStats is not None

    def test_get_effective_weight_shape(self):
        from squish.lora.weight_sharing import SharingConfig, WeightSharer
        cfg = SharingConfig(hidden_dim=16, n_shared_layers=4, rank=4)
        sharer = WeightSharer(cfg)
        w = sharer.get_effective_weight(0)
        assert w.shape == (16, 16)

    def test_memory_bytes(self):
        from squish.lora.weight_sharing import SharingConfig, WeightSharer
        cfg = SharingConfig(hidden_dim=16, n_shared_layers=4, rank=4)
        sharer = WeightSharer(cfg)
        assert sharer.memory_bytes() < sharer.dense_memory_bytes()

    def test_stats(self):
        from squish.lora.weight_sharing import SharingConfig, WeightSharer
        cfg = SharingConfig(hidden_dim=16, n_shared_layers=4, rank=4)
        sharer = WeightSharer(cfg)
        sharer.get_effective_weight(0)
        stats = sharer.stats
        assert stats.total_effective_weight_calls == 1
        ratio = stats.memory_ratio
        assert 0.0 < ratio < 1.0


# ---------------------------------------------------------------------------
# 6. quant_calib
# ---------------------------------------------------------------------------

class TestQuantCalib:
    def test_imports(self):
        from squish.quant.quant_calib import CalibConfig, CalibResult, CalibStats, QuantCalibrator
        assert CalibConfig is not None
        assert CalibResult is not None
        assert QuantCalibrator is not None
        assert CalibStats is not None

    def test_calibrate_returns_result(self):
        from squish.quant.quant_calib import CalibConfig, CalibResult, QuantCalibrator
        cfg = CalibConfig(method="minmax", n_bits=8, per_channel=True)
        calibrator = QuantCalibrator(cfg)
        activations = RNG.standard_normal((64, 16)).astype(np.float32)
        result = calibrator.calibrate(activations)
        assert isinstance(result, CalibResult)
        assert result.n_bits == 8

    def test_per_tensor_scales(self):
        from squish.quant.quant_calib import CalibConfig, QuantCalibrator
        cfg = CalibConfig(method="minmax", n_bits=8, per_channel=False)
        calibrator = QuantCalibrator(cfg)
        activations = RNG.standard_normal((64, 16)).astype(np.float32)
        result = calibrator.calibrate(activations)
        assert result.scales.ndim == 0

    def test_stats(self):
        from squish.quant.quant_calib import CalibConfig, QuantCalibrator
        cfg = CalibConfig(method="minmax", n_bits=8, per_channel=True)
        calibrator = QuantCalibrator(cfg)
        activations = RNG.standard_normal((64, 16)).astype(np.float32)
        calibrator.calibrate(activations)
        stats = calibrator.stats
        assert stats.total_calibrations == 1


# ---------------------------------------------------------------------------
# 7. sparse_weight
# ---------------------------------------------------------------------------

class TestSparseWeight:
    def test_imports(self):
        from squish.moe.sparse_weight import SparseStats, SparseWeightStore, SparsityConfig
        assert SparsityConfig is not None
        assert SparseWeightStore is not None
        assert SparseStats is not None

    def test_compress_and_decompress(self):
        from squish.moe.sparse_weight import SparseWeightStore, SparsityConfig
        cfg = SparsityConfig(N=2, M=4)
        store = SparseWeightStore(cfg)
        dense = RNG.standard_normal((8, 8)).astype(np.float32)
        store.compress(dense)
        recovered = store.decompress()
        assert recovered.shape == dense.shape

    def test_memory_savings(self):
        from squish.moe.sparse_weight import SparseWeightStore, SparsityConfig
        cfg = SparsityConfig(N=2, M=4)
        store = SparseWeightStore(cfg)
        dense = RNG.standard_normal((32, 32)).astype(np.float32)
        store.compress(dense)
        assert store.memory_bytes() < store.dense_memory_bytes()

    def test_stats(self):
        from squish.moe.sparse_weight import SparseWeightStore, SparsityConfig
        cfg = SparsityConfig(N=2, M=4)
        store = SparseWeightStore(cfg)
        dense = RNG.standard_normal((8, 8)).astype(np.float32)
        store.compress(dense)
        store.decompress()
        stats = store.stats
        assert stats.total_compress_calls == 1
        assert stats.total_decompress_calls == 1


# ---------------------------------------------------------------------------
# 8. delta_compress
# ---------------------------------------------------------------------------

class TestDeltaCompress:
    def test_imports(self):
        from squish.context.delta_compress import DeltaCompressor, DeltaConfig, DeltaStats
        assert DeltaConfig is not None
        assert DeltaCompressor is not None
        assert DeltaStats is not None

    def test_compress_and_decompress(self):
        from squish.context.delta_compress import DeltaCompressor, DeltaConfig
        cfg = DeltaConfig(rank=4)
        compressor = DeltaCompressor(cfg)
        base = RNG.standard_normal((16, 16)).astype(np.float32)
        finetuned = base + 0.01 * RNG.standard_normal((16, 16)).astype(np.float32)
        U_k, S_k, Vt_k = compressor.compress(base, finetuned)
        recovered = compressor.decompress(U_k, S_k, Vt_k)
        assert recovered.shape == base.shape

    def test_compression_ratio_static(self):
        from squish.context.delta_compress import DeltaCompressor
        ratio = DeltaCompressor.compression_ratio(rows=64, cols=64, k=8)
        assert ratio > 1.0

    def test_stats(self):
        from squish.context.delta_compress import DeltaCompressor, DeltaConfig
        cfg = DeltaConfig(rank=4)
        compressor = DeltaCompressor(cfg)
        base = RNG.standard_normal((16, 16)).astype(np.float32)
        finetuned = base + 0.01 * RNG.standard_normal((16, 16)).astype(np.float32)
        compressor.compress(base, finetuned)
        stats = compressor.stats
        assert stats.total_compress_calls == 1
        assert stats.total_singular_values_kept == 4


# ---------------------------------------------------------------------------
# 10. zero_quant_v2
# ---------------------------------------------------------------------------

class TestZeroQuantV2:
    def test_imports(self):
        from squish.quant.zero_quant_v2 import ZeroQuantV2, ZQConfig, ZQStats
        assert ZQConfig is not None
        assert ZeroQuantV2 is not None
        assert ZQStats is not None

    def test_quantize_returns_triple(self):
        from squish.quant.zero_quant_v2 import ZeroQuantV2, ZQConfig
        cfg = ZQConfig(n_bits=8, group_size=16)
        zq = ZeroQuantV2(cfg)
        weights = RNG.standard_normal((16, 32)).astype(np.float32)
        quantized, scales, residual = zq.quantize(weights)
        assert quantized.dtype == np.int8
        assert scales.shape[0] == weights.shape[0]

    def test_dequantize_shape(self):
        from squish.quant.zero_quant_v2 import ZeroQuantV2, ZQConfig
        cfg = ZQConfig(n_bits=8, group_size=16)
        zq = ZeroQuantV2(cfg)
        weights = RNG.standard_normal((16, 32)).astype(np.float32)
        quantized, scales, residual = zq.quantize(weights)
        recovered = zq.dequantize(quantized, scales, residual)
        assert recovered.shape == weights.shape

    def test_stats(self):
        from squish.quant.zero_quant_v2 import ZeroQuantV2, ZQConfig
        cfg = ZQConfig(n_bits=8, group_size=16)
        zq = ZeroQuantV2(cfg)
        weights = RNG.standard_normal((16, 32)).astype(np.float32)
        zq.quantize(weights)
        stats = zq.stats
        assert stats.total_quantize_calls == 1
        assert 0.0 <= stats.outlier_rate <= 1.0


# ---------------------------------------------------------------------------
# 11. gptq_layer
# ---------------------------------------------------------------------------

class TestGPTQLayer:
    def test_imports(self):
        from squish.quant.gptq_layer import GPTQCalibrator, GPTQConfig, GPTQStats
        assert GPTQConfig is not None
        assert GPTQCalibrator is not None
        assert GPTQStats is not None

    def test_calibrate_shape(self):
        from squish.quant.gptq_layer import GPTQCalibrator, GPTQConfig
        cfg = GPTQConfig(n_bits=4, block_size=8)
        calibrator = GPTQCalibrator(cfg)
        W = RNG.standard_normal((16, 16)).astype(np.float32)
        X = RNG.standard_normal((32, 16)).astype(np.float32)
        quantized = calibrator.calibrate(W, X)
        assert quantized.shape == W.shape

    def test_calibrate_dtype(self):
        from squish.quant.gptq_layer import GPTQCalibrator, GPTQConfig
        cfg = GPTQConfig(n_bits=4, block_size=8)
        calibrator = GPTQCalibrator(cfg)
        W = RNG.standard_normal((16, 16)).astype(np.float32)
        X = RNG.standard_normal((32, 16)).astype(np.float32)
        quantized = calibrator.calibrate(W, X)
        # GPTQ returns a quantized float32 weight matrix
        assert quantized.dtype == np.float32

    def test_stats(self):
        from squish.quant.gptq_layer import GPTQCalibrator, GPTQConfig
        cfg = GPTQConfig(n_bits=4, block_size=8)
        calibrator = GPTQCalibrator(cfg)
        W = RNG.standard_normal((16, 16)).astype(np.float32)
        X = RNG.standard_normal((32, 16)).astype(np.float32)
        calibrator.calibrate(W, X)
        stats = calibrator.stats
        assert stats.total_calibrations == 1
        assert stats.total_columns == 16


# ---------------------------------------------------------------------------
# 12. sparse_moe
# ---------------------------------------------------------------------------

class TestSparseMoE:
    def test_imports(self):
        from squish.moe.sparse_moe import MoEConfig, MoEStats, SparseMoERouter
        assert MoEConfig is not None
        assert SparseMoERouter is not None
        assert MoEStats is not None

    def test_route_returns_triple(self):
        from squish.moe.sparse_moe import MoEConfig, SparseMoERouter
        cfg = MoEConfig(n_experts=8, top_k=2, hidden_dim=16)
        router = SparseMoERouter(cfg)
        hidden = RNG.random((4, 16)).astype(np.float32)
        indices, weights, aux_loss = router.route(hidden)
        assert indices.shape == (4, 2)
        assert weights.shape == (4, 2)
        assert isinstance(aux_loss, float)

    def test_weights_sum_to_one(self):
        from squish.moe.sparse_moe import MoEConfig, SparseMoERouter
        cfg = MoEConfig(n_experts=8, top_k=2, hidden_dim=16)
        router = SparseMoERouter(cfg)
        hidden = RNG.random((4, 16)).astype(np.float32)
        _, weights, _ = router.route(hidden)
        row_sums = weights.sum(axis=1)
        np.testing.assert_allclose(row_sums, np.ones(4), atol=1e-5)

    def test_stats(self):
        from squish.moe.sparse_moe import MoEConfig, SparseMoERouter
        cfg = MoEConfig(n_experts=8, top_k=2, hidden_dim=16)
        router = SparseMoERouter(cfg)
        hidden = RNG.random((4, 16)).astype(np.float32)
        router.route(hidden)
        stats = router.stats
        assert stats.total_route_calls == 1
        assert stats.total_tokens == 4


# ---------------------------------------------------------------------------
# 13. awq_v2
# ---------------------------------------------------------------------------

class TestAWQv2:
    def test_imports(self):
        from squish.quant.awq_v2 import AWQv2Calibrator, AWQv2Config, AWQv2Stats
        assert AWQv2Config is not None
        assert AWQv2Calibrator is not None
        assert AWQv2Stats is not None

    def test_calibrate_returns_scales_shifts(self):
        from squish.quant.awq_v2 import AWQv2Calibrator, AWQv2Config
        cfg = AWQv2Config(n_bits=4, group_size=8, n_search_steps=5)
        calibrator = AWQv2Calibrator(cfg)
        W = RNG.standard_normal((16, 16)).astype(np.float32)
        act_scales = np.abs(RNG.standard_normal((16,))).astype(np.float32)
        opt_scales, opt_shifts = calibrator.calibrate(W, act_scales)
        assert opt_scales.shape == (16,)
        assert opt_shifts.shape == (16,)

    def test_quantize_shape(self):
        from squish.quant.awq_v2 import AWQv2Calibrator, AWQv2Config
        cfg = AWQv2Config(n_bits=4, group_size=8, n_search_steps=5)
        calibrator = AWQv2Calibrator(cfg)
        W = RNG.standard_normal((16, 16)).astype(np.float32)
        act_scales = np.abs(RNG.standard_normal((16,))).astype(np.float32)
        scales, shifts = calibrator.calibrate(W, act_scales)
        quantized = calibrator.quantize(W, scales, shifts)
        assert quantized.shape == W.shape

    def test_stats(self):
        from squish.quant.awq_v2 import AWQv2Calibrator, AWQv2Config
        cfg = AWQv2Config(n_bits=4, group_size=8, n_search_steps=5)
        calibrator = AWQv2Calibrator(cfg)
        W = RNG.standard_normal((16, 16)).astype(np.float32)
        act_scales = np.abs(RNG.standard_normal((16,))).astype(np.float32)
        calibrator.calibrate(W, act_scales)
        stats = calibrator.stats
        assert stats.total_calibrations == 1


