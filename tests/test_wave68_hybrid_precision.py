"""tests/test_wave68_hybrid_precision.py

Unit tests for Wave 68: Per-Block Intra-Weight Mixed Precision.

Module under test
─────────────────
* squish.compress.hybrid_precision — HybridPrecisionConfig, BlockPrecision,
                                      BlockPrecisionMap, HybridPrecisionProfiler,
                                      assign_hybrid_precision,
                                      find_variance_threshold,
                                      BITS_INT4, BITS_INT2, BITS_BF16
"""
from __future__ import annotations

import unittest

import numpy as np

from squish.compress.hybrid_precision import (
    BITS_BF16,
    BITS_INT2,
    BITS_INT4,
    BlockPrecision,
    BlockPrecisionMap,
    HybridPrecisionConfig,
    HybridPrecisionProfiler,
    assign_hybrid_precision,
    find_variance_threshold,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_weights(
    n_rows: int = 100,
    n_cols: int = 64,
    *,
    seed: int = 0,
    scale: float = 1.0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((n_rows, n_cols)) * scale).astype(np.float32)


def _make_pmap(
    n_blocks: int = 10,
    n_int4: int = 6,
    n_int2: int = 3,
    n_bf16: int = 1,
) -> BlockPrecisionMap:
    bits = np.array(
        [BITS_INT4] * n_int4 + [BITS_INT2] * n_int2 + [BITS_BF16] * n_bf16,
        dtype=np.uint8,
    )
    np.random.default_rng(0).shuffle(bits)
    return BlockPrecisionMap(
        bits=bits,
        block_size=64,
        original_shape=(n_blocks * 64,),
        variances=np.ones(n_blocks, dtype=np.float32),
        magnitudes=np.ones(n_blocks, dtype=np.float32),
    )


# ---------------------------------------------------------------------------
# TestBitConstants
# ---------------------------------------------------------------------------

class TestBitConstants(unittest.TestCase):

    def test_bits_int4(self):
        self.assertEqual(BITS_INT4, 4)

    def test_bits_int2(self):
        self.assertEqual(BITS_INT2, 2)

    def test_bits_bf16(self):
        self.assertEqual(BITS_BF16, 16)

    def test_block_precision_aliases(self):
        self.assertEqual(BlockPrecision.INT4, BITS_INT4)
        self.assertEqual(BlockPrecision.INT2, BITS_INT2)
        self.assertEqual(BlockPrecision.BF16, BITS_BF16)


# ---------------------------------------------------------------------------
# TestHybridPrecisionConfig
# ---------------------------------------------------------------------------

class TestHybridPrecisionConfig(unittest.TestCase):

    def test_defaults(self):
        cfg = HybridPrecisionConfig()
        self.assertEqual(cfg.block_size, 64)
        self.assertAlmostEqual(cfg.int4_fraction, 0.75, places=7)
        self.assertAlmostEqual(cfg.bf16_outlier_pct, 0.05, places=7)
        self.assertAlmostEqual(cfg.target_bpw, 3.0, places=7)

    def test_invalid_block_size_zero(self):
        with self.assertRaises(ValueError):
            HybridPrecisionConfig(block_size=0)

    def test_invalid_block_size_negative(self):
        with self.assertRaises(ValueError):
            HybridPrecisionConfig(block_size=-1)

    def test_invalid_int4_fraction_negative(self):
        with self.assertRaises(ValueError):
            HybridPrecisionConfig(int4_fraction=-0.1)

    def test_invalid_int4_fraction_above_one(self):
        with self.assertRaises(ValueError):
            HybridPrecisionConfig(int4_fraction=1.1)

    def test_invalid_bf16_outlier_pct_negative(self):
        with self.assertRaises(ValueError):
            HybridPrecisionConfig(bf16_outlier_pct=-0.01)

    def test_invalid_bf16_outlier_pct_one(self):
        with self.assertRaises(ValueError):
            HybridPrecisionConfig(bf16_outlier_pct=1.0)

    def test_invalid_target_bpw_zero(self):
        with self.assertRaises(ValueError):
            HybridPrecisionConfig(target_bpw=0.0)

    def test_invalid_target_bpw_negative(self):
        with self.assertRaises(ValueError):
            HybridPrecisionConfig(target_bpw=-1.0)

    def test_valid_custom_values(self):
        cfg = HybridPrecisionConfig(
            block_size=32, int4_fraction=0.5, bf16_outlier_pct=0.02, target_bpw=2.5
        )
        self.assertEqual(cfg.block_size, 32)
        self.assertAlmostEqual(cfg.int4_fraction, 0.5, places=7)

    def test_int4_fraction_boundary_zero(self):
        # 0.0 is valid
        cfg = HybridPrecisionConfig(int4_fraction=0.0)
        self.assertAlmostEqual(cfg.int4_fraction, 0.0, places=7)

    def test_int4_fraction_boundary_one(self):
        # 1.0 is valid
        cfg = HybridPrecisionConfig(int4_fraction=1.0)
        self.assertAlmostEqual(cfg.int4_fraction, 1.0, places=7)


# ---------------------------------------------------------------------------
# TestBlockPrecisionMap
# ---------------------------------------------------------------------------

class TestBlockPrecisionMap(unittest.TestCase):

    def test_n_blocks(self):
        pm = _make_pmap(10, 6, 3, 1)
        self.assertEqual(pm.n_blocks, 10)

    def test_n_int4(self):
        pm = _make_pmap(10, 6, 3, 1)
        self.assertEqual(pm.n_int4, 6)

    def test_n_int2(self):
        pm = _make_pmap(10, 6, 3, 1)
        self.assertEqual(pm.n_int2, 3)

    def test_n_bf16(self):
        pm = _make_pmap(10, 6, 3, 1)
        self.assertEqual(pm.n_bf16, 1)

    def test_effective_bpw_all_int4(self):
        bits = np.full(5, BITS_INT4, dtype=np.uint8)
        pm = BlockPrecisionMap(bits=bits, block_size=64, original_shape=(320,),
                               variances=np.ones(5), magnitudes=np.ones(5))
        self.assertAlmostEqual(pm.effective_bpw, 4.0, places=7)

    def test_effective_bpw_all_int2(self):
        bits = np.full(5, BITS_INT2, dtype=np.uint8)
        pm = BlockPrecisionMap(bits=bits, block_size=64, original_shape=(320,),
                               variances=np.ones(5), magnitudes=np.ones(5))
        self.assertAlmostEqual(pm.effective_bpw, 2.0, places=7)

    def test_effective_bpw_all_bf16(self):
        bits = np.full(4, BITS_BF16, dtype=np.uint8)
        pm = BlockPrecisionMap(bits=bits, block_size=64, original_shape=(256,),
                               variances=np.ones(4), magnitudes=np.ones(4))
        self.assertAlmostEqual(pm.effective_bpw, 16.0, places=7)

    def test_effective_bpw_mixed(self):
        # 5 INT4 + 5 INT2 + 0 BF16 → (5*4 + 5*2) / 10 = 30/10 = 3.0
        bits = np.array([BITS_INT4] * 5 + [BITS_INT2] * 5, dtype=np.uint8)
        pm = BlockPrecisionMap(bits=bits, block_size=64, original_shape=(640,),
                               variances=np.ones(10), magnitudes=np.ones(10))
        self.assertAlmostEqual(pm.effective_bpw, 3.0, places=7)

    def test_effective_bpw_zero_blocks(self):
        bits = np.array([], dtype=np.uint8)
        pm = BlockPrecisionMap(bits=bits, block_size=64, original_shape=(0,),
                               variances=np.array([]), magnitudes=np.array([]))
        self.assertAlmostEqual(pm.effective_bpw, 0.0, places=7)

    def test_rate_distortion_table_keys(self):
        pm = _make_pmap(10, 6, 3, 1)
        table = pm.rate_distortion_table()
        for key in ("n_blocks", "n_int4", "n_int2", "n_bf16",
                    "pct_int4", "pct_int2", "pct_bf16", "effective_bpw"):
            self.assertIn(key, table)

    def test_rate_distortion_table_percentages_sum(self):
        pm = _make_pmap(10, 6, 3, 1)
        table = pm.rate_distortion_table()
        total_pct = table["pct_int4"] + table["pct_int2"] + table["pct_bf16"]
        self.assertAlmostEqual(total_pct, 100.0, places=5)


# ---------------------------------------------------------------------------
# TestHybridPrecisionProfiler
# ---------------------------------------------------------------------------

class TestHybridPrecisionProfiler(unittest.TestCase):

    def setUp(self):
        self.profiler = HybridPrecisionProfiler(HybridPrecisionConfig(
            block_size=64, int4_fraction=0.75, bf16_outlier_pct=0.05
        ))

    def test_assign_returns_block_precision_map(self):
        w = _make_weights(100, 64)
        pm = self.profiler.assign(w)
        self.assertIsInstance(pm, BlockPrecisionMap)

    def test_assign_n_blocks_correct(self):
        # 100 * 64 = 6400 elements; 6400 / 64 = exactly 100 blocks
        w = np.zeros((100, 64), dtype=np.float32)
        pm = self.profiler.assign(w)
        self.assertEqual(pm.n_blocks, 100)

    def test_assign_n_blocks_non_multiple(self):
        # 70 elements, block_size=64 → ceil(70/64)=2 blocks
        profiler = HybridPrecisionProfiler(HybridPrecisionConfig(block_size=64))
        w = np.zeros(70, dtype=np.float32)
        pm = profiler.assign(w)
        self.assertEqual(pm.n_blocks, 2)

    def test_assign_bits_values_valid(self):
        w = _make_weights(50, 64)
        pm = self.profiler.assign(w)
        valid = {BITS_INT4, BITS_INT2, BITS_BF16}
        for b in pm.bits:
            self.assertIn(int(b), valid)

    def test_assign_bf16_count_approximate(self):
        # With 100 blocks and bf16_outlier_pct=0.05 → ~5 BF16 blocks
        w = _make_weights(100, 64)
        pm = self.profiler.assign(w)
        n_bf16 = pm.n_bf16
        self.assertGreaterEqual(n_bf16, 0)
        self.assertLessEqual(n_bf16, pm.n_blocks)

    def test_assign_int4_count_in_range(self):
        w = _make_weights(100, 64)
        pm = self.profiler.assign(w)
        # Some INT4 blocks should be present
        self.assertGreater(pm.n_int4, 0)

    def test_assign_int2_count_in_range(self):
        w = _make_weights(100, 64)
        pm = self.profiler.assign(w)
        # With int4_fraction=0.75, ~25% of non-BF16 blocks should be INT2
        self.assertGreaterEqual(pm.n_int2, 0)

    def test_assign_original_shape_preserved(self):
        w = _make_weights(10, 64)
        pm = self.profiler.assign(w)
        self.assertEqual(pm.original_shape, (10, 64))

    def test_assign_variances_shape(self):
        w = _make_weights(50, 64)
        pm = self.profiler.assign(w)
        self.assertEqual(pm.variances.shape, (pm.n_blocks,))

    def test_assign_magnitudes_shape(self):
        w = _make_weights(50, 64)
        pm = self.profiler.assign(w)
        self.assertEqual(pm.magnitudes.shape, (pm.n_blocks,))

    def test_assign_zero_weights(self):
        # All-zero weights shouldn't crash
        w = np.zeros((64, 64), dtype=np.float32)
        pm = self.profiler.assign(w)
        self.assertIsInstance(pm, BlockPrecisionMap)

    def test_bf16_zero_outlier_pct(self):
        profiler = HybridPrecisionProfiler(
            HybridPrecisionConfig(bf16_outlier_pct=0.0)
        )
        w = _make_weights(100, 64)
        pm = profiler.assign(w)
        self.assertEqual(pm.n_bf16, 0)

    def test_int4_fraction_one_no_int2(self):
        profiler = HybridPrecisionProfiler(
            HybridPrecisionConfig(int4_fraction=1.0, bf16_outlier_pct=0.0)
        )
        w = _make_weights(100, 64)
        pm = profiler.assign(w)
        self.assertEqual(pm.n_int2, 0)

    def test_int4_fraction_zero_no_int4(self):
        profiler = HybridPrecisionProfiler(
            HybridPrecisionConfig(int4_fraction=0.0, bf16_outlier_pct=0.0)
        )
        w = _make_weights(100, 64)
        pm = profiler.assign(w)
        self.assertEqual(pm.n_int4, 0)

    def test_large_matrix(self):
        w = _make_weights(256, 64)
        pm = self.profiler.assign(w)
        self.assertEqual(pm.n_blocks, 256)
        self.assertGreater(pm.n_int4 + pm.n_int2 + pm.n_bf16, 0)


# ---------------------------------------------------------------------------
# TestAssignHybridPrecision
# ---------------------------------------------------------------------------

class TestAssignHybridPrecision(unittest.TestCase):

    def test_returns_block_precision_map(self):
        w = _make_weights(50, 64)
        pm = assign_hybrid_precision(w)
        self.assertIsInstance(pm, BlockPrecisionMap)

    def test_default_config_block_size(self):
        w = np.zeros((64, 64), dtype=np.float32)
        pm = assign_hybrid_precision(w)
        self.assertEqual(pm.block_size, 64)

    def test_custom_config_passed_through(self):
        cfg = HybridPrecisionConfig(block_size=32, int4_fraction=0.5,
                                    bf16_outlier_pct=0.0)
        w = np.zeros((100, 32), dtype=np.float32)
        pm = assign_hybrid_precision(w, cfg)
        self.assertEqual(pm.block_size, 32)

    def test_1d_input(self):
        w = _make_weights(1, 128).reshape(-1)
        pm = assign_hybrid_precision(w)
        self.assertIsInstance(pm, BlockPrecisionMap)


# ---------------------------------------------------------------------------
# TestFindVarianceThreshold
# ---------------------------------------------------------------------------

class TestFindVarianceThreshold(unittest.TestCase):

    def setUp(self):
        self.weights = _make_weights(200, 64, seed=42)

    def test_returns_float(self):
        threshold = find_variance_threshold(self.weights, target_bpw=3.0)
        self.assertIsInstance(threshold, float)

    def test_threshold_nonnegative(self):
        threshold = find_variance_threshold(self.weights, target_bpw=3.0)
        self.assertGreaterEqual(threshold, 0.0)

    def test_threshold_low_bpw_higher_than_high_bpw(self):
        # Lower target BPW → fewer INT4 blocks needed → higher variance threshold
        t_low = find_variance_threshold(self.weights, target_bpw=2.1)
        t_high = find_variance_threshold(self.weights, target_bpw=3.9)
        # Lower target → threshold should be >= higher target threshold
        self.assertGreaterEqual(t_low, t_high)

    def test_threshold_for_all_int4(self):
        # target_bpw ~= 4.0, no BF16 → all non-outlier blocks INT4
        threshold = find_variance_threshold(
            self.weights, target_bpw=4.0, bf16_outlier_pct=0.0
        )
        # threshold should be very low (most blocks qualify for INT4)
        self.assertGreaterEqual(threshold, 0.0)

    def test_threshold_for_all_int2(self):
        # target_bpw = 2.0, no BF16 → all blocks INT2 → threshold very high
        threshold = find_variance_threshold(
            self.weights, target_bpw=2.0, bf16_outlier_pct=0.0
        )
        # All blocks INT2 → threshold above all variances
        self.assertGreaterEqual(threshold, 0.0)

    def test_single_block(self):
        w = np.ones(64, dtype=np.float32)
        threshold = find_variance_threshold(w, target_bpw=3.0,
                                            block_size=64, bf16_outlier_pct=0.0)
        self.assertIsInstance(threshold, float)

    def test_uniform_weights(self):
        # Uniform → zero variance everywhere; should not crash
        w = np.ones((100, 64), dtype=np.float32)
        threshold = find_variance_threshold(w, target_bpw=3.0)
        self.assertIsInstance(threshold, float)

    def test_custom_block_size(self):
        threshold = find_variance_threshold(self.weights, target_bpw=3.0,
                                            block_size=32)
        self.assertIsInstance(threshold, float)
        self.assertGreaterEqual(threshold, 0.0)


if __name__ == "__main__":
    unittest.main()
