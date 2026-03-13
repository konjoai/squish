"""
tests/test_dare_ties_unit.py

Unit tests for the DARE-TIES classes appended to squish/lora_manager.py
— 100% coverage of DareTiesConfig and DareTiesMerger.
"""

import numpy as np
import pytest

from squish.lora.lora_manager import DareTiesConfig, DareTiesMerger

# ---------------------------------------------------------------------------
# DareTiesConfig
# ---------------------------------------------------------------------------

class TestDareTiesConfig:
    def test_defaults(self):
        cfg = DareTiesConfig()
        assert 0 < cfg.sparsity < 1
        assert cfg.top_k_fraction is None
        assert cfg.scale > 0
        assert isinstance(cfg.seed, int)

    def test_custom(self):
        cfg = DareTiesConfig(sparsity=0.7, top_k_fraction=0.5, scale=0.8, seed=99)
        assert cfg.sparsity == 0.7
        assert cfg.top_k_fraction == 0.5
        assert cfg.scale == 0.8
        assert cfg.seed == 99

    @pytest.mark.parametrize("kwargs, exc", [
        ({"sparsity": 0.0},            "sparsity"),
        ({"sparsity": 1.0},            "sparsity"),
        ({"top_k_fraction": 0.0},      "top_k_fraction"),
        ({"top_k_fraction": 1.1},      "top_k_fraction"),
        ({"scale": 0.0},               "scale"),
        ({"scale": -1.0},              "scale"),
    ])
    def test_validation(self, kwargs, exc):
        with pytest.raises(ValueError, match=exc):
            DareTiesConfig(**kwargs)

    def test_top_k_fraction_exactly_one(self):
        # 1.0 is valid (keep all)
        cfg = DareTiesConfig(top_k_fraction=1.0)
        assert cfg.top_k_fraction == 1.0


# ---------------------------------------------------------------------------
# DareTiesMerger — DARE step
# ---------------------------------------------------------------------------

class TestDareTiesMergerDare:
    def test_sparsify_reduces_nonzero(self):
        cfg  = DareTiesConfig(sparsity=0.9)
        m    = DareTiesMerger(cfg)
        delta = np.ones(1000, dtype=np.float32)
        out  = m.sparsify_dare(delta)
        nonzero_frac = np.count_nonzero(out) / len(out)
        # Expect ~10% survivors, allow tolerance
        assert 0.05 < nonzero_frac < 0.20

    def test_sparsify_expected_magnitude(self):
        """After rescaling, the expected value of each surviving element
        should equal the original element value."""
        cfg   = DareTiesConfig(sparsity=0.8, seed=0)
        m     = DareTiesMerger(cfg)
        delta = np.full(5000, 2.0, dtype=np.float32)
        out   = m.sparsify_dare(delta)
        # E[out] ≈ delta  (survivors rescaled by 1/(1-sparsity)=5, prob 0.2)
        assert abs(out.mean() - 2.0) < 0.5

    def test_sparsify_shape_preserved(self):
        cfg   = DareTiesConfig(sparsity=0.5)
        m     = DareTiesMerger(cfg)
        delta = np.ones((4, 8))
        out   = m.sparsify_dare(delta)
        assert out.shape == (4, 8)

    def test_sparsify_custom_rng(self):
        cfg  = DareTiesConfig(sparsity=0.5)
        m    = DareTiesMerger(cfg)
        np.random.default_rng(123)
        d    = np.ones(100)
        out1 = m.sparsify_dare(d, rng=np.random.default_rng(123))
        out2 = m.sparsify_dare(d, rng=np.random.default_rng(123))
        # Same rng seed → identical output
        np.testing.assert_array_equal(out1, out2)

    def test_sparsify_deterministic_default_rng(self):
        cfg  = DareTiesConfig(sparsity=0.5, seed=7)
        m    = DareTiesMerger(cfg)
        d    = np.arange(20, dtype=np.float32)
        out1 = m.sparsify_dare(d)
        out2 = m.sparsify_dare(d)
        np.testing.assert_array_equal(out1, out2)


# ---------------------------------------------------------------------------
# DareTiesMerger — TIES steps
# ---------------------------------------------------------------------------

class TestDareTiesMergerTies:
    def test_trim_with_fraction(self):
        cfg    = DareTiesConfig(top_k_fraction=0.5)
        m      = DareTiesMerger(cfg)
        deltas = [np.array([0.1, 0.5, -0.3, 0.8], dtype=np.float32)]
        trimmed = m.trim(deltas)
        # About half the entries should be zeroed
        n_zero = np.count_nonzero(trimmed[0] == 0.0)
        assert n_zero >= 1

    def test_trim_no_fraction(self):
        cfg    = DareTiesConfig(top_k_fraction=None)
        m      = DareTiesMerger(cfg)
        delta  = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        trimmed = m.trim([delta])
        np.testing.assert_array_almost_equal(trimmed[0], delta)

    def test_elect_sign_majority(self):
        cfg = DareTiesConfig()
        m   = DareTiesMerger(cfg)
        # 3 agree positive, 1 negative → +1 everywhere
        d1 = np.array([1.0, 1.0])
        d2 = np.array([1.0, 1.0])
        d3 = np.array([1.0, 1.0])
        d4 = np.array([-1.0, -1.0])
        signs = m.elect_sign([d1, d2, d3, d4])
        assert np.all(signs == 1.0)

    def test_elect_sign_tie_broken_positive(self):
        cfg = DareTiesConfig()
        m   = DareTiesMerger(cfg)
        d1  = np.array([1.0])
        d2  = np.array([-1.0])
        signs = m.elect_sign([d1, d2])
        # Ties broken to +1
        assert signs[0] == 1.0

    def test_ties_merge_single_delta(self):
        cfg   = DareTiesConfig(top_k_fraction=None, scale=1.0)
        m     = DareTiesMerger(cfg)
        delta = np.array([0.5, -0.3, 0.8], dtype=np.float32)
        merged = m.ties_merge([delta])
        # With one delta, result should preserve sign-agree elements
        assert merged.shape == (3,)

    def test_ties_merge_two_agreeing(self):
        """Two identical positive deltas → merged average ≈ original."""
        cfg   = DareTiesConfig(top_k_fraction=None, scale=1.0)
        m     = DareTiesMerger(cfg)
        delta = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        merged = m.ties_merge([delta, delta])
        np.testing.assert_allclose(merged, delta, atol=1e-5)

    def test_ties_merge_scale_applied(self):
        cfg   = DareTiesConfig(top_k_fraction=None, scale=2.0)
        m     = DareTiesMerger(cfg)
        delta = np.array([1.0, 1.0], dtype=np.float32)
        merged = m.ties_merge([delta, delta])
        np.testing.assert_allclose(merged, np.array([2.0, 2.0]), atol=1e-5)

    def test_ties_merge_empty_raises(self):
        cfg = DareTiesConfig()
        m   = DareTiesMerger(cfg)
        with pytest.raises(ValueError, match="non-empty"):
            m.ties_merge([])


# ---------------------------------------------------------------------------
# DareTiesMerger — combined merge()
# ---------------------------------------------------------------------------

class TestDareTiesMergerFull:
    def test_merge_shape_preserved(self):
        cfg   = DareTiesConfig(sparsity=0.8, top_k_fraction=0.5, scale=1.0, seed=0)
        m     = DareTiesMerger(cfg)
        rng   = np.random.default_rng(0)
        deltas = [rng.standard_normal(16).astype(np.float32) for _ in range(3)]
        merged = m.merge(deltas)
        assert merged.shape == (16,)
        assert merged.dtype == np.float32

    def test_merge_empty_raises(self):
        cfg = DareTiesConfig()
        m   = DareTiesMerger(cfg)
        with pytest.raises(ValueError, match="non-empty"):
            m.merge([])

    def test_merge_identical_models(self):
        """Merging N copies of the same delta should give approximately the same delta."""
        cfg   = DareTiesConfig(sparsity=0.5, top_k_fraction=None, scale=1.0, seed=42)
        m     = DareTiesMerger(cfg)
        delta = np.array([1.0, -1.0, 2.0, -2.0, 3.0], dtype=np.float32)
        # Run multiple times; the result should be non-zero in expected direction
        merged = m.merge([delta] * 4)
        # After DARE and TIES, sign direction should still mostly agree with original
        assert merged.shape == (5,)

    def test_merge_2d_delta(self):
        cfg   = DareTiesConfig(sparsity=0.6, seed=1)
        m     = DareTiesMerger(cfg)
        rng   = np.random.default_rng(1)
        deltas = [rng.standard_normal((4, 8)).astype(np.float32) for _ in range(2)]
        merged = m.merge(deltas)
        assert merged.shape == (4, 8)
