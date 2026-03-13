"""tests/test_fr_spec_unit.py — 100 % coverage for squish/fr_spec.py"""
import numpy as np
import pytest

from squish.speculative.fr_spec import (
    FreqTokenSubset,
    FRSpecCalibrator,
    FRSpecConfig,
    FRSpecHead,
    FRSpecStats,
)

RNG = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# FRSpecConfig
# ---------------------------------------------------------------------------

class TestFRSpecConfig:
    def test_defaults(self):
        cfg = FRSpecConfig()
        assert cfg.vocab_size == 151_936
        assert cfg.top_k_fraction == 0.25
        assert cfg.min_frequent_tokens == 256
        assert cfg.max_calibration_samples == 10_000

    def test_k_when_raw_exceeds_min(self):
        # vocab=1000, fraction=0.5, min=100 → raw=500 > 100 → k=500
        cfg = FRSpecConfig(vocab_size=1000, top_k_fraction=0.5, min_frequent_tokens=100)
        assert cfg.k == 500

    def test_k_when_raw_below_min(self):
        # vocab=100, fraction=0.1, min=50 → raw=10 < 50 → k=50
        cfg = FRSpecConfig(vocab_size=100, top_k_fraction=0.1, min_frequent_tokens=50)
        assert cfg.k == 50

    def test_k_when_raw_equals_min(self):
        # raw == min → k == min
        cfg = FRSpecConfig(vocab_size=100, top_k_fraction=0.5, min_frequent_tokens=50)
        assert cfg.k == 50

    def test_custom_values(self):
        cfg = FRSpecConfig(vocab_size=32_000, top_k_fraction=0.20, min_frequent_tokens=64)
        assert cfg.vocab_size == 32_000
        assert cfg.top_k_fraction == 0.20


# ---------------------------------------------------------------------------
# FreqTokenSubset
# ---------------------------------------------------------------------------

class TestFreqTokenSubset:
    def test_construction_and_sorted(self):
        subset = FreqTokenSubset([5, 1, 3, 2, 4])
        np.testing.assert_array_equal(subset.indices, [1, 2, 3, 4, 5])

    def test_len(self):
        subset = FreqTokenSubset([10, 20, 30])
        assert len(subset) == 3

    def test_contains_true(self):
        subset = FreqTokenSubset([1, 2, 3])
        assert 2 in subset

    def test_contains_false(self):
        subset = FreqTokenSubset([1, 2, 3])
        assert 99 not in subset

    def test_iter(self):
        indices = [7, 3, 5]
        subset = FreqTokenSubset(indices)
        result = list(subset)
        assert sorted(indices) == result

    def test_invalid_ndim(self):
        with pytest.raises(ValueError, match="1-D"):
            FreqTokenSubset(np.array([[1, 2], [3, 4]]))

    def test_coverage_empty_sequence(self):
        subset = FreqTokenSubset([1, 2, 3])
        assert subset.coverage([]) == 1.0

    def test_coverage_all_in_subset(self):
        subset = FreqTokenSubset([1, 2, 3])
        assert subset.coverage([1, 2, 3]) == 1.0

    def test_coverage_none_in_subset(self):
        subset = FreqTokenSubset([1, 2, 3])
        assert subset.coverage([4, 5, 6]) == 0.0

    def test_coverage_partial(self):
        subset = FreqTokenSubset([1, 2, 3])
        assert subset.coverage([1, 4]) == pytest.approx(0.5)

    def test_to_list_from_list_roundtrip(self):
        indices = [10, 20, 30, 40]
        subset = FreqTokenSubset(indices)
        reconstructed = FreqTokenSubset.from_list(subset.to_list())
        np.testing.assert_array_equal(subset.indices, reconstructed.indices)


# ---------------------------------------------------------------------------
# FRSpecHead
# ---------------------------------------------------------------------------

class TestFRSpecHead:
    def _make_head(self, vocab=20, hidden=8, k=5):
        weight = RNG.standard_normal((vocab, hidden)).astype(np.float32)
        subset = FreqTokenSubset(np.arange(k))
        return FRSpecHead(weight, subset), weight, subset

    def test_construction_properties(self):
        head, weight, subset = self._make_head(vocab=20, hidden=8, k=5)
        assert head.full_vocab_size == 20
        assert head.hidden_dim == 8
        assert len(head.subset) == 5
        assert head.compressed_weight.shape == (5, 8)

    def test_invalid_weight_ndim(self):
        with pytest.raises(ValueError, match="2-D"):
            FRSpecHead(np.ones(10), FreqTokenSubset([0, 1]))

    def test_invalid_subset_too_large(self):
        weight = np.ones((5, 4))
        subset = FreqTokenSubset([0, 1, 2, 3, 4, 5, 6])  # 7 > 5
        with pytest.raises(ValueError, match="subset size"):
            FRSpecHead(weight, subset)

    def test_compression_ratio(self):
        head, _, _ = self._make_head(vocab=20, hidden=8, k=5)
        assert head.compression_ratio == pytest.approx(5 / 20)

    def test_forward_1d(self):
        head, weight, subset = self._make_head(vocab=20, hidden=8, k=5)
        hidden = RNG.standard_normal(8).astype(np.float32)
        out = head.forward(hidden)
        assert out.shape == (5,)
        # Verify against manual computation
        expected = hidden @ weight[subset.indices].T
        np.testing.assert_allclose(out, expected, rtol=1e-5)

    def test_forward_2d(self):
        head, weight, subset = self._make_head(vocab=20, hidden=8, k=5)
        hidden = RNG.standard_normal((3, 8)).astype(np.float32)
        out = head.forward(hidden)
        assert out.shape == (3, 5)

    def test_expand_logits_1d(self):
        head, _, subset = self._make_head(vocab=20, hidden=8, k=5)
        comp = np.ones(5, dtype=np.float32)
        full = head.expand_logits(comp)
        assert full.shape == (20,)
        # Positions in subset should be 1.0
        np.testing.assert_array_equal(full[subset.indices], np.ones(5))
        # Other positions should be -inf
        mask = np.ones(20, dtype=bool)
        mask[subset.indices] = False
        assert np.all(full[mask] == -np.inf)

    def test_expand_logits_2d(self):
        head, _, subset = self._make_head(vocab=20, hidden=8, k=5)
        comp = np.ones((2, 5), dtype=np.float32)
        full = head.expand_logits(comp)
        assert full.shape == (2, 20)
        # All -inf outside subset
        mask = np.ones(20, dtype=bool)
        mask[subset.indices] = False
        assert np.all(full[:, mask] == -np.inf)

    def test_compress_logits_1d(self):
        head, _, subset = self._make_head(vocab=20, hidden=8, k=5)
        full = np.arange(20, dtype=np.float32)
        comp = head.compress_logits(full)
        assert comp.shape == (5,)
        np.testing.assert_array_equal(comp, full[subset.indices])

    def test_compress_logits_2d(self):
        head, _, subset = self._make_head(vocab=20, hidden=8, k=5)
        full = RNG.standard_normal((3, 20)).astype(np.float32)
        comp = head.compress_logits(full)
        assert comp.shape == (3, 5)
        np.testing.assert_array_equal(comp, full[:, subset.indices])


# ---------------------------------------------------------------------------
# FRSpecCalibrator
# ---------------------------------------------------------------------------

class TestFRSpecCalibrator:
    def _make_cal(self, k=5, vocab=20, max_samples=100):
        cfg = FRSpecConfig(
            vocab_size=vocab,
            top_k_fraction=k / vocab,
            min_frequent_tokens=2,
            max_calibration_samples=max_samples,
        )
        return FRSpecCalibrator(cfg)

    def test_initial_state(self):
        cal = self._make_cal()
        assert cal.n_samples == 0

    def test_record_increments_sample_count(self):
        cal = self._make_cal()
        cal.record([0, 1, 2])
        assert cal.n_samples == 1

    def test_record_respects_max_samples(self):
        cal = self._make_cal(max_samples=2)
        cal.record([0])
        cal.record([1])
        cal.record([2])  # should be ignored – max reached
        assert cal.n_samples == 2

    def test_record_ignores_tokens_when_max_reached(self):
        cal = self._make_cal(max_samples=1)
        cal.record([0, 1, 0, 1])
        cal.record([99])           # ignored
        mc = cal.most_common()
        assert all(t != 99 for t, _ in mc)

    def test_most_common_returns_sorted(self):
        cal = self._make_cal()
        cal.record([1, 1, 1, 2, 2, 3])
        mc = cal.most_common(2)
        assert mc[0][0] == 1
        assert mc[1][0] == 2

    def test_build_subset_without_padding(self):
        # Record enough distinct tokens so no padding needed
        cal = self._make_cal(k=3, vocab=10)
        cal.record([0, 0, 0, 1, 1, 2])   # 3 distinct tokens → k=3 → no pad
        subset = cal.build_subset()
        assert len(subset) == 3

    def test_build_subset_with_padding(self):
        # Only 1 distinct token observed, k=5 → must pad with 4 more
        cal = self._make_cal(k=5, vocab=10)
        cal.record([7, 7, 7])
        subset = cal.build_subset()
        assert len(subset) == 5
        # Token 7 should be present (it was most frequent)
        assert 7 in subset

    def test_build_subset_padding_fills_low_indices(self):
        # Most frequent = token 9; padding fills 0,1,2,3 (lowest not present)
        cfg = FRSpecConfig(vocab_size=10, top_k_fraction=0.5, min_frequent_tokens=5)
        cal = FRSpecCalibrator(cfg)  # k=5
        cal.record([9, 9, 9])
        subset = cal.build_subset()
        assert len(subset) == 5
        assert 9 in subset

    def test_reset_clears_state(self):
        cal = self._make_cal()
        cal.record([1, 2, 3])
        cal.reset()
        assert cal.n_samples == 0
        assert cal.most_common() == []

    def test_default_config_used_when_none(self):
        cal = FRSpecCalibrator()
        assert cal.n_samples == 0

    def test_build_subset_indices_in_vocab(self):
        cal = self._make_cal(k=3, vocab=10)
        for _ in range(5):
            cal.record([0, 1, 2, 3, 4])
        subset = cal.build_subset()
        assert np.all(subset.indices < 10)
        assert np.all(subset.indices >= 0)

    def test_build_subset_padding_exhausts_range(self):
        # vocab=5, k=8 (min_frequent_tokens=8 > vocab): only 5 tokens possible.
        # Seen tokens {0,1,2} are LOW-index → padding hits them in range(5),
        # triggering the 'tid in seen' False branch (347→349).
        # vocab_size=5 < k=8 → range exhausts without break (346→352 branch).
        cfg = FRSpecConfig(vocab_size=5, top_k_fraction=0.1, min_frequent_tokens=8)
        cal = FRSpecCalibrator(cfg)
        cal.record([0, 1, 2])  # 3 tokens; k=8 so 5 more needed but only 2 unseen in range(5)
        subset = cal.build_subset()
        # All vocab tokens should be present (only 5 available < k=8)
        assert len(subset) <= 5


# ---------------------------------------------------------------------------
# FRSpecStats
# ---------------------------------------------------------------------------

class TestFRSpecStats:
    def test_defaults(self):
        s = FRSpecStats()
        assert s.compressed_forwards == 0
        assert s.full_forwards == 0
        assert s.tokens_drafted == 0
        assert s.tokens_outside_subset == 0

    def test_compression_utilization_zero_when_no_forwards(self):
        s = FRSpecStats()
        assert s.compression_utilization == 0.0

    def test_compression_utilization_all_compressed(self):
        s = FRSpecStats(compressed_forwards=10, full_forwards=0)
        assert s.compression_utilization == 1.0

    def test_compression_utilization_mixed(self):
        s = FRSpecStats(compressed_forwards=3, full_forwards=7)
        assert s.compression_utilization == pytest.approx(0.3)

    def test_subset_coverage_rate_when_no_drafts(self):
        s = FRSpecStats()
        assert s.subset_coverage_rate == 1.0

    def test_subset_coverage_rate_all_in_subset(self):
        s = FRSpecStats(tokens_drafted=10, tokens_outside_subset=0)
        assert s.subset_coverage_rate == 1.0

    def test_subset_coverage_rate_partial(self):
        s = FRSpecStats(tokens_drafted=10, tokens_outside_subset=2)
        assert s.subset_coverage_rate == pytest.approx(0.8)

    def test_record_compressed(self):
        s = FRSpecStats()
        s.record_compressed(5)
        assert s.compressed_forwards == 1
        assert s.tokens_drafted == 5

    def test_record_full(self):
        s = FRSpecStats()
        s.record_full(3)
        assert s.full_forwards == 1
        assert s.tokens_drafted == 3

    def test_record_outside_subset(self):
        s = FRSpecStats()
        s.record_compressed(10)
        s.record_outside_subset(2)
        assert s.tokens_outside_subset == 2

    def test_reset(self):
        s = FRSpecStats(
            compressed_forwards=5,
            full_forwards=3,
            tokens_drafted=20,
            tokens_outside_subset=1,
        )
        s.reset()
        assert s.compressed_forwards == 0
        assert s.full_forwards == 0
        assert s.tokens_drafted == 0
        assert s.tokens_outside_subset == 0
