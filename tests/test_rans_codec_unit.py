"""tests/test_rans_codec_unit.py — unit tests for squish/rans_codec.py"""
import numpy as np
import pytest

from squish.rans_codec import RANSCodec

RNG = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestRANSCodecInit:
    def test_uniform_freq(self):
        freq = {i: 10 for i in range(256)}
        codec = RANSCodec(freq)
        assert codec._m_bits == 14

    def test_custom_m_bits(self):
        freq = {i: 1 for i in range(4)}
        codec = RANSCodec(freq, m_bits=10)
        assert codec._m_bits == 10

    def test_empty_freq_raises(self):
        with pytest.raises((ValueError, ZeroDivisionError, Exception)):
            RANSCodec({})

    def test_repr_or_no_crash(self):
        freq = {0: 5, 1: 3}
        codec = RANSCodec(freq)
        # Should not raise
        _ = repr(codec) if hasattr(codec, "__repr__") else str(codec)


# ---------------------------------------------------------------------------
# Encode / Decode round-trip
# ---------------------------------------------------------------------------

class TestRANSCodecRoundTrip:
    def _make_codec(self, symbols: np.ndarray) -> RANSCodec:
        vals, counts = np.unique(symbols, return_counts=True)
        freq = {int(v): int(c) for v, c in zip(vals, counts, strict=True)}
        return RANSCodec(freq)

    def test_single_symbol(self):
        symbols = np.array([7, 7, 7, 7], dtype=np.uint8)
        codec = self._make_codec(symbols)
        data = codec.encode(symbols)
        decoded = codec.decode(data, len(symbols))
        np.testing.assert_array_equal(decoded, symbols)

    def test_two_symbols(self):
        symbols = np.array([0, 1, 0, 1, 1, 0], dtype=np.uint8)
        codec = self._make_codec(symbols)
        data = codec.encode(symbols)
        decoded = codec.decode(data, len(symbols))
        np.testing.assert_array_equal(decoded, symbols)

    def test_all_256_symbols(self):
        symbols = np.arange(256, dtype=np.uint8)
        codec = self._make_codec(symbols)
        data = codec.encode(symbols)
        decoded = codec.decode(data, len(symbols))
        np.testing.assert_array_equal(decoded, symbols)

    def test_random_100_symbols(self):
        symbols = RNG.integers(0, 16, size=100, dtype=np.uint8)
        codec = self._make_codec(symbols)
        data = codec.encode(symbols)
        decoded = codec.decode(data, len(symbols))
        np.testing.assert_array_equal(decoded, symbols)

    def test_random_1000_symbols_skewed(self):
        # Skewed distribution: symbol 0 is ~80% likely
        raw = RNG.choice([0, 1, 2, 3], size=1000, p=[0.8, 0.1, 0.07, 0.03])
        symbols = raw.astype(np.uint8)
        codec = self._make_codec(symbols)
        data = codec.encode(symbols)
        decoded = codec.decode(data, len(symbols))
        np.testing.assert_array_equal(decoded, symbols)

    def test_empty_symbols(self):
        symbols = np.array([], dtype=np.uint8)
        codec = RANSCodec({0: 1, 1: 1})
        data = codec.encode(symbols)
        decoded = codec.decode(data, 0)
        assert len(decoded) == 0

    def test_symbols_not_in_training_freq(self):
        """Codec must handle symbols seen at decode that weren't in training freq."""
        freq = {0: 100, 1: 50}   # only 0 and 1 in freq
        codec = RANSCodec(freq)  # RANSCodec assigns min prob 1 to all 256 symbols
        symbols = np.array([0, 2, 1, 3], dtype=np.uint8)  # 2 and 3 are rare
        data = codec.encode(symbols)
        decoded = codec.decode(data, len(symbols))
        np.testing.assert_array_equal(decoded, symbols)


# ---------------------------------------------------------------------------
# to_dict / from_code_dict round-trip
# ---------------------------------------------------------------------------

class TestRANSCodecDict:
    def test_to_dict_has_type_key(self):
        freq = {0: 5, 1: 3, 2: 2}
        codec = RANSCodec(freq)
        d = codec.to_dict()
        assert d["type"] == "rans"
        assert "freq" in d
        assert "m_bits" in d

    def test_from_code_dict_round_trip(self):
        symbols = np.array([0, 0, 1, 2, 1, 0, 3], dtype=np.uint8)
        freq = {int(v): int(c) for v, c in zip(*np.unique(symbols, return_counts=True), strict=True)}
        codec = RANSCodec(freq)
        d = codec.to_dict()
        codec2 = RANSCodec.from_code_dict(d)
        data = codec.encode(symbols)
        decoded = codec2.decode(data, len(symbols))
        np.testing.assert_array_equal(decoded, symbols)

    def test_from_code_dict_raises_on_wrong_type(self):
        with pytest.raises(ValueError, match="(?i)rans"):
            RANSCodec.from_code_dict({"type": "huffman", "codes": {}})

    def test_from_code_dict_raises_on_no_type(self):
        with pytest.raises((ValueError, KeyError)):
            RANSCodec.from_code_dict({"codes": {}})


# ---------------------------------------------------------------------------
# compressed_size_estimate
# ---------------------------------------------------------------------------

class TestRANSCodecSizeEstimate:
    def test_estimate_returns_positive(self):
        symbols = np.array([0, 0, 0, 1, 1, 2], dtype=np.uint8)
        freq = {0: 3, 1: 2, 2: 1}
        codec = RANSCodec(freq)
        est = codec.compressed_size_estimate(symbols)
        assert est > 0.0

    def test_estimate_skewed_less_than_uniform(self):
        """Skewed distribution should compress better than uniform."""
        n = 1000
        skewed  = np.array([0] * 900 + [1] * 100, dtype=np.uint8)
        uniform = np.arange(n, dtype=np.uint8) % 2

        freq_sk  = {0: 900, 1: 100}
        freq_uni = {0: 500, 1: 500}
        est_sk  = RANSCodec(freq_sk).compressed_size_estimate(skewed)
        est_uni = RANSCodec(freq_uni).compressed_size_estimate(uniform)
        assert est_sk < est_uni  # more skewed = smaller estimate
