"""Tests for WeightDecompressStream — async double-buffer decompression pipeline."""
from __future__ import annotations

import numpy as np
import pytest

from squish.io.weight_decompress_stream import (
    WeightDecompressStream,
    WeightStreamConfig,
    WeightStreamHandle,
)


def _stream(**kw):
    return WeightDecompressStream(WeightStreamConfig(**kw))


class TestConfig:
    def test_defaults(self):
        c = WeightStreamConfig()
        assert c.bits == 4 and c.n_layers == 32 and c.n_threads == 2

    @pytest.mark.parametrize("kw", [
        {"n_layers": 0}, {"bits": 5}, {"bits": 7},
        {"chunk_size": 0}, {"n_threads": 0}, {"lookahead": -1},
    ])
    def test_invalid(self, kw):
        with pytest.raises(ValueError):
            WeightStreamConfig(**kw)


class TestHandle:
    def test_valid_statuses(self):
        for s in ("pending", "ready", "consumed"):
            assert WeightStreamHandle(0, status=s).status == s

    def test_invalid_status_raises(self):
        with pytest.raises(ValueError):
            WeightStreamHandle(0, status="bogus")


class TestCompressRoundTrip:
    @pytest.mark.parametrize("bits", [2, 3, 4, 8])
    def test_shape_preserved_and_finite(self, bits):
        rng = np.random.default_rng(0)
        W = rng.standard_normal((4, 6)).astype(np.float32)
        comp = WeightDecompressStream.compress_weight(W, bits=bits)
        assert comp.dtype == np.uint8 and comp.ndim == 1
        out = WeightDecompressStream.decompress_weight(comp, bits=bits)
        assert out.shape == W.shape
        assert np.isfinite(out).all()

    def test_bits8_reasonable_accuracy(self):
        rng = np.random.default_rng(1)
        W = rng.standard_normal((8, 8)).astype(np.float32)
        out = WeightDecompressStream.decompress_weight(
            WeightDecompressStream.compress_weight(W, bits=8), bits=8
        )
        # 8-bit symmetric quant on unit Gaussian: max abs error well under 0.1.
        assert np.abs(out - W).max() < 0.1

    def test_bits16_near_lossless(self):
        rng = np.random.default_rng(2)
        W = rng.standard_normal((3, 5)).astype(np.float32)
        out = WeightDecompressStream.decompress_weight(
            WeightDecompressStream.compress_weight(W, bits=16), bits=16
        )
        np.testing.assert_allclose(out, W, rtol=0, atol=1e-2)

    def test_all_zero_weight_no_divide_by_zero(self):
        # abs_max==0 is guarded to 1.0 → no divide-by-zero / NaN. The midpoint
        # offset (q_max/2 = 7.5 for bits=4) means zero maps to a small non-zero
        # value, so we assert finiteness + a tight bound, not exact zero.
        W = np.zeros((2, 3), np.float32)
        out = WeightDecompressStream.decompress_weight(
            WeightDecompressStream.compress_weight(W, bits=4), bits=4
        )
        assert out.shape == (2, 3)
        assert np.isfinite(out).all()
        assert np.abs(out).max() < 0.1

    def test_shape_inferred_from_header(self):
        W = np.arange(12, dtype=np.float32).reshape(3, 4)
        comp = WeightDecompressStream.compress_weight(W, bits=8)
        # shape=None → recovered from the embedded header
        out = WeightDecompressStream.decompress_weight(comp, bits=8, shape=None)
        assert out.shape == (3, 4)

    def test_explicit_shape_override(self):
        W = np.arange(12, dtype=np.float32).reshape(3, 4)
        comp = WeightDecompressStream.compress_weight(W, bits=8)
        out = WeightDecompressStream.decompress_weight(comp, bits=8, shape=(2, 6))
        assert out.shape == (2, 6)

    def test_compress_invalid_bits(self):
        with pytest.raises(ValueError):
            WeightDecompressStream.compress_weight(np.zeros((2, 2), np.float32), bits=5)

    def test_decompress_invalid_bits(self):
        with pytest.raises(ValueError):
            WeightDecompressStream.decompress_weight(np.zeros(16, np.uint8), bits=5)

    def test_recover_shape_matches_header(self):
        W = np.zeros((7, 2), np.float32)
        comp = WeightDecompressStream.compress_weight(W, bits=4)
        assert WeightDecompressStream._recover_shape(comp) == (7, 2)


class TestAsyncLifecycle:
    def test_submit_fetch_roundtrip(self):
        s = _stream(bits=8)
        W = np.random.default_rng(0).standard_normal((4, 4)).astype(np.float32)
        comp = WeightDecompressStream.compress_weight(W, bits=8)
        h = s.submit(0, comp)
        assert isinstance(h, WeightStreamHandle) and h.layer_idx == 0
        out = s.fetch(h)
        assert out.shape == (4, 4)
        np.testing.assert_allclose(
            out, WeightDecompressStream.decompress_weight(comp, bits=8), atol=0
        )
        assert h.status == "consumed"
        s.reset()

    def test_is_ready_transitions(self):
        s = _stream(bits=4)
        comp = WeightDecompressStream.compress_weight(np.ones((2, 2), np.float32), bits=4)
        h = s.submit(1, comp)
        h._future.result()              # force completion
        assert s.is_ready(h) is True
        s.fetch(h)
        assert s.is_ready(h) is False    # consumed → not ready
        s.reset()

    def test_fetch_consumed_raises(self):
        s = _stream(bits=4)
        comp = WeightDecompressStream.compress_weight(np.ones((2, 2), np.float32), bits=4)
        h = s.submit(0, comp)
        s.fetch(h)
        with pytest.raises(RuntimeError, match="already consumed"):
            s.fetch(h)
        s.reset()

    def test_fetch_without_future_raises(self):
        s = _stream()
        with pytest.raises(RuntimeError, match="no associated future"):
            s.fetch(WeightStreamHandle(3, status="pending"))
        s.reset()

    def test_is_ready_false_for_no_future(self):
        s = _stream()
        assert s.is_ready(WeightStreamHandle(0, status="pending")) is False
        s.reset()

    def test_prefetch_range_and_missing_layer(self):
        s = _stream(bits=8)
        comp = {i: WeightDecompressStream.compress_weight(
            np.ones((2, 2), np.float32) * i, bits=8) for i in (0, 1, 2)}
        handles = s.prefetch_range([0, 1, 2], comp)
        assert [h.layer_idx for h in handles] == [0, 1, 2]
        for h in handles:
            assert s.fetch(h).shape == (2, 2)
        with pytest.raises(KeyError):
            s.prefetch_range([9], comp)
        s.reset()


class TestStatsAndReset:
    def test_stats_track_submit_fetch(self):
        s = _stream(bits=8, n_threads=2, lookahead=3)
        comp = WeightDecompressStream.compress_weight(
            np.ones((4, 4), np.float32), bits=8)
        h = s.submit(0, comp)
        st = s.stats()
        assert st["n_submitted"] == 1 and st["n_pending"] == 1
        assert st["bits"] == 8 and st["n_threads"] == 2 and st["lookahead"] == 3
        s.fetch(h)
        st2 = s.stats()
        assert st2["n_fetched"] == 1 and st2["n_pending"] == 0
        assert st2["total_bytes_in"] > 0 and st2["total_bytes_out"] > 0
        assert st2["compression_ratio"] > 0
        s.reset()

    def test_reset_clears_stats(self):
        s = _stream(bits=8)
        comp = WeightDecompressStream.compress_weight(
            np.ones((2, 2), np.float32), bits=8)
        s.fetch(s.submit(0, comp))
        s.reset()
        st = s.stats()
        assert st["n_submitted"] == 0 and st["n_fetched"] == 0
        assert st["total_bytes_in"] == 0
        # Stream is reusable after reset.
        assert s.fetch(s.submit(0, comp)).shape == (2, 2)
        s.reset()

    def test_del_swallows_shutdown_errors(self):
        # __del__ must not raise even if the executor is already gone (GC /
        # interpreter-shutdown ordering) — the defensive handler swallows it.
        s = _stream()
        s._executor.shutdown(wait=False)
        del s._executor          # now self._executor.shutdown → AttributeError
        s.__del__()              # exercises the except handler; must not raise
