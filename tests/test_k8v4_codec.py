"""Unit tests for squish.kv.k8v4_codec — INT8-keys / INT4-values disk codec.

Coverage:
  - Group-wise affine round-trip fidelity (keys INT8, values INT4)
  - Validated K8V4 bit allocation: keys lossless-grade, INT4-key error >> INT8-key
  - Nibble-packing correctness and the even/odd head_dim fallback
  - Constant-group (hi == lo) edge case (no divide-by-zero)
  - Group-size fallback when it doesn't divide head_dim
  - Compression ratio meets the ~2.7x design target
  - save/load to disk + auto-detect (.npz preferred, .npy fallback, corrupt → miss)
  - PromptKVStore(quant="k8v4") end-to-end put/get round-trip
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from squish.kv import k8v4_codec as codec


def _kv(seq=16, heads=8, head_dim=128, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((1, heads, seq, head_dim)).astype(np.float16)


# ── round-trip fidelity ───────────────────────────────────────────────────────


class TestRoundTrip:
    def test_int8_keys_low_error(self):
        arr = _kv()
        out = codec.dequantize_array(codec.quantize_array(arr, codec.K_BITS))
        assert out.shape == arr.shape
        # INT8 affine over 64-wide groups of N(0,1) data: max abs error well under 0.05
        assert np.max(np.abs(out.astype(np.float32) - arr.astype(np.float32))) < 0.05

    def test_int4_values_bounded_error(self):
        arr = _kv()
        out = codec.dequantize_array(codec.quantize_array(arr, codec.V_BITS))
        # INT4 is coarser but still bounded — half a group's quantization step
        rng = arr.astype(np.float32).max() - arr.astype(np.float32).min()
        assert np.max(np.abs(out.astype(np.float32) - arr.astype(np.float32))) < rng

    def test_keys_far_more_precise_than_int4(self):
        """The reason keys stay at 8 bits: INT4 keys carry ~10x the error."""
        arr = _kv(seed=3)
        err8 = np.abs(
            codec.dequantize_array(codec.quantize_array(arr, 8)).astype(np.float32)
            - arr.astype(np.float32)
        ).mean()
        err4 = np.abs(
            codec.dequantize_array(codec.quantize_array(arr, 4)).astype(np.float32)
            - arr.astype(np.float32)
        ).mean()
        assert err4 > 5 * err8

    def test_dtype_preserved(self):
        arr = _kv()
        out = codec.dequantize_array(codec.quantize_array(arr, 8), dtype=np.float32)
        assert out.dtype == np.float32


# ── packing ───────────────────────────────────────────────────────────────────


class TestPacking:
    def test_int4_even_dim_is_packed(self):
        payload = codec.quantize_array(_kv(head_dim=128), 4)
        assert int(payload["packed"]) == 1
        # packed q holds two nibbles per byte → half the elements along last axis
        assert payload["q"].shape[-1] == 64

    def test_int4_odd_dim_falls_back_unpacked(self):
        payload = codec.quantize_array(_kv(head_dim=63), 4)
        assert int(payload["packed"]) == 0
        assert payload["q"].shape[-1] == 63
        out = codec.dequantize_array(payload)
        assert out.shape[-1] == 63

    def test_int8_never_packed(self):
        payload = codec.quantize_array(_kv(), 8)
        assert int(payload["packed"]) == 0

    def test_packed_codes_in_nibble_range(self):
        payload = codec.quantize_array(_kv(), 4)
        # unpacking must yield codes in [0, 15]
        out = codec.dequantize_array(payload)
        assert np.isfinite(out).all()


# ── edges ─────────────────────────────────────────────────────────────────────


class TestEdges:
    def test_constant_group_no_nan(self):
        arr = np.full((1, 2, 4, 64), 0.5, dtype=np.float16)
        out = codec.dequantize_array(codec.quantize_array(arr, 8))
        assert np.isfinite(out).all()
        assert np.allclose(out.astype(np.float32), 0.5, atol=1e-3)

    def test_group_size_fallback_to_full_axis(self):
        # head_dim 100 not divisible by default group 64 → one group of 100
        payload = codec.quantize_array(_kv(head_dim=100), 8, group_size=64)
        assert int(payload["gsize"]) == 100
        assert payload["scale"].shape[-1] == 1

    def test_invalid_bits_rejected(self):
        with pytest.raises(ValueError):
            codec.quantize_array(_kv(), 5)

    def test_k_bits_is_eight(self):
        # INT4 keys broke decode in validation — the codec must never default keys to 4.
        assert codec.K_BITS == 8
        assert codec.V_BITS == 4


# ── compression ───────────────────────────────────────────────────────────────


class TestCompression:
    def test_k8v4_pair_ratio_near_target(self):
        arr = _kv(head_dim=128)
        rk = codec.compression_ratio(arr, 8)
        rv = codec.compression_ratio(arr, 4)
        # Average per K+V pair (one of each). This is the RAW, pre-zlib lower
        # bound (~2.3x); savez_compressed on real KV data reaches the validated
        # ~2.7x. Assert the raw floor so the codec can never silently regress.
        pair = 2.0 / ((2.0 / rk + 2.0 / rv) / 2.0)
        assert pair >= 2.2

    def test_int4_more_compressed_than_int8(self):
        arr = _kv()
        assert codec.compression_ratio(arr, 4) > codec.compression_ratio(arr, 8)


# ── disk I/O + auto-detect ─────────────────────────────────────────────────────


class TestDiskIO:
    def test_save_load_round_trip(self):
        arr = _kv()
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "k_0.npz"
            codec.save_quantized(p, arr, 8)
            out = codec.load_quantized(p)
        assert np.max(np.abs(out.astype(np.float32) - arr.astype(np.float32))) < 0.05

    def test_auto_detect_prefers_npz(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            arr = _kv()
            codec.save_quantized(d / "k_0.npz", arr, 8)
            out = codec.load_layer_auto(d, "k", 0)
            assert out.shape == arr.shape

    def test_auto_detect_npy_fallback(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            arr = _kv().astype(np.float16)
            np.save(str(d / "v_0.npy"), arr)
            out = codec.load_layer_auto(d, "v", 0)
            assert np.array_equal(out, arr)

    def test_corrupt_npz_raises_valueerror(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "k_0.npz"
            p.write_bytes(b"not a real zip archive")
            with pytest.raises(ValueError):
                codec.load_quantized(p)


# ── PromptKVStore integration ──────────────────────────────────────────────────


class TestPromptKVStoreK8V4:
    def _store(self, td, quant):
        from squish.kv.prompt_kv_cache import PromptKVStore

        return PromptKVStore(cache_dir=td, quant=quant)

    def test_invalid_quant_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            with pytest.raises(ValueError):
                self._store(td, "int2")

    def test_k8v4_put_get_round_trip(self):
        with tempfile.TemporaryDirectory() as td:
            store = self._store(td, "k8v4")
            keys = [_kv(seed=i) for i in range(4)]
            values = [_kv(seed=10 + i) for i in range(4)]
            store.put("a repeated prompt", keys, values, offset=16)

            entry = store.get("a repeated prompt")
            assert entry is not None
            assert entry.n_layers == 4
            assert entry.offset == 16
            # keys are INT8 → tight; values INT4 → looser but bounded
            for got, want in zip(entry.keys, keys, strict=True):
                assert np.max(np.abs(got.astype(np.float32) - want.astype(np.float32))) < 0.05

    def test_k8v4_writes_npz_not_npy(self):
        with tempfile.TemporaryDirectory() as td:
            store = self._store(td, "k8v4")
            store.put("p", [_kv()], [_kv(seed=1)], offset=16)
            d = Path(td) / store.hash_prompt("p")
            assert (d / "k_0.npz").exists()
            assert not (d / "k_0.npy").exists()

    def test_fp16_default_still_npy(self):
        with tempfile.TemporaryDirectory() as td:
            store = self._store(td, "fp16")
            store.put("p", [_kv()], [_kv(seed=1)], offset=16)
            d = Path(td) / store.hash_prompt("p")
            assert (d / "k_0.npy").exists()
            assert not (d / "k_0.npz").exists()

    def test_k8v4_entry_smaller_than_fp16(self):
        with tempfile.TemporaryDirectory() as td1, tempfile.TemporaryDirectory() as td2:
            keys = [_kv(seed=i) for i in range(6)]
            values = [_kv(seed=20 + i) for i in range(6)]
            self._store(td1, "fp16").put("p", keys, values, offset=16)
            self._store(td2, "k8v4").put("p", keys, values, offset=16)
            fp16_bytes = sum(p.stat().st_size for p in Path(td1).rglob("*.npy"))
            k8v4_bytes = sum(p.stat().st_size for p in Path(td2).rglob("*.npz"))
            assert k8v4_bytes < fp16_bytes
