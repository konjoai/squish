"""Tests for squish.kv.delta — cache snapshot diff/restore."""
from __future__ import annotations

import numpy as np
import pytest

from squish.kv.delta import KVCacheDelta, delta_from_layer_caches


def _rand_stack(n_tokens, n_heads=2, head_dim=4, seed=0, dtype=np.float16):
    rng = np.random.default_rng(seed)
    k = rng.standard_normal((n_tokens, n_heads, head_dim)).astype(dtype)
    v = rng.standard_normal((n_tokens, n_heads, head_dim)).astype(dtype)
    return k, v


class TestCompute:
    def test_append_only(self):
        bk, bv = _rand_stack(4, seed=1)
        # Target = base + 3 new tokens
        extra_k, extra_v = _rand_stack(3, seed=2)
        tk = np.concatenate([bk, extra_k], axis=0)
        tv = np.concatenate([bv, extra_v], axis=0)
        delta = KVCacheDelta.compute(bk, bv, tk, tv)
        assert delta.base_len == 4
        assert delta.truncate == 0
        assert delta.n_new == 3

    def test_truncate_and_replace(self):
        bk, bv = _rand_stack(6, seed=3)
        # Keep first 4, drop last 2, add 3 new
        new_k, new_v = _rand_stack(3, seed=4)
        tk = np.concatenate([bk[:4], new_k], axis=0)
        tv = np.concatenate([bv[:4], new_v], axis=0)
        delta = KVCacheDelta.compute(bk, bv, tk, tv)
        assert delta.base_len == 4
        assert delta.truncate == 2
        assert delta.n_new == 3

    def test_identical_yields_empty_delta(self):
        bk, bv = _rand_stack(5, seed=5)
        delta = KVCacheDelta.compute(bk, bv, bk.copy(), bv.copy())
        assert delta.is_empty
        assert delta.base_len == 5
        assert delta.n_new == 0
        assert delta.truncate == 0

    def test_truncate_only(self):
        # Target is a prefix of base.
        bk, bv = _rand_stack(5, seed=6)
        delta = KVCacheDelta.compute(bk, bv, bk[:3], bv[:3])
        assert delta.base_len == 3
        assert delta.truncate == 2
        assert delta.n_new == 0

    def test_completely_different(self):
        bk, bv = _rand_stack(4, seed=7)
        tk, tv = _rand_stack(4, seed=8)
        delta = KVCacheDelta.compute(bk, bv, tk, tv)
        assert delta.base_len == 0
        assert delta.truncate == 4
        assert delta.n_new == 4

    def test_empty_base(self):
        bk = np.zeros((0, 2, 4), dtype=np.float16)
        bv = np.zeros((0, 2, 4), dtype=np.float16)
        tk, tv = _rand_stack(3, seed=9)
        delta = KVCacheDelta.compute(bk, bv, tk, tv)
        assert delta.base_len == 0
        assert delta.truncate == 0
        assert delta.n_new == 3

    def test_shape_mismatch_raises(self):
        bk, bv = _rand_stack(3, n_heads=2, head_dim=4)
        tk, tv = _rand_stack(3, n_heads=2, head_dim=8)        # different head_dim
        with pytest.raises(ValueError, match="shape mismatch"):
            KVCacheDelta.compute(bk, bv, tk, tv)

    def test_atol_matches_near_equal_prefix(self):
        bk, bv = _rand_stack(4, seed=10, dtype=np.float32)
        tk = bk.copy()
        tv = bv.copy()
        # Inject tiny noise within tolerance into first 3 tokens
        tk[:3] += 1e-6
        tv[:3] += 1e-6
        # Modify the 4th token significantly so it should NOT match.
        tk[3] += 1.0
        delta = KVCacheDelta.compute(bk, bv, tk, tv, atol=1e-4)
        assert delta.base_len == 3
        assert delta.n_new == 1


class TestApply:
    def test_apply_reconstructs_target(self):
        bk, bv = _rand_stack(5, seed=11)
        new_k, new_v = _rand_stack(2, seed=12)
        tk = np.concatenate([bk[:3], new_k], axis=0)
        tv = np.concatenate([bv[:3], new_v], axis=0)
        delta = KVCacheDelta.compute(bk, bv, tk, tv)
        rk, rv = delta.apply(bk, bv)
        assert np.array_equal(rk, tk)
        assert np.array_equal(rv, tv)

    def test_apply_with_empty_delta_returns_base_prefix(self):
        bk, bv = _rand_stack(4, seed=13)
        delta = KVCacheDelta.compute(bk, bv, bk.copy(), bv.copy())
        rk, rv = delta.apply(bk, bv)
        assert np.array_equal(rk, bk)
        assert np.array_equal(rv, bv)
        # Returned arrays must be copies, not views
        assert rk is not bk

    def test_apply_to_wrong_base_raises(self):
        bk, bv = _rand_stack(5, seed=14)
        new_k, new_v = _rand_stack(2, seed=15)
        tk = np.concatenate([bk[:3], new_k], axis=0)
        tv = np.concatenate([bv[:3], new_v], axis=0)
        delta = KVCacheDelta.compute(bk, bv, tk, tv)

        wrong_bk, wrong_bv = _rand_stack(7, seed=16)
        with pytest.raises(ValueError, match="does not match base"):
            delta.apply(wrong_bk, wrong_bv)


class TestWireFormat:
    def test_roundtrip_with_payload(self):
        bk, bv = _rand_stack(3, seed=20, dtype=np.float16)
        new_k, new_v = _rand_stack(2, seed=21, dtype=np.float16)
        tk = np.concatenate([bk, new_k], axis=0)
        tv = np.concatenate([bv, new_v], axis=0)
        delta = KVCacheDelta.compute(bk, bv, tk, tv)
        buf = delta.encode_bytes()
        decoded = KVCacheDelta.decode_bytes(buf)
        assert decoded.base_len == delta.base_len
        assert decoded.truncate == delta.truncate
        assert decoded.n_new == delta.n_new
        assert np.array_equal(decoded.new_keys,   delta.new_keys)
        assert np.array_equal(decoded.new_values, delta.new_values)

    def test_roundtrip_empty(self):
        bk, bv = _rand_stack(3, seed=22)
        delta = KVCacheDelta.compute(bk, bv, bk.copy(), bv.copy())
        buf = delta.encode_bytes()
        decoded = KVCacheDelta.decode_bytes(buf)
        assert decoded.is_empty
        assert decoded.base_len == 3

    def test_fp32_roundtrip(self):
        bk = np.arange(24, dtype=np.float32).reshape(3, 2, 4)
        bv = bk.copy()
        new_k = np.full((1, 2, 4), 99.0, dtype=np.float32)
        new_v = new_k.copy()
        tk = np.concatenate([bk, new_k], axis=0)
        tv = np.concatenate([bv, new_v], axis=0)
        delta = KVCacheDelta.compute(bk, bv, tk, tv)
        buf = delta.encode_bytes()
        decoded = KVCacheDelta.decode_bytes(buf)
        assert decoded.new_keys.dtype == np.float32
        assert np.array_equal(decoded.new_keys, new_k)

    def test_bad_magic_raises(self):
        with pytest.raises(ValueError, match="bad magic"):
            KVCacheDelta.decode_bytes(b"NOTSQDLT" + b"\x00" * 20)

    def test_short_buffer_raises(self):
        with pytest.raises(ValueError, match="too short"):
            KVCacheDelta.decode_bytes(b"\x00" * 4)

    def test_truncated_payload_raises(self):
        bk, bv = _rand_stack(2, seed=30)
        tk = np.concatenate([bk, _rand_stack(1, seed=31)[0]], axis=0)
        tv = np.concatenate([bv, _rand_stack(1, seed=31)[1]], axis=0)
        delta = KVCacheDelta.compute(bk, bv, tk, tv)
        buf = delta.encode_bytes()
        with pytest.raises(ValueError, match="buffer length"):
            KVCacheDelta.decode_bytes(buf[:-4])


class TestSize:
    def test_size_bytes_matches_payload(self):
        bk, bv = _rand_stack(2, seed=40)
        new_k, new_v = _rand_stack(3, seed=41)
        tk = np.concatenate([bk, new_k], axis=0)
        tv = np.concatenate([bv, new_v], axis=0)
        delta = KVCacheDelta.compute(bk, bv, tk, tv)
        # 3 tokens × 2 heads × 4 head_dim × 2 bytes × 2 buffers
        assert delta.size_bytes() == 3 * 2 * 4 * 2 * 2


class TestMultiLayer:
    def test_layer_helper_one_per_layer(self):
        base   = [_rand_stack(4, seed=i)     for i in range(3)]
        target = [_rand_stack(6, seed=i + 100) for i in range(3)]
        deltas = delta_from_layer_caches(base, target)
        assert len(deltas) == 3
        # base and target are completely different (different seeds)
        for d in deltas:
            assert d.n_new == 6
            assert d.truncate == 4

    def test_layer_helper_length_mismatch(self):
        with pytest.raises(ValueError, match="layer count mismatch"):
            delta_from_layer_caches(
                [_rand_stack(3, seed=1)],
                [_rand_stack(3, seed=2), _rand_stack(3, seed=3)],
            )
