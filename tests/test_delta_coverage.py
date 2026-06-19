"""Behavioral coverage for the validation / wire-format error paths of
``squish.kv.delta`` left untested by the baseline suite. Pure-Python numpy.
"""
from __future__ import annotations

import numpy as np
import pytest

from squish.kv.delta import KVCacheDelta, delta_from_layer_caches


def _stack(n, heads=2, dim=3, fill=1.0):
    return np.full((n, heads, dim), fill, dtype=np.float16)


# ── compute / apply roundtrip ────────────────────────────────────────────────


def test_compute_apply_roundtrip():
    base_k, base_v = _stack(3), _stack(3) + 5
    tgt_k = np.concatenate([base_k, _stack(2, fill=9)], axis=0)
    tgt_v = np.concatenate([base_v, _stack(2, fill=9)], axis=0)
    delta = KVCacheDelta.compute(base_k, base_v, tgt_k, tgt_v)
    assert delta.base_len == 3 and delta.truncate == 0 and delta.n_new == 2
    out_k, out_v = delta.apply(base_k, base_v)
    np.testing.assert_array_equal(out_k, tgt_k)
    np.testing.assert_array_equal(out_v, tgt_v)


def test_compute_with_atol_and_truncation():
    base_k, base_v = _stack(4), _stack(4)
    tgt_k, tgt_v = _stack(2), _stack(2)  # shorter → truncation
    delta = KVCacheDelta.compute(base_k, base_v, tgt_k, tgt_v, atol=0.01)
    assert delta.truncate == 2 and delta.n_new == 0
    assert delta.is_empty is False  # truncate != 0


def test_apply_empty_delta_copies_base():
    base_k, base_v = _stack(3), _stack(3)
    delta = KVCacheDelta(base_len=3, truncate=0,
                         new_keys=np.zeros((0, 2, 3), np.float16),
                         new_values=np.zeros((0, 2, 3), np.float16))
    out_k, _ = delta.apply(base_k, base_v)
    np.testing.assert_array_equal(out_k, base_k)
    assert out_k is not base_k  # fresh copy


# ── apply validation ─────────────────────────────────────────────────────────


def test_apply_base_length_mismatch():
    delta = KVCacheDelta(base_len=3, truncate=0,
                         new_keys=_stack(1), new_values=_stack(1))
    with pytest.raises(ValueError, match="does not match base"):
        delta.apply(_stack(5), _stack(5))  # base has 5, expected 3


def test_apply_payload_shape_mismatch():
    # base_len + truncate == base_n, but the new-token head/dim differs (184).
    delta = KVCacheDelta(base_len=5, truncate=0,
                         new_keys=_stack(1, heads=9, dim=9),
                         new_values=_stack(1, heads=9, dim=9))
    with pytest.raises(ValueError, match="payload shape"):
        delta.apply(_stack(5), _stack(5))


# ── wire format ──────────────────────────────────────────────────────────────


def test_encode_decode_roundtrip():
    delta = KVCacheDelta(base_len=2, truncate=1,
                         new_keys=_stack(2, fill=7), new_values=_stack(2, fill=8))
    buf = delta.encode_bytes()
    out = KVCacheDelta.decode_bytes(buf)
    assert out.base_len == 2 and out.truncate == 1 and out.n_new == 2
    np.testing.assert_array_equal(out.new_keys, delta.new_keys)


def test_encode_decode_empty():
    delta = KVCacheDelta(base_len=4, truncate=2,
                         new_keys=np.zeros((0, 0, 0), np.float16),
                         new_values=np.zeros((0, 0, 0), np.float16))
    out = KVCacheDelta.decode_bytes(delta.encode_bytes())
    assert out.base_len == 4 and out.truncate == 2 and out.n_new == 0


def test_encode_unsupported_dtype():
    delta = KVCacheDelta(base_len=0, truncate=0,
                         new_keys=_stack(1).astype(np.float64),
                         new_values=_stack(1).astype(np.float64))
    with pytest.raises(ValueError, match="unsupported dtype for wire format"):
        delta.encode_bytes()


def test_decode_too_short():
    with pytest.raises(ValueError, match="too short"):
        KVCacheDelta.decode_bytes(b"x")


def test_decode_bad_magic():
    buf = bytearray(KVCacheDelta(base_len=0, truncate=0,
                                 new_keys=np.zeros((0, 0, 0), np.float16),
                                 new_values=np.zeros((0, 0, 0), np.float16)).encode_bytes())
    buf[:6] = b"BADMAG"
    with pytest.raises(ValueError, match="bad magic"):
        KVCacheDelta.decode_bytes(bytes(buf))


def test_decode_unsupported_dtype_char():
    delta = KVCacheDelta(base_len=2, truncate=0,
                         new_keys=_stack(2), new_values=_stack(2))
    buf = bytearray(delta.encode_bytes())
    buf[22] = ord("Z")  # invalid dtype char (249)
    with pytest.raises(ValueError, match="unsupported dtype char"):
        KVCacheDelta.decode_bytes(bytes(buf))


def test_decode_length_mismatch():
    delta = KVCacheDelta(base_len=0, truncate=0,
                         new_keys=_stack(2), new_values=_stack(2))
    buf = delta.encode_bytes()
    with pytest.raises(ValueError, match="!= expected"):
        KVCacheDelta.decode_bytes(buf[:-4])  # truncated payload


# ── _validate_kv_stack (via compute) ─────────────────────────────────────────


def test_validate_rejects_non_3d():
    with pytest.raises(ValueError, match="must be 3-D"):
        KVCacheDelta.compute(np.zeros((3, 2)), np.zeros((3, 2)),
                             _stack(3), _stack(3))  # base is 2-D (286)


def test_validate_rejects_key_value_shape_mismatch():
    with pytest.raises(ValueError, match="!= base_values shape"):
        KVCacheDelta.compute(_stack(3), _stack(3, dim=4),  # keys vs values differ (291)
                             _stack(3), _stack(3))


def test_compute_head_dim_mismatch_base_vs_target():
    with pytest.raises(ValueError, match="shape mismatch"):
        KVCacheDelta.compute(_stack(3, heads=2), _stack(3, heads=2),
                             _stack(3, heads=4), _stack(3, heads=4))


# ── delta_from_layer_caches ──────────────────────────────────────────────────


def test_delta_from_layer_caches():
    base = [(_stack(2), _stack(2)), (_stack(2), _stack(2))]
    target = [(_stack(3), _stack(3)), (_stack(2), _stack(2))]
    deltas = delta_from_layer_caches(base, target)
    assert len(deltas) == 2 and deltas[0].n_new == 1


def test_delta_from_layer_caches_count_mismatch():
    with pytest.raises(ValueError, match="layer count mismatch"):
        delta_from_layer_caches([(_stack(2), _stack(2))], [])
