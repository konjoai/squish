"""tests/test_kv_int4.py — W105 INT4 KV cache extension tests.

Coverage:
  - Per-channel INT4 quantize/dequantize roundtrip + reconstruction SNR
  - Nibble-packing layout (2 indices per uint8 byte along head_dim,
    low nibble = even col, high nibble = odd col)
  - Storage size: head_dim/2 bytes vs head_dim for INT8 (2× reduction)
  - QuantizedKVCache(mode="int4"): construction, append, get_full_kv shape
  - HadamardKVCache(mode="int4"): SNR ordering INT8 > INT4 > INT2
  - Mode validation: rejects int4 + svd_rank / comm_vq_bits / qfilter_rank
  - Disk-tier guardrail rejects int4
  - recommended_kv_mode 3-tier dispatch (W105) and 2-tier compatibility (W104)
  - INT8 + INT2 paths unchanged after W105 wiring (no regressions)
"""

import numpy as np
import pytest

from squish.kv.kv_cache import (
    HadamardKVCache,
    KV_INT2_AUTO_THRESHOLD,
    KV_INT4_DEFAULT_THRESHOLD,
    KVLayerCache,
    QuantizedKVCache,
    _KV_INT2_LEVELS,
    _KV_INT4_LEVELS,
    _dequantize_int2_per_channel,
    _dequantize_int4_per_channel,
    _dequantize_int8_per_channel,
    _kv_dequantize_per_channel,
    _kv_quantize_per_channel,
    _quantize_int2_per_channel,
    _quantize_int4_per_channel,
    _quantize_int8_per_channel,
    recommended_kv_mode,
    recommended_kv_mode_3tier,
)


def _snr_db(signal: np.ndarray, recon: np.ndarray) -> float:
    s = signal.astype(np.float64)
    r = recon.astype(np.float64)
    sig = float(np.mean(s * s))
    err = float(np.mean((s - r) ** 2))
    return 10.0 * np.log10(sig / err) if err > 0 else float("inf")


# ---------------------------------------------------------------------------
# 1. Codebook + low-level quant/dequant helpers
# ---------------------------------------------------------------------------


def test_int4_codebook_is_symmetric_uniform_16_level():
    expected = np.arange(16, dtype=np.float32) - 7.5    # [-7.5, ..., 7.5]
    np.testing.assert_array_equal(_KV_INT4_LEVELS, expected)
    # Symmetry: levels[k] == -levels[15-k]
    np.testing.assert_array_almost_equal(
        _KV_INT4_LEVELS, -_KV_INT4_LEVELS[::-1]
    )


def test_int4_pack_shape():
    """Packed shape is (n_tokens, head_dim/2) uint8."""
    arr = np.random.randn(7, 64).astype(np.float16)
    packed, scale = _quantize_int4_per_channel(arr)
    assert packed.shape == (7, 32)
    assert packed.dtype == np.uint8
    assert scale.shape == (7,)
    assert scale.dtype == np.float32


def test_int4_pack_unpack_indices_roundtrip():
    """All 16 codebook values quantize to their own index and round-trip exact."""
    # Levels [-7.5, ..., 7.5] · scale=1 → exactly the codebook
    levels = _KV_INT4_LEVELS                                     # (16,)
    arr = levels[None, :].repeat(2, axis=0).astype(np.float16)   # (2, 16)
    packed, scale = _quantize_int4_per_channel(arr)
    deq = _dequantize_int4_per_channel(packed, scale, 16).astype(np.float32)
    np.testing.assert_allclose(deq, arr.astype(np.float32), atol=1e-3)


def test_int4_nibble_packing_layout():
    """Low nibble holds even columns, high nibble holds odd columns."""
    # idx values 0..15 in alternating order across 4 cols.
    # We construct values that map deterministically to indices 0,1,2,3.
    # scale = max(|v|)/7.5; with v_max = 7.5, scale=1.  Values:
    #   idx=0 → -7.5,  idx=1 → -6.5,  idx=2 → -5.5,  idx=3 → -4.5
    arr = np.array(
        [[-7.5, -6.5, -5.5, -4.5]] * 1, dtype=np.float16
    )                                                  # (1, 4)
    packed, _ = _quantize_int4_per_channel(arr)
    # Two bytes for 4 cols.  byte0: low=idx0(0), high=idx1(1) → 0x10
    #                       byte1: low=idx2(2), high=idx3(3) → 0x32
    np.testing.assert_array_equal(packed, np.array([[0x10, 0x32]], dtype=np.uint8))


def test_int4_storage_is_half_of_int8():
    """Nibble-packed INT4 occupies head_dim/2 bytes per token vs head_dim for INT8."""
    arr = np.random.randn(100, 128).astype(np.float16)
    packed_int4, _ = _quantize_int4_per_channel(arr)
    int8_q, _ = _quantize_int8_per_channel(arr)
    assert packed_int4.nbytes * 2 == int8_q.nbytes


def test_int4_storage_is_double_of_int2():
    """INT4 takes 2× the space of INT2 (4-per-byte vs 2-per-byte)."""
    arr = np.random.randn(64, 128).astype(np.float16)
    p4, _ = _quantize_int4_per_channel(arr)
    p2, _ = _quantize_int2_per_channel(arr)
    assert p4.nbytes == 2 * p2.nbytes


def test_int4_rejects_odd_head_dim():
    arr = np.random.randn(4, 7).astype(np.float16)
    with pytest.raises(ValueError, match="divisible by 2"):
        _quantize_int4_per_channel(arr)


def test_int4_rejects_wrong_rank_input():
    arr = np.random.randn(4, 8, 16).astype(np.float16)
    with pytest.raises(ValueError, match="2-D"):
        _quantize_int4_per_channel(arr)


def test_int4_dequant_rejects_packed_shape_mismatch():
    arr = np.random.randn(3, 64).astype(np.float16)
    packed, scale = _quantize_int4_per_channel(arr)
    # packed is (3, 32); claim head_dim=128 → expects (3, 64) → mismatch
    with pytest.raises(ValueError, match="!= head_dim/2"):
        _dequantize_int4_per_channel(packed, scale, 128)


def test_int4_dequant_rejects_odd_head_dim():
    packed = np.zeros((1, 4), dtype=np.uint8)
    scale = np.ones((1,), dtype=np.float32)
    with pytest.raises(ValueError, match="divisible by 2"):
        _dequantize_int4_per_channel(packed, scale, 7)


def test_int4_zero_input_does_not_blow_up():
    arr = np.zeros((5, 16), dtype=np.float16)
    packed, scale = _quantize_int4_per_channel(arr)
    deq = _dequantize_int4_per_channel(packed, scale, 16).astype(np.float32)
    # All-zero input → idx=8 (the closest level to 0 is +0.5·1e-8 ≈ 0).
    # Check magnitude is bounded by one quantisation step (scale * 1.0).
    assert np.abs(deq).max() < 1e-7


def test_int4_per_token_independence():
    """Each token row gets its own scale — different magnitudes don't bleed."""
    arr = np.array(
        [[1e-3] * 16,
         [10.0]  * 16],
        dtype=np.float16,
    )
    packed, scale = _quantize_int4_per_channel(arr)
    deq = _dequantize_int4_per_channel(packed, scale, 16).astype(np.float32)
    # Constant rows reconstruct at the closest level (with offset 7.5 the
    # codebook never has 0; closest is ±0.5 → so a constant non-zero row
    # quantises to idx=8 → 0.5·scale = magnitude/15.0 of constant value).
    assert np.abs(deq[0]).max() < 0.005
    assert np.abs(deq[1]).max() > 0.3


# ---------------------------------------------------------------------------
# 2. SNR ordering — INT8 > INT4 > INT2 on the same input
# ---------------------------------------------------------------------------


def test_int4_snr_floor_on_uniform_noise():
    """On bounded near-uniform inputs INT4 reconstruction SNR is ≥ 18 dB."""
    rng = np.random.default_rng(42)
    arr = rng.uniform(-1.0, 1.0, size=(256, 128)).astype(np.float16)
    packed, scale = _quantize_int4_per_channel(arr)
    deq = _dequantize_int4_per_channel(packed, scale, 128)
    assert _snr_db(arr, deq) >= 18.0


def test_int4_snr_strictly_between_int8_and_int2():
    """INT8 > INT4 > INT2 on any non-degenerate input."""
    rng = np.random.default_rng(123)
    H = HadamardKVCache._build_hadamard(128, np.random.default_rng(0)).astype(np.float32)
    arr = (rng.standard_normal((256, 128)).astype(np.float32) * 0.3 @ H).astype(np.float16)

    deq8 = _dequantize_int8_per_channel(*_quantize_int8_per_channel(arr))
    deq4 = _dequantize_int4_per_channel(
        *_quantize_int4_per_channel(arr), 128)
    deq2 = _dequantize_int2_per_channel(
        *_quantize_int2_per_channel(arr), 128)

    snr8, snr4, snr2 = _snr_db(arr, deq8), _snr_db(arr, deq4), _snr_db(arr, deq2)
    assert snr8 > snr4 > snr2, f"INT8={snr8:.2f} INT4={snr4:.2f} INT2={snr2:.2f}"
    # INT4 must beat INT2 by ≥ 6 dB (each extra bit ≈ +6 dB Shannon bound).
    assert snr4 - snr2 >= 6.0


def test_hadamard_rotation_helps_int4_too():
    """Rotation lifts INT4 SNR on heavy-tailed inputs (smaller margin than INT2)."""
    rng = np.random.default_rng(7)
    base = rng.standard_normal((128, 128)).astype(np.float32) * 0.1
    base[:, ::20] *= 10.0
    raw = base.astype(np.float16)

    H = HadamardKVCache._build_hadamard(128, np.random.default_rng(7)).astype(np.float32)
    rotated = (base @ H).astype(np.float16)

    snr_raw = _snr_db(raw, _dequantize_int4_per_channel(
        *_quantize_int4_per_channel(raw), 128))
    snr_rot = _snr_db(rotated, _dequantize_int4_per_channel(
        *_quantize_int4_per_channel(rotated), 128))

    # INT4 has more headroom than INT2, so a smaller lift; require ≥ 0.5 dB.
    assert snr_rot - snr_raw >= 0.5, f"raw={snr_raw:.2f} rot={snr_rot:.2f}"


# ---------------------------------------------------------------------------
# 3. Dispatch helpers
# ---------------------------------------------------------------------------


def test_kv_quantize_dispatches_int4():
    arr = np.random.randn(4, 64).astype(np.float16)
    q, s = _kv_quantize_per_channel(arr, "int4")
    assert q.dtype == np.uint8 and q.shape == (4, 32)


def test_kv_dequantize_int4_requires_head_dim():
    arr = np.random.randn(4, 64).astype(np.float16)
    q, s = _kv_quantize_per_channel(arr, "int4")
    with pytest.raises(ValueError, match="head_dim is required for INT4"):
        _kv_dequantize_per_channel(q, s, "int4")


def test_kv_dispatch_unchanged_for_int8_and_int2():
    """W104 INT2 + INT8 dispatch behaviour is preserved verbatim."""
    arr = np.random.randn(4, 64).astype(np.float16)
    q8, s8 = _kv_quantize_per_channel(arr, "int8")
    q2, s2 = _kv_quantize_per_channel(arr, "int2")
    assert q8.dtype == np.int8 and q8.shape == (4, 64)
    assert q2.dtype == np.uint8 and q2.shape == (4, 16)


# ---------------------------------------------------------------------------
# 4. KVLayerCache + QuantizedKVCache wiring
# ---------------------------------------------------------------------------


def test_kvlayer_accepts_int4():
    layer = KVLayerCache(window=4, kv_mode="int4")
    assert layer._kv_mode == "int4"


def test_kvlayer_rejects_unknown_mode_includes_int4_in_message():
    with pytest.raises(ValueError, match="int4"):
        KVLayerCache(window=4, kv_mode="int5")


def test_quantized_kvcache_mode_int4_propagates_to_layers():
    cache = QuantizedKVCache(n_layers=3, window=4, mode="int4")
    assert all(L._kv_mode == "int4" for L in cache._layers)


def test_quantized_kvcache_rejects_int4_with_svd():
    with pytest.raises(ValueError, match="svd_rank"):
        QuantizedKVCache(n_layers=2, window=4, mode="int4", svd_rank=16)


def test_quantized_kvcache_rejects_int4_with_commvq():
    with pytest.raises(ValueError, match="comm_vq_bits"):
        QuantizedKVCache(n_layers=2, window=4, mode="int4", comm_vq_bits=4)


def test_quantized_kvcache_rejects_int4_with_qfilter():
    with pytest.raises(ValueError, match="qfilter_rank"):
        QuantizedKVCache(n_layers=2, window=4, mode="int4", qfilter_rank=32)


def test_quantized_kvcache_mode_message_lists_int4():
    with pytest.raises(ValueError, match="int4"):
        QuantizedKVCache(n_layers=2, window=4, mode="bogus")


# ---------------------------------------------------------------------------
# 5. End-to-end append + get_full_kv
# ---------------------------------------------------------------------------


def test_int4_cache_appends_and_dequantizes_to_correct_shape():
    cache = QuantizedKVCache(n_layers=1, window=4, mode="int4")
    rng = np.random.default_rng(0)
    n_heads, head_dim = 8, 64
    n_tokens = 12
    for _ in range(n_tokens):
        k = rng.standard_normal((n_heads, head_dim)).astype(np.float16)
        v = rng.standard_normal((n_heads, head_dim)).astype(np.float16)
        cache.update(0, k, v)

    layer = cache._layers[0]
    full_k, full_v = layer.get_full_kv()
    assert full_k.shape == (n_heads, n_tokens, head_dim)
    assert full_v.shape == (n_heads, n_tokens, head_dim)
    assert full_k.dtype == np.float16


def test_int4_cache_storage_shape_is_packed():
    """After eviction the old-tier buffer is (n_heads, n_old, head_dim/2) uint8."""
    cache = QuantizedKVCache(n_layers=1, window=2, mode="int4")
    rng = np.random.default_rng(1)
    for _ in range(8):
        k = rng.standard_normal((4, 64)).astype(np.float16)
        v = rng.standard_normal((4, 64)).astype(np.float16)
        cache.update(0, k, v)
    layer = cache._layers[0]
    # 8 tokens − window=2 → 6 evicted to int4 storage
    assert layer.keys_old_q.shape == (4, 6, 32)        # head_dim/2 = 32
    assert layer.keys_old_q.dtype == np.uint8
    assert layer.keys_old_s.shape == (4, 6)


def test_int8_cache_storage_unchanged_after_w105():
    """INT8 storage path is untouched: dtype int8, shape (n_heads, n_old, head_dim)."""
    cache = QuantizedKVCache(n_layers=1, window=2, mode="int8")
    rng = np.random.default_rng(2)
    for _ in range(8):
        k = rng.standard_normal((4, 64)).astype(np.float16)
        v = rng.standard_normal((4, 64)).astype(np.float16)
        cache.update(0, k, v)
    layer = cache._layers[0]
    assert layer.keys_old_q.shape == (4, 6, 64)
    assert layer.keys_old_q.dtype == np.int8


def test_int2_cache_storage_unchanged_after_w105():
    """INT2 storage path (W104) unchanged: head_dim/4 packed uint8."""
    cache = QuantizedKVCache(n_layers=1, window=2, mode="int2")
    rng = np.random.default_rng(2)
    for _ in range(8):
        k = rng.standard_normal((4, 64)).astype(np.float16)
        v = rng.standard_normal((4, 64)).astype(np.float16)
        cache.update(0, k, v)
    layer = cache._layers[0]
    assert layer.keys_old_q.shape == (4, 6, 16)
    assert layer.keys_old_q.dtype == np.uint8


def test_hadamard_kvcache_int4_end_to_end():
    cache = HadamardKVCache(n_layers=1, window=2, mode="int4", seed=99)
    rng = np.random.default_rng(99)
    for _ in range(10):
        k = (rng.standard_normal((4, 64)) * 0.3).astype(np.float16)
        v = (rng.standard_normal((4, 64)) * 0.3).astype(np.float16)
        cache.update(0, k, v)
    full_k, full_v = cache._layers[0].get_full_kv()
    assert full_k.shape == (4, 10, 64)
    assert np.isfinite(full_k).all()
    assert np.isfinite(full_v).all()


def test_int4_cache_uses_about_half_of_int8():
    """At fixed token count, INT4 RAM ≈ 0.5× INT8 (ignoring fp16 recent window)."""
    rng = np.random.default_rng(3)
    n_heads, head_dim, n_tokens = 8, 128, 64

    c4 = QuantizedKVCache(n_layers=1, window=2, mode="int4")
    c8 = QuantizedKVCache(n_layers=1, window=2, mode="int8")
    for _ in range(n_tokens):
        k = rng.standard_normal((n_heads, head_dim)).astype(np.float16)
        v = rng.standard_normal((n_heads, head_dim)).astype(np.float16)
        c4.update(0, k, v)
        c8.update(0, k, v)
    bytes_int4 = c4._layers[0].memory_bytes
    bytes_int8 = c8._layers[0].memory_bytes
    # Expect ≥ 1.7× total reduction (asymptotic 2× — overhead from scales +
    # fp16 recent window pulls the ratio down at small token counts).
    assert bytes_int8 / bytes_int4 >= 1.7


# ---------------------------------------------------------------------------
# 6. Disk-tier guardrail
# ---------------------------------------------------------------------------


def test_enable_disk_tier_rejects_int4_layer(tmp_path):
    layer = KVLayerCache(window=4, kv_mode="int4")
    with pytest.raises(ValueError, match="INT8-only"):
        layer.enable_disk_tier(
            threshold=4, max_disk_tokens=16,
            cache_dir=tmp_path, n_heads=2, head_dim=64,
        )


# ---------------------------------------------------------------------------
# 7. recommended_kv_mode + 3-tier helper
# ---------------------------------------------------------------------------


def test_recommended_kv_mode_2tier_unchanged_w104():
    """W104 default API: int8 → int2 above 8 K, ignoring optional medium params."""
    assert recommended_kv_mode(0) == "int8"
    assert recommended_kv_mode(KV_INT2_AUTO_THRESHOLD) == "int8"
    assert recommended_kv_mode(KV_INT2_AUTO_THRESHOLD + 1) == "int2"


def test_recommended_kv_mode_3tier_inline_args():
    """Inline 3-tier via medium_mode + medium_threshold + explicit threshold."""
    kw = dict(medium_mode="int4", medium_threshold=8192, threshold=16384)
    assert recommended_kv_mode(4000,  **kw) == "int8"
    assert recommended_kv_mode(12000, **kw) == "int4"
    assert recommended_kv_mode(20000, **kw) == "int2"


def test_recommended_kv_mode_3tier_helper():
    assert recommended_kv_mode_3tier(0) == "int8"
    assert recommended_kv_mode_3tier(KV_INT2_AUTO_THRESHOLD) == "int8"
    assert recommended_kv_mode_3tier(KV_INT2_AUTO_THRESHOLD + 1) == "int4"
    assert recommended_kv_mode_3tier(KV_INT4_DEFAULT_THRESHOLD) == "int4"
    assert recommended_kv_mode_3tier(KV_INT4_DEFAULT_THRESHOLD + 1) == "int2"
    assert recommended_kv_mode_3tier(64_000) == "int2"


def test_recommended_kv_mode_rejects_partial_medium_args():
    """Both medium_mode + medium_threshold must be set together."""
    with pytest.raises(ValueError, match="must both be set"):
        recommended_kv_mode(8000, medium_mode="int4")
    with pytest.raises(ValueError, match="must both be set"):
        recommended_kv_mode(8000, medium_threshold=4096)


def test_recommended_kv_mode_rejects_inverted_thresholds():
    """medium_threshold must be ≤ threshold (medium sits between)."""
    with pytest.raises(ValueError, match="must be ≤|<= "):
        recommended_kv_mode(
            10_000, medium_mode="int4",
            medium_threshold=20_000, threshold=8000,
        )


def test_recommended_kv_mode_rejects_negative_w105():
    """Negative-context guard from W104 still applies."""
    with pytest.raises(ValueError, match="0"):
        recommended_kv_mode_3tier(-1)
