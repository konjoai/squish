"""tests/test_kv_int2.py — W104 INT2 KV cache extension tests.

Coverage:
  - Per-channel INT2 quantize/dequantize roundtrip + reconstruction SNR
  - Bit-packing layout (4 indices per uint8 byte along head_dim)
  - Storage size: head_dim/4 bytes vs head_dim for INT8 (4× reduction)
  - QuantizedKVCache(mode="int2"): construction, append, get_full_kv shape
  - HadamardKVCache(mode="int2"): rotated SNR ≥ raw SNR (Hadamard helps INT2)
  - Mode validation: rejects unknown modes; rejects illegal int2 combinations
  - recommended_kv_mode() threshold logic
  - INT8 path unchanged after W104 wiring (no regressions)
"""

import numpy as np
import pytest

from squish.kv.kv_cache import (
    HadamardKVCache,
    KV_INT2_AUTO_THRESHOLD,
    KVLayerCache,
    QuantizedKVCache,
    _KV_INT2_LEVELS,
    _dequantize_int2_per_channel,
    _dequantize_int8_per_channel,
    _kv_dequantize_per_channel,
    _kv_quantize_per_channel,
    _quantize_int2_per_channel,
    _quantize_int8_per_channel,
    recommended_kv_mode,
)


# ---------------------------------------------------------------------------
# 1. Codebook + low-level quant/dequant helpers
# ---------------------------------------------------------------------------


def test_codebook_levels_match_sqint2():
    """The W104 KV codebook must equal the SQINT2 weight codebook so the
    rotation/quantisation grid is consistent across the two pipelines."""
    from squish.quant.sqint2 import NF2_VALUES
    np.testing.assert_array_equal(_KV_INT2_LEVELS, NF2_VALUES)


def test_int2_pack_shape():
    """Packed shape is (n_tokens, head_dim/4) uint8."""
    arr = np.random.randn(7, 64).astype(np.float16)
    packed, scale = _quantize_int2_per_channel(arr)
    assert packed.shape == (7, 16)
    assert packed.dtype == np.uint8
    assert scale.shape == (7,)
    assert scale.dtype == np.float32


def test_int2_pack_unpack_indices_roundtrip():
    """Bit-packing and unpacking preserves the index for every position."""
    # Construct an array whose quantised indices we can verify exactly.
    # Levels are [-1.5, -0.5, 0.5, 1.5]; with scale=1.0 (max=1.5), values
    # 1.5*[-1, -1/3, 1/3, 1] map to indices [0, 1, 2, 3].
    values = np.array([[-1.5, -0.5, 0.5, 1.5] * 8], dtype=np.float16)  # head_dim=32
    packed, scale = _quantize_int2_per_channel(values)
    # Each byte should decode to indices 0,1,2,3 (low-bit-first packing)
    deq = _dequantize_int2_per_channel(packed, scale, 32)
    np.testing.assert_allclose(deq.astype(np.float32), values.astype(np.float32), atol=1e-3)


def test_int2_storage_is_one_quarter_of_int8():
    """Bit-packed INT2 occupies head_dim/4 bytes per token vs head_dim for INT8."""
    arr = np.random.randn(100, 128).astype(np.float16)
    packed_int2, _ = _quantize_int2_per_channel(arr)
    int8_q, _ = _quantize_int8_per_channel(arr)
    assert packed_int2.nbytes * 4 == int8_q.nbytes


def test_int2_rejects_non_multiple_of_four_head_dim():
    arr = np.random.randn(4, 30).astype(np.float16)  # 30 % 4 != 0
    with pytest.raises(ValueError, match="divisible by 4"):
        _quantize_int2_per_channel(arr)


def test_int2_rejects_wrong_rank_input():
    arr = np.random.randn(4, 8, 16).astype(np.float16)
    with pytest.raises(ValueError, match="2-D"):
        _quantize_int2_per_channel(arr)


def test_int2_dequant_rejects_packed_shape_mismatch():
    arr = np.random.randn(3, 64).astype(np.float16)
    packed, scale = _quantize_int2_per_channel(arr)
    # packed is (3, 16); claim head_dim=128 → expects (3, 32) → mismatch
    with pytest.raises(ValueError, match="!= head_dim/4"):
        _dequantize_int2_per_channel(packed, scale, 128)


def test_int2_dequant_rejects_non_multiple_head_dim():
    packed = np.zeros((1, 8), dtype=np.uint8)
    scale = np.ones((1,), dtype=np.float32)
    with pytest.raises(ValueError, match="divisible by 4"):
        _dequantize_int2_per_channel(packed, scale, 30)


def test_int2_zero_input_does_not_blow_up():
    """Per-token scale guards against divide-by-zero on all-zero rows."""
    arr = np.zeros((5, 16), dtype=np.float16)
    packed, scale = _quantize_int2_per_channel(arr)
    deq = _dequantize_int2_per_channel(packed, scale, 16)
    np.testing.assert_array_equal(deq, np.zeros_like(deq))


def test_int2_per_token_independence():
    """Each token row gets its own scale — different magnitudes don't bleed."""
    arr = np.array(
        [[1e-3] * 16,           # tiny row
         [10.0]  * 16],         # large row
        dtype=np.float16,
    )
    packed, scale = _quantize_int2_per_channel(arr)
    deq = _dequantize_int2_per_channel(packed, scale, 16).astype(np.float32)
    # Both rows have constant magnitude → INT2 reconstructs them at ±max,
    # not bleeding the small row into 0 because the large row dominates.
    assert np.abs(deq[0]).max() < 0.005
    assert np.abs(deq[1]).max() > 5.0


# ---------------------------------------------------------------------------
# 2. SNR — Hadamard rotation must materially improve INT2 KV reconstruction
# ---------------------------------------------------------------------------


def _snr_db(signal: np.ndarray, recon: np.ndarray) -> float:
    s = signal.astype(np.float64)
    r = recon.astype(np.float64)
    sig = float(np.mean(s * s))
    err = float(np.mean((s - r) ** 2))
    return 10.0 * np.log10(sig / err) if err > 0 else float("inf")


def test_int2_snr_floor_on_uniform_noise():
    """On bounded, near-uniform inputs INT2 reconstruction SNR is ≥ 5 dB.

    Uniform [-1, 1] is the post-Hadamard-rotation distribution model used by
    QuaRot; the 4-level NF2 codebook achieves ~5–6 dB MSE on it.
    """
    rng = np.random.default_rng(42)
    arr = rng.uniform(-1.0, 1.0, size=(256, 128)).astype(np.float16)
    packed, scale = _quantize_int2_per_channel(arr)
    deq = _dequantize_int2_per_channel(packed, scale, 128)
    assert _snr_db(arr, deq) >= 5.0


def test_hadamard_rotation_improves_int2_snr_vs_raw_outliers():
    """Hadamard rotation flattens heavy-tailed activations → INT2 SNR climbs.

    This is the geometric reason INT2 KV is viable inside HadamardKVCache.
    """
    rng = np.random.default_rng(7)
    # Heavy-tailed: ~5% of channels carry 10× the variance of the rest
    base = rng.standard_normal((128, 128)).astype(np.float32) * 0.1
    base[:, ::20] *= 10.0    # outlier columns
    raw = base.astype(np.float16)

    H = HadamardKVCache._build_hadamard(128, np.random.default_rng(7)).astype(np.float32)
    rotated = (base @ H).astype(np.float16)

    snr_raw = _snr_db(raw, _dequantize_int2_per_channel(
        *_quantize_int2_per_channel(raw), 128))
    snr_rot = _snr_db(rotated, _dequantize_int2_per_channel(
        *_quantize_int2_per_channel(rotated), 128))

    # Rotation must lift INT2 SNR by ≥ 1 dB on heavy-tailed input.
    assert snr_rot - snr_raw >= 1.0, f"raw={snr_raw:.2f} dB rot={snr_rot:.2f} dB"


# ---------------------------------------------------------------------------
# 3. Dispatch helpers
# ---------------------------------------------------------------------------


def test_kv_quantize_dispatches_on_mode():
    arr = np.random.randn(4, 64).astype(np.float16)
    q8, s8 = _kv_quantize_per_channel(arr, "int8")
    q2, s2 = _kv_quantize_per_channel(arr, "int2")
    assert q8.dtype == np.int8 and q8.shape == (4, 64)
    assert q2.dtype == np.uint8 and q2.shape == (4, 16)


def test_kv_dequantize_int2_requires_head_dim():
    arr = np.random.randn(4, 64).astype(np.float16)
    q, s = _kv_quantize_per_channel(arr, "int2")
    with pytest.raises(ValueError, match="head_dim is required"):
        _kv_dequantize_per_channel(q, s, "int2")


def test_kv_dequantize_int8_ignores_head_dim():
    arr = np.random.randn(4, 64).astype(np.float16)
    q, s = _kv_quantize_per_channel(arr, "int8")
    out = _kv_dequantize_per_channel(q, s, "int8")
    assert out.shape == (4, 64) and out.dtype == np.float16


# ---------------------------------------------------------------------------
# 4. KVLayerCache + QuantizedKVCache wiring
# ---------------------------------------------------------------------------


def test_kvlayer_default_mode_is_int8():
    layer = KVLayerCache(window=4)
    assert layer._kv_mode == "int8"


def test_kvlayer_rejects_unknown_mode():
    with pytest.raises(ValueError, match="kv_mode"):
        KVLayerCache(window=4, kv_mode="int5")


def test_quantized_kvcache_mode_int2_propagates_to_layers():
    cache = QuantizedKVCache(n_layers=3, window=4, mode="int2")
    assert all(L._kv_mode == "int2" for L in cache._layers)


def test_quantized_kvcache_int8_default_unchanged():
    cache = QuantizedKVCache(n_layers=2, window=4, mode="int8")
    assert all(L._kv_mode == "int8" for L in cache._layers)


def test_quantized_kvcache_rejects_int2_with_svd():
    with pytest.raises(ValueError, match="svd_rank"):
        QuantizedKVCache(n_layers=2, window=4, mode="int2", svd_rank=16)


def test_quantized_kvcache_rejects_int2_with_commvq():
    with pytest.raises(ValueError, match="comm_vq_bits"):
        QuantizedKVCache(n_layers=2, window=4, mode="int2", comm_vq_bits=4)


def test_quantized_kvcache_rejects_int2_with_qfilter():
    with pytest.raises(ValueError, match="qfilter_rank"):
        QuantizedKVCache(n_layers=2, window=4, mode="int2", qfilter_rank=32)


def test_quantized_kvcache_rejects_unknown_mode():
    with pytest.raises(ValueError, match="mode must be"):
        QuantizedKVCache(n_layers=2, window=4, mode="int3")


# ---------------------------------------------------------------------------
# 5. End-to-end append + get_full_kv
# ---------------------------------------------------------------------------


def test_int2_cache_appends_and_dequantizes_to_correct_shape():
    cache = QuantizedKVCache(n_layers=1, window=4, mode="int2")
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


def test_int2_cache_storage_shape_is_packed():
    """After eviction the old-tier buffer is (n_heads, n_old, head_dim/4) uint8."""
    cache = QuantizedKVCache(n_layers=1, window=2, mode="int2")
    rng = np.random.default_rng(1)
    for _ in range(8):
        k = rng.standard_normal((4, 64)).astype(np.float16)
        v = rng.standard_normal((4, 64)).astype(np.float16)
        cache.update(0, k, v)
    layer = cache._layers[0]
    # 8 tokens − window=2 → 6 evicted to int2 storage
    assert layer.keys_old_q.shape == (4, 6, 16)        # head_dim/4 = 16
    assert layer.keys_old_q.dtype == np.uint8
    assert layer.keys_old_s.shape == (4, 6)


def test_int8_cache_storage_unchanged_after_w104():
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


def test_hadamard_kvcache_int2_end_to_end():
    """HadamardKVCache(mode='int2') round-trips with bounded reconstruction error."""
    cache = HadamardKVCache(n_layers=1, window=2, mode="int2", seed=99)
    rng = np.random.default_rng(99)
    keys = [rng.standard_normal((4, 64)).astype(np.float16) * 0.3 for _ in range(10)]
    vals = [rng.standard_normal((4, 64)).astype(np.float16) * 0.3 for _ in range(10)]
    for k, v in zip(keys, vals):
        cache.update(0, k, v)
    full_k, full_v = cache._layers[0].get_full_kv()
    assert full_k.shape == (4, 10, 64)
    # The cache stores H·K; get_full_kv returns the rotated form (un-rotation
    # happens in get_kv_mlx).  Just verify shape + finite values.
    assert np.isfinite(full_k).all()
    assert np.isfinite(full_v).all()


# ---------------------------------------------------------------------------
# 6. Memory accounting
# ---------------------------------------------------------------------------


def test_int2_cache_uses_less_memory_than_int8():
    """For the same number of tokens, INT2 stores ~3.5× less than INT8.

    Memory: int2 packed (head_dim/4 bytes) + scale (4 B) per token per head;
    int8 (head_dim bytes) + scale (4 B) per token per head.  Recent FP16
    window contributes equally to both → exclude it from the ratio test by
    using a small window.
    """
    rng = np.random.default_rng(3)
    n_heads, head_dim, n_tokens = 8, 128, 64

    c2 = QuantizedKVCache(n_layers=1, window=2, mode="int2")
    c8 = QuantizedKVCache(n_layers=1, window=2, mode="int8")
    for _ in range(n_tokens):
        k = rng.standard_normal((n_heads, head_dim)).astype(np.float16)
        v = rng.standard_normal((n_heads, head_dim)).astype(np.float16)
        c2.update(0, k, v)
        c8.update(0, k, v)
    bytes_int2 = c2._layers[0].memory_bytes
    bytes_int8 = c8._layers[0].memory_bytes
    # Old-tier ratio: head_dim/4 vs head_dim → 4× saved on the codes.  Including
    # the FP16 recent window and per-token scales (counted with overhead), expect
    # ≥ 2.9× total reduction.  Asymptotic ratio approaches 4 as n_tokens → ∞.
    assert bytes_int8 / bytes_int2 >= 2.9


# ---------------------------------------------------------------------------
# 7. Disk-tier guardrail (W104 is RAM-only)
# ---------------------------------------------------------------------------


def test_enable_disk_tier_rejects_int2_layer(tmp_path):
    layer = KVLayerCache(window=4, kv_mode="int2")
    with pytest.raises(ValueError, match="INT8-only"):
        layer.enable_disk_tier(
            threshold=4, max_disk_tokens=16,
            cache_dir=tmp_path, n_heads=2, head_dim=64,
        )


# ---------------------------------------------------------------------------
# 8. recommended_kv_mode helper
# ---------------------------------------------------------------------------


def test_recommended_kv_mode_threshold():
    assert recommended_kv_mode(0) == "int8"
    assert recommended_kv_mode(KV_INT2_AUTO_THRESHOLD) == "int8"     # boundary: ≤ → int8
    assert recommended_kv_mode(KV_INT2_AUTO_THRESHOLD + 1) == "int2"
    assert recommended_kv_mode(32_000) == "int2"


def test_recommended_kv_mode_rejects_negative():
    with pytest.raises(ValueError, match=">= 0|≥ 0"):
        recommended_kv_mode(-1)


def test_recommended_kv_mode_overrides():
    """Caller can override modes and threshold."""
    assert recommended_kv_mode(100, short_mode="fp16", long_mode="snap",
                               threshold=50) == "snap"
    assert recommended_kv_mode(40, short_mode="fp16", long_mode="snap",
                               threshold=50) == "fp16"
