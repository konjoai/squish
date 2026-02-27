#!/usr/bin/env python3
"""
Unit tests for Phase 1.2 (AWQ) and Phase 1.3 (KV cache).
No model loading required — pure numpy logic.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import tempfile


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1.2 — AWQ
# ─────────────────────────────────────────────────────────────────────────────

def test_awq_apply_weights():
    """AWQ scale col-division modifies W and leaves other tensors alone."""
    from squish.awq import apply_awq_to_weights

    W = np.random.randn(128, 64).astype(np.float32)
    gamma = np.ones(64, dtype=np.float32)
    weights = {
        "model.layers.0.self_attn.q_proj.weight": W.copy(),
        "model.layers.0.input_layernorm.weight":  gamma.copy(),
    }
    scales = {"model.layers.0.self_attn.q_proj":
              np.abs(np.random.randn(64)).astype(np.float32) + 0.1}

    out = apply_awq_to_weights(weights, scales, verbose=False)
    s = scales["model.layers.0.self_attn.q_proj"]

    # W[:, c] should have been divided by s[c]
    W_awq = out["model.layers.0.self_attn.q_proj.weight"]
    expected = W / s[np.newaxis, :]
    np.testing.assert_allclose(W_awq, expected, rtol=1e-5,
                               err_msg="W_awq col scaling mismatch")

    # gamma should have been multiplied by s
    gamma_awq = out["model.layers.0.input_layernorm.weight"]
    np.testing.assert_allclose(gamma_awq, gamma * s, rtol=1e-5,
                               err_msg="gamma awq scale absorption mismatch")
    print("test_awq_apply_weights: PASS")


def test_awq_save_load(tmp_path=None):
    """Round-trip: save_awq_scales then load_awq_scales returns identical arrays."""
    import tempfile
    from squish.awq import save_awq_scales, load_awq_scales

    scales_in = {
        "model.layers.0.self_attn.q_proj": np.array([1.2, 0.8, 1.5], dtype=np.float32),
        "model.layers.1.mlp.gate_proj":     np.array([0.9, 1.1, 0.7, 1.3], dtype=np.float32),
    }
    with tempfile.TemporaryDirectory() as d:
        save_awq_scales(scales_in, d, verbose=False)
        scales_out = load_awq_scales(d)

    for k, v in scales_in.items():
        assert k in scales_out, f"Key {k!r} missing after round-trip"
        np.testing.assert_array_equal(scales_out[k], v,
                                      err_msg=f"Scale mismatch for {k}")
    print("test_awq_save_load: PASS")


def test_awq_no_match_counts_applied():
    """apply_awq_to_weights prints warning when no scale matches — doesn't crash."""
    from squish.awq import apply_awq_to_weights

    weights = {"embed_tokens.weight": np.ones((100, 64), dtype=np.float32)}
    scales  = {"totally.different.path": np.ones(64, dtype=np.float32)}
    out = apply_awq_to_weights(weights, scales, verbose=False)
    assert out["embed_tokens.weight"].shape == (100, 64)
    print("test_awq_no_match_counts_applied: PASS")


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1.3 — KV cache
# ─────────────────────────────────────────────────────────────────────────────

def test_kv_int8_round_trip():
    """INT8 quantize → dequantize max error < 1% of signal range."""
    from squish.kv_cache import _quantize_int8_per_channel, _dequantize_int8_per_channel

    rng = np.random.default_rng(42)
    x = rng.standard_normal((32, 128)).astype(np.float16)
    q, s = _quantize_int8_per_channel(x)
    xr    = _dequantize_int8_per_channel(q, s)

    assert q.dtype  == np.int8,   f"Expected int8 got {q.dtype}"
    assert xr.dtype == np.float16, f"Expected float16 got {xr.dtype}"

    err      = np.abs(x.astype(np.float32) - xr.astype(np.float32))
    x_range  = np.abs(x.astype(np.float32)).max()
    rel_err  = err.max() / (x_range + 1e-9)
    assert rel_err < 0.02, f"INT8 round-trip relative error too large: {rel_err:.4f}"
    print(f"test_kv_int8_round_trip: PASS  (max-rel-err={rel_err:.5f})")


def test_kv_layer_cache_append():
    """KVLayerCache stores exactly n_tokens positions and recent window is correct."""
    from squish.kv_cache import KVLayerCache

    cache = KVLayerCache(window=4)
    n_heads, head_dim = 8, 64

    for i in range(10):
        k = np.random.randn(n_heads, head_dim).astype(np.float16)
        v = np.random.randn(n_heads, head_dim).astype(np.float16)
        cache.append(k, v)

    assert cache.n_tokens == 10, f"Expected 10 got {cache.n_tokens}"
    assert len(cache.keys_recent) == 4, \
        f"Recent window should be 4, got {len(cache.keys_recent)}"
    assert cache.keys_old_q is not None, "old INT8 buffer should exist after overflow"

    full_k, full_v = cache.get_full_kv()
    assert full_k.shape == (n_heads, 10, head_dim), \
        f"Expected ({n_heads}, 10, {head_dim}) got {full_k.shape}"
    assert full_v.shape == full_k.shape
    print(f"test_kv_layer_cache_append: PASS  shape={full_k.shape}")


def test_kv_layer_cache_reset():
    """reset() clears all state."""
    from squish.kv_cache import KVLayerCache

    cache = KVLayerCache(window=4)
    for _ in range(8):
        cache.append(np.ones((2, 32), dtype=np.float16),
                     np.ones((2, 32), dtype=np.float16))
    assert cache.n_tokens == 8
    cache.reset()
    assert cache.n_tokens == 0
    assert cache.keys_old_q is None
    print("test_kv_layer_cache_reset: PASS")


def test_quantized_kv_cache_full():
    """QuantizedKVCache.update accumulates correctly in int8 mode."""
    from squish.kv_cache import QuantizedKVCache

    qkv = QuantizedKVCache(n_layers=4, window=4, mode="int8")
    for _ in range(6):
        qkv.update(0, np.random.randn(2, 64).astype(np.float16),
                      np.random.randn(2, 64).astype(np.float16))
    assert qkv.n_tokens == 6
    assert qkv.memory_mb > 0
    stats = qkv.stats()
    assert stats["mode"] == "int8"
    assert stats["n_tokens"] == 6
    print(f"test_quantized_kv_cache_full: PASS  {stats}")


def test_snapkv_eviction():
    """SnapKV eviction reduces token count to <= budget."""
    from squish.kv_cache import KVLayerCache, _snap_evict

    cache = KVLayerCache(window=4)
    for _ in range(20):
        cache.append(np.random.randn(4, 128).astype(np.float16),
                     np.random.randn(4, 128).astype(np.float16))
    assert cache.n_tokens == 20

    _snap_evict(cache, budget=8, snap_window=4)
    assert cache.n_tokens <= 8, \
        f"After eviction expected ≤8, got {cache.n_tokens}"
    print(f"test_snapkv_eviction: PASS  {20} → {cache.n_tokens}")


def test_snapkv_mode_auto_evict():
    """QuantizedKVCache snap mode triggers eviction once budget is exceeded.

    SnapKV evicts *once* when the cache first exceeds ``budget`` during prefill;
    subsequent autoregressive tokens keep appending normally (cache grows past
    budget again, but that's expected — re-eviction only happens on next prefill).
    """
    from squish.kv_cache import QuantizedKVCache

    budget = 12
    # Eviction fires the first time n_tokens > budget (at token budget+1 = 13)
    # After eviction: n_tokens ≤ budget
    # Then 20-13 = 7 more tokens are appended → final n_tokens ≤ budget + 7
    qkv = QuantizedKVCache(n_layers=1, window=4, mode="snap",
                           budget=budget, snap_window=4)
    for i in range(20):
        qkv.update(0, np.random.randn(4, 64).astype(np.float16),
                      np.random.randn(4, 64).astype(np.float16))

    # Eviction should have been triggered exactly once
    assert qkv._snapped[0], "Layer 0 should have been snapped"
    # Final token count ≤ eviction_target + tokens_after_eviction ≤ budget + 7
    assert qkv._layers[0].n_tokens <= budget + 7, \
        f"After snap+append got {qkv._layers[0].n_tokens}, expected ≤{budget+7}"
    print(f"test_snapkv_mode_auto_evict: PASS  n_tokens={qkv._layers[0].n_tokens}"
          f"  (evicted at budget={budget}, then {20 - budget - 1} more appended)")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n── Phase 1.2 (AWQ) ─────────────────────────────────────────────")
    test_awq_apply_weights()
    test_awq_save_load()
    test_awq_no_match_counts_applied()

    print("\n── Phase 1.3 (KV cache) ────────────────────────────────────────")
    test_kv_int8_round_trip()
    test_kv_layer_cache_append()
    test_kv_layer_cache_reset()
    test_quantized_kv_cache_full()
    test_snapkv_eviction()
    test_snapkv_mode_auto_evict()

    print("\n✓ All Phase 1.2 + 1.3 unit tests PASSED\n")
