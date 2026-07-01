"""Unit tests for KVLayerCache prompt-prefix reuse: trim / snapshot / restore.

Pure-numpy — no MLX or loaded model. These guard the losslessness contract that
in-memory prefix reuse depends on: a trimmed-and-restored prefix must be
byte-identical to a cold prefill, and reuse must be gated off for any lossy
(quantized / evicting) cache mode.
"""

from __future__ import annotations

import numpy as np
import pytest

from squish.kv.kv_cache import KVLayerCache

H, D, W = 4, 8, 16


def _mk(mode: str = "fp16", sink: int = 0) -> KVLayerCache:
    return KVLayerCache(window=W, kv_mode=mode, sink_count=sink)


def _fill(c: KVLayerCache, n: int, base: int = 0) -> None:
    for t in range(n):
        k = np.full((H, D), base + t, dtype=np.float16)
        v = np.full((H, D), base + 100 + t, dtype=np.float16)
        c.append(k, v)


def _full(c: KVLayerCache) -> tuple[np.ndarray, np.ndarray]:
    k, v = c.get_full_kv()
    return np.array(k), np.array(v)


def test_is_trimmable_only_for_lossless_fp16():
    assert _mk("fp16").is_trimmable() is True
    assert _mk("int8").is_trimmable() is False
    assert _mk("int4").is_trimmable() is False
    # A sink pin makes the head non-trimmable (oldest tokens are special-cased).
    c = _mk("fp16", sink=2)
    _fill(c, 5)
    assert c.is_trimmable() is False


def test_trim_keeps_exact_prefix_across_window_spill():
    c = _mk()
    _fill(c, 200)  # >> window, so tokens spill to the fp16 old tier
    assert c.n_tokens == 200
    k_full, v_full = _full(c)
    trimmed = c.trim(80)
    assert trimmed == 80 and c.n_tokens == 120
    k_tr, v_tr = _full(c)
    np.testing.assert_array_equal(k_tr, k_full[:, :120, :])
    np.testing.assert_array_equal(v_tr, v_full[:, :120, :])


def test_trim_clamps_and_noops_on_nonpositive():
    c = _mk()
    _fill(c, 10)
    assert c.trim(0) == 0 and c.n_tokens == 10
    assert c.trim(-5) == 0 and c.n_tokens == 10
    assert c.trim(999) == 10 and c.n_tokens == 0


def test_snapshot_restore_round_trips_through_reset():
    c = _mk()
    _fill(c, 150)
    snap = c.snapshot()
    k0, v0 = _full(c)
    c.reset()
    assert c.n_tokens == 0
    c.restore(snap)
    assert c.n_tokens == 150
    k1, v1 = _full(c)
    np.testing.assert_array_equal(k0, k1)
    np.testing.assert_array_equal(v0, v1)


def test_restore_trim_extend_equals_cold_prefill():
    """The core losslessness contract: reusing a shared prefix then prefilling a
    new suffix yields KV byte-identical to prefilling the whole thing cold."""
    shared, tail_a, tail_b = 100, 50, 40
    # Cold reference: prefix + tail_b in one prefill.
    cold = _mk()
    _fill(cold, shared)
    _fill(cold, tail_b, base=1000)
    kc, vc = _full(cold)
    # Reuse path: a prior prompt (prefix + tail_a) snapshotted, then restored,
    # trimmed back to the shared prefix, and extended with tail_b.
    prior = _mk()
    _fill(prior, shared)
    _fill(prior, tail_a, base=500)
    snap = prior.snapshot()
    reuse = _mk()
    reuse.restore(snap)
    reuse.trim(reuse.n_tokens - shared)
    assert reuse.n_tokens == shared
    _fill(reuse, tail_b, base=1000)
    kr, vr = _full(reuse)
    np.testing.assert_array_equal(kc, kr)
    np.testing.assert_array_equal(vc, vr)


def test_snapshot_is_decoupled_from_later_mutation():
    c = _mk()
    _fill(c, 30)
    snap = c.snapshot()
    _fill(c, 20, base=900)  # mutate after snapshot
    c.reset()
    other = _mk()
    other.restore(snap)
    assert other.n_tokens == 30  # snapshot still reflects the 30-token state
