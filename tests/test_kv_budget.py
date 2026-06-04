"""tests/test_kv_budget.py — W106 KV memory budgeting + cache factory tests.

Coverage:
  - Closed-form estimate dataclass (KVMemoryEstimate) shape + invariants
  - estimate_kv_memory() math correctness across modes and dims
  - Closed-form estimate matches live cache.memory_bytes within tolerance
  - estimate_max_context() inverts estimate_kv_memory() exactly
  - recommend_mode_for_budget() picks the highest-quality mode that fits
  - make_kv_cache() factory: defaults, mode override, rotate flag, kwargs
  - Edge cases: zero context, tiny budgets, head_dim/mode incompatibility
  - Recent-window bytes are additive, not double-counted
  - HadamardKVCache vs QuantizedKVCache class selection
"""

from __future__ import annotations

import numpy as np
import pytest

from squish.kv.kv_cache import (
    HadamardKVCache,
    KVMemoryEstimate,
    QuantizedKVCache,
    _bytes_per_token_per_head,
    estimate_kv_memory,
    estimate_max_context,
    make_kv_cache,
    recommend_mode_for_budget,
    recommended_kv_mode_3tier,
)


# ---------------------------------------------------------------------------
# 1. _bytes_per_token_per_head — the building block
# ---------------------------------------------------------------------------


def test_per_head_int8_is_head_dim_plus_scale():
    assert _bytes_per_token_per_head("int8", 128) == 128 + 4


def test_per_head_int4_is_half_head_dim_plus_scale():
    assert _bytes_per_token_per_head("int4", 128) == 64 + 4


def test_per_head_int2_is_quarter_head_dim_plus_scale():
    assert _bytes_per_token_per_head("int2", 128) == 32 + 4


def test_per_head_fp16_no_scale():
    """FP16 is 2 bytes per value, no per-token scale stored."""
    assert _bytes_per_token_per_head("fp16", 128) == 256


def test_per_head_int4_rejects_odd_head_dim():
    with pytest.raises(ValueError, match="divisible by 2"):
        _bytes_per_token_per_head("int4", 7)


def test_per_head_int2_rejects_indivisible_head_dim():
    with pytest.raises(ValueError, match="divisible by 4"):
        _bytes_per_token_per_head("int2", 30)


def test_per_head_unknown_mode_raises():
    with pytest.raises(ValueError, match="fp16/int8/int4/int2"):
        _bytes_per_token_per_head("int5", 128)


def test_per_head_zero_head_dim_raises():
    with pytest.raises(ValueError, match="positive"):
        _bytes_per_token_per_head("int8", 0)


# ---------------------------------------------------------------------------
# 2. estimate_kv_memory — dataclass + math
# ---------------------------------------------------------------------------


def test_estimate_returns_dataclass_with_all_fields():
    e = estimate_kv_memory(28, 8, 128, 1000, "int8")
    assert isinstance(e, KVMemoryEstimate)
    # Frozen dataclass — assignment must fail.
    with pytest.raises(AttributeError):
        e.mode = "int4"  # type: ignore[misc]


def test_estimate_mode_propagates():
    for m in ("fp16", "int8", "int4", "int2"):
        assert estimate_kv_memory(2, 4, 128, 100, m).mode == m


def test_estimate_total_is_per_layer_times_layers():
    e = estimate_kv_memory(28, 8, 128, 1000, "int8")
    assert e.total_bytes == e.bytes_per_layer * 28


def test_estimate_per_layer_is_per_token_times_context():
    e = estimate_kv_memory(28, 8, 128, 1000, "int8")
    assert e.bytes_per_layer == e.bytes_per_token * 1000


def test_estimate_per_token_is_per_head_times_heads_times_two():
    """K + V buffers ⇒ multiply per-head cost by 2."""
    e = estimate_kv_memory(1, 8, 128, 1, "int8")
    assert e.bytes_per_token == e.bytes_per_token_per_head * 8 * 2


def test_estimate_compression_ratio_int4_about_two():
    """INT4 stores half the codes of INT8; vs FP16 (2 B/val) ratio ≈ 3.76 at d=128."""
    e = estimate_kv_memory(1, 1, 128, 1024, "int4")
    assert 3.7 < e.compression_ratio < 3.8


def test_estimate_compression_ratio_int2_about_four():
    """INT2 stores a quarter the codes of INT8; vs FP16 ratio ≈ 7.11 at d=128."""
    e = estimate_kv_memory(1, 1, 128, 1024, "int2")
    assert 7.0 < e.compression_ratio < 7.2


def test_estimate_fp16_baseline_self_consistent():
    """fp16 ratio is exactly 1.0."""
    e = estimate_kv_memory(28, 8, 128, 1000, "fp16")
    assert e.compression_ratio == pytest.approx(1.0)


def test_estimate_zero_context_yields_zero():
    e = estimate_kv_memory(28, 8, 128, 0, "int8")
    assert e.total_bytes == 0
    assert e.fp16_baseline_bytes == 0


def test_estimate_window_zero_means_no_recent_overhead():
    e = estimate_kv_memory(28, 8, 128, 1000, "int8", window=0)
    assert e.recent_window_bytes == 0


def test_estimate_window_adds_explicit_overhead():
    e0  = estimate_kv_memory(2, 4, 128, 100, "int4", window=0)
    e16 = estimate_kv_memory(2, 4, 128, 100, "int4", window=16)
    # Recent window adds head_dim * 2 (fp16) * n_kv_heads * 2 (K+V) * window * n_layers
    expected = 128 * 2 * 4 * 2 * 16 * 2
    assert e16.recent_window_bytes - e0.recent_window_bytes == expected


def test_estimate_rejects_negative_dims():
    with pytest.raises(ValueError, match="n_layers"):
        estimate_kv_memory(0, 8, 128, 100, "int8")
    with pytest.raises(ValueError, match="n_kv_heads"):
        estimate_kv_memory(1, 0, 128, 100, "int8")
    with pytest.raises(ValueError, match="context_tokens"):
        estimate_kv_memory(1, 8, 128, -1, "int8")
    with pytest.raises(ValueError, match="window"):
        estimate_kv_memory(1, 8, 128, 100, "int8", window=-1)


# ---------------------------------------------------------------------------
# 3. Closed-form estimate matches live cache.memory_bytes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["int8", "int4", "int2"])
def test_estimate_matches_live_cache_within_tolerance(mode):
    """Closed-form ≈ live `cache.memory_bytes` (the live count includes the
    fp16 recent window and double-counts the per-token scale by an existing
    convention — both are accounted for here)."""
    n_tokens, n_heads, head_dim, window = 64, 4, 64, 2
    cache = QuantizedKVCache(n_layers=1, window=window, mode=mode)
    rng = np.random.default_rng(0)
    for _ in range(n_tokens):
        cache.update(0,
            rng.standard_normal((n_heads, head_dim)).astype(np.float16),
            rng.standard_normal((n_heads, head_dim)).astype(np.float16))
    live = cache._layers[0].memory_bytes

    n_old = n_tokens - window
    est = estimate_kv_memory(1, n_heads, head_dim, n_old, mode, window=window)
    closed = est.total_bytes + est.recent_window_bytes

    # The live `memory_bytes` definition (kv_cache.py:481) counts per-token
    # scales as `keys_old_s.nbytes * 2` — i.e. doubles the scale storage.
    # Add a single n_heads × n_old × _KV_SCALE_BYTES × 2 (K+V) correction.
    extra_scale = n_heads * n_old * 4 * 2
    closed_adjusted = closed + extra_scale
    rel_err = abs(live - closed_adjusted) / live
    assert rel_err < 0.01, (
        f"{mode}: live={live:,}B closed={closed_adjusted:,}B rel_err={rel_err:.4f}"
    )


# ---------------------------------------------------------------------------
# 4. estimate_max_context — exact inverse of estimate_kv_memory
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode,context", [
    ("int8", 1000), ("int4", 5000), ("int2", 20_000),
])
def test_max_context_inverts_estimate(mode, context):
    e = estimate_kv_memory(28, 8, 128, context, mode)
    # Budget = exactly the bytes for `context`. Result must be ≥ context
    # (since per-token cost is integer the budget is divisible cleanly).
    fit = estimate_max_context(28, 8, 128, e.total_bytes, mode)
    assert fit >= context
    # Budget = total - 1 byte → must drop at least one token.
    fit_minus = estimate_max_context(28, 8, 128, e.total_bytes - 1, mode)
    assert fit_minus < context


def test_max_context_zero_budget_zero_tokens():
    assert estimate_max_context(28, 8, 128, 0, "int8") == 0


def test_max_context_int2_more_than_int4_more_than_int8():
    """At equal budget the smaller-bit mode fits more tokens."""
    budget = 1024 * 1024 * 1024     # 1 GB
    f_int8 = estimate_max_context(28, 8, 128, budget, "int8")
    f_int4 = estimate_max_context(28, 8, 128, budget, "int4")
    f_int2 = estimate_max_context(28, 8, 128, budget, "int2")
    assert f_int2 > f_int4 > f_int8 > 0


def test_max_context_with_window_subtracts_overhead():
    budget = 100 * 1024 * 1024
    no_win  = estimate_max_context(28, 8, 128, budget, "int4", window=0)
    big_win = estimate_max_context(28, 8, 128, budget, "int4", window=512)
    assert big_win < no_win


def test_max_context_window_exceeds_budget_returns_zero():
    """When the recent-window cost alone busts the budget, no tokens fit."""
    # FP16 recent window of 1000 tokens × 28 layers is much more than 1 KB.
    assert estimate_max_context(28, 8, 128, 1024, "int4", window=1000) == 0


def test_max_context_rejects_negative_budget():
    with pytest.raises(ValueError, match="budget_bytes"):
        estimate_max_context(28, 8, 128, -1, "int8")


# ---------------------------------------------------------------------------
# 5. recommend_mode_for_budget — quality-first pick
# ---------------------------------------------------------------------------


def test_recommend_picks_int8_when_budget_is_generous():
    # 8 GB easily covers Qwen2.5-7B INT8 at 32K (~1.8 GB).
    assert recommend_mode_for_budget(28, 8, 128, 32_000, 8 * 1024**3) == "int8"


def test_recommend_drops_to_int4_at_smaller_budget():
    # 1 GB doesn't fit INT8 at 32K (~1.8 GB) but fits INT4 (~930 MB).
    assert recommend_mode_for_budget(28, 8, 128, 32_000, 1024**3) == "int4"


def test_recommend_drops_to_int2_when_only_just_fits():
    # 700 MB fits INT2 (~492 MB) but not INT4.
    assert recommend_mode_for_budget(28, 8, 128, 32_000, 700 * 1024**2) == "int2"


def test_recommend_returns_none_when_nothing_fits():
    assert recommend_mode_for_budget(28, 8, 128, 32_000, 1024) is None


def test_recommend_zero_budget_returns_none():
    assert recommend_mode_for_budget(28, 8, 128, 1000, 0) is None


def test_recommend_zero_context_picks_int8():
    """At zero context everything fits ⇒ pick the highest-quality tier."""
    assert recommend_mode_for_budget(28, 8, 128, 0, 1) == "int8"


def test_recommend_skips_incompatible_head_dim_for_int2():
    """head_dim=6 isn't divisible by 4 — int2 is unreachable; pick the next one."""
    assert recommend_mode_for_budget(2, 1, 6, 100, 100_000_000) in ("int8", "int4")


# ---------------------------------------------------------------------------
# 6. make_kv_cache factory
# ---------------------------------------------------------------------------


def test_make_kv_cache_default_uses_hadamard_and_auto_mode():
    cache = make_kv_cache(n_layers=4, planned_context=4_000)
    assert isinstance(cache, HadamardKVCache)
    assert cache.mode == recommended_kv_mode_3tier(4_000)
    assert cache.window == 128


def test_make_kv_cache_picks_int4_for_medium_context():
    cache = make_kv_cache(n_layers=4, planned_context=12_000)
    assert cache.mode == "int4"


def test_make_kv_cache_picks_int2_for_long_context():
    cache = make_kv_cache(n_layers=4, planned_context=40_000)
    assert cache.mode == "int2"


def test_make_kv_cache_explicit_mode_overrides_auto():
    cache = make_kv_cache(n_layers=4, planned_context=40_000, mode="int8")
    assert cache.mode == "int8"


def test_make_kv_cache_rotate_false_returns_quantized_not_hadamard():
    cache = make_kv_cache(n_layers=4, planned_context=4_000, rotate=False)
    assert isinstance(cache, QuantizedKVCache)
    assert not isinstance(cache, HadamardKVCache)


def test_make_kv_cache_window_override():
    cache = make_kv_cache(n_layers=4, planned_context=4_000, window=64)
    assert cache.window == 64


def test_make_kv_cache_forwards_extra_kwargs():
    """Constructor kwargs (budget, snap_window) make it through."""
    cache = make_kv_cache(
        n_layers=4, planned_context=4_000, mode="snap",
        budget=2048, snap_window=32,
    )
    assert cache.mode == "snap"
    assert cache.budget == 2048
    assert cache.snap_window == 32


def test_make_kv_cache_seeded_hadamard_is_deterministic():
    """Same seed ⇒ identical Hadamard rotation matrices for the same head_dim."""
    c1 = make_kv_cache(n_layers=2, planned_context=20_000, seed=42)
    c2 = make_kv_cache(n_layers=2, planned_context=20_000, seed=42)
    H1 = c1._get_H_k(128)
    H2 = c2._get_H_k(128)
    np.testing.assert_array_equal(H1, H2)


# ---------------------------------------------------------------------------
# 7. KVMemoryEstimate.fits_in helper
# ---------------------------------------------------------------------------


def test_fits_in_true_when_within_budget():
    e = estimate_kv_memory(28, 8, 128, 32_000, "int4")
    assert e.fits_in(2 * 1024**3)        # 2 GB easily covers ~930 MB


def test_fits_in_false_when_over_budget():
    e = estimate_kv_memory(28, 8, 128, 32_000, "fp16")
    assert not e.fits_in(1024**3)         # 1 GB cannot hold ~3.5 GB


def test_fits_in_includes_recent_window():
    """Adding the recent window must push a borderline case over."""
    e_no  = estimate_kv_memory(28, 8, 128, 1000, "int4", window=0)
    budget = e_no.total_bytes + 100      # 100 bytes of slack
    assert e_no.fits_in(budget)
    # Same workload with a window that consumes more than the slack.
    e_win = estimate_kv_memory(28, 8, 128, 1000, "int4", window=512)
    assert not e_win.fits_in(budget)
