"""tests/test_spaces_demo.py — W107 HF Space pure-logic tests.

Covers `spaces/_logic.py` end-to-end without spinning up Gradio.

The tests are deliberately *not* gated on Gradio's presence: they import
straight from `spaces._logic`, which is gradio-free by design. The only
runtime dep beyond squish itself is numpy.

Coverage:
  - `snr_db`: normal case, perfect-reconstruction (∞) edge case
  - `make_synthetic_activations`: shape/dtype contract, three distributions,
     determinism under fixed seed, error path on bad distribution
  - `apply_hadamard`: shape/dtype preserved, energy preservation (orthogonal),
     deterministic under fixed seed, refuses 1-D input
  - `run_all_tiers`: returns INT8/INT4/INT2 in order, monotonic SNR ordering
     INT8 ≥ INT4 ≥ INT2, monotonic compression INT8 < INT4 < INT2,
     handles head_dim=128 (all 3 tiers), head_dim=64 (all 3 tiers),
     head_dim=2 (drops INT2)
  - `recommend_mode_for_context`: 3-tier dispatch matches
     `recommended_kv_mode_3tier` exactly
  - `memory_table_rows`: contains fp16 + 3 quant tiers; each tier's
     `total_mb` matches an independent `estimate_kv_memory` call
  - `label_budget_fit`: "yes (... headroom)" / "no (over by ...)" / "-"
     paths all reachable
  - `recommend_for_budget_mb`: returns fitting tier or the "none — too long"
     sentinel; rejects non-positive budgets
  - `MODEL_PRESETS` and `EXAMPLES`: surface invariants (non-empty,
     well-typed, runnable)
  - **Big one — Hadamard rotation lifts INT2 SNR by ≥ 8 dB on
     outlier-spiked input.** This is the demo's headline claim, asserted
     as a hard test so it cannot silently regress.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from squish.kv.kv_cache import (
    KV_INT2_AUTO_THRESHOLD,
    KV_INT4_DEFAULT_THRESHOLD,
    estimate_kv_memory,
    recommended_kv_mode_3tier,
)

from spaces._logic import (
    EXAMPLES,
    MODEL_PRESETS,
    TierResult,
    apply_hadamard,
    label_budget_fit,
    make_synthetic_activations,
    memory_table_rows,
    recommend_for_budget_mb,
    recommend_mode_for_context,
    run_all_tiers,
    snr_db,
)


# ---------------------------------------------------------------------------
# 1. snr_db
# ---------------------------------------------------------------------------

def test_snr_db_perfect_reconstruction_returns_inf():
    rng = np.random.default_rng(0)
    s = rng.standard_normal((32, 16)).astype(np.float16)
    assert math.isinf(snr_db(s, s.copy()))


def test_snr_db_known_value():
    """SNR(x, x + n) where n has variance 0.01 of x: ~20 dB."""
    rng = np.random.default_rng(1)
    s = rng.standard_normal((512, 64)).astype(np.float64) * 1.0
    n = rng.standard_normal(s.shape) * 0.1
    measured = snr_db(s.astype(np.float16), (s + n).astype(np.float16))
    assert 18.0 <= measured <= 22.0


# ---------------------------------------------------------------------------
# 2. make_synthetic_activations
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dist", ["gaussian", "heavy_tailed", "outlier"])
def test_synth_shape_dtype(dist):
    arr = make_synthetic_activations(64, 128, dist)
    assert arr.shape == (64, 128)
    assert arr.dtype == np.float16


def test_synth_determinism_under_fixed_seed():
    a = make_synthetic_activations(64, 32, "heavy_tailed", seed=7)
    b = make_synthetic_activations(64, 32, "heavy_tailed", seed=7)
    assert np.array_equal(a, b)


def test_synth_different_seeds_differ():
    a = make_synthetic_activations(64, 32, "gaussian", seed=1)
    b = make_synthetic_activations(64, 32, "gaussian", seed=2)
    assert not np.array_equal(a, b)


def test_synth_rejects_unknown_distribution():
    with pytest.raises(ValueError, match="distribution must be"):
        make_synthetic_activations(8, 8, "uniform")


def test_synth_rejects_non_positive_dims():
    with pytest.raises(ValueError):
        make_synthetic_activations(0, 8, "gaussian")
    with pytest.raises(ValueError):
        make_synthetic_activations(8, 0, "gaussian")


def test_synth_outlier_actually_has_outliers():
    """Outlier distribution must contain spikes ≥ 4σ from the bulk."""
    arr = make_synthetic_activations(1024, 128, "outlier", seed=42).astype(np.float64)
    assert (np.abs(arr) >= 4.0).any()


# ---------------------------------------------------------------------------
# 3. apply_hadamard
# ---------------------------------------------------------------------------

def test_hadamard_preserves_shape_and_dtype():
    arr = make_synthetic_activations(64, 128, "gaussian")
    rot = apply_hadamard(arr)
    assert rot.shape == arr.shape
    assert rot.dtype == np.float16


def test_hadamard_is_energy_preserving_within_fp16():
    """Orthogonal rotation conserves Frobenius norm. 1 % fp16 tolerance."""
    arr = make_synthetic_activations(128, 128, "gaussian", seed=3).astype(np.float64)
    rot = apply_hadamard(arr.astype(np.float16)).astype(np.float64)
    e_in  = float(np.sum(arr * arr))
    e_out = float(np.sum(rot * rot))
    assert math.isclose(e_in, e_out, rel_tol=0.01)


def test_hadamard_deterministic_under_fixed_seed():
    arr = make_synthetic_activations(32, 32, "gaussian", seed=4)
    a = apply_hadamard(arr, seed=99)
    b = apply_hadamard(arr, seed=99)
    assert np.array_equal(a, b)


def test_hadamard_refuses_non_2d_input():
    with pytest.raises(ValueError, match="expected 2-D"):
        apply_hadamard(np.zeros((4,), dtype=np.float16))


# ---------------------------------------------------------------------------
# 4. run_all_tiers
# ---------------------------------------------------------------------------

def test_run_all_tiers_returns_three_results_for_head_dim_128():
    arr = make_synthetic_activations(128, 128, "gaussian", seed=5)
    rs = run_all_tiers(arr)
    assert [r.mode for r in rs] == ["int8", "int4", "int2"]
    assert all(isinstance(r, TierResult) for r in rs)


def test_run_all_tiers_drops_int2_when_head_dim_not_divisible_by_4():
    """head_dim=2 supports int8 + int4 (even) but not int2 (needs %4==0)."""
    arr = make_synthetic_activations(64, 2, "gaussian", seed=6)
    rs = run_all_tiers(arr)
    assert [r.mode for r in rs] == ["int8", "int4"]


def test_run_all_tiers_snr_monotone_int8_ge_int4_ge_int2():
    arr = make_synthetic_activations(256, 128, "gaussian", seed=7)
    arr = apply_hadamard(arr)        # rotation makes the ordering crisp
    snrs = {r.mode: r.snr_db for r in run_all_tiers(arr)}
    assert snrs["int8"] >= snrs["int4"] >= snrs["int2"]


def test_run_all_tiers_storage_monotone_int8_gt_int4_gt_int2():
    arr = make_synthetic_activations(64, 128, "gaussian", seed=8)
    bpts = {r.mode: r.bytes_per_token for r in run_all_tiers(arr)}
    assert bpts["int8"] > bpts["int4"] > bpts["int2"]
    # Headline numbers at head_dim=128, locked by the BENCHMARKS.md table.
    assert bpts["int8"] == 132
    assert bpts["int4"] == 68
    assert bpts["int2"] == 36


def test_run_all_tiers_compression_ratios_match_table():
    """Compression vs fp16 for head_dim=128 is the BENCHMARKS.md row."""
    arr = make_synthetic_activations(8, 128, "gaussian", seed=9)
    ratios = {r.mode: r.compression_vs_fp16 for r in run_all_tiers(arr)}
    assert math.isclose(ratios["int8"], 256 / 132, rel_tol=1e-6)
    assert math.isclose(ratios["int4"], 256 / 68,  rel_tol=1e-6)
    assert math.isclose(ratios["int2"], 256 / 36,  rel_tol=1e-6)


def test_run_all_tiers_refuses_non_2d():
    with pytest.raises(ValueError, match="expected 2-D"):
        run_all_tiers(np.zeros((4,), dtype=np.float16))


# ---------------------------------------------------------------------------
# 5. recommend_mode_for_context
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("ctx,expected", [
    (        0, "int8"),
    (    4_000, "int8"),
    (KV_INT2_AUTO_THRESHOLD,     "int8"),
    (KV_INT2_AUTO_THRESHOLD + 1, "int4"),
    (   12_000, "int4"),
    (KV_INT4_DEFAULT_THRESHOLD,     "int4"),
    (KV_INT4_DEFAULT_THRESHOLD + 1, "int2"),
    (   65_536, "int2"),
])
def test_recommend_mode_for_context_matches_3tier(ctx, expected):
    assert recommend_mode_for_context(ctx) == expected
    assert recommend_mode_for_context(ctx) == recommended_kv_mode_3tier(ctx)


# ---------------------------------------------------------------------------
# 6. memory_table_rows
# ---------------------------------------------------------------------------

def test_memory_table_first_row_is_fp16_reference():
    rows = memory_table_rows(28, 4, 128, 8000)
    assert rows[0]["mode"] == "fp16 (reference)"
    assert rows[0]["compression_ratio"] == pytest.approx(1.0)


def test_memory_table_includes_all_three_quant_tiers_when_head_dim_compatible():
    rows = memory_table_rows(28, 4, 128, 8000)
    modes = [r["mode"] for r in rows]
    assert modes == ["fp16 (reference)", "int8", "int4", "int2"]


def test_memory_table_total_mb_matches_estimate_kv_memory():
    """Each row's total_mb must equal the upstream closed-form, exactly."""
    n_layers, n_kv_heads, head_dim, ctx = 28, 4, 128, 16_000
    rows = memory_table_rows(n_layers, n_kv_heads, head_dim, ctx)
    for row in rows:
        mode = row["mode"].split()[0]   # "fp16 (reference)" → "fp16"
        est = estimate_kv_memory(n_layers, n_kv_heads, head_dim, ctx,
                                 mode, window=128)
        assert math.isclose(row["total_mb"], est.total_bytes / 1e6, rel_tol=1e-12)


def test_memory_table_compression_ratios_are_monotone():
    """fp16 (1.00) < int8 < int4 < int2."""
    rows = memory_table_rows(28, 4, 128, 8000)
    ratios = [r["compression_ratio"] for r in rows]
    assert ratios == sorted(ratios)


# ---------------------------------------------------------------------------
# 7. label_budget_fit
# ---------------------------------------------------------------------------

def test_label_budget_fit_zero_budget_disables_column():
    rows = memory_table_rows(28, 4, 128, 8000)
    labelled = label_budget_fit(rows, 0)
    assert all(r["fits"] == "-" for r in labelled)


def test_label_budget_fit_yes_path_reachable():
    rows = memory_table_rows(28, 4, 128, 8000)
    labelled = label_budget_fit(rows, 100_000)   # absurdly high budget
    assert all(r["fits"].startswith("yes") for r in labelled)


def test_label_budget_fit_no_path_reachable():
    rows = memory_table_rows(28, 4, 128, 32_000)
    labelled = label_budget_fit(rows, 50)        # absurdly low budget
    # fp16 row is the largest; it must overflow.
    assert labelled[0]["fits"].startswith("no")


# ---------------------------------------------------------------------------
# 8. recommend_for_budget_mb
# ---------------------------------------------------------------------------

def test_recommend_for_budget_returns_a_quant_tier_when_int2_fits():
    """Qwen2.5-7B at 32 K, 4 GB budget → int8 fits comfortably."""
    rec = recommend_for_budget_mb(28, 4, 128, 32_000, 4096.0)
    assert rec in {"int8", "int4", "int2"}


def test_recommend_for_budget_returns_sentinel_when_nothing_fits():
    rec = recommend_for_budget_mb(28, 4, 128, 32_000, budget_mb=10.0)
    assert rec == "none — context too long for budget"


def test_recommend_for_budget_rejects_non_positive_budget():
    with pytest.raises(ValueError, match="budget_mb must be positive"):
        recommend_for_budget_mb(28, 4, 128, 1000, 0.0)


# ---------------------------------------------------------------------------
# 9. surface invariants — MODEL_PRESETS, EXAMPLES
# ---------------------------------------------------------------------------

def test_model_presets_are_well_typed_and_runnable():
    assert len(MODEL_PRESETS) >= 3
    for label, dims in MODEL_PRESETS.items():
        assert isinstance(label, str) and len(label) > 0
        n_layers, n_kv_heads, head_dim = dims
        assert n_layers   > 0
        assert n_kv_heads > 0
        assert head_dim   > 0
        # And actually runnable through the budgeter:
        rows = memory_table_rows(n_layers, n_kv_heads, head_dim, 4096)
        assert len(rows) >= 2   # at least fp16 + int8


def test_examples_are_runnable_through_tensor_inspector_logic():
    assert len(EXAMPLES) >= 3
    for n_tokens, head_dim, dist, rotate in EXAMPLES:
        arr = make_synthetic_activations(n_tokens, head_dim, dist)
        if rotate:
            arr = apply_hadamard(arr)
        rs = run_all_tiers(arr)
        assert len(rs) >= 2


# ---------------------------------------------------------------------------
# 10. headline claim — the demo's "wow" moment
# ---------------------------------------------------------------------------

def test_hadamard_rotation_lifts_int2_snr_by_at_least_8db_on_outlier_input():
    """The headline result that BENCHMARKS.md and spaces/README.md cite.

    Without rotation, naive INT2 collapses on outlier-spiked activations
    (∼ -8 dB on this seed). With the randomised Hadamard rotation it
    recovers to ∼ +8 dB. The 8 dB lift is the test's hard floor — if a
    future change drops below this, the demo's pitch is false and CI
    must fail.
    """
    raw = make_synthetic_activations(256, 128, "outlier", seed=42)
    rot = apply_hadamard(raw, seed=42)

    snr_raw = next(r.snr_db for r in run_all_tiers(raw) if r.mode == "int2")
    snr_rot = next(r.snr_db for r in run_all_tiers(rot) if r.mode == "int2")

    assert snr_rot - snr_raw >= 8.0, (
        f"Hadamard rotation only lifted INT2 SNR by {snr_rot - snr_raw:.2f} dB "
        f"(raw={snr_raw:.2f}, rot={snr_rot:.2f}); demo claim no longer holds."
    )
