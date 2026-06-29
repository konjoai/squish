"""Unit tests for the reuse x context benchmark-matrix harness (pure-logic core).

Covers the platform-independent modules: corpus construction, paired statistics,
cache-hit measurement/classification, memory/OOM classification, thermal math,
matrix spec, reporting, host parsing, and cell-level pure helpers. The bench-host
modules (systems/cell orchestration) are exercised on Apple Silicon, not here.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from benchmarks.ollama_vs_squish.matrix import (
    cache_probe,
    corpus,
    host,
    matrix_spec,
    memory,
    report,
    stats_ext,
    thermal,
)
from benchmarks.ollama_vs_squish.matrix.cell import _paired, inter_token_stats


# ── mock tokenizer ────────────────────────────────────────────────────────────


class MockTokenizer:
    """Whitespace tokenizer with a stable, reversible (in count) round-trip."""

    def __init__(self) -> None:
        self.vocab: dict[str, int] = {}
        self.inv: dict[int, str] = {}

    def encode(self, text: str) -> list[int]:
        ids = []
        for w in text.split():
            if w not in self.vocab:
                i = len(self.vocab) + 1
                self.vocab[w] = i
                self.inv[i] = w
            ids.append(self.vocab[w])
        return ids

    def decode(self, ids: list[int]) -> str:
        return " ".join(self.inv.get(i, "?") for i in ids)


# ── corpus ────────────────────────────────────────────────────────────────────


def _corpus() -> corpus.Corpus:
    return corpus.Corpus(MockTokenizer(), base_seed=42)


@pytest.mark.parametrize("reuse", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_prompt_hits_target_token_length(reuse):
    c = _corpus()
    spec = c.build_prompt(reuse, ctx_tokens=300, run_index=0)
    # exact-token slicing should land on target (mock tokenizer is exact)
    assert abs(spec.measured_tokens - 300) <= 2
    assert spec.reuse == reuse
    assert spec.shared_prefix_tokens == round(reuse * 300)


def test_zero_reuse_is_unique_per_run():
    c = _corpus()
    a = c.build_prompt(0.0, 300, 0)
    b = c.build_prompt(0.0, 300, 1)
    assert a.sha256 != b.sha256


def test_exact_reuse_is_identical_across_runs():
    c = _corpus()
    a = c.build_prompt(1.0, 300, 0)
    b = c.build_prompt(1.0, 300, 7)
    assert a.sha256 == b.sha256


def test_partial_reuse_shares_prefix_varies_tail():
    c = _corpus()
    tok = c.tok
    a = c.build_prompt(0.5, 400, 0)
    b = c.build_prompt(0.5, 400, 1)
    shared_n = round(0.5 * 400)
    ia, ib = tok.encode(a.text), tok.encode(b.text)
    assert ia[:shared_n] == ib[:shared_n]  # shared prefix identical
    assert a.sha256 != b.sha256  # tail differs


def test_determinism_same_coords_same_prompt():
    a = _corpus().build_prompt(0.5, 300, 3)
    b = _corpus().build_prompt(0.5, 300, 3)
    assert a.sha256 == b.sha256 and a.seed == b.seed


def test_expected_hit_fraction_matches_reuse():
    assert corpus.expected_hit_fraction(0.0) == 0.0
    assert corpus.expected_hit_fraction(0.5) == 0.5
    assert corpus.expected_hit_fraction(1.0) == 1.0


def test_save_cell_prompts_writes_files_and_manifest(tmp_path: Path):
    c = _corpus()
    prompts = [c.build_prompt(0.5, 200, i) for i in range(3)]
    manifest = corpus.save_cell_prompts(tmp_path, "r050_c200", prompts)
    assert manifest.exists()
    assert (tmp_path / "prompts" / "r050_c200" / "run_000.txt").exists()
    import json

    data = json.loads(manifest.read_text())
    assert data["n_prompts"] == 3 and len(data["runs"]) == 3
    assert "text" not in data["runs"][0]  # compact manifest


def test_corpus_uses_real_files_when_present(tmp_path: Path):
    (tmp_path / "doc1.txt").write_text("alpha beta gamma delta " * 50)
    c = corpus.Corpus(MockTokenizer(), corpus_dir=tmp_path)
    spec = c.build_prompt(0.0, 100, 0)
    assert spec.measured_tokens >= 95


def test_reuse_out_of_range_raises():
    with pytest.raises(ValueError):
        _corpus().build_prompt(1.5, 300, 0)


# ── stats_ext ─────────────────────────────────────────────────────────────────


def test_wilcoxon_exact_small_all_positive():
    a = [2, 4, 6, 8, 10, 12]
    b = [1, 2, 3, 4, 5, 6]
    res = stats_ext.wilcoxon_signed_rank(a, b)
    assert res.method == "exact"
    assert res.w_minus == 0 and res.w_plus == 21
    assert math.isclose(res.p_value, 2 / 64, rel_tol=1e-9)


def test_wilcoxon_normal_approx_large_n():
    a = list(range(1, 41))
    b = [x - 1 for x in a]  # constant +1 difference, ties present
    res = stats_ext.wilcoxon_signed_rank(a, b)
    assert res.method == "normal_approx"
    assert res.p_value is not None and res.p_value < 0.05


def test_wilcoxon_all_zero_is_degenerate():
    res = stats_ext.wilcoxon_signed_rank([1, 2, 3], [1, 2, 3])
    assert res.method == "degenerate" and res.n_effective == 0


def test_wilcoxon_length_mismatch_raises():
    with pytest.raises(ValueError):
        stats_ext.wilcoxon_signed_rank([1, 2], [1])


def test_cliffs_delta_full_dominance():
    assert stats_ext.cliffs_delta([5, 6, 7], [1, 2, 3])["delta"] == 1.0
    assert stats_ext.cliffs_delta([1, 2, 3], [5, 6, 7])["delta"] == -1.0
    mid = stats_ext.cliffs_delta([1, 2, 3], [1, 2, 3])["delta"]
    assert mid == 0.0


def test_cliffs_magnitude_bins():
    assert stats_ext.cliffs_delta([5, 6, 7], [1, 2, 3])["magnitude"] == "large"
    assert stats_ext.cliffs_delta([1, 2, 3], [1, 2, 3])["magnitude"] == "negligible"


def test_distribution_summary_iqr():
    d = stats_ext.distribution([1, 2, 3, 4, 5, 6, 7, 8, 9])
    assert d["median"] == 5
    assert d["q1"] == 3 and d["q3"] == 7 and d["iqr"] == 4
    assert d["n"] == 9


def test_distribution_empty():
    d = stats_ext.distribution([])
    assert d["n"] == 0 and d["median"] is None


def test_compare_paired_bundle():
    a = [10, 11, 12, 13, 14, 15]
    b = [1, 2, 3, 4, 5, 6]
    cmp = stats_ext.compare_paired("e2e_s", "Squish", "Ollama", a, b)
    assert cmp.is_significant()
    assert cmp.cliffs_delta == 1.0
    assert cmp.rank_biserial == 1.0
    assert cmp.median_ratio is not None


# ── cache_probe ───────────────────────────────────────────────────────────────


def test_parse_prometheus():
    text = (
        "# HELP x\n# TYPE x counter\nsquish_radix_prefix_hits_total 5\n"
        "squish_kv_cache_memory_mb 123.5\nbad line here\n"
    )
    m = cache_probe.parse_prometheus(text)
    assert m["squish_radix_prefix_hits_total"] == 5.0
    assert m["squish_kv_cache_memory_mb"] == 123.5


def test_ollama_hit_fraction_partial():
    frac, method = cache_probe.ollama_hit_fraction({"prompt_eval_count": 400}, 800)
    assert math.isclose(frac, 0.5)
    assert "prompt_eval_count" in method


def test_ollama_hit_fraction_full_cache_absent_field():
    frac, method = cache_probe.ollama_hit_fraction({}, 800)
    assert frac == 1.0 and "absent" in method


def test_ollama_hit_fraction_prefill_ratio_when_count_unreliable():
    # Build that reports the full prompt count despite real KV reuse: the count
    # signal reads ~0% reuse, so the prefill-time collapse must take over.
    frac, method = cache_probe.ollama_hit_fraction(
        {"prompt_eval_count": 800}, 800, prefill_cold_s=1.0, prefill_warm_s=0.5
    )
    assert math.isclose(frac, 0.5) and "prefill_ratio" in method


def test_ollama_hit_fraction_count_wins_when_reliable():
    # A genuine partial count is trusted even if a cold ref is present.
    frac, method = cache_probe.ollama_hit_fraction(
        {"prompt_eval_count": 400}, 800, prefill_cold_s=1.0, prefill_warm_s=0.9
    )
    assert math.isclose(frac, 0.5) and "prompt_eval_count" in method


def test_squish_hit_fraction_usage_cached_tokens():
    usage = {"prompt_tokens": 800, "prompt_tokens_details": {"cached_tokens": 600}}
    frac, method = cache_probe.squish_hit_fraction(usage, {}, {}, 800)
    assert math.isclose(frac, 0.75) and method.endswith("cached_tokens")


def test_squish_hit_fraction_exact_metric_delta():
    before = {"squish_prefix_cache_hits_total": 1}
    after = {"squish_prefix_cache_hits_total": 2}
    frac, method = cache_probe.squish_hit_fraction(None, before, after, 800)
    assert frac == 1.0 and "exact" in method


def test_squish_hit_fraction_radix_prefill_ratio():
    before = {"squish_radix_prefix_hits_total": 0}
    after = {"squish_radix_prefix_hits_total": 1}
    frac, method = cache_probe.squish_hit_fraction(
        None, before, after, 800, prefill_cold_s=1.0, prefill_warm_s=0.4
    )
    assert math.isclose(frac, 0.6) and "prefill_ratio" in method


def test_squish_hit_fraction_no_event():
    frac, method = cache_probe.squish_hit_fraction(None, {}, {}, 800)
    assert frac == 0.0 and "no_reuse" in method


def test_classify_zero_reuse_pass_and_fail():
    assert cache_probe.classify("ollama", 0.0, 0.02, "m").status == "ok"
    assert cache_probe.classify("ollama", 0.0, 0.4, "m").status == "mismatch"


def test_classify_partial_band():
    assert cache_probe.classify("squish", 0.5, 0.52, "m").status == "ok"
    assert cache_probe.classify("squish", 0.5, 0.9, "m").status == "mismatch"


def test_classify_exact_and_unknown_nan():
    assert cache_probe.classify("squish", 1.0, 0.99, "m").status == "ok"
    v = cache_probe.classify("squish", 0.5, float("nan"), "radix_only")
    assert v.status == "unknown" and v.measured is None


# ── memory ────────────────────────────────────────────────────────────────────

GB = 1024**3


def test_memory_fit():
    ms = memory.classify_memory_status(
        peak_rss_bytes=8 * GB, ram_bytes=16 * GB, request_failed=False
    )
    assert ms.status == "fit" and ms.fit


def test_memory_oom_on_failure():
    ms = memory.classify_memory_status(
        peak_rss_bytes=16 * GB, ram_bytes=16 * GB, request_failed=True, oom_signal=True
    )
    assert ms.status == "oom"


def test_memory_degraded_near_cap():
    ms = memory.classify_memory_status(
        peak_rss_bytes=int(15.8 * GB), ram_bytes=16 * GB, request_failed=False
    )
    assert ms.status == "degraded_via_governor"


def test_memory_degraded_throughput_collapse():
    ms = memory.classify_memory_status(
        peak_rss_bytes=8 * GB,
        ram_bytes=16 * GB,
        request_failed=False,
        decode_tps=5.0,
        baseline_tps=40.0,
    )
    assert ms.status == "degraded_via_governor"


def test_scan_log_for_signals():
    oom, gov = memory.scan_log_for_signals("... Out of memory: failed to allocate ...")
    assert oom is True
    _, gov2 = memory.scan_log_for_signals("memory pressure detected, compressed pages")
    assert gov2 is True


def test_kv_cache_mb_from_metrics():
    assert memory.kv_cache_mb_from_metrics({"squish_kv_cache_memory_mb": 42.0}) == 42.0
    assert memory.kv_cache_mb_from_metrics({}) is None


# ── thermal ───────────────────────────────────────────────────────────────────


def test_parse_powermetrics_temp():
    text = "CPU die temperature: 47.8 C\nother: 1"
    assert thermal.parse_powermetrics_temp(text) == 47.8


def test_drift_check_pass_and_fail():
    assert thermal.drift_check(20.0, 20.2).passed  # +1.0% <= 1.7%
    assert not thermal.drift_check(20.0, 21.0).passed  # +5.0%
    assert thermal.drift_check(0.0, 0.0).passed


def test_wait_for_baseline_no_sensor_skips():
    assert thermal.wait_for_baseline(reader=lambda: None, log=lambda *_: None) is True


def test_wait_for_baseline_reaches_target():
    seq = iter([70.0, 60.0, 49.0])
    ok = thermal.wait_for_baseline(
        target_c=50.0, poll_s=0.0, reader=lambda: next(seq), log=lambda *_: None
    )
    assert ok is True


def test_thermal_log_records_and_max():
    tl = thermal.ThermalLog()
    tl.record(40.0, "a")
    tl.record(None, "b")  # dropped
    tl.record(55.0, "c")
    assert tl.max_temp() == 55.0 and len(tl.as_rows()) == 2


# ── matrix_spec ───────────────────────────────────────────────────────────────


def test_all_cells_count():
    cells = matrix_spec.all_cells()
    assert len(cells) == len(matrix_spec.REUSE_LEVELS) * len(matrix_spec.CONTEXT_LENGTHS)
    assert cells[0].cell_id == "r000_c4000"


def test_kill_test_cell():
    c = matrix_spec.kill_test_cell()
    assert c.reuse == 0.5 and c.ctx_tokens == 8000


def test_counterbalanced_order_rotates():
    names = ["a", "b", "c"]
    assert matrix_spec.counterbalanced_order(names, 0) == ["a", "b", "c"]
    assert matrix_spec.counterbalanced_order(names, 1) == ["b", "c", "a"]
    assert matrix_spec.counterbalanced_order([], 3) == []


def test_second_model_rows():
    rows = matrix_spec.second_model_rows()
    assert all(r.reuse in (0.0, 0.5) for r in rows)


# ── host ──────────────────────────────────────────────────────────────────────


def test_parse_sysctl_memsize():
    assert host.parse_sysctl_memsize("hw.memsize: 17179869184") == 17179869184
    assert host.parse_sysctl_memsize("hw.memsize: ") is None


# ── cell pure helpers ─────────────────────────────────────────────────────────


def test_inter_token_stats():
    stamps = [0.0, 1.0, 1.1, 1.2, 1.3]  # first gap (TTFT) excluded
    s = inter_token_stats(stamps)
    assert s["itl_count"] == 3 and s["itl_p50_ms"] is not None


def test_inter_token_stats_too_few():
    assert inter_token_stats([0.0, 1.0])["itl_p50_ms"] is None


class _Run:
    def __init__(self, i, val, failed=False):
        self.run_index = i
        self.e2e_s = val
        self.failed = failed


class _Sys:
    def __init__(self, runs):
        self.runs = runs


def test_paired_aligns_common_indices():
    sa = _Sys([_Run(0, 1.0), _Run(1, 2.0), _Run(2, None)])
    sb = _Sys([_Run(0, 3.0), _Run(1, 4.0, failed=True), _Run(2, 5.0)])
    va, vb = _paired(sa, sb, "e2e_s")
    assert va == [1.0] and vb == [3.0]  # only index 0 valid in both


# ── report ────────────────────────────────────────────────────────────────────


def _fake_cell(reuse, ctx, squish_med, ollama_med, p, delta):
    return {
        "cell_id": f"r{int(reuse * 100):03d}_c{ctx}",
        "reuse": reuse,
        "ctx_tokens": ctx,
        "status": "ok",
        "systems": {
            "squish_int4": {
                "role": "head_to_head",
                "runs": [{"failed": False}] * 30,
                "cache_total_runs": 30,
                "cache_ok_runs": 30,
            },
            "ollama_q4km": {
                "role": "head_to_head",
                "runs": [{"failed": False}] * 30,
                "cache_total_runs": 30,
                "cache_ok_runs": 30,
            },
        },
        "comparisons": {
            "e2e_s": {
                "a_system": "squish_int4",
                "b_system": "ollama_q4km",
                "a": {"median": squish_med, "iqr": 0.1},
                "b": {"median": ollama_med, "iqr": 0.2},
                "wilcoxon": {"p_value": p},
                "cliffs_delta": delta,
                "cliffs_magnitude": "large",
            }
        },
    }


def test_speed_ratio_lower_is_better():
    cell = _fake_cell(0.0, 8000, squish_med=1.0, ollama_med=2.0, p=0.001, delta=-1.0)
    # e2e lower is better: ratio = ollama/squish = 2.0
    assert report.speed_ratio(cell, "e2e_s") == 2.0


def test_metric_table_contains_cells():
    cells = [_fake_cell(0.0, 8000, 1.0, 2.0, 0.001, -1.0)]
    table = report.metric_table(cells, "e2e_s", [0.0], [8000])
    assert "End-to-end" in table and "p=0.0010" in table


def test_one_screen_summary_headlines():
    cells = [
        _fake_cell(0.0, 8000, 1.0, 2.0, 0.001, -1.0),
        _fake_cell(1.0, 8000, 0.5, 2.0, 0.0001, -1.0),
    ]
    out = report.one_screen_summary(cells, [8000])
    assert "ONE-SCREEN SUMMARY" in out
    assert "Cold / unique" in out and "Exact-repeat" in out


def test_postflight_pass():
    cells = [_fake_cell(0.0, 8000, 1.0, 2.0, 0.001, -1.0)]
    out = report.postflight(cells, min_runs=30)
    assert "POST-FLIGHT VERIFICATION" in out
    assert "[FAIL]" not in out


def test_postflight_flags_cache_mismatch():
    cell = _fake_cell(0.0, 8000, 1.0, 2.0, 0.001, -1.0)
    cell["systems"]["squish_int4"]["cache_ok_runs"] = 10  # mismatch
    out = report.postflight([cell], min_runs=30)
    assert "[FAIL]" in out
