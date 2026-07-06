"""Unit tests for the benchmark-matrix harness — memory/thermal/reporting half.

Split out of test_benchmark_matrix.py (which covers corpus construction, paired
statistics, and cache-hit measurement/classification) to keep both files under
the 500-line gate. Covers: memory/OOM classification, thermal math, matrix
spec, host parsing, cell-level pure helpers, and reporting. The bench-host
modules (systems/cell orchestration) are exercised on Apple Silicon, not here.
"""

from __future__ import annotations

from benchmarks.ollama_vs_squish.matrix import host, matrix_spec, memory, report, thermal
from benchmarks.ollama_vs_squish.matrix.cell import _paired, inter_token_stats

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


def test_parse_macmon_temp_takes_max_of_cpu_gpu():
    line = '{"temp": {"cpu_temp_avg": 61.5, "gpu_temp_avg": 68.2}}'
    assert thermal.parse_macmon_temp(line) == 68.2


def test_parse_macmon_temp_uses_last_line_of_stream():
    text = '{"temp": {"cpu_temp_avg": 40.0}}\n{"temp": {"cpu_temp_avg": 50.0}}'
    assert thermal.parse_macmon_temp(text) == 50.0


def test_parse_macmon_temp_malformed_or_empty_returns_none():
    assert thermal.parse_macmon_temp("not json") is None
    assert thermal.parse_macmon_temp("") is None
    assert thermal.parse_macmon_temp('{"temp": {}}') is None


def test_plausible_die_temp_rejects_bogus_zero():
    # osx-cpu-temp on Apple Silicon reads Intel-only SMC keys and returns a
    # fixed 0.0 rather than failing — must be rejected, not trusted.
    assert thermal._plausible_die_temp(0.0) is False
    assert thermal._plausible_die_temp(47.8) is True
    assert thermal._plausible_die_temp(200.0) is False


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
