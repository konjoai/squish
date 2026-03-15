"""tests/benchmarks/test_bench_compare.py — Unit tests for squish/benchmarks/compare.py."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from squish.benchmarks.base import ResultRecord
from squish.benchmarks.compare import CompareConfig, ResultComparator


def _write_result(dirpath: Path, record: ResultRecord) -> None:
    fname = f"{record.track}_{record.engine}.json"
    (dirpath / fname).write_text(json.dumps(record.to_dict()))


def _make_result(track="quality", engine="squish", model="qwen3:8b", metrics=None) -> ResultRecord:
    return ResultRecord(
        track=track,
        engine=engine,
        model=model,
        timestamp="2026-01-01T00:00:00Z",
        metrics=metrics or {"acc": 0.75},
    )


# ---------------------------------------------------------------------------
# CompareConfig
# ---------------------------------------------------------------------------

class TestCompareConfig:
    def test_default_input_dir(self):
        assert CompareConfig().input_dir == "eval_output"

    def test_default_output_dir(self):
        assert CompareConfig().output_dir == "docs"

    def test_default_tracks(self):
        tracks = CompareConfig().tracks
        for t in ("quality", "code", "tools", "agent", "perf"):
            assert t in tracks

    def test_default_engines_is_empty(self):
        assert CompareConfig().engines == []

    def test_default_date_filter_is_empty(self):
        assert CompareConfig().date_filter == ""


# ---------------------------------------------------------------------------
# ResultComparator.load_results
# ---------------------------------------------------------------------------

class TestLoadResults:
    def test_loads_all_matching_json(self):
        with tempfile.TemporaryDirectory() as td:
            _write_result(Path(td), _make_result("quality", "squish"))
            _write_result(Path(td), _make_result("perf", "ollama"))
            comparator = ResultComparator(CompareConfig(input_dir=td))
            results = comparator.load_results()
        assert len(results) == 2

    def test_skips_non_json_files(self):
        with tempfile.TemporaryDirectory() as td:
            (Path(td) / "not_json.txt").write_text("hello")
            _write_result(Path(td), _make_result())
            comparator = ResultComparator(CompareConfig(input_dir=td))
            results = comparator.load_results()
        assert len(results) == 1

    def test_skips_malformed_json(self):
        with tempfile.TemporaryDirectory() as td:
            (Path(td) / "bad.json").write_text("{invalid json")
            _write_result(Path(td), _make_result())
            comparator = ResultComparator(CompareConfig(input_dir=td))
            results = comparator.load_results()
        assert len(results) == 1

    def test_filters_by_track(self):
        with tempfile.TemporaryDirectory() as td:
            _write_result(Path(td), _make_result("quality", "squish"))
            _write_result(Path(td), _make_result("perf", "squish2"))
            comparator = ResultComparator(CompareConfig(input_dir=td, tracks=["quality"]))
            results = comparator.load_results()
        assert all(r.track == "quality" for r in results)

    def test_filters_by_engine(self):
        with tempfile.TemporaryDirectory() as td:
            _write_result(Path(td), _make_result("quality", "squish"))
            _write_result(Path(td), _make_result("quality", "ollama"))
            comparator = ResultComparator(CompareConfig(input_dir=td, engines=["squish"]))
            results = comparator.load_results()
        assert all(r.engine == "squish" for r in results)

    def test_filters_by_date_prefix(self):
        with tempfile.TemporaryDirectory() as td:
            r1 = _make_result("quality", "squish")
            r1.timestamp = "2026-01-01T00:00:00Z"
            r2 = _make_result("quality", "ollama")
            r2.timestamp = "2025-12-15T00:00:00Z"
            _write_result(Path(td), r1)
            _write_result(Path(td), r2)
            comparator = ResultComparator(CompareConfig(input_dir=td, date_filter="2026"))
            results = comparator.load_results()
        assert len(results) == 1
        assert results[0].engine == "squish"

    def test_returns_empty_list_for_missing_dir(self):
        comparator = ResultComparator(CompareConfig(input_dir="/nonexistent_dir_xyz"))
        results = comparator.load_results()
        assert results == []


# ---------------------------------------------------------------------------
# ResultComparator.to_markdown
# ---------------------------------------------------------------------------

class TestToMarkdown:
    def test_empty_results_returns_no_results_comment(self):
        comp = ResultComparator(CompareConfig())
        md = comp.to_markdown([])
        assert "No benchmark results" in md

    def test_contains_track_header(self):
        comp = ResultComparator(CompareConfig())
        results = [_make_result("quality", "squish")]
        md = comp.to_markdown(results)
        assert "Quality" in md

    def test_contains_engine_name(self):
        comp = ResultComparator(CompareConfig())
        results = [_make_result("quality", "squish")]
        md = comp.to_markdown(results)
        assert "squish" in md

    def test_contains_model_name(self):
        comp = ResultComparator(CompareConfig())
        results = [_make_result("quality", "squish", model="qwen3:8b")]
        md = comp.to_markdown(results)
        assert "qwen3:8b" in md

    def test_contains_metric_keys(self):
        comp = ResultComparator(CompareConfig())
        results = [_make_result(metrics={"mmlu_acc": 0.75})]
        md = comp.to_markdown(results)
        assert "mmlu_acc" in md

    def test_skips_tracks_with_no_results(self):
        comp = ResultComparator(CompareConfig(tracks=["quality", "perf"]))
        results = [_make_result("quality")]
        md = comp.to_markdown(results)
        assert "Perf" not in md


# ---------------------------------------------------------------------------
# ResultComparator.to_csv
# ---------------------------------------------------------------------------

class TestToCsv:
    def test_empty_results_returns_header_only(self):
        comp = ResultComparator(CompareConfig())
        csv_str = comp.to_csv([])
        assert csv_str.strip() == "track,engine,model,timestamp"

    def test_contains_header_row(self):
        comp = ResultComparator(CompareConfig())
        results = [_make_result(metrics={"acc": 0.75})]
        csv_str = comp.to_csv(results)
        first_line = csv_str.splitlines()[0]
        assert "track" in first_line
        assert "engine" in first_line
        assert "model" in first_line

    def test_data_row_contains_metric_value(self):
        comp = ResultComparator(CompareConfig())
        results = [_make_result(metrics={"acc": 0.75})]
        csv_str = comp.to_csv(results)
        assert "0.75" in csv_str

    def test_multiple_rows_all_present(self):
        comp = ResultComparator(CompareConfig())
        results = [
            _make_result("quality", "squish", metrics={"acc": 0.80}),
            _make_result("quality", "ollama", metrics={"acc": 0.70}),
        ]
        csv_str = comp.to_csv(results)
        lines = csv_str.strip().splitlines()
        assert len(lines) == 3  # header + 2 data rows


# ---------------------------------------------------------------------------
# ResultComparator.generate
# ---------------------------------------------------------------------------

class TestGenerate:
    def test_generate_returns_markdown_and_csv_keys(self):
        with tempfile.TemporaryDirectory() as td:
            _write_result(Path(td), _make_result())
            comp = ResultComparator(CompareConfig(input_dir=td, output_dir=td))
            output = comp.generate(write_files=False)
        assert "markdown" in output
        assert "csv" in output

    def test_generate_write_files_false_creates_no_files(self):
        with tempfile.TemporaryDirectory() as td:
            in_dir = Path(td) / "eval"
            in_dir.mkdir()
            out_dir = Path(td) / "docs"
            _write_result(in_dir, _make_result())
            comp = ResultComparator(CompareConfig(input_dir=str(in_dir), output_dir=str(out_dir)))
            comp.generate(write_files=False)
        assert not out_dir.exists()

    def test_generate_writes_markdown_file_when_requested(self):
        with tempfile.TemporaryDirectory() as td:
            in_dir = Path(td) / "eval"
            in_dir.mkdir()
            out_dir = Path(td) / "docs"
            _write_result(in_dir, _make_result())
            comp = ResultComparator(CompareConfig(input_dir=str(in_dir), output_dir=str(out_dir)))
            comp.generate(write_files=True)
            md_files = list(out_dir.glob("comparison_*.md"))
            assert len(md_files) == 1
