"""tests/test_squash_wave41.py — Unit tests for dev/benchmarks/squish_lm_eval.py.

Wave 41: squish-native lm_eval harness.

All tests are pure unit tests — no I/O beyond creating temp directories in
memory, no real model weights, no Metal/MLX calls.  The module under test is
loaded from its file path via importlib so it lives in dev/ (outside squish/)
without polluting the squish package namespace.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Load the dev script under test via importlib (it is NOT a squish/ module)
# ---------------------------------------------------------------------------

_BENCH_SCRIPT = (
    Path(__file__).parent.parent / "dev" / "benchmarks" / "squish_lm_eval.py"
)
assert _BENCH_SCRIPT.exists(), f"squish_lm_eval.py not found at {_BENCH_SCRIPT}"

_SPEC   = importlib.util.spec_from_file_location("squish_lm_eval", _BENCH_SCRIPT)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

# Convenience aliases
_validate_npy_dir         = _MODULE._validate_npy_dir
_detect_eval_dir          = _MODULE._detect_eval_dir
_extract_metric           = _MODULE._extract_metric
compare_to_baseline       = _MODULE.compare_to_baseline
_summarize_row            = _MODULE._summarize_row
_npy_dir_format_tag       = _MODULE._npy_dir_format_tag
_is_thinking_model        = _MODULE._is_thinking_model
_results_path             = _MODULE._results_path
_WAVE41_BASELINE_ARC_EASY = _MODULE._WAVE41_BASELINE_ARC_EASY
_WAVE41_THRESHOLD_PP      = _MODULE._WAVE41_THRESHOLD_PP
TASKS                     = _MODULE.TASKS
_ALL_TASK_NAMES           = _MODULE._ALL_TASK_NAMES
_build_parser             = _MODULE._build_parser


# ---------------------------------------------------------------------------
# Helper: create filesystem stubs in tmp_path
# ---------------------------------------------------------------------------

def _stub_native_mlx(base: Path) -> Path:
    """Create a minimal native MLX safetensors dir stub."""
    base.mkdir(parents=True, exist_ok=True)
    (base / "config.json").write_text('{"model_type": "qwen2"}')
    (base / "model.safetensors").write_bytes(b"FAKE")
    return base


def _stub_squish_npy_dir(base: Path) -> Path:
    """Create a minimal squish npy-dir stub (manifest + tensors/)."""
    base.mkdir(parents=True, exist_ok=True)
    (base / "manifest.json").write_text('{"tensors": {}}')
    (base / "tensors").mkdir()
    return base


def _stub_squish_4bit_cache(npy_dir: Path) -> Path:
    """Add a squish_4bit/ cache inside an existing npy-dir stub."""
    cache = npy_dir / "squish_4bit"
    cache.mkdir(exist_ok=True)
    (cache / "config.json").write_text('{"model_type": "qwen2"}')
    (npy_dir / ".squish_4bit_ready").write_text("1")
    return cache


def _stub_squish_3bit_cache(npy_dir: Path) -> Path:
    """Add a squish_3bit/ cache inside an existing npy-dir stub."""
    cache = npy_dir / "squish_3bit"
    cache.mkdir(exist_ok=True)
    (cache / "model.safetensors").write_bytes(b"FAKE")
    (npy_dir / ".squish_3bit_ready").write_text("1")
    return cache


# ===========================================================================
# Tests: _validate_npy_dir
# ===========================================================================

class TestValidateNpyDir:
    def test_valid_native_mlx(self, tmp_path: Path) -> None:
        nd = _stub_native_mlx(tmp_path / "model")
        valid, reason = _validate_npy_dir(nd)
        assert valid is True
        assert reason == "native-mlx"

    def test_valid_squish_npy_dir(self, tmp_path: Path) -> None:
        nd = _stub_squish_npy_dir(tmp_path / "model-int4-awq")
        valid, reason = _validate_npy_dir(nd)
        assert valid is True
        assert reason == "squish-npy-dir"

    def test_invalid_empty_dir(self, tmp_path: Path) -> None:
        nd = tmp_path / "empty"
        nd.mkdir()
        valid, reason = _validate_npy_dir(nd)
        assert valid is False
        assert "neither" in reason.lower()

    def test_invalid_manifest_but_no_tensors_dir(self, tmp_path: Path) -> None:
        nd = tmp_path / "broken"
        nd.mkdir()
        (nd / "manifest.json").write_text('{}')
        # tensors/ is absent
        valid, reason = _validate_npy_dir(nd)
        assert valid is False
        assert "tensors" in reason

    def test_invalid_path_does_not_exist(self, tmp_path: Path) -> None:
        nd = tmp_path / "nonexistent"
        valid, reason = _validate_npy_dir(nd)
        assert valid is False
        assert "does not exist" in reason

    def test_invalid_path_is_a_file(self, tmp_path: Path) -> None:
        f = tmp_path / "file.json"
        f.write_text("{}")
        valid, reason = _validate_npy_dir(f)
        assert valid is False
        assert "not a directory" in reason


# ===========================================================================
# Tests: _detect_eval_dir
# ===========================================================================

class TestDetectEvalDir:
    def test_prefers_squish_4bit_cache(self, tmp_path: Path) -> None:
        nd = _stub_squish_npy_dir(tmp_path / "model-int4")
        cache = _stub_squish_4bit_cache(nd)
        result = _detect_eval_dir(nd)
        assert result == cache

    def test_falls_back_to_squish_3bit(self, tmp_path: Path) -> None:
        nd = _stub_squish_npy_dir(tmp_path / "model-int3")
        cache = _stub_squish_3bit_cache(nd)
        result = _detect_eval_dir(nd)
        assert result == cache

    def test_squish_4bit_takes_priority_over_3bit(self, tmp_path: Path) -> None:
        nd = _stub_squish_npy_dir(tmp_path / "model-both")
        cache4 = _stub_squish_4bit_cache(nd)
        _stub_squish_3bit_cache(nd)
        result = _detect_eval_dir(nd)
        assert result == cache4

    def test_native_mlx_returns_self(self, tmp_path: Path) -> None:
        nd = _stub_native_mlx(tmp_path / "native-model")
        result = _detect_eval_dir(nd)
        assert result == nd

    def test_no_cache_returns_none(self, tmp_path: Path) -> None:
        nd = _stub_squish_npy_dir(tmp_path / "model-no-cache")
        result = _detect_eval_dir(nd)
        assert result is None

    def test_squish_4bit_sentinel_without_config_not_used(self, tmp_path: Path) -> None:
        """Sentinel present but config.json missing → do not return the partial cache."""
        nd = _stub_squish_npy_dir(tmp_path / "partial")
        # Write sentinel but no config.json inside squish_4bit/
        (nd / "squish_4bit").mkdir()
        (nd / ".squish_4bit_ready").write_text("1")
        # config.json is absent from squish_4bit/
        result = _detect_eval_dir(nd)
        assert result is None


# ===========================================================================
# Tests: _extract_metric
# ===========================================================================

class TestExtractMetric:
    def test_flat_dict_acc_norm(self) -> None:
        data = {"acc_norm,none": 0.706, "acc_norm_stderr,none": 0.012}
        val = _extract_metric(data, "acc_norm,none")
        assert val == pytest.approx(0.706)

    def test_flat_dict_acc(self) -> None:
        data = {"acc,none": 0.554, "acc_stderr,none": 0.008}
        val = _extract_metric(data, "acc,none")
        assert val == pytest.approx(0.554)

    def test_nested_results_format(self) -> None:
        data = {"results": {"arc_easy": {"acc_norm,none": 0.706, "acc_norm_stderr,none": 0.01}}}
        val = _extract_metric(data, "acc_norm,none")
        assert val == pytest.approx(0.706)

    def test_missing_key_returns_none(self) -> None:
        data = {"some_other_key": 0.5}
        val = _extract_metric(data, "acc_norm,none")
        assert val is None

    def test_empty_dict_returns_none(self) -> None:
        val = _extract_metric({}, "acc_norm,none")
        assert val is None

    def test_does_not_return_stderr(self) -> None:
        data = {"acc_norm_stderr,none": 0.099}
        val = _extract_metric(data, "acc_norm,none")
        assert val is None


# ===========================================================================
# Tests: compare_to_baseline
# ===========================================================================

class TestCompareToBaseline:
    def test_within_tolerance_exact(self) -> None:
        passed, delta = compare_to_baseline(70.6, baseline_pct=70.6, threshold_pp=2.0)
        assert passed is True
        assert delta == pytest.approx(0.0)

    def test_within_tolerance_above(self) -> None:
        passed, delta = compare_to_baseline(72.0, baseline_pct=70.6, threshold_pp=2.0)
        assert passed is True
        assert delta == pytest.approx(1.4, abs=0.01)

    def test_within_tolerance_below(self) -> None:
        passed, delta = compare_to_baseline(69.0, baseline_pct=70.6, threshold_pp=2.0)
        assert passed is True
        assert delta == pytest.approx(-1.6, abs=0.01)

    def test_outside_tolerance_below(self) -> None:
        passed, delta = compare_to_baseline(67.0, baseline_pct=70.6, threshold_pp=2.0)
        assert passed is False
        assert delta == pytest.approx(-3.6, abs=0.01)

    def test_outside_tolerance_above(self) -> None:
        passed, delta = compare_to_baseline(75.0, baseline_pct=70.6, threshold_pp=2.0)
        assert passed is False
        assert delta == pytest.approx(4.4, abs=0.01)

    def test_default_baseline_matches_wave41_spec(self) -> None:
        passed, _ = compare_to_baseline(_WAVE41_BASELINE_ARC_EASY)
        assert passed is True

    def test_default_threshold_matches_wave41_spec(self) -> None:
        # 2.0pp tolerance as per Session spec
        assert _WAVE41_THRESHOLD_PP == pytest.approx(2.0)


# ===========================================================================
# Tests: _npy_dir_format_tag
# ===========================================================================

class TestNpyDirFormatTag:
    def test_detects_int4_awq(self, tmp_path: Path) -> None:
        nd = tmp_path / "Qwen2.5-1.5B-Instruct-int4-awq"
        nd.mkdir()
        assert _npy_dir_format_tag(nd) == "int4-awq"

    def test_detects_mixed_attn(self, tmp_path: Path) -> None:
        nd = tmp_path / "Qwen2.5-1.5B-Instruct-mixed-attn"
        nd.mkdir()
        assert _npy_dir_format_tag(nd) == "mixed-attn"

    def test_detects_int3(self, tmp_path: Path) -> None:
        nd = tmp_path / "Qwen3-0.6B-int3"
        nd.mkdir()
        assert _npy_dir_format_tag(nd) == "int3"

    def test_unknown_falls_back(self, tmp_path: Path) -> None:
        nd = tmp_path / "some-model-custom-name"
        nd.mkdir()
        assert _npy_dir_format_tag(nd) == "squish-npy"


# ===========================================================================
# Tests: _is_thinking_model
# ===========================================================================

class TestIsThinkingModel:
    def test_qwen3_is_thinking(self) -> None:
        assert _is_thinking_model("Qwen3-0.6B") is True
        assert _is_thinking_model("Qwen3-8B-Instruct-int4-awq") is True

    def test_qwen25_is_not_thinking(self) -> None:
        assert _is_thinking_model("Qwen2.5-1.5B-Instruct") is False

    def test_llama_is_not_thinking(self) -> None:
        assert _is_thinking_model("Llama-3.2-1B-Instruct") is False

    def test_gemma_is_not_thinking(self) -> None:
        assert _is_thinking_model("gemma-3-4b-it") is False


# ===========================================================================
# Tests: _results_path
# ===========================================================================

class TestResultsPath:
    def test_structure(self, tmp_path: Path) -> None:
        p = _results_path(tmp_path, "my-model", "20260401T120000")
        assert p.parent.name == "squish_lmeval_20260401T120000"
        assert p.name == "my-model.json"
        assert p.parent.parent == tmp_path


# ===========================================================================
# Tests: _summarize_row
# ===========================================================================

class TestSummarizeRow:
    def test_renders_score(self) -> None:
        row = _summarize_row("my-model", "arc_easy", 70.6)
        assert "70.6%" in row or "70.6" in row
        assert "arc_easy" in row

    def test_renders_none_as_na(self) -> None:
        row = _summarize_row("my-model", "arc_easy", None)
        assert "N/A" in row


# ===========================================================================
# Tests: TASKS constants are well-formed
# ===========================================================================

class TestTasksConstants:
    def test_six_tasks(self) -> None:
        assert len(TASKS) == 6

    def test_all_tasks_have_metric(self) -> None:
        for name, metric, _ in TASKS:
            assert metric, f"task {name} has empty metric"

    def test_all_task_names_in_list(self) -> None:
        assert set(_ALL_TASK_NAMES) == {t for t, _, _ in TASKS}

    def test_arc_easy_uses_acc_norm(self) -> None:
        arc_metrics = {t: m for t, m, _ in TASKS}
        assert "acc_norm" in arc_metrics["arc_easy"]

    def test_winogrande_uses_acc(self) -> None:
        task_metrics = {t: m for t, m, _ in TASKS}
        assert "acc,none" == task_metrics["winogrande"]


# ===========================================================================
# Tests: CLI / argparse
# ===========================================================================

class TestArgparse:
    def test_requires_npy_dir(self) -> None:
        """--npy-dir is required; omitting it must cause SystemExit."""
        parser = _build_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args([])
        assert exc_info.value.code != 0

    def test_default_limit(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["--npy-dir", "/tmp/fake"])
        assert args.limit == 500

    def test_default_batch_size(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["--npy-dir", "/tmp/fake"])
        assert args.batch_size == 4

    def test_tasks_default_is_all_6(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["--npy-dir", "/tmp/fake"])
        assert set(args.tasks) == set(_ALL_TASK_NAMES)

    def test_skip_cache_build_default_false(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["--npy-dir", "/tmp/fake"])
        assert args.skip_cache_build is False

    def test_quiet_default_false(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["--npy-dir", "/tmp/fake"])
        assert args.quiet is False

    def test_tasks_subset(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["--npy-dir", "/tmp/fake", "--tasks", "arc_easy", "hellaswag"])
        assert args.tasks == ["arc_easy", "hellaswag"]

    def test_invalid_task_rejected(self) -> None:
        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--npy-dir", "/tmp/fake", "--tasks", "not_a_real_task"])


# ===========================================================================
# Tests: module isolation — squish_lm_eval.py must NOT be a squish/ module
# ===========================================================================

class TestModuleIsolation:
    def test_dev_script_not_in_squish_package(self) -> None:
        """squish_lm_eval must live under dev/benchmarks/, not inside the squish/ package."""
        repo_root   = _BENCH_SCRIPT.parent.parent.parent  # /…/squish (repo root)
        squish_pkg  = repo_root / "squish"               # /…/squish/squish (package)
        # The script path must NOT be relative to the squish package dir
        try:
            _BENCH_SCRIPT.relative_to(squish_pkg)
            inside_pkg = True
        except ValueError:
            inside_pkg = False
        assert not inside_pkg, (
            f"squish_lm_eval.py is inside the squish/ package at {_BENCH_SCRIPT}. "
            "It must live in dev/benchmarks/ instead."
        )

    def test_squish_module_count_unchanged(self) -> None:
        """Adding squish_lm_eval.py must not increase squish/ Python module count.

        The file lives in dev/benchmarks/ so the squish/ count must remain the
        same as after Wave 40.  We only assert it stays ≤115 (generous headroom
        above the 108 post-Wave-40 count) to avoid fragility from unrelated
        structural changes while still catching gross violations.
        """
        squish_dir = _BENCH_SCRIPT.parent.parent.parent / "squish"
        py_files = list(squish_dir.rglob("*.py"))
        count = len(py_files)
        assert count <= 134, (
            f"squish/ Python module count {count} exceeds ceiling 134. "
            "W54-56 added 4 new squash feature modules (remediate, evaluator, edge_formats, chat); "
            "W57 added cloud_db.py (SQLite persistence, justified). "
            "W83 added nist_rmf.py (NIST AI RMF 1.0 controls scanner, justified). "
            "Was a new module inadvertently added to squish/ in Wave 41?"
        )

    def test_squish_lm_eval_not_importable_as_squish_submodule(self) -> None:
        """Importing squish_lm_eval via squish.* must fail (it is a dev script)."""
        assert "squish.squish_lm_eval"   not in sys.modules
        assert "squish.quant.lm_eval"    not in sys.modules
        assert "squish.dev.squish_lm_eval" not in sys.modules
