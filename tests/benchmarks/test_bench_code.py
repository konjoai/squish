"""tests/benchmarks/test_bench_code.py — Unit tests for squish/benchmarks/code_bench.py."""
from __future__ import annotations

import warnings
from unittest.mock import MagicMock, patch

import pytest

from squish.benchmarks.base import EngineConfig, ResultRecord
from squish.benchmarks.code_bench import (
    CODE_TASKS,
    SANDBOX_WARNING,
    CodeBenchConfig,
    CodeBenchRunner,
)

_ENGINE = EngineConfig("squish", "http://localhost:11434")


class TestCodeBenchConfig:
    def test_default_tasks_match_code_tasks(self):
        cfg = CodeBenchConfig()
        assert set(cfg.tasks) == set(CODE_TASKS.keys())

    def test_default_sandbox_is_false(self):
        assert CodeBenchConfig().sandbox is False

    def test_default_limit_is_none(self):
        assert CodeBenchConfig().limit is None

    def test_default_seed(self):
        assert CodeBenchConfig().seed == 42


class TestSandboxWarning:
    def test_sandbox_warning_is_a_string(self):
        assert isinstance(SANDBOX_WARNING, str)
        assert len(SANDBOX_WARNING) > 0


class TestCodeBenchRunner:
    def _make_runner(self, **kw):
        return CodeBenchRunner(CodeBenchConfig(**kw))

    def test_track_name_is_code(self):
        assert self._make_runner().track_name == "code"

    def test_run_returns_result_record(self):
        runner = self._make_runner(tasks=["humaneval"])
        runner._run_task = MagicMock(return_value={"humaneval_pass_at_1": 0.50})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = runner.run(_ENGINE, "qwen3:8b")
        assert isinstance(result, ResultRecord)

    def test_run_emits_user_warning_when_sandbox_false(self):
        runner = self._make_runner(tasks=["humaneval"], sandbox=False)
        runner._run_task = MagicMock(return_value={})
        with pytest.warns(UserWarning):
            runner.run(_ENGINE, "qwen3:8b")

    def test_run_no_warning_when_sandbox_true(self):
        runner = self._make_runner(tasks=["humaneval"], sandbox=True)
        runner._run_task = MagicMock(return_value={})
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            runner.run(_ENGINE, "qwen3:8b")
        user_warns = [x for x in w if issubclass(x.category, UserWarning)]
        assert len(user_warns) == 0

    def test_run_sandbox_flag_in_metrics(self):
        runner = self._make_runner(tasks=["humaneval"], sandbox=True)
        runner._run_task = MagicMock(return_value={})
        result = runner.run(_ENGINE, "qwen3:8b")
        assert result.metrics.get("sandbox_enabled") is True

    def test_run_captures_task_exception_as_error_key(self):
        runner = self._make_runner(tasks=["humaneval"])
        runner._run_task = MagicMock(side_effect=ValueError("oops"))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = runner.run(_ENGINE, "qwen3:8b")
        assert "humaneval_error" in result.metrics

    def test_run_task_metric_key_at_to_underscore(self):
        runner = self._make_runner()
        fake_lm_eval = MagicMock()
        fake_lm_eval.simple_evaluate.return_value = {
            "results": {"humaneval": {"pass@1": 0.45}}
        }
        with patch.dict("sys.modules", {"lm_eval": fake_lm_eval}):
            out = runner._run_task(_ENGINE, "m", "humaneval", None)
        # pass@1 → pass_at_1
        assert "humaneval_pass_at_1" in out
        assert out["humaneval_pass_at_1"] == pytest.approx(0.45, abs=1e-4)

    def test_run_task_fallback_to_comma_none_metric(self):
        runner = self._make_runner()
        fake_lm_eval = MagicMock()
        fake_lm_eval.simple_evaluate.return_value = {
            "results": {"mbpp": {"pass@1,none": 0.60}}
        }
        with patch.dict("sys.modules", {"lm_eval": fake_lm_eval}):
            out = runner._run_task(_ENGINE, "m", "mbpp", None)
        assert "mbpp_pass_at_1" in out
        assert out["mbpp_pass_at_1"] == pytest.approx(0.60, abs=1e-4)

    def test_run_task_returns_error_when_lm_eval_not_installed(self):
        runner = self._make_runner()
        with patch.dict("sys.modules", {"lm_eval": None}):
            out = runner._run_task(_ENGINE, "m", "humaneval", None)
        assert "humaneval_error" in out

    def test_run_metadata_contains_sandbox_flag(self):
        runner = self._make_runner(tasks=["humaneval"], sandbox=True)
        runner._run_task = MagicMock(return_value={})
        result = runner.run(_ENGINE, "qwen3:8b")
        assert result.metadata.get("sandbox") is True
