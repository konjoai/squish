"""tests/benchmarks/test_bench_quality.py — Unit tests for squish/benchmarks/quality_bench.py."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from squish.benchmarks.base import EngineConfig, ResultRecord
from squish.benchmarks.quality_bench import (
    QUALITY_TASKS,
    QualityBenchConfig,
    QualityBenchRunner,
)


_ENGINE = EngineConfig("squish", "http://localhost:11434")


class TestQualityBenchConfig:
    def test_default_tasks_match_quality_tasks(self):
        cfg = QualityBenchConfig()
        assert set(cfg.tasks) == set(QUALITY_TASKS.keys())

    def test_default_limit_is_none(self):
        assert QualityBenchConfig().limit is None

    def test_default_seed(self):
        assert QualityBenchConfig().seed == 42

    def test_default_batch_size(self):
        assert QualityBenchConfig().batch_size == 1


class TestQualityBenchRunner:
    def _make_runner(self, **kw):
        return QualityBenchRunner(QualityBenchConfig(**kw))

    def test_track_name_is_quality(self):
        assert self._make_runner().track_name == "quality"

    def test_run_returns_result_record(self):
        runner = self._make_runner(tasks=["mmlu"])
        runner._run_task = MagicMock(return_value={"mmlu_acc": 0.70})
        result = runner.run(_ENGINE, "qwen3:8b")
        assert isinstance(result, ResultRecord)

    def test_run_track_and_engine_set_correctly(self):
        runner = self._make_runner(tasks=["mmlu"])
        runner._run_task = MagicMock(return_value={"mmlu_acc": 0.70})
        result = runner.run(_ENGINE, "qwen3:8b")
        assert result.track == "quality"
        assert result.engine == "squish"
        assert result.model == "qwen3:8b"

    def test_run_merges_per_task_metrics(self):
        runner = self._make_runner(tasks=["mmlu", "gsm8k"])
        runner._run_task = MagicMock(side_effect=[
            {"mmlu_acc": 0.70},
            {"gsm8k_exact_match": 0.55},
        ])
        result = runner.run(_ENGINE, "qwen3:8b")
        assert "mmlu_acc" in result.metrics
        assert "gsm8k_exact_match" in result.metrics

    def test_run_captures_task_exception_as_error_key(self):
        runner = self._make_runner(tasks=["mmlu"])
        runner._run_task = MagicMock(side_effect=RuntimeError("boom"))
        result = runner.run(_ENGINE, "qwen3:8b")
        assert "mmlu_error" in result.metrics

    def test_run_override_limit(self):
        runner = self._make_runner(tasks=["mmlu"], limit=5)
        calls = []
        def capture(engine, model, task, lim):
            calls.append(lim)
            return {}
        runner._run_task = capture
        runner.run(_ENGINE, "qwen3:8b", limit=10)
        # caller limit=10 overrides config limit=5
        assert calls[0] == 10

    def test_run_uses_config_limit_when_not_overridden(self):
        runner = self._make_runner(tasks=["mmlu"], limit=7)
        calls = []
        def capture(engine, model, task, lim):
            calls.append(lim)
            return {}
        runner._run_task = capture
        runner.run(_ENGINE, "qwen3:8b")
        assert calls[0] == 7

    def test_run_metadata_contains_tasks_and_seed(self):
        runner = self._make_runner(tasks=["mmlu"])
        runner._run_task = MagicMock(return_value={})
        result = runner.run(_ENGINE, "qwen3:8b")
        assert "tasks" in result.metadata
        assert "seed" in result.metadata

    def test_run_task_returns_error_when_lm_eval_not_installed(self):
        runner = self._make_runner()
        with patch.dict("sys.modules", {"lm_eval": None}):
            out = runner._run_task(_ENGINE, "qwen3:8b", "mmlu", None)
        assert "mmlu_error" in out
        assert "lm_eval" in out["mmlu_error"]

    def test_run_task_parses_metric_key(self):
        runner = self._make_runner()
        fake_lm_eval = MagicMock()
        fake_lm_eval.simple_evaluate.return_value = {
            "results": {"mmlu": {"acc": 0.65}}
        }
        with patch.dict("sys.modules", {"lm_eval": fake_lm_eval}):
            out = runner._run_task(_ENGINE, "m", "mmlu", None)
        assert "mmlu_acc" in out
        assert out["mmlu_acc"] == pytest.approx(0.65, abs=1e-4)

    def test_output_path_for_contains_engine_and_model(self):
        runner = self._make_runner()
        path = runner.output_path_for("squish", "qwen3:8b")
        assert "squish" in str(path)
        assert "qwen3" in str(path)
