"""tests/benchmarks/test_bench_tool.py — Unit tests for squish/benchmarks/tool_bench.py."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from squish.benchmarks.base import EngineConfig, ResultRecord
from squish.benchmarks.tool_bench import (
    ToolBenchConfig,
    ToolBenchRunner,
    ToolEvaluator,
    _load_schemas,
)

_ENGINE = EngineConfig("squish", "http://localhost:11434")


def _make_tool_call_response(name: str, args: dict) -> dict:
    return {
        "choices": [{
            "message": {
                "tool_calls": [{
                    "function": {
                        "name": name,
                        "arguments": json.dumps(args),
                    }
                }]
            },
            "finish_reason": "tool_calls",
        }]
    }


def _no_tool_call_response() -> dict:
    return {
        "choices": [{"message": {"content": "Sure!"}, "finish_reason": "stop"}]
    }


# ---------------------------------------------------------------------------
# ToolEvaluator.schema_compliance
# ---------------------------------------------------------------------------

class TestSchemaCompliance:
    def test_valid_response_is_compliant(self):
        resp = _make_tool_call_response("get_weather", {"location": "NYC"})
        assert ToolEvaluator.schema_compliance(resp) is True

    def test_no_choices_not_compliant(self):
        assert ToolEvaluator.schema_compliance({}) is False

    def test_empty_choices_not_compliant(self):
        assert ToolEvaluator.schema_compliance({"choices": []}) is False

    def test_no_tool_calls_not_compliant(self):
        resp = {"choices": [{"message": {"content": "Hello"}}]}
        assert ToolEvaluator.schema_compliance(resp) is False

    def test_tool_call_non_dict_not_compliant(self):
        resp = {"choices": [{"message": {"tool_calls": ["not_a_dict"]}}]}
        assert ToolEvaluator.schema_compliance(resp) is False

    def test_tool_call_name_not_string_not_compliant(self):
        resp = {
            "choices": [{"message": {"tool_calls": [
                {"function": {"name": 42, "arguments": "{}"}}
            ]}}]
        }
        assert ToolEvaluator.schema_compliance(resp) is False

    def test_invalid_args_json_not_compliant(self):
        resp = {
            "choices": [{"message": {"tool_calls": [
                {"function": {"name": "f", "arguments": "not json"}}
            ]}}]
        }
        assert ToolEvaluator.schema_compliance(resp) is False

    def test_valid_empty_args_is_compliant(self):
        resp = _make_tool_call_response("ping", {})
        assert ToolEvaluator.schema_compliance(resp) is True


# ---------------------------------------------------------------------------
# ToolEvaluator.function_name_match
# ---------------------------------------------------------------------------

class TestFunctionNameMatch:
    def test_matching_name_returns_true(self):
        resp = _make_tool_call_response("calc", {"a": 1})
        assert ToolEvaluator.function_name_match(resp, "calc") is True

    def test_non_matching_name_returns_false(self):
        resp = _make_tool_call_response("calc", {"a": 1})
        assert ToolEvaluator.function_name_match(resp, "other") is False

    def test_empty_name_returns_false(self):
        resp = _make_tool_call_response("calc", {})
        assert ToolEvaluator.function_name_match(resp, "") is False

    def test_no_choices_returns_false(self):
        assert ToolEvaluator.function_name_match({}, "calc") is False

    def test_no_tool_calls_returns_false(self):
        resp = _no_tool_call_response()
        assert ToolEvaluator.function_name_match(resp, "calc") is False


# ---------------------------------------------------------------------------
# ToolEvaluator.argument_match
# ---------------------------------------------------------------------------

class TestArgumentMatch:
    def test_all_args_matched(self):
        resp = _make_tool_call_response("f", {"a": 1, "b": 2})
        matched, total = ToolEvaluator.argument_match(resp, {"a": 1, "b": 2})
        assert matched == 2
        assert total == 2

    def test_partial_args_matched(self):
        resp = _make_tool_call_response("f", {"a": 1})
        matched, total = ToolEvaluator.argument_match(resp, {"a": 1, "b": 2})
        assert matched == 1
        assert total == 2

    def test_no_tool_calls_returns_zero(self):
        resp = _no_tool_call_response()
        matched, total = ToolEvaluator.argument_match(resp, {"a": 1})
        assert matched == 0
        assert total == 1

    def test_empty_expected_args_returns_zero_total(self):
        resp = _make_tool_call_response("f", {"a": 1})
        matched, total = ToolEvaluator.argument_match(resp, {})
        assert total == 0

    def test_invalid_arg_json_returns_zero(self):
        resp = {
            "choices": [{"message": {"tool_calls": [
                {"function": {"name": "f", "arguments": "bad"}}
            ]}}]
        }
        matched, total = ToolEvaluator.argument_match(resp, {"a": 1})
        assert matched == 0


# ---------------------------------------------------------------------------
# ToolEvaluator.exact_match
# ---------------------------------------------------------------------------

class TestExactMatch:
    def test_full_match_returns_true(self):
        resp = _make_tool_call_response("calc", {"a": 1, "b": 2})
        assert ToolEvaluator.exact_match(resp, {"name": "calc", "arguments": {"a": 1, "b": 2}}) is True

    def test_name_mismatch_returns_false(self):
        resp = _make_tool_call_response("other", {"a": 1, "b": 2})
        assert ToolEvaluator.exact_match(resp, {"name": "calc", "arguments": {"a": 1, "b": 2}}) is False

    def test_missing_args_returns_false(self):
        resp = _make_tool_call_response("calc", {"a": 1})
        assert ToolEvaluator.exact_match(resp, {"name": "calc", "arguments": {"a": 1, "b": 2}}) is False

    def test_extra_args_in_response_still_matches_required(self):
        resp = _make_tool_call_response("calc", {"a": 1, "b": 2, "extra": "x"})
        assert ToolEvaluator.exact_match(resp, {"name": "calc", "arguments": {"a": 1, "b": 2}}) is True


# ---------------------------------------------------------------------------
# _load_schemas
# ---------------------------------------------------------------------------

class TestLoadSchemas:
    def test_bundled_schemas_load_successfully(self):
        schemas = _load_schemas()
        assert isinstance(schemas, list)
        assert len(schemas) > 0

    def test_schema_entries_have_required_keys(self):
        schemas = _load_schemas()
        for s in schemas:
            assert "tool" in s
            assert "prompt" in s

    def test_custom_path_loads_correctly(self):
        data = [{"id": "t1", "tool": {"name": "f"}, "prompt": "Call f.", "expected": {}}]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            fname = f.name
        loaded = _load_schemas(fname)
        assert len(loaded) == 1
        assert loaded[0]["id"] == "t1"
        Path(fname).unlink()


# ---------------------------------------------------------------------------
# ToolBenchConfig
# ---------------------------------------------------------------------------

class TestToolBenchConfig:
    def test_default_limit_is_none(self):
        assert ToolBenchConfig().limit is None

    def test_default_use_bfcl_is_false(self):
        assert ToolBenchConfig().use_bfcl is False

    def test_default_bfcl_limit(self):
        assert ToolBenchConfig().bfcl_limit == 200


# ---------------------------------------------------------------------------
# ToolBenchRunner
# ---------------------------------------------------------------------------

class TestToolBenchRunner:
    def _make_runner(self, **kw):
        return ToolBenchRunner(ToolBenchConfig(**kw))

    def test_track_name_is_tools(self):
        assert self._make_runner().track_name == "tools"

    def test_run_with_mocked_client_returns_result_record(self):
        runner = self._make_runner()
        runner._run_canonical = MagicMock(return_value=[
            {"schema_compliance": True, "name_match": True, "arg_match_ratio": 1.0, "exact_match": True}
        ])
        with patch("squish.benchmarks.tool_bench.EngineClient"):
            result = runner.run(_ENGINE, "qwen3:8b")
        assert isinstance(result, ResultRecord)
        assert result.track == "tools"

    def test_run_calculates_compliance_pct(self):
        runner = self._make_runner()
        case_results = [
            {"schema_compliance": True,  "name_match": True,  "arg_match_ratio": 1.0, "exact_match": True},
            {"schema_compliance": False, "name_match": False, "arg_match_ratio": 0.0, "exact_match": False},
        ]
        runner._run_canonical = MagicMock(return_value=case_results)
        with patch("squish.benchmarks.tool_bench.EngineClient"):
            result = runner.run(_ENGINE, "qwen3:8b")
        assert result.metrics["schema_compliance_pct"] == pytest.approx(0.5, abs=1e-4)
        assert result.metrics["exact_match_pct"] == pytest.approx(0.5, abs=1e-4)

    def test_run_applies_limit(self):
        runner = self._make_runner()
        runner._run_canonical = MagicMock(return_value=[])
        with patch("squish.benchmarks.tool_bench.EngineClient"), \
             patch("squish.benchmarks.tool_bench._load_schemas", return_value=[{"tool": {}, "prompt": ""}] * 10):
            runner.run(_ENGINE, "qwen3:8b", limit=3)
        args = runner._run_canonical.call_args[0]
        schemas_passed = args[2]
        assert len(schemas_passed) == 3

    def test_run_metadata_has_schemas_source(self):
        runner = self._make_runner()
        runner._run_canonical = MagicMock(return_value=[])
        with patch("squish.benchmarks.tool_bench.EngineClient"):
            result = runner.run(_ENGINE, "qwen3:8b")
        assert result.metadata.get("schemas_source") == "canonical"
