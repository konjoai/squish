"""tests/benchmarks/test_bench_agent.py — Unit tests for squish/benchmarks/agent_bench.py."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from squish.benchmarks.base import EngineConfig, ResultRecord
from squish.benchmarks.agent_bench import (
    AgentBenchConfig,
    AgentBenchRunner,
    AgentScenario,
    ToolFixtureReplay,
    _load_scenarios,
)

_ENGINE = EngineConfig("squish", "http://localhost:11434")

_SCENARIO_DICT = {
    "id": "test_1",
    "category": "file_ops",
    "goal": "Read /data/config.json",
    "tools": [
        {
            "name": "file_read",
            "description": "Read a file",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        }
    ],
    "tool_fixtures": {
        "file_read": {"{\"path\": \"/data/config.json\"}": {"content": "debug=true"}}
    },
    "expected_sequence": ["file_read"],
    "expected_final_answer": "debug|config",
}


# ---------------------------------------------------------------------------
# AgentScenario.from_dict
# ---------------------------------------------------------------------------

class TestAgentScenarioFromDict:
    def test_fields_populated_correctly(self):
        s = AgentScenario.from_dict(_SCENARIO_DICT)
        assert s.id == "test_1"
        assert s.category == "file_ops"
        assert s.goal == "Read /data/config.json"
        assert len(s.tools) == 1
        assert s.expected_sequence == ["file_read"]
        assert s.expected_final_answer == "debug|config"

    def test_missing_optional_fields_have_defaults(self):
        minimal = {"id": "x", "category": "c", "goal": "g", "tools": []}
        s = AgentScenario.from_dict(minimal)
        assert s.tool_fixtures == {}
        assert s.expected_sequence == []
        assert s.expected_final_answer == ""


# ---------------------------------------------------------------------------
# ToolFixtureReplay
# ---------------------------------------------------------------------------

class TestToolFixtureReplay:
    def _make_replay(self):
        return ToolFixtureReplay({
            "file_read": {
                "{\"path\": \"/data/config.json\"}": {"content": "hello"}
            }
        })

    def test_exact_match_returns_correct_fixture(self):
        replay = self._make_replay()
        result = replay.call("file_read", {"path": "/data/config.json"})
        assert result == {"content": "hello"}

    def test_fallback_to_first_response_when_no_exact_match(self):
        replay = self._make_replay()
        result = replay.call("file_read", {"path": "/other/path.txt"})
        assert result == {"content": "hello"}

    def test_unknown_tool_returns_default_response(self):
        replay = self._make_replay()
        result = replay.call("file_write", {"path": "/x", "content": "y"})
        assert isinstance(result, dict)

    def test_empty_fixtures_returns_default_response(self):
        replay = ToolFixtureReplay({})
        result = replay.call("any_tool", {})
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# _load_scenarios
# ---------------------------------------------------------------------------

class TestLoadScenarios:
    def test_bundled_scenarios_load_successfully(self):
        scenarios = _load_scenarios()
        assert isinstance(scenarios, list)
        assert len(scenarios) > 0

    def test_scenarios_have_required_keys(self):
        scenarios = _load_scenarios()
        for s in scenarios:
            assert "id" in s
            assert "goal" in s
            assert "tools" in s

    def test_custom_path_loads_correctly(self):
        data = [_SCENARIO_DICT]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            fname = f.name
        loaded = _load_scenarios(fname)
        assert len(loaded) == 1
        assert loaded[0]["id"] == "test_1"
        Path(fname).unlink()


# ---------------------------------------------------------------------------
# AgentBenchConfig
# ---------------------------------------------------------------------------

class TestAgentBenchConfig:
    def test_default_max_turns(self):
        assert AgentBenchConfig().max_turns == 10

    def test_default_limit_is_none(self):
        assert AgentBenchConfig().limit is None


# ---------------------------------------------------------------------------
# AgentBenchRunner
# ---------------------------------------------------------------------------

class TestAgentBenchRunner:
    def _make_runner(self, **kw):
        return AgentBenchRunner(AgentBenchConfig(**kw))

    def _make_scenario_result(self, completed=True, seq_acc=1.0, tokens=50):
        return {
            "scenario_id": "x", "category": "c",
            "completed": completed, "sequence_accuracy": seq_acc,
            "step_efficiency": 1.0, "actual_steps": 1,
            "optimal_steps": 1, "tokens_consumed": tokens,
            "actual_sequence": ["f"],
        }

    def test_track_name_is_agent(self):
        assert self._make_runner().track_name == "agent"

    def test_run_returns_result_record(self):
        runner = self._make_runner()
        runner._run_scenario = MagicMock(return_value=self._make_scenario_result())
        with patch("squish.benchmarks.agent_bench.EngineClient"), \
             patch("squish.benchmarks.agent_bench._load_scenarios", return_value=[_SCENARIO_DICT]):
            result = runner.run(_ENGINE, "qwen3:8b")
        assert isinstance(result, ResultRecord)
        assert result.track == "agent"

    def test_run_computes_completion_rate(self):
        runner = self._make_runner()
        results_seq = [
            self._make_scenario_result(completed=True, tokens=10),
            self._make_scenario_result(completed=False, seq_acc=0.0, tokens=20),
        ]
        idx = [0]
        def side_effect(client, model, scenario):
            r = results_seq[idx[0]]
            idx[0] += 1
            return r
        runner._run_scenario = side_effect
        with patch("squish.benchmarks.agent_bench.EngineClient"), \
             patch("squish.benchmarks.agent_bench._load_scenarios",
                   return_value=[_SCENARIO_DICT, _SCENARIO_DICT]):
            result = runner.run(_ENGINE, "qwen3:8b")
        assert result.metrics["completion_rate"] == pytest.approx(0.5, abs=1e-4)
        assert result.metrics["total_tokens_consumed"] == 30

    def test_run_applies_limit(self):
        runner = self._make_runner()
        runner._run_scenario = MagicMock(return_value=self._make_scenario_result())
        with patch("squish.benchmarks.agent_bench.EngineClient"), \
             patch("squish.benchmarks.agent_bench._load_scenarios",
                   return_value=[_SCENARIO_DICT] * 5):
            runner.run(_ENGINE, "qwen3:8b", limit=2)
        assert runner._run_scenario.call_count == 2

    def test_run_scenario_handles_client_exception(self):
        runner = self._make_runner(max_turns=1)
        mock_client = MagicMock()
        mock_client.chat.side_effect = ConnectionError("no server")
        scenario = AgentScenario.from_dict(_SCENARIO_DICT)
        result = runner._run_scenario(mock_client, "m", scenario)
        assert "scenario_id" in result

    def test_run_scenario_sequence_accuracy_no_tools_called(self):
        """When model produces a plain text answer (no tool calls), sequence_accuracy=0."""
        runner = self._make_runner(max_turns=1)
        mock_client = MagicMock()
        mock_client.chat.return_value = {
            "choices": [{"message": {"content": "I cannot help.", "tool_calls": []}, "finish_reason": "stop"}],
            "usage": {"total_tokens": 20},
        }
        scenario = AgentScenario.from_dict(_SCENARIO_DICT)
        result = runner._run_scenario(mock_client, "m", scenario)
        assert result["sequence_accuracy"] == pytest.approx(0.0, abs=1e-4)

    def test_run_metadata_contains_max_turns(self):
        runner = self._make_runner(max_turns=5)
        runner._run_scenario = MagicMock(return_value=self._make_scenario_result())
        with patch("squish.benchmarks.agent_bench.EngineClient"), \
             patch("squish.benchmarks.agent_bench._load_scenarios", return_value=[_SCENARIO_DICT]):
            result = runner.run(_ENGINE, "qwen3:8b")
        assert result.metadata["max_turns"] == 5
