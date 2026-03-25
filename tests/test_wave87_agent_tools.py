"""tests/test_wave87_agent_tools.py

Wave 87 — Fix VSCode/Web UI Agent Tool Execution

Tests for:
  - parse_tool_calls(): Strategy 0.5 — truncated <tool_call> (no closing tag)
  - _is_tool_call(): accepts "input" key (Claude/Anthropic format)
  - _normalise(): normalizes "input" → "arguments"
  - Round-trip tool name mapping (normalize_for_backend + normalize_for_client)
  - VSCode unprefixed names work with normalize_for_backend
  - normalize_for_client strips squish_ prefix for known tools
  - tool_name_map exports VSCODE_TO_BACKEND and BACKEND_TO_VSCODE
"""
from __future__ import annotations

import os
import sys
import unittest

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)


# ============================================================================
# TestStrategy05 — truncated <tool_call> parsing
# ============================================================================

class TestStrategy05(unittest.TestCase):
    """parse_tool_calls() must handle <tool_call> without closing </tool_call>."""

    def test_truncated_tool_call_parsed(self):
        """Strategy 0.5 must extract JSON when </tool_call> is absent."""
        from squish.serving.tool_calling import parse_tool_calls
        text = '<tool_call>{"name": "read_file", "arguments": {"path": "/tmp/x.txt"}}'
        result = parse_tool_calls(text)
        assert result is not None, "Strategy 0.5 must parse truncated <tool_call>"
        assert len(result) >= 1
        assert result[0]["name"] == "read_file"

    def test_truncated_tool_call_arguments_preserved(self):
        from squish.serving.tool_calling import parse_tool_calls
        text = '<tool_call>{"name": "write_file", "arguments": {"path": "/a.py", "content": "hello"}}'
        result = parse_tool_calls(text)
        assert result is not None
        args = result[0]["arguments"]
        assert args["path"] == "/a.py"
        assert args["content"] == "hello"

    def test_complete_tool_call_still_works(self):
        """Strategy 0 (complete tags) must still operate correctly."""
        from squish.serving.tool_calling import parse_tool_calls
        text = '<tool_call>{"name": "list_directory", "arguments": {"path": "."}}</tool_call>'
        result = parse_tool_calls(text)
        assert result is not None
        assert result[0]["name"] == "list_directory"

    def test_truncated_with_think_block_stripped(self):
        """Think-block stripping must happen before Strategy 0.5."""
        from squish.serving.tool_calling import parse_tool_calls
        text = '<think>thinking...</think>\n<tool_call>{"name": "run_terminal", "arguments": {"cmd": "ls"}}'
        result = parse_tool_calls(text)
        assert result is not None
        assert result[0]["name"] == "run_terminal"

    def test_plain_text_still_returns_none(self):
        """Plain assistant text must return None (no false positives)."""
        from squish.serving.tool_calling import parse_tool_calls
        result = parse_tool_calls("Sure, I'll help you with that.")
        assert result is None


# ============================================================================
# TestIsToolCallInputKey — _is_tool_call() with "input" key
# ============================================================================

class TestIsToolCallInputKey(unittest.TestCase):
    """_is_tool_call() must accept both 'arguments' and 'input' keys."""

    def test_arguments_key_accepted(self):
        from squish.serving.tool_calling import _is_tool_call
        assert _is_tool_call({"name": "read_file", "arguments": {}})

    def test_input_key_accepted(self):
        """Claude-format uses 'input' instead of 'arguments'."""
        from squish.serving.tool_calling import _is_tool_call
        assert _is_tool_call({"name": "read_file", "input": {"path": "/x"}})

    def test_missing_both_keys_rejected(self):
        from squish.serving.tool_calling import _is_tool_call
        assert not _is_tool_call({"name": "read_file"})

    def test_list_with_input_key_accepted(self):
        from squish.serving.tool_calling import _is_tool_call
        calls = [
            {"name": "read_file", "input": {"path": "/a"}},
            {"name": "write_file", "input": {"path": "/b", "content": "x"}},
        ]
        assert _is_tool_call(calls)

    def test_empty_list_rejected(self):
        from squish.serving.tool_calling import _is_tool_call
        assert not _is_tool_call([])


# ============================================================================
# TestNormaliseInputKey — _normalise() normalizes "input" → "arguments"
# ============================================================================

class TestNormaliseInputKey(unittest.TestCase):
    """_normalise() must convert 'input' key to 'arguments'."""

    def test_input_key_renamed_to_arguments(self):
        from squish.serving.tool_calling import _normalise
        obj = {"name": "read_file", "input": {"path": "/x"}}
        result = _normalise(obj)
        assert len(result) == 1
        assert "arguments" in result[0], "input key must be renamed to arguments"
        assert "input" not in result[0], "input key must be removed after rename"

    def test_arguments_key_unchanged(self):
        from squish.serving.tool_calling import _normalise
        obj = {"name": "read_file", "arguments": {"path": "/x"}}
        result = _normalise(obj)
        assert "arguments" in result[0]
        assert result[0]["arguments"]["path"] == "/x"

    def test_list_input_key_normalised(self):
        from squish.serving.tool_calling import _normalise
        objs = [
            {"name": "a", "input": {"x": 1}},
            {"name": "b", "arguments": {"y": 2}},
        ]
        result = _normalise(objs)
        assert "arguments" in result[0]
        assert "arguments" in result[1]
        assert "input" not in result[0]

    def test_parse_tool_calls_input_format(self):
        """End-to-end: Claude-format 'input' key is parsed and normalized."""
        from squish.serving.tool_calling import parse_tool_calls
        text = '<tool_call>{"name": "delete_file", "input": {"path": "/tmp/x"}}</tool_call>'
        result = parse_tool_calls(text)
        assert result is not None
        assert result[0]["name"] == "delete_file"
        assert "arguments" in result[0], "input key must be normalized to arguments"
        assert result[0]["arguments"]["path"] == "/tmp/x"


# ============================================================================
# TestToolNameMap — VSCODE_TO_BACKEND and round-trip normalization
# ============================================================================

class TestToolNameMap(unittest.TestCase):
    """tool_name_map must provide correct bidirectional name mapping."""

    def test_vscode_to_backend_dict_present(self):
        from squish.agent.tool_name_map import VSCODE_TO_BACKEND
        assert isinstance(VSCODE_TO_BACKEND, dict)
        assert len(VSCODE_TO_BACKEND) >= 6

    def test_backend_to_vscode_dict_present(self):
        from squish.agent.tool_name_map import BACKEND_TO_VSCODE
        assert isinstance(BACKEND_TO_VSCODE, dict)

    def test_normalize_for_backend_known_name(self):
        from squish.agent.tool_name_map import normalize_for_backend
        assert normalize_for_backend("create_file") == "squish_create_file"

    def test_normalize_for_backend_unknown_passthrough(self):
        from squish.agent.tool_name_map import normalize_for_backend
        assert normalize_for_backend("custom_tool") == "custom_tool"

    def test_normalize_for_client_known_name(self):
        from squish.agent.tool_name_map import normalize_for_client
        assert normalize_for_client("squish_create_file") == "create_file"

    def test_normalize_for_client_unknown_passthrough(self):
        from squish.agent.tool_name_map import normalize_for_client
        assert normalize_for_client("custom_tool") == "custom_tool"

    def test_round_trip_backend_to_client(self):
        """normalize_for_backend then normalize_for_client must round-trip."""
        from squish.agent.tool_name_map import normalize_for_backend, normalize_for_client
        original = "write_file"
        backend = normalize_for_backend(original)
        assert backend == "squish_write_file"
        restored = normalize_for_client(backend)
        assert restored == original

    def test_all_vscode_tools_have_squish_prefix_in_backend(self):
        from squish.agent.tool_name_map import VSCODE_TO_BACKEND
        for vscode, backend in VSCODE_TO_BACKEND.items():
            assert backend.startswith("squish_"), (
                f"Backend name for {vscode!r} must start with 'squish_', got {backend!r}"
            )

    def test_read_file_mapping_present(self):
        from squish.agent.tool_name_map import normalize_for_backend
        assert normalize_for_backend("read_file") == "squish_read_file"

    def test_list_directory_mapping_present(self):
        from squish.agent.tool_name_map import normalize_for_backend
        assert normalize_for_backend("list_directory") == "squish_list_directory"


if __name__ == "__main__":
    unittest.main()
