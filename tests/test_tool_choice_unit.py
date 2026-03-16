"""Unit tests for tool_choice handling in squish/server.py.

Covers:
  - _build_tool_union_schema (pure function, tests 1-7)
  - tool_choice logic simulation (tests 8-11)
"""

import pytest

from squish.server import _build_tool_union_schema

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tool(name: str, parameters: dict | None = None) -> dict:
    """Return a minimal OpenAI-style tool dict."""
    fn: dict = {"name": name}
    if parameters is not None:
        fn["parameters"] = parameters
    return {"type": "function", "function": fn}


# ---------------------------------------------------------------------------
# Tests 1-7: _build_tool_union_schema (pure function)
# ---------------------------------------------------------------------------

class TestBuildToolUnionSchema:

    def test_single_tool_name_in_enum(self):
        """Single tool produces an enum containing exactly that tool's name."""
        tools = [_make_tool("my_tool")]
        schema = _build_tool_union_schema(tools)
        name_prop = schema["properties"]["name"]
        assert "enum" in name_prop, "Expected 'enum' key in name property"
        assert name_prop["enum"] == ["my_tool"]

    def test_multiple_tools_all_names_in_enum(self):
        """Multiple tools include every name in the enum, in order."""
        tools = [_make_tool("alpha"), _make_tool("beta"), _make_tool("gamma")]
        schema = _build_tool_union_schema(tools)
        name_prop = schema["properties"]["name"]
        assert "enum" in name_prop
        assert name_prop["enum"] == ["alpha", "beta", "gamma"]

    def test_zero_tools_returns_plain_string_name(self):
        """Empty tools list produces a plain string type (no enum) for 'name'."""
        schema = _build_tool_union_schema([])
        name_prop = schema["properties"]["name"]
        assert "enum" not in name_prop, "No enum expected for zero tools"
        assert name_prop == {"type": "string"}

    def test_schema_always_has_required_name(self):
        """'required' always contains 'name', regardless of tool count."""
        for tools in ([], [_make_tool("x")], [_make_tool("a"), _make_tool("b")]):
            schema = _build_tool_union_schema(tools)
            assert "required" in schema
            assert "name" in schema["required"]

    def test_schema_always_has_parameters_as_object(self):
        """'parameters' property is always present with type 'object'."""
        for tools in ([], [_make_tool("x")]):
            schema = _build_tool_union_schema(tools)
            params_prop = schema["properties"]["parameters"]
            assert params_prop == {"type": "object"}

    def test_ignores_tools_without_function_key(self):
        """Tools missing the 'function' key are silently skipped."""
        tools = [
            {"type": "function"},              # no 'function' key at all
            _make_tool("valid_tool"),
        ]
        schema = _build_tool_union_schema(tools)
        name_prop = schema["properties"]["name"]
        assert "enum" in name_prop
        assert name_prop["enum"] == ["valid_tool"]

    def test_ignores_tools_with_empty_function_name(self):
        """Tools whose function name is an empty string are silently skipped."""
        tools = [
            {"type": "function", "function": {"name": ""}},  # empty name
            _make_tool("real_tool"),
        ]
        schema = _build_tool_union_schema(tools)
        name_prop = schema["properties"]["name"]
        assert "enum" in name_prop
        assert name_prop["enum"] == ["real_tool"]


# ---------------------------------------------------------------------------
# Tests 8-11: tool_choice logic (simulated inline, matching server.py logic)
# ---------------------------------------------------------------------------

class TestToolChoiceLogic:

    # Replicated from server.py lines 1838-1908 (pure Python, no HTTP)
    @staticmethod
    def _apply_tool_choice(tools: list[dict], tool_choice: object) -> tuple[list[dict], "dict | None"]:
        """Simulate the tool_choice branching from chat_completions handler.

        Returns (effective_tools, _tc_schema).
        """
        # tool_choice == "none": agent explicitly disables tools for this turn
        if tool_choice == "none":
            tools = []

        _tc_schema: dict | None = None

        if tools:  # only evaluate choice when tools are present
            if tool_choice == "required":
                _tc_schema = _build_tool_union_schema(tools)
            elif isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
                _forced_name = tool_choice.get("function", {}).get("name", "")
                _match = next(
                    (t for t in tools if t.get("function", {}).get("name") == _forced_name),
                    None,
                )
                if _match:
                    _tc_schema = _match.get("function", {}).get("parameters") or {}

        return tools, _tc_schema

    def test_tool_choice_none_empties_tools(self):
        """tool_choice='none' must produce an empty tools list."""
        tools = [_make_tool("fn_a"), _make_tool("fn_b")]
        effective_tools, _tc_schema = self._apply_tool_choice(tools, "none")
        assert effective_tools == [], "tools should be emptied when tool_choice='none'"
        assert _tc_schema is None

    def test_tool_choice_required_returns_union_schema(self):
        """tool_choice='required' must return the union schema from _build_tool_union_schema."""
        tools = [_make_tool("search"), _make_tool("calculator")]
        _, _tc_schema = self._apply_tool_choice(tools, "required")
        expected = _build_tool_union_schema(tools)
        assert _tc_schema == expected

    def test_named_tool_choice_returns_matching_parameters(self):
        """Named tool_choice selects the parameters schema of the matching function."""
        params = {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        }
        tools = [
            _make_tool("unrelated"),
            _make_tool("known_fn", parameters=params),
        ]
        tool_choice = {"type": "function", "function": {"name": "known_fn"}}
        _, _tc_schema = self._apply_tool_choice(tools, tool_choice)
        assert _tc_schema == params

    def test_named_tool_choice_unknown_function_yields_none(self):
        """Named tool_choice with an unknown function name must produce _tc_schema=None."""
        tools = [_make_tool("fn_a"), _make_tool("fn_b")]
        tool_choice = {"type": "function", "function": {"name": "nonexistent_fn"}}
        _, _tc_schema = self._apply_tool_choice(tools, tool_choice)
        assert _tc_schema is None, "_tc_schema must be None for unknown function name"
