#!/usr/bin/env python3
"""
tests/serving/test_tool_choice_unit.py

Unit tests for the tool_choice resolution logic in squish/server.py (Phase 15).

Because the tool_choice dispatch lives inline inside the /v1/chat/completions
endpoint handler rather than as a standalone function, this module takes two
complementary approaches:

  1. Import the pure helper ``_build_tool_union_schema`` directly from
     squish.server and test it in isolation.

  2. Define a local ``_resolve_tool_choice`` mirror that faithfully reproduces
     the same branch logic so every path can be exercised without an HTTP
     request, a loaded model, or a grammar engine.

Coverage targets
----------------
_resolve_tool_choice (local mirror of the server.py inline logic)
  - "auto"  leaves tools unchanged, returns no schema
  - "none"  clears the tools list, returns no schema
  - "required"  returns a union schema covering all tool names
  - dict {"type":"function","function":{"name":"X"}} returns the matching
    tool's parameters schema
  - dict where the named tool is NOT present → no schema (None)
  - Empty tools list with "required" → resolution short-circuits, no schema
  - Invalid / unknown string value → treated as "auto" (no schema)
  - Empty function name in tool_choice dict → no match, no schema
  - tool_choice dict missing the "function" key → no match, no schema
  - tool_choice dict with wrong "type" field → no match, no schema

_build_tool_union_schema
  - names are populated from the tools list
  - tools with no name key are excluded from the enum
  - empty tools list produces an unconstrained string name (no enum)
  - the "required" field always contains "name"
  - the "parameters" property always has type "object"
  - the top-level schema type is "object"
"""
from __future__ import annotations

from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Import pure helper from squish.server; skip file if unavailable
# ---------------------------------------------------------------------------
try:
    from squish.server import _build_tool_union_schema
except ImportError:  # pragma: no cover
    pytest.skip(
        "squish.server not importable in this environment",
        allow_module_level=True,
    )


# ---------------------------------------------------------------------------
# Local mirror of the inline tool_choice resolution logic (server.py ~L1832)
# ---------------------------------------------------------------------------

def _resolve_tool_choice(
    tool_choice: Any,
    tools: list[dict],
) -> tuple[list[dict], dict | None]:
    """
    Mirror of the server.py inline tool_choice handling found in the
    /v1/chat/completions endpoint.

    Returns
    -------
    (effective_tools, tc_schema)
        effective_tools : the tools list after any "none" clearing
        tc_schema       : JSON schema to enforce via grammar, or None when no
                          grammar enforcement is needed
    """
    effective_tools: list[dict] = list(tools)

    # "none": the agent explicitly disables tool use for this turn
    if tool_choice == "none":
        return [], None

    # Grammar enforcement is only meaningful when tools are present
    if not effective_tools:
        return effective_tools, None

    tc_schema: dict | None = None

    if tool_choice == "required":
        tc_schema = _build_tool_union_schema(effective_tools)
    elif isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
        forced_name: str = tool_choice.get("function", {}).get("name", "")
        match = next(
            (
                t for t in effective_tools
                if t.get("function", {}).get("name") == forced_name
            ),
            None,
        )
        if match:
            tc_schema = match.get("function", {}).get("parameters") or {}

    return effective_tools, tc_schema


# ---------------------------------------------------------------------------
# Shared test fixtures
# ---------------------------------------------------------------------------

_TOOL_WEATHER = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Return the current weather for a location.",
        "parameters": {
            "type": "object",
            "properties": {"location": {"type": "string"}},
            "required": ["location"],
        },
    },
}

_TOOL_SEARCH = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
        },
    },
}

_TOOL_NO_NAME = {
    "type": "function",
    "function": {
        "description": "A nameless tool — should be excluded from enums.",
        "parameters": {"type": "object"},
    },
}

_TOOL_EMPTY_PARAMS = {
    "type": "function",
    "function": {
        "name": "ping",
        "description": "No-op ping.",
        # no "parameters" key
    },
}

_TWO_TOOLS = [_TOOL_WEATHER, _TOOL_SEARCH]


# ---------------------------------------------------------------------------
# Tests: tool_choice="auto"
# ---------------------------------------------------------------------------

class TestToolChoiceAuto:
    """tool_choice="auto" is the OpenAI default: the model may or may not call
    a tool.  No grammar enforcement schema should be produced."""

    def test_auto_does_not_clear_tools_list(self):
        effective, _ = _resolve_tool_choice("auto", _TWO_TOOLS)
        assert len(effective) == 2

    def test_auto_preserves_original_tool_objects(self):
        effective, _ = _resolve_tool_choice("auto", _TWO_TOOLS)
        assert effective[0] is _TOOL_WEATHER
        assert effective[1] is _TOOL_SEARCH

    def test_auto_returns_no_schema(self):
        _, schema = _resolve_tool_choice("auto", _TWO_TOOLS)
        assert schema is None

    def test_auto_with_single_tool_returns_no_schema(self):
        _, schema = _resolve_tool_choice("auto", [_TOOL_WEATHER])
        assert schema is None

    def test_auto_with_empty_tools_returns_no_schema(self):
        effective, schema = _resolve_tool_choice("auto", [])
        assert effective == []
        assert schema is None


# ---------------------------------------------------------------------------
# Tests: tool_choice="none"
# ---------------------------------------------------------------------------

class TestToolChoiceNone:
    """tool_choice="none" instructs the server to disable tools entirely for
    this turn, regardless of what is in the tools list."""

    def test_none_clears_full_tools_list(self):
        effective, _ = _resolve_tool_choice("none", _TWO_TOOLS)
        assert effective == []

    def test_none_returns_no_schema(self):
        _, schema = _resolve_tool_choice("none", _TWO_TOOLS)
        assert schema is None

    def test_none_on_already_empty_list_is_safe(self):
        effective, schema = _resolve_tool_choice("none", [])
        assert effective == []
        assert schema is None

    def test_none_does_not_mutate_original_list(self):
        original = list(_TWO_TOOLS)
        _resolve_tool_choice("none", original)
        assert len(original) == 2  # caller's list must be untouched


# ---------------------------------------------------------------------------
# Tests: tool_choice="required"
# ---------------------------------------------------------------------------

class TestToolChoiceRequired:
    """tool_choice="required" forces the model to output a syntactically valid
    tool call JSON object selected from the provided tools."""

    def test_required_returns_a_schema(self):
        _, schema = _resolve_tool_choice("required", _TWO_TOOLS)
        assert schema is not None

    def test_required_schema_top_level_type_is_object(self):
        _, schema = _resolve_tool_choice("required", _TWO_TOOLS)
        assert schema["type"] == "object"

    def test_required_schema_includes_all_tool_names_in_enum(self):
        _, schema = _resolve_tool_choice("required", _TWO_TOOLS)
        enum = schema["properties"]["name"]["enum"]
        assert "get_weather" in enum
        assert "web_search" in enum

    def test_required_schema_name_is_in_required_array(self):
        _, schema = _resolve_tool_choice("required", _TWO_TOOLS)
        assert "name" in schema["required"]

    def test_required_schema_has_parameters_object_property(self):
        _, schema = _resolve_tool_choice("required", _TWO_TOOLS)
        assert schema["properties"]["parameters"]["type"] == "object"

    def test_required_with_empty_tools_returns_no_schema(self):
        """Empty tools list causes early return before schema building."""
        _, schema = _resolve_tool_choice("required", [])
        assert schema is None

    def test_required_does_not_alter_effective_tools(self):
        effective, _ = _resolve_tool_choice("required", _TWO_TOOLS)
        assert len(effective) == 2


# ---------------------------------------------------------------------------
# Tests: tool_choice as dict with type="function"
# ---------------------------------------------------------------------------

class TestToolChoiceFunctionDict:
    """tool_choice={"type":"function","function":{"name":"X"}} forces the
    model to call exactly the named function, constrained by its parameter
    schema."""

    def test_matching_name_returns_a_schema(self):
        tc = {"type": "function", "function": {"name": "get_weather"}}
        _, schema = _resolve_tool_choice(tc, _TWO_TOOLS)
        assert schema is not None

    def test_matching_name_returns_tool_parameters_schema(self):
        tc = {"type": "function", "function": {"name": "get_weather"}}
        _, schema = _resolve_tool_choice(tc, _TWO_TOOLS)
        assert schema == _TOOL_WEATHER["function"]["parameters"]

    def test_second_tool_matching_returns_its_parameters(self):
        tc = {"type": "function", "function": {"name": "web_search"}}
        _, schema = _resolve_tool_choice(tc, _TWO_TOOLS)
        assert schema == _TOOL_SEARCH["function"]["parameters"]

    def test_non_matching_name_returns_no_schema(self):
        tc = {"type": "function", "function": {"name": "nonexistent_tool"}}
        _, schema = _resolve_tool_choice(tc, _TWO_TOOLS)
        assert schema is None

    def test_empty_function_name_returns_no_schema(self):
        tc = {"type": "function", "function": {"name": ""}}
        _, schema = _resolve_tool_choice(tc, _TWO_TOOLS)
        assert schema is None

    def test_missing_function_key_returns_no_schema(self):
        tc = {"type": "function"}
        _, schema = _resolve_tool_choice(tc, _TWO_TOOLS)
        assert schema is None

    def test_wrong_type_field_returns_no_schema(self):
        tc = {"type": "not_function", "function": {"name": "get_weather"}}
        _, schema = _resolve_tool_choice(tc, _TWO_TOOLS)
        assert schema is None

    def test_tool_with_no_parameters_returns_empty_dict_schema(self):
        """A tool whose function dict has no 'parameters' key yields {}."""
        tc = {"type": "function", "function": {"name": "ping"}}
        _, schema = _resolve_tool_choice(tc, [_TOOL_EMPTY_PARAMS])
        assert schema == {}

    def test_dict_tool_choice_does_not_clear_tools_list(self):
        tc = {"type": "function", "function": {"name": "get_weather"}}
        effective, _ = _resolve_tool_choice(tc, _TWO_TOOLS)
        assert len(effective) == 2


# ---------------------------------------------------------------------------
# Tests: invalid / unknown tool_choice values
# ---------------------------------------------------------------------------

class TestToolChoiceInvalidValues:
    """Any value that is not "none", "required", or a recognized dict should
    be treated permissively (no schema, tools unchanged) rather than raising."""

    def test_unknown_string_returns_no_schema(self):
        _, schema = _resolve_tool_choice("invalid_value", _TWO_TOOLS)
        assert schema is None

    def test_integer_tool_choice_returns_no_schema(self):
        _, schema = _resolve_tool_choice(42, _TWO_TOOLS)
        assert schema is None

    def test_python_none_tool_choice_returns_no_schema(self):
        """Python None (distinct from the string "none") must not crash."""
        _, schema = _resolve_tool_choice(None, _TWO_TOOLS)
        assert schema is None

    def test_list_tool_choice_returns_no_schema(self):
        _, schema = _resolve_tool_choice([], _TWO_TOOLS)
        assert schema is None


# ---------------------------------------------------------------------------
# Tests: _build_tool_union_schema (imported directly)
# ---------------------------------------------------------------------------

class TestBuildToolUnionSchema:
    """Direct unit tests for the pure _build_tool_union_schema helper."""

    def test_schema_top_level_type_is_object(self):
        schema = _build_tool_union_schema(_TWO_TOOLS)
        assert schema["type"] == "object"

    def test_names_populated_in_enum(self):
        schema = _build_tool_union_schema(_TWO_TOOLS)
        assert schema["properties"]["name"]["enum"] == ["get_weather", "web_search"]

    def test_empty_tools_list_produces_unconstrained_string_name(self):
        schema = _build_tool_union_schema([])
        name_prop = schema["properties"]["name"]
        assert name_prop == {"type": "string"}
        assert "enum" not in name_prop

    def test_tool_with_no_name_excluded_from_enum(self):
        schema = _build_tool_union_schema([_TOOL_NO_NAME, _TOOL_WEATHER])
        enum = schema["properties"]["name"]["enum"]
        assert "get_weather" in enum
        assert len(enum) == 1

    def test_required_array_contains_name(self):
        schema = _build_tool_union_schema(_TWO_TOOLS)
        assert "name" in schema["required"]

    def test_parameters_property_is_object_type(self):
        schema = _build_tool_union_schema(_TWO_TOOLS)
        assert schema["properties"]["parameters"]["type"] == "object"

    def test_single_tool_enum_has_one_entry(self):
        schema = _build_tool_union_schema([_TOOL_WEATHER])
        assert schema["properties"]["name"]["enum"] == ["get_weather"]

    def test_all_nameless_tools_produces_unconstrained_string(self):
        schema = _build_tool_union_schema([_TOOL_NO_NAME])
        name_prop = schema["properties"]["name"]
        assert name_prop == {"type": "string"}
