"""tests/grammar/test_tool_choice_unit.py

Phase 15B — test_tool_choice_unit.py

Unit tests for tool_choice enforcement in squish/server.py.
Tests exercise _build_tool_union_schema() and the surrounding
tool_choice dispatch logic via direct imports (no live server).

Coverage targets
────────────────
_build_tool_union_schema
  - tools with named functions → schema enum includes all names
  - tools with no names → schema uses plain string (no enum)
  - tools with some names missing → only valid names in enum
  - empty tools list → valid base schema, no enum
  - single tool → name in enum
  - schema has required type:object
  - schema has required ["name"]
  - schema has parameters:object property

tool_choice dispatch logic (extracted helpers)
  - tool_choice "none" disables tools (tools list becomes empty)
  - tool_choice "auto" leaves tools unchanged
  - tool_choice "required" selects union schema
  - tool_choice {"type":"function","function":{"name":"X"}} selects named tool schema
  - named tool_choice with unknown function name → no schema (None)
  - named tool_choice with matching function → schema is function.parameters
"""
from __future__ import annotations

import json
import sys

import pytest

# ---------------------------------------------------------------------------
# Import _build_tool_union_schema from squish.server.
# squish.server calls _require("fastapi") at module level; when fastapi is
# absent _require() exits unless "--help" is in sys.argv.  We temporarily
# inject "--help" so the module loads in environments without fastapi.
# ---------------------------------------------------------------------------
_saved_argv = sys.argv[:]
sys.argv = ["squish.server", "--help"]
try:
    from squish.server import _build_tool_union_schema
finally:
    sys.argv = _saved_argv


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _tool(name: str, **extra_params) -> dict:
    """Build a minimal tool object."""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": f"{name} function",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                },
                "required": ["query"],
                **extra_params,
            },
        },
    }


# ---------------------------------------------------------------------------
# _build_tool_union_schema — schema structure
# ---------------------------------------------------------------------------


class TestBuildToolUnionSchemaStructure:
    def test_result_is_dict(self):
        schema = _build_tool_union_schema([_tool("search")])
        assert isinstance(schema, dict)

    def test_type_is_object(self):
        schema = _build_tool_union_schema([_tool("search")])
        assert schema["type"] == "object"

    def test_properties_has_name(self):
        schema = _build_tool_union_schema([_tool("search")])
        assert "name" in schema["properties"]

    def test_properties_has_parameters(self):
        schema = _build_tool_union_schema([_tool("search")])
        assert "parameters" in schema["properties"]

    def test_parameters_property_is_object_type(self):
        schema = _build_tool_union_schema([_tool("search")])
        assert schema["properties"]["parameters"]["type"] == "object"

    def test_required_includes_name(self):
        schema = _build_tool_union_schema([_tool("search")])
        assert "name" in schema["required"]

    def test_serialises_to_json(self):
        schema = _build_tool_union_schema([_tool("search"), _tool("write")])
        json_str = json.dumps(schema)
        assert json.loads(json_str) == schema


# ---------------------------------------------------------------------------
# _build_tool_union_schema — name enumeration
# ---------------------------------------------------------------------------


class TestBuildToolUnionSchemaNames:
    def test_single_tool_name_in_enum(self):
        schema = _build_tool_union_schema([_tool("search")])
        name_prop = schema["properties"]["name"]
        assert name_prop.get("enum") == ["search"]

    def test_multiple_tools_all_names_in_enum(self):
        tools = [_tool("search"), _tool("write"), _tool("compute")]
        schema = _build_tool_union_schema(tools)
        enum_vals = schema["properties"]["name"].get("enum", [])
        assert set(enum_vals) == {"search", "write", "compute"}

    def test_empty_tools_no_enum(self):
        schema = _build_tool_union_schema([])
        name_prop = schema["properties"]["name"]
        # With no tools, enum should be absent or empty; type must still be string
        assert name_prop.get("type") == "string"
        assert "enum" not in name_prop or name_prop["enum"] == []

    def test_tool_missing_name_excluded(self):
        tools = [
            _tool("good_tool"),
            {"type": "function", "function": {"description": "no name here", "parameters": {}}},
        ]
        schema = _build_tool_union_schema(tools)
        enum_vals = schema["properties"]["name"].get("enum", [])
        assert "good_tool" in enum_vals
        # Tool without name should NOT produce an empty-string enum entry
        assert "" not in enum_vals

    def test_tool_with_empty_name_excluded(self):
        tools = [_tool("valid"), {"type": "function", "function": {"name": "", "parameters": {}}}]
        schema = _build_tool_union_schema(tools)
        enum_vals = schema["properties"]["name"].get("enum", [])
        assert "valid" in enum_vals
        assert "" not in enum_vals


# ---------------------------------------------------------------------------
# tool_choice dispatch logic — extracted from server.py request handler
# ---------------------------------------------------------------------------


def _simulate_tool_choice_dispatch(tool_choice, tools):
    """
    Replicate the server.py tool_choice dispatch logic for unit testing.

    Returns (effective_tools, tc_schema) where tc_schema is the grammar
    enforcement schema (None if not enforced).
    """
    # Replicate: tool_choice == "none" disables tools
    if tool_choice == "none":
        tools = []

    tc_schema = None
    if tools:
        if tool_choice == "required":
            tc_schema = _build_tool_union_schema(tools)
        elif isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
            forced_name = tool_choice.get("function", {}).get("name", "")
            match = next(
                (t for t in tools if t.get("function", {}).get("name") == forced_name),
                None,
            )
            if match:
                tc_schema = match.get("function", {}).get("parameters") or {}
    return tools, tc_schema


class TestToolChoiceDispatch:
    def setup_method(self):
        self.tools = [_tool("search"), _tool("write")]

    def test_none_disables_all_tools(self):
        effective_tools, tc_schema = _simulate_tool_choice_dispatch("none", self.tools)
        assert effective_tools == []
        assert tc_schema is None

    def test_auto_keeps_tools_unchanged(self):
        effective_tools, tc_schema = _simulate_tool_choice_dispatch("auto", self.tools)
        assert len(effective_tools) == 2
        assert tc_schema is None

    def test_required_builds_union_schema(self):
        _, tc_schema = _simulate_tool_choice_dispatch("required", self.tools)
        assert tc_schema is not None
        assert tc_schema["type"] == "object"
        enum_vals = tc_schema["properties"]["name"].get("enum", [])
        assert "search" in enum_vals and "write" in enum_vals

    def test_function_choice_named_tool_found(self):
        tc = {"type": "function", "function": {"name": "search"}}
        _, tc_schema = _simulate_tool_choice_dispatch(tc, self.tools)
        # Schema should be the parameters of the "search" tool
        assert tc_schema is not None
        assert tc_schema.get("type") == "object"

    def test_function_choice_unknown_name_no_schema(self):
        tc = {"type": "function", "function": {"name": "nonexistent"}}
        _, tc_schema = _simulate_tool_choice_dispatch(tc, self.tools)
        assert tc_schema is None

    def test_function_choice_empty_tools_no_schema(self):
        tc = {"type": "function", "function": {"name": "search"}}
        _, tc_schema = _simulate_tool_choice_dispatch(tc, [])
        assert tc_schema is None

    def test_required_empty_tools_no_schema(self):
        _, tc_schema = _simulate_tool_choice_dispatch("required", [])
        assert tc_schema is None

    def test_function_choice_parameters_preserved(self):
        """The tc_schema for a named tool must include the correct parameters."""
        tc = {"type": "function", "function": {"name": "write"}}
        _, tc_schema = _simulate_tool_choice_dispatch(tc, self.tools)
        assert tc_schema is not None
        assert "properties" in tc_schema

    def test_required_with_single_tool(self):
        single = [_tool("only_tool")]
        _, tc_schema = _simulate_tool_choice_dispatch("required", single)
        assert tc_schema is not None
        enum_vals = tc_schema["properties"]["name"].get("enum", [])
        assert enum_vals == ["only_tool"]
