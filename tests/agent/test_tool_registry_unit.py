"""Unit tests for ToolRegistry — registration, validation, and dispatch contract."""
from __future__ import annotations

import pytest

from squish.agent.tool_registry import (
    ToolCallError,
    ToolDefinition,
    ToolRegistry,
    ToolResult,
)


def _def(name="echo", **kw):
    params = kw.pop("parameters", {
        "type": "object",
        "properties": {"x": {"type": "string"}},
        "required": ["x"],
    })
    fn = kw.pop("fn", lambda x: x)
    return ToolDefinition(name=name, description=kw.pop("description", "d"),
                          parameters=params, fn=fn, **kw)


class TestToolDefinition:
    def test_to_openai_schema(self):
        schema = _def().to_openai_schema()
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "echo"
        assert schema["function"]["parameters"]["required"] == ["x"]

    def test_default_source_is_user(self):
        assert _def().source == "user"


class TestToolResult:
    def test_ok_true_when_no_error(self):
        assert ToolResult("t", "id", output="hi").ok is True
        assert ToolResult("t", "id", output=None, error="boom").ok is False

    def test_to_message_string_output(self):
        msg = ToolResult("t", "c1", output="hello").to_message()
        assert msg == {"role": "tool", "tool_call_id": "c1", "content": "hello"}

    def test_to_message_serializes_non_string(self):
        msg = ToolResult("t", "c1", output={"a": 1}).to_message()
        assert msg["content"] == '{"a": 1}'

    def test_to_message_error_takes_precedence(self):
        msg = ToolResult("t", "c1", output="ignored", error="bad").to_message()
        assert msg["content"] == "[ERROR] bad"


class TestRegistration:
    def test_register_and_get(self):
        r = ToolRegistry()
        d = _def()
        r.register(d)
        assert r.get("echo") is d
        assert "echo" in r
        assert len(r) == 1

    def test_duplicate_register_raises(self):
        r = ToolRegistry()
        r.register(_def())
        with pytest.raises(ValueError, match="already registered"):
            r.register(_def())

    def test_decorator_defaults_name_and_docstring(self):
        r = ToolRegistry()

        @r.tool()
        def my_tool(a):
            """Docstring description."""
            return a

        assert my_tool("x") == "x"            # returns unmodified fn
        defn = r.get("my_tool")
        assert defn is not None
        assert defn.description == "Docstring description."
        # Default params when none supplied.
        assert defn.parameters == {"type": "object", "properties": {}, "required": []}

    def test_decorator_name_override_and_source(self):
        r = ToolRegistry()

        @r.tool(name="renamed", description="x", source="mcp:srv")
        def f():
            return 1

        assert "renamed" in r and "f" not in r
        assert r.get("renamed").source == "mcp:srv"

    def test_unregister_and_clear(self):
        r = ToolRegistry()
        r.register(_def("a"))
        r.register(_def("b"))
        r.unregister("a")
        r.unregister("missing")   # no-op, must not raise
        assert r.names() == ["b"]
        r.clear()
        assert len(r) == 0

    def test_names_sorted_and_openai_schemas(self):
        r = ToolRegistry()
        r.register(_def("zebra"))
        r.register(_def("alpha"))
        assert r.names() == ["alpha", "zebra"]
        assert len(r.to_openai_schemas()) == 2


class TestValidateCall:
    def _reg(self):
        r = ToolRegistry()
        r.register(_def("t", parameters={
            "type": "object",
            "properties": {
                "s": {"type": "string"},
                "n": {"type": "number"},
                "mode": {"type": "string", "enum": ["fast", "slow"]},
            },
            "required": ["s"],
        }))
        return r

    def test_unknown_tool(self):
        with pytest.raises(ToolCallError, match="Unknown tool"):
            ToolRegistry().validate_call("nope", {})

    def test_missing_required(self):
        with pytest.raises(ToolCallError, match="missing required"):
            self._reg().validate_call("t", {"n": 1})

    def test_type_mismatch(self):
        with pytest.raises(ToolCallError, match="expected string"):
            self._reg().validate_call("t", {"s": 123})

    def test_number_accepts_int_and_float(self):
        self._reg().validate_call("t", {"s": "ok", "n": 3})
        self._reg().validate_call("t", {"s": "ok", "n": 3.5})

    def test_enum_violation(self):
        with pytest.raises(ToolCallError, match="not in allowed enum"):
            self._reg().validate_call("t", {"s": "ok", "mode": "turbo"})

    def test_valid_passes(self):
        self._reg().validate_call("t", {"s": "ok", "n": 1, "mode": "fast"})

    def test_property_without_type_skips_type_check(self):
        # A property whose schema declares no "type" must not be type-checked.
        r = ToolRegistry()
        r.register(_def("t", parameters={
            "type": "object",
            "properties": {"anything": {}},   # no "type" key
            "required": [],
        }))
        r.validate_call("t", {"anything": 12345})  # must not raise

    def test_unknown_type_is_ignored(self):
        r = ToolRegistry()
        r.register(_def("t", parameters={
            "type": "object",
            "properties": {"v": {"type": "weirdtype"}},
            "required": [],
        }))
        r.validate_call("t", {"v": object()})  # unknown type → no enforcement


class TestCall:
    def test_success_sets_output_and_timing(self):
        r = ToolRegistry()
        r.register(_def("add", parameters={
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
        }, fn=lambda a, b: a + b))
        res = r.call("add", {"a": 2, "b": 3})
        assert res.ok and res.output == 5
        assert res.elapsed_ms >= 0.0
        assert res.call_id.startswith("add_")

    def test_custom_call_id(self):
        r = ToolRegistry()
        r.register(_def("echo", fn=lambda x: x))
        assert r.call("echo", {"x": "hi"}, call_id="abc").call_id == "abc"

    def test_validation_failure_returns_error_result(self):
        res = self_reg().call("t", {})  # missing required "s"
        assert not res.ok and "missing required" in res.error

    def test_unknown_tool_returns_error_result(self):
        res = ToolRegistry().call("ghost", {})
        assert not res.ok and "Unknown tool" in res.error

    def test_unknown_tool_with_validate_false_still_errors(self):
        # Skips validation, so the post-lookup `defn is None` guard handles it.
        res = ToolRegistry().call("ghost", {}, validate=False)
        assert not res.ok and "Unknown tool" in res.error and res.output is None

    def test_tool_exception_is_captured(self):
        r = ToolRegistry()

        def boom(x):
            raise RuntimeError("kaboom")

        r.register(_def("boom", parameters={
            "type": "object", "properties": {"x": {"type": "string"}}, "required": [],
        }, fn=boom))
        res = r.call("boom", {"x": "v"})
        assert not res.ok
        assert "RuntimeError: kaboom" in res.error
        assert res.output is None

    def test_validate_false_skips_validation(self):
        r = ToolRegistry()
        r.register(_def("echo", fn=lambda x="default": x))
        # Missing required "x" would fail validation, but validate=False bypasses
        # and the fn supplies its own default.
        res = r.call("echo", {}, validate=False)
        assert res.ok and res.output == "default"


def self_reg():
    r = ToolRegistry()
    r.register(_def("t", parameters={
        "type": "object",
        "properties": {"s": {"type": "string"}},
        "required": ["s"],
    }))
    return r
