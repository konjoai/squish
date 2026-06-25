"""tests/test_wave129_robust_tools.py

Wave 129 — robust tool-call handling for small model families + tools opt-in.

A 1B model in (auto-enabled) agent mode produced: a leaked ``<|python_tag|>``
tool call in the chat bubble, a duplicate call, off-topic degeneration, and a
hard ``Tool ... argument 'timeout': expected integer, got str`` because it
passed ``"10"`` (a string) for an integer param.

This pins the fixes:
- tool_registry: scalar args are coerced to the declared JSON-schema type, so
  ``"10" -> 10`` runs instead of erroring; genuinely wrong types still raise.
- tool_calling: Llama ``<|python_tag|>`` and Mistral ``[TOOL_CALLS]`` envelopes
  parse; ``ToolCallStreamFilter`` suppresses those (and bare-JSON) calls so they
  never reach a chat bubble; ``TOOL_CALL_STOPS`` covers the family end markers.
- web UI: agent/tool mode is opt-in (no auto-enable on startup).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from squish.agent.tool_registry import ToolRegistry, _coerce_arg
from squish.serving import tool_calling as tc

_INT_SCHEMA = {
    "type": "object",
    "properties": {
        "code": {"type": "string"},
        "timeout": {"type": "integer"},
        "max_memory_mb": {"type": "integer"},
    },
    "required": ["code"],
}


def _registry():
    reg = ToolRegistry()

    @reg.tool(name="repl", description="run", parameters=_INT_SCHEMA)
    def repl(code: str, timeout: int = 10, max_memory_mb: int = 512) -> str:
        # Assert the function actually receives ints, not strings.
        assert isinstance(timeout, int) and isinstance(max_memory_mb, int)
        return f"{code}:{timeout}:{max_memory_mb}"

    return reg


# ──────────────────────────────────────────────────────────────────────────────
# 1.  Argument coercion
# ──────────────────────────────────────────────────────────────────────────────


class TestCoerceArg:
    def test_string_to_int(self):
        assert _coerce_arg("10", "integer") == 10
        assert _coerce_arg("512", "integer") == 512

    def test_integral_float_to_int(self):
        assert _coerce_arg(10.0, "integer") == 10
        assert _coerce_arg("10.0", "integer") == 10

    def test_string_to_number(self):
        assert _coerce_arg("3.5", "number") == 3.5

    def test_string_to_bool(self):
        assert _coerce_arg("true", "boolean") is True
        assert _coerce_arg("False", "boolean") is False
        assert _coerce_arg("1", "boolean") is True

    def test_json_string_to_array_object(self):
        assert _coerce_arg('["a", "b"]', "array") == ["a", "b"]
        assert _coerce_arg('{"k": 1}', "object") == {"k": 1}

    def test_unrecoverable_left_unchanged(self):
        # Non-numeric string for an int stays a string so validate raises clearly.
        assert _coerce_arg("abc", "integer") == "abc"

    def test_bool_not_treated_as_int(self):
        # bool is an int subclass; coercion must not turn True into 1 for an int.
        assert _coerce_arg(True, "integer") is True


class TestRegistryCoercionEndToEnd:
    def test_string_args_run_after_coercion(self):
        # The exact failing call: timeout/max_memory_mb passed as strings.
        res = _registry().call("repl", {"code": "x", "timeout": "10", "max_memory_mb": "512"})
        assert res.ok, res.error
        assert res.output == "x:10:512"

    def test_arguments_dict_mutated_in_place(self):
        reg = _registry()
        args = {"code": "x", "timeout": "10"}
        reg.call("repl", args)
        assert args["timeout"] == 10  # coerced in place

    def test_genuinely_wrong_type_still_errors(self):
        res = _registry().call("repl", {"code": "x", "timeout": "not-a-number"})
        assert not res.ok
        assert "timeout" in res.error and "integer" in res.error


# ──────────────────────────────────────────────────────────────────────────────
# 2.  Tool-call parsing across model families
# ──────────────────────────────────────────────────────────────────────────────


class TestParseAcrossFamilies:
    def test_llama_python_tag(self):
        # The user's exact leaked shape: marker + {"type","name","parameters"} +
        # eom marker + trailing junk.
        out = (
            '<|python_tag|>{"type": "function", "name": "squish_python_repl", '
            '"parameters": {"code": "x", "timeout": "10"}}<|eom_id|> blah blah'
        )
        calls = tc.parse_tool_calls(out)
        assert calls and calls[0]["name"] == "squish_python_repl"
        assert calls[0]["arguments"]["timeout"] == "10"

    def test_mistral_tool_calls(self):
        calls = tc.parse_tool_calls('[TOOL_CALLS][{"name": "f", "arguments": {"x": 1}}]')
        assert calls and calls[0]["name"] == "f"

    def test_qwen_hermes_still_works(self):
        calls = tc.parse_tool_calls('<tool_call>{"name": "g", "arguments": {}}</tool_call>')
        assert calls and calls[0]["name"] == "g"

    def test_plain_prose_is_not_a_tool_call(self):
        assert tc.parse_tool_calls("The KV cache stores past keys and values.") is None


# ──────────────────────────────────────────────────────────────────────────────
# 3.  Stream filter suppresses the raw call across families
# ──────────────────────────────────────────────────────────────────────────────


class TestStreamFilter:
    def _run(self, tokens):
        f = tc.ToolCallStreamFilter()
        vis = "".join(f.feed(t) for t in tokens)
        return vis, f.suppressing

    def test_suppresses_python_tag(self):
        vis, sup = self._run(["Sure! ", "<|python", "_tag|>", '{"name":"f"}'])
        assert vis == "Sure! " and sup is True

    def test_suppresses_tool_calls_marker(self):
        vis, sup = self._run(["ok ", "[TOOL_CALLS]", '[{"name":"f"}]'])
        assert vis == "ok " and sup is True

    def test_suppresses_bare_json_call(self):
        vis, sup = self._run(['{"name": ', '"f", "arguments": {}}'])
        assert sup is True
        assert '"f"' not in vis  # the call body never leaks

    def test_qwen_tag_still_suppressed(self):
        vis, sup = self._run(["hi ", "<tool_call>", '{"name":"f"}'])
        assert vis == "hi " and sup is True

    def test_plain_text_passes_through(self):
        vis, sup = self._run(["The ", "KV ", "cache ", "is great."])
        f = tc.ToolCallStreamFilter()
        final = "".join(f.feed(t) for t in ["The ", "KV ", "cache ", "is great."]) + f.feed(
            "", final=True
        )
        assert "KV cache is great." in final and sup is False


# ──────────────────────────────────────────────────────────────────────────────
# 4.  Stop strings + web-UI opt-in
# ──────────────────────────────────────────────────────────────────────────────


class TestStopsAndOptIn:
    def test_tool_call_stops_cover_families(self):
        assert "</tool_call>" in tc.TOOL_CALL_STOPS  # Qwen/Hermes
        assert "<|eom_id|>" in tc.TOOL_CALL_STOPS  # Llama tool turn
        assert "<|eot_id|>" in tc.TOOL_CALL_STOPS  # Llama end of turn

    def test_webui_does_not_auto_enable_agent_mode(self):
        html = (ROOT / "squish" / "static" / "index.html").read_text(encoding="utf-8")
        assert "let agentMode    = false;" in html or "agentMode = false" in html
        # The startup auto-enable (toggleAgentMode when tools exist) must be gone.
        assert "(td.tools || []).length > 0 && !agentMode" not in html


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__, "-q"]))
