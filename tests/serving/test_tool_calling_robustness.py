"""
tests/serving/test_tool_calling_robustness.py

Regression coverage for the tool-call parser's tolerance of the malformed /
variant JSON shapes real quantized models emit. Without these, the agent loop
(``/v1/agent/run``) and tool-aware ``/v1/chat/completions`` silently fail to
execute any tools.

Observed live with Qwen2.5-7B-Instruct-int4:
  - ``"parameters"`` used in place of ``"arguments"`` (Qwen native template)
  - positional tuple ``["tool_name", {args}]`` (compact quantized form)
  - doubled outer braces ``{{"name": ...}}`` (Qwen2.5 template Jinja artifact)
"""

from __future__ import annotations

from squish.serving.tool_calling import parse_tool_calls


class TestParametersKey:
    def test_parameters_key_normalised_to_arguments(self):
        out = parse_tool_calls('{"name": "squish_list_dir", "parameters": {"path": "."}}')
        assert out == [{"name": "squish_list_dir", "arguments": {"path": "."}}]

    def test_input_key_still_supported(self):
        out = parse_tool_calls('{"name": "f", "input": {"x": 1}}')
        assert out == [{"name": "f", "arguments": {"x": 1}}]


class TestStringEncodedArguments:
    """OpenAI's wire format (and some models) encode arguments as a JSON
    *string*. It must be parsed to a dict so tool execution never crashes on
    ``'str' object has no attribute 'items'``."""

    def test_json_string_arguments_parsed_to_dict(self):
        out = parse_tool_calls(
            '{"name": "squish_read_file", "arguments": "{\\"path\\": \\"/tmp/x\\"}"}'
        )
        assert out == [{"name": "squish_read_file", "arguments": {"path": "/tmp/x"}}]

    def test_non_json_string_arguments_become_empty_dict(self):
        out = parse_tool_calls('{"name": "f", "arguments": "not json"}')
        assert out == [{"name": "f", "arguments": {}}]

    def test_parameters_as_json_string(self):
        out = parse_tool_calls('{"name": "f", "parameters": "{\\"a\\": 1}"}')
        assert out == [{"name": "f", "arguments": {"a": 1}}]


class TestPositionalTuple:
    def test_single_positional_call(self):
        out = parse_tool_calls('["squish_list_dir", {"path": "."}]')
        assert out == [{"name": "squish_list_dir", "arguments": {"path": "."}}]

    def test_positional_inside_tool_call_tag(self):
        out = parse_tool_calls('<tool_call>["f", {"a": 1}]</tool_call>')
        assert out == [{"name": "f", "arguments": {"a": 1}}]

    def test_two_element_non_tool_array_is_not_a_call(self):
        # A plain [str, str] pair must not be mistaken for a positional call.
        assert parse_tool_calls('["hello", "world"]') is None


class TestDoubledBraces:
    def test_doubled_braces_object(self):
        raw = '{{"name": "squish_create_file", "arguments": {"path": "/tmp/x", "content": "hi"}}}'
        assert parse_tool_calls(raw) == [
            {"name": "squish_create_file", "arguments": {"path": "/tmp/x", "content": "hi"}}
        ]

    def test_doubled_braces_inside_tool_call_tag(self):
        raw = '<tool_call>\n{{"name": "squish_run_shell", "arguments": {"command": "ls", "timeout": 30}}}\n</tool_call>'
        assert parse_tool_calls(raw) == [
            {"name": "squish_run_shell", "arguments": {"command": "ls", "timeout": 30}}
        ]

    def test_doubled_braces_open_tag_only(self):
        # Closing tag consumed by the stop sequence (native-template path).
        raw = '<tool_call>\n{{"name": "f", "arguments": {"a": 1}}}'
        assert parse_tool_calls(raw) == [{"name": "f", "arguments": {"a": 1}}]


class TestPlainTextUnaffected:
    def test_prose_is_not_a_tool_call(self):
        assert parse_tool_calls("This project is a local inference server.") is None

    def test_well_formed_call_still_parses(self):
        out = parse_tool_calls('<tool_call>{"name": "f", "arguments": {"a": 1}}</tool_call>')
        assert out == [{"name": "f", "arguments": {"a": 1}}]
