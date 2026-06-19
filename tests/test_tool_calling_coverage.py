"""Behavioral coverage for ``squish.serving.tool_calling`` — the tools-prompt
injection, the JSON extraction/repair/normalisation helpers, the streaming
tool-call suppression filter, the multi-strategy parser, the OpenAI response
builder, the SSE streamer, and grammar-assisted parsing.

All pure-Python (json/regex/uuid/async) — no MLX.
"""
from __future__ import annotations

import asyncio
import json

from squish.serving import tool_calling as tc
from squish.serving.tool_calling import (
    ToolCallStreamFilter,
    _coerce_args,
    _extract_json_objects,
    _is_tool_call,
    _normalise,
    _repair_doubled_braces,
    _try_parse,
    build_tool_calls_response,
    format_tools_prompt,
    parse_tool_calls,
    parse_tool_calls_with_grammar,
    stream_tool_calls_response,
)

_TOOL = {"type": "function", "function": {"name": "search", "description": "d",
                                          "parameters": {"type": "object"}}}


# ── format_tools_prompt ─────────────────────────────────────────────────────


def test_format_tools_prompt_no_tools_returns_input():
    msgs = [{"role": "user", "content": "hi"}]
    assert format_tools_prompt(msgs, []) is msgs


def test_format_tools_prompt_merges_into_existing_system():
    msgs = [{"role": "system", "content": "base"}, {"role": "user", "content": "hi"}]
    out = format_tools_prompt(msgs, [_TOOL])
    assert out[0]["role"] == "system"
    assert "base" in out[0]["content"] and "search" in out[0]["content"]
    assert out[1] == {"role": "user", "content": "hi"}


def test_format_tools_prompt_prepends_system_when_absent():
    msgs = [{"role": "user", "content": "hi"}]
    out = format_tools_prompt(msgs, [{"name": "bare_fn", "description": "x"}])
    assert out[0]["role"] == "system" and "bare_fn" in out[0]["content"]
    assert out[1]["content"] == "hi"


# ── _extract_json_objects ───────────────────────────────────────────────────


def test_extract_json_objects_nested_and_strings():
    text = 'noise {"a": {"b": 1}} more ["x", "y"] end'
    found = _extract_json_objects(text)
    assert '{"a": {"b": 1}}' in found
    assert '["x", "y"]' in found


def test_extract_json_objects_ignores_braces_in_strings():
    text = '{"k": "a } b { c"}'  # braces inside the string must not break balance
    assert _extract_json_objects(text) == ['{"k": "a } b { c"}']


def test_extract_json_objects_handles_escaped_quote():
    text = r'{"k": "a \" b"}'  # escaped quote inside string
    assert _extract_json_objects(text) == [r'{"k": "a \" b"}']


# ── _repair_doubled_braces / _try_parse ─────────────────────────────────────


def test_repair_doubled_braces():
    assert _repair_doubled_braces('{{"a": 1}}') == '{"a": 1}'
    assert _repair_doubled_braces('[[1, 2]]') == "[1, 2]"
    assert _repair_doubled_braces('{"a": 1}') is None


def test_try_parse_valid_and_repaired_and_failed():
    assert _try_parse('{"a": 1}') == {"a": 1}
    assert _try_parse('{{"a": 1}}') == {"a": 1}        # via repair
    assert _try_parse('{{not json}}') is None          # repair applies but still invalid
    assert _try_parse("not json at all") is None       # no repair, plain failure


# ── tool-call shape detectors ───────────────────────────────────────────────


def test_is_tool_call_shapes():
    assert _is_tool_call({"name": "f", "arguments": {}}) is True
    assert _is_tool_call({"name": "f", "input": {}}) is True       # Anthropic key
    assert _is_tool_call({"name": "f", "parameters": {}}) is True  # Qwen key
    assert _is_tool_call(["f", {"x": 1}]) is True                  # positional
    assert _is_tool_call([{"name": "a", "arguments": {}}]) is True  # list
    assert _is_tool_call({"name": "f"}) is False                   # no args key
    assert _is_tool_call([]) is False                              # empty list
    assert _is_tool_call("string") is False


# ── _normalise / _coerce_args ───────────────────────────────────────────────


def test_normalise_folds_alt_keys_and_positional():
    assert _normalise({"name": "f", "input": {"a": 1}}) == [{"name": "f", "arguments": {"a": 1}}]
    assert _normalise(["f", {"a": 1}]) == [{"name": "f", "arguments": {"a": 1}}]
    # list with a non-dict element → the bad element is skipped.
    out = _normalise([{"name": "a", "arguments": {}}, 42])
    assert out == [{"name": "a", "arguments": {}}]


def test_coerce_args_variants():
    assert _coerce_args({"a": 1}) == {"a": 1}
    assert _coerce_args('{"a": 1}') == {"a": 1}   # JSON string → dict
    assert _coerce_args("not json") == {}          # unparseable string → {}
    assert _coerce_args(123) == {}                 # non-str/dict → {}


def test_normalise_parameters_key_and_no_args_key():
    # "parameters" alt key is folded into "arguments" (the break path).
    assert _normalise({"name": "f", "parameters": {"a": 1}}) == \
        [{"name": "f", "arguments": {"a": 1}}]
    # No args key at all → the alt loop completes without break → {} default.
    assert _normalise({"name": "f"}) == [{"name": "f", "arguments": {}}]


# ── ToolCallStreamFilter ────────────────────────────────────────────────────


def test_stream_filter_emits_text_then_suppresses():
    f = ToolCallStreamFilter()
    out = "".join(f.feed(t) for t in ["Let me ", "search", "<tool_call>", '{"name"'])
    assert "Let me search" in out
    assert f.suppressing is True
    # Once suppressing, further tokens emit nothing.
    assert f.feed("more") == ""


def test_stream_filter_holds_back_split_marker():
    f = ToolCallStreamFilter()
    # The marker arrives split; the holdback prevents leaking a partial "<tool".
    f.feed("hello <tool")
    # Final flush reveals nothing past the marker since the full marker forms.
    rest = f.feed("_call>", final=False)
    assert "<tool_call>" not in (("hello <tool") + rest)
    assert f.suppressing is True


def test_stream_filter_final_flushes_holdback():
    f = ToolCallStreamFilter()
    out = f.feed("plain text", final=True)
    assert out == "plain text"  # final=True emits the held-back tail


def test_stream_filter_empty_token_is_noop():
    f = ToolCallStreamFilter()
    assert f.feed("abc") == ""              # held back (marker is 11 chars > 3)
    # Empty token skips the buffer append (297→299); final=True flushes the tail.
    assert f.feed("", final=True) == "abc"


# ── parse_tool_calls (strategies) ───────────────────────────────────────────


def test_parse_tool_calls_tag_format():
    text = '<tool_call>\n{"name": "search", "arguments": {"q": "x"}}\n</tool_call>'
    calls = parse_tool_calls(text)
    assert calls == [{"name": "search", "arguments": {"q": "x"}}]


def test_parse_tool_calls_strips_think_block():
    text = '<think>reasoning</think><tool_call>{"name": "f", "arguments": {}}</tool_call>'
    assert parse_tool_calls(text) == [{"name": "f", "arguments": {}}]


def test_parse_tool_calls_open_tag_without_close():
    text = '<tool_call>{"name": "f", "arguments": {"a": 1}}'
    assert parse_tool_calls(text) == [{"name": "f", "arguments": {"a": 1}}]


def test_parse_tool_calls_fenced_json():
    text = 'sure:\n```json\n{"name": "f", "arguments": {}}\n```'
    assert parse_tool_calls(text) == [{"name": "f", "arguments": {}}]


def test_parse_tool_calls_bare_json():
    assert parse_tool_calls('{"name": "f", "arguments": {"a": 1}}') == \
        [{"name": "f", "arguments": {"a": 1}}]


def test_parse_tool_calls_embedded_in_prose():
    text = 'I will call {"name": "f", "arguments": {"a": 1}} now.'
    assert parse_tool_calls(text) == [{"name": "f", "arguments": {"a": 1}}]


def test_parse_tool_calls_plain_text_returns_none():
    assert parse_tool_calls("just a normal answer, no tools") is None


def test_parse_tool_calls_tag_with_invalid_json_falls_through():
    # A tag whose body isn't a tool call → tag strategy yields nothing, and the
    # surrounding text has no other JSON → None.
    assert parse_tool_calls("<tool_call>not json</tool_call>") is None


def test_parse_tool_calls_fenced_non_tool_json_continues():
    # Fenced JSON that isn't a tool call → fenced strategy continues (357→355),
    # and no other strategy matches → None.
    assert parse_tool_calls('```json\n{"x": 1}\n```') is None


def test_parse_tool_calls_embedded_non_tool_json_continues():
    # An extractable JSON object that isn't a tool call → extract loop continues
    # (368→366) and ultimately returns None.
    assert parse_tool_calls('here is data {"x": 1} ok') is None


# ── build_tool_calls_response ───────────────────────────────────────────────


def test_build_tool_calls_response_arg_forms():
    out = build_tool_calls_response([
        {"name": "a", "arguments": {"x": 1}},   # dict → json string
        {"name": "b", "arguments": '{"y": 2}'},  # already-serialised string
        {"name": "c", "arguments": None},        # other → json.dumps(None)
    ])
    assert out[0]["function"]["arguments"] == '{"x": 1}'
    assert out[1]["function"]["arguments"] == '{"y": 2}'
    assert out[2]["function"]["arguments"] == "null"
    assert all(c["type"] == "function" and c["id"].startswith("call_") for c in out)


# ── stream_tool_calls_response (async SSE) ──────────────────────────────────


def _collect(agen):
    async def run():
        return [c async for c in agen]
    return asyncio.run(run())


def test_stream_tool_calls_response_sequence():
    raw = [{"name": "search", "arguments": {"query": "weather today"}}]
    chunks = _collect(stream_tool_calls_response("cid1", "model-x", raw, chunk_size=4))
    assert chunks[-1] == "data: [DONE]\n\n"
    # First chunk is the opening role delta.
    first = json.loads(chunks[0][len("data: "):])
    assert first["choices"][0]["delta"] == {"role": "assistant", "content": None}
    # A start chunk announces the function name with empty arguments.
    bodies = [json.loads(c[len("data: "):]) for c in chunks[1:-1]]
    start = bodies[0]["choices"][0]["delta"]["tool_calls"][0]
    assert start["function"]["name"] == "search" and start["function"]["arguments"] == ""
    # Reassembling the argument deltas yields the full JSON arguments.
    arg_pieces = [
        b["choices"][0]["delta"]["tool_calls"][0]["function"].get("arguments", "")
        for b in bodies[1:]
        if b["choices"][0]["delta"].get("tool_calls")
    ]
    assert json.loads("".join(arg_pieces)) == {"query": "weather today"}
    # The final chunk carries the tool_calls finish reason.
    final = json.loads(chunks[-2][len("data: "):])
    assert final["choices"][0]["finish_reason"] == "tool_calls"


# ── parse_tool_calls_with_grammar ───────────────────────────────────────────


class _Grammar:
    def __init__(self, available):
        self._available = available

    def is_available(self):
        return self._available


def test_grammar_parse_direct_when_available():
    text = '{"name": "f", "arguments": {"a": 1}}'
    assert parse_tool_calls_with_grammar(text, _Grammar(True)) == \
        [{"name": "f", "arguments": {"a": 1}}]


def test_grammar_parse_falls_back_to_heuristic_on_bad_json():
    # Grammar available but output isn't bare JSON → falls back to parse_tool_calls.
    text = 'prefix <tool_call>{"name": "f", "arguments": {}}</tool_call>'
    assert parse_tool_calls_with_grammar(text, _Grammar(True)) == [{"name": "f", "arguments": {}}]


def test_grammar_parse_none_engine_uses_heuristic():
    text = '{"name": "f", "arguments": {}}'
    assert parse_tool_calls_with_grammar(text, None) == [{"name": "f", "arguments": {}}]


def test_grammar_parse_unavailable_engine_uses_heuristic():
    text = '{"name": "f", "arguments": {}}'
    assert parse_tool_calls_with_grammar(text, _Grammar(False)) == [{"name": "f", "arguments": {}}]


def test_grammar_parse_valid_non_tool_json_falls_through():
    # Grammar available, output is valid JSON but not a tool call → the direct
    # parse falls through to the heuristic (551→555), which also finds nothing.
    assert parse_tool_calls_with_grammar('{"x": 1}', _Grammar(True)) is None
