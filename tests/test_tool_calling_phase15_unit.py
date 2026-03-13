#!/usr/bin/env python3
"""
tests/test_tool_calling_phase15_unit.py

Unit tests for Phase 15A: stream_tool_calls_response() in squish/tool_calling.py.

Coverage targets
────────────────
stream_tool_calls_response
  - yields opening role chunk first
  - role chunk delta contains role=assistant and content=null
  - yields tool call start chunk with correct id, type, name, empty arguments
  - yields argument streaming chunks (content split by chunk_size)
  - all argument chunks for a tool call reconstitute the full arguments string
  - yields final chunk with finish_reason="tool_calls" and empty delta
  - last yielded line is "data: [DONE]\n\n"
  - multiple tool calls: index increments correctly
  - empty arguments string: no argument chunks emitted (only start + final)
  - chunk_size=1: one chunk per argument character
  - all yielded strings are valid "data: <json>\n\n" format
"""
from __future__ import annotations

import asyncio
import json

import pytest

from squish.tool_calling import stream_tool_calls_response


# ---------------------------------------------------------------------------
# Async helpers
# ---------------------------------------------------------------------------

def _collect(raw_calls, chunk_size=8):
    """Collect all chunks from stream_tool_calls_response synchronously."""

    async def _gather():
        chunks = []
        async for chunk in stream_tool_calls_response("cid-test", "model-x",
                                                       raw_calls, chunk_size=chunk_size):
            chunks.append(chunk)
        return chunks

    return asyncio.run(_gather())


def _parse_sse(chunk: str) -> dict:
    """Parse a single 'data: <json>\n\n' SSE line to dict."""
    assert chunk.startswith("data: "), f"expected 'data: ...' got: {repr(chunk)}"
    payload = chunk.removeprefix("data: ").strip()
    return json.loads(payload)


_SINGLE_CALL = [{"name": "get_weather", "arguments": {"location": "Paris"}}]
_MULTI_CALLS = [
    {"name": "search",    "arguments": {"q": "hello"}},
    {"name": "get_price", "arguments": {"sku": "ABC123"}},
]
_EMPTY_ARGS  = [{"name": "ping", "arguments": ""}]  # pre-serialised empty string → no arg chunks


# ---------------------------------------------------------------------------
# Structure tests
# ---------------------------------------------------------------------------

class TestStreamToolCallsResponseStructure:
    def test_first_chunk_is_role(self):
        chunks = _collect(_SINGLE_CALL)
        obj = _parse_sse(chunks[0])
        delta = obj["choices"][0]["delta"]
        assert delta.get("role") == "assistant"
        assert delta.get("content") is None

    def test_last_chunk_is_done(self):
        chunks = _collect(_SINGLE_CALL)
        assert chunks[-1] == "data: [DONE]\n\n"

    def test_second_to_last_chunk_has_tool_calls_finish_reason(self):
        chunks = _collect(_SINGLE_CALL)
        obj = _parse_sse(chunks[-2])
        choice = obj["choices"][0]
        assert choice["finish_reason"] == "tool_calls"
        assert choice["delta"] == {}

    def test_all_chunks_are_valid_sse(self):
        chunks = _collect(_SINGLE_CALL)
        for chunk in chunks:
            if chunk == "data: [DONE]\n\n":
                continue
            obj = _parse_sse(chunk)
            assert "choices" in obj
            assert obj["id"] == "cid-test"
            assert obj["model"] == "model-x"

    def test_chunk_contains_created_field(self):
        chunks = _collect(_SINGLE_CALL)
        obj = _parse_sse(chunks[0])
        assert "created" in obj
        assert isinstance(obj["created"], int)


# ---------------------------------------------------------------------------
# Tool call start chunk
# ---------------------------------------------------------------------------

class TestToolCallStartChunk:
    def test_start_chunk_has_id_type_name_empty_args(self):
        chunks = _collect(_SINGLE_CALL)
        # start chunk comes right after the role chunk (index 1)
        obj = _parse_sse(chunks[1])
        tc = obj["choices"][0]["delta"]["tool_calls"][0]
        assert tc["index"] == 0
        assert tc["type"] == "function"
        assert "id" in tc and tc["id"].startswith("call_")
        assert tc["function"]["name"] == "get_weather"
        assert tc["function"]["arguments"] == ""

    def test_start_chunk_index_zero(self):
        chunks = _collect(_SINGLE_CALL)
        obj = _parse_sse(chunks[1])
        tc = obj["choices"][0]["delta"]["tool_calls"][0]
        assert tc["index"] == 0


# ---------------------------------------------------------------------------
# Argument chunks
# ---------------------------------------------------------------------------

class TestArgumentChunks:
    def test_argument_chunks_reconstitute_full_arguments(self):
        chunks = _collect(_SINGLE_CALL, chunk_size=4)
        # Collect all argument text from chunks that have tool_calls with no name key
        from squish.tool_calling import build_tool_calls_response
        expected_args = build_tool_calls_response(_SINGLE_CALL)[0]["function"]["arguments"]

        arg_chunks = []
        for chunk in chunks:
            if chunk == "data: [DONE]\n\n":
                continue
            obj = _parse_sse(chunk)
            delta = obj["choices"][0]["delta"]
            if "tool_calls" in delta:
                tc = delta["tool_calls"][0]
                fn = tc.get("function", {})
                if "arguments" in fn and "name" not in fn:
                    arg_chunks.append(fn["arguments"])

        assert "".join(arg_chunks) == expected_args

    def test_chunk_size_1_produces_one_char_per_chunk(self):
        chunks = _collect(_SINGLE_CALL, chunk_size=1)
        from squish.tool_calling import build_tool_calls_response
        expected_args = build_tool_calls_response(_SINGLE_CALL)[0]["function"]["arguments"]

        arg_chunks = []
        for chunk in chunks:
            if chunk == "data: [DONE]\n\n":
                continue
            obj = _parse_sse(chunk)
            delta = obj["choices"][0]["delta"]
            if "tool_calls" in delta:
                fn = delta["tool_calls"][0].get("function", {})
                if "arguments" in fn and "name" not in fn:
                    arg_chunks.append(fn["arguments"])

        for ac in arg_chunks:
            assert len(ac) == 1
        assert "".join(arg_chunks) == expected_args

    def test_empty_arguments_produces_no_arg_chunks(self):
        chunks = _collect(_EMPTY_ARGS)
        arg_only_chunks = []
        for chunk in chunks:
            if chunk == "data: [DONE]\n\n":
                continue
            obj = _parse_sse(chunk)
            delta = obj["choices"][0]["delta"]
            if "tool_calls" in delta:
                tc = delta["tool_calls"][0]
                fn = tc.get("function", {})
                if "arguments" in fn and "name" not in fn:
                    arg_only_chunks.append(fn["arguments"])
        # No argument-only chunks for empty args
        assert arg_only_chunks == []


# ---------------------------------------------------------------------------
# Multiple tool calls
# ---------------------------------------------------------------------------

class TestMultipleToolCalls:
    def test_indices_increment(self):
        chunks = _collect(_MULTI_CALLS)
        start_indices = []
        for chunk in chunks:
            if chunk == "data: [DONE]\n\n":
                continue
            obj = _parse_sse(chunk)
            delta = obj["choices"][0]["delta"]
            if "tool_calls" in delta:
                tc = delta["tool_calls"][0]
                if "name" in tc.get("function", {}):
                    start_indices.append(tc["index"])
        assert start_indices == [0, 1]

    def test_all_names_present(self):
        chunks = _collect(_MULTI_CALLS)
        names = []
        for chunk in chunks:
            if chunk == "data: [DONE]\n\n":
                continue
            obj = _parse_sse(chunk)
            delta = obj["choices"][0]["delta"]
            if "tool_calls" in delta:
                fn = delta["tool_calls"][0].get("function", {})
                if "name" in fn:
                    names.append(fn["name"])
        assert "search" in names
        assert "get_price" in names

    def test_argument_chunks_tagged_with_correct_index(self):
        chunks = _collect(_MULTI_CALLS, chunk_size=4)
        for chunk in chunks:
            if chunk == "data: [DONE]\n\n":
                continue
            obj = _parse_sse(chunk)
            delta = obj["choices"][0]["delta"]
            if "tool_calls" in delta:
                tc = delta["tool_calls"][0]
                assert tc["index"] in (0, 1)
