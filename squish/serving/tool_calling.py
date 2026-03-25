#!/usr/bin/env python3
"""
squish/tool_calling.py

OpenAI-compatible function/tool calling support for Squish.

Provides:
    format_tools_prompt(messages, tools) → str
        Injects a JSON‑schema tool manifest into the system prompt so that
        instruction-tuned models (Qwen 2.5, Llama 3.1+, Mistral) know what
        functions are available.

    parse_tool_calls(text) → list[dict] | None
        Heuristically extracts tool-call JSON blocks from model output.
        Returns None if no tool calls are found (regular text reply).

    build_tool_calls_response(tool_calls) → dict
        Formats the tool_calls list for inclusion in an OpenAI ChatCompletion
        response.

Design
──────
The approach follows Qwen2.5 and Hermes-style "tool use" prompts:

    System message injection:
        "You have access to the following tools:
        [JSON schema list]
        When you want to call a tool, reply ONLY with a JSON object on a
        single line in one of these forms:
        { \"name\": \"<tool_name>\", \"arguments\": { ... } }
        You may call multiple tools by returning a JSON array."

    Output parsing:
        Look for a JSON blob (```json...``` fenced block, or a bare object/
        array) that contains  {"name": ..., "arguments": ...}  patterns.

Models that natively support tool calling (they know the schema from training)
will generally comply without any special parsing — the JSON injection just
reinforces behaviour for models that haven't seen a tool-calling system prompt.
"""

import json
import re
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

# ── Prompt formatting ──────────────────────────────────────────────────────────

_TOOL_SYSTEM_PREFIX = """\
You have access to the following tools. To use a tool, write ONLY a JSON \
object (no surrounding text) that matches one of the schemas below. You may \
call multiple tools by returning a JSON array of objects.

Tool schemas:
{schemas}

Rules:
- If you need to call a tool, respond with ONLY the JSON object or array.
- If no tool call is needed, respond normally in plain text.
- Do not mix prose and a tool call in the same message."""


def format_tools_prompt(messages: list[dict], tools: list[dict]) -> list[dict]:
    """
    Return a copy of `messages` with a tools-manifest injected into the
    system prompt (or prepended as a new system message if none exists).

    Parameters
    ----------
    messages : OpenAI message list  [{role, content}, ...]
    tools    : OpenAI tools list    [{type, function:{name, description, parameters}}, ...]

    Returns
    -------
    list[dict]   — messages with tool schema injected
    """
    if not tools:
        return messages

    # Build compact schema strings
    schemas_parts: list[str] = []
    for t in tools:
        fn = t.get("function", t)  # handle both {type, function} and bare fn dict
        name    = fn.get("name", "unknown")
        desc    = fn.get("description", "")
        params  = fn.get("parameters", {})
        schemas_parts.append(
            f"  {json.dumps({'name': name, 'description': desc, 'parameters': params})}"
        )
    schemas_str = "\n".join(schemas_parts)
    tool_system = _TOOL_SYSTEM_PREFIX.format(schemas=schemas_str)

    msgs = list(messages)  # shallow copy
    # Find existing system message
    for i, m in enumerate(msgs):
        if m.get("role") == "system":
            existing = m.get("content", "")
            msgs[i] = {**m, "content": f"{tool_system}\n\n{existing}".strip()}
            return msgs
    # No system message — prepend one
    msgs.insert(0, {"role": "system", "content": tool_system})
    return msgs


# ── Output parsing ─────────────────────────────────────────────────────────────

# Patterns to extract JSON from model output
_FENCED_JSON  = re.compile(r"```(?:json)?\s*\n?([\s\S]*?)```", re.MULTILINE)
# Qwen3 / Hermes native tool-call tags
_TOOL_CALL_TAG = re.compile(r"<tool_call>\s*([\s\S]*?)\s*</tool_call>", re.MULTILINE)
# Strategy 0.5 — opening tag only (closing tag consumed by stop string)
_OPEN_TOOL_CALL_TAG = re.compile(r"<tool_call>\s*([\s\S]+)$", re.MULTILINE)
# Think-block stripper (Qwen3 reasoning traces before the tool call)
_THINK_BLOCK   = re.compile(r"<think>[\s\S]*?</think>", re.MULTILINE)


def _extract_json_objects(text: str) -> list[str]:
    """
    Walk `text` character-by-character finding balanced JSON objects/arrays.
    Returns a list of candidate JSON substrings (handles nested braces).
    """
    candidates = []
    for opener, closer in (('{', '}'), ('[', ']')):
        depth = 0
        start = None
        in_string = False
        escape    = False
        for i, ch in enumerate(text):
            if escape:
                escape = False
                continue
            if ch == '\\' and in_string:
                escape = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == opener:
                if depth == 0:
                    start = i
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0 and start is not None:
                    candidates.append(text[start:i+1])
                    start = None
    return candidates


def _try_parse(text: str) -> Any | None:
    """Try to parse `text` as JSON, return None on failure."""
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        return None


def _is_tool_call(obj: Any) -> bool:
    """Return True if `obj` looks like a tool call or list of tool calls.

    Accepts both ``"arguments"`` (OpenAI/Qwen format) and ``"input"``
    (Claude/Anthropic tool-use format) as the arguments key.
    """
    def _single(x: Any) -> bool:
        return isinstance(x, dict) and "name" in x and ("arguments" in x or "input" in x)
    if isinstance(obj, dict):
        return _single(obj)
    if isinstance(obj, list):
        return all(_single(x) for x in obj) and len(obj) > 0
    return False


def _normalise(obj: Any) -> list[dict]:
    """Wrap single tool call in a list and normalize ``"input"`` → ``"arguments"``."""
    items = [obj] if isinstance(obj, dict) else list(obj)
    for item in items:
        if isinstance(item, dict) and "input" in item and "arguments" not in item:
            item["arguments"] = item.pop("input")
    return items


def parse_tool_calls(text: str) -> list[dict] | None:
    """
    Scan model output for tool-call JSON.

    Returns a list of raw tool-call dicts ({"name": ..., "arguments": ...})
    or None if no valid tool call is found (meaning the reply is plain text).

    Handles:
    - Qwen3 / Hermes ``<tool_call>...</tool_call>`` XML-tag format
    - Fenced JSON blocks (```json...```)
    - Full-text JSON (model output is only JSON)
    - Balanced JSON objects/arrays embedded in prose
    - ``<think>`` blocks are stripped before parsing
    """
    # Strip think blocks so they don't confuse the JSON extractor
    stripped = _THINK_BLOCK.sub("", text).strip()

    # 0. Qwen3 / Hermes <tool_call> tag (highest confidence for those models)
    tag_matches = _TOOL_CALL_TAG.findall(stripped)
    if tag_matches:
        calls: list[dict] = []
        for raw in tag_matches:
            obj = _try_parse(raw)
            if obj is not None and _is_tool_call(obj):
                calls.extend(_normalise(obj))
        if calls:
            return calls

    # 0.5. Opening <tool_call> tag without closing tag (consumed by stop string)
    open_match = _OPEN_TOOL_CALL_TAG.search(stripped)
    if open_match:
        obj = _try_parse(open_match.group(1).strip())
        if obj is not None and _is_tool_call(obj):
            return _normalise(obj)

    # 1. Try fenced JSON blocks
    for m in _FENCED_JSON.finditer(stripped):
        obj = _try_parse(m.group(1))
        if obj is not None and _is_tool_call(obj):
            return _normalise(obj)

    # 2. Try the stripped text as bare JSON (model output is only a JSON blob)
    obj = _try_parse(stripped)
    if obj is not None and _is_tool_call(obj):
        return _normalise(obj)

    # 3. Extract balanced JSON objects/arrays from surrounding prose
    for candidate in _extract_json_objects(stripped):
        obj = _try_parse(candidate)
        if obj is not None and _is_tool_call(obj):
            return _normalise(obj)

    return None


# ── Response building ──────────────────────────────────────────────────────────

def build_tool_calls_response(raw_calls: list[dict]) -> list[dict]:
    """
    Convert raw tool-call dicts into the OpenAI `tool_calls` list format.

    Each item becomes:
    {
        "id":       "call_<hex>",
        "type":     "function",
        "function": {
            "name":      "<name>",
            "arguments": "<JSON string>"
        }
    }

    Parameters
    ----------
    raw_calls : list returned by parse_tool_calls()

    Returns
    -------
    list[dict]  in OpenAI tool_calls format
    """
    result = []
    for raw in raw_calls:
        name = raw.get("name", "unknown")
        args = raw.get("arguments", {})
        # arguments must be a JSON string in OpenAI format
        if isinstance(args, dict):
            args_str = json.dumps(args)
        elif isinstance(args, str):
            args_str = args  # already serialised
        else:
            args_str = json.dumps(args)
        result.append({
            "id":       f"call_{uuid.uuid4().hex[:12]}",
            "type":     "function",
            "function": {
                "name":      name,
                "arguments": args_str,
            },
        })
    return result


async def stream_tool_calls_response(
    cid: str,
    model_id: str,
    raw_calls: list[dict],
    chunk_size: int = 8,
) -> AsyncIterator[str]:
    """
    Yield SSE ``data:`` lines that replay a tool-call response in the OpenAI
    streaming (``chat.completion.chunk``) format.

    The function generates tool calls in full non-streaming mode internally,
    then emits the result as a structured sequence of delta chunks so that
    streaming-aware agent frameworks (LangChain, pydantic-ai, llama-index …)
    receive the proper ``delta.tool_calls`` events.

    Chunk sequence per tool call
    ----------------------------
    1. Opening role chunk: ``delta = {"role": "assistant", "content": null}``
    2. Per tool call — start chunk: announces ``name`` and ``id``; ``arguments`` is ``""``
    3. Per tool call — argument chunks: ``arguments`` text in ``chunk_size``-character pieces
    4. Final chunk: ``delta = {}``, ``finish_reason = "tool_calls"``
    5. ``data: [DONE]``

    Parameters
    ----------
    cid : str
        Chat completion ID to embed in every chunk.
    model_id : str
        Model name string to embed in every chunk.
    raw_calls : list[dict]
        Parsed tool calls as returned by :func:`parse_tool_calls`.
    chunk_size : int
        Number of argument characters per streaming chunk (default 8).
    """

    def _make_chunk(delta: dict, finish_reason: Any = None) -> str:
        payload = {
            "id":      cid,
            "object":  "chat.completion.chunk",
            "created": int(time.time()),
            "model":   model_id,
            "choices": [{
                "index":         0,
                "delta":         delta,
                "finish_reason": finish_reason,
            }],
        }
        return f"data: {json.dumps(payload)}\n\n"

    tc_list = build_tool_calls_response(raw_calls)

    # 1. Opening role chunk
    yield _make_chunk({"role": "assistant", "content": None})

    for i, tc in enumerate(tc_list):
        # 2. Tool call start: announce id, name, empty arguments
        yield _make_chunk({
            "tool_calls": [{
                "index":    i,
                "id":       tc["id"],
                "type":     "function",
                "function": {
                    "name":      tc["function"]["name"],
                    "arguments": "",
                },
            }],
        })
        # 3. Stream arguments in small chunks
        args_str = tc["function"]["arguments"]
        for j in range(0, len(args_str), chunk_size):
            yield _make_chunk({
                "tool_calls": [{
                    "index":    i,
                    "function": {
                        "arguments": args_str[j : j + chunk_size],
                    },
                }],
            })

    # 4. Final stop chunk
    yield _make_chunk({}, finish_reason="tool_calls")
    # 5. SSE terminator
    yield "data: [DONE]\n\n"


# ── Grammar-assisted parsing ───────────────────────────────────────────────────

def parse_tool_calls_with_grammar(
    text: str,
    grammar_engine: Any | None = None,
) -> list[dict] | None:
    """
    Parse tool calls from *text*, leveraging the grammar engine when available.

    When *grammar_engine* is provided and grammar-constrained sampling was used,
    the model output should already be valid JSON.  This function attempts a
    direct ``json.loads`` parse first (trusting the grammar guarantee) before
    falling back to the heuristic :func:`parse_tool_calls` approach.

    Parameters
    ----------
    text:
        Raw model output string.
    grammar_engine:
        A ``GrammarEngine`` instance, or ``None`` to use heuristic parsing only.

    Returns
    -------
    list[dict] | None
        Same semantics as :func:`parse_tool_calls`: a list of raw tool-call
        dicts ``{"name": ..., "arguments": ...}``, or ``None`` if no valid
        tool call is found.
    """
    if grammar_engine is not None and grammar_engine.is_available():
        # Grammar-constrained output should be well-formed JSON; try direct parse.
        try:
            obj = json.loads(text.strip())
            if _is_tool_call(obj):
                return _normalise(obj)
        except json.JSONDecodeError:
            pass
    return parse_tool_calls(text)
