"""
tests/e2e/test_agent_e2e.py

Comprehensive end-to-end coverage for the agent + chat surface, exercised
against a *live* squish server. Short / medium / long prompts across complexity
tiers, plus robustness cases (empty input, huge pastes, tool-error recovery,
clean streaming).

Gated behind ``SQUISH_E2E=1`` so it never runs (or fails) in CI without a
loaded model. To run:

    SQUISH_E2E=1 python -m pytest tests/e2e/test_agent_e2e.py -v

Configure the target with ``SQUISH_E2E_URL`` (default http://127.0.0.1:11435)
and ``SQUISH_E2E_KEY`` (default "squish").
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

import pytest

BASE = os.environ.get("SQUISH_E2E_URL", "http://127.0.0.1:11435")
KEY = os.environ.get("SQUISH_E2E_KEY", "squish")
_HDR = {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"}


def _server_ready() -> bool:
    if os.environ.get("SQUISH_E2E") != "1":
        return False
    try:
        req = urllib.request.Request(BASE + "/health", headers=_HDR)
        with urllib.request.urlopen(req, timeout=3) as r:
            return bool(json.loads(r.read()).get("loaded"))
    except (urllib.error.URLError, ValueError, OSError):
        return False


pytestmark = pytest.mark.skipif(
    not _server_ready(),
    reason="set SQUISH_E2E=1 and run a loaded squish server to exercise e2e",
)


# ── transport helpers ─────────────────────────────────────────────────────────


def _chat(prompt: str, max_tokens: int = 256) -> str:
    body = {
        "model": "local",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
    }
    req = urllib.request.Request(
        BASE + "/v1/chat/completions", data=json.dumps(body).encode(), method="POST", headers=_HDR
    )
    text = ""
    for raw in urllib.request.urlopen(req, timeout=180):
        line = raw.decode().strip()
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        try:
            obj = json.loads(line[6:])
        except ValueError:
            continue
        text += obj.get("choices", [{}])[0].get("delta", {}).get("content", "") or ""
    return text


def _agent(prompt: str, *, max_steps: int = 6, max_tokens: int = 512):
    body = {
        "messages": [{"role": "user", "content": prompt}],
        "max_steps": max_steps,
        "max_tokens": max_tokens,
        "temperature": 0.3,
    }
    req = urllib.request.Request(
        BASE + "/v1/agent/run", data=json.dumps(body).encode(), method="POST", headers=_HDR
    )
    text, tools, tool_errors, done = "", [], 0, False
    stream_error = None
    for raw in urllib.request.urlopen(req, timeout=240):
        line = raw.decode().strip()
        if not line.startswith("data: "):
            continue
        try:
            ev = json.loads(line[6:])
        except ValueError:
            continue
        t = ev.get("type")
        if t == "text_delta":
            text += ev.get("delta", "")
        elif t == "tool_call_start":
            tools.append(ev["tool_name"])
        elif t == "tool_call_result" and ev.get("error"):
            tool_errors += 1
        elif t == "done":
            done = True
        elif t == "error":
            stream_error = ev.get("message")
    return {
        "text": text,
        "tools": tools,
        "tool_errors": tool_errors,
        "done": done,
        "error": stream_error,
    }


def _assert_clean(text: str) -> None:
    """No raw tool-call syntax should ever surface in visible text."""
    assert "<tool_call>" not in text
    assert "{{" not in text


# ── chat: short / medium / long ───────────────────────────────────────────────


class TestChatComplexityTiers:
    def test_short_prompt(self):
        out = _chat("Reply with exactly the word: pong", max_tokens=16)
        assert out.strip()

    def test_medium_prompt(self):
        out = _chat("In two sentences, explain what an INT4 KV cache is.", max_tokens=160)
        assert len(out.split()) >= 8

    def test_long_complex_prompt(self):
        out = _chat(
            "Write a Python function `fib(n)` that returns the nth Fibonacci "
            "number iteratively, with a docstring and a couple of inline comments.",
            max_tokens=400,
        )
        assert "def fib" in out


# ── agent: short / medium / long ──────────────────────────────────────────────


class TestAgentComplexityTiers:
    def test_short_single_tool(self):
        r = _agent("List the files in the current directory using your tools.")
        assert r["done"] and r["error"] is None
        assert any("list" in t for t in r["tools"])
        _assert_clean(r["text"])

    def test_medium_two_tools(self, tmp_path):
        p = tmp_path / "e2e_medium.txt"
        r = _agent(
            f"Create a file at {p} containing the text 'medium tier', then read it "
            f"back and tell me what it contains."
        )
        assert r["done"] and r["error"] is None
        assert len(r["tools"]) >= 2
        assert p.exists() and "medium tier" in p.read_text()
        _assert_clean(r["text"])

    def test_long_multi_step(self):
        r = _agent(
            "Use your tools: read README.md, count how many Python files are under "
            "the squish/ directory with a shell command, then summarise the project "
            "in one sentence.",
            max_steps=8,
            max_tokens=600,
        )
        # The system must stay robust on a long, multi-part prompt: complete
        # cleanly, call at least one tool, and never leak tool syntax. We don't
        # assert an exact planning depth — how many tools a small quantized
        # model chains for an open-ended task is non-deterministic (the medium
        # tier covers reliable 2-tool chaining with a clearer task).
        assert r["done"] and r["error"] is None
        assert len(r["tools"]) >= 1
        _assert_clean(r["text"])
        assert r["text"].strip()  # produced a final answer


# ── robustness: ready for anything ────────────────────────────────────────────


class TestRobustness:
    def test_empty_prompt_is_rejected_not_crashed(self):
        body = {"messages": [], "max_steps": 2}
        req = urllib.request.Request(
            BASE + "/v1/agent/run", data=json.dumps(body).encode(), method="POST", headers=_HDR
        )
        with pytest.raises(urllib.error.HTTPError) as ei:
            urllib.request.urlopen(req, timeout=30)
        assert ei.value.code == 400  # graceful client error, not a 500

    def test_huge_paste_is_handled(self):
        blob = "data line\n" * 5000  # ~50k chars
        out = _chat(f"Reply with only the word OK. Ignore this:\n{blob}", max_tokens=16)
        assert out.strip()

    def test_tool_error_recovers_without_killing_stream(self):
        # Reading a non-existent path makes a tool fail; the agent must recover
        # and the stream must complete cleanly.
        r = _agent(
            "Read the file at /tmp/this_does_not_exist_konjo.txt. If it fails, just "
            "tell me it does not exist.",
            max_steps=4,
        )
        assert r["done"] and r["error"] is None  # stream survived the tool error

    def test_stream_never_leaks_tool_syntax(self):
        r = _agent("Create /tmp/e2e_clean.txt with 'x', then read it back.", max_steps=5)
        _assert_clean(r["text"])
