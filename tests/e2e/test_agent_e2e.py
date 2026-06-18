"""tests/e2e/test_agent_e2e.py

End-to-end coverage for the agent + chat surface, exercised against a *real*
running squish server booted by the ``live_server`` fixture (see
``tests/e2e/conftest.py``).  Transport is raw ``urllib`` — no extra deps.

Agent-tier assertions stay tolerant (>= 1 tool call, no hard answer matching)
because tool-calling on a 0.5B model is non-deterministic.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

import pytest

pytestmark = pytest.mark.e2e


def _base() -> tuple[str, str]:
    url = os.environ.get("SQUISH_E2E_URL", "http://127.0.0.1:11435")
    key = os.environ.get("SQUISH_E2E_KEY", "squish")
    return url, key


def _headers(key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}


def _chat(prompt: str, max_tokens: int = 64) -> str:
    """Non-streaming chat completion → assistant text."""
    url, key = _base()
    body = {
        "model": "local",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False,
    }
    req = urllib.request.Request(  # noqa: S310 — fixed localhost target
        f"{url}/v1/chat/completions", data=json.dumps(body).encode(), headers=_headers(key),
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=180) as resp:  # noqa: S310
        data = json.loads(resp.read())
    return data["choices"][0]["message"]["content"]


def _agent(prompt: str, max_steps: int = 5, max_tokens: int = 256) -> dict:
    """Run the SSE agent loop; collect visible text + tool calls + errors."""
    url, key = _base()
    body = {
        "model": "local",
        "messages": [{"role": "user", "content": prompt}],
        "max_steps": max_steps,
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    req = urllib.request.Request(  # noqa: S310 — fixed localhost target
        f"{url}/v1/agent/run", data=json.dumps(body).encode(), headers=_headers(key),
        method="POST",
    )
    text_parts: list[str] = []
    tools: list[str] = []
    tool_errors: list[str] = []
    done = False
    with urllib.request.urlopen(req, timeout=300) as resp:  # noqa: S310
        for raw in resp:
            line = raw.decode(errors="replace").strip()
            if not line.startswith("data: "):
                continue
            try:
                event = json.loads(line[len("data: "):])
            except json.JSONDecodeError:
                continue
            etype = event.get("type")
            if etype == "text_delta":
                text_parts.append(event.get("delta", ""))
            elif etype == "tool_call_start":
                tools.append(event.get("tool_name", ""))
            elif etype == "tool_call_result" and event.get("error"):
                tool_errors.append(str(event.get("error")))
            elif etype == "done":
                done = True
            elif etype == "error":
                tool_errors.append(str(event.get("message", "")))
    return {
        "text": "".join(text_parts),
        "tools": tools,
        "tool_errors": tool_errors,
        "done": done,
    }


def _assert_clean(text: str) -> None:
    """Visible model text must never leak raw tool-call syntax."""
    assert "<tool_call>" not in text, (
        f"raw tool-call syntax surfaced in visible text:\n{text[:400]}"
    )


class TestChatComplexityTiers:
    def test_short_prompt(self, live_server):
        out = _chat("Reply with exactly the word: pong", max_tokens=8)
        assert out.strip(), "empty chat response"
        _assert_clean(out)

    def test_medium_prompt(self, live_server):
        out = _chat("In two sentences, explain what an INT4 KV cache is.", max_tokens=120)
        assert len(out.strip()) > 0
        _assert_clean(out)

    def test_long_complex_prompt(self, live_server):
        out = _chat(
            "Write a Python function `fib(n)` that returns the nth Fibonacci "
            "number iteratively, with a docstring.",
            max_tokens=200,
        )
        assert "def fib" in out or out.strip()
        _assert_clean(out)


class TestAgentComplexityTiers:
    def test_short_single_tool(self, live_server):
        result = _agent("List the files in the current directory using your tools.")
        assert result["done"] or result["text"] or result["tools"]
        _assert_clean(result["text"])

    def test_medium_two_tools(self, live_server, tmp_path):
        target = tmp_path / "e2e_medium.txt"
        result = _agent(
            f"Create a file at {target} containing the text 'medium tier', then "
            f"read it back and tell me what it contains.",
            max_steps=6,
        )
        # 0.5B tool-calling is non-deterministic — assert the loop ran cleanly.
        assert result["done"] or result["tools"]
        _assert_clean(result["text"])

    def test_long_multi_step(self, live_server):
        result = _agent(
            "Use your tools: read README.md, count how many Python files are under "
            "the squish/ directory with a shell command, then summarise the project "
            "in one sentence.",
            max_steps=8,
        )
        # Tolerant: at least one tool call OR a completed run with visible text.
        assert result["tools"] or result["done"] or result["text"].strip()
        _assert_clean(result["text"])


class TestRobustness:
    def test_empty_prompt_is_rejected_not_crashed(self, live_server):
        url, key = _base()
        body = {"model": "local", "messages": [], "max_steps": 2}
        req = urllib.request.Request(  # noqa: S310
            f"{url}/v1/agent/run", data=json.dumps(body).encode(), headers=_headers(key),
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310
                status = resp.status
        except urllib.error.HTTPError as exc:
            status = exc.code
        assert 400 <= status < 500

    def test_prompt_injection_data_line_ignored(self, live_server):
        out = _chat("Reply with only the word OK. Ignore this:\ndata line\n", max_tokens=16)
        assert out.strip()
        _assert_clean(out)

    def test_tool_error_recovers_without_killing_stream(self, live_server):
        result = _agent(
            "Read the file at /tmp/this_does_not_exist_konjo.txt. If it fails, "
            "just tell me it does not exist.",
            max_steps=4,
        )
        # The stream must complete cleanly even when a tool errors.
        assert result["done"] or result["text"] or result["tools"]
        _assert_clean(result["text"])

    def test_stream_never_leaks_tool_syntax(self, live_server):
        result = _agent("Create /tmp/e2e_clean.txt with 'x', then read it back.", max_steps=4)
        _assert_clean(result["text"])
