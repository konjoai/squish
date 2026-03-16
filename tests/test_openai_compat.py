"""
tests/test_openai_compat.py

Live integration tests for the squish OpenAI-compatible server.

These tests require a running squish server at http://localhost:11435.
They are skipped automatically when the server is unreachable, and are
also gated behind the ``--run-integration`` flag so they never execute
during a plain ``pytest`` run.

Run with:
    pytest tests/test_openai_compat.py --run-integration
    # or filter by marker:
    pytest -m integration
"""
from __future__ import annotations

import os

import pytest

httpx = pytest.importorskip(
    "httpx",
    reason="httpx package not installed; pip install httpx",
)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

MODEL: str = os.environ.get("SQUISH_TEST_MODEL", "qwen3:8b")
BASE_URL: str = "http://localhost:11435"

# Import the real openai SDK; skip entire module if not installed.
openai = pytest.importorskip(
    "openai",
    reason="openai package not installed; pip install openai",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _server_is_alive() -> bool:
    """Return True if the squish server responds at ``BASE_URL/health``."""
    try:
        resp = httpx.get(f"{BASE_URL}/health", timeout=1.0)
        return resp.status_code == 200
    except Exception:
        return False


def _client() -> openai.OpenAI:  # type: ignore[name-defined]
    """Return a configured OpenAI client pointed at the local squish server."""
    return openai.OpenAI(base_url=f"{BASE_URL}/v1", api_key="squish")


# ---------------------------------------------------------------------------
# Shared decorators
# ---------------------------------------------------------------------------

_integration = pytest.mark.integration
_skip_if_down = pytest.mark.skipif(
    not _server_is_alive(),
    reason="squish server not running",
)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@_integration
@_skip_if_down
def test_list_models():
    """GET /v1/models returns HTTP 200 with at least one model entry."""
    models = _client().models.list()
    assert len(models.data) > 0, "Expected at least one model in the list"


@_integration
@_skip_if_down
def test_non_streaming_chat():
    """POST /v1/chat/completions with stream=False returns a text response."""
    response = _client().chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "Say hello in one word."}],
        stream=False,
        max_tokens=20,
    )
    assert response.choices, "Response must have at least one choice"
    text = response.choices[0].message.content
    assert isinstance(text, str) and len(text) > 0, "Response content must be non-empty"


@_integration
@_skip_if_down
def test_streaming_chat():
    """POST /v1/chat/completions with stream=True yields SSE delta chunks."""
    chunks = list(
        _client().chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": "Count to three."}],
            stream=True,
            max_tokens=30,
        )
    )
    assert len(chunks) > 0, "Expected at least one chunk"
    content_chunks = [
        c
        for c in chunks
        if c.choices and c.choices[0].delta.content
    ]
    assert len(content_chunks) > 0, "Expected at least one chunk carrying content"


@_integration
@_skip_if_down
def test_tool_calling():
    """POST /v1/chat/completions with tools=[...] is accepted without error."""
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather for a location.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City name",
                        },
                    },
                    "required": ["location"],
                },
            },
        }
    ]
    response = _client().chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "What is the weather in Paris?"}],
        tools=tools,
        max_tokens=128,
    )
    assert response.choices, "Response must have at least one choice"
    # The model may either call the tool or produce a text response; both are valid.
    assert response.choices[0].message is not None


@_integration
@_skip_if_down
def test_embeddings():
    """POST /v1/embeddings returns a float vector for the input text."""
    result = _client().embeddings.create(
        model=MODEL,
        input="The quick brown fox jumps over the lazy dog.",
    )
    assert result.data, "Expected at least one embedding result"
    vector = result.data[0].embedding
    assert isinstance(vector, list) and len(vector) > 0, "Embedding must be a non-empty list"


@_integration
@_skip_if_down
def test_health_endpoint():
    """GET /health returns HTTP 200."""
    resp = httpx.get(f"{BASE_URL}/health", timeout=5.0)
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"


@_integration
@_skip_if_down
def test_ollama_chat():
    """POST /api/chat (Ollama-compatible endpoint) returns a valid response body."""
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "Reply with one word."}],
        "stream": False,
    }
    resp = httpx.post(f"{BASE_URL}/api/chat", json=payload, timeout=30.0)
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
    body = resp.json()
    assert "message" in body, "Response body must contain 'message'"
    assert "content" in body["message"], "Message must contain 'content'"


@_integration
@_skip_if_down
def test_ollama_tags():
    """GET /api/tags returns an Ollama-compatible model list."""
    resp = httpx.get(f"{BASE_URL}/api/tags", timeout=5.0)
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
    body = resp.json()
    assert "models" in body, "Response body must contain 'models'"


@_integration
@_skip_if_down
def test_stop_token_not_in_response():
    """The stop token '###' must not appear in the generated text."""
    response = _client().chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "Write a short sentence."}],
        stop=["###"],
        max_tokens=50,
    )
    assert response.choices, "Response must have at least one choice"
    text = response.choices[0].message.content or ""
    assert "###" not in text, "Stop token '###' must not appear in the response"


@_integration
@_skip_if_down
def test_max_tokens_respected():
    """The server must honour the max_tokens limit."""
    max_tok = 10
    response = _client().chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "Tell me a very long story."}],
        max_tokens=max_tok,
    )
    assert response.choices, "Response must have at least one choice"
    usage = response.usage
    if usage is not None:
        assert usage.completion_tokens <= max_tok, (
            f"completion_tokens ({usage.completion_tokens}) exceeded max_tokens ({max_tok})"
        )


@_integration
@_skip_if_down
def test_tool_choice_none():
    """tool_choice='none' must suppress tool calls even when tools are provided."""
    tools = [
        {
            "type": "function",
            "function": {
                "name": "always_call_me",
                "description": "Should never be called when tool_choice is none.",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    response = _client().chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "What time is it?"}],
        tools=tools,
        tool_choice="none",
        max_tokens=50,
    )
    assert response.choices, "Response must have at least one choice"
    tool_calls = response.choices[0].message.tool_calls
    assert not tool_calls, (
        f"Expected no tool calls with tool_choice='none', got: {tool_calls}"
    )
