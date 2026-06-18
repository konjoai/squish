"""tests/e2e/test_server_battle.py

Adversarial payload matrix against a **real** running squish server.

The contract under test: *every* malformed / hostile request must come back as
a clean 4xx — never a 5xx and never a raw traceback.  Plus a handful of real
tiny round-trips to prove the happy path still works once the defences are in.

Runs only on Apple Silicon under ``SQUISH_E2E=1`` / ``--run-e2e`` (see
``tests/e2e/conftest.py``).
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request

import pytest

pytestmark = pytest.mark.e2e


# ── HTTP helpers (raw urllib — matches the existing agent-e2e transport) ──────
def _request(
    server: "ServerHandle",  # noqa: F821 — runtime type from conftest
    method: str,
    path: str,
    *,
    body: object = None,
    raw_body: bytes | None = None,
    auth: bool = True,
    content_type: str = "application/json",
) -> tuple[int, bytes]:
    """Send one request; return (status_code, body_bytes). Never raises on 4xx/5xx."""
    headers: dict[str, str] = {}
    if content_type:
        headers["Content-Type"] = content_type
    if auth:
        headers["Authorization"] = f"Bearer {server.api_key}"

    data: bytes | None
    if raw_body is not None:
        data = raw_body
    elif body is not None:
        data = json.dumps(body).encode()
    else:
        data = None

    req = urllib.request.Request(  # noqa: S310 — fixed localhost target
        f"{server.url}{path}", data=data, headers=headers, method=method
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:  # noqa: S310
            return resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def _assert_clean_4xx(status: int, payload: bytes, label: str) -> None:
    """A defended endpoint must answer 4xx with no server-side 5xx/traceback."""
    text = payload.decode(errors="replace")
    assert 400 <= status < 500, (
        f"[{label}] expected a 4xx client error, got {status}.\nbody: {text[:600]}"
    )
    assert "Traceback (most recent call last)" not in text, (
        f"[{label}] response leaked a Python traceback:\n{text[:600]}"
    )


# ── Adversarial sampling-parameter matrix ────────────────────────────────────
_BAD_SAMPLING = [
    ("max_tokens=abc", {"max_tokens": "abc"}),
    ("max_tokens=null-string", {"max_tokens": "null"}),
    ("max_tokens=-5", {"max_tokens": -5}),
    ("max_tokens=huge-string", {"max_tokens": "99999999999999999999999999"}),
    ("max_tokens=list", {"max_tokens": [1, 2, 3]}),
    ("temperature=hot", {"temperature": "hot"}),
    ("temperature=NaN", {"temperature": float("nan")}),
    ("temperature=1e9", {"temperature": 1e9}),
    ("temperature=-1", {"temperature": -1}),
    ("top_p=-1", {"top_p": -1}),
    ("top_p=5", {"top_p": 5}),
    ("top_p=NaN", {"top_p": float("nan")}),
]


def _chat_body(**overrides) -> dict:
    body = {
        "model": "local",
        "messages": [{"role": "user", "content": "hi"}],
    }
    body.update(overrides)
    return body


class TestChatCompletionsBattle:
    @pytest.mark.parametrize("label,override", _BAD_SAMPLING, ids=[b[0] for b in _BAD_SAMPLING])
    def test_bad_sampling_params_return_4xx(self, live_server, label, override):
        # NaN is not valid JSON via the standard encoder unless we allow it.
        body = _chat_body(**override)
        status, payload = _request(
            live_server, "POST", "/v1/chat/completions",
            raw_body=json.dumps(body, allow_nan=True).encode(),
        )
        _assert_clean_4xx(status, payload, f"chat:{label}")

    def test_empty_messages_400(self, live_server):
        status, payload = _request(
            live_server, "POST", "/v1/chat/completions", body=_chat_body(messages=[]),
        )
        _assert_clean_4xx(status, payload, "chat:empty-messages")

    def test_malformed_json_400(self, live_server):
        status, payload = _request(
            live_server, "POST", "/v1/chat/completions",
            raw_body=b'{"model": "local", "messages": [',
        )
        _assert_clean_4xx(status, payload, "chat:malformed-json")

    def test_non_object_body_400(self, live_server):
        status, payload = _request(
            live_server, "POST", "/v1/chat/completions", raw_body=b"[1, 2, 3]",
        )
        _assert_clean_4xx(status, payload, "chat:non-object-body")

    def test_missing_auth_401(self, live_server):
        status, payload = _request(
            live_server, "POST", "/v1/chat/completions", body=_chat_body(), auth=False,
        )
        assert status == 401, payload[:300]

    def test_bad_auth_401(self, live_server):
        req = urllib.request.Request(  # noqa: S310
            f"{live_server.url}/v1/chat/completions",
            data=json.dumps(_chat_body()).encode(),
            headers={"Authorization": "Bearer wrong-key", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310
                status = resp.status
        except urllib.error.HTTPError as exc:
            status = exc.code
        assert status == 401

    def test_50k_char_paste_is_handled(self, live_server):
        huge = "spam " * 10_000  # ~50k chars
        status, payload = _request(
            live_server, "POST", "/v1/chat/completions",
            body=_chat_body(messages=[{"role": "user", "content": huge}], max_tokens=4),
        )
        # Either a clean success or a clean 4xx — never a 5xx/traceback.
        text = payload.decode(errors="replace")
        assert status < 500, f"huge paste caused a 5xx: {status}\n{text[:400]}"
        assert "Traceback (most recent call last)" not in text


class TestCompletionsBattle:
    @pytest.mark.parametrize("label,override", _BAD_SAMPLING, ids=[b[0] for b in _BAD_SAMPLING])
    def test_bad_sampling_params_return_4xx(self, live_server, label, override):
        body = {"model": "local", "prompt": "hello", **override}
        status, payload = _request(
            live_server, "POST", "/v1/completions",
            raw_body=json.dumps(body, allow_nan=True).encode(),
        )
        _assert_clean_4xx(status, payload, f"completions:{label}")

    def test_empty_prompt_400(self, live_server):
        status, payload = _request(
            live_server, "POST", "/v1/completions", body={"model": "local", "prompt": ""},
        )
        _assert_clean_4xx(status, payload, "completions:empty-prompt")

    def test_malformed_json_400(self, live_server):
        status, payload = _request(
            live_server, "POST", "/v1/completions", raw_body=b'{"prompt": ',
        )
        _assert_clean_4xx(status, payload, "completions:malformed-json")


class TestEmbeddingsBattle:
    def test_empty_input_400(self, live_server):
        status, payload = _request(
            live_server, "POST", "/v1/embeddings", body={"model": "local", "input": ""},
        )
        _assert_clean_4xx(status, payload, "embeddings:empty-input")

    def test_missing_input_400(self, live_server):
        status, payload = _request(
            live_server, "POST", "/v1/embeddings", body={"model": "local"},
        )
        _assert_clean_4xx(status, payload, "embeddings:missing-input")

    def test_empty_list_400(self, live_server):
        status, payload = _request(
            live_server, "POST", "/v1/embeddings", body={"input": []},
        )
        _assert_clean_4xx(status, payload, "embeddings:empty-list")

    def test_malformed_json_400(self, live_server):
        status, payload = _request(
            live_server, "POST", "/v1/embeddings", raw_body=b"{not json",
        )
        _assert_clean_4xx(status, payload, "embeddings:malformed-json")


class TestAgentRunBattle:
    def _agent_body(self, **overrides) -> dict:
        body = {"model": "local", "messages": [{"role": "user", "content": "hi"}]}
        body.update(overrides)
        return body

    @pytest.mark.parametrize(
        "label,override",
        [
            ("max_steps=abc", {"max_steps": "abc"}),
            ("max_steps=0", {"max_steps": 0}),
            ("max_steps=-1", {"max_steps": -1}),
            ("max_tokens=abc", {"max_tokens": "abc"}),
            ("temperature=NaN", {"temperature": float("nan")}),
            ("temperature=1e9", {"temperature": 1e9}),
            ("top_p=5", {"top_p": 5}),
        ],
    )
    def test_bad_params_return_4xx(self, live_server, label, override):
        status, payload = _request(
            live_server, "POST", "/v1/agent/run",
            raw_body=json.dumps(self._agent_body(**override), allow_nan=True).encode(),
        )
        _assert_clean_4xx(status, payload, f"agent:{label}")

    def test_empty_messages_400(self, live_server):
        status, payload = _request(
            live_server, "POST", "/v1/agent/run", body=self._agent_body(messages=[]),
        )
        _assert_clean_4xx(status, payload, "agent:empty-messages")

    def test_malformed_json_400(self, live_server):
        status, payload = _request(
            live_server, "POST", "/v1/agent/run", raw_body=b'{"messages":',
        )
        _assert_clean_4xx(status, payload, "agent:malformed-json")


class TestTokenizeBattle:
    def test_malformed_json_400(self, live_server):
        status, payload = _request(
            live_server, "POST", "/v1/tokenize", raw_body=b"{bad",
        )
        _assert_clean_4xx(status, payload, "tokenize:malformed-json")

    def test_missing_keys_400(self, live_server):
        status, payload = _request(
            live_server, "POST", "/v1/tokenize", body={"nonsense": 1},
        )
        _assert_clean_4xx(status, payload, "tokenize:missing-keys")


class TestReadEndpointsAuthAndShape:
    """GET endpoints: auth enforced where required; never 5xx on a bare GET."""

    @pytest.mark.parametrize(
        "path",
        ["/health", "/model/status", "/v1/models", "/v1/agent/tools", "/v1/agent/mcp", "/v1/metrics"],
    )
    def test_get_endpoint_no_5xx(self, live_server, path):
        status, payload = _request(live_server, "GET", path)
        text = payload.decode(errors="replace")
        assert status < 500, f"GET {path} returned {status}\n{text[:400]}"
        assert "Traceback (most recent call last)" not in text

    def test_health_open_without_auth(self, live_server):
        status, payload = _request(live_server, "GET", "/health", auth=False)
        assert status == 200
        assert json.loads(payload)["loaded"] is True


# ── Real happy-path smoke round-trips ────────────────────────────────────────
class TestHappyPathSmoke:
    def test_real_chat_roundtrip(self, live_server):
        status, payload = _request(
            live_server, "POST", "/v1/chat/completions",
            body=_chat_body(
                messages=[{"role": "user", "content": "Reply with exactly: pong"}],
                max_tokens=8, temperature=0.0,
            ),
        )
        assert status == 200, payload[:400]
        data = json.loads(payload)
        text = data["choices"][0]["message"]["content"]
        assert isinstance(text, str) and text.strip()

    def test_real_completion_roundtrip(self, live_server):
        status, payload = _request(
            live_server, "POST", "/v1/completions",
            body={"model": "local", "prompt": "The capital of France is", "max_tokens": 8, "temperature": 0.0},
        )
        assert status == 200, payload[:400]
        data = json.loads(payload)
        assert data["choices"][0]["text"] is not None

    def test_real_tokenize_roundtrip(self, live_server):
        status, payload = _request(
            live_server, "POST", "/v1/tokenize", body={"text": "hello world"},
        )
        assert status == 200, payload[:400]
        data = json.loads(payload)
        assert data["token_count"] >= 1
