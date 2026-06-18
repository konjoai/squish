"""tests/e2e/test_embeddings_e2e.py

Real ``/v1/embeddings`` round-trip plus bad-input rejection, against a live
server.  Validates fixes 1a (MLX import gated) and 1b (input validation).
"""
from __future__ import annotations

import json
import math
import urllib.error
import urllib.request

import pytest

pytestmark = pytest.mark.e2e


def _post(server, body=None, raw_body=None):
    data = raw_body if raw_body is not None else json.dumps(body).encode()
    req = urllib.request.Request(  # noqa: S310 — fixed localhost target
        f"{server.url}/v1/embeddings",
        data=data,
        headers={
            "Authorization": f"Bearer {server.api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:  # noqa: S310
            return resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


class TestEmbeddingsRoundTrip:
    def test_single_string_returns_vector(self, live_server):
        status, payload = _post(live_server, {"model": "local", "input": "hello world"})
        assert status == 200, payload[:400]
        data = json.loads(payload)
        assert data["object"] == "list"
        assert len(data["data"]) == 1
        vec = data["data"][0]["embedding"]
        assert len(vec) > 0
        # L2-normalised → unit norm (allow numerical slack).
        norm = math.sqrt(sum(x * x for x in vec))
        assert norm == pytest.approx(1.0, abs=1e-3) or norm == pytest.approx(0.0, abs=1e-6)

    def test_batch_input_returns_one_vector_each(self, live_server):
        status, payload = _post(live_server, {"model": "local", "input": ["a", "b", "c"]})
        assert status == 200, payload[:400]
        data = json.loads(payload)
        assert len(data["data"]) == 3
        assert [d["index"] for d in data["data"]] == [0, 1, 2]


class TestEmbeddingsBadInput:
    @pytest.mark.parametrize(
        "label,body",
        [
            ("empty-string", {"input": ""}),
            ("missing-input", {"model": "local"}),
            ("empty-list", {"input": []}),
            ("list-with-int", {"input": ["ok", 5]}),
            ("null-input", {"input": None}),
        ],
    )
    def test_bad_input_returns_4xx(self, live_server, label, body):
        status, payload = _post(live_server, body)
        text = payload.decode(errors="replace")
        assert 400 <= status < 500, f"[{label}] expected 4xx, got {status}: {text[:300]}"
        assert "Traceback (most recent call last)" not in text

    def test_malformed_json_returns_4xx(self, live_server):
        status, payload = _post(live_server, raw_body=b"{not valid")
        assert 400 <= status < 500, payload[:300]
