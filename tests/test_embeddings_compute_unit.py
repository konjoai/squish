"""tests/test_embeddings_compute_unit.py

Covers the Darwin/MLX compute path of ``/v1/embeddings`` (server.py ~3837-3893)
that the existing guard test cannot reach.  ``platform.system`` is faked to
``"Darwin"`` and a tiny ``mlx.core`` stand-in (numpy-backed) is injected so the
mean-pooled last-hidden-state computation runs on any host, including Linux CI.
"""

from __future__ import annotations

import asyncio
import sys
import types

import numpy as np
import pytest
from fastapi import HTTPException

import squish.server as server


class _FakeRequest:
    def __init__(self, body):
        self._body = body

    async def json(self):
        return self._body


class _FakeTokenizer:
    """Has ``.encode`` → list[int] (the preferred path at line 3858)."""

    def encode(self, text):
        return [1, 2, 3, len(text)]


def _install_fake_mlx_core(monkeypatch):
    fake_mx = types.ModuleType("mlx.core")
    fake_mx.int32 = "int32"
    fake_mx.array = lambda ids, dtype=None: np.asarray(ids)
    fake_mx.mean = lambda a, axis=None: np.mean(np.asarray(a), axis=axis)
    pkg = types.ModuleType("mlx")
    pkg.core = fake_mx
    monkeypatch.setitem(sys.modules, "mlx", pkg)
    monkeypatch.setitem(sys.modules, "mlx.core", fake_mx)
    return fake_mx


@pytest.fixture
def darwin_loaded(monkeypatch):
    """Model resident + platform reports Darwin so we reach the compute path."""
    monkeypatch.setattr(server, "_API_KEY", None, raising=False)
    monkeypatch.setattr(server.platform, "system", lambda: "Darwin")
    server._LOAD_COMPLETE.set()
    monkeypatch.setattr(server._state, "tokenizer", _FakeTokenizer(), raising=False)
    yield monkeypatch
    server._LOAD_COMPLETE.clear()


def _model_returning(hidden_fn):
    """A model whose ``.model(x)`` returns a (1, seq, D) hidden tensor."""
    inner = types.SimpleNamespace(__call__=None)
    model = types.SimpleNamespace()
    model.model = lambda x: hidden_fn(x)
    return model


def test_embeddings_happy_path_single_string(darwin_loaded):
    mp = darwin_loaded
    _install_fake_mlx_core(mp)
    mp.setattr(
        server._state,
        "model",
        _model_returning(lambda x: np.random.randn(1, np.asarray(x).shape[1], 8)),
        raising=False,
    )

    resp = asyncio.run(server.embeddings(_FakeRequest({"input": "hello"}), creds=None))
    import json as _json

    payload = _json.loads(bytes(resp.body))
    assert payload["object"] == "list"
    assert len(payload["data"]) == 1
    vec = payload["data"][0]["embedding"]
    assert len(vec) == 8
    # L2-normalised → unit norm
    assert abs(float(np.linalg.norm(vec)) - 1.0) < 1e-5
    assert payload["usage"]["total_tokens"] == 4  # _FakeTokenizer → 4 ids


def test_embeddings_list_input_and_zero_norm(darwin_loaded):
    mp = darwin_loaded
    _install_fake_mlx_core(mp)
    # All-zero hidden state → norm == 0 → division skipped (branch 3879 false)
    mp.setattr(
        server._state,
        "model",
        _model_returning(lambda x: np.zeros((1, np.asarray(x).shape[1], 4))),
        raising=False,
    )

    resp = asyncio.run(server.embeddings(_FakeRequest({"input": ["a", "bb"]}), creds=None))
    import json as _json

    payload = _json.loads(bytes(resp.body))
    assert len(payload["data"]) == 2
    # zero hidden → zero embedding, left un-normalised
    assert payload["data"][0]["embedding"] == [0.0, 0.0, 0.0, 0.0]
    assert payload["data"][1]["index"] == 1


def test_embeddings_503_when_mlx_unimportable(darwin_loaded):
    mp = darwin_loaded
    # Darwin reported, but mlx cannot import → clean 503 (lines 3840-3846).
    mp.setitem(sys.modules, "mlx", None)
    mp.setitem(sys.modules, "mlx.core", None)
    mp.setattr(server._state, "model", object(), raising=False)

    with pytest.raises(HTTPException) as ei:
        asyncio.run(server.embeddings(_FakeRequest({"input": "x"}), creds=None))
    assert ei.value.status_code == 503
    assert "MLX" in ei.value.detail or "Apple Silicon" in ei.value.detail


def test_embeddings_triggers_lazy_load_when_not_complete(monkeypatch):
    """When the model isn't resident yet, the endpoint blocks on the lazy
    loader (lines 3822-3823) before re-checking ``_state.model``."""
    monkeypatch.setattr(server, "_API_KEY", None, raising=False)
    server._LOAD_COMPLETE.clear()  # force the lazy-load branch
    loaded = {"called": False}

    def _fake_load():
        loaded["called"] = True  # leaves _state.model as None → 503 afterwards

    monkeypatch.setattr(server, "_ensure_loaded_blocking", _fake_load, raising=False)
    monkeypatch.setattr(server._state, "model", None, raising=False)
    try:
        with pytest.raises(HTTPException) as ei:
            asyncio.run(server.embeddings(_FakeRequest({"input": "x"}), creds=None))
        assert ei.value.status_code == 503
        assert loaded["called"] is True
    finally:
        server._LOAD_COMPLETE.clear()
