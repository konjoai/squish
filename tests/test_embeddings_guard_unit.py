"""tests/test_embeddings_guard_unit.py

Linux-runnable unit test for the MLX platform guard added to the
``/v1/embeddings`` endpoint in ``squish.server``.

The endpoint computes embeddings via ``mlx.core`` and therefore must refuse to
run — with a clean ``503`` — on a host without the MLX backend, instead of
crashing on an unguarded ``import mlx.core``.  We exercise that branch by
faking ``platform.system()`` so the test runs anywhere (including Linux CI).
"""
from __future__ import annotations

import asyncio

import pytest
from fastapi import HTTPException

import squish.server as server


class _FakeRequest:
    async def json(self):
        return {"input": "hello"}


@pytest.fixture
def loaded_state(monkeypatch):
    """Pretend a model is resident so we reach the platform guard."""
    monkeypatch.setattr(server, "_API_KEY", None, raising=False)
    server._LOAD_COMPLETE.set()
    monkeypatch.setattr(server._state, "model", object(), raising=False)
    monkeypatch.setattr(server._state, "tokenizer", object(), raising=False)
    yield
    server._LOAD_COMPLETE.clear()


def test_embeddings_refuses_on_non_darwin(monkeypatch, loaded_state):
    monkeypatch.setattr(server.platform, "system", lambda: "Linux")

    with pytest.raises(HTTPException) as ei:
        asyncio.run(server.embeddings(_FakeRequest(), creds=None))

    assert ei.value.status_code == 503
    assert "MLX" in ei.value.detail or "Apple Silicon" in ei.value.detail


def test_embeddings_model_not_loaded_503(monkeypatch):
    monkeypatch.setattr(server, "_API_KEY", None, raising=False)
    server._LOAD_COMPLETE.set()
    monkeypatch.setattr(server._state, "model", None, raising=False)
    try:
        with pytest.raises(HTTPException) as ei:
            asyncio.run(server.embeddings(_FakeRequest(), creds=None))
        assert ei.value.status_code == 503
        assert "not loaded" in ei.value.detail.lower()
    finally:
        server._LOAD_COMPLETE.clear()
