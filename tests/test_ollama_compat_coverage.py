"""Behavioral coverage for ``squish.serving.ollama_compat`` — the Ollama-
compatible FastAPI routes mounted by ``mount_ollama``.

State, the generate function, and the tokenizer are injected via the
``get_state`` / ``get_generate`` / ``get_tokenizer`` callables, so every route
(tags, show, generate, chat, ps, stubs) is exercised with a TestClient and no
real model / MLX. The embeddings route is ``# pragma: no cover`` (MLX-only).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from squish.serving import ollama_compat


class _State:
    def __init__(self, model="loaded", model_name="qwen2.5-7b"):
        self.model = model
        self.model_name = model_name


class _ChatTok:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "TEMPLATED_PROMPT"


class _NoTemplateTok:
    pass  # no apply_chat_template → _messages_to_prompt uses the fallback


def _gen_fn(prompt, max_tokens, temperature, top_p, stop, seed):
    yield ("Hello", None)
    yield (" world", None)
    yield ("", "stop")


def _gen_no_finish(prompt, max_tokens, temperature, top_p, stop, seed):
    # Exhausts without ever emitting a finish marker → the consumer loop exits
    # by iterator exhaustion rather than by `break`.
    yield ("A", None)
    yield ("B", None)


def _client(state=None, tokenizer=None, models_dir=None, generate=None):
    app = FastAPI()
    state = state if state is not None else _State()
    ollama_compat.mount_ollama(
        app,
        get_state=lambda: state,
        get_generate=lambda: (generate or _gen_fn),
        get_tokenizer=lambda: (tokenizer or _ChatTok()),
        models_dir=models_dir,
    )
    return TestClient(app)


# ── Metadata routes ─────────────────────────────────────────────────────────


def test_version(tmp_path):
    r = _client(models_dir=tmp_path).get("/api/version")
    assert r.status_code == 200 and "version" in r.json()


def _patch_scanner(monkeypatch, models):
    """Patch LocalModelScanner so find_all() returns a fixed model list."""
    import squish.serving.local_model_scanner as lms

    class _FakeScanner:
        def __init__(self, **kw):
            pass

        def find_all(self):
            return list(models)

    monkeypatch.setattr(lms, "LocalModelScanner", _FakeScanner)


def test_tags_empty_dir_falls_back_to_single_card(tmp_path, monkeypatch):
    _patch_scanner(monkeypatch, [])  # no models discovered anywhere
    r = _client(models_dir=tmp_path).get("/api/tags")
    models = r.json()["models"]
    assert len(models) == 1 and models[0]["name"].endswith(":latest")


def test_tags_lists_discovered_squish_model(tmp_path, monkeypatch):
    import types as _t
    model = _t.SimpleNamespace(name="qwen2.5-7b", size_bytes=4_000_000_000,
                               family="qwen2", params="7B", source="squish")
    _patch_scanner(monkeypatch, [model])
    r = _client(models_dir=tmp_path).get("/api/tags")
    names = [m["name"] for m in r.json()["models"]]
    assert names == ["qwen2.5-7b:latest"]  # ":latest" appended when no tag present


def _scanner_raises(monkeypatch):
    import squish.serving.local_model_scanner as lms
    monkeypatch.setattr(lms, "LocalModelScanner",
                        lambda **k: (_ for _ in ()).throw(ValueError("boom")))


def test_tags_scanner_failure_uses_direct_fallback(tmp_path, monkeypatch):
    (tmp_path / "Llama-3-8B").mkdir()
    (tmp_path / ".hidden").mkdir()  # dotdir → skipped in the fallback scan (120)
    _scanner_raises(monkeypatch)
    names = [m["name"] for m in _client(models_dir=tmp_path).get("/api/tags").json()["models"]]
    assert any(n.startswith("Llama-3-8B") for n in names)
    assert not any(n.startswith(".hidden") for n in names)


def test_tags_scanner_failure_missing_dir_single_card(tmp_path, monkeypatch):
    _scanner_raises(monkeypatch)
    missing = tmp_path / "does-not-exist"
    # Fallback scan: models_dir absent → loop skipped (117→137) → single card.
    models = _client(models_dir=missing).get("/api/tags").json()["models"]
    assert len(models) == 1


def test_show_returns_card_with_family(tmp_path):
    r = _client(models_dir=tmp_path).post("/api/show", json={"name": "mistral-7b"})
    body = r.json()
    assert body["details"]["family"] == "mistral"
    assert body["details"]["parameter_size"] == "7B"


def test_show_family_guesses(tmp_path):
    c = _client(models_dir=tmp_path)

    def fam(name):
        return c.post("/api/show", json={"name": name}).json()["details"]["family"]

    assert fam("qwen3-8b") == "qwen2"
    assert fam("llama3") == "llama"
    assert fam("gemma2-2b") == "gemma"
    assert fam("something-else") == "unknown"


def test_ps_no_model_returns_empty():
    r = _client(state=_State(model=None)).get("/api/ps")
    assert r.json() == {"models": []}


def test_ps_with_model_returns_card(tmp_path):
    r = _client(models_dir=tmp_path).get("/api/ps")
    models = r.json()["models"]
    assert len(models) == 1 and models[0]["model"].endswith(":latest")


# ── Stub routes ─────────────────────────────────────────────────────────────


def test_pull_streams_local_message(tmp_path):
    r = _client(models_dir=tmp_path).post("/api/pull", json={"name": "x"})
    lines = [json.loads(ln) for ln in r.text.splitlines() if ln.strip()]
    assert lines[-1]["status"] == "done"


def test_create_streams_message(tmp_path):
    r = _client(models_dir=tmp_path).post("/api/create", json={"name": "x"})
    lines = [json.loads(ln) for ln in r.text.splitlines() if ln.strip()]
    assert lines[-1]["status"] == "success"


def test_delete_rejected(tmp_path):
    assert _client(models_dir=tmp_path).request("DELETE", "/api/delete").status_code == 400


def test_blobs_head_404(tmp_path):
    assert _client(models_dir=tmp_path).head("/api/blobs/abc").status_code == 404


def test_blobs_push_400(tmp_path):
    assert _client(models_dir=tmp_path).post("/api/blobs/abc").status_code == 400


def test_copy_400(tmp_path):
    assert _client(models_dir=tmp_path).post("/api/copy", json={}).status_code == 400


# ── /api/generate ───────────────────────────────────────────────────────────


def test_generate_model_not_loaded_503():
    r = _client(state=_State(model=None)).post("/api/generate", json={"prompt": "hi"})
    assert r.status_code == 503


def test_generate_empty_prompt_400(tmp_path):
    r = _client(models_dir=tmp_path).post("/api/generate", json={"prompt": ""})
    assert r.status_code == 400


def test_generate_streaming(tmp_path):
    r = _client(models_dir=tmp_path).post("/api/generate",
                                          json={"prompt": "hi", "stream": True})
    lines = [json.loads(ln) for ln in r.text.splitlines() if ln.strip()]
    assert "".join(x.get("response", "") for x in lines) == "Hello world"
    assert lines[-1]["done"] is True and lines[-1]["eval_count"] == 2


def test_generate_non_streaming(tmp_path):
    r = _client(models_dir=tmp_path).post(
        "/api/generate",
        json={"prompt": "hi", "stream": False, "options": {"num_predict": 5, "seed": 1}},
    )
    body = r.json()
    assert body["response"] == "Hello world" and body["done"] is True


# ── /api/chat ───────────────────────────────────────────────────────────────


def test_chat_model_not_loaded_503():
    r = _client(state=_State(model=None)).post("/api/chat",
                                               json={"messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 503


def test_chat_empty_messages_400(tmp_path):
    r = _client(models_dir=tmp_path).post("/api/chat", json={"messages": []})
    assert r.status_code == 400


def test_chat_streaming_with_template(tmp_path):
    r = _client(models_dir=tmp_path).post(
        "/api/chat",
        json={"messages": [{"role": "user", "content": "hi"}], "stream": True},
    )
    lines = [json.loads(ln) for ln in r.text.splitlines() if ln.strip()]
    content = "".join(x.get("message", {}).get("content", "") for x in lines)
    assert content == "Hello world" and lines[-1]["done"] is True


def test_chat_non_streaming_with_fallback_prompt(tmp_path):
    # No apply_chat_template → _messages_to_prompt uses the fallback concatenation.
    r = _client(models_dir=tmp_path, tokenizer=_NoTemplateTok()).post(
        "/api/chat",
        json={"messages": [{"role": "user", "content": "hi"}], "stream": False},
    )
    assert r.json()["message"]["content"] == "Hello world"


def test_chat_template_exception_falls_back(tmp_path):
    class _BoomTok:
        def apply_chat_template(self, *a, **k):
            raise RuntimeError("jinja boom")

    r = _client(models_dir=tmp_path, tokenizer=_BoomTok()).post(
        "/api/chat",
        json={"messages": [{"role": "user", "content": "hi"}], "stream": False},
    )
    assert r.json()["message"]["content"] == "Hello world"


# ── generator-exhaustion paths (no finish marker → loop exits naturally) ─────


def test_generate_stream_loop_exhaustion(tmp_path):
    r = _client(models_dir=tmp_path, generate=_gen_no_finish).post(
        "/api/generate", json={"prompt": "hi", "stream": True})
    lines = [json.loads(ln) for ln in r.text.splitlines() if ln.strip()]
    assert "".join(x.get("response", "") for x in lines) == "AB"  # 288→306


def test_generate_nonstream_loop_exhaustion(tmp_path):
    r = _client(models_dir=tmp_path, generate=_gen_no_finish).post(
        "/api/generate", json={"prompt": "hi", "stream": False})
    assert r.json()["response"] == "AB"  # 325→332


def test_chat_stream_loop_exhaustion(tmp_path):
    r = _client(models_dir=tmp_path, generate=_gen_no_finish).post(
        "/api/chat",
        json={"messages": [{"role": "user", "content": "hi"}], "stream": True})
    lines = [json.loads(ln) for ln in r.text.splitlines() if ln.strip()]
    content = "".join(x.get("message", {}).get("content", "") for x in lines)
    assert content == "AB"  # 380→398


def test_chat_nonstream_loop_exhaustion(tmp_path):
    r = _client(models_dir=tmp_path, generate=_gen_no_finish).post(
        "/api/chat",
        json={"messages": [{"role": "user", "content": "hi"}], "stream": False})
    assert r.json()["message"]["content"] == "AB"  # 416→423
