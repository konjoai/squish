"""Behavioral coverage for the SSL/context, catalog-fetch, suggest/resolve,
hash-verify, and raw-model-dir-completeness paths of ``squish.catalog`` left
untested by the baseline suite. Pure-Python; network is mocked / injected.
"""
from __future__ import annotations

import json
import ssl

import pytest

from squish import catalog as cat


# ── _ssl_verify / _catalog_ssl_context ──────────────────────────────────────


def test_ssl_verify_env_and_default(monkeypatch):
    monkeypatch.delenv("REQUESTS_CA_BUNDLE", raising=False)
    monkeypatch.delenv("CURL_CA_BUNDLE", raising=False)
    assert cat._ssl_verify() is True
    monkeypatch.setenv("REQUESTS_CA_BUNDLE", "/etc/ca.pem")
    assert cat._ssl_verify() == "/etc/ca.pem"


def test_catalog_ssl_context_variants(monkeypatch):
    monkeypatch.setattr(cat, "_ssl_verify", lambda: True)
    assert cat._catalog_ssl_context() is None  # system CAs
    monkeypatch.setattr(cat, "_ssl_verify", lambda: False)
    ctx = cat._catalog_ssl_context()  # 644-647
    assert ctx.verify_mode == ssl.CERT_NONE and ctx.check_hostname is False
    monkeypatch.setattr(cat, "_ssl_verify", lambda: "/nonexistent-ca.pem")
    with pytest.raises(OSError):  # cafile path doesn't exist
        cat._catalog_ssl_context()


# ── _fetch_squishai_model_ids ────────────────────────────────────────────────


def _clear_squishai_cache():
    if hasattr(cat._fetch_squishai_model_ids, "_cache"):
        del cat._fetch_squishai_model_ids._cache


class _Resp:
    def __init__(self, data):
        self._data = data

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self):
        return self._data


def test_fetch_squishai_ids_success(monkeypatch):
    _clear_squishai_cache()
    monkeypatch.setattr(cat, "_ssl_verify", lambda: True)
    body = json.dumps([{"id": "squishai/qwen3-8b"}, {"no_id": 1}]).encode()
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: _Resp(body))
    ids = cat._fetch_squishai_model_ids()
    assert ids == {"squishai/qwen3-8b"}
    _clear_squishai_cache()


def test_fetch_squishai_ids_unverified_ssl_branch(monkeypatch):
    _clear_squishai_cache()
    monkeypatch.setattr(cat, "_ssl_verify", lambda: False)  # → unverified context (126)
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: _Resp(b"[]"))
    assert cat._fetch_squishai_model_ids() == set()
    _clear_squishai_cache()


def test_fetch_squishai_ids_network_error_degrades(monkeypatch):
    _clear_squishai_cache()
    monkeypatch.setattr(cat, "_ssl_verify", lambda: "/some/ca.pem")  # str branch (128)
    monkeypatch.setattr("urllib.request.urlopen",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("no net")))
    assert cat._fetch_squishai_model_ids() == set()  # graceful empty
    _clear_squishai_cache()


def test_fetch_squishai_ids_certifi_missing(monkeypatch):
    import builtins
    _clear_squishai_cache()
    monkeypatch.setattr(cat, "_ssl_verify", lambda: True)  # → default-CA path tries certifi
    real_import = builtins.__import__

    def _no_certifi(name, *a, **k):
        if name == "certifi":
            raise ImportError("certifi")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", _no_certifi)
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: _Resp(b"[]"))
    assert cat._fetch_squishai_model_ids() == set()  # certifi ImportError → default ctx (133-134)
    _clear_squishai_cache()


# ── _fetch_catalog_bytes (injectable opener/sleeper) ─────────────────────────


def test_fetch_catalog_bytes_success(monkeypatch):
    monkeypatch.setattr(cat, "_ssl_verify", lambda: True)
    out = cat._fetch_catalog_bytes(opener=lambda *a, **k: _Resp(b"catalog-data"))
    assert out == b"catalog-data"


def test_fetch_catalog_bytes_retries_then_none(monkeypatch):
    monkeypatch.setattr(cat, "_ssl_verify", lambda: True)
    attempts = {"n": 0}

    def _opener(*a, **k):
        attempts["n"] += 1
        raise OSError("transient")

    sleeps = []
    out = cat._fetch_catalog_bytes(max_attempts=3, opener=_opener, sleeper=sleeps.append)
    assert out is None and attempts["n"] == 3 and len(sleeps) == 2  # backoff between retries


def test_fetch_catalog_bytes_value_error_not_retried(monkeypatch):
    monkeypatch.setattr(cat, "_ssl_verify", lambda: True)

    def _opener(*a, **k):
        raise ValueError("bad url")

    assert cat._fetch_catalog_bytes(opener=_opener, sleeper=lambda s: None) is None


def test_fetch_catalog_bytes_zero_attempts(monkeypatch):
    monkeypatch.setattr(cat, "_ssl_verify", lambda: True)
    # max_attempts=0 → loop body never runs → final return None (691).
    assert cat._fetch_catalog_bytes(max_attempts=0, opener=lambda *a, **k: _Resp(b"x")) is None


# ── suggest / resolve ───────────────────────────────────────────────────────


def test_suggest_no_match_returns_empty():
    assert cat.suggest("zzz-nonexistent-model-xyz") == []


def test_suggest_finds_matches():
    out = cat.suggest("qwen", max_results=2)
    assert len(out) <= 2
    assert all("qwen" in e.id.lower() for e in out) or out == []


def test_resolve_exact_and_unknown():
    catalog = cat.load_catalog()
    some_id = next(iter(catalog))
    assert cat.resolve(some_id) is not None
    assert cat.resolve("definitely-not-a-real-model:999b") is None


# ── verify_hash ─────────────────────────────────────────────────────────────


def _entry(hf_sha256=""):
    catalog = cat.load_catalog()
    e = next(iter(catalog.values()))
    return cat.CatalogEntry(**{**e.__dict__, "hf_sha256": hf_sha256})


def test_verify_hash_no_expected_passes(tmp_path):
    ok, msg = cat.verify_hash(_entry(hf_sha256=""), tmp_path)
    assert ok and msg == ""


def test_verify_hash_no_sentinel_is_exempt(tmp_path):
    ok, msg = cat.verify_hash(_entry(hf_sha256="abc123"), tmp_path)
    assert ok and "skipping integrity check" in msg


def test_verify_hash_match_and_mismatch(tmp_path):
    entry = _entry(hf_sha256="deadbeef")
    cat.write_hash_sentinel(tmp_path, "deadbeef")
    ok, msg = cat.verify_hash(entry, tmp_path)
    assert ok and msg == ""
    cat.write_hash_sentinel(tmp_path, "different")
    ok, msg = cat.verify_hash(entry, tmp_path)
    assert not ok and "mismatch" in msg


# ── _is_raw_model_dir_complete ───────────────────────────────────────────────


def test_raw_dir_missing(tmp_path):
    ok, why = cat._is_raw_model_dir_complete(tmp_path / "nope")
    assert not ok and "missing" in why


def test_raw_dir_missing_config(tmp_path):
    ok, why = cat._is_raw_model_dir_complete(tmp_path)
    assert not ok and "config.json" in why


def test_raw_dir_temp_file(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model.safetensors.incomplete").write_text("x")
    ok, why = cat._is_raw_model_dir_complete(tmp_path)
    assert not ok and "temporary download file" in why


def test_raw_dir_safetensors_complete(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    ok, why = cat._is_raw_model_dir_complete(tmp_path)
    assert ok and why == ""


def test_raw_dir_no_weight_files(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    ok, why = cat._is_raw_model_dir_complete(tmp_path)
    assert not ok and "no weight files" in why


def test_raw_dir_empty_weight_file(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model.safetensors").write_bytes(b"")
    ok, why = cat._is_raw_model_dir_complete(tmp_path)
    assert not ok and "empty weight file" in why


def test_raw_dir_gguf_complete(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model.gguf").write_bytes(b"GGUF")
    ok, why = cat._is_raw_model_dir_complete(tmp_path)
    assert ok


def test_raw_dir_sharded_index_complete(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(
        {"weight_map": {"w1": "shard-0.safetensors", "w2": "shard-1.safetensors"}}))
    (tmp_path / "shard-0.safetensors").write_bytes(b"a")
    (tmp_path / "shard-1.safetensors").write_bytes(b"b")
    ok, why = cat._is_raw_model_dir_complete(tmp_path)
    assert ok and why == ""


def test_raw_dir_sharded_missing_shard(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(
        {"weight_map": {"w1": "shard-0.safetensors"}}))
    ok, why = cat._is_raw_model_dir_complete(tmp_path)
    assert not ok and "missing shard" in why


def test_raw_dir_sharded_empty_shard(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(
        {"weight_map": {"w1": "shard-0.safetensors"}}))
    (tmp_path / "shard-0.safetensors").write_bytes(b"")
    ok, why = cat._is_raw_model_dir_complete(tmp_path)
    assert not ok and "empty shard" in why


def test_raw_dir_sharded_bad_index(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model.safetensors.index.json").write_text("{ not json")
    ok, why = cat._is_raw_model_dir_complete(tmp_path)
    assert not ok and "invalid shard index" in why


def test_raw_dir_sharded_no_weight_map(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}}))
    ok, why = cat._is_raw_model_dir_complete(tmp_path)
    assert not ok and "weight_map" in why


def test_raw_dir_sharded_empty_shard_names(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    # weight_map present but all values are empty strings → no shard names (1050-1051).
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(
        {"weight_map": {"w1": "", "w2": ""}}))
    ok, why = cat._is_raw_model_dir_complete(tmp_path)
    assert not ok and "no shard file names" in why


def _stat_fails_on_safetensors(monkeypatch):
    """Patch stat to raise only for .safetensors files; is_file bypasses stat for
    those (returns True) but stays real for everything else (so a missing index
    still reads as absent)."""
    from pathlib import Path
    real_stat = Path.stat
    real_is_file = Path.is_file

    def flaky_stat(self, *a, **k):
        if self.suffix == ".safetensors":
            raise OSError("stat boom")
        return real_stat(self, *a, **k)

    def is_file(self):
        if self.suffix == ".safetensors":
            return True
        return real_is_file(self)

    monkeypatch.setattr(Path, "is_file", is_file)
    monkeypatch.setattr(Path, "stat", flaky_stat)


def test_raw_dir_shard_stat_error(tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(
        {"weight_map": {"w1": "shard-0.safetensors"}}))
    (tmp_path / "shard-0.safetensors").write_bytes(b"a")
    _stat_fails_on_safetensors(monkeypatch)
    ok, why = cat._is_raw_model_dir_complete(tmp_path)
    assert not ok and "cannot stat shard" in why  # 1060-1061


def test_raw_dir_weight_stat_error(tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model.safetensors").write_bytes(b"a")
    _stat_fails_on_safetensors(monkeypatch)
    ok, why = cat._is_raw_model_dir_complete(tmp_path)
    assert not ok and "cannot stat weight file" in why  # 1075-1076
