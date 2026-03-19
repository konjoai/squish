"""
tests/test_config_unit.py

Unit tests for squish/config.py — persistent user configuration.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest


# ── helpers ──────────────────────────────────────────────────────────────────

def _import_config():
    import squish.config as cfg
    return cfg


def _tmp_config(tmp_path: Path):
    """Return a context where the config is stored in tmp_path."""
    return patch.dict(os.environ, {"SQUISH_CONFIG_DIR": str(tmp_path)})


# ── config_path ───────────────────────────────────────────────────────────────

class TestConfigPath:
    def test_default_path_is_home_squish(self, tmp_path):
        cfg = _import_config()
        with patch.dict(os.environ, {}, clear=False):
            # Remove env override if set
            env = os.environ.copy()
            env.pop("SQUISH_CONFIG_DIR", None)
            with patch.dict(os.environ, env, clear=True):
                p = cfg.config_path()
            assert p.name == "config.json"
            assert ".squish" in str(p)

    def test_env_override_changes_path(self, tmp_path):
        cfg = _import_config()
        with _tmp_config(tmp_path):
            p = cfg.config_path()
        assert p == tmp_path / "config.json"

    def test_returns_path_object(self, tmp_path):
        cfg = _import_config()
        with _tmp_config(tmp_path):
            p = cfg.config_path()
        assert isinstance(p, Path)


# ── load ─────────────────────────────────────────────────────────────────────

class TestLoad:
    def test_load_returns_defaults_when_no_file(self, tmp_path):
        cfg = _import_config()
        with _tmp_config(tmp_path):
            result = cfg.load()
        assert result["port"] == 11435
        assert result["host"] == "127.0.0.1"
        assert result["api_key"] == "squish"
        assert result["auto_compress"] is True
        assert result["default_model"] is None

    def test_load_merges_with_defaults(self, tmp_path):
        cfg = _import_config()
        (tmp_path / "config.json").write_text(json.dumps({"port": 9999}))
        with _tmp_config(tmp_path):
            result = cfg.load()
        assert result["port"] == 9999
        # defaults still present for non-overridden keys
        assert result["host"] == "127.0.0.1"
        assert result["api_key"] == "squish"

    def test_load_deep_merges_nested_keys(self, tmp_path):
        cfg = _import_config()
        (tmp_path / "config.json").write_text(json.dumps({
            "whatsapp": {"access_token": "EAABxxx"}
        }))
        with _tmp_config(tmp_path):
            result = cfg.load()
        assert result["whatsapp"]["access_token"] == "EAABxxx"
        # Other nested keys still have defaults
        assert "verify_token" in result["whatsapp"]

    def test_load_handles_corrupt_json(self, tmp_path):
        cfg = _import_config()
        (tmp_path / "config.json").write_text("NOT VALID JSON {{")
        with _tmp_config(tmp_path):
            result = cfg.load()
        # Should fall back to defaults without raising
        assert result["port"] == 11435

    def test_load_handles_permission_error(self, tmp_path):
        cfg = _import_config()
        p = tmp_path / "config.json"
        p.write_text("{}")
        p.chmod(0o000)
        try:
            with _tmp_config(tmp_path):
                result = cfg.load()
            assert result["port"] == 11435
        finally:
            p.chmod(0o644)

    def test_load_returns_independent_copy_each_call(self, tmp_path):
        cfg = _import_config()
        with _tmp_config(tmp_path):
            r1 = cfg.load()
            r2 = cfg.load()
        r1["port"] = 9999
        assert r2["port"] == 11435  # r2 unaffected


# ── save ─────────────────────────────────────────────────────────────────────

class TestSave:
    def test_save_creates_file(self, tmp_path):
        cfg = _import_config()
        with _tmp_config(tmp_path):
            cfg.save({"port": 8080})
        assert (tmp_path / "config.json").exists()

    def test_save_creates_parent_directories(self, tmp_path):
        cfg = _import_config()
        nested = tmp_path / "deep" / "nested"
        with patch.dict(os.environ, {"SQUISH_CONFIG_DIR": str(nested)}):
            cfg.save({"port": 9000})
        assert (nested / "config.json").exists()

    def test_save_roundtrip(self, tmp_path):
        cfg = _import_config()
        data = {"port": 7777, "host": "0.0.0.0", "api_key": "test"}
        with _tmp_config(tmp_path):
            cfg.save(data)
            raw = json.loads((tmp_path / "config.json").read_text())
        assert raw["port"] == 7777
        assert raw["host"] == "0.0.0.0"

    def test_save_raises_on_unwritable_dir(self, tmp_path):
        cfg = _import_config()
        tmp_path.chmod(0o444)
        try:
            with _tmp_config(tmp_path):
                with pytest.raises((RuntimeError, PermissionError, OSError)):
                    cfg.save({"port": 1})
        finally:
            tmp_path.chmod(0o755)


# ── get ───────────────────────────────────────────────────────────────────────

class TestGet:
    def test_get_top_level_key(self, tmp_path):
        cfg = _import_config()
        (tmp_path / "config.json").write_text(json.dumps({"port": 9999}))
        with _tmp_config(tmp_path):
            assert cfg.get("port") == 9999

    def test_get_missing_key_returns_default(self, tmp_path):
        cfg = _import_config()
        with _tmp_config(tmp_path):
            assert cfg.get("nonexistent_key", "fallback") == "fallback"

    def test_get_missing_key_returns_none_by_default(self, tmp_path):
        cfg = _import_config()
        with _tmp_config(tmp_path):
            assert cfg.get("nonexistent_key") is None

    def test_get_dot_notation_nested(self, tmp_path):
        cfg = _import_config()
        (tmp_path / "config.json").write_text(json.dumps({
            "whatsapp": {"access_token": "EAABxxx"}
        }))
        with _tmp_config(tmp_path):
            assert cfg.get("whatsapp.access_token") == "EAABxxx"

    def test_get_dot_notation_missing_nested(self, tmp_path):
        cfg = _import_config()
        # 'whatsapp' dict exists but 'nonexistent' key is not in it
        # _dot_get returns default=None when key missing
        with _tmp_config(tmp_path):
            assert cfg.get("whatsapp.nonexistent") is None

    def test_get_dot_notation_deeply_nested_missing(self, tmp_path):
        cfg = _import_config()
        with _tmp_config(tmp_path):
            result = cfg.get("a.b.c.d", "sentinel")
        assert result == "sentinel"

    def test_get_dot_notation_non_dict_intermediate(self, tmp_path):
        cfg = _import_config()
        (tmp_path / "config.json").write_text(json.dumps({"port": 9999}))
        with _tmp_config(tmp_path):
            result = cfg.get("port.subkey", "fallback")
        assert result == "fallback"


# ── set ───────────────────────────────────────────────────────────────────────

class TestSet:
    def test_set_top_level_persists(self, tmp_path):
        cfg = _import_config()
        with _tmp_config(tmp_path):
            cfg.set("default_model", "qwen3:8b")
            assert cfg.get("default_model") == "qwen3:8b"
            # Verify it's actually written to disk
            raw = json.loads((tmp_path / "config.json").read_text())
        assert raw["default_model"] == "qwen3:8b"

    def test_set_dot_notation_nested(self, tmp_path):
        cfg = _import_config()
        with _tmp_config(tmp_path):
            cfg.set("whatsapp.access_token", "EAABxxx")
            assert cfg.get("whatsapp.access_token") == "EAABxxx"
            # verify other nested keys untouched
            assert cfg.get("whatsapp.verify_token") == ""

    def test_set_creates_intermediate_dict(self, tmp_path):
        cfg = _import_config()
        with _tmp_config(tmp_path):
            cfg.set("new_section.key", "value")
            assert cfg.get("new_section.key") == "value"

    def test_set_overwrites_existing(self, tmp_path):
        cfg = _import_config()
        with _tmp_config(tmp_path):
            cfg.set("port", 9000)
            cfg.set("port", 8080)
            assert cfg.get("port") == 8080

    def test_set_bool_value(self, tmp_path):
        cfg = _import_config()
        with _tmp_config(tmp_path):
            cfg.set("auto_compress", False)
            assert cfg.get("auto_compress") is False

    def test_set_none_value(self, tmp_path):
        cfg = _import_config()
        with _tmp_config(tmp_path):
            cfg.set("default_model", None)
            assert cfg.get("default_model") is None


# ── _deep_merge (internal) ────────────────────────────────────────────────────

class TestDeepMerge:
    def test_deep_merge_nested_dict(self):
        cfg = _import_config()
        base = {"a": {"x": 1, "y": 2}, "b": 3}
        override = {"a": {"x": 99}}
        cfg._deep_merge(base, override)
        assert base["a"]["x"] == 99
        assert base["a"]["y"] == 2  # preserved
        assert base["b"] == 3

    def test_deep_merge_non_dict_override_replaces(self):
        cfg = _import_config()
        base = {"a": {"x": 1}}
        override = {"a": "not a dict"}
        cfg._deep_merge(base, override)
        assert base["a"] == "not a dict"

    def test_deep_merge_adds_new_keys(self):
        cfg = _import_config()
        base = {"a": 1}
        override = {"b": 2}
        cfg._deep_merge(base, override)
        assert base["b"] == 2


# ── _dot_get / _dot_set (internal) ────────────────────────────────────────────

class TestDotAccess:
    def test_dot_get_single_key(self):
        cfg = _import_config()
        assert cfg._dot_get({"a": 1}, "a") == 1

    def test_dot_get_nested(self):
        cfg = _import_config()
        assert cfg._dot_get({"a": {"b": 2}}, "a.b") == 2

    def test_dot_get_missing_returns_default(self):
        cfg = _import_config()
        assert cfg._dot_get({}, "x", "default") == "default"

    def test_dot_set_single_key(self):
        cfg = _import_config()
        d = {}
        cfg._dot_set(d, "a", 1)
        assert d == {"a": 1}

    def test_dot_set_nested(self):
        cfg = _import_config()
        d = {}
        cfg._dot_set(d, "a.b", 2)
        assert d == {"a": {"b": 2}}

    def test_dot_set_creates_intermediate(self):
        cfg = _import_config()
        d = {"a": "not-a-dict"}
        cfg._dot_set(d, "a.b", 99)
        assert d["a"] == {"b": 99}
