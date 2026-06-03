"""
tests/test_catalog_branches.py

Branch coverage tests for squish/catalog.py:
  - _try_refresh_catalog SQUISH_OFFLINE path   (lines 294-316)
  - _try_refresh_catalog fresh TTL path        (lines 310-317)
  - load_catalog(refresh=True) unlinks cache   (line 369)
  - list_catalog with sort_key ValueError      (line 393)
  - _has_squish_weights                        (lines 508-512)
"""
from __future__ import annotations

import json
import os
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import squish.catalog as _cat

# ── Helper: write a minimal valid catalog JSON ────────────────────────────────

def _write_catalog(path: Path, entries=None) -> None:
    if entries is None:
        entries = [
            {
                "id":              "qwen3:8b",
                "name":           "Qwen3 8B",
                "hf_mlx_repo":    "some/repo",
                "size_gb":        5.0,
                "squished_size_gb": 1.5,
                "params":         "8B",
                "context":        32768,
            }
        ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"models": entries}))


# ── SQUISH_OFFLINE mode ───────────────────────────────────────────────────────

class TestCatalogOfflineMode:
    def test_offline_with_no_local_cache(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        """SQUISH_OFFLINE=1 and no local cache: returns empty/bundled catalog."""
        monkeypatch.setenv("SQUISH_OFFLINE", "1")
        monkeypatch.setattr(_cat, "_CATALOG_CACHE", None)
        # Point LOCAL_CATALOG_PATH to a non-existent file so exists() returns False
        nonexistent = tmp_path / "no_such_catalog.json"
        monkeypatch.setattr(_cat, "LOCAL_CATALOG_PATH", nonexistent)
        catalog = _cat._try_refresh_catalog({})
        assert isinstance(catalog, dict)

    def test_offline_with_local_cache_loads_it(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        """SQUISH_OFFLINE=1 and local cache exists: load it synchronously."""
        monkeypatch.setenv("SQUISH_OFFLINE", "1")
        monkeypatch.setattr(_cat, "_CATALOG_CACHE", None)
        cache_file = tmp_path / "catalog.json"
        _write_catalog(cache_file)
        monkeypatch.setattr(_cat, "LOCAL_CATALOG_PATH", cache_file)
        catalog = _cat._try_refresh_catalog({})
        assert "qwen3:8b" in catalog

    def test_offline_with_malformed_cache(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        """SQUISH_OFFLINE=1 with malformed JSON: silently fails, returns what we have."""
        monkeypatch.setenv("SQUISH_OFFLINE", "1")
        monkeypatch.setattr(_cat, "_CATALOG_CACHE", None)
        cache_file = tmp_path / "catalog.json"
        cache_file.write_text("not valid json")
        monkeypatch.setattr(_cat, "LOCAL_CATALOG_PATH", cache_file)
        catalog = _cat._try_refresh_catalog({})
        assert isinstance(catalog, dict)

    def test_offline_with_malformed_entry_skipped(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        """SQUISH_OFFLINE=1 with entry missing required keys: KeyError swallowed (line 297)."""
        monkeypatch.setenv("SQUISH_OFFLINE", "1")
        monkeypatch.setattr(_cat, "_CATALOG_CACHE", None)
        cache_file = tmp_path / "catalog.json"
        # Entry has 'id' but missing required 'name', 'hf_mlx_repo', etc.
        cache_file.write_text(json.dumps({"models": [{"id": "broken"}]}))
        monkeypatch.setattr(_cat, "LOCAL_CATALOG_PATH", cache_file)
        catalog = _cat._try_refresh_catalog({})
        assert isinstance(catalog, dict)
        assert "broken" not in catalog  # entry was skipped due to missing fields


# ── Fresh TTL path ────────────────────────────────────────────────────────────

class TestCatalogFreshTTL:
    def test_fresh_cache_served_from_disk(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        """Within TTL: load from disk, skip network."""
        monkeypatch.delenv("SQUISH_OFFLINE", raising=False)
        monkeypatch.setattr(_cat, "_CATALOG_CACHE", None)
        cache_file = tmp_path / "catalog.json"
        _write_catalog(cache_file)
        # Make the file appear fresh (mtime = now)
        monkeypatch.setattr(_cat, "LOCAL_CATALOG_PATH", cache_file)
        monkeypatch.setattr(_cat, "CATALOG_TTL", 9999)  # very long TTL
        catalog = _cat._try_refresh_catalog({})
        assert "qwen3:8b" in catalog

    def test_fresh_cache_with_malformed_json(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        """Within TTL but malformed JSON: falls through to background refresh."""
        monkeypatch.delenv("SQUISH_OFFLINE", raising=False)
        monkeypatch.setattr(_cat, "_CATALOG_CACHE", None)
        cache_file = tmp_path / "catalog.json"
        cache_file.write_text("{bad json")
        monkeypatch.setattr(_cat, "LOCAL_CATALOG_PATH", cache_file)
        monkeypatch.setattr(_cat, "CATALOG_TTL", 9999)
        # Should not raise
        catalog = _cat._try_refresh_catalog({})
        assert isinstance(catalog, dict)


# ── load_catalog(refresh=True) ────────────────────────────────────────────────

class TestLoadCatalogRefresh:
    def test_refresh_unlinks_local_cache(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        """load_catalog(refresh=True) should delete the local cache file."""
        monkeypatch.setattr(_cat, "_CATALOG_CACHE", None)
        cache_file = tmp_path / "catalog.json"
        _write_catalog(cache_file)
        assert cache_file.exists()
        monkeypatch.setattr(_cat, "LOCAL_CATALOG_PATH", cache_file)
        monkeypatch.setattr(_cat, "CATALOG_TTL", 0)  # force stale
        # threading is imported locally in _try_refresh_catalog, so patch threading.Thread directly
        with patch("threading.Thread") as mock_thread_cls:
            mock_thread_cls.return_value = MagicMock()
            _cat.load_catalog(refresh=True)
        assert not cache_file.exists()


# ── list_catalog sort_key ValueError ──────────────────────────────────────────

class TestListCatalogSortKey:
    def test_unknown_param_format_sorts_last(self, monkeypatch: pytest.MonkeyPatch):
        """Params like '??' have no recognized unit → sort key returns 9999."""
        monkeypatch.setattr(_cat, "_CATALOG_CACHE", None)
        entry = _cat.CatalogEntry(
            id="test:weird",
            name="Weird Model",
            hf_mlx_repo="x/y",
            size_gb=1.0,
            squished_size_gb=0.3,
            params="??",
            context=4096,
        )
        monkeypatch.setattr(_cat, "_CATALOG_CACHE",
                            {"test:weird": entry})
        entries = _cat.list_catalog()
        assert any(e.id == "test:weird" for e in entries)

    def test_param_with_float_conversion_error(self, monkeypatch: pytest.MonkeyPatch):
        """Params like 'XB' where X is not floatable triggers ValueError in sort_key."""
        monkeypatch.setattr(_cat, "_CATALOG_CACHE", None)
        entry = _cat.CatalogEntry(
            id="test:bad",
            name="Bad Model",
            hf_mlx_repo="x/y",
            size_gb=1.0,
            squished_size_gb=0.3,
            params="NANB",  # ends with B but 'NAN' is valid float → no error
            context=4096,
        )
        # Use a params that will fail float() conversion
        entry2 = _cat.CatalogEntry(
            id="test:bad2",
            name="Bad Model 2",
            hf_mlx_repo="x/y",
            size_gb=1.0,
            squished_size_gb=0.3,
            params="!!B",  # ends with B but '!!' is not floatable → ValueError
            context=4096,
        )
        monkeypatch.setattr(_cat, "_CATALOG_CACHE",
                            {"test:bad": entry, "test:bad2": entry2})
        entries = _cat.list_catalog()
        # Just verify it doesn't crash and returns both
        assert len(entries) == 2


# ── _has_squish_weights ───────────────────────────────────────────────────────

class TestHasSquishWeights:
    def test_returns_true_for_squish_weights_npz(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(_cat, "_hf_list_files", lambda repo, token=None: ["squish_weights.npz"])
        assert _cat._has_squish_weights("some/repo") is True

    def test_returns_true_for_squish_npy_dir(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(_cat, "_hf_list_files", lambda repo, token=None: ["squish_npy/weights.npy"])
        assert _cat._has_squish_weights("some/repo") is True

    def test_returns_false_when_no_squish_weights(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(_cat, "_hf_list_files", lambda repo, token=None: ["config.json", "model.safetensors"])
        assert _cat._has_squish_weights("some/repo") is False

    def test_empty_file_list(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(_cat, "_hf_list_files", lambda repo, token=None: [])
        assert _cat._has_squish_weights("some/repo") is False


# ── Stale cache load path ─────────────────────────────────────────────────────

class TestStaleCacheLoad:
    def test_stale_cache_loads_while_refreshing(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        """Stale cache: should load from disk immediately while background thread fetches."""
        monkeypatch.delenv("SQUISH_OFFLINE", raising=False)
        monkeypatch.setattr(_cat, "_CATALOG_CACHE", None)
        cache_file = tmp_path / "catalog.json"
        _write_catalog(cache_file)
        # Make the file appear stale (mtime = unix epoch)
        os.utime(cache_file, (0, 0))
        monkeypatch.setattr(_cat, "LOCAL_CATALOG_PATH", cache_file)
        monkeypatch.setattr(_cat, "CATALOG_TTL", 1)  # 1 second TTL
        # threading is imported locally in _try_refresh_catalog, so patch threading.Thread directly
        with patch("threading.Thread") as mock_thread_cls:
            mock_thread_cls.return_value = MagicMock()
            catalog = _cat._try_refresh_catalog({})
        # Should have loaded from stale cache
        assert "qwen3:8b" in catalog


# ── _is_raw_model_dir_complete + partial raw dir recovery ───────────────────

class TestRawModelCompleteness:
    def test_complete_single_shard_model(self, tmp_path: Path):
        (tmp_path / "config.json").write_text("{}", encoding="utf-8")
        (tmp_path / "model.safetensors").write_bytes(b"ok")
        ok, reason = _cat._is_raw_model_dir_complete(tmp_path)
        assert ok is True
        assert reason == ""

    def test_missing_config_is_incomplete(self, tmp_path: Path):
        (tmp_path / "model.safetensors").write_bytes(b"ok")
        ok, reason = _cat._is_raw_model_dir_complete(tmp_path)
        assert ok is False
        assert "config.json" in reason

    def test_transient_marker_marks_incomplete(self, tmp_path: Path):
        (tmp_path / "config.json").write_text("{}", encoding="utf-8")
        (tmp_path / "model.safetensors").write_bytes(b"ok")
        (tmp_path / "weights.incomplete").write_text("partial", encoding="utf-8")
        ok, reason = _cat._is_raw_model_dir_complete(tmp_path)
        assert ok is False
        assert "temporary download file" in reason

    def test_sharded_index_with_missing_shard_is_incomplete(self, tmp_path: Path):
        (tmp_path / "config.json").write_text("{}", encoding="utf-8")
        (tmp_path / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"x": "model-00001-of-00002.safetensors"}}),
            encoding="utf-8",
        )
        ok, reason = _cat._is_raw_model_dir_complete(tmp_path)
        assert ok is False
        assert "missing shard" in reason


class TestPullPartialRawRecovery:
    def test_pull_removes_partial_raw_dir_and_redownloads(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        entry = _cat.CatalogEntry(
            id="demo:1b",
            name="Demo 1B",
            hf_mlx_repo="mlx-community/Demo-1B-bf16",
            size_gb=1.0,
            params="1B",
            context=4096,
            squished_size_gb=0.4,
            squish_repo=None,
        )

        raw_dir = tmp_path / entry.dir_name
        raw_dir.mkdir(parents=True)
        (raw_dir / "config.json").write_text("{}", encoding="utf-8")
        (raw_dir / "weights.incomplete").write_text("partial", encoding="utf-8")

        monkeypatch.setattr(_cat, "resolve", lambda name, refresh=False: entry)
        monkeypatch.setattr(_cat, "_has_squish_weights", lambda repo, token=None: False)

        download_calls = {"n": 0}

        def _fake_hf_download(repo: str, local_dir: Path, token=None):
            download_calls["n"] += 1
            local_dir.mkdir(parents=True, exist_ok=True)
            (local_dir / "config.json").write_text("{}", encoding="utf-8")
            (local_dir / "model.safetensors").write_bytes(b"ok")

        monkeypatch.setattr(_cat, "_hf_download", _fake_hf_download)
        monkeypatch.setattr("subprocess.run", lambda *a, **k: types.SimpleNamespace(returncode=0))

        _cat.pull("demo:1b", models_dir=tmp_path, int4=False, quant_mode="int8")

        assert download_calls["n"] == 1
        assert (raw_dir / "weights.incomplete").exists() is False
        ok, reason = _cat._is_raw_model_dir_complete(raw_dir)
        assert ok is True
        assert reason == ""
