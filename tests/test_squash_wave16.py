"""Wave 16 — VexCache tests."""
from __future__ import annotations

import datetime
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch


class TestVexCacheInit(unittest.TestCase):
    def test_default_cache_dir(self):
        from squish.squash.vex import VexCache
        cache = VexCache()
        self.assertIsInstance(cache._cache_dir, Path)
        self.assertIn(".squish", str(cache._cache_dir))

    def test_custom_cache_dir(self):
        with tempfile.TemporaryDirectory() as td:
            from squish.squash.vex import VexCache
            cache = VexCache(cache_dir=Path(td))
            self.assertEqual(cache._cache_dir, Path(td))


class TestVexCacheStale(unittest.TestCase):
    def test_is_stale_no_manifest(self):
        """No manifest → always stale."""
        with tempfile.TemporaryDirectory() as td:
            from squish.squash.vex import VexCache
            cache = VexCache(cache_dir=Path(td))
            self.assertTrue(cache.is_stale())

    def test_is_stale_fresh_manifest(self):
        """Fresh manifest → not stale."""
        with tempfile.TemporaryDirectory() as td:
            from squish.squash.vex import VexCache
            cache = VexCache(cache_dir=Path(td))
            # Write a manifest with 'last_fetched' = now
            now_str = datetime.datetime.now(datetime.timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            )
            manifest = {"last_fetched": now_str, "url": "https://example.com/vex.json",
                        "last_modified": None, "statement_count": 0}
            Path(td).mkdir(parents=True, exist_ok=True)
            (Path(td) / "cache-manifest.json").write_text(json.dumps(manifest))
            self.assertFalse(cache.is_stale(max_age_hours=24))

    def test_is_stale_old_manifest(self):
        """Old manifest (>24h) → stale."""
        with tempfile.TemporaryDirectory() as td:
            from squish.squash.vex import VexCache
            cache = VexCache(cache_dir=Path(td))
            old_ts = (
                datetime.datetime.now(datetime.timezone.utc)
                - datetime.timedelta(hours=48)
            ).strftime("%Y-%m-%dT%H:%M:%SZ")
            manifest = {"last_fetched": old_ts, "url": "https://example.com",
                        "last_modified": None, "statement_count": 0}
            (Path(td) / "cache-manifest.json").write_text(json.dumps(manifest))
            self.assertTrue(cache.is_stale(max_age_hours=24))


class TestVexCacheManifest(unittest.TestCase):
    def test_manifest_returns_dict(self):
        with tempfile.TemporaryDirectory() as td:
            from squish.squash.vex import VexCache
            cache = VexCache(cache_dir=Path(td))
            result = cache.manifest()
            self.assertIsInstance(result, dict)

    def test_manifest_empty_when_no_file(self):
        with tempfile.TemporaryDirectory() as td:
            from squish.squash.vex import VexCache
            cache = VexCache(cache_dir=Path(td))
            result = cache.manifest()
            self.assertEqual(result, {})

    def test_manifest_returns_contents(self):
        with tempfile.TemporaryDirectory() as td:
            from squish.squash.vex import VexCache
            cache = VexCache(cache_dir=Path(td))
            data = {"url": "https://example.com", "statement_count": 42,
                    "last_fetched": "2025-01-01T00:00:00Z", "last_modified": None}
            (Path(td) / "cache-manifest.json").write_text(json.dumps(data))
            result = cache.manifest()
            self.assertEqual(result.get("statement_count"), 42)


class TestVexCacheClear(unittest.TestCase):
    def test_clear_removes_files(self):
        with tempfile.TemporaryDirectory() as td:
            from squish.squash.vex import VexCache
            cache_dir = Path(td) / "vex-cache"
            cache_dir.mkdir()
            (cache_dir / "cache-manifest.json").write_text("{}")
            (cache_dir / "some-feed.json").write_text("{}")
            cache = VexCache(cache_dir=cache_dir)
            cache.clear()
            self.assertFalse((cache_dir / "cache-manifest.json").exists())

    def test_clear_no_error_when_empty(self):
        with tempfile.TemporaryDirectory() as td:
            from squish.squash.vex import VexCache
            cache = VexCache(cache_dir=Path(td) / "empty-cache")
            # Should not raise even if dir doesn't exist
            try:
                cache.clear()
            except Exception as e:
                self.fail(f"clear() raised unexpectedly: {e}")


class TestVexCacheLoadOrFetch(unittest.TestCase):
    def test_load_or_fetch_fetches_when_stale(self):
        """load_or_fetch calls _fetch when cache is stale."""
        with tempfile.TemporaryDirectory() as td:
            from squish.squash.vex import VexCache, VexFeed
            cache = VexCache(cache_dir=Path(td))

            mock_feed = VexFeed.empty()
            with patch.object(cache, "_fetch", return_value=mock_feed) as mock_f:
                result = cache.load_or_fetch("https://example.com/vex.json")
            mock_f.assert_called_once()
            self.assertIsInstance(result, VexFeed)

    def test_load_or_fetch_returns_vex_feed(self):
        with tempfile.TemporaryDirectory() as td:
            from squish.squash.vex import VexCache, VexFeed
            cache = VexCache(cache_dir=Path(td))
            with patch.object(cache, "_fetch", return_value=VexFeed.empty()):
                result = cache.load_or_fetch("https://example.com/vex.json")
            self.assertIsInstance(result, VexFeed)


class TestVexCacheDtypeContracts(unittest.TestCase):
    def test_manifest_type(self):
        with tempfile.TemporaryDirectory() as td:
            from squish.squash.vex import VexCache
            cache = VexCache(cache_dir=Path(td))
            self.assertIsInstance(cache.manifest(), dict)

    def test_is_stale_type(self):
        with tempfile.TemporaryDirectory() as td:
            from squish.squash.vex import VexCache
            cache = VexCache(cache_dir=Path(td))
            self.assertIsInstance(cache.is_stale(), bool)


if __name__ == "__main__":
    unittest.main()
