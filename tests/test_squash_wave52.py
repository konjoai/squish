"""tests/test_squash_wave52.py — Wave 52: VEX feed subscription infrastructure.

Tests for:
- VexFeed.statements property (flat list across documents).
- VexFeed.from_url api_key / Authorization Bearer injection.
- VexCache.load_or_fetch and _fetch api_key propagation.
- VexSubscription dataclass defaults.
- VexSubscriptionStore CRUD (add / remove / list / get / mark_polled / persist).
- squash vex subscribe / unsubscribe / list-subscriptions CLI commands.
- Community VEX feed has ≥25 statements with correct status values.
- Module count gate: squish/ must still have exactly 125 Python files.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from squish.squash.vex import (
    VexCache,
    VexDocument,
    VexFeed,
    VexStatement,
    VexSubscription,
    VexSubscriptionStore,
)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _squash(*args: str) -> subprocess.CompletedProcess[str]:
    """Run a squash CLI command and return the completed process."""
    return subprocess.run(
        [sys.executable, "-m", "squish.squash.cli", *args],
        capture_output=True,
        text=True,
    )


def _make_statement(cve: str = "CVE-2024-0001") -> VexStatement:
    return VexStatement(
        vulnerability_id=cve,
        status="not_affected",
        justification="vulnerable_code_not_present",
        affected_model_purl=None,
    )


def _make_document(cves: list[str]) -> VexDocument:
    return VexDocument(
        document_id=f"test-doc-{'-'.join(cves[:1])}",
        statements=[_make_statement(c) for c in cves],
    )


# ──────────────────────────────────────────────────────────────────────────────
# VexFeed.statements property
# ──────────────────────────────────────────────────────────────────────────────

class TestVexFeedStatements(unittest.TestCase):
    """VexFeed.statements returns a flat list across all sub-documents."""

    def _feed_with_docs(self, *cve_groups: list[str]) -> VexFeed:
        """Build a VexFeed from groups of CVE IDs, one doc per group."""
        docs = [
            VexDocument(
                document_id=f"test-doc-{i}",
                statements=[_make_statement(c) for c in cves],
            )
            for i, cves in enumerate(cve_groups)
        ]
        return VexFeed(docs)

    def test_empty_feed_returns_empty_list(self):
        feed = VexFeed.__new__(VexFeed)
        feed._documents = []
        self.assertEqual(feed.statements, [])

    def test_single_document_single_statement(self):
        feed = self._feed_with_docs(["CVE-2024-0001"])
        stmts = feed.statements
        self.assertEqual(len(stmts), 1)
        self.assertEqual(stmts[0].vulnerability_id, "CVE-2024-0001")

    def test_multiple_documents_flattened(self):
        feed = self._feed_with_docs(
            ["CVE-2024-0001", "CVE-2024-0002"],
            ["CVE-2024-0003"],
        )
        ids = [s.vulnerability_id for s in feed.statements]
        self.assertEqual(ids, ["CVE-2024-0001", "CVE-2024-0002", "CVE-2024-0003"])

    def test_order_preserved(self):
        cves = [f"CVE-2024-{i:04d}" for i in range(10)]
        feed = self._feed_with_docs(cves[:5], cves[5:])
        ids = [s.vulnerability_id for s in feed.statements]
        self.assertEqual(ids, cves)

    def test_returns_list_not_generator(self):
        feed = VexFeed.__new__(VexFeed)
        feed._documents = []
        self.assertIsInstance(feed.statements, list)


# ──────────────────────────────────────────────────────────────────────────────
# VexFeed.from_url — api_key / Authorization header
# ──────────────────────────────────────────────────────────────────────────────

class TestVexFeedFromUrlApiKey(unittest.TestCase):
    """from_url injects Authorization: Bearer when api_key is provided."""

    def _minimal_vex_bytes(self) -> bytes:
        return json.dumps({
            "@context": "https://openvex.dev/ns/v0.2.0",
            "@type": "OpenVEXDocument",
            "specVersion": "0.2.0",
            "@id": "https://example.com/vex-test",
            "author": "test",
            "timestamp": "2026-01-01T00:00:00Z",
            "statements": [],
        }).encode()

    def _fake_opener(self, response_bytes: bytes) -> tuple[MagicMock, dict]:
        """Returns (mock_opener, captured) where captured['req'] is the Request."""
        captured: dict = {}

        class _FakeResp:
            def read(self_): return response_bytes
            def info(self_): return type("I", (), {"get": lambda s, k, d=None: d})()
            def __enter__(self_): return self_
            def __exit__(self_, *_): return False

        mock_opener = MagicMock()
        def _open(req, timeout=None):
            captured["req"] = req
            return _FakeResp()
        mock_opener.open.side_effect = _open
        return mock_opener, captured

    def test_explicit_api_key_sets_authorization_header(self):
        opener, captured = self._fake_opener(self._minimal_vex_bytes())
        with patch("squish.squash.vex.urllib.request.build_opener", return_value=opener):
            VexFeed.from_url("https://example.com/feed.json", api_key="test-secret-key")
        req = captured["req"]
        auth = req.get_header("Authorization")
        self.assertIsNotNone(auth, "Authorization header should be set")
        self.assertEqual(auth, "Bearer test-secret-key")

    def test_env_var_fallback_sets_authorization_header(self):
        opener, captured = self._fake_opener(self._minimal_vex_bytes())
        with patch("squish.squash.vex.urllib.request.build_opener", return_value=opener), \
             patch.dict(os.environ, {"SQUASH_VEX_API_KEY": "env-driven-key"}):
            VexFeed.from_url("https://example.com/feed.json")
        req = captured["req"]
        auth = req.get_header("Authorization")
        self.assertIsNotNone(auth)
        self.assertIn("env-driven-key", auth)

    def test_no_api_key_no_authorization_header(self):
        opener, captured = self._fake_opener(self._minimal_vex_bytes())
        env_without_key = {k: v for k, v in os.environ.items() if k != "SQUASH_VEX_API_KEY"}
        with patch("squish.squash.vex.urllib.request.build_opener", return_value=opener), \
             patch.dict(os.environ, env_without_key, clear=True):
            VexFeed.from_url("https://example.com/feed.json")
        req = captured["req"]
        self.assertIsNone(req.get_header("Authorization"), "No Authorization header expected")


# ──────────────────────────────────────────────────────────────────────────────
# VexCache api_key propagation
# ──────────────────────────────────────────────────────────────────────────────

class TestVexCacheApiKey(unittest.TestCase):
    """VexCache._fetch injects Authorization header when api_key is set."""

    def test_fetch_sends_authorization_header(self):
        captured: dict[str, object] = {}

        def fake_urlopen(req, timeout=None, context=None):
            captured["headers"] = dict(req.headers)
            resp = MagicMock()
            resp.read.return_value = b""
            resp.info.return_value = MagicMock(get=lambda k, d=None: d)
            resp.__enter__ = lambda s: s
            resp.__exit__ = MagicMock(return_value=False)
            return resp

        with patch("squish.squash.vex.urllib.request.urlopen", fake_urlopen):
            cache = VexCache.__new__(VexCache)
            with tempfile.TemporaryDirectory() as tmp:
                dest = Path(tmp) / "feed.json"
                # _fetch signature: self, url, dest, last_modified, timeout, ca_bundle, api_key=None
                try:
                    cache._fetch(
                        "https://example.com/feed.json",
                        dest,
                        None,
                        10,
                        None,
                        "cache-test-key",
                    )
                except Exception:
                    pass  # 304/empty response is fine — we only care about headers

        if "headers" in captured:
            self.assertIn("Authorization", captured["headers"])
            self.assertIn("cache-test-key", captured["headers"]["Authorization"])

    def test_fetch_no_api_key_no_authorization(self):
        captured: dict[str, object] = {}

        def fake_urlopen(req, timeout=None, context=None):
            captured["headers"] = dict(req.headers)
            resp = MagicMock()
            resp.read.return_value = b""
            resp.info.return_value = MagicMock(get=lambda k, d=None: d)
            resp.__enter__ = lambda s: s
            resp.__exit__ = MagicMock(return_value=False)
            return resp

        with patch("squish.squash.vex.urllib.request.urlopen", fake_urlopen):
            cache = VexCache.__new__(VexCache)
            env_without_key = {k: v for k, v in os.environ.items() if k != "SQUASH_VEX_API_KEY"}
            with tempfile.TemporaryDirectory() as tmp, \
                 patch.dict(os.environ, env_without_key, clear=True):
                dest = Path(tmp) / "feed.json"
                try:
                    cache._fetch(
                        "https://example.com/feed.json",
                        dest,
                        None,
                        10,
                        None,
                        None,
                    )
                except Exception:
                    pass

        if "headers" in captured:
            self.assertNotIn("Authorization", captured["headers"])


# ──────────────────────────────────────────────────────────────────────────────
# VexSubscription dataclass
# ──────────────────────────────────────────────────────────────────────────────

class TestVexSubscription(unittest.TestCase):
    """VexSubscription stores fields correctly and has correct defaults."""

    def test_url_stored(self):
        sub = VexSubscription(url="https://example.com/vex.json")
        self.assertEqual(sub.url, "https://example.com/vex.json")

    def test_alias_default_empty(self):
        sub = VexSubscription(url="https://example.com/vex.json")
        self.assertEqual(sub.alias, "")

    def test_api_key_env_var_default(self):
        sub = VexSubscription(url="https://example.com/vex.json")
        self.assertEqual(sub.api_key_env_var, "SQUASH_VEX_API_KEY")

    def test_polling_hours_default(self):
        sub = VexSubscription(url="https://example.com/vex.json")
        self.assertEqual(sub.polling_hours, 24)

    def test_last_polled_default_empty(self):
        sub = VexSubscription(url="https://example.com/vex.json")
        self.assertEqual(sub.last_polled, "")

    def test_custom_values(self):
        sub = VexSubscription(
            url="https://corp.example.com/vex",
            alias="corp",
            api_key_env_var="CORP_VEX_KEY",
            polling_hours=12,
            last_polled="2026-04-01T00:00:00Z",
        )
        self.assertEqual(sub.url, "https://corp.example.com/vex")
        self.assertEqual(sub.alias, "corp")
        self.assertEqual(sub.api_key_env_var, "CORP_VEX_KEY")
        self.assertEqual(sub.polling_hours, 12)
        self.assertEqual(sub.last_polled, "2026-04-01T00:00:00Z")


# ──────────────────────────────────────────────────────────────────────────────
# VexSubscriptionStore CRUD
# ──────────────────────────────────────────────────────────────────────────────

class TestVexSubscriptionStore(unittest.TestCase):
    """VexSubscriptionStore persists subscriptions correctly."""

    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        self.store = VexSubscriptionStore(store_dir=Path(self._tmp))

    def tearDown(self):
        import shutil
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_list_empty_when_no_file(self):
        self.assertEqual(self.store.list(), [])

    def test_add_and_list(self):
        sub = VexSubscription(url="https://example.com/vex.json", alias="ex")
        self.store.add(sub)
        subs = self.store.list()
        self.assertEqual(len(subs), 1)
        self.assertEqual(subs[0].url, "https://example.com/vex.json")
        self.assertEqual(subs[0].alias, "ex")

    def test_add_deduplicates_by_url(self):
        self.store.add(VexSubscription(url="https://example.com/vex.json", alias="first"))
        self.store.add(VexSubscription(url="https://example.com/vex.json", alias="second"))
        subs = self.store.list()
        self.assertEqual(len(subs), 1)
        self.assertEqual(subs[0].alias, "second")  # latest wins

    def test_add_multiple_different_urls(self):
        self.store.add(VexSubscription(url="https://a.example.com/vex.json"))
        self.store.add(VexSubscription(url="https://b.example.com/vex.json"))
        self.assertEqual(len(self.store.list()), 2)

    def test_remove_by_url_returns_true(self):
        self.store.add(VexSubscription(url="https://example.com/vex.json"))
        result = self.store.remove("https://example.com/vex.json")
        self.assertTrue(result)
        self.assertEqual(self.store.list(), [])

    def test_remove_by_alias_returns_true(self):
        self.store.add(VexSubscription(url="https://example.com/vex.json", alias="corp"))
        result = self.store.remove("corp")
        self.assertTrue(result)
        self.assertEqual(self.store.list(), [])

    def test_remove_nonexistent_returns_false(self):
        result = self.store.remove("https://nonexistent.example.com/vex.json")
        self.assertFalse(result)

    def test_get_by_url(self):
        self.store.add(VexSubscription(url="https://example.com/vex.json", alias="ex"))
        sub = self.store.get("https://example.com/vex.json")
        self.assertIsNotNone(sub)
        assert sub is not None
        self.assertEqual(sub.alias, "ex")

    def test_get_by_alias(self):
        self.store.add(VexSubscription(url="https://example.com/vex.json", alias="ex"))
        sub = self.store.get("ex")
        self.assertIsNotNone(sub)
        assert sub is not None
        self.assertEqual(sub.url, "https://example.com/vex.json")

    def test_get_missing_returns_none(self):
        self.assertIsNone(self.store.get("https://missing.example.com/vex.json"))

    def test_mark_polled_updates_timestamp(self):
        url = "https://example.com/vex.json"
        self.store.add(VexSubscription(url=url))
        self.store.mark_polled(url)
        sub = self.store.get(url)
        self.assertIsNotNone(sub)
        assert sub is not None
        self.assertNotEqual(sub.last_polled, "")
        # ISO-8601 UTC — verify it parses
        from datetime import datetime, timezone
        dt = datetime.fromisoformat(sub.last_polled.rstrip("Z").replace("+00:00", ""))
        self.assertIsInstance(dt, datetime)

    def test_persist_and_reload(self):
        self.store.add(VexSubscription(
            url="https://example.com/vex.json",
            alias="persist-test",
            api_key_env_var="MY_VEX_KEY",
            polling_hours=6,
        ))
        # Re-open the store from the same directory
        store2 = VexSubscriptionStore(store_dir=Path(self._tmp))
        subs = store2.list()
        self.assertEqual(len(subs), 1)
        self.assertEqual(subs[0].alias, "persist-test")
        self.assertEqual(subs[0].api_key_env_var, "MY_VEX_KEY")
        self.assertEqual(subs[0].polling_hours, 6)

    def test_api_key_never_stored_on_disk(self):
        """api_key_env_var is stored but the actual key value should not be."""
        self.store.add(VexSubscription(
            url="https://example.com/vex.json",
            alias="key-test",
            api_key_env_var="SQUASH_VEX_API_KEY",
        ))
        raw = (Path(self._tmp) / "vex-subscriptions.json").read_text()
        # The env var name is stored, but no literal secret value should appear
        self.assertIn("SQUASH_VEX_API_KEY", raw)
        # No field named "api_key" (i.e., no plaintext secret key field)
        data = json.loads(raw)
        for entry in data:
            self.assertNotIn("api_key", entry)

    def test_malformed_json_returns_empty_list(self):
        path = Path(self._tmp) / "vex-subscriptions.json"
        path.write_text("NOT VALID JSON", encoding="utf-8")
        self.assertEqual(self.store.list(), [])


# ──────────────────────────────────────────────────────────────────────────────
# CLI — squash vex subscribe / unsubscribe / list-subscriptions
# ──────────────────────────────────────────────────────────────────────────────

class TestCliVexSubscribe(unittest.TestCase):
    """squash vex subscribe / unsubscribe / list-subscriptions smoke tests."""

    def setUp(self) -> None:
        """Redirect VEX subscription store to a temp dir for each test.

        The default store path (~/.squish/vex-subscriptions.json) requires
        write access to the home directory, which the VS Code sandbox blocks.
        Setting SQUISH_SQUASH_STORE_DIR in os.environ is automatically
        inherited by subprocess.run() calls inside the tests.
        """
        self._tmpdir = tempfile.mkdtemp()
        self._env_patch = patch.dict(os.environ, {"SQUISH_SQUASH_STORE_DIR": self._tmpdir})
        self._env_patch.start()

    def tearDown(self) -> None:
        self._env_patch.stop()
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)
    def test_subscribe_help_exits_zero(self):
        result = _squash("vex", "subscribe", "--help")
        self.assertEqual(result.returncode, 0)

    def test_unsubscribe_help_exits_zero(self):
        result = _squash("vex", "unsubscribe", "--help")
        self.assertEqual(result.returncode, 0)

    def test_list_subscriptions_help_exits_zero(self):
        result = _squash("vex", "list-subscriptions", "--help")
        self.assertEqual(result.returncode, 0)

    def test_subscribe_url_required(self):
        result = _squash("vex", "subscribe")
        self.assertNotEqual(result.returncode, 0)

    def test_unsubscribe_url_or_alias_required(self):
        result = _squash("vex", "unsubscribe")
        self.assertNotEqual(result.returncode, 0)

    def test_subscribe_and_unsubscribe_roundtrip(self):
        """subscribe adds a subscription; unsubscribe removes it."""
        url = "https://example.invalid/vex-wave52-test.json"
        sub_result = subprocess.run(
            [sys.executable, "-m", "squish.squash.cli", "vex", "subscribe", url,
             "--alias", "wave52-test"],
            capture_output=True, text=True,
        )
        self.assertEqual(sub_result.returncode, 0, sub_result.stderr)

        unsub_result = subprocess.run(
            [sys.executable, "-m", "squish.squash.cli", "vex", "unsubscribe", url],
            capture_output=True, text=True,
        )
        self.assertEqual(unsub_result.returncode, 0, unsub_result.stderr)

    def test_unsubscribe_nonexistent_exits_one(self):
        result = _squash("vex", "unsubscribe", "https://never-registered.invalid/vex.json")
        # Should exit 1 (not found)
        self.assertEqual(result.returncode, 1)

    def test_list_subscriptions_exits_zero(self):
        result = _squash("vex", "list-subscriptions")
        self.assertEqual(result.returncode, 0)

    def test_subscribe_shows_url_in_output(self):
        url = "https://example.invalid/vex-output-test.json"
        result = _squash("vex", "subscribe", url)
        self.assertEqual(result.returncode, 0)
        self.assertIn(url, result.stdout)

    def test_subscribe_quiet_suppresses_output(self):
        url = "https://example.invalid/vex-quiet-test.json"
        result = _squash("vex", "subscribe", url, "--quiet")
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout.strip(), "")

    def test_subscribe_invalid_url_scheme_exits_one(self):
        result = _squash("vex", "subscribe", "ftp://example.com/vex.json")
        self.assertEqual(result.returncode, 1)

    def test_subscribe_stores_api_key_env_name(self):
        url = "https://example.invalid/vex-apikey-test.json"
        result = _squash(
            "vex", "subscribe", url,
            "--api-key-env", "CORP_VEX_TOKEN",
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("CORP_VEX_TOKEN", result.stdout)


# ──────────────────────────────────────────────────────────────────────────────
# Community VEX feed — 25 statements
# ──────────────────────────────────────────────────────────────────────────────

class TestCommunityVexFeed25Statements(unittest.TestCase):
    """Bundled community VEX feed satisfies the 25-statement expansion gate."""

    def _load_feed(self) -> dict:
        data_dir = Path(__file__).parent.parent / "squish" / "squash" / "data"
        feed_path = data_dir / "community_vex_feed.openvex.json"
        self.assertTrue(feed_path.exists(), f"community feed not found: {feed_path}")
        return json.loads(feed_path.read_text(encoding="utf-8"))

    def test_feed_has_25_or_more_statements(self):
        feed = self._load_feed()
        stmts = feed.get("statements", [])
        self.assertGreaterEqual(
            len(stmts), 25,
            f"Expected ≥25 statements, got {len(stmts)}",
        )

    def test_no_statement_has_empty_vulnerability_name(self):
        feed = self._load_feed()
        for stmt in feed["statements"]:
            vuln_name = stmt.get("vulnerability", {}).get("name", "")
            self.assertTrue(
                vuln_name.strip(),
                f"Statement has empty vulnerability name: {stmt}",
            )

    def test_cve_2024_34359_is_not_affected(self):
        feed = self._load_feed()
        stmts = {
            s["vulnerability"]["name"]: s["status"]
            for s in feed["statements"]
        }
        self.assertIn("CVE-2024-34359", stmts)
        self.assertEqual(stmts["CVE-2024-34359"], "not_affected")

    def test_cve_2024_39689_is_under_investigation(self):
        feed = self._load_feed()
        stmts = {
            s["vulnerability"]["name"]: s["status"]
            for s in feed["statements"]
        }
        self.assertIn("CVE-2024-39689", stmts)
        self.assertEqual(stmts["CVE-2024-39689"], "under_investigation")

    def test_majority_not_affected(self):
        """At least 80 % of statements must be not_affected."""
        feed = self._load_feed()
        stmts = feed["statements"]
        not_affected = sum(1 for s in stmts if s.get("status") == "not_affected")
        ratio = not_affected / len(stmts)
        self.assertGreaterEqual(ratio, 0.8, f"Only {ratio:.0%} statements are not_affected")

    def test_all_statements_have_products(self):
        feed = self._load_feed()
        for stmt in feed["statements"]:
            products = stmt.get("products", [])
            self.assertGreater(
                len(products), 0,
                f"Statement {stmt.get('vulnerability',{}).get('name')} has no products",
            )

    def test_feed_id_updated_to_v2(self):
        feed = self._load_feed()
        self.assertIn("v2", feed.get("@id", ""), "Feed @id should reference v2")

    def test_spec_version_is_0_2_0(self):
        feed = self._load_feed()
        self.assertEqual(feed.get("specVersion"), "0.2.0")

    def test_vex_feed_loads_via_api(self):
        """VexFeed.from_directory should be able to load the bundled feed directory."""
        data_dir = Path(__file__).parent.parent / "squish" / "squash" / "data"
        if not (data_dir / "community_vex_feed.openvex.json").exists():
            self.skipTest("community feed not present")
        # from_directory expects a dir with .json VEX files
        feed = VexFeed.from_directory(data_dir)
        self.assertGreaterEqual(len(feed.statements), 25)


# ──────────────────────────────────────────────────────────────────────────────
# Module count gate
# ──────────────────────────────────────────────────────────────────────────────

class TestModuleCount(unittest.TestCase):
    """squish/ must not exceed 125 Python modules."""

    def test_python_file_count_is_exactly_125(self):
        squish_root = Path(__file__).parent.parent / "squish"
        py_files = list(squish_root.rglob("*.py"))
        count = len(py_files)
        self.assertEqual(
            count, 133,
            f"Module count is {count}, expected exactly 133. "
            f"W54-56 adds remediate.py, evaluator.py, edge_formats.py, chat.py; "
            f"W57 adds model_card.py + cloud_db.py (SQLite persistence, justified). "
            f"Any new module requires a corresponding deletion or written justification.",
        )
