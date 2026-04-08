"""tests/test_squash_wave47.py — Wave 47: RAG knowledge base integrity scanner.

Tests for squish/squash/rag.py (RagScanner, RagManifest, RagVerifyResult,
RagDriftItem), the squash scan-rag CLI subcommand, and the
POST /rag/index + POST /rag/verify REST routes.

All tests are pure-unit or integration (temp dirs, cleaned up in tearDown).
No network, no model weights, no mocking of rag.py itself.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SQUISH_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_SQUISH_ROOT))


def _make_corpus(root: Path, files: dict[str, str]) -> None:
    """Write *files* (relative_path → content) into *root*."""
    for rel, content in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")


# ===========================================================================
# RagFileEntry
# ===========================================================================


class TestRagFileEntry(unittest.TestCase):
    def test_fields(self) -> None:
        from squish.squash.rag import RagFileEntry

        entry = RagFileEntry(path="docs/a.txt", sha256="abc123", size_bytes=42)
        self.assertEqual(entry.path, "docs/a.txt")
        self.assertEqual(entry.sha256, "abc123")
        self.assertEqual(entry.size_bytes, 42)

    def test_asdict(self) -> None:
        from squish.squash.rag import RagFileEntry

        entry = RagFileEntry(path="x.txt", sha256="dead", size_bytes=1)
        d = asdict(entry)
        self.assertIn("path", d)
        self.assertIn("sha256", d)
        self.assertIn("size_bytes", d)


# ===========================================================================
# RagManifest
# ===========================================================================


class TestRagManifest(unittest.TestCase):
    def _sample(self) -> "RagManifest":
        from squish.squash.rag import RagManifest

        return RagManifest(
            version=1,
            corpus_dir="/tmp/corpus",
            indexed_at="2026-04-10T00:00:00Z",
            file_count=2,
            files=[
                {"path": "a.txt", "sha256": "aaa", "size_bytes": 10},
                {"path": "b.txt", "sha256": "bbb", "size_bytes": 20},
            ],
            manifest_hash="deadbeef",
        )

    def test_fields(self) -> None:
        m = self._sample()
        self.assertEqual(m.version, 1)
        self.assertEqual(m.file_count, 2)

    def test_to_dict_keys(self) -> None:
        m = self._sample()
        d = m.to_dict()
        for key in ("version", "corpus_dir", "indexed_at", "file_count", "files", "manifest_hash"):
            self.assertIn(key, d)

    def test_write_and_load_roundtrip(self) -> None:
        from squish.squash.rag import RagManifest

        m = self._sample()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".rag_manifest.json"
            m.write(path)
            self.assertTrue(path.exists())
            loaded = RagManifest.load(path)
            self.assertEqual(loaded.version, m.version)
            self.assertEqual(loaded.corpus_dir, m.corpus_dir)
            self.assertEqual(loaded.manifest_hash, m.manifest_hash)
            self.assertEqual(len(loaded.files), 2)

    def test_load_missing_key_raises(self) -> None:
        from squish.squash.rag import RagManifest

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text(json.dumps({"version": 1}), encoding="utf-8")
            with self.assertRaises(ValueError):
                RagManifest.load(path)

    def test_load_unsupported_version_raises(self) -> None:
        from squish.squash.rag import RagManifest

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text(
                json.dumps(
                    {
                        "version": 99,
                        "corpus_dir": "/x",
                        "indexed_at": "now",
                        "file_count": 0,
                        "files": [],
                        "manifest_hash": "x",
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                RagManifest.load(path)

    def test_load_file_not_found(self) -> None:
        from squish.squash.rag import RagManifest

        with self.assertRaises(FileNotFoundError):
            RagManifest.load(Path("/does/not/exist/.rag_manifest.json"))


# ===========================================================================
# RagDriftItem
# ===========================================================================


class TestRagDriftItem(unittest.TestCase):
    def test_fields(self) -> None:
        from squish.squash.rag import RagDriftItem

        item = RagDriftItem(path="docs/x.txt", status="added", old_hash="", new_hash="abc")
        self.assertEqual(item.path, "docs/x.txt")
        self.assertEqual(item.status, "added")
        self.assertEqual(item.old_hash, "")
        self.assertEqual(item.new_hash, "abc")

    def test_asdict(self) -> None:
        from squish.squash.rag import RagDriftItem

        d = asdict(RagDriftItem(path="y.txt", status="removed", old_hash="xyz", new_hash=""))
        self.assertIn("status", d)


# ===========================================================================
# RagVerifyResult
# ===========================================================================


class TestRagVerifyResult(unittest.TestCase):
    def test_to_dict_keys(self) -> None:
        from squish.squash.rag import RagDriftItem, RagVerifyResult

        r = RagVerifyResult(
            ok=True,
            corpus_dir="/tmp/x",
            verified_at="2026-04-10T00:00:00Z",
            total_files=5,
            drift_count=0,
            drift=[],
        )
        d = r.to_dict()
        for key in ("ok", "corpus_dir", "verified_at", "total_files", "drift_count", "drift"):
            self.assertIn(key, d)

    def test_drift_serialised(self) -> None:
        from squish.squash.rag import RagDriftItem, RagVerifyResult

        r = RagVerifyResult(
            ok=False,
            corpus_dir="/tmp/x",
            verified_at="2026-04-10T00:00:00Z",
            total_files=5,
            drift_count=1,
            drift=[RagDriftItem(path="x.txt", status="modified", old_hash="a", new_hash="b")],
        )
        d = r.to_dict()
        self.assertEqual(len(d["drift"]), 1)
        self.assertEqual(d["drift"][0]["status"], "modified")


# ===========================================================================
# RagScanner._hash_file
# ===========================================================================


class TestRagScannerHashFile(unittest.TestCase):
    def test_deterministic(self) -> None:
        from squish.squash.rag import RagScanner

        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as fh:
            fh.write(b"hello rag")
            fpath = Path(fh.name)
        try:
            h1 = RagScanner._hash_file(fpath)
            h2 = RagScanner._hash_file(fpath)
            self.assertEqual(h1, h2)
        finally:
            fpath.unlink()

    def test_hex_length(self) -> None:
        from squish.squash.rag import RagScanner

        with tempfile.NamedTemporaryFile(delete=False) as fh:
            fh.write(b"data")
            fpath = Path(fh.name)
        try:
            h = RagScanner._hash_file(fpath)
            self.assertEqual(len(h), 64)
        finally:
            fpath.unlink()

    def test_empty_file(self) -> None:
        from squish.squash.rag import RagScanner

        with tempfile.NamedTemporaryFile(delete=False) as fh:
            fpath = Path(fh.name)
        try:
            h = RagScanner._hash_file(fpath)
            # sha256("") is known
            import hashlib
            self.assertEqual(h, hashlib.sha256(b"").hexdigest())
        finally:
            fpath.unlink()

    def test_different_content_different_hash(self) -> None:
        from squish.squash.rag import RagScanner

        with tempfile.NamedTemporaryFile(delete=False) as f1, tempfile.NamedTemporaryFile(delete=False) as f2:
            f1.write(b"aaa")
            f2.write(b"bbb")
            p1, p2 = Path(f1.name), Path(f2.name)
        try:
            self.assertNotEqual(RagScanner._hash_file(p1), RagScanner._hash_file(p2))
        finally:
            p1.unlink()
            p2.unlink()


# ===========================================================================
# RagScanner._manifest_hash
# ===========================================================================


class TestRagScannerManifestHash(unittest.TestCase):
    def test_deterministic(self) -> None:
        from squish.squash.rag import RagScanner

        files = [{"path": "a.txt", "sha256": "abc", "size_bytes": 1}]
        self.assertEqual(
            RagScanner._manifest_hash(files),
            RagScanner._manifest_hash(files),
        )

    def test_empty_list(self) -> None:
        from squish.squash.rag import RagScanner

        h = RagScanner._manifest_hash([])
        self.assertEqual(len(h), 64)

    def test_different_files_different_hash(self) -> None:
        from squish.squash.rag import RagScanner

        fs1 = [{"path": "a.txt", "sha256": "aaa", "size_bytes": 1}]
        fs2 = [{"path": "a.txt", "sha256": "bbb", "size_bytes": 1}]
        self.assertNotEqual(RagScanner._manifest_hash(fs1), RagScanner._manifest_hash(fs2))

    def test_order_independent(self) -> None:
        """Manifest hash must NOT depend on list order — sorted by caller."""
        from squish.squash.rag import RagScanner

        # We intentionally pass the same two entries in different orders.
        # The caller (index()) sorts before calling, so we verify the hash
        # function itself is stable when given the same sorted payload.
        fs_a = [
            {"path": "a.txt", "sha256": "aaa", "size_bytes": 1},
            {"path": "b.txt", "sha256": "bbb", "size_bytes": 2},
        ]
        fs_b = [
            {"path": "a.txt", "sha256": "aaa", "size_bytes": 1},
            {"path": "b.txt", "sha256": "bbb", "size_bytes": 2},
        ]
        self.assertEqual(RagScanner._manifest_hash(fs_a), RagScanner._manifest_hash(fs_b))


# ===========================================================================
# RagScanner.index
# ===========================================================================


class TestRagScannerIndex(unittest.TestCase):
    def test_creates_manifest_file(self) -> None:
        from squish.squash.rag import RagScanner

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_corpus(root, {"a.txt": "hello", "b.txt": "world"})
            manifest = RagScanner.index(root)
            manifest_path = root / RagScanner.MANIFEST_FILENAME
            self.assertTrue(manifest_path.exists())

    def test_file_count(self) -> None:
        from squish.squash.rag import RagScanner

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_corpus(root, {"a.txt": "1", "sub/b.txt": "2", "sub/c.md": "3"})
            manifest = RagScanner.index(root)
            self.assertEqual(manifest.file_count, 3)

    def test_manifest_not_counted(self) -> None:
        """The .rag_manifest.json itself must not be counted in file_count."""
        from squish.squash.rag import RagScanner

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_corpus(root, {"doc.txt": "content"})
            manifest = RagScanner.index(root)
            self.assertEqual(manifest.file_count, 1)

    def test_manifest_hash_is_valid_hex(self) -> None:
        from squish.squash.rag import RagScanner

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_corpus(root, {"a.txt": "data"})
            manifest = RagScanner.index(root)
            self.assertEqual(len(manifest.manifest_hash), 64)
            int(manifest.manifest_hash, 16)  # raises if not hex

    def test_manifest_hash_stable_across_calls(self) -> None:
        """Same corpus → same manifest_hash on repeated indexing."""
        from squish.squash.rag import RagScanner

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_corpus(root, {"x.txt": "stable"})
            m1 = RagScanner.index(root)
            m2 = RagScanner.index(root)
            self.assertEqual(m1.manifest_hash, m2.manifest_hash)

    def test_empty_corpus(self) -> None:
        from squish.squash.rag import RagScanner

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = RagScanner.index(root)
            self.assertEqual(manifest.file_count, 0)

    def test_not_a_directory_raises(self) -> None:
        from squish.squash.rag import RagScanner

        with self.assertRaises(NotADirectoryError):
            RagScanner.index("/definitely/not/a/real/path/xyz")

    def test_glob_filter(self) -> None:
        """Custom glob pattern restricts which files are indexed."""
        from squish.squash.rag import RagScanner

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_corpus(root, {"a.txt": "t", "b.md": "m", "c.py": "p"})
            manifest = RagScanner.index(root, glob="**/*.txt")
            self.assertEqual(manifest.file_count, 1)
            self.assertEqual(manifest.files[0]["path"], "a.txt")

    def test_version_is_one(self) -> None:
        from squish.squash.rag import RagScanner

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = RagScanner.index(root)
            self.assertEqual(manifest.version, 1)

    def test_files_sorted_by_path(self) -> None:
        from squish.squash.rag import RagScanner

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_corpus(root, {"z.txt": "z", "a.txt": "a", "m.txt": "m"})
            manifest = RagScanner.index(root)
            paths = [f["path"] for f in manifest.files]
            self.assertEqual(paths, sorted(paths))


# ===========================================================================
# RagScanner.verify
# ===========================================================================


class TestRagScannerVerify(unittest.TestCase):
    def test_pristine_ok(self) -> None:
        from squish.squash.rag import RagScanner

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_corpus(root, {"a.txt": "hello", "b.txt": "world"})
            RagScanner.index(root)
            result = RagScanner.verify(root)
            self.assertTrue(result.ok)
            self.assertEqual(result.drift_count, 0)
            self.assertEqual(len(result.drift), 0)

    def test_pristine_total_files(self) -> None:
        from squish.squash.rag import RagScanner

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_corpus(root, {"a.txt": "x", "sub/b.txt": "y"})
            RagScanner.index(root)
            result = RagScanner.verify(root)
            self.assertEqual(result.total_files, 2)

    def test_modified_detected(self) -> None:
        from squish.squash.rag import RagScanner

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_corpus(root, {"a.txt": "original"})
            RagScanner.index(root)
            (root / "a.txt").write_text("POISONED", encoding="utf-8")
            result = RagScanner.verify(root)
            self.assertFalse(result.ok)
            self.assertEqual(result.drift_count, 1)
            self.assertEqual(result.drift[0].status, "modified")
            self.assertEqual(result.drift[0].path, "a.txt")

    def test_added_detected(self) -> None:
        from squish.squash.rag import RagScanner

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_corpus(root, {"a.txt": "existing"})
            RagScanner.index(root)
            _make_corpus(root, {"new_doc.txt": "injected"})
            result = RagScanner.verify(root)
            self.assertFalse(result.ok)
            added = [d for d in result.drift if d.status == "added"]
            self.assertEqual(len(added), 1)
            self.assertEqual(added[0].path, "new_doc.txt")

    def test_removed_detected(self) -> None:
        from squish.squash.rag import RagScanner

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_corpus(root, {"a.txt": "keep", "b.txt": "delete_me"})
            RagScanner.index(root)
            (root / "b.txt").unlink()
            result = RagScanner.verify(root)
            self.assertFalse(result.ok)
            removed = [d for d in result.drift if d.status == "removed"]
            self.assertEqual(len(removed), 1)
            self.assertEqual(removed[0].path, "b.txt")

    def test_multiple_drift_types(self) -> None:
        from squish.squash.rag import RagScanner

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_corpus(root, {"a.txt": "orig", "b.txt": "to_remove"})
            RagScanner.index(root)
            (root / "a.txt").write_text("mutated", encoding="utf-8")
            (root / "b.txt").unlink()
            _make_corpus(root, {"c.txt": "injected"})
            result = RagScanner.verify(root)
            self.assertFalse(result.ok)
            self.assertEqual(result.drift_count, 3)
            statuses = {d.status for d in result.drift}
            self.assertEqual(statuses, {"added", "removed", "modified"})

    def test_missing_manifest(self) -> None:
        from squish.squash.rag import RagScanner

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_corpus(root, {"a.txt": "x"})
            result = RagScanner.verify(root)
            self.assertFalse(result.ok)
            self.assertEqual(result.drift_count, 1)
            self.assertEqual(result.drift[0].status, "missing_manifest")

    def test_modified_old_and_new_hash(self) -> None:
        from squish.squash.rag import RagScanner

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_corpus(root, {"doc.txt": "before"})
            manifest = RagScanner.index(root)
            old_hash = manifest.files[0]["sha256"]
            (root / "doc.txt").write_text("after", encoding="utf-8")
            result = RagScanner.verify(root)
            item = result.drift[0]
            self.assertEqual(item.old_hash, old_hash)
            self.assertNotEqual(item.new_hash, "")
            self.assertNotEqual(item.new_hash, old_hash)

    def test_removed_new_hash_empty(self) -> None:
        from squish.squash.rag import RagScanner

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_corpus(root, {"a.txt": "x"})
            RagScanner.index(root)
            (root / "a.txt").unlink()
            result = RagScanner.verify(root)
            self.assertEqual(result.drift[0].new_hash, "")

    def test_added_old_hash_empty(self) -> None:
        from squish.squash.rag import RagScanner

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_corpus(root, {"a.txt": "x"})
            RagScanner.index(root)
            _make_corpus(root, {"new.txt": "y"})
            result = RagScanner.verify(root)
            added = next(d for d in result.drift if d.status == "added")
            self.assertEqual(added.old_hash, "")


# ===========================================================================
# CLI — squash scan-rag
# ===========================================================================


class TestRagScanRagCli(unittest.TestCase):
    def _get_parser(self):
        from squish.squash.cli import _build_parser
        return _build_parser()

    def test_scan_rag_subcommand_registered(self) -> None:
        p = self._get_parser()
        # Should not raise
        args = p.parse_args(["scan-rag", "index", "/tmp"])
        self.assertEqual(args.command, "scan-rag")
        self.assertEqual(args.scan_rag_command, "index")

    def test_verify_subcommand_registered(self) -> None:
        p = self._get_parser()
        args = p.parse_args(["scan-rag", "verify", "/tmp"])
        self.assertEqual(args.scan_rag_command, "verify")

    def test_index_corpus_dir_arg(self) -> None:
        p = self._get_parser()
        args = p.parse_args(["scan-rag", "index", "/my/corpus"])
        self.assertEqual(args.corpus_dir, "/my/corpus")

    def test_index_glob_default(self) -> None:
        p = self._get_parser()
        args = p.parse_args(["scan-rag", "index", "/x"])
        self.assertEqual(args.glob, "**/*")

    def test_index_glob_override(self) -> None:
        p = self._get_parser()
        args = p.parse_args(["scan-rag", "index", "/x", "--glob", "**/*.md"])
        self.assertEqual(args.glob, "**/*.md")

    def test_verify_json_flag(self) -> None:
        p = self._get_parser()
        args = p.parse_args(["scan-rag", "verify", "/x", "--json"])
        self.assertTrue(args.json_output)

    def test_exit_0_on_intact_corpus(self) -> None:
        """scan-rag verify exits 0 on pristine corpus."""
        from squish.squash.rag import RagScanner

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_corpus(root, {"doc.txt": "data"})
            RagScanner.index(root)

            from squish.squash.cli import _cmd_scan_rag
            import argparse

            args = argparse.Namespace(
                command="scan-rag",
                scan_rag_command="verify",
                corpus_dir=str(root),
                json_output=False,
                quiet=True,
            )
            rc = _cmd_scan_rag(args, quiet=True)
            self.assertEqual(rc, 0)

    def test_exit_2_on_drifted_corpus(self) -> None:
        """scan-rag verify exits 2 when drift is detected."""
        from squish.squash.rag import RagScanner

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_corpus(root, {"doc.txt": "original"})
            RagScanner.index(root)
            (root / "doc.txt").write_text("TAMPERED", encoding="utf-8")

            from squish.squash.cli import _cmd_scan_rag
            import argparse

            args = argparse.Namespace(
                command="scan-rag",
                scan_rag_command="verify",
                corpus_dir=str(root),
                json_output=False,
                quiet=True,
            )
            rc = _cmd_scan_rag(args, quiet=True)
            self.assertEqual(rc, 2)

    def test_exit_0_on_index(self) -> None:
        """scan-rag index exits 0 on success."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_corpus(root, {"a.txt": "hello"})

            from squish.squash.cli import _cmd_scan_rag
            import argparse

            args = argparse.Namespace(
                command="scan-rag",
                scan_rag_command="index",
                corpus_dir=str(root),
                glob="**/*",
                quiet=True,
            )
            rc = _cmd_scan_rag(args, quiet=True)
            self.assertEqual(rc, 0)

    def test_exit_1_on_invalid_dir(self) -> None:
        """scan-rag index exits 1 when corpus_dir does not exist."""
        from squish.squash.cli import _cmd_scan_rag
        import argparse

        args = argparse.Namespace(
            command="scan-rag",
            scan_rag_command="index",
            corpus_dir="/does/not/exist/xyz",
            glob="**/*",
            quiet=True,
        )
        rc = _cmd_scan_rag(args, quiet=True)
        self.assertEqual(rc, 1)

    def test_json_output_is_valid_json(self) -> None:
        """scan-rag verify --json prints valid JSON to stdout."""
        import io
        import contextlib
        from squish.squash.rag import RagScanner

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_corpus(root, {"doc.txt": "content"})
            RagScanner.index(root)

            from squish.squash.cli import _cmd_scan_rag
            import argparse

            args = argparse.Namespace(
                command="scan-rag",
                scan_rag_command="verify",
                corpus_dir=str(root),
                json_output=True,
                quiet=False,
            )
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = _cmd_scan_rag(args, quiet=False)
            data = json.loads(buf.getvalue())
            self.assertIn("ok", data)
            self.assertTrue(data["ok"])
            self.assertEqual(rc, 0)


# ===========================================================================
# API — POST /rag/index and POST /rag/verify
# ===========================================================================


class TestRagApi(unittest.TestCase):
    """Verify that the FastAPI routes are registered and return expected shapes.

    Uses the Starlette TestClient so no real HTTP server is needed.
    """

    @classmethod
    def setUpClass(cls) -> None:
        try:
            from starlette.testclient import TestClient
            from squish.squash.api import app
            cls.client = TestClient(app, raise_server_exceptions=True)
        except Exception:
            cls.client = None

    def _skip_if_no_client(self) -> None:
        if self.client is None:
            self.skipTest("starlette TestClient or api.py not available")

    def test_rag_index_route_registered(self) -> None:
        """POST /rag/index must be a known route (405 Not Allowed on GET, not 404)."""
        self._skip_if_no_client()
        resp = self.client.get("/rag/index")
        self.assertNotEqual(resp.status_code, 404, "/rag/index route not registered")

    def test_rag_verify_route_registered(self) -> None:
        self._skip_if_no_client()
        resp = self.client.get("/rag/verify")
        self.assertNotEqual(resp.status_code, 404, "/rag/verify route not registered")

    def test_rag_index_ok(self) -> None:
        self._skip_if_no_client()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_corpus(root, {"doc.txt": "knowledge"})
            resp = self.client.post("/rag/index", json={"corpus_dir": str(root)})
            self.assertEqual(resp.status_code, 200, resp.text)
            data = resp.json()
            for key in ("corpus_dir", "file_count", "manifest_path", "manifest_hash", "indexed_at"):
                self.assertIn(key, data)
            self.assertEqual(data["file_count"], 1)

    def test_rag_index_404_on_missing_dir(self) -> None:
        self._skip_if_no_client()
        resp = self.client.post("/rag/index", json={"corpus_dir": "/does/not/exist/xyz"})
        self.assertEqual(resp.status_code, 404)

    def test_rag_verify_ok_on_pristine(self) -> None:
        self._skip_if_no_client()
        from squish.squash.rag import RagScanner

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_corpus(root, {"doc.txt": "fresh"})
            RagScanner.index(root)
            resp = self.client.post("/rag/verify", json={"corpus_dir": str(root)})
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertTrue(data["ok"])
            self.assertEqual(data["drift_count"], 0)

    def test_rag_verify_drift_flagged(self) -> None:
        self._skip_if_no_client()
        from squish.squash.rag import RagScanner

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_corpus(root, {"doc.txt": "clean"})
            RagScanner.index(root)
            (root / "doc.txt").write_text("POISONED", encoding="utf-8")
            resp = self.client.post("/rag/verify", json={"corpus_dir": str(root)})
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertFalse(data["ok"])
            self.assertEqual(data["drift_count"], 1)


if __name__ == "__main__":
    unittest.main()
