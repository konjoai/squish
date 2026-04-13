"""Unit tests for squish.squash.remediate (Wave 54).

Test taxonomy: pure unit — no I/O to external services; temp dirs cleaned in tearDown.
"""

from __future__ import annotations

import hashlib
import json
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch


class TestRemediateResult(unittest.TestCase):
    """Tests for RemediateResult properties and summary."""

    def _make_result(self, converted=(), failed=()):
        from squish.squash.remediate import RemediateResult
        return RemediateResult(
            model_path=Path("/some/model"),
            target_format="safetensors",
            converted=list(converted),
            failed=list(failed),
            sbom_patch={},
        )

    def test_empty_result_is_fully_remediated(self):
        result = self._make_result()
        self.assertTrue(result.fully_remediated)
        self.assertFalse(result.partial)

    def test_partial_when_both_converted_and_failed(self):
        from squish.squash.remediate import ConvertedFile, FailedFile
        conv = ConvertedFile(
            source=Path("/m/w.bin"),
            destination=Path("/m/w.safetensors"),
            source_sha256="aa" * 32,
            destination_sha256="bb" * 32,
            tensor_count=3,
        )
        fail = FailedFile(source=Path("/m/bad.bin"), reason="load_error")
        result = self._make_result(converted=[conv], failed=[fail])
        self.assertFalse(result.fully_remediated)
        self.assertTrue(result.partial)

    def test_summary_contains_counts(self):
        from squish.squash.remediate import ConvertedFile
        conv = ConvertedFile(
            source=Path("/m/w.bin"),
            destination=Path("/m/w.safetensors"),
            source_sha256="aa" * 32,
            destination_sha256="bb" * 32,
            tensor_count=2,
        )
        result = self._make_result(converted=[conv])
        summary = result.summary()
        self.assertIn("1", summary)
        self.assertIn("safetensors", summary.lower())


class TestRemediatorFindPickleFiles(unittest.TestCase):
    """Tests for _find_pickle_files static method."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_finds_bin_pt_pth(self):
        from squish.squash.remediate import Remediator
        base = Path(self.tmpdir)
        for name in ("model.bin", "weights.pt", "state.pth", "config.json"):
            (base / name).write_text("dummy")
        found = Remediator._find_pickle_files(base)
        names = {f.name for f in found}
        self.assertIn("model.bin", names)
        self.assertIn("weights.pt", names)
        self.assertIn("state.pth", names)
        self.assertNotIn("config.json", names)

    def test_single_bin_file_input(self):
        from squish.squash.remediate import Remediator
        p = Path(self.tmpdir) / "model.bin"
        p.write_text("dummy")
        found = Remediator._find_pickle_files(p)
        self.assertEqual([p], found)

    def test_non_pickle_extension_ignored(self):
        from squish.squash.remediate import Remediator
        base = Path(self.tmpdir)
        (base / "model.safetensors").write_text("dummy")
        found = Remediator._find_pickle_files(base)
        self.assertEqual([], found)


class TestRemediatorSha256(unittest.TestCase):
    """Tests for _sha256 helper."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_sha256_matches_hashlib(self):
        from squish.squash.remediate import Remediator
        p = Path(self.tmpdir) / "data.bin"
        p.write_bytes(b"hello squash")
        expected = hashlib.sha256(b"hello squash").hexdigest()
        self.assertEqual(expected, Remediator._sha256(p))


class TestRemediatorPatchSbom(unittest.TestCase):
    """Tests for patch_sbom class method."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _make_cyclonedx_bom(self, sha: str) -> dict:
        return {
            "bomFormat": "CycloneDX",
            "components": [
                {
                    "name": "weights",
                    "hashes": [{"alg": "SHA-256", "content": sha}],
                }
            ],
        }

    def test_patch_updates_matching_hash(self):
        from squish.squash.remediate import Remediator
        old_sha = "aabb" * 16
        new_sha = "ccdd" * 16
        bom = self._make_cyclonedx_bom(old_sha)
        bom_path = Path(self.tmpdir) / "cyclonedx-mlbom.json"
        bom_path.write_text(json.dumps(bom))

        patch = {old_sha: {"new_sha256": new_sha, "new_file": "weights.safetensors"}}
        result = Remediator.patch_sbom(bom_path, patch)

        self.assertTrue(result)
        updated = json.loads(bom_path.read_text())
        hashes = updated["components"][0]["hashes"]
        self.assertTrue(any(h["content"] == new_sha for h in hashes))

    def test_patch_returns_false_for_missing_file(self):
        from squish.squash.remediate import Remediator
        result = Remediator.patch_sbom(Path(self.tmpdir) / "nonexistent.json", {"x": {"new_sha256": "y"}})
        self.assertFalse(result)

    def test_patch_returns_false_for_invalid_json(self):
        from squish.squash.remediate import Remediator
        p = Path(self.tmpdir) / "bad.json"
        p.write_text("not-json")
        result = Remediator.patch_sbom(p, {"x": {}})
        self.assertFalse(result)


class TestRemediatorConvertDryRun(unittest.TestCase):
    """Integration-style test: dry_run=True must not write any files."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_dry_run_produces_no_output_files(self):
        from squish.squash.remediate import Remediator
        base = Path(self.tmpdir)
        (base / "model.bin").write_bytes(b"\x80\x04dummy pickle bytes")

        with patch.object(Remediator, "_safe_load", side_effect=RuntimeError("torch not available")):
            result = Remediator.convert(base, target_format="safetensors", dry_run=True, overwrite=False)

        safetensors_files = list(base.glob("*.safetensors"))
        self.assertEqual([], safetensors_files)
        # dry_run means nothing should be on disk regardless of success/failure
        self.assertEqual(0, len(result.converted))


if __name__ == "__main__":
    unittest.main()
