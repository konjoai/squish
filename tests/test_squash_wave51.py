"""tests/test_squash_wave51.py — Wave 51: SBOM drift detection.

Tests for squish/squash/drift.py (DriftConfig, DriftHit, DriftResult,
check_drift) and the squash drift-check CLI command.

Coverage:
- DriftConfig default values and field types.
- DriftHit.missing / DriftHit.tampered properties.
- DriftResult.__post_init__ summary auto-build (ok and drift cases).
- _parse_bom_hashes: empty BOM, missing components key, per-file props.
- check_drift: clean model, tampered file, missing file, extra files
  ignored, invalid JSON, no hash properties → ValueError.
- CLI drift-check: clean exit-0, drift exit-0 without --fail-on-drift,
  drift exit-2 with --fail-on-drift, --output-json, --quiet, --help,
  missing BOM → 1, missing model_dir → 1, invalid JSON BOM → 1.
- Module count gate (squish/ must have exactly 125 Python files).
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import patch


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _sha256_of(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_bom(directory: Path, file_entries: dict[str, bytes]) -> Path:
    """Write a minimal squish CycloneDX BOM with squish:weight_hash: properties.

    *file_entries* maps ``rel_path → file_bytes``.  The SHA-256 digest is
    computed from the bytes and stored as a BOM property.  The files
    themselves are NOT written by this helper.
    """
    properties = [
        {
            "name": f"squish:weight_hash:{rel}",
            "value": _sha256_of(data),
        }
        for rel, data in sorted(file_entries.items())
    ]
    bom: dict = {
        "bomFormat": "CycloneDX",
        "specVersion": "1.7",
        "components": [
            {
                "type": "machine-learning-model",
                "name": "test-model",
                "properties": properties,
            }
        ],
    }
    bom_path = directory / "cyclonedx-mlbom.json"
    bom_path.write_text(json.dumps(bom), encoding="utf-8")
    return bom_path


def _write_model_files(directory: Path, file_entries: dict[str, bytes]) -> None:
    """Write files in *file_entries* into *directory* (creating sub-dirs)."""
    for rel, data in file_entries.items():
        full_path = directory / rel
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_bytes(data)


# ──────────────────────────────────────────────────────────────────────────────
# TestDriftConfig
# ──────────────────────────────────────────────────────────────────────────────

class TestDriftConfig(unittest.TestCase):

    def test_fields_stored(self):
        from squish.squash.drift import DriftConfig
        cfg = DriftConfig(bom_path=Path("/x/bom.json"), model_dir=Path("/x/model"))
        self.assertEqual(cfg.bom_path, Path("/x/bom.json"))
        self.assertEqual(cfg.model_dir, Path("/x/model"))

    def test_tolerance_default_zero(self):
        from squish.squash.drift import DriftConfig
        cfg = DriftConfig(bom_path=Path("/x"), model_dir=Path("/y"))
        self.assertEqual(cfg.tolerance, 0.0)

    def test_tolerance_configurable(self):
        from squish.squash.drift import DriftConfig
        cfg = DriftConfig(bom_path=Path("/x"), model_dir=Path("/y"), tolerance=0.01)
        self.assertAlmostEqual(cfg.tolerance, 0.01)

    def test_bom_path_is_path(self):
        from squish.squash.drift import DriftConfig
        cfg = DriftConfig(bom_path=Path("/x"), model_dir=Path("/y"))
        self.assertIsInstance(cfg.bom_path, Path)
        self.assertIsInstance(cfg.model_dir, Path)


# ──────────────────────────────────────────────────────────────────────────────
# TestDriftHit
# ──────────────────────────────────────────────────────────────────────────────

class TestDriftHit(unittest.TestCase):

    def test_missing_when_actual_empty(self):
        from squish.squash.drift import DriftHit
        hit = DriftHit(path="a.npy", expected_digest="abc", actual_digest="")
        self.assertTrue(hit.missing)
        self.assertFalse(hit.tampered)

    def test_tampered_when_digests_differ(self):
        from squish.squash.drift import DriftHit
        hit = DriftHit(path="a.npy", expected_digest="aaa", actual_digest="bbb")
        self.assertFalse(hit.missing)
        self.assertTrue(hit.tampered)

    def test_neither_when_digests_match(self):
        from squish.squash.drift import DriftHit
        hit = DriftHit(path="a.npy", expected_digest="abc", actual_digest="abc")
        self.assertFalse(hit.missing)
        self.assertFalse(hit.tampered)

    def test_fields_stored(self):
        from squish.squash.drift import DriftHit
        hit = DriftHit(path="x/y.safetensors", expected_digest="e3b0", actual_digest="")
        self.assertEqual(hit.path, "x/y.safetensors")
        self.assertEqual(hit.expected_digest, "e3b0")
        self.assertEqual(hit.actual_digest, "")


# ──────────────────────────────────────────────────────────────────────────────
# TestDriftResult
# ──────────────────────────────────────────────────────────────────────────────

class TestDriftResult(unittest.TestCase):

    def test_ok_true_no_hits(self):
        from squish.squash.drift import DriftResult
        r = DriftResult(hits=[], files_checked=3, ok=True)
        self.assertTrue(r.ok)
        self.assertIn("3", r.summary)

    def test_summary_clean(self):
        from squish.squash.drift import DriftResult
        r = DriftResult(hits=[], files_checked=5, ok=True)
        self.assertIn("No drift", r.summary)
        self.assertIn("5 file", r.summary)

    def test_summary_drift_tampered(self):
        from squish.squash.drift import DriftHit, DriftResult
        hits = [DriftHit(path="a", expected_digest="e", actual_digest="x")]
        r = DriftResult(hits=hits, files_checked=2, ok=False)
        self.assertIn("Drift detected", r.summary)
        self.assertIn("tampered", r.summary)

    def test_summary_drift_missing(self):
        from squish.squash.drift import DriftHit, DriftResult
        hits = [DriftHit(path="b", expected_digest="e", actual_digest="")]
        r = DriftResult(hits=hits, files_checked=2, ok=False)
        self.assertIn("Drift detected", r.summary)
        self.assertIn("missing", r.summary)

    def test_summary_drift_both(self):
        from squish.squash.drift import DriftHit, DriftResult
        hits = [
            DriftHit(path="a", expected_digest="e", actual_digest=""),
            DriftHit(path="b", expected_digest="e", actual_digest="x"),
        ]
        r = DriftResult(hits=hits, files_checked=3, ok=False)
        self.assertIn("tampered", r.summary)
        self.assertIn("missing", r.summary)

    def test_custom_summary_preserved(self):
        from squish.squash.drift import DriftResult
        r = DriftResult(hits=[], files_checked=0, ok=True, summary="custom")
        self.assertEqual(r.summary, "custom")

    def test_files_checked_default_zero(self):
        from squish.squash.drift import DriftResult
        r = DriftResult()
        self.assertEqual(r.files_checked, 0)
        self.assertTrue(r.ok)

    def test_hits_default_empty_list(self):
        from squish.squash.drift import DriftResult
        r = DriftResult()
        self.assertEqual(r.hits, [])


# ──────────────────────────────────────────────────────────────────────────────
# TestParseBomHashes
# ──────────────────────────────────────────────────────────────────────────────

class TestParseBomHashes(unittest.TestCase):

    def _parse(self, bom: dict) -> dict:
        from squish.squash.drift import _parse_bom_hashes
        return _parse_bom_hashes(bom)

    def test_empty_bom_no_components(self):
        result = self._parse({})
        self.assertEqual(result, {})

    def test_empty_components_list(self):
        result = self._parse({"components": []})
        self.assertEqual(result, {})

    def test_no_weight_hash_properties(self):
        bom = {"components": [{"properties": [{"name": "other:key", "value": "v"}]}]}
        result = self._parse(bom)
        self.assertEqual(result, {})

    def test_single_weight_hash(self):
        bom = {
            "components": [
                {
                    "properties": [
                        {"name": "squish:weight_hash:tensors/w.npy", "value": "abc123"}
                    ]
                }
            ]
        }
        result = self._parse(bom)
        self.assertEqual(result, {"tensors/w.npy": "abc123"})

    def test_multiple_weight_hashes(self):
        bom = {
            "components": [
                {
                    "properties": [
                        {"name": "squish:weight_hash:a.npy", "value": "aaa"},
                        {"name": "squish:weight_hash:b.npy", "value": "bbb"},
                        {"name": "squish:other", "value": "ignored"},
                    ]
                }
            ]
        }
        result = self._parse(bom)
        self.assertEqual(result, {"a.npy": "aaa", "b.npy": "bbb"})

    def test_digest_lowercased(self):
        bom = {
            "components": [
                {
                    "properties": [
                        {"name": "squish:weight_hash:x.npy", "value": "ABC123DEF"}
                    ]
                }
            ]
        }
        result = self._parse(bom)
        self.assertEqual(result["x.npy"], "abc123def")

    def test_only_first_component_used(self):
        bom = {
            "components": [
                {
                    "properties": [
                        {"name": "squish:weight_hash:first.npy", "value": "111"}
                    ]
                },
                {
                    "properties": [
                        {"name": "squish:weight_hash:second.npy", "value": "222"}
                    ]
                },
            ]
        }
        result = self._parse(bom)
        self.assertIn("first.npy", result)
        self.assertNotIn("second.npy", result)

    def test_property_missing_name_ignored(self):
        bom = {
            "components": [
                {
                    "properties": [
                        {"value": "orphan"},
                        {"name": "squish:weight_hash:good.npy", "value": "ggg"},
                    ]
                }
            ]
        }
        result = self._parse(bom)
        self.assertEqual(result, {"good.npy": "ggg"})


# ──────────────────────────────────────────────────────────────────────────────
# TestCheckDrift
# ──────────────────────────────────────────────────────────────────────────────

class TestCheckDrift(unittest.TestCase):

    def setUp(self):
        import tempfile
        self._tmpdir = tempfile.mkdtemp()
        self.tmp = Path(self._tmpdir)

    def tearDown(self):
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_clean_model_ok(self):
        from squish.squash.drift import DriftConfig, check_drift
        files = {"tensors/w.npy": b"weight_data_v1"}
        bom_path = _write_bom(self.tmp, files)
        _write_model_files(self.tmp, files)
        result = check_drift(DriftConfig(bom_path=bom_path, model_dir=self.tmp))
        self.assertTrue(result.ok)
        self.assertEqual(result.hits, [])
        self.assertEqual(result.files_checked, 1)

    def test_clean_model_multiple_files(self):
        from squish.squash.drift import DriftConfig, check_drift
        files = {
            "tensors/model.0.weight.npy": b"layer0",
            "tensors/model.1.weight.npy": b"layer1",
            "config.json": b'{"model":"test"}',
        }
        bom_path = _write_bom(self.tmp, files)
        _write_model_files(self.tmp, files)
        result = check_drift(DriftConfig(bom_path=bom_path, model_dir=self.tmp))
        self.assertTrue(result.ok)
        self.assertEqual(result.files_checked, 3)

    def test_tampered_file_detected(self):
        from squish.squash.drift import DriftConfig, check_drift
        bom_files = {"weight.npy": b"original content"}
        bom_path = _write_bom(self.tmp, bom_files)
        # Write tampered content to disk
        _write_model_files(self.tmp, {"weight.npy": b"TAMPERED content!"})
        result = check_drift(DriftConfig(bom_path=bom_path, model_dir=self.tmp))
        self.assertFalse(result.ok)
        self.assertEqual(len(result.hits), 1)
        hit = result.hits[0]
        self.assertEqual(hit.path, "weight.npy")
        self.assertTrue(hit.tampered)
        self.assertFalse(hit.missing)
        self.assertNotEqual(hit.expected_digest, hit.actual_digest)
        self.assertTrue(len(hit.actual_digest) == 64)  # hex SHA-256

    def test_missing_file_detected(self):
        from squish.squash.drift import DriftConfig, check_drift
        bom_files = {
            "exists.npy": b"present",
            "gone.npy": b"missing_from_disk",
        }
        bom_path = _write_bom(self.tmp, bom_files)
        # Only write one of the two files
        _write_model_files(self.tmp, {"exists.npy": b"present"})
        result = check_drift(DriftConfig(bom_path=bom_path, model_dir=self.tmp))
        self.assertFalse(result.ok)
        self.assertEqual(len(result.hits), 1)
        hit = result.hits[0]
        self.assertEqual(hit.path, "gone.npy")
        self.assertTrue(hit.missing)
        self.assertEqual(hit.actual_digest, "")

    def test_extra_files_on_disk_ignored(self):
        from squish.squash.drift import DriftConfig, check_drift
        bom_files = {"attested.npy": b"attested"}
        bom_path = _write_bom(self.tmp, bom_files)
        # Write BOM file + extra file not in BOM
        _write_model_files(self.tmp, {
            "attested.npy": b"attested",
            "extra_unlisted.npy": b"extra data",
        })
        result = check_drift(DriftConfig(bom_path=bom_path, model_dir=self.tmp))
        self.assertTrue(result.ok)
        self.assertEqual(result.hits, [])
        self.assertEqual(result.files_checked, 1)

    def test_all_files_missing_reports_all(self):
        from squish.squash.drift import DriftConfig, check_drift
        bom_files = {
            "a.npy": b"aaa",
            "b.npy": b"bbb",
        }
        bom_path = _write_bom(self.tmp, bom_files)
        # Write no model files
        result = check_drift(DriftConfig(bom_path=bom_path, model_dir=self.tmp))
        self.assertFalse(result.ok)
        self.assertEqual(len(result.hits), 2)
        self.assertEqual(result.files_checked, 2)
        self.assertTrue(all(h.missing for h in result.hits))

    def test_invalid_json_raises(self):
        from squish.squash.drift import DriftConfig, check_drift
        bom_path = self.tmp / "bad.json"
        bom_path.write_text("not valid json", encoding="utf-8")
        with self.assertRaises(Exception):
            check_drift(DriftConfig(bom_path=bom_path, model_dir=self.tmp))

    def test_missing_bom_file_raises(self):
        from squish.squash.drift import DriftConfig, check_drift
        bom_path = self.tmp / "nonexistent.json"
        with self.assertRaises(FileNotFoundError):
            check_drift(DriftConfig(bom_path=bom_path, model_dir=self.tmp))

    def test_empty_bom_no_properties_raises_value_error(self):
        from squish.squash.drift import DriftConfig, check_drift
        bom_path = self.tmp / "empty.json"
        bom_path.write_text(
            json.dumps({"bomFormat": "CycloneDX", "components": []}),
            encoding="utf-8",
        )
        with self.assertRaises(ValueError):
            check_drift(DriftConfig(bom_path=bom_path, model_dir=self.tmp))

    def test_bom_with_no_squish_properties_raises_value_error(self):
        from squish.squash.drift import DriftConfig, check_drift
        bom = {
            "bomFormat": "CycloneDX",
            "components": [{"type": "library", "properties": [{"name": "other", "value": "x"}]}],
        }
        bom_path = self.tmp / "noprops.json"
        bom_path.write_text(json.dumps(bom), encoding="utf-8")
        with self.assertRaises(ValueError):
            check_drift(DriftConfig(bom_path=bom_path, model_dir=self.tmp))

    def test_files_checked_count(self):
        from squish.squash.drift import DriftConfig, check_drift
        files = {f"w{i}.npy": bytes([i, i, i]) for i in range(6)}
        bom_path = _write_bom(self.tmp, files)
        _write_model_files(self.tmp, files)
        result = check_drift(DriftConfig(bom_path=bom_path, model_dir=self.tmp))
        self.assertEqual(result.files_checked, 6)

    def test_subdirectory_path_round_trips(self):
        from squish.squash.drift import DriftConfig, check_drift
        files = {"tensors/deep/layer.safetensors": b"\x00\x01\x02"}
        bom_path = _write_bom(self.tmp, files)
        _write_model_files(self.tmp, files)
        result = check_drift(DriftConfig(bom_path=bom_path, model_dir=self.tmp))
        self.assertTrue(result.ok)

    def test_correct_expected_digest_recorded(self):
        from squish.squash.drift import DriftConfig, check_drift
        data = b"deterministic content for SHA-256 test"
        files = {"model.npy": data}
        bom_path = _write_bom(self.tmp, files)
        # Write tampered content so we get a hit with the expected digest
        _write_model_files(self.tmp, {"model.npy": b"other content"})
        result = check_drift(DriftConfig(bom_path=bom_path, model_dir=self.tmp))
        self.assertFalse(result.ok)
        expected = _sha256_of(data)
        self.assertEqual(result.hits[0].expected_digest, expected)


# ──────────────────────────────────────────────────────────────────────────────
# TestCliDriftCheck
# ──────────────────────────────────────────────────────────────────────────────

class TestCliDriftCheck(unittest.TestCase):

    def setUp(self):
        import tempfile
        self._tmpdir = tempfile.mkdtemp()
        self.tmp = Path(self._tmpdir)

    def tearDown(self):
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def _run(self, *argv: str) -> tuple[int, str, str]:
        """Run `squash drift-check ...` via subprocess; return (rc, stdout, stderr)."""
        result = subprocess.run(
            [sys.executable, "-m", "squish.squash.cli", "drift-check", *argv],
            capture_output=True,
            text=True,
        )
        return result.returncode, result.stdout, result.stderr

    def test_help_exits_zero(self):
        rc, stdout, _ = self._run("--help")
        self.assertEqual(rc, 0)
        self.assertIn("drift-check", stdout)

    def test_help_contains_fail_on_drift(self):
        rc, stdout, _ = self._run("--help")
        self.assertIn("--fail-on-drift", stdout)

    def test_help_contains_output_json(self):
        rc, stdout, _ = self._run("--help")
        self.assertIn("--output-json", stdout)

    def test_clean_model_exits_zero(self):
        files = {"w.npy": b"clean data"}
        bom_path = _write_bom(self.tmp, files)
        _write_model_files(self.tmp, files)
        rc, stdout, _ = self._run(str(self.tmp), "--bom", str(bom_path))
        self.assertEqual(rc, 0)
        self.assertIn("No drift", stdout)

    def test_drift_without_fail_flag_exits_zero(self):
        bom_files = {"w.npy": b"original"}
        bom_path = _write_bom(self.tmp, bom_files)
        _write_model_files(self.tmp, {"w.npy": b"TAMPERED"})
        rc, stdout, _ = self._run(str(self.tmp), "--bom", str(bom_path))
        self.assertEqual(rc, 0)
        self.assertIn("Drift detected", stdout)

    def test_drift_with_fail_flag_exits_two(self):
        bom_files = {"w.npy": b"original"}
        bom_path = _write_bom(self.tmp, bom_files)
        _write_model_files(self.tmp, {"w.npy": b"TAMPERED"})
        rc, _, _ = self._run(
            str(self.tmp), "--bom", str(bom_path), "--fail-on-drift"
        )
        self.assertEqual(rc, 2)

    def test_missing_file_with_fail_flag_exits_two(self):
        bom_files = {"present.npy": b"ok", "gone.npy": b"gone"}
        bom_path = _write_bom(self.tmp, bom_files)
        _write_model_files(self.tmp, {"present.npy": b"ok"})
        rc, _, _ = self._run(
            str(self.tmp), "--bom", str(bom_path), "--fail-on-drift"
        )
        self.assertEqual(rc, 2)

    def test_output_json_written(self):
        files = {"w.npy": b"data"}
        bom_path = _write_bom(self.tmp, files)
        _write_model_files(self.tmp, files)
        out_json = self.tmp / "result.json"
        rc, _, _ = self._run(
            str(self.tmp), "--bom", str(bom_path), "--output-json", str(out_json)
        )
        self.assertEqual(rc, 0)
        self.assertTrue(out_json.exists())
        payload = json.loads(out_json.read_text())
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["hits"], [])
        self.assertIn("files_checked", payload)

    def test_output_json_drift_has_hits(self):
        bom_files = {"w.npy": b"orig"}
        bom_path = _write_bom(self.tmp, bom_files)
        _write_model_files(self.tmp, {"w.npy": b"tampered"})
        out_json = self.tmp / "result.json"
        self._run(str(self.tmp), "--bom", str(bom_path), "--output-json", str(out_json))
        payload = json.loads(out_json.read_text())
        self.assertFalse(payload["ok"])
        self.assertEqual(len(payload["hits"]), 1)
        hit = payload["hits"][0]
        self.assertIn("path", hit)
        self.assertIn("expected_digest", hit)
        self.assertIn("actual_digest", hit)

    def test_quiet_suppresses_output(self):
        files = {"w.npy": b"data"}
        bom_path = _write_bom(self.tmp, files)
        _write_model_files(self.tmp, files)
        rc, stdout, _ = self._run(
            str(self.tmp), "--bom", str(bom_path), "--quiet"
        )
        self.assertEqual(rc, 0)
        self.assertEqual(stdout.strip(), "")

    def test_missing_bom_file_exits_one(self):
        rc, _, stderr = self._run(
            str(self.tmp), "--bom", str(self.tmp / "nonexistent.json")
        )
        self.assertEqual(rc, 1)
        self.assertIn("error", stderr.lower())

    def test_missing_model_dir_exits_one(self):
        bom_path = self.tmp / "bom.json"
        bom_path.write_text(json.dumps({"components": []}), encoding="utf-8")
        rc, _, stderr = self._run(
            str(self.tmp / "no_such_dir"), "--bom", str(bom_path)
        )
        self.assertEqual(rc, 1)
        self.assertIn("error", stderr.lower())

    def test_invalid_json_bom_exits_one(self):
        bad_bom = self.tmp / "bad.json"
        bad_bom.write_text("not json at all", encoding="utf-8")
        rc, _, stderr = self._run(str(self.tmp), "--bom", str(bad_bom))
        self.assertEqual(rc, 1)
        self.assertIn("error", stderr.lower())

    def test_bom_without_properties_exits_one(self):
        bom = {
            "bomFormat": "CycloneDX",
            "components": [{"type": "library"}],
        }
        bom_path = self.tmp / "empty_props.json"
        bom_path.write_text(json.dumps(bom), encoding="utf-8")
        rc, _, stderr = self._run(str(self.tmp), "--bom", str(bom_path))
        self.assertEqual(rc, 1)
        self.assertIn("error", stderr.lower())

    def test_drift_report_shows_tampered_label(self):
        bom_files = {"w.npy": b"original"}
        bom_path = _write_bom(self.tmp, bom_files)
        _write_model_files(self.tmp, {"w.npy": b"TAMPERED"})
        rc, stdout, _ = self._run(str(self.tmp), "--bom", str(bom_path))
        self.assertIn("TAMPERED", stdout)

    def test_drift_report_shows_missing_label(self):
        bom_files = {"gone.npy": b"data"}
        bom_path = _write_bom(self.tmp, bom_files)
        # Don't write the file
        rc, stdout, _ = self._run(str(self.tmp), "--bom", str(bom_path))
        self.assertIn("MISSING", stdout)


# ──────────────────────────────────────────────────────────────────────────────
# TestModuleCount — gate: squish/ must have exactly 125 Python files
# ──────────────────────────────────────────────────────────────────────────────

class TestModuleCount(unittest.TestCase):

    def test_module_count_is_125(self):
        """squish/ must contain exactly 125 Python files (124 pre-W51 + drift.py)."""
        squish_root = Path(__file__).parent.parent / "squish"
        py_files = list(squish_root.rglob("*.py"))
        count = len(py_files)
        self.assertEqual(
            count,
            125,
            msg=(
                f"Expected 125 Python files in squish/, found {count}. "
                "If you added a file, either justify it or delete an existing one."
            ),
        )


if __name__ == "__main__":
    unittest.main()
