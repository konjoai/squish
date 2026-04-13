"""Unit tests for squish.squash.edge_formats (Wave 56).

Test taxonomy: pure unit — no external SDKs required; uses in-memory byte stubs.
"""

from __future__ import annotations

import hashlib
import json
import struct
import tempfile
import unittest
from pathlib import Path


def _make_tflite_bytes(schema_version: int = 3, valid_magic: bool = True) -> bytes:
    """Produce a minimal syntactically-valid TFLite FlatBuffer stub.

    The real FlatBuffer structure is complex; we only produce enough to satisfy
    the magic-byte heuristic check in TFLiteParser.
    """
    # TFLite magic: offset 4 = b"TFL3" (v3) or b"TFL2" (v2)
    magic = b"TFL3" if schema_version >= 3 else b"TFL2"
    if not valid_magic:
        magic = b"XXXX"
    # FlatBuffer root table offset at byte 0 (4 bytes LE)
    root_offset = struct.pack("<I", 12)
    buf = root_offset + magic + b"\x00" * 256
    return buf


class TestTFLiteParserMagic(unittest.TestCase):
    """TFLiteParser should detect valid / invalid magic bytes."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_valid_tfl3_magic_parses_without_error(self):
        from squish.squash.edge_formats import TFLiteParser
        p = Path(self.tmpdir) / "model.tflite"
        p.write_bytes(_make_tflite_bytes(schema_version=3, valid_magic=True))
        meta = TFLiteParser.parse(p)
        # parse_error should be None (magic accepted) or at most a FlatBuffer decode error —
        # what it must NOT be is an "unknown magic" error.
        if meta.parse_error:
            self.assertNotIn("unknown magic", meta.parse_error.lower())

    def test_invalid_magic_recorded_as_parse_error(self):
        from squish.squash.edge_formats import TFLiteParser
        p = Path(self.tmpdir) / "bad.tflite"
        p.write_bytes(_make_tflite_bytes(valid_magic=False))
        meta = TFLiteParser.parse(p)
        self.assertIsNotNone(meta.parse_error)

    def test_sha256_is_hex64(self):
        from squish.squash.edge_formats import TFLiteParser
        p = Path(self.tmpdir) / "model.tflite"
        raw = _make_tflite_bytes()
        p.write_bytes(raw)
        meta = TFLiteParser.parse(p)
        expected_sha = hashlib.sha256(raw).hexdigest()
        self.assertEqual(expected_sha, meta.sha256)
        self.assertEqual(64, len(meta.sha256))

    def test_file_path_stored(self):
        from squish.squash.edge_formats import TFLiteParser
        p = Path(self.tmpdir) / "model.tflite"
        p.write_bytes(_make_tflite_bytes())
        meta = TFLiteParser.parse(p)
        self.assertEqual(p, meta.file_path)


class TestTFLiteParserCycloneDXProperties(unittest.TestCase):
    """to_cyclonedx_properties must return a list of {name, value} dicts."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_properties_have_squash_keys(self):
        from squish.squash.edge_formats import TFLiteParser
        p = Path(self.tmpdir) / "model.tflite"
        p.write_bytes(_make_tflite_bytes())
        meta = TFLiteParser.parse(p)
        props = meta.to_cyclonedx_properties()
        self.assertIsInstance(props, list)
        keys = {prop["name"] for prop in props}
        # At least the format key should be present
        self.assertTrue(any("squash:edge" in k for k in keys))


class TestCoreMLParser(unittest.TestCase):
    """CoreMLParser must parse a minimal .mlpackage directory."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _make_mlpackage(self, spec_version: int = 7) -> Path:
        pkg = Path(self.tmpdir) / "MyModel.mlpackage"
        pkg.mkdir()
        (pkg / "Manifest.json").write_text(json.dumps({
            "fileFormatVersion": "1.0.0",
            "modelVersion": "1.2",
            "itemByIdentifier": {
                "com.apple.CoreML.model": {
                    "path": "Data/com.apple.CoreML/model.mlmodel",
                }
            },
        }))
        data_dir = pkg / "Data" / "com.apple.CoreML"
        data_dir.mkdir(parents=True)
        (data_dir / "model.mlmodel").write_bytes(b"\x00" * 64)
        # metadata.json must be at the path the parser expects
        (data_dir / "metadata.json").write_text(json.dumps({
            "specificationVersion": spec_version,
            "shortDescription": "Test model",
            "author": "SquashTest",
        }))
        return pkg

    def test_parse_returns_metadata(self):
        from squish.squash.edge_formats import CoreMLParser
        pkg = self._make_mlpackage()
        meta = CoreMLParser.parse(pkg)
        self.assertEqual(pkg, meta.package_path)
        self.assertIsNotNone(meta.sha256)
        self.assertEqual(64, len(meta.sha256))

    def test_spec_version_read(self):
        from squish.squash.edge_formats import CoreMLParser
        pkg = self._make_mlpackage(spec_version=7)
        meta = CoreMLParser.parse(pkg)
        self.assertEqual(7, meta.spec_version)

    def test_model_version_and_description(self):
        from squish.squash.edge_formats import CoreMLParser
        pkg = self._make_mlpackage()
        meta = CoreMLParser.parse(pkg)
        self.assertEqual("1.2", meta.model_version)
        self.assertEqual("Test model", meta.short_description)

    def test_properties_returned(self):
        from squish.squash.edge_formats import CoreMLParser
        pkg = self._make_mlpackage()
        meta = CoreMLParser.parse(pkg)
        props = meta.to_cyclonedx_properties()
        self.assertIsInstance(props, list)
        keys = {p["name"] for p in props}
        self.assertTrue(any("squash:edge" in k for k in keys))


class TestEdgeSecurityScanner(unittest.TestCase):
    """EdgeSecurityScanner must flag known security issues."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_invalid_magic_triggers_edge_tflite_001(self):
        from squish.squash.edge_formats import EdgeSecurityScanner
        p = Path(self.tmpdir) / "bad.tflite"
        p.write_bytes(_make_tflite_bytes(valid_magic=False))
        findings = EdgeSecurityScanner.scan(p)
        ids = {f.finding_id for f in findings}
        self.assertIn("EDGE-TFLITE-001", ids)

    def test_valid_tflite_no_critical_findings(self):
        from squish.squash.edge_formats import EdgeSecurityScanner
        p = Path(self.tmpdir) / "ok.tflite"
        p.write_bytes(_make_tflite_bytes(valid_magic=True))
        findings = EdgeSecurityScanner.scan(p)
        critical = [f for f in findings if f.severity == "critical"]
        self.assertEqual(0, len(critical))

    def test_coreml_missing_model_data_triggers_edge_coreml_001(self):
        from squish.squash.edge_formats import EdgeSecurityScanner
        pkg = Path(self.tmpdir) / "EmptyModel.mlpackage"
        pkg.mkdir()
        (pkg / "Manifest.json").write_text(json.dumps({"fileFormatVersion": "1.0.0"}))
        # No Data/ directory — model data is missing
        findings = EdgeSecurityScanner.scan(pkg)
        ids = {f.finding_id for f in findings}
        self.assertIn("EDGE-COREML-001", ids)

    def test_coreml_objc_injection_pattern_detected(self):
        from squish.squash.edge_formats import EdgeSecurityScanner
        pkg = Path(self.tmpdir) / "InjectModel.mlpackage"
        pkg.mkdir()
        # Create model data so COREML-001 is NOT triggered
        data_dir = pkg / "Data" / "com.apple.CoreML"
        data_dir.mkdir(parents=True)
        (data_dir / "model.mlmodel").write_bytes(b"\x00" * 64)
        # Write a metadata JSON containing an objc_msgSend pattern (matches _OBJC_INJECTION_PATTERNS)
        (pkg / "Manifest.json").write_text(json.dumps({
            "fileFormatVersion": "1.0.0",
            "specialKey": "exec call detected in payload",
        }))
        findings = EdgeSecurityScanner.scan(pkg)
        ids = {f.finding_id for f in findings}
        self.assertIn("EDGE-COREML-002", ids)

    def test_unsupported_extension_returns_empty(self):
        from squish.squash.edge_formats import EdgeSecurityScanner
        p = Path(self.tmpdir) / "model.onnx"
        p.write_bytes(b"\x00" * 32)
        findings = EdgeSecurityScanner.scan(p)
        self.assertEqual([], findings)


class TestEdgeFindingSeverities(unittest.TestCase):
    """EdgeFinding dataclass must store severity and IDs."""

    def test_finding_fields(self):
        from squish.squash.edge_formats import EdgeFinding
        f = EdgeFinding(
            severity="high",
            finding_id="EDGE-TFLITE-001",
            title="Invalid magic",
            detail="Expected TFL3",
            file_path="/tmp/model.tflite",
        )
        self.assertEqual("high", f.severity)
        self.assertEqual("EDGE-TFLITE-001", f.finding_id)


if __name__ == "__main__":
    unittest.main()
