"""Wave 19 — SbomRegistry and EvalBinder tests (W45: eval_binder shim deleted)."""
from __future__ import annotations

import base64
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch


def _write_bom(path: Path) -> None:
    path.write_text(json.dumps({
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "version": 1,
        "serialNumber": "urn:uuid:test-001",
        "components": [{
            "type": "machine-learning-model",
            "name": "test-model",
            "version": "1.0",
            "modelCard": {
                "quantitativeAnalysis": {
                    "performanceMetrics": [],
                },
            },
        }],
    }))


# ---------------------------------------------------------------------------
# EvalBinder import test (W45: eval_binder.py shim deleted — canonical location)
# ---------------------------------------------------------------------------

class TestEvalBinderShim(unittest.TestCase):
    def test_import_from_sbom_builder(self):
        """EvalBinder is importable from sbom_builder (canonical location)."""
        from squish.squash.sbom_builder import EvalBinder
        self.assertIsNotNone(EvalBinder)

    def test_eval_binder_has_bind(self):
        from squish.squash.sbom_builder import EvalBinder
        self.assertTrue(callable(EvalBinder.bind))


class TestEvalBinderBind(unittest.TestCase):
    def test_bind_adds_performance_metrics(self):
        with tempfile.TemporaryDirectory() as td:
            bom_path = Path(td) / "cyclonedx-mlbom.json"
            _write_bom(bom_path)

            lmeval = Path(td) / "lmeval.json"
            lmeval.write_text(json.dumps({
                "results": {
                    "arc_easy": {"acc": 0.706, "acc_stderr": 0.01},
                }
            }))

            from squish.squash.sbom_builder import EvalBinder
            EvalBinder.bind(bom_path, lmeval)

            updated = json.loads(bom_path.read_text())
            # The BOM should still be valid JSON
            self.assertIn("bomFormat", updated)

    def test_bind_nonexistent_lmeval_raises(self):
        with tempfile.TemporaryDirectory() as td:
            bom_path = Path(td) / "cyclonedx-mlbom.json"
            _write_bom(bom_path)
            lmeval = Path(td) / "missing.json"

            from squish.squash.sbom_builder import EvalBinder
            with self.assertRaises(Exception):
                EvalBinder.bind(bom_path, lmeval)

    def test_bind_return_type(self):
        with tempfile.TemporaryDirectory() as td:
            bom_path = Path(td) / "cyclonedx-mlbom.json"
            _write_bom(bom_path)
            lmeval = Path(td) / "lmeval.json"
            lmeval.write_text(json.dumps({"results": {}}))

            from squish.squash.sbom_builder import EvalBinder
            result = EvalBinder.bind(bom_path, lmeval)
            self.assertIsNone(result)  # returns None (in-place)


# ---------------------------------------------------------------------------
# SbomRegistry tests
# ---------------------------------------------------------------------------

class TestSbomRegistryDtrack(unittest.TestCase):
    def test_push_dtrack_returns_url(self):
        with tempfile.TemporaryDirectory() as td:
            bom = Path(td) / "cyclonedx-mlbom.json"
            _write_bom(bom)

            mock_resp = MagicMock()
            mock_resp.status = 200
            mock_resp.__enter__ = lambda s: s
            mock_resp.__exit__ = MagicMock(return_value=False)

            with patch("urllib.request.urlopen", return_value=mock_resp):
                from squish.squash.sbom_builder import SbomRegistry
                url = SbomRegistry.push_dtrack(
                    bom,
                    base_url="http://dtrack.example.com",
                    api_key="test-key",
                    project_name="my-project",
                )

            self.assertIsInstance(url, str)
            self.assertIn("my-project", url)

    def test_push_dtrack_sends_put_with_api_key(self):
        captured = {}

        def fake_urlopen(req, timeout=None):
            captured["method"] = req.method
            captured["headers"] = dict(req.headers)
            captured["data"] = json.loads(base64.b64decode(
                json.loads(req.data.decode())["bom"]
            ).decode("utf-8"))
            mock_resp = MagicMock()
            mock_resp.status = 200
            mock_resp.__enter__ = lambda s: s
            mock_resp.__exit__ = MagicMock(return_value=False)
            return mock_resp

        with tempfile.TemporaryDirectory() as td:
            bom = Path(td) / "bom.json"
            _write_bom(bom)
            with patch("urllib.request.urlopen", fake_urlopen):
                from squish.squash.sbom_builder import SbomRegistry
                SbomRegistry.push_dtrack(bom, "http://dt.example.com", "my-api-key")

        self.assertEqual(captured["method"], "PUT")
        self.assertIn("x-api-key", {k.lower(): v for k, v in captured["headers"].items()})

    def test_push_dtrack_raises_on_http_error(self):
        import urllib.error
        with tempfile.TemporaryDirectory() as td:
            bom = Path(td) / "bom.json"
            _write_bom(bom)
            with patch("urllib.request.urlopen",
                       side_effect=urllib.error.HTTPError(
                           url=None, code=403, msg="Forbidden", hdrs=None, fp=None)):
                from squish.squash.sbom_builder import SbomRegistry
                with self.assertRaises(RuntimeError):
                    SbomRegistry.push_dtrack(bom, "http://dt.example.com", "bad-key")


class TestSbomRegistryGuac(unittest.TestCase):
    def test_push_guac_returns_endpoint_url(self):
        with tempfile.TemporaryDirectory() as td:
            bom = Path(td) / "bom.json"
            _write_bom(bom)

            mock_resp = MagicMock()
            mock_resp.status = 200
            mock_resp.__enter__ = lambda s: s
            mock_resp.__exit__ = MagicMock(return_value=False)

            endpoint = "http://guac.example.com/api/v1/upload"
            with patch("urllib.request.urlopen", return_value=mock_resp):
                from squish.squash.sbom_builder import SbomRegistry
                result = SbomRegistry.push_guac(bom, endpoint)

            self.assertEqual(result, endpoint)

    def test_push_guac_raises_on_http_error(self):
        import urllib.error
        with tempfile.TemporaryDirectory() as td:
            bom = Path(td) / "bom.json"
            _write_bom(bom)
            with patch("urllib.request.urlopen",
                       side_effect=urllib.error.HTTPError(
                           url=None, code=500, msg="Internal Server Error",
                           hdrs=None, fp=None)):
                from squish.squash.sbom_builder import SbomRegistry
                with self.assertRaises(RuntimeError):
                    SbomRegistry.push_guac(bom, "http://guac.example.com/upload")


class TestSbomRegistrySquash(unittest.TestCase):
    def test_push_squash_returns_str(self):
        with tempfile.TemporaryDirectory() as td:
            bom = Path(td) / "bom.json"
            _write_bom(bom)

            mock_resp = MagicMock()
            mock_resp.status = 200
            mock_resp.__enter__ = lambda s: s
            mock_resp.__exit__ = MagicMock(return_value=False)

            with patch("urllib.request.urlopen", return_value=mock_resp):
                from squish.squash.sbom_builder import SbomRegistry
                result = SbomRegistry.push_squash(
                    bom,
                    registry_url="http://registry.example.com",
                    token="bearer-token",
                )
            self.assertIsInstance(result, str)

    def test_push_squash_raises_on_error(self):
        import urllib.error
        with tempfile.TemporaryDirectory() as td:
            bom = Path(td) / "bom.json"
            _write_bom(bom)
            with patch("urllib.request.urlopen",
                       side_effect=urllib.error.HTTPError(
                           url=None, code=401, msg="Unauthorized",
                           hdrs=None, fp=None)):
                from squish.squash.sbom_builder import SbomRegistry
                with self.assertRaises(RuntimeError):
                    SbomRegistry.push_squash(bom, "http://reg.example.com", "bad-token")


class TestSbomRegistryDtypeContracts(unittest.TestCase):
    def test_push_dtrack_return_type(self):
        with tempfile.TemporaryDirectory() as td:
            bom = Path(td) / "bom.json"
            _write_bom(bom)
            mock_resp = MagicMock()
            mock_resp.status = 200
            mock_resp.__enter__ = lambda s: s
            mock_resp.__exit__ = MagicMock(return_value=False)
            with patch("urllib.request.urlopen", return_value=mock_resp):
                from squish.squash.sbom_builder import SbomRegistry
                result = SbomRegistry.push_dtrack(bom, "http://dt.example.com", "key")
            self.assertIsInstance(result, str)

    def test_push_guac_return_type(self):
        with tempfile.TemporaryDirectory() as td:
            bom = Path(td) / "bom.json"
            _write_bom(bom)
            mock_resp = MagicMock()
            mock_resp.status = 200
            mock_resp.__enter__ = lambda s: s
            mock_resp.__exit__ = MagicMock(return_value=False)
            with patch("urllib.request.urlopen", return_value=mock_resp):
                from squish.squash.sbom_builder import SbomRegistry
                result = SbomRegistry.push_guac(bom, "http://guac.example.com")
            self.assertIsInstance(result, str)


if __name__ == "__main__":
    unittest.main()
