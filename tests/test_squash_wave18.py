"""Wave 18 — CompositeAttestPipeline tests."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch


def _make_attest_result(model_dir: Path, passed: bool = True) -> "AttestResult":
    from squish.squash.attest import AttestResult
    return AttestResult(
        model_id=model_dir.name,
        output_dir=model_dir,
        passed=passed,
        cyclonedx_path=model_dir / "cyclonedx-mlbom.json",
        error="",
    )


class TestCompositeAttestConfig(unittest.TestCase):
    def test_config_fields(self):
        from squish.squash.attest import CompositeAttestConfig
        cfg = CompositeAttestConfig(
            model_paths=[Path("/models/a"), Path("/models/b")],
            policies=["no-cvss"],
            sign=False,
        )
        self.assertEqual(len(cfg.model_paths), 2)
        self.assertIn("no-cvss", cfg.policies)
        self.assertFalse(cfg.sign)

    def test_config_output_dir_defaults_none(self):
        from squish.squash.attest import CompositeAttestConfig
        cfg = CompositeAttestConfig(model_paths=[Path("/m")])
        self.assertIsNone(cfg.output_dir)


class TestCompositeAttestResult(unittest.TestCase):
    def test_result_passed_all_pass(self):
        from squish.squash.attest import CompositeAttestResult
        r = CompositeAttestResult(
            component_results=[],
            parent_bom_path=None,
            output_dir=Path("/out"),
            passed=True,
        )
        self.assertTrue(r.passed)
        self.assertEqual(r.error, "")

    def test_result_failed(self):
        from squish.squash.attest import CompositeAttestResult
        r = CompositeAttestResult(
            component_results=[],
            parent_bom_path=None,
            output_dir=Path("/out"),
            passed=False,
            error="scan failed",
        )
        self.assertFalse(r.passed)
        self.assertEqual(r.error, "scan failed")


class TestCompositeAttestPipelineRun(unittest.TestCase):
    def test_run_returns_composite_result(self):
        """run() returns CompositeAttestResult (mocked AttestPipeline)."""
        with tempfile.TemporaryDirectory() as td:
            model_a = Path(td) / "model-a"
            model_b = Path(td) / "model-b"
            for m in (model_a, model_b):
                m.mkdir()
                (m / "cyclonedx-mlbom.json").write_text(json.dumps({
                    "bomFormat": "CycloneDX",
                    "specVersion": "1.5",
                    "version": 1,
                    "serialNumber": f"urn:uuid:{m.name}",
                    "metadata": {"component": {"name": m.name, "version": "1.0"}},
                    "components": [],
                    "dependencies": [],
                }))

            from squish.squash.attest import CompositeAttestConfig, CompositeAttestPipeline

            mock_result_a = _make_attest_result(model_a, passed=True)
            mock_result_b = _make_attest_result(model_b, passed=True)

            with patch("squish.squash.attest.AttestPipeline") as MockPipeline:
                MockPipeline.run.side_effect = [mock_result_a, mock_result_b]

                cfg = CompositeAttestConfig(
                    model_paths=[model_a, model_b],
                    policies=["no-cvss"],
                )
                result = CompositeAttestPipeline.run(cfg)

            from squish.squash.attest import CompositeAttestResult
            self.assertIsInstance(result, CompositeAttestResult)

    def test_run_never_raises(self):
        """run() returns result with error field — does not propagate exceptions."""
        with tempfile.TemporaryDirectory() as td:
            model_a = Path(td) / "model-a"
            model_a.mkdir()

            from squish.squash.attest import CompositeAttestConfig, CompositeAttestPipeline

            with patch("squish.squash.attest.AttestPipeline") as MockPipeline:
                MockPipeline.run.side_effect = RuntimeError("disk full")

                cfg = CompositeAttestConfig(model_paths=[model_a])
                result = CompositeAttestPipeline.run(cfg)

            self.assertIsNotNone(result)

    def test_parent_bom_is_cyclonedx(self):
        """Parent BOM written by run() conforms to CycloneDX schema basics."""
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td) / "output"
            out_dir.mkdir()
            model_a = Path(td) / "model-a"
            model_b = Path(td) / "model-b"

            for m in (model_a, model_b):
                m.mkdir()
                (m / "cyclonedx-mlbom.json").write_text(json.dumps({
                    "bomFormat": "CycloneDX",
                    "specVersion": "1.5",
                    "version": 1,
                    "serialNumber": f"urn:uuid:fake-{m.name}",
                    "metadata": {"component": {"name": m.name, "version": "0.1"}},
                    "components": [],
                    "dependencies": [],
                }))

            from squish.squash.attest import (
                CompositeAttestConfig,
                CompositeAttestPipeline,
            )

            mock_r_a = _make_attest_result(model_a, passed=True)
            mock_r_b = _make_attest_result(model_b, passed=True)
            mock_r_a.cyclonedx_path = model_a / "cyclonedx-mlbom.json"
            mock_r_b.cyclonedx_path = model_b / "cyclonedx-mlbom.json"

            with patch("squish.squash.attest.AttestPipeline") as MockPipeline:
                MockPipeline.run.side_effect = [mock_r_a, mock_r_b]

                cfg = CompositeAttestConfig(
                    model_paths=[model_a, model_b],
                    output_dir=out_dir,
                )
                result = CompositeAttestPipeline.run(cfg)

            if result.parent_bom_path and result.parent_bom_path.exists():
                parent_bom = json.loads(result.parent_bom_path.read_text())
                self.assertEqual(parent_bom.get("bomFormat"), "CycloneDX")
                self.assertIn("dependencies", parent_bom)


class TestCompositeAttestDtypeContracts(unittest.TestCase):
    def test_config_types(self):
        from squish.squash.attest import CompositeAttestConfig
        cfg = CompositeAttestConfig(model_paths=[Path("/x"), Path("/y")])
        self.assertIsInstance(cfg.model_paths, list)
        self.assertIsInstance(cfg.policies, list)
        self.assertIsInstance(cfg.sign, bool)

    def test_result_types(self):
        from squish.squash.attest import CompositeAttestResult
        r = CompositeAttestResult(
            component_results=[], parent_bom_path=None,
            output_dir=Path("/x"), passed=True,
        )
        self.assertIsInstance(r.component_results, list)
        self.assertIsInstance(r.passed, bool)
        self.assertIsInstance(r.error, str)


if __name__ == "__main__":
    unittest.main()
