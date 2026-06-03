"""W84: Reusable GitHub Actions composite integration tests.

Static schema/structure tests for:
- .github/actions/squash-scan/action.yml
- .github/actions/squash-compress/action.yml
- .github/actions/squash-attest/action.yml
- docs/github-actions.md

These tests are pure unit tests: local file reads only, no network/process/model I/O.
"""

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_yaml(rel_path: str) -> dict:
    full = REPO_ROOT / rel_path
    assert full.exists(), f"Expected file not found: {full}"
    with full.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _steps(doc: dict) -> list[dict]:
    runs = doc.get("runs", {})
    assert isinstance(runs, dict), "action.yml must define a runs mapping"
    steps = runs.get("steps", [])
    assert isinstance(steps, list), "runs.steps must be a list"
    assert len(steps) >= 1, "runs.steps must not be empty"
    return steps


class TestSquashScanAction:
    ACTION = ".github/actions/squash-scan/action.yml"

    def test_parseable(self):
        doc = _load_yaml(self.ACTION)
        assert isinstance(doc, dict)

    def test_required_top_level_keys(self):
        doc = _load_yaml(self.ACTION)
        for key in ("name", "description", "inputs", "outputs", "runs"):
            assert key in doc, f"scan action missing key: {key}"

    def test_runs_composite(self):
        doc = _load_yaml(self.ACTION)
        assert doc["runs"]["using"] == "composite"

    def test_required_inputs_exist(self):
        doc = _load_yaml(self.ACTION)
        inputs = doc.get("inputs", {})
        assert "model-path" in inputs
        assert inputs["model-path"].get("required") is True
        assert "strict" in inputs
        assert str(inputs["strict"].get("default", "")).lower() == "false"

    def test_expected_outputs_exist(self):
        doc = _load_yaml(self.ACTION)
        outputs = doc.get("outputs", {})
        assert "scan-result" in outputs
        assert "report-path" in outputs
        assert "steps.scan.outputs.scan-result" in outputs["scan-result"].get("value", "")

    def test_setup_python_step_present(self):
        steps = _steps(_load_yaml(self.ACTION))
        assert any(str(s.get("uses", "")).startswith("actions/setup-python@") for s in steps)

    def test_scan_step_invokes_cli(self):
        steps = _steps(_load_yaml(self.ACTION))
        run_blocks = "\n".join(str(s.get("run", "")) for s in steps)
        assert "squash scan" in run_blocks


class TestSquashCompressAction:
    ACTION = ".github/actions/squash-compress/action.yml"

    def test_parseable(self):
        doc = _load_yaml(self.ACTION)
        assert isinstance(doc, dict)

    def test_required_top_level_keys(self):
        doc = _load_yaml(self.ACTION)
        for key in ("name", "description", "inputs", "outputs", "runs"):
            assert key in doc, f"compress action missing key: {key}"

    def test_runs_composite(self):
        doc = _load_yaml(self.ACTION)
        assert doc["runs"]["using"] == "composite"

    def test_input_contract(self):
        doc = _load_yaml(self.ACTION)
        inputs = doc.get("inputs", {})
        assert "model-path" in inputs
        assert inputs["model-path"].get("required") is True
        assert str(inputs.get("method", {}).get("default")) == "hqq"
        assert str(inputs.get("nbits", {}).get("default")) == "4"

    def test_output_contract(self):
        doc = _load_yaml(self.ACTION)
        outputs = doc.get("outputs", {})
        assert "compression-ratio" in outputs
        assert "bom-path" in outputs
        assert "steps.compress.outputs.compression-ratio" in outputs["compression-ratio"].get("value", "")

    def test_setup_python_step_present(self):
        steps = _steps(_load_yaml(self.ACTION))
        assert any(str(s.get("uses", "")).startswith("actions/setup-python@") for s in steps)

    def test_compress_step_invokes_cli(self):
        steps = _steps(_load_yaml(self.ACTION))
        run_blocks = "\n".join(str(s.get("run", "")) for s in steps)
        assert "squish compress" in run_blocks
        assert "INPUT_METHOD" in run_blocks
        assert "INPUT_NBITS" in run_blocks


class TestSquashAttestAction:
    ACTION = ".github/actions/squash-attest/action.yml"

    def test_parseable(self):
        doc = _load_yaml(self.ACTION)
        assert isinstance(doc, dict)

    def test_required_top_level_keys(self):
        doc = _load_yaml(self.ACTION)
        for key in ("name", "description", "inputs", "outputs", "runs"):
            assert key in doc, f"attest action missing key: {key}"

    def test_runs_composite(self):
        doc = _load_yaml(self.ACTION)
        assert doc["runs"]["using"] == "composite"

    def test_input_contract(self):
        doc = _load_yaml(self.ACTION)
        inputs = doc.get("inputs", {})
        assert "model-path" in inputs
        assert inputs["model-path"].get("required") is True
        assert str(inputs.get("policies", {}).get("default")) == "enterprise-strict"

    def test_output_contract(self):
        doc = _load_yaml(self.ACTION)
        outputs = doc.get("outputs", {})
        assert "passed" in outputs
        assert "attestation-path" in outputs
        assert "steps.attest.outputs.passed" in outputs["passed"].get("value", "")

    def test_setup_python_step_present(self):
        steps = _steps(_load_yaml(self.ACTION))
        assert any(str(s.get("uses", "")).startswith("actions/setup-python@") for s in steps)

    def test_attest_step_invokes_cli(self):
        steps = _steps(_load_yaml(self.ACTION))
        run_blocks = "\n".join(str(s.get("run", "")) for s in steps)
        assert "squash attest" in run_blocks
        assert "INPUT_POLICIES" in run_blocks


class TestGithubActionsDocumentation:
    DOC_PATH = "docs/github-actions.md"

    def test_doc_exists(self):
        full = REPO_ROOT / self.DOC_PATH
        assert full.exists(), "docs/github-actions.md must exist"

    def test_doc_mentions_reusable_action_paths(self):
        text = (REPO_ROOT / self.DOC_PATH).read_text(encoding="utf-8")
        assert "./.github/actions/squash-scan" in text
        assert "./.github/actions/squash-compress" in text
        assert "./.github/actions/squash-attest" in text

    def test_doc_contains_end_to_end_workflow_example(self):
        text = (REPO_ROOT / self.DOC_PATH).read_text(encoding="utf-8")
        assert "name: squash-compliance" in text
        assert "Enforce pass gate" in text
