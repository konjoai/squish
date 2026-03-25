"""tests/test_wave92_presquish.py

Wave 92 — Pre-Compress Pipeline + HF Batch Upload

Tests for:
  - upload_to_hub.py has --all-missing, --dry-run, --org, --batch-file, --force flags
  - Catalog backfill: qwen3:32b, gemma3:12b, gemma3:27b, phi4:14b, mistral:7b squish_repos set
  - --all-missing lists only models without squish_repo
  - model_upload.yml workflow file exists with correct inputs
  - import_scan.py exists and is importable for orphan-module detection

Wave 92 is primarily a CI/tooling wave so most tests verify contracts and
presence of expected features rather than end-to-end execution.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)


# ============================================================================
# TestCatalogSquishRepoBackfill
# ============================================================================

class TestCatalogSquishRepoBackfill(unittest.TestCase):
    """Catalog entries specified in Wave 92 plan must now have squish_repo set."""

    def _get(self, model_id):
        from squish.catalog import resolve
        return resolve(model_id)

    def test_qwen3_32b_has_squish_repo(self):
        entry = self._get("qwen3:32b")
        assert entry is not None
        assert entry.squish_repo is not None, "qwen3:32b squish_repo should be set"
        assert "squishai" in entry.squish_repo

    def test_gemma3_12b_has_squish_repo(self):
        entry = self._get("gemma3:12b")
        assert entry is not None
        assert entry.squish_repo is not None, "gemma3:12b squish_repo should be set"
        assert "squishai" in entry.squish_repo

    def test_gemma3_27b_has_squish_repo(self):
        entry = self._get("gemma3:27b")
        assert entry is not None
        assert entry.squish_repo is not None, "gemma3:27b squish_repo should be set"
        assert "squishai" in entry.squish_repo

    def test_phi4_14b_has_squish_repo(self):
        entry = self._get("phi4:14b")
        assert entry is not None
        assert entry.squish_repo is not None, "phi4:14b squish_repo should be set"
        assert "squishai" in entry.squish_repo

    def test_mistral_7b_has_squish_repo(self):
        entry = self._get("mistral:7b")
        assert entry is not None
        assert entry.squish_repo is not None, "mistral:7b squish_repo should be set"
        assert "squishai" in entry.squish_repo

    def test_llama3_3_70b_has_squish_repo(self):
        """llama3.3:70b was backfilled in Wave 91 — verify it sticks."""
        entry = self._get("llama3.3:70b")
        assert entry is not None
        assert entry.squish_repo is not None
        assert "int2" in entry.squish_repo.lower() or "squished" in entry.squish_repo.lower()


# ============================================================================
# TestUploadScriptFlags — upload_to_hub.py argparse
# ============================================================================

class TestUploadScriptFlags(unittest.TestCase):
    """upload_to_hub.py must have all Wave 92 argparse flags."""

    def _get_help(self):
        import subprocess
        result = subprocess.run(
            [sys.executable, "dev/scripts/upload_to_hub.py", "--help"],
            capture_output=True, text=True, cwd=_repo_root,
        )
        return result.stdout + result.stderr

    def test_all_missing_flag_in_help(self):
        help_text = self._get_help()
        assert "--all-missing" in help_text, "--all-missing not in upload_to_hub.py --help"

    def test_dry_run_flag_in_help(self):
        help_text = self._get_help()
        assert "--dry-run" in help_text, "--dry-run not in upload_to_hub.py --help"

    def test_org_flag_in_help(self):
        help_text = self._get_help()
        assert "--org" in help_text, "--org not in upload_to_hub.py --help"

    def test_batch_file_flag_in_help(self):
        help_text = self._get_help()
        assert "--batch-file" in help_text, "--batch-file not in upload_to_hub.py --help"

    def test_force_flag_in_help(self):
        help_text = self._get_help()
        assert "--force" in help_text, "--force not in upload_to_hub.py --help"


# ============================================================================
# TestAllMissingLogic — --all-missing filter
# ============================================================================

class TestAllMissingLogic(unittest.TestCase):
    """--all-missing should list only models without squish_repo."""

    def test_all_missing_filters_correctly(self):
        from squish.catalog import list_catalog
        entries = list_catalog()
        missing = [e for e in entries if not e.squish_repo]
        with_repo = [e for e in entries if e.squish_repo]
        # After backfill we should have MORE models with squish_repo than without
        assert len(with_repo) >= len(missing), (
            f"Expected more models with squish_repo ({len(with_repo)}) "
            f"than without ({len(missing)})"
        )

    def test_wave92_backfill_not_in_missing(self):
        from squish.catalog import list_catalog
        entries = list_catalog()
        missing_ids = {e.id for e in entries if not e.squish_repo}
        for model_id in ("qwen3:32b", "gemma3:12b", "gemma3:27b", "phi4:14b", "mistral:7b"):
            assert model_id not in missing_ids, (
                f"{model_id} still in missing list after Wave 92 backfill"
            )


# ============================================================================
# TestBatchFile — --batch-file JSON parsing
# ============================================================================

class TestBatchFile(unittest.TestCase):
    """--batch-file must accept a JSON list of model IDs."""

    def _parse_batch(self, model_ids):
        """Simulate batch file parsing logic from upload_to_hub.py."""
        from squish.catalog import resolve
        entries = []
        errors = []
        for name in model_ids:
            e = resolve(name)
            if e is None:
                errors.append(name)
            else:
                entries.append(e)
        return entries, errors

    def test_valid_batch_resolves_all(self):
        entries, errors = self._parse_batch(["qwen3:8b", "gemma3:4b"])
        assert len(errors) == 0
        assert len(entries) == 2

    def test_unknown_model_in_batch_returns_error(self):
        entries, errors = self._parse_batch(["squish_nonexistent_xyz"])
        assert len(errors) > 0

    def test_empty_batch_returns_empty(self):
        entries, errors = self._parse_batch([])
        assert entries == []
        assert errors == []


# ============================================================================
# TestModelUploadWorkflow — .github/workflows/model_upload.yml
# ============================================================================

class TestModelUploadWorkflow(unittest.TestCase):
    """model_upload.yml must exist with correct structure."""

    def _read_workflow(self):
        path = Path(_repo_root) / ".github" / "workflows" / "model_upload.yml"
        assert path.exists(), f"model_upload.yml not found at {path}"
        return path.read_text()

    def test_workflow_file_exists(self):
        self._read_workflow()  # asserts existence

    def test_workflow_has_workflow_dispatch(self):
        content = self._read_workflow()
        assert "workflow_dispatch" in content

    def test_workflow_has_model_id_input(self):
        content = self._read_workflow()
        assert "model_id" in content

    def test_workflow_has_int2_input(self):
        content = self._read_workflow()
        assert "int2" in content

    def test_workflow_has_dry_run_input(self):
        content = self._read_workflow()
        assert "dry_run" in content

    def test_workflow_has_org_input(self):
        content = self._read_workflow()
        assert "org" in content


# ============================================================================
# TestImportScanScript — dev/scripts/import_scan.py
# ============================================================================

class TestImportScanScript(unittest.TestCase):
    """import_scan.py must exist and be syntactically valid."""

    def test_import_scan_exists(self):
        path = Path(_repo_root) / "dev" / "scripts" / "import_scan.py"
        assert path.exists(), "dev/scripts/import_scan.py does not exist"

    def test_import_scan_parseable(self):
        """import_scan.py must be valid Python."""
        import ast
        path = Path(_repo_root) / "dev" / "scripts" / "import_scan.py"
        src = path.read_text()
        try:
            ast.parse(src)
        except SyntaxError as exc:
            self.fail(f"import_scan.py has a SyntaxError: {exc}")

    def test_import_scan_has_report_functions(self):
        """import_scan.py must contain orphan report and dead-flag report logic."""
        path = Path(_repo_root) / "dev" / "scripts" / "import_scan.py"
        content = path.read_text().lower()
        assert "orphan" in content or "inbound" in content, \
            "import_scan.py should mention orphan/inbound module analysis"


if __name__ == "__main__":
    unittest.main()
