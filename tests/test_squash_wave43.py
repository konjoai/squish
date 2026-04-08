"""Wave 43: Publish CircleCI Orb + Helm chart — static YAML validation tests.

All tests are pure unit tests: no I/O beyond file reads from the local repo,
no network calls, no Metal, no model weights.
"""
from pathlib import Path
import yaml
import pytest

REPO_ROOT = Path(__file__).parent.parent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_yaml(rel_path: str) -> dict:
    full = REPO_ROOT / rel_path
    assert full.exists(), f"Expected file not found: {full}"
    with full.open() as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# CircleCI orb YAML
# ---------------------------------------------------------------------------

class TestCircleCIOrbYaml:
    ORB_PATH = "squish/squash/integrations/circleci/orb.yml"

    def test_orb_parseable(self):
        doc = _load_yaml(self.ORB_PATH)
        assert isinstance(doc, dict), "orb.yml must parse to a mapping"

    def test_orb_has_required_top_level_keys(self):
        doc = _load_yaml(self.ORB_PATH)
        for key in ("version", "description", "commands"):
            assert key in doc, f"orb.yml missing required key: {key!r}"

    def test_orb_version_is_2_1(self):
        doc = _load_yaml(self.ORB_PATH)
        assert doc["version"] == 2.1 or str(doc["version"]) == "2.1", (
            f"Expected orb version 2.1, got {doc['version']!r}"
        )

    def test_orb_commands_structure(self):
        doc = _load_yaml(self.ORB_PATH)
        commands = doc["commands"]
        assert isinstance(commands, dict), "orb.yml 'commands' must be a mapping"
        assert len(commands) >= 1, "orb.yml must define at least one command"
        for name, body in commands.items():
            assert "description" in body, f"Command {name!r} missing 'description'"
            assert "steps" in body, f"Command {name!r} missing 'steps'"
            assert isinstance(body["steps"], list), (
                f"Command {name!r} 'steps' must be a list"
            )
            assert len(body["steps"]) >= 1, (
                f"Command {name!r} 'steps' must not be empty"
            )

    def test_orb_expected_commands_present(self):
        doc = _load_yaml(self.ORB_PATH)
        commands = doc["commands"]
        for expected in ("attest", "check", "policy-gate"):
            assert expected in commands, (
                f"Expected command {expected!r} not found in orb.yml"
            )

    def test_orb_display_block(self):
        doc = _load_yaml(self.ORB_PATH)
        assert "display" in doc, "orb.yml missing 'display' block"
        display = doc["display"]
        assert "home_url" in display or "source_url" in display, (
            "orb.yml display block must contain home_url or source_url"
        )


# ---------------------------------------------------------------------------
# Helm Chart.yaml
# ---------------------------------------------------------------------------

class TestHelmChartYaml:
    CHART_PATH = "helm/squish-serve/Chart.yaml"

    def test_chart_parseable(self):
        doc = _load_yaml(self.CHART_PATH)
        assert isinstance(doc, dict)

    def test_chart_required_keys(self):
        doc = _load_yaml(self.CHART_PATH)
        for key in ("apiVersion", "name", "version", "description"):
            assert key in doc, f"Chart.yaml missing required key: {key!r}"

    def test_chart_api_version(self):
        doc = _load_yaml(self.CHART_PATH)
        assert doc["apiVersion"] == "v2", (
            f"Expected Helm apiVersion v2, got {doc['apiVersion']!r}"
        )

    def test_chart_name(self):
        doc = _load_yaml(self.CHART_PATH)
        assert doc["name"] == "squish-serve", (
            f"Expected chart name 'squish-serve', got {doc['name']!r}"
        )

    def test_chart_version_semver_format(self):
        import re
        doc = _load_yaml(self.CHART_PATH)
        ver = str(doc["version"])
        assert re.match(r"^\d+\.\d+\.\d+", ver), (
            f"Chart version {ver!r} does not match semver pattern"
        )

    def test_chart_app_version_present(self):
        doc = _load_yaml(self.CHART_PATH)
        assert "appVersion" in doc, "Chart.yaml should specify appVersion"


# ---------------------------------------------------------------------------
# Artifact Hub repo metadata
# ---------------------------------------------------------------------------

class TestArtifacthubRepoYml:
    REPO_FILE = "artifacthub-repo.yml"

    def test_file_parseable(self):
        doc = _load_yaml(self.REPO_FILE)
        assert isinstance(doc, dict)

    def test_has_repository_id_key(self):
        doc = _load_yaml(self.REPO_FILE)
        assert "repositoryID" in doc, "artifacthub-repo.yml must have repositoryID key"

    def test_has_owners(self):
        doc = _load_yaml(self.REPO_FILE)
        assert "owners" in doc, "artifacthub-repo.yml must have owners key"
        owners = doc["owners"]
        assert isinstance(owners, list), "owners must be a list"
        assert len(owners) >= 1, "owners list must not be empty"
        for owner in owners:
            assert "name" in owner, "Each owner entry must have a 'name'"


# ---------------------------------------------------------------------------
# Publish workflow YAMLs
# ---------------------------------------------------------------------------

class TestPublishOrbWorkflow:
    WF_PATH = ".github/workflows/publish-orb.yml"

    def test_parseable(self):
        doc = _load_yaml(self.WF_PATH)
        assert isinstance(doc, dict)

    def test_has_on_trigger(self):
        doc = _load_yaml(self.WF_PATH)
        assert "on" in doc or True in doc, "workflow must have an 'on' trigger"

    def test_has_expected_jobs(self):
        doc = _load_yaml(self.WF_PATH)
        jobs = doc.get("jobs", {})
        assert "validate" in jobs, "publish-orb.yml must have a 'validate' job"
        assert "publish-dev" in jobs, "publish-orb.yml must have a 'publish-dev' job"
        assert "publish-prod" in jobs, "publish-orb.yml must have a 'publish-prod' job"

    def test_prod_job_needs_validate(self):
        doc = _load_yaml(self.WF_PATH)
        prod_needs = doc["jobs"]["publish-prod"].get("needs", [])
        if isinstance(prod_needs, str):
            prod_needs = [prod_needs]
        assert "validate" in prod_needs, (
            "publish-prod must depend on validate job"
        )


class TestPublishHelmWorkflow:
    WF_PATH = ".github/workflows/publish-helm.yml"

    def test_parseable(self):
        doc = _load_yaml(self.WF_PATH)
        assert isinstance(doc, dict)

    def test_has_publish_job(self):
        doc = _load_yaml(self.WF_PATH)
        jobs = doc.get("jobs", {})
        assert "publish" in jobs, "publish-helm.yml must have a 'publish' job"

    def test_has_packages_write_permission(self):
        doc = _load_yaml(self.WF_PATH)
        perms = doc.get("permissions", {})
        assert perms.get("packages") == "write", (
            "publish-helm.yml must grant packages:write permission for GHCR push"
        )

    def test_publish_job_runs_on_ubuntu(self):
        doc = _load_yaml(self.WF_PATH)
        runner = doc["jobs"]["publish"].get("runs-on", "")
        assert "ubuntu" in str(runner), (
            "publish job should run on ubuntu-latest"
        )
