"""
tests/test_helm_chart_unit.py

Unit tests for the squish-serve Helm chart (helm/squish-serve/).

Tests:
  1. All required chart files exist
  2. Chart.yaml is valid YAML with required apiVersion/name/version fields
  3. values.yaml is valid YAML with expected top-level keys
  4. All template files are syntactically valid Helm (via `helm lint` when
     the `helm` CLI is available, otherwise structural file checks)
  5. `helm template` dry-run produces parseable Kubernetes manifests for key
     value overrides (GPU, CPU-only, autoscaling enabled, KEDA enabled)
  6. Key resource types are rendered in the manifests
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

_REPO   = Path(__file__).resolve().parent.parent
_CHART  = _REPO / "helm" / "squish-serve"
_TMPLS  = _CHART / "templates"

_HELM_AVAILABLE = shutil.which("helm") is not None

# ── File existence ─────────────────────────────────────────────────────────────

class TestHelmChartFiles:
    def test_chart_yaml_exists(self):
        assert (_CHART / "Chart.yaml").is_file()

    def test_values_yaml_exists(self):
        assert (_CHART / "values.yaml").is_file()

    def test_helmignore_exists(self):
        assert (_CHART / ".helmignore").is_file()

    def test_readme_exists(self):
        assert (_CHART / "README.md").is_file()

    def test_templates_dir_exists(self):
        assert _TMPLS.is_dir()

    def test_helpers_tpl_exists(self):
        assert (_TMPLS / "_helpers.tpl").is_file()

    def test_notes_txt_exists(self):
        assert (_TMPLS / "NOTES.txt").is_file()

    def test_deployment_template_exists(self):
        assert (_TMPLS / "deployment.yaml").is_file()

    def test_service_template_exists(self):
        assert (_TMPLS / "service.yaml").is_file()

    def test_configmap_template_exists(self):
        assert (_TMPLS / "configmap.yaml").is_file()

    def test_serviceaccount_template_exists(self):
        assert (_TMPLS / "serviceaccount.yaml").is_file()

    def test_pvc_template_exists(self):
        assert (_TMPLS / "pvc.yaml").is_file()

    def test_hpa_template_exists(self):
        assert (_TMPLS / "hpa.yaml").is_file()

    def test_keda_scaledobject_template_exists(self):
        assert (_TMPLS / "keda-scaledobject.yaml").is_file()


# ── Chart.yaml validation ──────────────────────────────────────────────────────

class TestChartYaml:
    @pytest.fixture(scope="class")
    def chart(self):
        return yaml.safe_load((_CHART / "Chart.yaml").read_text())

    def test_api_version_is_v2(self, chart):
        assert chart["apiVersion"] == "v2"

    def test_has_name(self, chart):
        assert chart["name"] == "squish-serve"

    def test_has_version(self, chart):
        assert "version" in chart
        # Semantic version pattern: major.minor.patch
        parts = chart["version"].split(".")
        assert len(parts) == 3
        assert all(p.isdigit() for p in parts)

    def test_has_app_version(self, chart):
        assert "appVersion" in chart

    def test_type_is_application(self, chart):
        assert chart.get("type") == "application"

    def test_has_description(self, chart):
        assert "description" in chart
        assert len(chart["description"].strip()) > 10

    def test_has_keywords(self, chart):
        assert "keywords" in chart
        assert isinstance(chart["keywords"], list)
        assert len(chart["keywords"]) > 0

    def test_has_maintainers(self, chart):
        assert "maintainers" in chart
        assert isinstance(chart["maintainers"], list)


# ── values.yaml validation ─────────────────────────────────────────────────────

class TestValuesYaml:
    @pytest.fixture(scope="class")
    def vals(self):
        return yaml.safe_load((_CHART / "values.yaml").read_text())

    def test_image_section_exists(self, vals):
        assert "image" in vals

    def test_image_has_repository(self, vals):
        assert "repository" in vals["image"]
        assert "squishai" in vals["image"]["repository"]

    def test_image_has_tag(self, vals):
        assert "tag" in vals["image"]

    def test_image_has_flavour(self, vals):
        assert "flavour" in vals["image"]
        assert vals["image"]["flavour"] in ("cuda", "cpu", "")

    def test_model_section_exists(self, vals):
        assert "model" in vals
        assert "id" in vals["model"]

    def test_server_section_exists(self, vals):
        assert "server" in vals
        assert "port" in vals["server"]

    def test_service_section_exists(self, vals):
        assert "service" in vals
        assert vals["service"]["type"] in ("ClusterIP", "NodePort", "LoadBalancer")
        assert vals["service"]["port"] == 8080

    def test_resources_section_exists(self, vals):
        assert "resources" in vals
        assert "limits" in vals["resources"]
        assert "requests" in vals["resources"]

    def test_resources_has_gpu(self, vals):
        limits = vals["resources"]["limits"]
        assert "nvidia.com/gpu" in limits

    def test_persistence_section_exists(self, vals):
        p = vals["persistence"]
        assert "enabled" in p
        assert "size" in p
        assert "mountPath" in p

    def test_autoscaling_section_exists(self, vals):
        a = vals["autoscaling"]
        assert "enabled" in a
        assert "minReplicas" in a
        assert "maxReplicas" in a

    def test_keda_section_exists(self, vals):
        k = vals["keda"]
        assert "enabled" in k
        assert "prometheus" in k

    def test_liveness_probe_section_exists(self, vals):
        assert "livenessProbe" in vals

    def test_readiness_probe_section_exists(self, vals):
        assert "readinessProbe" in vals

    def test_service_account_section_exists(self, vals):
        assert "serviceAccount" in vals

    def test_replica_count_exists(self, vals):
        assert "replicaCount" in vals
        assert vals["replicaCount"] >= 1

    def test_node_selector_exists(self, vals):
        assert "nodeSelector" in vals

    def test_tolerations_exist(self, vals):
        assert "tolerations" in vals

    def test_affinity_exists(self, vals):
        assert "affinity" in vals


# ── Template content smoke tests (raw string checks) ──────────────────────────

class TestTemplateContent:
    def test_deployment_references_models_volume(self):
        content = (_TMPLS / "deployment.yaml").read_text()
        # The mount path is sourced from values: {{ .Values.persistence.mountPath }}
        assert "persistence.mountPath" in content or "/models" in content

    def test_deployment_has_liveness_probe(self):
        content = (_TMPLS / "deployment.yaml").read_text()
        assert "livenessProbe" in content

    def test_deployment_has_readiness_probe(self):
        content = (_TMPLS / "deployment.yaml").read_text()
        assert "readinessProbe" in content

    def test_deployment_uses_configmap_ref(self):
        content = (_TMPLS / "deployment.yaml").read_text()
        assert "configMapRef" in content

    def test_deployment_references_pvc(self):
        content = (_TMPLS / "deployment.yaml").read_text()
        assert "persistentVolumeClaim" in content

    def test_service_has_selector(self):
        content = (_TMPLS / "service.yaml").read_text()
        assert "selector" in content

    def test_configmap_sets_squish_model(self):
        content = (_TMPLS / "configmap.yaml").read_text()
        assert "SQUISH_MODEL" in content

    def test_configmap_sets_squish_host(self):
        content = (_TMPLS / "configmap.yaml").read_text()
        assert "SQUISH_HOST" in content

    def test_configmap_sets_squish_port(self):
        content = (_TMPLS / "configmap.yaml").read_text()
        assert "SQUISH_PORT" in content

    def test_pvc_uses_storage_size(self):
        content = (_TMPLS / "pvc.yaml").read_text()
        assert "storage" in content

    def test_hpa_references_deployment(self):
        content = (_TMPLS / "hpa.yaml").read_text()
        assert "Deployment" in content

    def test_hpa_has_min_max_replicas(self):
        content = (_TMPLS / "hpa.yaml").read_text()
        assert "minReplicas" in content
        assert "maxReplicas" in content

    def test_keda_scaledobject_references_prometheus(self):
        content = (_TMPLS / "keda-scaledobject.yaml").read_text()
        assert "prometheus" in content

    def test_keda_scaledobject_has_threshold(self):
        content = (_TMPLS / "keda-scaledobject.yaml").read_text()
        assert "threshold" in content

    def test_helpers_defines_fullname(self):
        content = (_TMPLS / "_helpers.tpl").read_text()
        assert "squish-serve.fullname" in content

    def test_helpers_defines_image(self):
        content = (_TMPLS / "_helpers.tpl").read_text()
        assert "squish-serve.image" in content

    def test_helpers_defines_pvc_name(self):
        content = (_TMPLS / "_helpers.tpl").read_text()
        assert "squish-serve.pvcName" in content

    def test_helpers_flavour_logic(self):
        """_helpers.tpl must have logic that appends the image flavour."""
        content = (_TMPLS / "_helpers.tpl").read_text()
        assert "flavour" in content


# ── helm lint (requires helm CLI) ─────────────────────────────────────────────

@pytest.mark.skipif(not _HELM_AVAILABLE, reason="helm CLI not installed")
class TestHelmLint:
    def test_helm_lint_default_values(self):
        """helm lint with default values must pass with --strict."""
        result = subprocess.run(
            ["helm", "lint", str(_CHART), "--strict"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, \
            f"helm lint failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"

    def test_helm_lint_cpu_overrides(self):
        """helm lint with CPU-only overrides must pass."""
        result = subprocess.run(
            [
                "helm", "lint", str(_CHART), "--strict",
                "--set", "image.flavour=cpu",
                "--set", "autoscaling.enabled=true",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, \
            f"helm lint (cpu) failed:\n{result.stdout}\n{result.stderr}"

    def test_helm_lint_keda_enabled(self):
        """helm lint with KEDA enabled must pass."""
        result = subprocess.run(
            [
                "helm", "lint", str(_CHART), "--strict",
                "--set", "keda.enabled=true",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, \
            f"helm lint (keda) failed:\n{result.stdout}\n{result.stderr}"


# ── helm template dry-run renders (requires helm CLI) ─────────────────────────

@pytest.mark.skipif(not _HELM_AVAILABLE, reason="helm CLI not installed")
class TestHelmTemplate:
    @staticmethod
    def _template(extra_args: list[str]) -> list[dict]:
        """Run `helm template` and parse the rendered manifests."""
        result = subprocess.run(
            ["helm", "template", "squish-test", str(_CHART)] + extra_args,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, \
            f"helm template failed:\n{result.stdout}\n{result.stderr}"
        docs = list(yaml.safe_load_all(result.stdout))
        return [d for d in docs if d is not None]

    def test_default_render_produces_deployment(self):
        docs = self._template([])
        kinds = [d.get("kind") for d in docs]
        assert "Deployment" in kinds

    def test_default_render_produces_service(self):
        docs = self._template([])
        kinds = [d.get("kind") for d in docs]
        assert "Service" in kinds

    def test_default_render_produces_configmap(self):
        docs = self._template([])
        kinds = [d.get("kind") for d in docs]
        assert "ConfigMap" in kinds

    def test_default_render_produces_pvc(self):
        docs = self._template([])
        kinds = [d.get("kind") for d in docs]
        assert "PersistentVolumeClaim" in kinds

    def test_default_render_produces_service_account(self):
        docs = self._template([])
        kinds = [d.get("kind") for d in docs]
        assert "ServiceAccount" in kinds

    def test_hpa_disabled_by_default(self):
        docs = self._template([])
        kinds = [d.get("kind") for d in docs]
        assert "HorizontalPodAutoscaler" not in kinds

    def test_hpa_enabled_when_flag_set(self):
        docs = self._template(["--set", "autoscaling.enabled=true"])
        kinds = [d.get("kind") for d in docs]
        assert "HorizontalPodAutoscaler" in kinds

    def test_keda_disabled_by_default(self):
        docs = self._template([])
        kinds = [d.get("kind") for d in docs]
        assert "ScaledObject" not in kinds

    def test_keda_enabled_when_flag_set(self):
        docs = self._template(["--set", "keda.enabled=true"])
        kinds = [d.get("kind") for d in docs]
        assert "ScaledObject" in kinds

    def test_configmap_contains_squish_model(self):
        docs = self._template([
            "--set", "model.id=/models/my-model",
        ])
        cm = next(d for d in docs if d.get("kind") == "ConfigMap")
        assert cm["data"]["SQUISH_MODEL"] == "/models/my-model"

    def test_configmap_contains_squish_host(self):
        docs = self._template([])
        cm = next(d for d in docs if d.get("kind") == "ConfigMap")
        assert "SQUISH_HOST" in cm["data"]

    def test_deployment_model_volume_mount(self):
        docs = self._template([])
        dep = next(d for d in docs if d.get("kind") == "Deployment")
        containers = dep["spec"]["template"]["spec"]["containers"]
        mounts = containers[0]["volumeMounts"]
        mount_paths = [m["mountPath"] for m in mounts]
        assert "/models" in mount_paths

    def test_deployment_image_cuda_default(self):
        docs = self._template([])
        dep = next(d for d in docs if d.get("kind") == "Deployment")
        image = dep["spec"]["template"]["spec"]["containers"][0]["image"]
        assert "cuda" in image

    def test_deployment_image_cpu_flavour(self):
        docs = self._template(["--set", "image.flavour=cpu"])
        dep = next(d for d in docs if d.get("kind") == "Deployment")
        image = dep["spec"]["template"]["spec"]["containers"][0]["image"]
        assert "cpu" in image

    def test_pvc_not_rendered_when_existing_claim(self):
        docs = self._template([
            "--set", "persistence.existingClaim=my-pvc",
            "--set", "persistence.enabled=false",
        ])
        kinds = [d.get("kind") for d in docs]
        assert "PersistentVolumeClaim" not in kinds
