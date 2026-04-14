"""tests/test_squash_wave50.py — Wave 50: Shadow AI detection.

Coverage
--------
- :data:`SHADOW_AI_MODEL_EXTENSIONS` — expected extension set
- :func:`scan_pod_for_model_files` — host paths, volume mounts, env vars, args,
  init containers, no-hit pods, empty spec
- :class:`ShadowAiScanner` — scan_pod_list, scan_namespace, empty list, namespace
  filtering, multi-pod batches
- :class:`ShadowAiScanResult` — ok semantics, summary format, pods_scanned count
- :class:`ShadowAiConfig` — defaults and overrides
- :class:`ShadowAiHit` — field correctness
- :class:`WebhookConfig` — shadow_ai_scan_mode field default
- CLI ``squash shadow-ai scan`` — arg parsing, --fail-on-hits exit codes,
  stdin JSON, --output-json, --namespace filter, --extensions override
- Module count gate — squish/ must stay at 124 Python files
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent

import pytest


# ---------------------------------------------------------------------------
# Pod fixture helpers
# ---------------------------------------------------------------------------


def _clean_pod(name: str = "app-pod", namespace: str = "default") -> dict:
    """Minimal pod spec with no model file references."""
    return {
        "metadata": {"name": name, "namespace": namespace},
        "spec": {
            "containers": [
                {
                    "name": "app",
                    "image": "nginx:latest",
                    "volumeMounts": [{"mountPath": "/data", "name": "data"}],
                    "env": [{"name": "HOME", "value": "/root"}],
                    "args": ["--port", "8080"],
                }
            ],
            "volumes": [{"name": "data", "emptyDir": {}}],
        },
    }


def _pod_with_host_path(path: str, name: str = "shadow-pod") -> dict:
    return {
        "metadata": {"name": name, "namespace": "default"},
        "spec": {
            "containers": [{"name": "llm", "image": "pytorch:latest"}],
            "volumes": [{"name": "models", "hostPath": {"path": path, "type": "File"}}],
        },
    }


def _pod_with_volume_mount(mount_path: str) -> dict:
    return {
        "metadata": {"name": "vm-pod", "namespace": "prod"},
        "spec": {
            "containers": [
                {
                    "name": "worker",
                    "image": "worker:v1",
                    "volumeMounts": [{"mountPath": mount_path, "name": "models"}],
                }
            ],
        },
    }


def _pod_with_env(value: str) -> dict:
    return {
        "metadata": {"name": "env-pod", "namespace": "staging"},
        "spec": {
            "containers": [
                {
                    "name": "runner",
                    "image": "runner:v1",
                    "env": [{"name": "MODEL_PATH", "value": value}],
                }
            ],
        },
    }


def _pod_with_args(args: list) -> dict:
    return {
        "metadata": {"name": "arg-pod", "namespace": "default"},
        "spec": {
            "containers": [
                {
                    "name": "serve",
                    "image": "serve:v1",
                    "args": args,
                }
            ],
        },
    }


def _pod_with_init_container(mount_path: str) -> dict:
    return {
        "metadata": {"name": "init-pod", "namespace": "default"},
        "spec": {
            "containers": [{"name": "app", "image": "app:v1"}],
            "initContainers": [
                {
                    "name": "init",
                    "image": "busybox",
                    "volumeMounts": [{"mountPath": mount_path, "name": "models"}],
                }
            ],
        },
    }


def _pod_list(*pods) -> dict:
    return {"apiVersion": "v1", "kind": "PodList", "items": list(pods)}


# ---------------------------------------------------------------------------
# SHADOW_AI_MODEL_EXTENSIONS
# ---------------------------------------------------------------------------


class TestShadowAiModelExtensions:
    def test_gguf_present(self):
        from squish.squash.integrations.kubernetes import SHADOW_AI_MODEL_EXTENSIONS
        assert ".gguf" in SHADOW_AI_MODEL_EXTENSIONS

    def test_safetensors_present(self):
        from squish.squash.integrations.kubernetes import SHADOW_AI_MODEL_EXTENSIONS
        assert ".safetensors" in SHADOW_AI_MODEL_EXTENSIONS

    def test_pt_present(self):
        from squish.squash.integrations.kubernetes import SHADOW_AI_MODEL_EXTENSIONS
        assert ".pt" in SHADOW_AI_MODEL_EXTENSIONS

    def test_onnx_present(self):
        from squish.squash.integrations.kubernetes import SHADOW_AI_MODEL_EXTENSIONS
        assert ".onnx" in SHADOW_AI_MODEL_EXTENSIONS

    def test_tflite_present(self):
        from squish.squash.integrations.kubernetes import SHADOW_AI_MODEL_EXTENSIONS
        assert ".tflite" in SHADOW_AI_MODEL_EXTENSIONS

    def test_mlmodel_present(self):
        from squish.squash.integrations.kubernetes import SHADOW_AI_MODEL_EXTENSIONS
        assert ".mlmodel" in SHADOW_AI_MODEL_EXTENSIONS

    def test_bin_and_pkl_present(self):
        from squish.squash.integrations.kubernetes import SHADOW_AI_MODEL_EXTENSIONS
        assert ".bin" in SHADOW_AI_MODEL_EXTENSIONS
        assert ".pkl" in SHADOW_AI_MODEL_EXTENSIONS

    def test_is_frozenset(self):
        from squish.squash.integrations.kubernetes import SHADOW_AI_MODEL_EXTENSIONS
        assert isinstance(SHADOW_AI_MODEL_EXTENSIONS, frozenset)

    def test_all_lowercase_with_dot(self):
        from squish.squash.integrations.kubernetes import SHADOW_AI_MODEL_EXTENSIONS
        for ext in SHADOW_AI_MODEL_EXTENSIONS:
            assert ext.startswith("."), f"extension {ext!r} missing leading dot"
            assert ext == ext.lower(), f"extension {ext!r} is not lowercase"


# ---------------------------------------------------------------------------
# ShadowAiConfig
# ---------------------------------------------------------------------------


class TestShadowAiConfig:
    def test_defaults(self):
        from squish.squash.integrations.kubernetes import (
            ShadowAiConfig,
            SHADOW_AI_MODEL_EXTENSIONS,
        )
        cfg = ShadowAiConfig()
        assert cfg.scan_extensions == SHADOW_AI_MODEL_EXTENSIONS
        assert cfg.scan_volume_mounts is True
        assert cfg.scan_env_vars is True
        assert cfg.scan_args is True
        assert cfg.namespaces_include == []

    def test_custom_extensions(self):
        from squish.squash.integrations.kubernetes import ShadowAiConfig
        custom = frozenset({".gguf"})
        cfg = ShadowAiConfig(scan_extensions=custom)
        assert cfg.scan_extensions == custom

    def test_namespaces_include(self):
        from squish.squash.integrations.kubernetes import ShadowAiConfig
        cfg = ShadowAiConfig(namespaces_include=["prod", "staging"])
        assert "prod" in cfg.namespaces_include

    def test_disable_env_scan(self):
        from squish.squash.integrations.kubernetes import ShadowAiConfig
        cfg = ShadowAiConfig(scan_env_vars=False)
        assert cfg.scan_env_vars is False


# ---------------------------------------------------------------------------
# scan_pod_for_model_files
# ---------------------------------------------------------------------------


class TestScanPodForModelFiles:
    def _scan(self, pod, **kwargs):
        from squish.squash.integrations.kubernetes import (
            ShadowAiConfig,
            scan_pod_for_model_files,
        )
        return scan_pod_for_model_files(pod, ShadowAiConfig(**kwargs))

    def test_clean_pod_no_hits(self):
        hits = self._scan(_clean_pod())
        assert hits == []

    def test_empty_spec_no_hits(self):
        hits = self._scan({"metadata": {"name": "x", "namespace": "y"}, "spec": {}})
        assert hits == []

    def test_bare_minimal_pod_no_hits(self):
        hits = self._scan({})
        assert hits == []

    def test_host_path_gguf_hit(self):
        pod = _pod_with_host_path("/mnt/models/llama3.gguf")
        hits = self._scan(pod)
        assert len(hits) == 1
        assert hits[0].location_type == "host_path"
        assert hits[0].extension == ".gguf"
        assert hits[0].matched_value == "/mnt/models/llama3.gguf"
        assert hits[0].container_name == ""

    def test_host_path_safetensors_hit(self):
        pod = _pod_with_host_path("/data/model.safetensors")
        hits = self._scan(pod)
        assert len(hits) == 1
        assert hits[0].extension == ".safetensors"

    def test_host_path_no_model_extension_no_hit(self):
        pod = _pod_with_host_path("/data/myapp.txt")
        hits = self._scan(pod)
        assert hits == []

    def test_volume_mount_pt_hit(self):
        pod = _pod_with_volume_mount("/models/weights.pt")
        hits = self._scan(pod)
        assert len(hits) == 1
        assert hits[0].location_type == "volume_mount"
        assert hits[0].extension == ".pt"
        assert hits[0].namespace == "prod"

    def test_volume_mount_no_hit_for_directory(self):
        pod = _pod_with_volume_mount("/models/")
        hits = self._scan(pod)
        assert hits == []

    def test_env_var_gguf_hit(self):
        pod = _pod_with_env("/volumes/llama.gguf")
        hits = self._scan(pod)
        assert len(hits) == 1
        assert hits[0].location_type == "env"
        assert hits[0].extension == ".gguf"

    def test_env_var_no_model_value_no_hit(self):
        pod = _pod_with_env("production")
        hits = self._scan(pod)
        assert hits == []

    def test_arg_model_path_hit(self):
        pod = _pod_with_args(["--model", "/models/qwen.gguf", "--port", "8080"])
        hits = self._scan(pod)
        assert len(hits) == 1
        assert hits[0].location_type == "arg"
        assert hits[0].matched_value == "/models/qwen.gguf"

    def test_arg_ckpt_hit(self):
        pod = _pod_with_args(["/checkpoints/run.ckpt"])
        hits = self._scan(pod)
        assert len(hits) == 1
        assert hits[0].extension == ".ckpt"

    def test_init_container_volume_mount_hit(self):
        pod = _pod_with_init_container("/init/model.safetensors")
        hits = self._scan(pod)
        assert len(hits) == 1
        assert hits[0].container_name == "init"
        assert hits[0].location_type == "volume_mount"

    def test_pod_name_and_namespace_in_hit(self):
        pod = _pod_with_host_path("/m.gguf", name="my-pod")
        pod["metadata"]["namespace"] = "my-ns"
        hits = self._scan(pod)
        assert hits[0].pod_name == "my-pod"
        assert hits[0].namespace == "my-ns"

    def test_disable_env_scan_skips_env_hits(self):
        pod = _pod_with_env("/vol/model.gguf")
        hits = self._scan(pod, scan_env_vars=False)
        assert hits == []

    def test_disable_args_scan_skips_arg_hits(self):
        pod = _pod_with_args(["--model=/vol/weights.bin"])
        hits = self._scan(pod, scan_args=False)
        assert hits == []

    def test_disable_volume_mount_scan_skips_mount_hits(self):
        pod = _pod_with_volume_mount("/m/model.pt")
        hits = self._scan(pod, scan_volume_mounts=False)
        assert hits == []

    def test_multiple_containers_multiple_hits(self):
        pod = {
            "metadata": {"name": "multi", "namespace": "default"},
            "spec": {
                "containers": [
                    {
                        "name": "a",
                        "image": "img",
                        "env": [{"name": "M", "value": "/a/a.gguf"}],
                    },
                    {
                        "name": "b",
                        "image": "img",
                        "args": ["/b/b.safetensors"],
                    },
                ]
            },
        }
        hits = self._scan(pod)
        assert len(hits) == 2
        names = {h.container_name for h in hits}
        assert names == {"a", "b"}

    def test_custom_extension_set(self):
        from squish.squash.integrations.kubernetes import ShadowAiConfig, scan_pod_for_model_files
        pod = _pod_with_env("/vol/model.xyz")
        hits = scan_pod_for_model_files(pod, ShadowAiConfig(scan_extensions=frozenset({".xyz"})))
        assert len(hits) == 1
        assert hits[0].extension == ".xyz"

    def test_extension_matching_is_case_insensitive(self):
        pod = _pod_with_env("/vol/MODEL.GGUF")
        hits = self._scan(pod)
        assert len(hits) == 1
        assert hits[0].extension == ".gguf"


# ---------------------------------------------------------------------------
# ShadowAiScanResult
# ---------------------------------------------------------------------------


class TestShadowAiScanResult:
    def _result(self, hits, pods_scanned=1):
        from squish.squash.integrations.kubernetes import ShadowAiScanResult
        ok = len(hits) == 0
        summary = "clean" if ok else f"{len(hits)} hit(s)"
        return ShadowAiScanResult(hits=hits, pods_scanned=pods_scanned, ok=ok, summary=summary)

    def test_ok_true_when_no_hits(self):
        r = self._result([])
        assert r.ok is True

    def test_ok_false_when_hits_present(self):
        from squish.squash.integrations.kubernetes import ShadowAiHit
        hit = ShadowAiHit("p", "ns", "c", "arg", "/m.gguf", ".gguf")
        r = self._result([hit])
        assert r.ok is False

    def test_pods_scanned_field(self):
        r = self._result([], pods_scanned=42)
        assert r.pods_scanned == 42

    def test_summary_string_nonempty(self):
        r = self._result([])
        assert isinstance(r.summary, str)
        assert r.summary


# ---------------------------------------------------------------------------
# ShadowAiScanner
# ---------------------------------------------------------------------------


class TestShadowAiScanner:
    def _scanner(self, **kwargs):
        from squish.squash.integrations.kubernetes import ShadowAiConfig, ShadowAiScanner
        return ShadowAiScanner(ShadowAiConfig(**kwargs))

    def test_scan_empty_pod_list(self):
        scanner = self._scanner()
        result = scanner.scan_pod_list({"items": []})
        assert result.ok is True
        assert result.pods_scanned == 0

    def test_scan_pod_list_missing_items_key(self):
        scanner = self._scanner()
        result = scanner.scan_pod_list({})
        assert result.ok is True
        assert result.pods_scanned == 0

    def test_clean_pod_list_ok(self):
        scanner = self._scanner()
        result = scanner.scan_pod_list(_pod_list(_clean_pod()))
        assert result.ok is True
        assert result.pods_scanned == 1

    def test_pods_scanned_counts_all_items(self):
        scanner = self._scanner()
        pl = _pod_list(_clean_pod("p1"), _clean_pod("p2"), _clean_pod("p3"))
        result = scanner.scan_pod_list(pl)
        assert result.pods_scanned == 3

    def test_hit_pod_in_list_not_ok(self):
        scanner = self._scanner()
        pl = _pod_list(_pod_with_host_path("/m.gguf"), _clean_pod())
        result = scanner.scan_pod_list(pl)
        assert result.ok is False
        assert result.pods_scanned == 2
        assert len(result.hits) == 1

    def test_multiple_hits_across_pods(self):
        scanner = self._scanner()
        pl = _pod_list(
            _pod_with_host_path("/a.gguf"),
            _pod_with_env("/b.safetensors"),
        )
        result = scanner.scan_pod_list(pl)
        assert len(result.hits) == 2

    def test_summary_contains_count(self):
        scanner = self._scanner()
        pl = _pod_list(_pod_with_host_path("/a.gguf"))
        result = scanner.scan_pod_list(pl)
        assert "1" in result.summary

    def test_scan_namespace_entry_point(self):
        scanner = self._scanner()
        pods = [_clean_pod(), _pod_with_host_path("/m.gguf")]
        result = scanner.scan_namespace(pods)
        assert result.pods_scanned == 2
        assert not result.ok

    def test_namespace_include_filter_skips_other_namespaces(self):
        scanner = self._scanner(namespaces_include=["prod"])
        pod_a = _pod_with_env("/m.gguf")
        pod_a["metadata"]["namespace"] = "prod"
        pod_b = _pod_with_env("/m.gguf")
        pod_b["metadata"]["namespace"] = "staging"
        result = scanner.scan_pod_list(_pod_list(pod_a, pod_b))
        assert result.pods_scanned == 1
        assert not result.ok

    def test_namespace_include_filter_empty_means_all(self):
        scanner = self._scanner(namespaces_include=[])
        pod_a = _pod_with_env("/m.gguf")
        pod_a["metadata"]["namespace"] = "nsA"
        pod_b = _pod_with_env("/m.gguf")
        pod_b["metadata"]["namespace"] = "nsB"
        result = scanner.scan_pod_list(_pod_list(pod_a, pod_b))
        assert result.pods_scanned == 2

    def test_default_config_used_when_none_passed(self):
        from squish.squash.integrations.kubernetes import ShadowAiScanner
        scanner = ShadowAiScanner()
        assert scanner.config is not None


# ---------------------------------------------------------------------------
# WebhookConfig — shadow_ai_scan_mode field
# ---------------------------------------------------------------------------


class TestWebhookConfigShadowAiField:
    def test_default_shadow_ai_scan_mode_is_false(self):
        from squish.squash.integrations.kubernetes import WebhookConfig
        cfg = WebhookConfig()
        assert cfg.shadow_ai_scan_mode is False

    def test_shadow_ai_scan_mode_can_be_enabled(self):
        from squish.squash.integrations.kubernetes import WebhookConfig
        cfg = WebhookConfig(shadow_ai_scan_mode=True)
        assert cfg.shadow_ai_scan_mode is True


# ---------------------------------------------------------------------------
# ANNOTATION_SHADOW_AI constant
# ---------------------------------------------------------------------------


class TestAnnotationShadowAI:
    def test_constant_value(self):
        from squish.squash.integrations.kubernetes import ANNOTATION_SHADOW_AI
        assert ANNOTATION_SHADOW_AI == "squash.ai/shadow-ai-detected"


# ---------------------------------------------------------------------------
# CLI — squash shadow-ai scan
# ---------------------------------------------------------------------------


class TestCliShadowAiScan:
    """Tests for `squash shadow-ai scan` via _build_parser / _cmd_shadow_ai."""

    def _parser(self):
        sys.path.insert(0, str(_REPO_ROOT))
        from squish.squash.cli import _build_parser
        return _build_parser()

    def _make_pod_list_file(self, tmp_path, pods: list | None = None) -> Path:
        pl = {"apiVersion": "v1", "kind": "PodList", "items": pods or []}
        p = tmp_path / "pods.json"
        p.write_text(json.dumps(pl, indent=2), encoding="utf-8")
        return p

    def test_help_flag_exits_zero(self):
        result = subprocess.run(
            [sys.executable, "-m", "squish.squash.cli", "shadow-ai", "--help"],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "shadow" in result.stdout.lower()

    def test_scan_help_flag(self):
        result = subprocess.run(
            [sys.executable, "-m", "squish.squash.cli", "shadow-ai", "scan", "--help"],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "POD_LIST_JSON" in result.stdout

    def test_clean_pod_list_exits_zero(self, tmp_path):
        pl_file = self._make_pod_list_file(tmp_path)
        result = subprocess.run(
            [sys.executable, "-m", "squish.squash.cli", "shadow-ai", "scan", str(pl_file)],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0

    def test_shadow_pod_exits_zero_without_fail_on_hits(self, tmp_path):
        pod = _pod_with_host_path("/m.gguf")
        pl_file = self._make_pod_list_file(tmp_path, [pod])
        result = subprocess.run(
            [sys.executable, "-m", "squish.squash.cli", "shadow-ai", "scan", str(pl_file)],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0

    def test_shadow_pod_exits_two_with_fail_on_hits(self, tmp_path):
        pod = _pod_with_host_path("/m.gguf")
        pl_file = self._make_pod_list_file(tmp_path, [pod])
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "squish.squash.cli",
                "shadow-ai",
                "scan",
                str(pl_file),
                "--fail-on-hits",
            ],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 2

    def test_stdin_input(self, tmp_path):
        pl = json.dumps({"items": []})
        result = subprocess.run(
            [sys.executable, "-m", "squish.squash.cli", "shadow-ai", "scan", "-"],
            cwd=_REPO_ROOT,
            input=pl,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0

    def test_output_json_written(self, tmp_path):
        pl_file = self._make_pod_list_file(tmp_path)
        out_file = tmp_path / "result.json"
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "squish.squash.cli",
                "shadow-ai",
                "scan",
                str(pl_file),
                "--output-json",
                str(out_file),
            ],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        data = json.loads(out_file.read_text())
        assert "ok" in data
        assert "hits" in data
        assert "pods_scanned" in data
        assert data["ok"] is True

    def test_output_json_contains_hits(self, tmp_path):
        pod = _pod_with_host_path("/m.safetensors")
        pl_file = self._make_pod_list_file(tmp_path, [pod])
        out_file = tmp_path / "result.json"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "squish.squash.cli",
                "shadow-ai",
                "scan",
                str(pl_file),
                "--output-json",
                str(out_file),
            ],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
        )
        data = json.loads(out_file.read_text())
        assert data["ok"] is False
        assert len(data["hits"]) == 1
        assert data["hits"][0]["extension"] == ".safetensors"

    def test_namespace_filter(self, tmp_path):
        pod_prod = _pod_with_host_path("/m.gguf")
        pod_prod["metadata"]["namespace"] = "prod"
        pod_staging = _pod_with_host_path("/m.gguf")
        pod_staging["metadata"]["namespace"] = "staging"
        pl_file = self._make_pod_list_file(tmp_path, [pod_prod, pod_staging])
        out_file = tmp_path / "result.json"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "squish.squash.cli",
                "shadow-ai",
                "scan",
                str(pl_file),
                "--namespace",
                "staging",
                "--output-json",
                str(out_file),
            ],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
        )
        data = json.loads(out_file.read_text())
        assert data["pods_scanned"] == 1

    def test_extensions_override(self, tmp_path):
        pod = _pod_with_env("/model.xyz")
        pl_file = self._make_pod_list_file(tmp_path, [pod])
        out_file = tmp_path / "result.json"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "squish.squash.cli",
                "shadow-ai",
                "scan",
                str(pl_file),
                "--extensions",
                ".xyz",
                "--output-json",
                str(out_file),
            ],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
        )
        data = json.loads(out_file.read_text())
        assert data["ok"] is False

    def test_invalid_json_exits_one(self, tmp_path):
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("NOT JSON", encoding="utf-8")
        result = subprocess.run(
            [sys.executable, "-m", "squish.squash.cli", "shadow-ai", "scan", str(bad_file)],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 1
        assert "error" in result.stderr.lower()

    def test_missing_file_exits_one(self, tmp_path):
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "squish.squash.cli",
                "shadow-ai",
                "scan",
                str(tmp_path / "nonexistent.json"),
            ],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 1
        assert "error" in result.stderr.lower()

    def test_quiet_suppresses_output(self, tmp_path):
        pod = _pod_with_host_path("/m.gguf")
        pl_file = self._make_pod_list_file(tmp_path, [pod])
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "squish.squash.cli",
                "shadow-ai",
                "scan",
                str(pl_file),
                "--quiet",
            ],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert result.stdout.strip() == ""


# ---------------------------------------------------------------------------
# Module count gate
# ---------------------------------------------------------------------------


class TestModuleCount:
    def test_squish_module_count_still_124(self):
        """squish/ Python file count must be 125 (W51 adds drift.py — SBOM drift detection, new security domain)."""
        squish_dir = _REPO_ROOT / "squish"
        count = len(list(squish_dir.rglob("*.py")))
        assert count == 133, (
            f"squish/ Python file count is {count}, expected 133. "
            "W54-56 adds remediate.py, evaluator.py, edge_formats.py, chat.py; "
            "W57 adds model_card.py + cloud_db.py (SQLite persistence, justified). "
            "A new file was added — either remove it or update this gate with written justification."
        )
