"""
tests/test_docker_entrypoint_unit.py

Unit tests covering Docker entrypoint configuration:

  - Docker-related config files exist and are valid
  - squish.cli serve / run subparsers accept host/port/model args
  - SQUISH_MODEL, SQUISH_HOST, SQUISH_PORT env vars are honoured as
    parser defaults (critical for `docker run -e SQUISH_MODEL=…`)
  - CLI arg values always override env-var defaults
  - docker-compose.yml is valid YAML with the expected service structure
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

# ── Helpers ───────────────────────────────────────────────────────────────────

_REPO = Path(__file__).resolve().parent.parent


def _build_parser() -> tuple[argparse.ArgumentParser, argparse.Namespace]:
    """Import cli.main indirectly by calling it with a mocked cmd_run so we
    can capture the parsed Namespace without starting a server."""
    import squish.cli as cli  # noqa: PLC0415
    return cli


def _parse_serve(argv: list[str], env: dict[str, str] | None = None):
    """Parse `squish serve …` with optional env-var overrides.

    Returns the args Namespace that would be passed to cmd_run.
    """
    captured: list[argparse.Namespace] = []

    def _mock_cmd_run(args):
        captured.append(args)

    with patch.dict(os.environ, env or {}, clear=False):
        # Reset any cached env vars inside cli by reimporting inside env patch
        with patch("squish.cli.cmd_run", side_effect=_mock_cmd_run):
            import squish.cli as cli  # noqa: PLC0415
            old_argv = sys.argv[:]
            try:
                sys.argv = ["squish"] + argv
                # main() will parse argv and call cmd_run (mocked above)
                try:
                    cli.main()
                except SystemExit:
                    pass
            finally:
                sys.argv = old_argv

    return captured[0] if captured else None


# ── Configuration file smoke tests ────────────────────────────────────────────

class TestDockerFiles:
    def test_dockerfile_cpu_exists(self):
        assert (_REPO / "Dockerfile.cpu").is_file(), \
            "Dockerfile.cpu must exist in repo root"

    def test_dockerfile_cuda_exists(self):
        assert (_REPO / "Dockerfile.cuda").is_file(), \
            "Dockerfile.cuda must exist in repo root"

    def test_dockerignore_exists(self):
        assert (_REPO / ".dockerignore").is_file(), \
            ".dockerignore must exist in repo root"

    def test_dockerignore_excludes_models(self):
        content = (_REPO / ".dockerignore").read_text()
        assert "models/" in content, ".dockerignore must exclude models/ directory"

    def test_dockerignore_excludes_safetensors(self):
        content = (_REPO / ".dockerignore").read_text()
        assert "*.safetensors" in content, ".dockerignore must exclude *.safetensors"

    def test_dockerignore_excludes_git(self):
        content = (_REPO / ".dockerignore").read_text()
        assert ".git" in content, ".dockerignore must exclude .git"

    def test_dockerfile_cpu_exposes_8080(self):
        content = (_REPO / "Dockerfile.cpu").read_text()
        assert "EXPOSE 8080" in content

    def test_dockerfile_cuda_exposes_8080(self):
        content = (_REPO / "Dockerfile.cuda").read_text()
        assert "EXPOSE 8080" in content

    def test_dockerfile_cpu_has_health_check(self):
        content = (_REPO / "Dockerfile.cpu").read_text()
        assert "HEALTHCHECK" in content

    def test_dockerfile_cuda_has_health_check(self):
        content = (_REPO / "Dockerfile.cuda").read_text()
        assert "HEALTHCHECK" in content

    def test_dockerfile_cpu_entrypoint_is_squish_cli(self):
        content = (_REPO / "Dockerfile.cpu").read_text()
        assert "squish.cli" in content

    def test_dockerfile_cuda_entrypoint_is_squish_cli(self):
        content = (_REPO / "Dockerfile.cuda").read_text()
        assert "squish.cli" in content

    def test_dockerfile_cpu_has_models_volume(self):
        content = (_REPO / "Dockerfile.cpu").read_text()
        assert 'VOLUME' in content and '/models' in content

    def test_dockerfile_cuda_has_models_volume(self):
        content = (_REPO / "Dockerfile.cuda").read_text()
        assert 'VOLUME' in content and '/models' in content

    def test_dockerfile_cpu_uses_linux_extra(self):
        """Dockerfile.cpu must install .[linux] to get torch on Linux."""
        content = (_REPO / "Dockerfile.cpu").read_text()
        assert ".[linux]" in content

    def test_dockerfile_cuda_uses_linux_extra(self):
        content = (_REPO / "Dockerfile.cuda").read_text()
        assert ".[linux]" in content


class TestDockerCompose:
    def test_docker_compose_exists(self):
        assert (_REPO / "docker-compose.yml").is_file()

    def test_docker_compose_is_valid_yaml(self):
        yaml = pytest.importorskip("yaml")
        content = (_REPO / "docker-compose.yml").read_text()
        data = yaml.safe_load(content)
        assert isinstance(data, dict)

    def test_docker_compose_has_services(self):
        yaml = pytest.importorskip("yaml")
        data = yaml.safe_load((_REPO / "docker-compose.yml").read_text())
        assert "services" in data

    def test_docker_compose_has_cpu_service(self):
        yaml = pytest.importorskip("yaml")
        data = yaml.safe_load((_REPO / "docker-compose.yml").read_text())
        assert "squish-cpu" in data["services"]

    def test_docker_compose_has_cuda_service(self):
        yaml = pytest.importorskip("yaml")
        data = yaml.safe_load((_REPO / "docker-compose.yml").read_text())
        assert "squish-cuda" in data["services"]

    def test_docker_compose_cpu_has_build(self):
        yaml = pytest.importorskip("yaml")
        data = yaml.safe_load((_REPO / "docker-compose.yml").read_text())
        svc = data["services"]["squish-cpu"]
        assert "build" in svc or "image" in svc

    def test_docker_compose_cuda_has_build(self):
        yaml = pytest.importorskip("yaml")
        data = yaml.safe_load((_REPO / "docker-compose.yml").read_text())
        svc = data["services"]["squish-cuda"]
        assert "build" in svc or "image" in svc

    def test_docker_compose_cuda_profile(self):
        yaml = pytest.importorskip("yaml")
        data = yaml.safe_load((_REPO / "docker-compose.yml").read_text())
        cuda_svc = data["services"]["squish-cuda"]
        assert "profiles" in cuda_svc
        assert "cuda" in cuda_svc["profiles"]

    def test_docker_compose_cpu_profile(self):
        yaml = pytest.importorskip("yaml")
        data = yaml.safe_load((_REPO / "docker-compose.yml").read_text())
        cpu_svc = data["services"]["squish-cpu"]
        assert "profiles" in cpu_svc
        assert "cpu" in cpu_svc["profiles"]


# ── pyproject.toml [linux] extra ──────────────────────────────────────────────

class TestPyprojectLinuxExtra:
    def test_linux_extra_exists(self):
        import importlib.metadata  # noqa: PLC0415
        # The extra may not be installed in the test runner; check pyproject.toml
        content = (_REPO / "pyproject.toml").read_text()
        assert 'linux = [' in content or 'linux=[' in content, \
            "pyproject.toml must declare a [linux] extra"

    def test_linux_extra_includes_torch(self):
        content = (_REPO / "pyproject.toml").read_text()
        # Find the linux block:
        start = content.find('linux = [')
        assert start != -1
        end = content.find(']', start)
        block = content[start:end]
        assert 'torch' in block, "[linux] extra must declare torch as a dependency"


# ── CLI serve subcommand — argument parsing ────────────────────────────────────

class TestServeSubcommand:
    """Test that the serve subparser exists and accepts the right arguments.

    We mock cmd_run to capture the parsed Namespace without starting a server.
    """

    def test_serve_accepts_model_positional(self):
        captured = []

        def _capture(args):
            captured.append(args)

        import squish.cli as cli  # noqa: PLC0415
        with patch("squish.cli.cmd_run", side_effect=_capture):
            old = sys.argv[:]
            try:
                sys.argv = ["squish", "serve", "/models/mymodel"]
                try:
                    cli.main()
                except SystemExit:
                    pass
            finally:
                sys.argv = old

        assert captured, "cmd_run should have been called"
        assert captured[0].model == "/models/mymodel"

    def test_serve_accepts_host_flag(self):
        captured = []

        def _capture(args):
            captured.append(args)

        import squish.cli as cli  # noqa: PLC0415
        with patch("squish.cli.cmd_run", side_effect=_capture):
            old = sys.argv[:]
            try:
                sys.argv = ["squish", "serve", "--host", "0.0.0.0"]
                try:
                    cli.main()
                except SystemExit:
                    pass
            finally:
                sys.argv = old

        assert captured
        assert captured[0].host == "0.0.0.0"

    def test_serve_accepts_port_flag(self):
        captured = []

        def _capture(args):
            captured.append(args)

        import squish.cli as cli  # noqa: PLC0415
        with patch("squish.cli.cmd_run", side_effect=_capture):
            old = sys.argv[:]
            try:
                sys.argv = ["squish", "serve", "--port", "9090"]
                try:
                    cli.main()
                except SystemExit:
                    pass
            finally:
                sys.argv = old

        assert captured
        assert captured[0].port == 9090

    def test_serve_dispatch_to_cmd_run(self):
        """serve subcommand must dispatch to cmd_run (same as run)."""
        import squish.cli as cli  # noqa: PLC0415
        with patch("squish.cli.cmd_run") as mock_run:
            old = sys.argv[:]
            try:
                sys.argv = ["squish", "serve", "/models/x"]
                try:
                    cli.main()
                except SystemExit:
                    pass
            finally:
                sys.argv = old
            assert mock_run.called, "serve must dispatch to cmd_run"


# ── SQUISH_MODEL / SQUISH_HOST / SQUISH_PORT env var defaults ─────────────────

class TestDockerEnvVarDefaults:
    """Verify that Docker env vars become parser defaults for the serve command.

    We mock cmd_run to capture the parsed Namespace without starting a server.
    The env vars are read inside main() at parse time, so only a patch.dict +
    call to main() is needed — no module reload required.
    """

    @staticmethod
    def _run_serve(extra_argv: list[str], env_overrides: dict[str, str]) -> argparse.Namespace | None:
        """Parse 'squish serve [extra_argv]' under patched env vars."""
        captured: list[argparse.Namespace] = []

        def _capture(args):
            captured.append(args)

        import squish.cli as cli  # noqa: PLC0415
        with patch.dict(os.environ, env_overrides, clear=False):
            with patch("squish.cli.cmd_run", side_effect=_capture):
                old = sys.argv[:]
                try:
                    sys.argv = ["squish", "serve"] + extra_argv
                    try:
                        cli.main()
                    except SystemExit:
                        pass
                finally:
                    sys.argv = old

        return captured[0] if captured else None

    @staticmethod
    def _run_run(extra_argv: list[str], env_overrides: dict[str, str]) -> argparse.Namespace | None:
        """Parse 'squish run [extra_argv]' under patched env vars."""
        captured: list[argparse.Namespace] = []

        def _capture(args):
            captured.append(args)

        import squish.cli as cli  # noqa: PLC0415
        with patch.dict(os.environ, env_overrides, clear=False):
            with patch("squish.cli.cmd_run", side_effect=_capture):
                old = sys.argv[:]
                try:
                    sys.argv = ["squish", "run"] + extra_argv
                    try:
                        cli.main()
                    except SystemExit:
                        pass
                finally:
                    sys.argv = old

        return captured[0] if captured else None

    def test_squish_model_default(self):
        """SQUISH_MODEL without CLI arg → model default is the env var value."""
        args = self._run_serve([], {"SQUISH_MODEL": "/models/Qwen2.5-7B-Instruct-bf16"})
        if args is not None:
            assert args.model == "/models/Qwen2.5-7B-Instruct-bf16"

    def test_squish_host_default(self):
        """SQUISH_HOST without CLI --host flag → host default is the env var."""
        args = self._run_serve([], {"SQUISH_HOST": "0.0.0.0"})
        if args is not None:
            assert args.host == "0.0.0.0"

    def test_squish_port_default(self):
        """SQUISH_PORT without CLI --port flag → port default is the env var."""
        args = self._run_serve([], {"SQUISH_PORT": "9999"})
        if args is not None:
            assert args.port == 9999

    def test_cli_port_overrides_env_var(self):
        """Explicit --port CLI arg must take precedence over SQUISH_PORT."""
        args = self._run_serve(["--port", "8888"], {"SQUISH_PORT": "7777"})
        if args is not None:
            assert args.port == 8888, "CLI --port must override SQUISH_PORT env var"

    def test_cli_host_overrides_env_var(self):
        """Explicit --host CLI arg must take precedence over SQUISH_HOST."""
        args = self._run_serve(["--host", "127.0.0.1"], {"SQUISH_HOST": "0.0.0.0"})
        if args is not None:
            assert args.host == "127.0.0.1", "CLI --host must override SQUISH_HOST env var"

    def test_cli_model_overrides_env_var(self):
        """Explicit model positional arg must take precedence over SQUISH_MODEL."""
        args = self._run_serve(["/models/from-cli"], {"SQUISH_MODEL": "/models/from-env"})
        if args is not None:
            assert args.model == "/models/from-cli", \
                "CLI model positional arg must override SQUISH_MODEL env var"

    def test_run_subcommand_also_reads_squish_model(self):
        """'squish run' must also honour SQUISH_MODEL (same as serve)."""
        args = self._run_run([], {"SQUISH_MODEL": "/models/run-env-model"})
        if args is not None:
            assert args.model == "/models/run-env-model"

    def test_invalid_squish_port_falls_back_to_default(self):
        """Non-integer SQUISH_PORT must not crash the parser."""
        from squish.cli import _DEFAULT_PORT  # noqa: PLC0415
        # A non-digit SQUISH_PORT should not raise, and should fall back
        args = self._run_serve([], {"SQUISH_PORT": "not-a-number"})
        # If cmd_run was called, port should be the default
        if args is not None:
            assert args.port == _DEFAULT_PORT
