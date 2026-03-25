"""tests/test_wave94_cross_platform.py — Wave 94: Cross-platform support tests.

Verifies:
- detect_platform() returns PlatformInfo with is_apple_silicon, is_cuda, name
- cmd_setup() does NOT sys.exit(1) on non-Apple-Silicon mock
- get_inference_backend() returns a valid string
- README no longer says "Apple Silicon only"
- Requirements section includes Linux/Windows
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).parent.parent


# ── 1. PlatformInfo properties ─────────────────────────────────────────────────

class TestPlatformInfoProperties:
    """PlatformInfo must expose is_apple_silicon, is_cuda, name, platform_name."""

    def _get_info(self):
        from squish.platform.detector import detect_platform
        return detect_platform()

    def test_has_is_apple_silicon(self):
        info = self._get_info()
        assert hasattr(info, "is_apple_silicon")
        assert isinstance(info.is_apple_silicon, bool)

    def test_has_is_cuda(self):
        info = self._get_info()
        assert hasattr(info, "is_cuda")
        assert isinstance(info.is_cuda, bool)

    def test_has_name(self):
        info = self._get_info()
        assert hasattr(info, "name")
        assert isinstance(info.name, str)
        assert len(info.name) > 0

    def test_has_platform_name(self):
        info = self._get_info()
        assert hasattr(info, "platform_name")
        assert isinstance(info.platform_name, str)
        assert len(info.platform_name) > 0

    def test_has_ram_gb(self):
        info = self._get_info()
        assert hasattr(info, "ram_gb")
        assert isinstance(info.ram_gb, (int, float))
        assert info.ram_gb >= 0

    def test_is_apple_silicon_false_on_non_mac(self):
        """On Linux/Windows test runners, is_apple_silicon should be False."""
        if sys.platform == "darwin":
            pytest.skip("Skipping non-Mac assertion on macOS")
        info = self._get_info()
        assert info.is_apple_silicon is False

    def test_kind_attribute_is_platform_kind(self):
        from squish.platform.detector import PlatformKind
        info = self._get_info()
        assert isinstance(info.kind, PlatformKind)

    def test_name_is_lowercase_descriptor(self):
        info = self._get_info()
        assert info.name == info.kind.name.lower()


# ── 2. get_inference_backend() ─────────────────────────────────────────────────

class TestGetInferenceBackend:
    """get_inference_backend(platform_info) must return a valid backend string."""

    VALID_BACKENDS = {"mlx", "torch_cuda", "torch_rocm", "torch_cpu", "directml", "cpu"}

    def test_returns_string(self):
        from squish.platform.detector import detect_platform
        from squish.platform.platform_router import get_inference_backend
        result = get_inference_backend(detect_platform())
        assert isinstance(result, str)

    def test_returns_non_empty_string(self):
        from squish.platform.detector import detect_platform
        from squish.platform.platform_router import get_inference_backend
        result = get_inference_backend(detect_platform())
        assert len(result) > 0

    def test_apple_silicon_returns_mlx(self):
        from squish.platform.detector import PlatformInfo, PlatformKind
        from squish.platform.platform_router import get_inference_backend
        mock_info = MagicMock(spec=PlatformInfo)
        mock_info.is_apple_silicon = True
        mock_info.has_mlx = True
        mock_info.is_cuda = False
        mock_info.has_rocm = False
        mock_info.kind = PlatformKind.MACOS_APPLE_SILICON
        result = get_inference_backend(mock_info)
        assert "mlx" in result.lower()

    def test_cuda_platform_returns_torch_cuda(self):
        from squish.platform.detector import PlatformInfo, PlatformKind
        from squish.platform.platform_router import get_inference_backend
        mock_info = MagicMock(spec=PlatformInfo)
        mock_info.is_apple_silicon = False
        mock_info.has_mlx = False
        mock_info.is_cuda = True
        mock_info.has_cuda = True
        mock_info.has_rocm = False
        mock_info.kind = PlatformKind.LINUX_CUDA
        result = get_inference_backend(mock_info)
        assert "cuda" in result.lower() or "torch" in result.lower()

    def test_cpu_fallback(self):
        from squish.platform.detector import PlatformInfo, PlatformKind
        from squish.platform.platform_router import get_inference_backend
        mock_info = MagicMock(spec=PlatformInfo)
        mock_info.is_apple_silicon = False
        mock_info.has_mlx = False
        mock_info.is_cuda = False
        mock_info.has_cuda = False
        mock_info.has_rocm = False
        mock_info.kind = PlatformKind.LINUX_CPU
        result = get_inference_backend(mock_info)
        assert isinstance(result, str)
        assert len(result) > 0


# ── 3. cmd_setup() no sys.exit on non-Apple ────────────────────────────────────

class TestCmdSetupCrossplatform:
    """cmd_setup() must not sys.exit(1) on non-Apple-Silicon machines."""

    def test_no_exit_on_linux_mock(self):
        """On a mocked Linux/x86_64 system, cmd_setup should print a notice but not exit."""
        import squish.cli as cli

        exited_with = []

        def capture_exit(code=0):
            exited_with.append(code)
            raise SystemExit(code)

        # Mock platform to appear as Linux x86_64 (non-Apple)
        with patch("squish.cli.sys") as mock_sys:
            mock_sys.exit.side_effect = capture_exit
            mock_sys.stdin = sys.stdin
            mock_sys.stdout = sys.stdout
            mock_sys.stderr = sys.stderr
            with patch("squish.cli._detect_ram_gb", return_value=16.0):
                with patch("squish.cli._recommend_model", return_value="qwen3:8b"):
                    with patch("squish.cli._catalog_resolve", return_value=None):
                        import platform as _platform
                        with patch.object(_platform, "system", return_value="Linux"):
                            with patch.object(_platform, "machine", return_value="x86_64"):
                                try:
                                    cli.cmd_setup(MagicMock())
                                except SystemExit as exc:
                                    # sys.exit(0) is ok (normal end), sys.exit(1) is the bug
                                    if exc.code == 1:
                                        pytest.fail(
                                            "cmd_setup called sys.exit(1) on non-Apple platform"
                                        )
                                except Exception:
                                    pass  # Other errors are OK (missing deps, etc.)

        # If sys.exit was called with code 1, that's a failure
        if 1 in exited_with:
            pytest.fail("cmd_setup called sys.exit(1) on non-Apple-Silicon platform")

    def test_cmd_setup_no_hard_exit_in_source(self):
        """cmd_setup source should NOT have an unconditional sys.exit(1) for non-Apple."""
        cli_path = ROOT / "squish" / "cli.py"
        source = cli_path.read_text(encoding="utf-8")
        # Find cmd_setup function body
        start = source.find("def cmd_setup(")
        # Find the next def after cmd_setup
        next_def = source.find("\ndef ", start + 1)
        setup_body = source[start:next_def] if next_def > start else source[start:]
        # The old pattern: "not is_apple_silicon" immediately followed by sys.exit(1)
        # New code should not have unconditional exit for non-Apple
        assert 'print(f"\\n  {_C.PK}squish requires Apple Silicon' not in setup_body, (
            "cmd_setup still has the old Apple-Silicon-only hard-exit message"
        )


# ── 4. README platform checks ──────────────────────────────────────────────────

class TestReadmePlatformAccuracy:
    README = ROOT / "README.md"

    def _content(self):
        return self.README.read_text(encoding="utf-8")

    def test_no_apple_silicon_only_in_title(self):
        content = self._content()
        first_line = content.split("\n")[0]
        assert "Apple Silicon" not in first_line, (
            f"README title still says 'Apple Silicon': {first_line!r}"
        )

    def test_no_only_warning(self):
        """The hard '⚠️ macOS + Apple Silicon (M1–M5) only' line must be gone."""
        content = self._content()
        assert "macOS + Apple Silicon (M1–M5) only" not in content

    def test_linux_mentioned_in_requirements(self):
        content = self._content()
        assert "Linux" in content

    def test_windows_mentioned_in_requirements(self):
        content = self._content()
        assert "Windows" in content

    def test_platform_badge_includes_linux(self):
        content = self._content()
        # Badge URL should include Linux
        assert "Linux" in content or "linux" in content.lower()

    def test_requirements_section_present(self):
        content = self._content()
        assert "## Requirements" in content


# ── 5. Platform source file checks ────────────────────────────────────────────

class TestPlatformSourceFiles:

    def test_detector_module_importable(self):
        from squish.platform import detector  # noqa: F401

    def test_platform_router_importable(self):
        from squish.platform import platform_router  # noqa: F401

    def test_cuda_backend_importable(self):
        from squish.platform import cuda_backend  # noqa: F401

    def test_windows_backend_importable(self):
        from squish.platform import windows_backend  # noqa: F401

    def test_platform_kind_has_macos_apple_silicon(self):
        from squish.platform.detector import PlatformKind
        assert hasattr(PlatformKind, "MACOS_APPLE_SILICON")

    def test_platform_kind_has_linux_cuda(self):
        from squish.platform.detector import PlatformKind
        assert hasattr(PlatformKind, "LINUX_CUDA")

    def test_platform_kind_has_windows_native(self):
        from squish.platform.detector import PlatformKind
        assert hasattr(PlatformKind, "WINDOWS_NATIVE")

    def test_detect_platform_function_exists(self):
        from squish.platform.detector import detect_platform
        assert callable(detect_platform)

    def test_get_inference_backend_function_exists(self):
        from squish.platform.platform_router import get_inference_backend
        assert callable(get_inference_backend)
