"""Regression: a plain Linux CPU host must not be described as 'Unknown'.

platform_name had branches for Apple Silicon / CUDA / ROCm / WSL / Windows but
no LINUX_CPU branch, so a correctly classified CPU-only Linux box fell through
to the "Unknown (...)" fallback in diagnostics and logs.
"""
from __future__ import annotations

from squish.platform.detector import PlatformInfo, PlatformKind


def _linux_cpu_info(arch: str = "x86_64") -> PlatformInfo:
    return PlatformInfo(
        kind=PlatformKind.LINUX_CPU,
        os_name="linux",
        python_version="3.11",
        arch=arch,
        has_mlx=False,
        has_cuda=False,
        has_rocm=False,
        is_wsl=False,
        cuda_info=None,
        apple_chip=None,
        ram_gb=16.0,
    )


def test_linux_cpu_not_labeled_unknown():
    info = _linux_cpu_info()
    assert "Unknown" not in info.platform_name
    assert info.platform_name == "Linux CPU (x86_64)"


def test_linux_cpu_arm():
    assert _linux_cpu_info("aarch64").platform_name == "Linux CPU (aarch64)"
