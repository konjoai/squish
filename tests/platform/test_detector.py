"""Tests for UnifiedPlatformDetector — every probe + classification branch.

All platform/hardware probes are monkeypatched (or fed fakes), so the tests are
fully host-independent: nothing depends on the runner being Linux or macOS.
"""
from __future__ import annotations

import ctypes
import ctypes.util  # imported up-front so the darwin RAM test doesn't re-import
import sys          # it under a patched sys.platform (which pulls in macholib)
import types

import pytest

from squish.platform import detector as d
from squish.platform.detector import (
    CUDAInfo,
    PlatformInfo,
    PlatformKind,
    UnifiedPlatformDetector,
    detect_platform,
)

UPD = UnifiedPlatformDetector


def _patch_probes(monkeypatch, *, mlx=False, cuda=(False, None), rocm=False,
                  wsl=False, chip="", ram=0.0):
    monkeypatch.setattr(UPD, "_check_mlx", staticmethod(lambda: mlx))
    monkeypatch.setattr(UPD, "_check_cuda", staticmethod(lambda: cuda))
    monkeypatch.setattr(UPD, "_check_rocm", staticmethod(lambda: rocm))
    monkeypatch.setattr(UPD, "_check_wsl", staticmethod(lambda: wsl))
    monkeypatch.setattr(UPD, "_read_apple_chip", staticmethod(lambda: chip))
    monkeypatch.setattr(UPD, "_read_ram_gb", staticmethod(lambda: ram))


class TestClassification:
    def test_apple_silicon(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "darwin")
        _patch_probes(monkeypatch, mlx=True, chip="Apple M3 Pro", ram=36.0)
        info = UPD().detect()
        assert info.kind == PlatformKind.MACOS_APPLE_SILICON
        assert info.is_apple_silicon and info.apple_chip == "Apple M3 Pro"
        assert info.ram_gb == 36.0

    def test_wsl(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "linux")
        _patch_probes(monkeypatch, wsl=True)
        assert UPD().detect().kind == PlatformKind.WINDOWS_WSL

    def test_linux_cuda(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "linux")
        info_cuda = CUDAInfo(1, "A100", 80.0, "8.0", True)
        _patch_probes(monkeypatch, cuda=(True, info_cuda))
        info = UPD().detect()
        assert info.kind == PlatformKind.LINUX_CUDA and info.is_cuda
        assert info.cuda_info.device_name == "A100"

    def test_linux_rocm(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "linux")
        _patch_probes(monkeypatch, rocm=True)
        assert UPD().detect().kind == PlatformKind.LINUX_ROCM

    def test_linux_cpu(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "linux")
        _patch_probes(monkeypatch)
        assert UPD().detect().kind == PlatformKind.LINUX_CPU

    def test_windows_native(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "win32")
        _patch_probes(monkeypatch)
        assert UPD().detect().kind == PlatformKind.WINDOWS_NATIVE

    def test_unknown(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "freebsd13")
        _patch_probes(monkeypatch)
        assert UPD().detect().kind == PlatformKind.UNKNOWN


class TestDetectLifecycle:
    def test_caching_and_stats(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "linux")
        _patch_probes(monkeypatch)
        det = UPD()
        first = det.detect()
        second = det.detect()
        assert first is second
        assert det.stats.detection_calls == 2 and det.stats.cache_hits == 1
        assert det.stats.last_detection_ms >= 0.0

    def test_reset_reruns(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "linux")
        _patch_probes(monkeypatch)
        det = UPD()
        det.detect()
        det.reset()
        det.detect()
        assert det.stats.detection_calls == 2 and det.stats.cache_hits == 0

    def test_repr_before_and_after(self, monkeypatch):
        det = UPD()
        assert "not yet detected" in repr(det)
        monkeypatch.setattr(sys, "platform", "linux")
        _patch_probes(monkeypatch, ram=8.0)
        det.detect()
        assert "kind='LINUX_CPU'" in repr(det)

    def test_detect_platform_singleton(self):
        info = detect_platform()
        assert isinstance(info, PlatformInfo)


class TestMlxProbe:
    def test_non_darwin_short_circuits(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "linux")
        assert UPD._check_mlx() is False

    def test_darwin_with_fake_mlx(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "darwin")
        fake = types.ModuleType("mlx.core")
        fake.array = lambda *a, **k: object()
        fake.int32 = "int32"
        monkeypatch.setitem(sys.modules, "mlx", types.ModuleType("mlx"))
        monkeypatch.setitem(sys.modules, "mlx.core", fake)
        assert UPD._check_mlx() is True

    def test_darwin_mlx_import_error(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "darwin")
        monkeypatch.setitem(sys.modules, "mlx.core", None)  # ImportError
        assert UPD._check_mlx() is False


def _fake_torch(monkeypatch, **attrs):
    mod = types.ModuleType("torch")
    for k, v in attrs.items():
        setattr(mod, k, v)
    monkeypatch.setitem(sys.modules, "torch", mod)
    return mod


class TestCudaProbe:
    def test_no_torch(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch", None)  # ImportError
        assert UPD._check_cuda() == (False, None)

    def test_cuda_available(self, monkeypatch):
        props = types.SimpleNamespace(total_memory=80_000_000_000, major=8, minor=0)
        cuda_ns = types.SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: 1,
            get_device_name=lambda i: "A100",
            get_device_properties=lambda i: props,
        )
        _fake_torch(monkeypatch, cuda=cuda_ns)
        ok, info = UPD._check_cuda()
        assert ok and info.device_name == "A100"
        assert info.total_memory_gb == 80.0 and info.compute_capability == "8.0"

    def test_cuda_not_available(self, monkeypatch):
        _fake_torch(monkeypatch, cuda=types.SimpleNamespace(is_available=lambda: False))
        assert UPD._check_cuda() == (False, None)


class TestRocmProbe:
    def test_no_torch(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch", None)
        assert UPD._check_rocm() is False

    def test_rocm_present(self, monkeypatch):
        _fake_torch(
            monkeypatch,
            version=types.SimpleNamespace(hip="6.0"),
            cuda=types.SimpleNamespace(is_available=lambda: True),
        )
        assert UPD._check_rocm() is True


class TestWslProbe:
    def test_microsoft_in_proc_version(self, monkeypatch, tmp_path):
        import builtins
        real_open = builtins.open

        def fake_open(path, *a, **k):
            if str(path) == "/proc/version":
                return real_open(tmp_path / "v", "r")
            return real_open(path, *a, **k)

        (tmp_path / "v").write_text("Linux ... microsoft-standard-WSL2 ...")
        monkeypatch.setattr(builtins, "open", fake_open)
        assert UPD._check_wsl() is True

    def test_no_proc_version_falls_back_to_env(self, monkeypatch):
        import builtins

        def fake_open(path, *a, **k):
            if str(path) == "/proc/version":
                raise OSError("no /proc")
            return builtins.open(path, *a, **k)

        monkeypatch.setattr(builtins, "open", fake_open)
        monkeypatch.setenv("WSL_DISTRO_NAME", "Ubuntu")
        assert UPD._check_wsl() is True


class TestAppleChipProbe:
    def test_sysctl_success(self, monkeypatch):
        import subprocess
        monkeypatch.setattr(
            subprocess, "run",
            lambda *a, **k: types.SimpleNamespace(returncode=0, stdout="Apple M2 Max\n"),
        )
        assert UPD._read_apple_chip() == "Apple M2 Max"

    def test_sysctl_failure(self, monkeypatch):
        import subprocess

        def _raise(*a, **k):
            raise OSError("no sysctl")

        monkeypatch.setattr(subprocess, "run", _raise)
        assert UPD._read_apple_chip() == ""

    def test_sysctl_nonzero_returncode(self, monkeypatch):
        import subprocess
        monkeypatch.setattr(
            subprocess, "run",
            lambda *a, **k: types.SimpleNamespace(returncode=1, stdout=""),
        )
        assert UPD._read_apple_chip() == ""  # falls through to empty


class TestRamProbe:
    def test_linux_meminfo(self, monkeypatch, tmp_path):
        import builtins
        real_open = builtins.open
        # A non-MemTotal line first → exercises the loop's skip branch.
        (tmp_path / "meminfo").write_text("MemFree: 100 kB\nMemTotal:       16000000 kB\n")

        def fake_open(path, *a, **k):
            if str(path) == "/proc/meminfo":
                return real_open(tmp_path / "meminfo", "r")
            return real_open(path, *a, **k)

        monkeypatch.setattr(sys, "platform", "linux")
        monkeypatch.setattr(builtins, "open", fake_open)
        assert UPD._read_ram_gb() == pytest.approx(16.0, abs=0.1)

    def test_failure_returns_zero(self, monkeypatch):
        import builtins

        def fake_open(path, *a, **k):
            if str(path) == "/proc/meminfo":
                raise OSError("nope")
            return builtins.open(path, *a, **k)

        monkeypatch.setattr(sys, "platform", "linux")
        monkeypatch.setattr(builtins, "open", fake_open)
        assert UPD._read_ram_gb() == 0.0

    def test_darwin_sysctl_path(self, monkeypatch):
        # Exercise the darwin ctypes branch host-independently with a fake libc
        # whose sysctlbyname writes a known hw.memsize.
        monkeypatch.setattr(sys, "platform", "darwin")

        def _fake_cdll(_name):
            def _sysctlbyname(name, oldp, oldlenp, newp, newlen):
                ctypes.cast(oldp, ctypes.POINTER(ctypes.c_uint64))[0] = 32_000_000_000
                return 0
            return types.SimpleNamespace(sysctlbyname=_sysctlbyname)

        monkeypatch.setattr(ctypes, "CDLL", _fake_cdll)
        monkeypatch.setattr(ctypes.util, "find_library", lambda n: "c")
        assert UPD._read_ram_gb() == pytest.approx(32.0, abs=0.1)

    def test_non_posix_platform_returns_zero(self, monkeypatch):
        # Neither darwin nor linux → both branches skipped → 0.0.
        monkeypatch.setattr(sys, "platform", "win32")
        assert UPD._read_ram_gb() == 0.0

    def test_linux_meminfo_without_memtotal(self, monkeypatch, tmp_path):
        import builtins
        real_open = builtins.open
        (tmp_path / "meminfo").write_text("MemFree: 100 kB\nBuffers: 5 kB\n")

        def fake_open(path, *a, **k):
            if str(path) == "/proc/meminfo":
                return real_open(tmp_path / "meminfo", "r")
            return real_open(path, *a, **k)

        monkeypatch.setattr(sys, "platform", "linux")
        monkeypatch.setattr(builtins, "open", fake_open)
        assert UPD._read_ram_gb() == 0.0  # loop exhausts without MemTotal


class TestProperties:
    def _info(self, **kw):
        base = dict(
            kind=PlatformKind.LINUX_CPU, os_name="linux", python_version="3.11",
            arch="x86_64", has_mlx=False, has_cuda=False, has_rocm=False,
            is_wsl=False, cuda_info=None, apple_chip="", ram_gb=8.0,
        )
        base.update(kw)
        return PlatformInfo(**base)

    def test_name_lowercases_kind(self):
        assert self._info().name == "linux_cpu"

    def test_platform_name_apple(self):
        i = self._info(kind=PlatformKind.MACOS_APPLE_SILICON, os_name="darwin",
                       has_mlx=True, apple_chip="Apple M3")
        assert i.platform_name == "Apple Silicon (Apple M3)"

    def test_platform_name_apple_no_chip(self):
        i = self._info(kind=PlatformKind.MACOS_APPLE_SILICON, os_name="darwin", has_mlx=True)
        assert i.platform_name == "Apple Silicon (Apple Silicon)"

    def test_platform_name_cuda(self):
        i = self._info(has_cuda=True, cuda_info=CUDAInfo(1, "A100", 80.0, "8.0", True))
        assert i.platform_name == "Linux CUDA (A100)"

    def test_platform_name_cuda_no_info(self):
        i = self._info(has_cuda=True, cuda_info=None)
        assert i.platform_name == "Linux CUDA (CUDA)"

    def test_platform_name_rocm(self):
        assert self._info(has_rocm=True).platform_name == "Linux ROCm (AMD)"

    def test_platform_name_wsl(self):
        assert self._info(is_wsl=True).platform_name == "Windows (WSL2)"

    def test_platform_name_windows_native(self):
        assert self._info(os_name="win32").platform_name == "Windows (native)"

    def test_platform_name_linux_cpu(self):
        assert self._info(arch="aarch64").platform_name == "Linux CPU (aarch64)"

    def test_platform_name_unknown(self):
        assert "Unknown" in self._info(os_name="freebsd").platform_name

    def test_is_cuda_alias(self):
        assert self._info(has_cuda=True).is_cuda is True

    def test_cache_hit_rate(self):
        from squish.platform.detector import PlatformDetectorStats
        assert PlatformDetectorStats().cache_hit_rate == 0.0
        s = PlatformDetectorStats(detection_calls=4, cache_hits=3)
        assert s.cache_hit_rate == 0.75
