"""Tests for ChipDetector — Apple-Silicon generation detection + tuning helpers.

The `_override` injection point and monkeypatched platform/subprocess calls keep
every test host-independent (no dependence on the runner being Apple Silicon).
Lives in tests/ root (not tests/hardware, which is in norecursedirs) so the
default suite and the coverage job collect it.
"""
from __future__ import annotations

import json

import pytest

from squish.hardware import chip_detector as cd
from squish.hardware.chip_detector import (
    CHIP_PROFILES,
    AppleChipGeneration,
    ChipDetector,
)


class TestParseAndProfiles:
    @pytest.mark.parametrize("chip,gen", [
        ("Apple M1", AppleChipGeneration.M1),
        ("Apple M2 Pro", AppleChipGeneration.M2),
        ("Apple M3 Max", AppleChipGeneration.M3),
        ("Apple M4", AppleChipGeneration.M4),
        ("Apple M5 Ultra", AppleChipGeneration.M5),
        ("Intel Core i9", AppleChipGeneration.UNKNOWN),
        ("Apple M15", AppleChipGeneration.UNKNOWN),  # \bM5\b must not match M15
    ])
    def test_generation_parse(self, chip, gen):
        assert ChipDetector(_override=chip).detect().generation == gen

    def test_unknown_uses_fallback_profile(self):
        prof = ChipDetector(_override="something else").detect()
        assert prof.generation == AppleChipGeneration.UNKNOWN
        assert prof.recommended_chunk_prefill == 512  # _UNKNOWN_PROFILE

    def test_detect_is_cached(self):
        det = ChipDetector(_override="Apple M3")
        first = det.detect()
        det._override = "Apple M5"  # ignored — result is cached
        assert det.detect() is first

    def test_repr(self):
        assert "gen=M3" in repr(ChipDetector(_override="Apple M3"))


class TestTuningHelpers:
    def _det(self, chip="Apple M3"):
        return ChipDetector(_override=chip)

    def test_recommended_model_bits_rules(self):
        d = self._det("Apple M3")
        assert d.get_recommended_model_bits(0.0, 7.0) == d.detect().recommended_model_bits
        assert d.get_recommended_model_bits(16.0, 7.0) == 2
        assert d.get_recommended_model_bits(20.0, 7.0) == 3
        assert d.get_recommended_model_bits(24.0, 7.0) == 4
        assert d.get_recommended_model_bits(64.0, 7.0) == d.detect().recommended_model_bits
        assert d.get_recommended_model_bits(16.0, 1.5) == 4  # small model → INT4

    def test_optimal_chunk_size_scales_down(self):
        d = self._det("Apple M5")  # base 2048
        assert d.get_optimal_chunk_size(4.0) == 2048
        small = d.get_optimal_chunk_size(32.0)
        assert 128 <= small < 2048

    def test_ttft_chunk_size(self):
        d = self._det("Apple M5")  # ttft base 256
        assert d.get_ttft_chunk_size(4.0) == 256
        assert d.get_ttft_chunk_size(32.0) == 128  # halved for large model

    def test_recommended_kv_bits(self):
        d = self._det("Apple M1")  # kv_bits 8
        assert d.get_recommended_kv_bits(32.0) == 8
        assert d.get_recommended_kv_bits(8.0) == 4   # low RAM → 4

    def test_should_enable_metal_dispatch(self):
        assert self._det("Apple M3").should_enable_metal_dispatch() is True

    def test_bandwidth_ratio_vs_m3(self):
        assert self._det("Apple M3").bandwidth_ratio_vs_m3() == pytest.approx(1.0)
        ratio = self._det("Apple M5").bandwidth_ratio_vs_m3()
        assert ratio == pytest.approx(
            153.0 / CHIP_PROFILES[AppleChipGeneration.M3].memory_bandwidth_gbps)


class TestReadChipString:
    def test_non_darwin_returns_empty(self, monkeypatch):
        monkeypatch.setattr(cd.platform, "system", lambda: "Linux")
        assert ChipDetector()._read_chip_string() == ""

    def test_disk_cache_hit(self, monkeypatch):
        monkeypatch.setattr(cd.platform, "system", lambda: "Darwin")
        monkeypatch.setattr(cd, "_load_disk_cache", lambda: "Apple M4 Pro")
        assert ChipDetector()._read_chip_string() == "Apple M4 Pro"

    def test_sysctl_success(self, monkeypatch):
        import types
        monkeypatch.setattr(cd.platform, "system", lambda: "Darwin")
        monkeypatch.setattr(cd, "_load_disk_cache", lambda: None)
        monkeypatch.setattr(cd, "_save_disk_cache", lambda s: None)
        monkeypatch.setattr(cd.subprocess, "run",
                            lambda *a, **k: types.SimpleNamespace(returncode=0, stdout="Apple M3 Pro\n"))
        assert ChipDetector()._read_chip_string() == "Apple M3 Pro"

    def test_system_profiler_fallback(self, monkeypatch):
        import types
        monkeypatch.setattr(cd.platform, "system", lambda: "Darwin")
        monkeypatch.setattr(cd, "_load_disk_cache", lambda: None)
        monkeypatch.setattr(cd, "_save_disk_cache", lambda s: None)

        def _run(cmd, *a, **k):
            if cmd[0] == "sysctl":
                raise FileNotFoundError("no sysctl")
            return types.SimpleNamespace(
                returncode=0, stdout="Model: Mac\n      Chip: Apple M4\n")

        monkeypatch.setattr(cd.subprocess, "run", _run)
        assert "Apple M4" in ChipDetector()._read_chip_string()

    def test_sysctl_nonzero_then_profiler(self, monkeypatch):
        # sysctl returns non-zero (empty) → falls through to system_profiler.
        import types
        monkeypatch.setattr(cd.platform, "system", lambda: "Darwin")
        monkeypatch.setattr(cd, "_load_disk_cache", lambda: None)
        monkeypatch.setattr(cd, "_save_disk_cache", lambda s: None)

        def _run(cmd, *a, **k):
            if cmd[0] == "sysctl":
                return types.SimpleNamespace(returncode=1, stdout="")
            return types.SimpleNamespace(returncode=0, stdout="Chip: Apple M5\n")

        monkeypatch.setattr(cd.subprocess, "run", _run)
        assert "Apple M5" in ChipDetector()._read_chip_string()

    def test_profiler_nonzero_returncode(self, monkeypatch):
        import types
        monkeypatch.setattr(cd.platform, "system", lambda: "Darwin")
        monkeypatch.setattr(cd, "_load_disk_cache", lambda: None)
        monkeypatch.setattr(cd, "_save_disk_cache", lambda s: None)
        monkeypatch.setattr(cd.subprocess, "run",
                            lambda cmd, *a, **k: types.SimpleNamespace(returncode=1, stdout=""))
        assert ChipDetector()._read_chip_string() == ""

    def test_profiler_no_chip_line(self, monkeypatch):
        import types
        monkeypatch.setattr(cd.platform, "system", lambda: "Darwin")
        monkeypatch.setattr(cd, "_load_disk_cache", lambda: None)
        monkeypatch.setattr(cd, "_save_disk_cache", lambda s: None)

        def _run(cmd, *a, **k):
            if cmd[0] == "sysctl":
                return types.SimpleNamespace(returncode=1, stdout="")
            return types.SimpleNamespace(returncode=0, stdout="Model: Mac\nMemory: 16 GB\n")

        monkeypatch.setattr(cd.subprocess, "run", _run)
        assert ChipDetector()._read_chip_string() == ""  # no Chip/Processor line

    def test_both_probes_fail(self, monkeypatch):
        monkeypatch.setattr(cd.platform, "system", lambda: "Darwin")
        monkeypatch.setattr(cd, "_load_disk_cache", lambda: None)
        saved = {}
        monkeypatch.setattr(cd, "_save_disk_cache", lambda s: saved.update(v=s))

        def _run(cmd, *a, **k):
            raise FileNotFoundError("nothing")

        monkeypatch.setattr(cd.subprocess, "run", _run)
        assert ChipDetector()._read_chip_string() == ""
        assert saved["v"] == ""  # empty result persisted


class TestDiskCache:
    def test_key_is_machine_plus_version(self, monkeypatch):
        monkeypatch.setattr(cd.platform, "machine", lambda: "arm64")
        monkeypatch.setattr(cd.platform, "version", lambda: "v1")
        assert cd._disk_cache_key() == "arm64v1"

    def test_save_then_load_roundtrip(self, monkeypatch, tmp_path):
        monkeypatch.setattr(cd, "_DISK_CACHE_PATH", tmp_path / "hw.json")
        monkeypatch.setattr(cd.platform, "machine", lambda: "arm64")
        monkeypatch.setattr(cd.platform, "version", lambda: "v1")
        cd._save_disk_cache("Apple M3")
        assert cd._load_disk_cache() == "Apple M3"

    def test_load_key_mismatch_returns_none(self, monkeypatch, tmp_path):
        p = tmp_path / "hw.json"
        p.write_text(json.dumps({"key": "stale", "chip_str": "Apple M2"}))
        monkeypatch.setattr(cd, "_DISK_CACHE_PATH", p)
        monkeypatch.setattr(cd.platform, "machine", lambda: "arm64")
        monkeypatch.setattr(cd.platform, "version", lambda: "v1")
        assert cd._load_disk_cache() is None

    def test_load_missing_file_returns_none(self, monkeypatch, tmp_path):
        monkeypatch.setattr(cd, "_DISK_CACHE_PATH", tmp_path / "absent.json")
        assert cd._load_disk_cache() is None

    def test_save_failure_is_swallowed(self, monkeypatch):
        # Unwritable path → OSError swallowed (best-effort, no raise).
        monkeypatch.setattr(cd, "_DISK_CACHE_PATH", cd.pathlib.Path("/proc/cannot/write.json"))
        cd._save_disk_cache("x")


class TestDetectRam:
    def test_non_darwin_zero(self, monkeypatch):
        monkeypatch.setattr(cd.platform, "system", lambda: "Linux")
        assert ChipDetector.detect_ram_gb() == 0.0

    def test_darwin_sysctl(self, monkeypatch):
        import types
        monkeypatch.setattr(cd.platform, "system", lambda: "Darwin")
        monkeypatch.setattr(cd.subprocess, "run",
                            lambda *a, **k: types.SimpleNamespace(returncode=0, stdout=str(32 * 1024**3)))
        assert ChipDetector.detect_ram_gb() == pytest.approx(32.0, abs=0.1)

    def test_darwin_failure_zero(self, monkeypatch):
        monkeypatch.setattr(cd.platform, "system", lambda: "Darwin")

        def _raise(*a, **k):
            raise FileNotFoundError("no sysctl")

        monkeypatch.setattr(cd.subprocess, "run", _raise)
        assert ChipDetector.detect_ram_gb() == 0.0

    def test_darwin_nonzero_returncode_zero(self, monkeypatch):
        import types
        monkeypatch.setattr(cd.platform, "system", lambda: "Darwin")
        monkeypatch.setattr(cd.subprocess, "run",
                            lambda *a, **k: types.SimpleNamespace(returncode=1, stdout=""))
        assert ChipDetector.detect_ram_gb() == 0.0
