"""Coverage for cli.py RAM/shard pre-flight helpers and the compress preflight.

All pure-Python (no MLX). Absent-dependency paths are forced via sys.modules so
they're deterministic on Linux *and* the macOS runners (where squish_quant /
safetensors may actually be importable).
"""

import subprocess
import sys
import types

import numpy as np
import pytest
from safetensors.numpy import save_file

from squish import cli


# ── _ram_available_gb ──────────────────────────────────────────────────────────


def test_ram_available_linux_branch():
    total, free = cli._ram_available_gb()
    assert total >= 0.0 and free >= 0.0  # /proc/meminfo path on the Linux runner


def test_ram_available_darwin_branch(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")

    class _Libc:
        def sysctlbyname(self, *_a):
            return 0

    import ctypes

    monkeypatch.setattr(ctypes, "CDLL", lambda *_a, **_k: _Libc())
    vm = (
        "Mach VM stats:\n"
        "no-colon header line\n"  # exercises the ":" not-in-line branch
        "Pages bad: notanint.\n"  # exercises the int() ValueError branch
        "Pages free: 100.\n"
        "Pages inactive: 50.\n"
        "Pages speculative: 10.\n"
    )
    monkeypatch.setattr(subprocess, "check_output", lambda *_a, **_k: vm)
    total, free = cli._ram_available_gb()
    assert free > 0.0  # parsed from the mocked vm_stat


def test_ram_available_swallows_errors(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    import ctypes

    def _boom(*_a, **_k):
        raise OSError("no libSystem")

    monkeypatch.setattr(ctypes, "CDLL", _boom)
    assert cli._ram_available_gb() == (0.0, 0.0)  # except branch → defaults


# ── _bf16_native_available ─────────────────────────────────────────────────────


def test_bf16_native_unavailable(monkeypatch):
    monkeypatch.setitem(sys.modules, "squish_quant", None)  # force ImportError
    assert cli._bf16_native_available() is False


def test_bf16_native_available(monkeypatch):
    fake = types.ModuleType("squish_quant")
    fake.quantize_int8_bf16 = lambda *a: None
    monkeypatch.setitem(sys.modules, "squish_quant", fake)
    assert cli._bf16_native_available() is True


# ── _max_tensor_gb_from_shards ─────────────────────────────────────────────────


def test_max_tensor_from_real_safetensors(tmp_path):
    p = tmp_path / "model.safetensors"
    save_file({"a": np.zeros((16, 16), np.float32), "b": np.zeros((16, 16), np.float32)}, str(p))
    assert cli._max_tensor_gb_from_shards([p]) > 0.0


def test_max_tensor_fallback_without_safetensors(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "safetensors", None)  # force ImportError
    p = tmp_path / "shard.bin"
    p.write_bytes(b"x" * 1024)
    assert cli._max_tensor_gb_from_shards([p]) > 0.0  # shard-size fallback


def test_max_tensor_safe_open_error_falls_back(monkeypatch, tmp_path):
    import safetensors

    def _raise(*_a, **_k):
        raise ValueError("corrupt header")

    monkeypatch.setattr(safetensors, "safe_open", _raise)
    p = tmp_path / "x.safetensors"
    p.write_bytes(b"x" * 2048)
    # safe_open raises a caught error → max_bytes stays 0 → shard-size fallback
    assert cli._max_tensor_gb_from_shards([p]) > 0.0


# ── _peak_ram_estimate_gb ──────────────────────────────────────────────────────


def test_peak_ram_shard_path(monkeypatch, tmp_path):
    save_file({"w": np.zeros((16, 16), np.float32)}, str(tmp_path / "m.safetensors"))
    monkeypatch.setattr(cli, "_bf16_native_available", lambda: False)
    size, peak, mx = cli._peak_ram_estimate_gb(tmp_path, run_awq=False)
    assert size >= 0.0 and peak > 0.0 and mx >= 0.0


def test_peak_ram_awq_path(tmp_path):
    save_file({"w": np.zeros((8, 8), np.float32)}, str(tmp_path / "m.safetensors"))
    size, peak, mx = cli._peak_ram_estimate_gb(tmp_path, run_awq=True)
    assert peak == pytest.approx(size * 2.0 + 2.0)


# ── _cmd_compress_inner pre-flight hard block ──────────────────────────────────


def _args():
    return types.SimpleNamespace(model="qwen3:7b")


def test_compress_preflight_hard_block_no_awq(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "_peak_ram_estimate_gb", lambda *a, **k: (10.0, 999.0, 5.0))
    monkeypatch.setattr(cli, "_ram_available_gb", lambda: (8.0, 4.0))
    monkeypatch.setattr(cli, "_bf16_native_available", lambda: False)
    with pytest.raises(SystemExit):
        cli._cmd_compress_inner(_args(), tmp_path, tmp_path, True, True, False)


def test_compress_preflight_hard_block_awq_suggests_no_awq(monkeypatch, tmp_path):
    # AWQ peak too big, but the no-AWQ re-estimate fits → "Try --no-awq" branch.
    calls = iter([(10.0, 999.0, 5.0), (10.0, 8.0, 5.0)])
    monkeypatch.setattr(cli, "_peak_ram_estimate_gb", lambda *a, **k: next(calls))
    monkeypatch.setattr(cli, "_ram_available_gb", lambda: (8.0, 4.0))
    with pytest.raises(SystemExit):
        cli._cmd_compress_inner(_args(), tmp_path, tmp_path, True, False, True)


def test_compress_preflight_hard_block_awq_no_fit(monkeypatch, tmp_path):
    # Both AWQ and no-AWQ peaks exceed total → multi-option branch.
    monkeypatch.setattr(cli, "_peak_ram_estimate_gb", lambda *a, **k: (10.0, 999.0, 5.0))
    monkeypatch.setattr(cli, "_ram_available_gb", lambda: (8.0, 4.0))
    with pytest.raises(SystemExit):
        cli._cmd_compress_inner(_args(), tmp_path, tmp_path, True, False, True)


def test_ram_available_other_platform_returns_zero(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")  # neither darwin nor linux
    assert cli._ram_available_gb() == (0.0, 0.0)
