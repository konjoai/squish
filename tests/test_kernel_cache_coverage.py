"""Behavioral coverage for ``squish.serving.kernel_cache`` — MLX Metal kernel
cache helpers. Every Darwin / non-Darwin and MLX-present / MLX-absent branch is
driven via a monkeypatched ``platform.system`` and an injected fake ``mlx.core``
so the suite is host-agnostic (passes on both macOS and Linux CI).
"""

from __future__ import annotations

import sys
import types

import pytest

from squish.serving import kernel_cache as kc


def _fake_mlx(*, metal=None, version="0.22.0"):
    """A minimal fake ``mlx.core`` module."""
    m = types.ModuleType("mlx.core")
    m.__version__ = version
    if metal is not None:
        m.metal = metal
    m.int32 = "int32"
    m.array = lambda data, dtype=None: ("array", data)
    m.eval = lambda out: None
    return m


def _install_mlx(monkeypatch, core):
    """Install ``core`` as both the ``mlx`` package's ``.core`` attribute AND
    ``sys.modules['mlx.core']``. Both are required because ``import mlx.core as
    mx`` binds ``mx`` via the package attribute — overriding only
    ``sys.modules['mlx.core']`` is bypassed once a sibling test has imported the
    real mlx. ``core=None`` makes the import raise (ImportError path)."""
    if core is None:
        monkeypatch.setitem(sys.modules, "mlx", None)
        monkeypatch.setitem(sys.modules, "mlx.core", None)
        return
    pkg = types.ModuleType("mlx")
    pkg.core = core
    monkeypatch.setitem(sys.modules, "mlx", pkg)
    monkeypatch.setitem(sys.modules, "mlx.core", core)


def _force_darwin(monkeypatch):
    monkeypatch.setattr(kc.platform, "system", lambda: "Darwin")


def _force_linux(monkeypatch):
    monkeypatch.setattr(kc.platform, "system", lambda: "Linux")


# ── ensure_kernel_cache_dir ──────────────────────────────────────────────────


def test_ensure_kernel_cache_dir_explicit(tmp_path, monkeypatch):
    monkeypatch.delenv(kc._MLX_KERNEL_CACHE_ENV, raising=False)
    target = tmp_path / "kernels"
    out = kc.ensure_kernel_cache_dir(target)
    assert out == target and target.is_dir()
    assert kc.os.environ[kc._MLX_KERNEL_CACHE_ENV] == str(target)


def test_ensure_kernel_cache_dir_default(tmp_path, monkeypatch):
    monkeypatch.delenv(kc._MLX_KERNEL_CACHE_ENV, raising=False)
    default = tmp_path / "default_kernels"
    monkeypatch.setattr(kc, "_DEFAULT_KERNEL_CACHE_DIR", default)
    out = kc.ensure_kernel_cache_dir(None)
    assert out == default and default.is_dir()


# ── mlx_supports_kernel_cache ────────────────────────────────────────────────


def test_supports_false_off_darwin(monkeypatch):
    _force_linux(monkeypatch)
    assert kc.mlx_supports_kernel_cache() is False


def test_supports_true_when_api_present(monkeypatch):
    _force_darwin(monkeypatch)
    metal = types.SimpleNamespace(set_cache_limit=lambda n: None, save_kernel_cache=lambda p: None)
    _install_mlx(monkeypatch, _fake_mlx(metal=metal))
    assert kc.mlx_supports_kernel_cache() is True


def test_supports_false_when_api_absent(monkeypatch):
    _force_darwin(monkeypatch)
    metal = types.SimpleNamespace()  # no set_cache_limit / save_kernel_cache
    _install_mlx(monkeypatch, _fake_mlx(metal=metal))
    assert kc.mlx_supports_kernel_cache() is False


def test_supports_false_on_import_error(monkeypatch):
    _force_darwin(monkeypatch)
    _install_mlx(monkeypatch, None)  # → ImportError
    assert kc.mlx_supports_kernel_cache() is False


# ── run_warmup_pass ──────────────────────────────────────────────────────────


def test_warmup_no_model_returns_zero():
    assert kc.run_warmup_pass(model=None) == 0.0


def test_warmup_off_darwin_returns_zero(monkeypatch):
    _force_linux(monkeypatch)
    assert kc.run_warmup_pass(model=lambda ids: ids) == 0.0


def test_warmup_runs_forward_pass(monkeypatch):
    _force_darwin(monkeypatch)
    _install_mlx(monkeypatch, _fake_mlx())
    calls = []
    elapsed = kc.run_warmup_pass(model=lambda ids: calls.append(ids) or ids)
    assert elapsed >= 0.0 and len(calls) == 1


def test_warmup_swallows_forward_error(monkeypatch):
    _force_darwin(monkeypatch)
    _install_mlx(monkeypatch, _fake_mlx())

    def _boom(ids):
        raise RuntimeError("metal exploded")

    elapsed = kc.run_warmup_pass(model=_boom)
    assert elapsed >= 0.0  # exception logged, not raised


def test_warmup_import_error(monkeypatch):
    _force_darwin(monkeypatch)
    _install_mlx(monkeypatch, None)
    assert kc.run_warmup_pass(model=lambda ids: ids) >= 0.0


# ── metal_cache_info ─────────────────────────────────────────────────────────


def test_info_off_darwin_omits_system_keys(monkeypatch):
    _force_linux(monkeypatch)
    info = kc.metal_cache_info()
    assert info["env_var"] == kc._MLX_KERNEL_CACHE_ENV
    assert "system_metal_cache" not in info and "mlx_version" not in info


def test_info_on_darwin_reports_mlx_version(monkeypatch):
    _force_darwin(monkeypatch)
    _install_mlx(monkeypatch, _fake_mlx(version="0.99.0"))
    info = kc.metal_cache_info()
    assert info["mlx_version"] == "0.99.0"
    assert "system_metal_cache" in info


def test_info_on_darwin_mlx_not_installed(monkeypatch):
    _force_darwin(monkeypatch)
    _install_mlx(monkeypatch, None)
    info = kc.metal_cache_info()
    assert info["mlx_version"] == "not installed"
