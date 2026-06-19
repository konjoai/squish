"""Behavioral coverage for ``squish/__init__.py`` — dist_version fallback, the
lazy __getattr__ import path, and the vendored-squish_quant auto-installer
(exercised via a fresh module copy since it is deleted after import).

Pure-Python; platform / subprocess / metadata are monkeypatched.
"""
from __future__ import annotations

import importlib.metadata
import importlib.util

import pytest

import squish


# ── dist_version ────────────────────────────────────────────────────────────


def test_dist_version_success(monkeypatch):
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "9.99.9")
    assert squish.dist_version() == "9.99.9"


def test_dist_version_fallback_to_pinned(monkeypatch):
    def _missing(name):
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "version", _missing)
    assert squish.dist_version() == squish.__version__  # 81-82 fallback


# ── lazy __getattr__ ─────────────────────────────────────────────────────────


def test_getattr_unknown_attribute_raises():
    with pytest.raises(AttributeError):
        squish.this_name_is_not_registered_anywhere  # noqa: B018


def test_getattr_import_failure_becomes_attribute_error(monkeypatch):
    monkeypatch.setitem(squish._LAZY_IMPORTS, "_probe_bad", "squish._no_such_module_xyz")
    monkeypatch.setattr(squish, "_lazy_cache", dict(squish._lazy_cache))
    with pytest.raises(AttributeError, match="has no attribute"):
        squish._probe_bad  # noqa: B018  (430-431)


def test_getattr_lazy_load_and_cache(monkeypatch):
    monkeypatch.setattr(squish, "_lazy_cache", dict(squish._lazy_cache))
    squish._lazy_cache.pop("CatalogEntry", None)  # force a fresh lazy load
    first = squish.CatalogEntry  # lazy import + cache
    assert first is squish.CatalogEntry  # second access served from cache (414-415)


def test_getattr_catalog_alias(monkeypatch):
    monkeypatch.setattr(squish, "_lazy_cache", dict(squish._lazy_cache))
    squish._lazy_cache.pop("pull_model", None)
    fn = squish.pull_model  # aliased to squish.catalog.pull (423-427)
    assert callable(fn) and fn is squish.pull_model


# ── _install_vendored_squish_quant (fresh module copy) ──────────────────────


def _exec_fresh_init(monkeypatch, *, system="Darwin", machine="arm64",
                     squish_quant_importable=False, wheels=None, run=None):
    import builtins
    import glob
    import platform
    import subprocess
    import sys

    monkeypatch.setattr(platform, "system", lambda: system)
    monkeypatch.setattr(platform, "machine", lambda: machine)
    monkeypatch.setattr(glob, "glob", lambda pat: list(wheels) if wheels is not None else [])
    if run is not None:
        monkeypatch.setattr(subprocess, "run", run)

    if squish_quant_importable:
        import types
        monkeypatch.setitem(sys.modules, "squish_quant", types.ModuleType("squish_quant"))
    else:
        # Force `import squish_quant` to fail inside the installer.
        monkeypatch.setitem(sys.modules, "squish_quant", None)

    spec = importlib.util.spec_from_file_location("squish._init_probe", squish.__file__)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # runs _install_vendored_squish_quant() during exec
    return mod


def test_install_vendored_non_darwin_returns(monkeypatch):
    _exec_fresh_init(monkeypatch, system="Linux")  # early return (30) — no raise


def test_install_vendored_already_importable(monkeypatch):
    _exec_fresh_init(monkeypatch, squish_quant_importable=True)  # return at 34


def test_install_vendored_no_wheels(monkeypatch):
    _exec_fresh_init(monkeypatch, wheels=[])  # no wheel found → return (46)


def test_install_vendored_runs_pip(monkeypatch):
    calls = []
    _exec_fresh_init(monkeypatch, wheels=["/v/squish_quant-0.1.0.whl"],
                     run=lambda *a, **k: calls.append(a))
    assert calls  # subprocess.run invoked with the vendored wheel


def test_install_vendored_subprocess_error_swallowed(monkeypatch):
    def _boom(*a, **k):
        raise OSError("pip unavailable")

    # subprocess failure must be swallowed (55-56) — exec must not raise.
    _exec_fresh_init(monkeypatch, wheels=["/v/squish_quant-0.1.0.whl"], run=_boom)
