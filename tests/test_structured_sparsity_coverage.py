"""Supplementary coverage for ``squish.experimental.structured_sparsity`` — the
MLX ``apply_mask`` path, which the existing suite leaves uncovered. A fake
``mlx.core`` is injected so the branch runs on both macOS and Linux CI.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from squish.experimental.structured_sparsity import StructuredFfnSparsity


class _FakeTensor:
    """Non-ndarray tensor that mimics the MLX ``shape`` + ``*`` contract."""

    shape = (3,)

    def __mul__(self, other):
        return ("masked", other)


def _sparsity():
    return StructuredFfnSparsity({0: np.array([1.0, 0.0, 1.0], dtype=np.float32)})


def _install_fake_mlx(monkeypatch):
    """Inject a fake ``mlx`` package + ``mlx.core``. Both are needed because
    ``import mlx.core as mx`` binds ``mx`` via the package's ``core`` attribute,
    so overriding only ``sys.modules['mlx.core']`` is bypassed when a sibling
    test has already imported the real mlx."""
    fake_mx = types.ModuleType("mlx.core")
    fake_mx.array = lambda m: ("mlxarr", m)
    fake_pkg = types.ModuleType("mlx")
    fake_pkg.core = fake_mx
    monkeypatch.setitem(sys.modules, "mlx", fake_pkg)
    monkeypatch.setitem(sys.modules, "mlx.core", fake_mx)
    return fake_mx


def test_apply_mask_no_mask_returns_unchanged():
    sp = _sparsity()
    sentinel = object()
    assert sp.apply_mask(99, sentinel) is sentinel  # no mask for layer 99


def test_apply_mask_numpy_path():
    sp = _sparsity()
    out = sp.apply_mask(0, np.ones(3, dtype=np.float32))
    assert out.tolist() == [1.0, 0.0, 1.0]


def test_apply_mask_mlx_path(monkeypatch):
    _install_fake_mlx(monkeypatch)
    out = _sparsity().apply_mask(0, _FakeTensor())
    assert out[0] == "masked" and out[1][0] == "mlxarr"


def test_apply_mask_mlx_import_error_returns_unchanged(monkeypatch):
    monkeypatch.setitem(sys.modules, "mlx", None)
    monkeypatch.setitem(sys.modules, "mlx.core", None)  # → ImportError
    t = _FakeTensor()
    assert _sparsity().apply_mask(0, t) is t


def test_apply_mask_unknown_type_returns_unchanged(monkeypatch):
    _install_fake_mlx(monkeypatch)
    # int has no ``shape`` → hasattr False → returned unchanged.
    assert _sparsity().apply_mask(0, 42) == 42
