"""Cover _restore_type's mlx.array branch via a mocked mlx.core (host-agnostic)."""

import sys
import types

import numpy as np

from squish.integrations import hf


def test_restore_type_mlx_branch(monkeypatch):
    fake_mx = types.ModuleType("mlx.core")
    fake_mx.array = lambda a: ("MX", a)
    pkg = types.ModuleType("mlx")
    pkg.core = fake_mx
    monkeypatch.setitem(sys.modules, "mlx", pkg)
    monkeypatch.setitem(sys.modules, "mlx.core", fake_mx)
    # ref whose class __module__ contains "mlx" → routes to the mlx.array branch
    ref = type("Arr", (), {"__module__": "mlx.core"})()
    out = hf._restore_type(np.zeros(3, np.float32), ref)
    assert out[0] == "MX"
