"""Cover _ActivationHook.__call__'s mlx→numpy conversion via a mocked mlx.core."""

import sys
import types

import numpy as np

from squish.quant import awq


def test_activation_hook_mlx_to_numpy(monkeypatch):
    fake_mx = types.ModuleType("mlx.core")
    fake_mx.float32 = "f32"
    pkg = types.ModuleType("mlx")
    pkg.core = fake_mx
    monkeypatch.setitem(sys.modules, "mlx", pkg)
    monkeypatch.setitem(sys.modules, "mlx.core", fake_mx)
    hook = awq._ActivationHook()
    x = types.SimpleNamespace(astype=lambda _dt: np.ones((2, 4), np.float32))
    hook(None, (x,), None)
    # line 245 (mlx → numpy conversion) ran without falling to the except path
    assert hook.channel_sum is not None and hook.channel_count >= 1
