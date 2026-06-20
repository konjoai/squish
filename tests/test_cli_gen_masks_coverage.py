"""Coverage for squish.cli cmd_gen_masks — structured FFN sparsity mask
generation. mlx / mlx_lm and the model are faked (numpy-backed) so the
calibration + mask-computation flow runs host-agnostically.
"""

from __future__ import annotations

import argparse
import sys
import types

import numpy as np
import pytest

from squish import cli


def _ns(model, tmp_out=None, **kw):
    kw.setdefault("samples", 12)
    kw.setdefault("activation_threshold", 0.05)
    kw.setdefault("keep_threshold", 0.1)
    kw.setdefault("output", str(tmp_out) if tmp_out else "")
    return argparse.Namespace(model=str(model), **kw)


def _fake_model(n_layers=2, hidden=8, *, with_layers=True):
    rng = np.random.default_rng(0)

    class MLP:
        flag = "delegated"  # accessed via the wrapper's __getattr__ delegate

        def __call__(self, x):
            return rng.standard_normal((1, 3, hidden)).astype(np.float32)

    class Layer:
        def __init__(self):
            self.mlp = MLP()

    layers = [Layer() for _ in range(n_layers)]

    class Model:
        def __init__(self):
            if with_layers:
                self.layers = layers

        def __call__(self, inp):
            for layer in self.layers:
                _ = layer.mlp.flag  # missing on _CaptureMLP → __getattr__ delegate
                layer.mlp(None)  # triggers the _CaptureMLP wrapper → records output
            return None

    return Model()


class _Tok:
    def __init__(self, mode="ok"):
        self.mode = mode

    def encode(self, prompt):
        if self.mode == "empty":
            return []
        if "France" in prompt:
            return []  # empty-toks → continue
        if "transformer" in prompt:
            raise ValueError("tokenize failed")  # calibration except
        return [1, 2, 3]


def _install(monkeypatch, model, tok):
    core = types.ModuleType("mlx.core")
    core.array = lambda x: x
    core.eval = lambda *a, **k: None
    pkg = types.ModuleType("mlx")
    pkg.core = core
    mlx_lm = types.ModuleType("mlx_lm")
    mlx_lm.load = lambda path: (model, tok)
    monkeypatch.setitem(sys.modules, "mlx", pkg)
    monkeypatch.setitem(sys.modules, "mlx.core", core)
    monkeypatch.setitem(sys.modules, "mlx_lm", mlx_lm)


# ── guard branches ───────────────────────────────────────────────────────────


def test_gen_masks_mlx_missing(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "mlx", None)
    monkeypatch.setitem(sys.modules, "mlx.core", None)
    with pytest.raises(SystemExit):
        cli.cmd_gen_masks(_ns(tmp_path))


def test_gen_masks_alias_not_found(tmp_path, monkeypatch):
    _install(monkeypatch, _fake_model(), _Tok())
    import squish.serving.local_model_scanner as lms

    monkeypatch.setattr(
        lms, "LocalModelScanner", lambda: types.SimpleNamespace(find_all=lambda: [])
    )
    with pytest.raises(SystemExit):
        cli.cmd_gen_masks(_ns("ghost-alias"))


def test_gen_masks_scanner_error(monkeypatch):
    _install(monkeypatch, _fake_model(), _Tok())
    import squish.serving.local_model_scanner as lms

    def _boom():
        raise OSError("scan failed")

    monkeypatch.setattr(lms, "LocalModelScanner", lambda: types.SimpleNamespace(find_all=_boom))
    with pytest.raises(SystemExit):
        cli.cmd_gen_masks(_ns("ghost-alias"))


def test_gen_masks_dir_not_found_via_alias(tmp_path, monkeypatch):
    _install(monkeypatch, _fake_model(), _Tok())
    import squish.serving.local_model_scanner as lms

    ghost = tmp_path / "ghost" / "model"  # parent "ghost" does not exist
    cand = types.SimpleNamespace(name="x", path=ghost)
    monkeypatch.setattr(
        lms,
        "LocalModelScanner",
        lambda: types.SimpleNamespace(find_all=lambda: [cand]),
    )
    with pytest.raises(SystemExit):
        cli.cmd_gen_masks(_ns("x"))


def test_gen_masks_alias_resolves_to_dir(tmp_path, monkeypatch, capsys):
    _install(monkeypatch, _fake_model(n_layers=1), _Tok("ok"))
    import squish.serving.local_model_scanner as lms

    model_dir = tmp_path / "qwen-compressed"
    model_dir.mkdir()
    cand = types.SimpleNamespace(name="qwen3:8b", path=model_dir)  # path IS a dir
    monkeypatch.setattr(
        lms,
        "LocalModelScanner",
        lambda: types.SimpleNamespace(find_all=lambda: [cand]),
    )
    cli.cmd_gen_masks(_ns("qwen3:8b", samples=2))
    assert "Saved" in capsys.readouterr().out


def test_gen_masks_no_layers(tmp_path, monkeypatch):
    _install(monkeypatch, _fake_model(with_layers=False), _Tok())
    with pytest.raises(SystemExit):
        cli.cmd_gen_masks(_ns(tmp_path))


# ── full calibration + mask computation ──────────────────────────────────────


def test_gen_masks_success(tmp_path, monkeypatch, capsys):
    out = tmp_path / "sparse_masks.npz"
    _install(monkeypatch, _fake_model(n_layers=2), _Tok("ok"))
    cli.cmd_gen_masks(_ns(tmp_path, tmp_out=out))
    assert out.exists()
    loaded = np.load(out)
    assert "layer_0" in loaded and "layer_1" in loaded
    o = capsys.readouterr().out
    assert "Saved 2 layer masks" in o and "prompts done" in o  # %10 progress


def test_gen_masks_all_empty_calibration(tmp_path, monkeypatch, capsys):
    # every prompt tokenizes empty → no forward passes → empty hooks → sparsity 0
    _install(monkeypatch, _fake_model(n_layers=2), _Tok("empty"))
    cli.cmd_gen_masks(_ns(tmp_path, samples=3))
    out = capsys.readouterr().out
    assert "Mean sparsity" in out


def test_gen_masks_default_output_path(tmp_path, monkeypatch, capsys):
    # output="" → defaults to comp_dir/sparse_masks.npz
    _install(monkeypatch, _fake_model(n_layers=1), _Tok("ok"))
    cli.cmd_gen_masks(_ns(tmp_path, samples=2))
    assert (tmp_path / "sparse_masks.npz").exists()
