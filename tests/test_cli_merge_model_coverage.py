"""Coverage for squish.cli cmd_merge_model + _find_adapter_safetensors —
offline DARE+TIES multi-adapter merge. Real safetensors adapters; host-agnostic.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pytest
from safetensors.numpy import save_file

from squish import cli


def _ns(output_path, adapters, method="dare-ties", base_model="qwen3:8b"):
    return argparse.Namespace(
        output_path=str(output_path),
        adapters=adapters,
        method=method,
        base_model=base_model,
    )


def _adapter(tmp_path, name, weights):
    d = tmp_path / name
    d.mkdir()
    save_file(weights, str(d / "adapter_model.safetensors"))
    return d


# ── _find_adapter_safetensors ────────────────────────────────────────────────


def test_find_adapter_file_passthrough(tmp_path):
    f = tmp_path / "adapter_model.safetensors"
    save_file({"w": np.ones(2, np.float32)}, str(f))
    assert cli._find_adapter_safetensors(f) == f


def test_find_adapter_in_dir(tmp_path):
    d = _adapter(tmp_path, "a", {"w": np.ones(2, np.float32)})
    assert cli._find_adapter_safetensors(d).name == "adapter_model.safetensors"


def test_find_adapter_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        cli._find_adapter_safetensors(tmp_path)


# ── cmd_merge_model guards ───────────────────────────────────────────────────


def test_merge_invalid_spec(tmp_path):
    with pytest.raises(SystemExit):
        cli.cmd_merge_model(_ns(tmp_path / "out", ["no-colon-spec"]))


def test_merge_adapter_path_missing(tmp_path):
    with pytest.raises(SystemExit):
        cli.cmd_merge_model(_ns(tmp_path / "out", [f"legal:{tmp_path / 'absent'}"]))


def test_merge_safetensors_missing(tmp_path, monkeypatch):
    a = _adapter(tmp_path, "legal", {"w": np.ones(2, np.float32)})
    monkeypatch.setitem(sys.modules, "safetensors.numpy", None)  # → ImportError → _die
    with pytest.raises(SystemExit):
        cli.cmd_merge_model(_ns(tmp_path / "out", [f"legal:{a}"]))


def test_merge_load_failure(tmp_path):
    # adapter dir exists but has no .safetensors → _find raises → _die
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(SystemExit):
        cli.cmd_merge_model(_ns(tmp_path / "out", [f"legal:{empty}"]))


# ── merge methods ────────────────────────────────────────────────────────────


def test_merge_dare_ties_two_adapters(tmp_path, capsys):
    a = _adapter(tmp_path, "legal", {"w": np.array([1.0, -1.0, 2.0], np.float32)})
    b = _adapter(tmp_path, "code", {"w": np.array([1.0, 1.0, -2.0], np.float32)})
    out = tmp_path / "out"
    cli.cmd_merge_model(_ns(out, [f"legal:{a}", f"code:{b}"], method="dare-ties"))
    assert (out / "adapter_model.safetensors").exists()
    assert "Sign conflict rate" in capsys.readouterr().out


def test_merge_ties_two_adapters(tmp_path, capsys):
    a = _adapter(tmp_path, "legal", {"w": np.array([1.0, -1.0], np.float32)})
    b = _adapter(tmp_path, "code", {"w": np.array([1.0, 1.0], np.float32)})
    cli.cmd_merge_model(_ns(tmp_path / "out", [f"legal:{a}", f"code:{b}"], method="ties"))
    assert "Merged adapter saved" in capsys.readouterr().out


def test_merge_dare_single_adapter(tmp_path, capsys):
    a = _adapter(tmp_path, "legal", {"w": np.ones(4, np.float32)})
    # single adapter → len(deltas)==1 → simple-average else branch
    cli.cmd_merge_model(_ns(tmp_path / "out", [f"legal:{a}"], method="dare"))
    assert "Merged adapter saved" in capsys.readouterr().out
