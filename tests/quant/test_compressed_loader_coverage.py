"""Supplementary coverage for ``squish.quant.compressed_loader`` — the
``_available_ram_gb`` RAM probe and the ``discover_npy_dir_metadata`` entry
point, which the existing unit suite leaves uncovered. Host-agnostic: the
Darwin path is driven by monkeypatching ``platform.system`` + ``vm_stat``.
"""

from __future__ import annotations

import json
import subprocess

import pytest

from squish.quant import compressed_loader as cl


# ── _available_ram_gb ────────────────────────────────────────────────────────


def test_available_ram_inf_off_darwin(monkeypatch):
    monkeypatch.setattr(cl._platform, "system", lambda: "Linux")
    assert cl._available_ram_gb() == float("inf")


def test_available_ram_parses_vm_stat(monkeypatch):
    monkeypatch.setattr(cl._platform, "system", lambda: "Darwin")
    vm_stat = (
        "Mach Virtual Memory Statistics:\n"
        "Pages free:                               100000.\n"
        "Pages inactive:                           200000.\n"
    )
    monkeypatch.setattr(subprocess, "check_output", lambda *a, **k: vm_stat)
    gb = cl._available_ram_gb()
    # (100000 + 200000) * 16384 / 1e9 ≈ 4.9152
    assert gb == pytest.approx((300000 * 16384) / 1e9)


def test_available_ram_inf_when_vm_stat_missing(monkeypatch):
    monkeypatch.setattr(cl._platform, "system", lambda: "Darwin")

    def _boom(*a, **k):
        raise FileNotFoundError("vm_stat")

    monkeypatch.setattr(subprocess, "check_output", _boom)
    assert cl._available_ram_gb() == float("inf")


# ── discover_npy_dir_metadata ────────────────────────────────────────────────


def test_discover_metadata_missing_manifest(tmp_path):
    with pytest.raises(FileNotFoundError, match="manifest.json not found"):
        cl.discover_npy_dir_metadata(tmp_path)


def test_discover_metadata_missing_tensors_dir(tmp_path):
    (tmp_path / "manifest.json").write_text("{}")
    with pytest.raises(FileNotFoundError, match="tensors/ directory not found"):
        cl.discover_npy_dir_metadata(tmp_path)


def test_discover_metadata_success(tmp_path):
    manifest = {"model.layers.0.mlp.down_proj": "model.layers.0.mlp.down_proj"}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    tensors = tmp_path / "tensors"
    tensors.mkdir()
    (tensors / "model.layers.0.mlp.down_proj__q4.npy").write_bytes(b"\x00")

    tensor_dir, base_keys, safe_to_original = cl.discover_npy_dir_metadata(tmp_path)
    assert tensor_dir == tensors
    assert base_keys == ["model.layers.0.mlp.down_proj"]
    assert safe_to_original == {"model.layers.0.mlp.down_proj": "model.layers.0.mlp.down_proj"}
