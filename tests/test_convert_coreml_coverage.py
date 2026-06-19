"""Coverage for ``squish.experimental.convert_coreml`` — the CoreML export
pipeline. coremltools is optional; the real conversion path is already
``# pragma: no cover``. Everything else (numpy simulation, chunk planning,
appendix writing, checksums) is pure numpy/filesystem → host-agnostic.
"""

from __future__ import annotations

import importlib.util
import json
import struct
import sys
import types

import numpy as np

from squish.experimental import convert_coreml as cc


def test_import_succeeds_when_coremltools_present(monkeypatch):
    """Cover the import-success branch (_COREMLTOOLS_AVAILABLE = True) via a fresh
    module copy with a fake coremltools injected."""
    monkeypatch.setitem(sys.modules, "coremltools", types.ModuleType("coremltools"))
    spec = importlib.util.spec_from_file_location("convert_coreml_fresh", cc.__file__)
    fresh = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "convert_coreml_fresh", fresh)  # dataclass needs it
    spec.loader.exec_module(fresh)
    assert fresh._COREMLTOOLS_AVAILABLE is True


from squish.experimental.convert_coreml import (
    CoreMLChunk,
    CoreMLConversionConfig,
    CoreMLConverter,
    CoreMLPackage,
    _dir_checksum,
)


# ── config / package dataclasses ─────────────────────────────────────────────


def test_config_defaults():
    c = CoreMLConversionConfig()
    assert c.chunk_size_gb == 2.0 and c.quantization == "int4" and c.target_chip == "ane"


def test_package_chunk_count_and_manifest():
    chunk = CoreMLChunk(
        index=0,
        mlpackage_path=__import__("pathlib").Path("/tmp/x"),
        layer_start=0,
        layer_end=1,
        param_count=10,
        size_bytes=20,
        checksum="abc",
    )
    pkg = CoreMLPackage(chunks=[chunk], total_param_count=10, coremltools_used=False)
    assert pkg.chunk_count == 1
    m = pkg.manifest()
    assert m["chunk_count"] == 1 and m["chunks"][0]["checksum"] == "abc"
    assert m["header_bit"] == cc.SQUIZD_ANE_COREML_BIT


# ── convert (numpy simulation) ───────────────────────────────────────────────


def _weights():
    return {
        "model.layers.0.mlp.weight": np.ones((32, 32), np.float32),
        "model.layers.1.mlp.weight": np.ones((32, 32), np.float32),
    }


def test_convert_numpy_simulation(tmp_path):
    cfg = CoreMLConversionConfig(output_dir=str(tmp_path))
    pkg = CoreMLConverter(cfg).convert(_weights())
    assert pkg.coremltools_used is False
    assert pkg.chunk_count >= 1
    # simulation writes a manifest.json inside each chunk dir
    manifest = json.loads((pkg.chunks[0].mlpackage_path / "manifest.json").read_text())
    assert manifest["simulation"] is True


def test_convert_uses_coremltools_branch(tmp_path, monkeypatch):
    # Force the _COREMLTOOLS_AVAILABLE + target_chip=='ane' arm without running
    # the real (pragma'd) coremltools conversion: stub _convert_coremltools.
    monkeypatch.setattr(cc, "_COREMLTOOLS_AVAILABLE", True)
    sentinel = [
        CoreMLChunk(
            index=0,
            mlpackage_path=tmp_path,
            layer_start=0,
            layer_end=0,
            param_count=1,
            size_bytes=2,
            checksum="z",
        )
    ]
    monkeypatch.setattr(CoreMLConverter, "_convert_coremltools", lambda self, w, p: sentinel)
    pkg = CoreMLConverter(CoreMLConversionConfig(target_chip="ane")).convert(_weights())
    assert pkg.coremltools_used is True and pkg.chunks == sentinel


def test_convert_gpu_target_uses_simulation(tmp_path, monkeypatch):
    # coremltools available but target_chip != 'ane' → simulation arm.
    monkeypatch.setattr(cc, "_COREMLTOOLS_AVAILABLE", True)
    cfg = CoreMLConversionConfig(output_dir=str(tmp_path), target_chip="gpu")
    pkg = CoreMLConverter(cfg).convert(_weights())
    assert pkg.coremltools_used is False


def test_convert_default_output_dir_is_tempdir():
    # output_dir empty → a temp dir is created automatically.
    pkg = CoreMLConverter(CoreMLConversionConfig()).convert(_weights())
    assert pkg.chunks[0].mlpackage_path.exists()


# ── _infer_layer_count ───────────────────────────────────────────────────────


def test_infer_layer_count_from_names():
    assert CoreMLConverter._infer_layer_count({"model.layers.5.w": np.zeros(1)}) == 6


def test_infer_layer_count_defaults_to_one():
    assert CoreMLConverter._infer_layer_count({"weight": np.zeros(1)}) == 1


# ── _plan_chunks quantization branches + multi-chunk ─────────────────────────


def _plan(quant, chunk_gb=2.0, n_layers=2, total=2048):
    conv = CoreMLConverter(CoreMLConversionConfig(quantization=quant, chunk_size_gb=chunk_gb))
    return conv._plan_chunks({}, n_layers=n_layers, total_params=total)


def test_plan_chunks_quantization_byte_rates():
    assert _plan("int4")  # 0.5 B/param
    assert _plan("int8")  # 1.0 B/param
    assert _plan("fp16")  # 2.0 B/param (else)


def test_plan_chunks_splits_when_over_budget():
    # Tiny budget + 4 layers + many params → one layer per chunk → 4 chunks.
    plans = _plan("fp16", chunk_gb=1e-9, n_layers=4, total=4_000_000)
    assert len(plans) == 4
    assert plans[0]["layer_start"] == 0 and plans[-1]["layer_end"] == 3


# ── write_squizd_appendix ────────────────────────────────────────────────────


def test_write_squizd_appendix(tmp_path):
    f = tmp_path / "model.squizd"
    f.write_bytes(b"SQUIZDHEADER")
    pkg = CoreMLConverter(CoreMLConversionConfig(output_dir=str(tmp_path))).convert(_weights())
    n = CoreMLConverter().write_squizd_appendix(pkg, f)
    data = f.read_bytes()
    assert data[12:16] == cc.SQUIZD_APPENDIX_TAG  # tag after the original header
    payload_len = struct.unpack("<Q", data[16:24])[0]
    assert n == 12 + payload_len


# ── _dir_checksum ────────────────────────────────────────────────────────────


def test_dir_checksum_dir_file_and_missing(tmp_path):
    d = tmp_path / "d"
    d.mkdir()
    (d / "a.txt").write_text("x")
    dir_sum = _dir_checksum(d)

    f = tmp_path / "f.bin"
    f.write_bytes(b"hello")
    file_sum = _dir_checksum(f)

    missing_sum = _dir_checksum(tmp_path / "nope")
    assert len({dir_sum, file_sum, missing_sum}) == 3  # all distinct, no errors
