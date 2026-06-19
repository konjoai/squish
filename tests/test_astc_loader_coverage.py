"""Behavioral coverage for ``squish.experimental.astc_loader`` — the ASTC weight
texture loader. ``squish.compress.astc_encoder`` is not installed in this
distribution, so the encoder symbols and the Metal bindings are replaced with
fakes via monkeypatch. Every Metal-available / simulation / fallback branch is
driven deterministically, so the suite is host-agnostic (macOS + Linux CI).
"""

from __future__ import annotations

import importlib.util
import sys
import types

import numpy as np
import pytest

from squish.experimental import astc_loader as astc


def test_import_succeeds_when_encoder_present(monkeypatch):
    """Cover the import-success branch (_ASTC_ENCODER_AVAILABLE = True) by loading
    a fresh copy of the module with a fake squish.compress.astc_encoder present."""
    pkg = types.ModuleType("squish.compress")
    enc = types.ModuleType("squish.compress.astc_encoder")
    for attr in ("ASTC_BLOCK_BYTES", "ASTC_BLOCK_X", "ASTC_BLOCK_Y"):
        setattr(enc, attr, 6)
    for attr in ("ASTCEncodeResult", "ASTCEncoderConfig", "ASTCEncoder"):
        setattr(enc, attr, object)
    monkeypatch.setitem(sys.modules, "squish.compress", pkg)
    monkeypatch.setitem(sys.modules, "squish.compress.astc_encoder", enc)

    spec = importlib.util.spec_from_file_location("astc_loader_fresh", astc.__file__)
    fresh = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "astc_loader_fresh", fresh)  # dataclass needs it
    spec.loader.exec_module(fresh)
    assert fresh._ASTC_ENCODER_AVAILABLE is True


# ── fakes ────────────────────────────────────────────────────────────────────


class FakeEncodeResult:
    original_shape = (8, 6)
    padded_shape = (12, 6)  # (rows, cols)
    scale_table = np.ones(2, dtype=np.float32)
    n_blocks = 2
    block_bytes = b"\x00" * 32


class _FakeEncCfg:
    def __init__(self, block_x, block_y):
        self.block_x, self.block_y = block_x, block_y


class _FakeEnc:
    def __init__(self, config, force_numpy_fallback=False):
        self.config = config

    def decode(self, result):
        return np.zeros(result.original_shape, dtype=np.float32)


@pytest.fixture
def encoder_fakes(monkeypatch):
    """Install fake encoder symbols so decode paths are exercisable."""
    monkeypatch.setattr(astc, "ASTC_BLOCK_X", 6)
    monkeypatch.setattr(astc, "ASTC_BLOCK_Y", 6)
    monkeypatch.setattr(astc, "ASTCEncoderConfig", _FakeEncCfg)
    monkeypatch.setattr(astc, "ASTCEncoder", _FakeEnc)


def _fake_metalcompute(monkeypatch, *, name="Apple M3", buffer_ok=True, device_ok=True):
    mod = types.ModuleType("metalcompute")

    class Device:
        def __init__(self):
            if not device_ok:
                raise RuntimeError("no device")

        def name(self):
            return name

        def buffer(self, data):
            if not buffer_ok:
                raise RuntimeError("buffer failed")
            return ("buf", len(data))

    mod.Device = Device
    monkeypatch.setitem(sys.modules, "metalcompute", mod)


# ── _probe_metal / is_metal_available ────────────────────────────────────────


def test_probe_cached(monkeypatch):
    monkeypatch.setattr(astc, "_METAL_AVAILABLE", True)
    assert astc.is_metal_available() is True


def test_probe_force_simulation(monkeypatch):
    monkeypatch.setattr(astc, "_METAL_AVAILABLE", None)
    monkeypatch.setenv("SQUISH_FORCE_METAL_SIMULATION", "1")
    assert astc._probe_metal() is False


def test_probe_force_available(monkeypatch):
    monkeypatch.setattr(astc, "_METAL_AVAILABLE", None)
    monkeypatch.delenv("SQUISH_FORCE_METAL_SIMULATION", raising=False)
    monkeypatch.setenv("SQUISH_FORCE_METAL_AVAILABLE", "1")
    assert astc._probe_metal() is True


def test_probe_via_metalcompute(monkeypatch):
    monkeypatch.setattr(astc, "_METAL_AVAILABLE", None)
    monkeypatch.delenv("SQUISH_FORCE_METAL_SIMULATION", raising=False)
    monkeypatch.delenv("SQUISH_FORCE_METAL_AVAILABLE", raising=False)
    _fake_metalcompute(monkeypatch)
    assert astc._probe_metal() is True


def test_probe_falls_back_to_pyobjc(monkeypatch):
    monkeypatch.setattr(astc, "_METAL_AVAILABLE", None)
    monkeypatch.delenv("SQUISH_FORCE_METAL_SIMULATION", raising=False)
    monkeypatch.delenv("SQUISH_FORCE_METAL_AVAILABLE", raising=False)
    monkeypatch.setitem(sys.modules, "metalcompute", None)  # ImportError
    metal = types.ModuleType("Metal")
    metal.MTLCreateSystemDefaultDevice = lambda: object()
    monkeypatch.setitem(sys.modules, "Metal", metal)
    assert astc._probe_metal() is True


def test_probe_none_available(monkeypatch):
    monkeypatch.setattr(astc, "_METAL_AVAILABLE", None)
    monkeypatch.delenv("SQUISH_FORCE_METAL_SIMULATION", raising=False)
    monkeypatch.delenv("SQUISH_FORCE_METAL_AVAILABLE", raising=False)
    monkeypatch.setitem(sys.modules, "metalcompute", None)
    monkeypatch.setitem(sys.modules, "Metal", None)
    assert astc._probe_metal() is False


# ── ASTCLoaderConfig + ASTCWeightTexture ─────────────────────────────────────


def test_config_defaults():
    c = astc.ASTCLoaderConfig()
    assert c.allow_simulation is True and c.device_index == 0


def _texture(backend="simulation", mtl=None):
    return astc.ASTCWeightTexture(
        encode_result=FakeEncodeResult(),
        backend=backend,
        mtl_texture=mtl,
        layer_name="L",
    )


def test_texture_properties():
    tex = _texture()
    assert tex.original_shape == (8, 6)
    assert tex.padded_shape == (12, 6)
    assert tex.n_blocks == 2
    assert tex.scale_table.shape == (2,)


def test_texture_descriptor_dict():
    d = _texture().texture_descriptor_dict()
    assert d["pixelFormat"] == astc.METAL_FORMAT_ASTC_6x6_HDR
    assert d["width"] == 6 and d["height"] == 12


def test_texture_decode_simulation(encoder_fakes):
    out = _texture(backend="simulation").decode()
    assert out.shape == (8, 6)


def test_texture_decode_metal_reads_back(encoder_fakes):
    # metal backend with a texture object → _readback_metal → _decode_simulation
    out = _texture(backend="metal", mtl=("buf", 1)).decode()
    assert out.shape == (8, 6)


# ── ASTCLoader.create_texture ────────────────────────────────────────────────


def test_create_texture_simulation(monkeypatch):
    monkeypatch.setattr(astc, "is_metal_available", lambda: False)
    loader = astc.ASTCLoader()
    assert loader.config.allow_simulation is True
    tex = loader.create_texture(FakeEncodeResult(), layer_name="x")
    assert tex.backend == "simulation" and tex.mtl_texture is None


def test_create_texture_raises_without_simulation(monkeypatch):
    monkeypatch.setattr(astc, "is_metal_available", lambda: False)
    loader = astc.ASTCLoader(astc.ASTCLoaderConfig(allow_simulation=False))
    with pytest.raises(RuntimeError, match="Metal is not available"):
        loader.create_texture(FakeEncodeResult())


def test_create_texture_metal_path_with_verify(monkeypatch, encoder_fakes):
    monkeypatch.setattr(astc, "is_metal_available", lambda: True)
    _fake_metalcompute(monkeypatch)
    loader = astc.ASTCLoader(astc.ASTCLoaderConfig(verify_on_load=True))
    tex = loader.create_texture(FakeEncodeResult(), layer_name="m")
    assert tex.backend == "metal" and tex.mtl_texture is not None


# ── _try_metal caching ───────────────────────────────────────────────────────


def test_try_metal_caches(monkeypatch):
    calls = {"n": 0}

    def _avail():
        calls["n"] += 1
        return False

    monkeypatch.setattr(astc, "is_metal_available", _avail)
    loader = astc.ASTCLoader()
    assert loader._try_metal(FakeEncodeResult()) is False
    loader._try_metal(FakeEncodeResult())  # cached → is_metal_available not re-called
    assert calls["n"] == 1


# ── _create_metal_texture branches ───────────────────────────────────────────


def test_create_metal_texture_too_large_falls_back():
    class Big(FakeEncodeResult):
        padded_shape = (20000, 6)

    tex = astc.ASTCLoader()._create_metal_texture(Big(), layer_name="big")
    assert tex.backend == "simulation"


def test_create_metal_texture_success(monkeypatch):
    _fake_metalcompute(monkeypatch)
    tex = astc.ASTCLoader()._create_metal_texture(FakeEncodeResult(), layer_name="ok")
    assert tex.backend == "metal" and tex.mtl_texture == ("buf", 32)


def test_create_metal_texture_exception_falls_back(monkeypatch):
    _fake_metalcompute(monkeypatch, buffer_ok=False)
    tex = astc.ASTCLoader()._create_metal_texture(FakeEncodeResult(), layer_name="e")
    assert tex.backend == "simulation" and tex.mtl_texture is None


# ── supports_astc_6x6_hdr ────────────────────────────────────────────────────


def test_supports_false_without_metal(monkeypatch):
    monkeypatch.setattr(astc, "is_metal_available", lambda: False)
    assert astc.ASTCLoader().supports_astc_6x6_hdr() is False


def test_supports_true_on_apple_gpu(monkeypatch):
    monkeypatch.setattr(astc, "is_metal_available", lambda: True)
    _fake_metalcompute(monkeypatch, name="Apple M3 Max")
    assert astc.ASTCLoader().supports_astc_6x6_hdr() is True


def test_supports_false_on_radeon(monkeypatch):
    monkeypatch.setattr(astc, "is_metal_available", lambda: True)
    _fake_metalcompute(monkeypatch, name="AMD Radeon Pro 5500M")
    assert astc.ASTCLoader().supports_astc_6x6_hdr() is False


def test_supports_false_on_probe_exception(monkeypatch):
    monkeypatch.setattr(astc, "is_metal_available", lambda: True)
    _fake_metalcompute(monkeypatch, device_ok=False)
    assert astc.ASTCLoader().supports_astc_6x6_hdr() is False


# ── load_from_file ───────────────────────────────────────────────────────────


def test_load_from_file(tmp_path, monkeypatch):
    monkeypatch.setattr(astc, "is_metal_available", lambda: False)

    class _FakeResult(FakeEncodeResult):
        @staticmethod
        def deserialise(payload):
            assert payload == b"PAYLOAD"
            return FakeEncodeResult()

    monkeypatch.setattr(astc, "ASTCEncodeResult", _FakeResult)
    f = tmp_path / "layer.squizd"
    f.write_bytes(b"HEADER" + b"PAYLOAD")
    tex = astc.ASTCLoader().load_from_file(f, layer_offset=6, layer_name="f")
    assert tex.backend == "simulation" and tex.layer_name == "f"


# ── _verify_texture ──────────────────────────────────────────────────────────


def test_verify_texture_shape_mismatch_is_soft(monkeypatch, encoder_fakes):
    # decode returns a shape that does not match original → soft `pass` branch.
    class _MismatchEnc(_FakeEnc):
        def decode(self, result):
            return np.zeros((3, 3, 3), dtype=np.float32)

    monkeypatch.setattr(astc, "ASTCEncoder", _MismatchEnc)
    astc.ASTCLoader()._verify_texture(_texture())  # must not raise
