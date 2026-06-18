"""Tests for io.loader_utils — the multi-format npy/npz dequant dispatcher.

Real fixtures are used for the pure-numpy formats (NF4, INT8, passthrough, npz).
The Rust-backed INT4 dequant and the forward-looking VPTQ / DFloat11 backends are
covered by patching the dispatcher's dequant entry points / lazy getters — the
loader's branch + shape-handling logic is what's under test here.
"""
from __future__ import annotations

import io
import json
import pickle
import sys
import types

import numpy as np
import pytest

from squish.io import loader_utils as lu
from squish.io.loader_utils import (
    _build_model_args,
    _dequantize,
    _dequantize_npy,
    _load_npy_path,
    _safe_key_to_original,
    _unique_base_keys,
)
from squish.quant.nf4_quant import quantize_nf4
from squish.quant.quantizer import quantize_embeddings


def _save(tmp, name, arr):
    np.save(tmp / name, arr)


# ── _load_npy_path ─────────────────────────────────────────────────────────

class TestLoadNpyPath:
    def test_plain_npy(self, tmp_path):
        _save(tmp_path, "x.npy", np.arange(4, dtype=np.float32))
        out = _load_npy_path(tmp_path / "x.npy", mmap_mode=None)
        np.testing.assert_array_equal(out, np.arange(4, dtype=np.float32))

    def test_zst_compressed(self, tmp_path):
        import zstandard as zstd
        arr = np.arange(6, dtype=np.float32)
        buf = io.BytesIO()
        np.save(buf, arr)
        (tmp_path / "y.npy.zst").write_bytes(zstd.ZstdCompressor().compress(buf.getvalue()))
        out = _load_npy_path(tmp_path / "y.npy", mmap_mode=None)
        np.testing.assert_array_equal(out, arr)

    def test_missing_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            _load_npy_path(tmp_path / "nope.npy")


# ── helpers ────────────────────────────────────────────────────────────────

class TestHelpers:
    def test_safe_key_to_original_inverts(self, tmp_path):
        (tmp_path / "m.json").write_text(json.dumps({"orig.name": "safe__key"}))
        assert _safe_key_to_original(str(tmp_path / "m.json")) == {"safe__key": "orig.name"}

    def test_unique_base_keys(self):
        files = ["m__embed__q", "m__embed__s", "m__embed__shape", "m__w__pt", "junk"]
        assert _unique_base_keys(files) == {"m__embed", "m__w"}

    def test_build_model_args_from_dict(self):
        class MA:
            @staticmethod
            def from_dict(cfg):
                return ("from_dict", cfg)

        assert _build_model_args(MA, {"a": 1})[0] == "from_dict"

    def test_build_model_args_filters_fields(self):
        import dataclasses

        @dataclasses.dataclass
        class MA:
            x: int = 0
            y: int = 0

        out = _build_model_args(MA, {"x": 5, "z": 99})  # z dropped
        assert out.x == 5 and out.y == 0

    def test_get_zstd_dctx(self):
        assert lu._get_zstd_dctx() is not None  # zstandard installed in dev env

    def test_get_dfloat11_via_stub_module(self, monkeypatch):
        mod = types.ModuleType("squish.quant.dfloat11")
        mod.DFloat11Compressor = type("C", (), {})
        mod.DFloat11Config = type("Cfg", (), {})
        monkeypatch.setitem(sys.modules, "squish.quant.dfloat11", mod)
        cfg, comp = lu._get_dfloat11()
        assert cfg is mod.DFloat11Config and comp is mod.DFloat11Compressor

    def test_get_vptq_via_stub_module(self, monkeypatch):
        mod = types.ModuleType("squish.quant.vptq")
        for n in ("VPTQConfig", "VPTQCodebook", "VPTQLayer", "VPTQQuantizer"):
            setattr(mod, n, type(n, (), {}))
        monkeypatch.setitem(sys.modules, "squish.quant.vptq", mod)
        cfg, cb, layer, quant = lu._get_vptq()
        assert (cfg, cb, layer, quant) == (
            mod.VPTQConfig, mod.VPTQCodebook, mod.VPTQLayer, mod.VPTQQuantizer
        )


# ── _dequantize (npz path) ──────────────────────────────────────────────────

class TestDequantizeNpz:
    def test_pt_passthrough_with_shape(self):
        arr = np.arange(6, dtype=np.float32)
        npz = {"w__pt": arr, "w__shape": np.array([2, 3])}

        class _NF(dict):
            files = ["w__pt", "w__shape"]

        out = _dequantize(_NF(npz), "w")
        assert out.shape == (2, 3)

    def test_pt_passthrough_without_shape(self):
        arr = np.ones((2, 2), dtype=np.float16)

        class _NF(dict):
            files = ["w__pt"]

        out = _dequantize(_NF({"w__pt": arr}), "w")
        assert out.shape == (2, 2) and out.dtype == np.float32

    def test_int8_quantized_reconstruct(self):
        W = np.random.default_rng(0).standard_normal((3, 8)).astype(np.float32)
        res = quantize_embeddings(W, group_size=0, backend="numpy")

        class _NF(dict):
            files = ["w__q", "w__s", "w__shape"]

        npz = {"w__q": res.quantized, "w__s": res.scales, "w__shape": np.array([3, 8])}
        out = _dequantize(_NF(npz), "w")
        assert out.shape == (3, 8)
        assert np.abs(out - W).max() < 0.1


# ── _dequantize_npy: real pure-numpy formats ─────────────────────────────────

class TestDequantizeNpyRealFormats:
    def test_nf4(self, tmp_path):
        W = np.random.default_rng(1).standard_normal((4, 64)).astype(np.float32)
        packed, scales = quantize_nf4(W, group_size=32)
        _save(tmp_path, "t__nf4.npy", packed)
        _save(tmp_path, "t__s_nf4.npy", scales)
        _save(tmp_path, "t__shape.npy", np.array([4, 64]))
        out = _dequantize_npy(tmp_path, "t")
        assert out.shape == (4, 64)
        assert np.linalg.norm(out - W) / np.linalg.norm(W) < 0.15

    def test_nf4_without_shape(self, tmp_path):
        W = np.random.default_rng(2).standard_normal((2, 32)).astype(np.float32)
        packed, scales = quantize_nf4(W, group_size=32)
        _save(tmp_path, "t__nf4.npy", packed)
        _save(tmp_path, "t__s_nf4.npy", scales)
        out = _dequantize_npy(tmp_path, "t")
        assert out.shape == (2, 32)

    def test_passthrough_pt(self, tmp_path):
        arr = np.ones((3, 5), dtype=np.float16)
        _save(tmp_path, "t__pt.npy", arr)
        out = _dequantize_npy(tmp_path, "t")
        assert out.shape == (3, 5) and out.dtype == np.float32

    def test_int8_q_s(self, tmp_path):
        W = np.random.default_rng(3).standard_normal((3, 8)).astype(np.float32)
        res = quantize_embeddings(W, group_size=0, backend="numpy")
        _save(tmp_path, "t__q.npy", res.quantized)
        _save(tmp_path, "t__s.npy", res.scales)
        out = _dequantize_npy(tmp_path, "t")
        assert out.shape == (3, 8)
        assert np.abs(out - W).max() < 0.1


# ── _dequantize_npy: INT4 (Rust dequant patched out) ─────────────────────────

class TestDequantizeNpyInt4:
    def test_q4a_asymmetric(self, tmp_path, monkeypatch):
        monkeypatch.setattr(lu, "dequantize_int4_asymmetric",
                            lambda packed, scales, offsets, group_size: np.zeros((2, 8), np.float32))
        _save(tmp_path, "t__q4a.npy", np.zeros((2, 4), np.uint8))   # d//2 = 4 → d=8
        _save(tmp_path, "t__s4a.npy", np.ones((2, 2), np.float32))  # n_groups=2 → gs=4
        _save(tmp_path, "t__z4a.npy", np.zeros((2, 2), np.float32))
        _save(tmp_path, "t__shape.npy", np.array([2, 8]))
        out = _dequantize_npy(tmp_path, "t")
        assert out.shape == (2, 8)

    def test_q4a_without_shape(self, tmp_path, monkeypatch):
        monkeypatch.setattr(lu, "dequantize_int4_asymmetric",
                            lambda packed, scales, offsets, group_size: np.zeros((2, 8), np.float32))
        _save(tmp_path, "t__q4a.npy", np.zeros((2, 4), np.uint8))
        _save(tmp_path, "t__s4a.npy", np.ones((2, 2), np.float32))
        _save(tmp_path, "t__z4a.npy", np.zeros((2, 2), np.float32))
        out = _dequantize_npy(tmp_path, "t")  # no __shape → returns arr directly
        assert out.shape == (2, 8)

    def test_q4_symmetric(self, tmp_path, monkeypatch):
        monkeypatch.setattr(lu, "dequantize_int4",
                            lambda packed, scales, group_size: np.zeros((2, 8), np.float32))
        _save(tmp_path, "t__q4.npy", np.zeros((2, 4), np.uint8))
        _save(tmp_path, "t__s4.npy", np.ones((2, 2), np.float32))
        out = _dequantize_npy(tmp_path, "t")
        assert out.shape == (2, 8)


# ── _dequantize_npy: DFloat11 + VPTQ (stub backends) ─────────────────────────

class TestDequantizeNpyStubbedBackends:
    def test_dfloat11_passthrough(self, tmp_path, monkeypatch):
        class _Comp:
            def decompress_array(self, blocks):
                return np.ones(6, np.float32)

        monkeypatch.setattr(lu, "_get_dfloat11", lambda: (object, _Comp))
        # __pt_df11 holds pickled "blocks" as raw uint8 bytes.
        blob = np.frombuffer(pickle.dumps([1, 2, 3]), dtype=np.uint8)
        _save(tmp_path, "t__pt_df11.npy", blob)
        _save(tmp_path, "t__shape.npy", np.array([2, 3]))
        out = _dequantize_npy(tmp_path, "t")
        assert out.shape == (2, 3)

    def test_dfloat11_without_shape(self, tmp_path, monkeypatch):
        class _Comp:
            def decompress_array(self, blocks):
                return np.ones(6, np.float32)

        monkeypatch.setattr(lu, "_get_dfloat11", lambda: (object, _Comp))
        blob = np.frombuffer(pickle.dumps([1]), dtype=np.uint8)
        _save(tmp_path, "t__pt_df11.npy", blob)
        out = _dequantize_npy(tmp_path, "t")  # no __shape → flat return
        assert out.shape == (6,) and out.dtype == np.float32

    def test_q4_with_df11_scales(self, tmp_path, monkeypatch):
        class _Comp:
            def decompress_array(self, blocks):
                return np.ones(4, np.float32)  # → reshape (2, 2) scales

        monkeypatch.setattr(lu, "_get_dfloat11", lambda: (object, _Comp))
        monkeypatch.setattr(lu, "dequantize_int4",
                            lambda packed, scales, group_size: np.zeros((2, 8), np.float32))
        _save(tmp_path, "t__q4.npy", np.zeros((2, 4), np.uint8))
        blob = np.frombuffer(pickle.dumps(["x"]), dtype=np.uint8)
        _save(tmp_path, "t__s4_df11.npy", blob)
        out = _dequantize_npy(tmp_path, "t")
        assert out.shape == (2, 8)

    def _install_vptq(self, monkeypatch, n_rows, n_cols):
        class _CB:
            pass

        class _Layer:
            def __init__(self, **kw):
                self.kw = kw

        class _Cfg:
            def __init__(self, **kw):
                self.kw = kw

        class _Quant:
            def decompress(self, layer):
                return np.zeros((n_rows, n_cols), np.float32)

        monkeypatch.setattr(lu, "_get_vptq", lambda: (_Cfg, _CB, _Layer, _Quant))

    def test_vptq_primary_only(self, tmp_path, monkeypatch):
        self._install_vptq(monkeypatch, 3, 4)
        gs, n_cb = 2, 4
        _save(tmp_path, "t__vq_idx.npy", np.zeros((3, 2), np.int64))
        _save(tmp_path, "t__vq_cb.npy",
              np.concatenate([np.zeros(n_cb * gs, np.float32), np.array([0, 0, 1], np.float32)]))
        _save(tmp_path, "t__vq_res.npy", np.array([0], np.int64))   # size==1 → no residual
        _save(tmp_path, "t__vq_rescb.npy", np.zeros(1, np.float32))
        _save(tmp_path, "t__vq_meta.npy", np.array([3, 4, gs, n_cb, 0], np.int64))
        out = _dequantize_npy(tmp_path, "t")
        assert out.shape == (3, 4)

    def test_vptq_with_residual_and_col_scales(self, tmp_path, monkeypatch):
        self._install_vptq(monkeypatch, 3, 4)
        gs, n_cb, n_res = 2, 4, 2
        _save(tmp_path, "t__vq_idx.npy", np.zeros((3, 2), np.int64))
        _save(tmp_path, "t__vq_cb.npy",
              np.concatenate([np.zeros(n_cb * gs, np.float32), np.array([0, 0, 1], np.float32)]))
        _save(tmp_path, "t__vq_res.npy", np.zeros((3, 2), np.int64))  # size>1 → residual
        _save(tmp_path, "t__vq_rescb.npy",
              np.concatenate([np.zeros(n_res * gs, np.float32), np.array([0, 0, 1], np.float32)]))
        _save(tmp_path, "t__vq_cols.npy", np.ones(4, np.float32))     # col_scales present
        _save(tmp_path, "t__vq_meta.npy", np.array([3, 4, gs, n_cb, n_res], np.int64))
        _save(tmp_path, "t__shape.npy", np.array([3, 4]))
        out = _dequantize_npy(tmp_path, "t")
        assert out.shape == (3, 4)

    def test_vptq_empty_col_scales(self, tmp_path, monkeypatch):
        # __vq_cols present but empty → cs.size > 0 is False → col_scales stays None.
        self._install_vptq(monkeypatch, 3, 4)
        gs, n_cb = 2, 4
        _save(tmp_path, "t__vq_idx.npy", np.zeros((3, 2), np.int64))
        _save(tmp_path, "t__vq_cb.npy",
              np.concatenate([np.zeros(n_cb * gs, np.float32), np.array([0, 0, 1], np.float32)]))
        _save(tmp_path, "t__vq_res.npy", np.array([0], np.int64))
        _save(tmp_path, "t__vq_rescb.npy", np.zeros(1, np.float32))
        _save(tmp_path, "t__vq_cols.npy", np.zeros(0, np.float32))    # empty
        _save(tmp_path, "t__vq_meta.npy", np.array([3, 4, gs, n_cb, 0], np.int64))
        out = _dequantize_npy(tmp_path, "t")
        assert out.shape == (3, 4)
