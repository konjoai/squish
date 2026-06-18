"""Tests for the native GGUF v3 loader — header/KV/tensor parsing + dequant.

Drives the parser with synthetic in-memory GGUF byte buffers (no real model
file needed), exercising the documented header → KV → tensor-info → data layout.
"""
from __future__ import annotations

import struct

import numpy as np
import pytest

from squish.io.gguf_loader import (
    GGUFConfig,
    GGUFNativeLoader,
    GGUFTensor,
)

# GGUF value-type ids (mirror the loader's private constants).
T_UINT32, T_INT32, T_FLOAT32, T_BOOL, T_STRING, T_ARRAY = 4, 5, 6, 7, 8, 9
# ggml tensor-type ids.
GGML_F32, GGML_F16 = 0, 1


def _gguf_string(s: str) -> bytes:
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def _kv_value(vtype: int, value) -> bytes:
    if vtype == T_UINT32:
        return struct.pack("<I", value)
    if vtype == T_INT32:
        return struct.pack("<i", value)
    if vtype == T_FLOAT32:
        return struct.pack("<f", value)
    if vtype == T_BOOL:
        return struct.pack("<B", 1 if value else 0)
    if vtype == T_STRING:
        return _gguf_string(value)
    if vtype == T_ARRAY:
        elem_type, items = value
        out = struct.pack("<I", elem_type) + struct.pack("<Q", len(items))
        for it in items:
            out += _kv_value(elem_type, it)
        return out
    raise AssertionError(f"unsupported test vtype {vtype}")


def _build_gguf(path, kv: list[tuple], tensors: list[dict], version: int = 3) -> None:
    """Write a minimal valid GGUF file.

    kv:      list of (key, vtype, value)
    tensors: list of {name, shape, ggml_type, data(np.ndarray|None)}
    """
    body = bytearray()
    body += b"GGUF"
    body += struct.pack("<I", version)
    body += struct.pack("<Q", len(tensors))
    body += struct.pack("<Q", len(kv))
    for key, vtype, value in kv:
        body += _gguf_string(key)
        body += struct.pack("<I", vtype)
        body += _kv_value(vtype, value)

    # Tensor info section; assign sequential offsets into the data section.
    blobs: list[bytes] = []
    offset = 0
    for t in tensors:
        arr = t.get("data")
        if t["ggml_type"] == GGML_F32:
            blob = arr.astype(np.float32).tobytes() if arr is not None else b""
        elif t["ggml_type"] == GGML_F16:
            blob = arr.astype(np.float16).tobytes() if arr is not None else b""
        else:
            blob = t.get("raw", b"")
        body += _gguf_string(t["name"])
        body += struct.pack("<I", len(t["shape"]))
        for d in t["shape"]:
            body += struct.pack("<Q", d)
        body += struct.pack("<I", t["ggml_type"])
        body += struct.pack("<Q", offset)
        blobs.append(blob)
        offset += len(blob)

    # Align data section to a 32-byte boundary, then append blobs in order.
    pad = (-len(body)) % 32
    body += b"\x00" * pad
    for blob in blobs:
        body += blob

    path.write_bytes(bytes(body))


class TestConfig:
    def test_defaults(self):
        cfg = GGUFConfig()
        assert "Q4_K" in cfg.supported_qtypes and cfg.device == "cpu"

    def test_unknown_qtype_raises(self):
        with pytest.raises(ValueError, match="Unknown quantization type"):
            GGUFConfig(supported_qtypes=["Q9_Z"])

    def test_bad_device_raises(self):
        with pytest.raises(ValueError, match="device must be"):
            GGUFConfig(device="tpu")


class TestTensorDataclass:
    def test_n_elements(self):
        t = GGUFTensor(name="x", n_dims=2, shape=(3, 4), dtype="F32", offset=0)
        assert t.n_elements == 12


class TestHeaderAndKV:
    def test_metadata_and_kv_types(self, tmp_path):
        p = tmp_path / "m.gguf"
        kv = [
            ("general.architecture", T_STRING, "llama"),
            ("llama.block_count", T_UINT32, 32),
            ("answer", T_INT32, -7),
            ("scale", T_FLOAT32, 0.5),
            ("flag", T_BOOL, True),
            ("tokens", T_ARRAY, (T_STRING, ["a", "b", "c"])),
        ]
        _build_gguf(p, kv, [])
        meta = GGUFNativeLoader(GGUFConfig()).get_metadata(str(p))
        assert meta.magic == b"GGUF" and meta.version == 3
        assert meta.n_tensors == 0 and meta.n_kv == 6
        assert meta.kv["general.architecture"] == "llama"
        assert meta.kv["llama.block_count"] == 32
        assert meta.kv["answer"] == -7
        assert meta.kv["scale"] == pytest.approx(0.5)
        assert meta.kv["flag"] is True
        assert meta.kv["tokens"] == ["a", "b", "c"]

    def test_bad_magic_raises(self, tmp_path):
        p = tmp_path / "bad.gguf"
        p.write_bytes(b"NOPE" + b"\x00" * 24)
        with pytest.raises(ValueError, match="Not a GGUF file"):
            GGUFNativeLoader(GGUFConfig()).get_metadata(str(p))

    def test_unsupported_version_raises(self, tmp_path):
        p = tmp_path / "v9.gguf"
        _build_gguf(p, [], [], version=9)
        with pytest.raises(ValueError, match="Unsupported GGUF version"):
            GGUFNativeLoader(GGUFConfig()).get_metadata(str(p))

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            GGUFNativeLoader(GGUFConfig()).get_metadata("/no/such/file.gguf")

    def test_unknown_kv_value_type_becomes_none(self, tmp_path):
        # A KV pair with an unrecognized value type is stored as None (the parser
        # returns a placeholder rather than crashing).
        body = bytearray(b"GGUF")
        body += struct.pack("<I", 3)          # version
        body += struct.pack("<Q", 0)          # n_tensors
        body += struct.pack("<Q", 1)          # n_kv
        body += _gguf_string("mystery")
        body += struct.pack("<I", 99)         # unknown vtype → _read_value returns None
        p = tmp_path / "unk.gguf"
        p.write_bytes(bytes(body))
        meta = GGUFNativeLoader(GGUFConfig()).get_metadata(str(p))
        assert meta.kv["mystery"] is None


class TestTensorInfoAndLoad:
    def test_list_tensors_descriptors(self, tmp_path):
        p = tmp_path / "t.gguf"
        tensors = [
            {"name": "blk.0.attn_q.weight", "shape": (2, 3), "ggml_type": GGML_F32,
             "data": np.zeros((2, 3), np.float32)},
        ]
        _build_gguf(p, [], tensors)
        descs = GGUFNativeLoader(GGUFConfig()).list_tensors(str(p))
        assert len(descs) == 1
        d = descs[0]
        assert d.name == "blk.0.attn_q.weight" and d.n_dims == 2
        assert d.shape == (2, 3) and d.dtype == "F32" and d.offset == 0

    def test_load_f32_and_f16_roundtrip(self, tmp_path):
        p = tmp_path / "w.gguf"
        a = np.array([1.0, -2.0, 3.5, 4.25, 0.0, 7.0], np.float32)
        b = np.array([0.5, 1.5, -1.0, 2.0], np.float32)
        tensors = [
            {"name": "a_f32", "shape": (6,), "ggml_type": GGML_F32, "data": a},
            {"name": "b_f16", "shape": (4,), "ggml_type": GGML_F16, "data": b},
        ]
        _build_gguf(p, [], tensors)
        out = GGUFNativeLoader(GGUFConfig()).load(str(p))
        np.testing.assert_array_equal(out["a_f32"], a)
        # F16 storage is lossless for these exact half-representable values.
        np.testing.assert_allclose(out["b_f16"], b, rtol=0, atol=1e-3)

    def test_unsupported_qtype_is_skipped_on_load(self, tmp_path):
        # A loader that only supports F32 must skip an F16 tensor (no entry).
        p = tmp_path / "skip.gguf"
        tensors = [{"name": "h", "shape": (4,), "ggml_type": GGML_F16,
                    "data": np.ones(4, np.float32)}]
        _build_gguf(p, [], tensors)
        out = GGUFNativeLoader(GGUFConfig(supported_qtypes=["F32"])).load(str(p))
        assert "h" not in out


class TestDequantBlock:
    def test_dequant_block_f32(self):
        loader = GGUFNativeLoader(GGUFConfig())
        a = np.array([1.0, 2.0, 3.0], np.float32)
        out = loader.dequantize_block(a.tobytes(), "F32", 3)
        np.testing.assert_array_equal(out, a)

    def test_dequant_block_q8_0(self):
        loader = GGUFNativeLoader(GGUFConfig())
        scale = np.float16(0.25)
        ints = np.arange(-16, 16, dtype=np.int8)  # 32 values
        raw = struct.pack("<e", float(scale)) + ints.tobytes()
        out = loader.dequantize_block(raw, "Q8_0", 32)
        np.testing.assert_allclose(out, ints.astype(np.float32) * float(scale), atol=1e-3)

    def test_dequant_block_unsupported_raises(self):
        with pytest.raises(ValueError, match="Unsupported qtype"):
            GGUFNativeLoader(GGUFConfig()).dequantize_block(b"", "BOGUS", 0)

    def test_dequant_block_k_quant_path(self):
        # Q4_0: 18 bytes/block, 32 elems/block. Exercises _dequant_generic_k +
        # the cross-byte branch of _unpack_bits. We assert structure/finiteness,
        # not exact values (the K-quant path is a documented approximation).
        loader = GGUFNativeLoader(GGUFConfig())
        payload = bytes(range(14))                       # 14 payload bytes
        scale = struct.pack("<f", 2.0)                   # last 4 bytes = scale
        out = loader.dequantize_block(payload + scale, "Q4_0", 32)
        assert out.shape == (32,)
        assert np.isfinite(out).all()

    def test_q8_0_partial_block_is_handled(self):
        # Fewer bytes than a full 34-byte block → the loop breaks cleanly.
        out = GGUFNativeLoader(GGUFConfig()).dequantize_block(b"\x00" * 10, "Q8_0", 32)
        assert out.shape == (32,)

    def test_unpack_bits_known_pattern(self):
        # 0b10110100 → 4-bit lo nibble 0b0100=4, hi nibble 0b1011=11.
        out = GGUFNativeLoader._unpack_bits(bytes([0b10110100]), bits=4, n=2)
        assert out.tolist() == [4.0, 11.0]

    def test_unpack_bits_crosses_byte_boundary(self):
        # bits=3: the 3rd value (bit_pos=6) spans byte 0→1, exercising the
        # cross-byte merge branch.
        out = GGUFNativeLoader._unpack_bits(bytes([0b11111111, 0b00000001]), bits=3, n=5)
        assert out.tolist() == [7.0, 7.0, 7.0, 0.0, 0.0]

    def test_k_quant_zero_scale_guard(self):
        # A block whose super-scale reads as 0.0 must fall back to scale=1.0
        # instead of zeroing/NaNing the output.
        raw = bytes(14) + struct.pack("<f", 0.0)
        out = GGUFNativeLoader(GGUFConfig()).dequantize_block(raw, "Q4_0", 32)
        assert out.shape == (32,) and np.isfinite(out).all()

    def test_k_quant_tiny_block_breaks(self):
        # Fewer than 4 bytes → the per-block loop breaks without indexing past end.
        out = GGUFNativeLoader(GGUFConfig()).dequantize_block(b"\x00\x00", "Q4_0", 32)
        assert out.shape == (32,)


class TestSynthetic:
    def test_make_synthetic_shapes(self):
        loader = GGUFNativeLoader.make_synthetic(
            {"w1": (2, 3), "w2": (5,)}, qtype="F32", seed=1
        )
        assert loader._synthetic_tensors["w1"].shape == (2, 3)
        assert loader._synthetic_tensors["w2"].shape == (5,)
        assert loader._synthetic_tensors["w1"].dtype == np.float32
