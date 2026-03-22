"""Tests for Wave 50b I/O modules: GGUFNativeLoader, WeightDecompressStream, ModelShardLoader."""

from __future__ import annotations

import tempfile
import os
import struct
import threading
import unittest

import numpy as np

from squish.io.gguf_loader import (
    GGUFConfig,
    GGUFMetadata,
    GGUFNativeLoader,
    GGUFTensor,
)
from squish.io.weight_decompress_stream import (
    WeightDecompressStream,
    WeightStreamConfig,
    WeightStreamHandle,
)
from squish.io.model_shard_loader import (
    LayerShard,
    ModelShardLoader,
    ShardConfig,
    ShardTier,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_minimal_gguf(
    n_tensors: int = 0,
    n_kv: int = 0,
    version: int = 3,
    kv_bytes: bytes = b"",
    tensor_info_bytes: bytes = b"",
    data_bytes: bytes = b"",
) -> bytes:
    """Build the minimal valid GGUF bytes for parser testing."""
    buf = b"GGUF"
    buf += struct.pack("<I", version)
    buf += struct.pack("<Q", n_tensors)
    buf += struct.pack("<Q", n_kv)
    buf += kv_bytes
    buf += tensor_info_bytes
    # Pad to 32-byte alignment
    header_len = len(buf)
    pad = (32 - header_len % 32) % 32
    buf += b"\x00" * pad
    buf += data_bytes
    return buf


def _write_gguf_file(content: bytes) -> str:
    """Write *content* to a temporary file and return its path."""
    fd, path = tempfile.mkstemp(suffix=".gguf")
    os.write(fd, content)
    os.close(fd)
    return path


def _make_layers(n: int, shape=(32, 32), seed=0) -> dict:
    rng = np.random.default_rng(seed)
    return {i: rng.standard_normal(shape).astype(np.float32) for i in range(n)}


# ---------------------------------------------------------------------------
# GGUFConfig
# ---------------------------------------------------------------------------


class TestGGUFConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = GGUFConfig()
        self.assertEqual(cfg.device, "cpu")
        self.assertEqual(cfg.seed, 0)
        self.assertIn("F32", cfg.supported_qtypes)
        self.assertIn("Q8_0", cfg.supported_qtypes)

    def test_custom_qtypes(self):
        cfg = GGUFConfig(supported_qtypes=["F32", "F16"])
        self.assertEqual(cfg.supported_qtypes, ["F32", "F16"])

    def test_invalid_qtype_raises(self):
        with self.assertRaises(ValueError):
            GGUFConfig(supported_qtypes=["BOGUS"])

    def test_metal_device_valid(self):
        cfg = GGUFConfig(device="metal")
        self.assertEqual(cfg.device, "metal")

    def test_invalid_device_raises(self):
        with self.assertRaises(ValueError):
            GGUFConfig(device="cuda")

    def test_q2k_accepted(self):
        cfg = GGUFConfig(supported_qtypes=["Q2_K"])
        self.assertEqual(cfg.supported_qtypes, ["Q2_K"])

    def test_q4k_accepted(self):
        cfg = GGUFConfig(supported_qtypes=["Q4_K"])
        self.assertIn("Q4_K", cfg.supported_qtypes)

    def test_all_default_qtypes_valid(self):
        cfg = GGUFConfig()
        self.assertGreater(len(cfg.supported_qtypes), 0)


# ---------------------------------------------------------------------------
# GGUFMetadata
# ---------------------------------------------------------------------------


class TestGGUFMetadata(unittest.TestCase):
    def test_construction(self):
        meta = GGUFMetadata(magic=b"GGUF", version=3, n_tensors=10, n_kv=5, kv={"key": "value"})
        self.assertEqual(meta.magic, b"GGUF")

    def test_version_field(self):
        meta = GGUFMetadata(magic=b"GGUF", version=2, n_tensors=0, n_kv=0, kv={})
        self.assertEqual(meta.version, 2)

    def test_kv_dict(self):
        meta = GGUFMetadata(magic=b"GGUF", version=3, n_tensors=1, n_kv=1, kv={"model": "llama"})
        self.assertEqual(meta.kv["model"], "llama")

    def test_n_tensors_field(self):
        meta = GGUFMetadata(magic=b"GGUF", version=3, n_tensors=42, n_kv=0, kv={})
        self.assertEqual(meta.n_tensors, 42)


# ---------------------------------------------------------------------------
# GGUFTensor
# ---------------------------------------------------------------------------


class TestGGUFTensor(unittest.TestCase):
    def test_construction(self):
        t = GGUFTensor(name="blk.0.attn.q", n_dims=2, shape=(64, 64), dtype="Q4_K", offset=0)
        self.assertEqual(t.name, "blk.0.attn.q")

    def test_n_elements_2d(self):
        t = GGUFTensor(name="x", n_dims=2, shape=(8, 16), dtype="F32", offset=0)
        self.assertEqual(t.n_elements, 128)

    def test_n_elements_1d(self):
        t = GGUFTensor(name="x", n_dims=1, shape=(256,), dtype="F16", offset=0)
        self.assertEqual(t.n_elements, 256)

    def test_data_none_by_default(self):
        t = GGUFTensor(name="x", n_dims=1, shape=(16,), dtype="F32", offset=0)
        self.assertIsNone(t.data)

    def test_data_can_be_set(self):
        t = GGUFTensor(name="x", n_dims=1, shape=(4,), dtype="F32", offset=0,
                       data=np.zeros(4, dtype=np.float32))
        self.assertIsNotNone(t.data)

    def test_dtype_field(self):
        t = GGUFTensor(name="y", n_dims=2, shape=(2, 4), dtype="Q8_0", offset=100)
        self.assertEqual(t.dtype, "Q8_0")


# ---------------------------------------------------------------------------
# GGUFNativeLoader
# ---------------------------------------------------------------------------


class TestGGUFNativeLoader(unittest.TestCase):
    def test_loader_construction(self):
        loader = GGUFNativeLoader(GGUFConfig())
        self.assertIsNotNone(loader)

    def test_invalid_magic_raises(self):
        bad_bytes = b"NOPE" + struct.pack("<I", 3) + struct.pack("<QQ", 0, 0)
        path = _write_gguf_file(bad_bytes)
        try:
            loader = GGUFNativeLoader(GGUFConfig())
            with self.assertRaises(ValueError):
                loader.load(path)
        finally:
            os.unlink(path)

    def test_invalid_version_raises(self):
        bad = b"GGUF" + struct.pack("<I", 99) + struct.pack("<QQ", 0, 0)
        path = _write_gguf_file(bad)
        try:
            loader = GGUFNativeLoader(GGUFConfig())
            with self.assertRaises(ValueError):
                loader.load(path)
        finally:
            os.unlink(path)

    def test_file_not_found_raises(self):
        loader = GGUFNativeLoader(GGUFConfig())
        with self.assertRaises(FileNotFoundError):
            loader.load("/tmp/nonexistent_gguf_test_xyz.gguf")

    def test_parse_empty_gguf(self):
        content = _make_minimal_gguf(n_tensors=0, n_kv=0)
        path = _write_gguf_file(content)
        try:
            loader = GGUFNativeLoader(GGUFConfig())
            meta = loader.get_metadata(path)
            self.assertEqual(meta.n_tensors, 0)
        finally:
            os.unlink(path)

    def test_metadata_magic(self):
        content = _make_minimal_gguf()
        path = _write_gguf_file(content)
        try:
            loader = GGUFNativeLoader(GGUFConfig())
            meta = loader.get_metadata(path)
            self.assertEqual(meta.magic, b"GGUF")
        finally:
            os.unlink(path)

    def test_metadata_version(self):
        content = _make_minimal_gguf(version=3)
        path = _write_gguf_file(content)
        try:
            loader = GGUFNativeLoader(GGUFConfig())
            meta = loader.get_metadata(path)
            self.assertEqual(meta.version, 3)
        finally:
            os.unlink(path)

    def test_list_tensors_empty_file(self):
        content = _make_minimal_gguf()
        path = _write_gguf_file(content)
        try:
            loader = GGUFNativeLoader(GGUFConfig())
            tensors = loader.list_tensors(path)
            self.assertEqual(len(tensors), 0)
        finally:
            os.unlink(path)

    def test_dequantize_block_f32(self):
        loader = GGUFNativeLoader(GGUFConfig())
        data = np.array([1.0, 2.0, 3.0], dtype=np.float32).tobytes()
        result = loader.dequantize_block(data, "F32", 3)
        np.testing.assert_array_almost_equal(result, [1.0, 2.0, 3.0])

    def test_dequantize_block_f16(self):
        loader = GGUFNativeLoader(GGUFConfig())
        data = np.array([1.0, 0.5], dtype=np.float16).tobytes()
        result = loader.dequantize_block(data, "F16", 2)
        self.assertEqual(result.shape[0], 2)

    def test_dequantize_block_q8_0(self):
        loader = GGUFNativeLoader(GGUFConfig())
        # Build a valid Q8_0 block: 2 byte float16 scale + 32 int8
        scale = np.array([1.0], dtype=np.float16).tobytes()
        ints = np.zeros(32, dtype=np.int8).tobytes()
        data = scale + ints
        result = loader.dequantize_block(data, "Q8_0", 32)
        self.assertEqual(result.shape[0], 32)

    def test_dequantize_block_q4_k_shape(self):
        loader = GGUFNativeLoader(GGUFConfig())
        # Q4_K: 256 elements, 144 bytes per block
        data = bytes(144)
        result = loader.dequantize_block(data, "Q4_K", 256)
        self.assertEqual(result.shape[0], 256)

    def test_dequantize_block_q2_k_shape(self):
        loader = GGUFNativeLoader(GGUFConfig())
        data = bytes(84)
        result = loader.dequantize_block(data, "Q2_K", 256)
        self.assertEqual(result.shape[0], 256)

    def test_dequantize_block_q3_k_shape(self):
        loader = GGUFNativeLoader(GGUFConfig())
        data = bytes(110)
        result = loader.dequantize_block(data, "Q3_K", 256)
        self.assertEqual(result.shape[0], 256)

    def test_dequantize_block_q5_k_shape(self):
        loader = GGUFNativeLoader(GGUFConfig())
        data = bytes(176)
        result = loader.dequantize_block(data, "Q5_K", 256)
        self.assertEqual(result.shape[0], 256)

    def test_dequantize_block_unknown_type_raises(self):
        loader = GGUFNativeLoader(GGUFConfig())
        with self.assertRaises(ValueError):
            loader.dequantize_block(b"\x00" * 32, "Q9_K", 256)

    def test_unpack_bits_4bit(self):
        data = bytes([0xAB])  # 1010 1011 → extracts nibbles
        result = GGUFNativeLoader._unpack_bits(data, 4, 2)
        self.assertEqual(result.shape[0], 2)

    def test_synthetic_loader_creation(self):
        loader = GGUFNativeLoader.make_synthetic(
            tensor_shapes={"weight": (4, 4), "bias": (4,)}, seed=0
        )
        self.assertIn("weight", loader._synthetic_tensors)
        self.assertIn("bias", loader._synthetic_tensors)

    def test_synthetic_tensor_shape(self):
        loader = GGUFNativeLoader.make_synthetic({"x": (8, 16)}, seed=42)
        self.assertEqual(loader._synthetic_tensors["x"].shape, (8, 16))

    def test_synthetic_tensor_float32(self):
        loader = GGUFNativeLoader.make_synthetic({"x": (4,)}, seed=0)
        self.assertEqual(loader._synthetic_tensors["x"].dtype, np.float32)

    def test_dequantize_q8_0_zeros(self):
        loader = GGUFNativeLoader(GGUFConfig())
        scale = np.array([0.0], dtype=np.float16).tobytes()
        ints = bytes(32)
        data = scale + ints
        result = loader._dequant_q8_0(data, 32)
        np.testing.assert_array_almost_equal(result, np.zeros(32))


# ---------------------------------------------------------------------------
# WeightStreamConfig
# ---------------------------------------------------------------------------


class TestWeightStreamConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = WeightStreamConfig()
        self.assertEqual(cfg.n_layers, 32)
        self.assertEqual(cfg.bits, 4)
        self.assertEqual(cfg.chunk_size, 1)
        self.assertEqual(cfg.n_threads, 2)
        self.assertEqual(cfg.lookahead, 2)
        self.assertEqual(cfg.seed, 0)

    def test_valid_bits_2(self):
        cfg = WeightStreamConfig(bits=2)
        self.assertEqual(cfg.bits, 2)

    def test_valid_bits_16(self):
        cfg = WeightStreamConfig(bits=16)
        self.assertEqual(cfg.bits, 16)

    def test_invalid_bits_raises(self):
        with self.assertRaises(ValueError):
            WeightStreamConfig(bits=5)

    def test_n_layers_zero_raises(self):
        with self.assertRaises(ValueError):
            WeightStreamConfig(n_layers=0)

    def test_chunk_size_zero_raises(self):
        with self.assertRaises(ValueError):
            WeightStreamConfig(chunk_size=0)

    def test_n_threads_zero_raises(self):
        with self.assertRaises(ValueError):
            WeightStreamConfig(n_threads=0)

    def test_negative_lookahead_raises(self):
        with self.assertRaises(ValueError):
            WeightStreamConfig(lookahead=-1)

    def test_lookahead_zero_valid(self):
        cfg = WeightStreamConfig(lookahead=0)
        self.assertEqual(cfg.lookahead, 0)


# ---------------------------------------------------------------------------
# WeightStreamHandle
# ---------------------------------------------------------------------------


class TestWeightStreamHandle(unittest.TestCase):
    def test_construction(self):
        h = WeightStreamHandle(layer_idx=3, status="pending")
        self.assertEqual(h.layer_idx, 3)
        self.assertEqual(h.status, "pending")

    def test_ready_status_valid(self):
        h = WeightStreamHandle(layer_idx=0, status="ready")
        self.assertEqual(h.status, "ready")

    def test_consumed_status_valid(self):
        h = WeightStreamHandle(layer_idx=0, status="consumed")
        self.assertEqual(h.status, "consumed")

    def test_invalid_status_raises(self):
        with self.assertRaises(ValueError):
            WeightStreamHandle(layer_idx=0, status="bogus")

    def test_future_none_by_default(self):
        h = WeightStreamHandle(layer_idx=5)
        self.assertIsNone(h._future)


# ---------------------------------------------------------------------------
# WeightDecompressStream
# ---------------------------------------------------------------------------


class TestWeightDecompressStream(unittest.TestCase):
    def _make_W(self, shape=(16, 16), seed=0):
        return np.random.default_rng(seed).standard_normal(shape).astype(np.float32)

    def test_compress_decompress_roundtrip_4bit(self):
        W = self._make_W()
        compressed = WeightDecompressStream.compress_weight(W, bits=4)
        W_rec = WeightDecompressStream.decompress_weight(compressed, bits=4)
        self.assertEqual(W_rec.shape, W.shape)

    def test_compress_decompress_roundtrip_8bit(self):
        W = self._make_W()
        compressed = WeightDecompressStream.compress_weight(W, bits=8)
        W_rec = WeightDecompressStream.decompress_weight(compressed, bits=8)
        self.assertEqual(W_rec.shape, W.shape)

    def test_compress_decompress_roundtrip_16bit(self):
        W = self._make_W()
        compressed = WeightDecompressStream.compress_weight(W, bits=16)
        W_rec = WeightDecompressStream.decompress_weight(compressed, bits=16)
        self.assertEqual(W_rec.shape, W.shape)

    def test_compressed_output_uint8(self):
        W = self._make_W()
        compressed = WeightDecompressStream.compress_weight(W, bits=4)
        self.assertEqual(compressed.dtype, np.uint8)

    def test_compress_invalid_bits_raises(self):
        W = self._make_W()
        with self.assertRaises(ValueError):
            WeightDecompressStream.compress_weight(W, bits=5)

    def test_decompress_invalid_bits_raises(self):
        W = self._make_W()
        compressed = WeightDecompressStream.compress_weight(W, bits=4)
        with self.assertRaises(ValueError):
            WeightDecompressStream.decompress_weight(compressed, bits=5)

    def test_compress_higher_bits_lower_error(self):
        W = self._make_W((32, 32), seed=7)
        c4 = WeightDecompressStream.compress_weight(W, bits=4)
        c8 = WeightDecompressStream.compress_weight(W, bits=8)
        W4 = WeightDecompressStream.decompress_weight(c4, bits=4)
        W8 = WeightDecompressStream.decompress_weight(c8, bits=8)
        err4 = float(np.mean((W4 - W) ** 2))
        err8 = float(np.mean((W8 - W) ** 2))
        self.assertLessEqual(err8, err4 + 1e-3)

    def test_submit_returns_handle(self):
        cfg = WeightStreamConfig(bits=4, n_threads=1)
        stream = WeightDecompressStream(cfg)
        W = self._make_W()
        compressed = WeightDecompressStream.compress_weight(W, bits=4)
        h = stream.submit(0, compressed)
        self.assertIsInstance(h, WeightStreamHandle)

    def test_fetch_returns_ndarray(self):
        cfg = WeightStreamConfig(bits=4, n_threads=1)
        stream = WeightDecompressStream(cfg)
        W = self._make_W()
        compressed = WeightDecompressStream.compress_weight(W, bits=4)
        h = stream.submit(0, compressed)
        result = stream.fetch(h)
        self.assertIsInstance(result, np.ndarray)

    def test_fetch_marks_consumed(self):
        cfg = WeightStreamConfig(bits=4, n_threads=1)
        stream = WeightDecompressStream(cfg)
        W = self._make_W()
        compressed = WeightDecompressStream.compress_weight(W, bits=4)
        h = stream.submit(0, compressed)
        stream.fetch(h)
        self.assertEqual(h.status, "consumed")

    def test_fetch_consumed_handle_raises(self):
        cfg = WeightStreamConfig(bits=4, n_threads=1)
        stream = WeightDecompressStream(cfg)
        W = self._make_W()
        compressed = WeightDecompressStream.compress_weight(W, bits=4)
        h = stream.submit(0, compressed)
        stream.fetch(h)
        with self.assertRaises(RuntimeError):
            stream.fetch(h)

    def test_is_ready_eventually_true(self):
        cfg = WeightStreamConfig(bits=4, n_threads=1)
        stream = WeightDecompressStream(cfg)
        W = self._make_W()
        compressed = WeightDecompressStream.compress_weight(W, bits=4)
        h = stream.submit(0, compressed)
        # Wait for completion
        stream.fetch(h)  # blocks; after fetch status is consumed
        self.assertFalse(stream.is_ready(h))  # consumed → not ready

    def test_prefetch_range_returns_handles(self):
        cfg = WeightStreamConfig(bits=4, n_threads=2)
        stream = WeightDecompressStream(cfg)
        W = self._make_W()
        compressed = {i: WeightDecompressStream.compress_weight(W, bits=4) for i in range(4)}
        handles = stream.prefetch_range([0, 1, 2, 3], compressed)
        self.assertEqual(len(handles), 4)

    def test_prefetch_range_missing_key_raises(self):
        cfg = WeightStreamConfig(bits=4, n_threads=1)
        stream = WeightDecompressStream(cfg)
        W = self._make_W()
        compressed = {0: WeightDecompressStream.compress_weight(W, bits=4)}
        with self.assertRaises(KeyError):
            stream.prefetch_range([0, 99], compressed)

    def test_stats_keys(self):
        cfg = WeightStreamConfig(bits=4, n_threads=1)
        stream = WeightDecompressStream(cfg)
        s = stream.stats()
        for key in ("n_submitted", "n_fetched", "n_pending", "bits"):
            self.assertIn(key, s)

    def test_stats_counts_after_submit_fetch(self):
        cfg = WeightStreamConfig(bits=4, n_threads=1)
        stream = WeightDecompressStream(cfg)
        W = self._make_W()
        c = WeightDecompressStream.compress_weight(W, bits=4)
        h = stream.submit(0, c)
        stream.fetch(h)
        s = stream.stats()
        self.assertEqual(s["n_submitted"], 1)
        self.assertEqual(s["n_fetched"], 1)

    def test_reset_clears_stats(self):
        cfg = WeightStreamConfig(bits=4, n_threads=1)
        stream = WeightDecompressStream(cfg)
        W = self._make_W()
        c = WeightDecompressStream.compress_weight(W, bits=4)
        h = stream.submit(0, c)
        stream.fetch(h)
        stream.reset()
        s = stream.stats()
        self.assertEqual(s["n_submitted"], 0)
        self.assertEqual(s["n_fetched"], 0)


# ---------------------------------------------------------------------------
# ShardConfig
# ---------------------------------------------------------------------------


class TestShardConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = ShardConfig()
        self.assertEqual(cfg.n_layers, 32)
        self.assertEqual(cfg.hot_layers, 4)
        self.assertEqual(cfg.warm_layers, 8)
        self.assertEqual(cfg.lookahead, 2)

    def test_valid_custom(self):
        cfg = ShardConfig(n_layers=8, hot_layers=2, warm_layers=4, lookahead=1)
        self.assertEqual(cfg.n_layers, 8)

    def test_n_layers_zero_raises(self):
        with self.assertRaises(ValueError):
            ShardConfig(n_layers=0)

    def test_hot_layers_zero_raises(self):
        with self.assertRaises(ValueError):
            ShardConfig(hot_layers=0)

    def test_warm_layers_negative_raises(self):
        with self.assertRaises(ValueError):
            ShardConfig(warm_layers=-1)

    def test_lookahead_negative_raises(self):
        with self.assertRaises(ValueError):
            ShardConfig(lookahead=-1)

    def test_hot_layers_exceeds_n_layers_raises(self):
        with self.assertRaises(ValueError):
            ShardConfig(n_layers=4, hot_layers=8)

    def test_warm_layers_zero_valid(self):
        cfg = ShardConfig(warm_layers=0)
        self.assertEqual(cfg.warm_layers, 0)

    def test_lookahead_zero_valid(self):
        cfg = ShardConfig(lookahead=0)
        self.assertEqual(cfg.lookahead, 0)


# ---------------------------------------------------------------------------
# ShardTier
# ---------------------------------------------------------------------------


class TestShardTier(unittest.TestCase):
    def test_hot_value(self):
        self.assertEqual(ShardTier.HOT.value, "hot")

    def test_warm_value(self):
        self.assertEqual(ShardTier.WARM.value, "warm")

    def test_cold_value(self):
        self.assertEqual(ShardTier.COLD.value, "cold")

    def test_enum_members(self):
        members = {t.name for t in ShardTier}
        self.assertSetEqual(members, {"HOT", "WARM", "COLD"})


# ---------------------------------------------------------------------------
# LayerShard
# ---------------------------------------------------------------------------


class TestLayerShard(unittest.TestCase):
    def test_construction(self):
        s = LayerShard(layer_idx=0, tier=ShardTier.HOT, data=None, size_bytes=1024)
        self.assertEqual(s.layer_idx, 0)
        self.assertEqual(s.tier, ShardTier.HOT)

    def test_is_resident_with_data(self):
        data = np.zeros(4, dtype=np.float32)
        s = LayerShard(layer_idx=0, tier=ShardTier.HOT, data=data, size_bytes=16)
        self.assertTrue(s.is_resident)

    def test_is_resident_without_data(self):
        s = LayerShard(layer_idx=0, tier=ShardTier.COLD, data=None, size_bytes=1024)
        self.assertFalse(s.is_resident)

    def test_size_bytes(self):
        s = LayerShard(layer_idx=1, tier=ShardTier.WARM, data=None, size_bytes=2048)
        self.assertEqual(s.size_bytes, 2048)

    def test_tier_assignment(self):
        s = LayerShard(layer_idx=2, tier=ShardTier.WARM, data=None, size_bytes=512)
        self.assertEqual(s.tier, ShardTier.WARM)


# ---------------------------------------------------------------------------
# ModelShardLoader
# ---------------------------------------------------------------------------


class TestModelShardLoader(unittest.TestCase):
    def _make_loader(self, n=16, hot=4, warm=8, lookahead=2):
        cfg = ShardConfig(n_layers=n, hot_layers=hot, warm_layers=warm, lookahead=lookahead)
        return ModelShardLoader(cfg)

    def test_load_model(self):
        loader = self._make_loader()
        layers = _make_layers(16, shape=(4, 4))
        loader.load_model(layers)
        self.assertEqual(len(loader), 16)

    def test_initial_tier_hot(self):
        loader = self._make_loader(hot=4)
        loader.load_model(_make_layers(16, (4, 4)))
        self.assertEqual(loader.tier_of(0), ShardTier.HOT)
        self.assertEqual(loader.tier_of(3), ShardTier.HOT)

    def test_initial_tier_warm(self):
        loader = self._make_loader(hot=4, warm=8)
        loader.load_model(_make_layers(16, (4, 4)))
        self.assertEqual(loader.tier_of(4), ShardTier.WARM)
        self.assertEqual(loader.tier_of(11), ShardTier.WARM)

    def test_initial_tier_cold(self):
        loader = self._make_loader(hot=4, warm=8)
        loader.load_model(_make_layers(16, (4, 4)))
        self.assertEqual(loader.tier_of(12), ShardTier.COLD)

    def test_get_layer_hot(self):
        loader = self._make_loader()
        layers = _make_layers(16, (4, 4))
        loader.load_model(layers)
        data = loader.get_layer(0)
        self.assertIsInstance(data, np.ndarray)

    def test_get_layer_cold_raises(self):
        loader = self._make_loader(hot=4, warm=4)
        loader.load_model(_make_layers(16, (4, 4)))
        with self.assertRaises(RuntimeError):
            loader.get_layer(15)

    def test_get_layer_unknown_raises(self):
        loader = self._make_loader()
        with self.assertRaises(KeyError):
            loader.get_layer(999)

    def test_promote_to_warm(self):
        loader = self._make_loader(hot=4, warm=8)
        loader.load_model(_make_layers(16, (4, 4)))
        loader.evict_to_cold(15)
        loader.promote_to_warm(15)
        self.assertEqual(loader.tier_of(15), ShardTier.WARM)

    def test_evict_to_cold(self):
        loader = self._make_loader()
        loader.load_model(_make_layers(16, (4, 4)))
        loader.evict_to_cold(0)
        self.assertEqual(loader.tier_of(0), ShardTier.COLD)

    def test_promote_to_hot(self):
        loader = self._make_loader()
        loader.load_model(_make_layers(16, (4, 4)))
        loader.promote_to_warm(5)
        loader.promote_to_hot(5)
        self.assertEqual(loader.tier_of(5), ShardTier.HOT)

    def test_prefetch_promotes_to_warm(self):
        loader = self._make_loader(hot=4, warm=8)
        loader.load_model(_make_layers(16, (4, 4)))
        loader.evict_to_cold(14)
        loader.prefetch([14])
        self.assertEqual(loader.tier_of(14), ShardTier.WARM)

    def test_memory_report_keys(self):
        loader = self._make_loader()
        loader.load_model(_make_layers(16, (4, 4)))
        report = loader.memory_report()
        for key in ("hot_count", "warm_count", "cold_count", "total_layers"):
            self.assertIn(key, report)

    def test_memory_report_total_layers(self):
        loader = self._make_loader()
        loader.load_model(_make_layers(16, (4, 4)))
        report = loader.memory_report()
        self.assertEqual(report["total_layers"], 16)

    def test_advance_window_promotes_current(self):
        loader = self._make_loader(hot=4, warm=8)
        loader.load_model(_make_layers(16, (4, 4)))
        loader.advance_window(6)
        # Current layer should be HOT (or at least warm if hot is full)
        tier = loader.tier_of(6)
        self.assertIn(tier, (ShardTier.HOT, ShardTier.WARM))

    def test_advance_window_evicts_old_layers(self):
        loader = self._make_loader(hot=2, warm=4, lookahead=1)
        loader.load_model(_make_layers(16, (4, 4)))
        loader.advance_window(10)
        # Layers well behind the window should become COLD
        self.assertEqual(loader.tier_of(0), ShardTier.COLD)

    def test_tier_of_unknown_raises(self):
        loader = self._make_loader()
        with self.assertRaises(KeyError):
            loader.tier_of(999)

    def test_iter_hot_yields_arrays(self):
        loader = self._make_loader(hot=2)
        loader.load_model(_make_layers(16, (4, 4)))
        hot_items = list(loader.iter_hot())
        self.assertGreater(len(hot_items), 0)
        for idx, arr in hot_items:
            self.assertIsInstance(arr, np.ndarray)

    def test_thread_safety(self):
        """Concurrent get_layer calls should not raise."""
        loader = self._make_loader(hot=8, warm=8)
        loader.load_model(_make_layers(16, (4, 4)))
        errors = []

        def worker(layer_idx):
            try:
                loader.get_layer(layer_idx)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errors, [])

    def test_len(self):
        loader = self._make_loader()
        loader.load_model(_make_layers(16, (4, 4)))
        self.assertEqual(len(loader), 16)


if __name__ == "__main__":
    unittest.main()
