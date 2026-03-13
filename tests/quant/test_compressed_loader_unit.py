"""
tests/test_compressed_loader_unit.py

Unit tests for the pure-Python / file-I/O helpers in squish/compressed_loader.py.
Does NOT require MLX, model files, or hardware.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from squish.quant.compressed_loader import (
    _collect_tensor_keys,
    _dequantize,
    _get_auto_tokenizer,
    _get_zstd_dctx,
    _load_npy_path,
    _npy_exists,
    _rss_mb,
    _rss_mb_throttled,
    _safe_key_to_original,
    _tensor_load_key,
    _unique_base_keys,
)

# ── _get_zstd_dctx ────────────────────────────────────────────────────────────

class TestGetZstdDctx:
    def test_returns_dctx_or_none(self):
        result = _get_zstd_dctx()
        # Either a ZstdDecompressor or None (if zstandard not installed)
        assert result is None or hasattr(result, "decompress")

    def test_returns_same_object_on_repeated_calls(self):
        """Should be lazy-singleton — two calls return the same object."""
        r1 = _get_zstd_dctx()
        r2 = _get_zstd_dctx()
        assert r1 is r2


# ── _load_npy_path ────────────────────────────────────────────────────────────

class TestLoadNpyPath:
    def test_loads_existing_npy(self, tmp_path):
        arr = np.arange(20, dtype=np.float32)
        p = tmp_path / "test.npy"
        np.save(str(p), arr)
        loaded = _load_npy_path(p)
        np.testing.assert_array_equal(loaded, arr)

    def test_raises_file_not_found_for_missing(self, tmp_path):
        p = tmp_path / "nonexistent.npy"
        with pytest.raises(FileNotFoundError):
            _load_npy_path(p)

    def test_falls_back_to_zst(self, tmp_path):
        """Verify _npy_exists reports True when only .npy.zst exists (loader path)."""
        npy_path = tmp_path / "weights.npy"
        zst_path = Path(str(npy_path) + ".zst")
        zst_path.write_bytes(b"\x00")  # just needs to exist
        # The zst path is present — confirm npy_exists detects it
        assert _npy_exists(npy_path) is True
        # The .npy itself is absent
        assert not npy_path.exists()

    def test_mmap_mode_accepted(self, tmp_path):
        arr = np.arange(100, dtype=np.float32)
        p = tmp_path / "mmap.npy"
        np.save(str(p), arr)
        loaded = _load_npy_path(p, mmap_mode=None)
        np.testing.assert_array_equal(loaded, arr)


# ── _rss_mb ────────────────────────────────────────────────────────────────────

class TestRssMb:
    def test_returns_positive_float(self):
        result = _rss_mb()
        assert isinstance(result, float)
        assert result > 0

    def test_reasonable_range(self):
        """A Python process should use more than 1 MB and less than 100 GB."""
        result = _rss_mb()
        assert 1.0 < result < 100_000.0


# ── _rss_mb_throttled ──────────────────────────────────────────────────────────

class TestRssMbThrottled:
    def test_returns_positive_float(self):
        result = _rss_mb_throttled(interval=1.0)
        assert isinstance(result, float)
        assert result > 0

    def test_same_within_interval(self):
        v1 = _rss_mb_throttled(interval=5.0)
        v2 = _rss_mb_throttled(interval=5.0)
        assert v1 == v2

    def test_updates_when_interval_zero(self):
        """With interval=0, always refreshes."""
        v1 = _rss_mb_throttled(interval=0.0)
        v2 = _rss_mb_throttled(interval=0.0)
        # Both should be valid positive floats
        assert v1 > 0 and v2 > 0


# ── _safe_key_to_original ──────────────────────────────────────────────────────

class TestSafeKeyToOriginal:
    def test_roundtrip(self, tmp_path):
        manifest = {
            "model.embed_tokens.weight": "model__embed_tokens__weight",
            "lm_head.weight":            "lm_head__weight",
        }
        p = tmp_path / "manifest.json"
        p.write_text(json.dumps(manifest))
        result = _safe_key_to_original(str(p))
        # inverted: safe_key → original
        assert result["model__embed_tokens__weight"] == "model.embed_tokens.weight"
        assert result["lm_head__weight"] == "lm_head.weight"

    def test_empty_manifest(self, tmp_path):
        p = tmp_path / "empty.json"
        p.write_text("{}")
        result = _safe_key_to_original(str(p))
        assert result == {}

    def test_raises_file_not_found(self, tmp_path):
        with pytest.raises((FileNotFoundError, OSError)):
            _safe_key_to_original(str(tmp_path / "missing.json"))


# ── _unique_base_keys ──────────────────────────────────────────────────────────

class TestUniqueBaseKeys:
    def test_extracts_q_suffix(self):
        files = ["layer_0__q", "layer_0__s", "layer_0__shape"]
        result = _unique_base_keys(files)
        assert "layer_0" in result

    def test_extracts_pt_suffix(self):
        files = ["embed__pt", "embed__shape"]
        result = _unique_base_keys(files)
        assert "embed" in result

    def test_ignores_non_matching(self):
        files = ["readme.txt", "some_random_key", "config"]
        result = _unique_base_keys(files)
        assert len(result) == 0

    def test_unique_keys_only(self):
        files = ["key1__q", "key1__s", "key1__shape", "key2__q", "key2__s"]
        result = _unique_base_keys(files)
        assert result == {"key1", "key2"}

    def test_empty_list(self):
        result = _unique_base_keys([])
        assert result == set()


# ── _npy_exists ────────────────────────────────────────────────────────────────

class TestNpyExists:
    def test_true_for_existing_npy(self, tmp_path):
        p = tmp_path / "arr.npy"
        np.save(str(p), np.array([1.0]))
        assert _npy_exists(p) is True

    def test_true_for_existing_zst(self, tmp_path):
        p = tmp_path / "arr.npy"
        # Only create the .npy.zst version
        zst = Path(str(p) + ".zst")
        zst.write_bytes(b"\x00")  # just needs to exist
        assert _npy_exists(p) is True

    def test_false_for_missing(self, tmp_path):
        p = tmp_path / "ghost.npy"
        assert _npy_exists(p) is False


# ── _dequantize ────────────────────────────────────────────────────────────────

class TestDequantize:
    def _make_npz(self, tmp_path: Path, sk: str, arr: np.ndarray) -> object:
        """Write a quantized npz record and return the opened NpzFile."""
        from squish.quant.quantizer import quantize_embeddings

        # Use asymmetric quantization so we get __s and __q keys
        two_d = arr.reshape(1, -1) if arr.ndim == 1 else arr
        r = quantize_embeddings(two_d, group_size=two_d.shape[1])
        original_shape = two_d.shape

        data = {
            sk + "__q":     r.quantized,
            sk + "__s":     r.scales,
            sk + "__shape": np.array(original_shape, dtype=np.int64),
        }
        path = tmp_path / "weights.npz"
        np.savez(str(path), **data)
        return np.load(str(path))

    def _make_pt_npz(self, tmp_path: Path, sk: str, arr: np.ndarray) -> object:
        """Write a passthrough (float) npz record."""
        path = tmp_path / "pt_weights.npz"
        data = {
            sk + "__pt":    arr,
            sk + "__shape": np.array(arr.shape, dtype=np.int64),
        }
        np.savez(str(path), **data)
        return np.load(str(path))

    def test_reconstructs_passthrough_array(self, tmp_path):
        sk = "model__embed"
        arr = np.random.default_rng(5).standard_normal((4, 32)).astype(np.float32)
        npz = self._make_pt_npz(tmp_path, sk, arr)
        result = _dequantize(npz, sk)
        assert result.shape == (4, 32)
        np.testing.assert_allclose(result, arr, rtol=1e-5)

    def test_quantized_reconstruct_shape_preserved(self, tmp_path):
        sk = "layer_weight"
        arr = np.random.default_rng(7).standard_normal((8, 64)).astype(np.float32)
        npz = self._make_npz(tmp_path, sk, arr)
        result = _dequantize(npz, sk)
        assert result.shape == (8, 64)

    def test_quantized_accuracy(self, tmp_path):
        sk = "attn_weight"
        arr = np.random.default_rng(0).standard_normal((4, 64)).astype(np.float32)
        npz = self._make_npz(tmp_path, sk, arr)
        result = _dequantize(npz, sk)
        from squish.quant.quantizer import mean_cosine_similarity
        assert mean_cosine_similarity(arr, result) > 0.95

    def test_pt_without_shape_key(self, tmp_path: Path):
        """__pt key present but no __shape: original_shape = pt.shape (line 271)."""
        path = tmp_path / "no_shape.npz"
        sk = "model.weight"
        arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        np.savez(str(path), **{f"{sk}__pt": arr})  # no __shape key
        npz = np.load(str(path))
        result = _dequantize(npz, sk)
        assert result.shape == (2, 2)
        np.testing.assert_allclose(result, arr, rtol=1e-5)


# ── _collect_tensor_keys ───────────────────────────────────────────────────────

class TestCollectTensorKeys:
    def _make_tensor_dir(self, tmp_path: Path, filenames: list[str]) -> Path:
        d = tmp_path / "tensors"
        d.mkdir()
        for name in filenames:
            (d / name).write_bytes(b"")
        return d

    def test_discovers_q_suffix(self, tmp_path):
        d = self._make_tensor_dir(tmp_path, ["layer_0__q.npy", "layer_0__s.npy"])
        keys = _collect_tensor_keys(d)
        assert "layer_0" in keys

    def test_discovers_pt_suffix(self, tmp_path):
        d = self._make_tensor_dir(tmp_path, ["embed__pt.npy"])
        keys = _collect_tensor_keys(d)
        assert "embed" in keys

    def test_discovers_shape_suffix(self, tmp_path):
        d = self._make_tensor_dir(tmp_path, ["layer_1__shape.npy"])
        keys = _collect_tensor_keys(d)
        assert "layer_1" in keys

    def test_discovers_zst_suffix(self, tmp_path):
        d = self._make_tensor_dir(tmp_path, ["layer_2__q.npy.zst", "layer_2__s.npy.zst"])
        keys = _collect_tensor_keys(d)
        assert "layer_2" in keys

    def test_ignores_non_tensor_files(self, tmp_path):
        d = self._make_tensor_dir(tmp_path, ["manifest.json", "config.txt", "README.md"])
        keys = _collect_tensor_keys(d)
        assert len(keys) == 0

    def test_unique_keys_only(self, tmp_path):
        d = self._make_tensor_dir(tmp_path, [
            "attn__q.npy", "attn__s.npy", "attn__shape.npy",
            "mlp__q.npy",  "mlp__s.npy",
        ])
        keys = _collect_tensor_keys(d)
        assert keys == {"attn", "mlp"}

    def test_mixed_zst_and_plain(self, tmp_path):
        d = self._make_tensor_dir(tmp_path, [
            "layer_3__q.npy",
            "layer_4__q.npy.zst",
        ])
        keys = _collect_tensor_keys(d)
        assert "layer_3" in keys
        assert "layer_4" in keys

    def test_empty_directory(self, tmp_path):
        d = self._make_tensor_dir(tmp_path, [])
        keys = _collect_tensor_keys(d)
        assert keys == set()

    def test_subdirectories_ignored(self, tmp_path):
        d = tmp_path / "tensors"
        d.mkdir()
        (d / "subdir").mkdir()
        # scandir should not yield the subdirectory as a tensor key
        keys = _collect_tensor_keys(d)
        assert len(keys) == 0


# ── _tensor_load_key ──────────────────────────────────────────────────────────

class TestTensorLoadKey:
    def test_attention_before_mlp(self):
        attn_key = _tensor_load_key("layers_0__self_attn__q_proj")
        mlp_key  = _tensor_load_key("layers_0__mlp__gate_proj")
        assert attn_key < mlp_key

    def test_mlp_before_embed(self):
        mlp_key   = _tensor_load_key("layers_0__mlp__up_proj")
        embed_key = _tensor_load_key("embed_tokens__weight")
        assert mlp_key < embed_key

    def test_same_group_sorted_by_layer(self):
        k0 = _tensor_load_key("layers_0__self_attn__q_proj")
        k5 = _tensor_load_key("layers_5__self_attn__q_proj")
        assert k0 < k5

    def test_returns_tuple(self):
        key = _tensor_load_key("layers_0__self_attn__q_proj")
        assert isinstance(key, tuple)
        assert len(key) == 3

    def test_other_group_for_norm(self):
        # Layer norms should be in group 2 (not attn=0, mlp=1, embed=3)
        group, _, _ = _tensor_load_key("layers_0__input_layernorm")
        assert group == 2

    def test_embed_group_is_last(self):
        embed_group, _, _ = _tensor_load_key("model__embed_tokens__weight")
        attn_group, _, _  = _tensor_load_key("model__layers_0__self_attn__q_proj")
        mlp_group, _, _   = _tensor_load_key("model__layers_0__mlp__gate_proj")
        assert embed_group > attn_group
        assert embed_group > mlp_group

    def test_q_proj_in_attention_group(self):
        group, _, _ = _tensor_load_key("layers_7__q_proj")
        assert group == 0

    def test_k_proj_in_attention_group(self):
        group, _, _ = _tensor_load_key("layers_7__k_proj")
        assert group == 0

    def test_v_proj_in_attention_group(self):
        group, _, _ = _tensor_load_key("layers_7__v_proj")
        assert group == 0

    def test_o_proj_in_attention_group(self):
        group, _, _ = _tensor_load_key("layers_7__o_proj")
        assert group == 0

    def test_gate_proj_in_mlp_group(self):
        group, _, _ = _tensor_load_key("layers_7__gate_proj")
        assert group == 1

    def test_down_proj_in_mlp_group(self):
        group, _, _ = _tensor_load_key("layers_7__down_proj")
        assert group == 1

    def test_up_proj_in_mlp_group(self):
        group, _, _ = _tensor_load_key("layers_7__up_proj")
        assert group == 1

    def test_no_layer_number_uses_9999(self):
        # lm_head has no layer number; should sort after all numbered layers
        no_layer_group, no_layer_num, _ = _tensor_load_key("lm_head__weight")
        with_layer_group, with_layer_num, _ = _tensor_load_key("layers_99__mlp__gate_proj")
        assert no_layer_num > with_layer_num  # both use layer 9999 or large number


# ── _get_auto_tokenizer ────────────────────────────────────────────────────────

class TestGetAutoTokenizer:
    def test_returns_auto_tokenizer_class(self):
        """_get_auto_tokenizer() should return the AutoTokenizer class (lazy import)."""
        try:
            cls = _get_auto_tokenizer()
            assert hasattr(cls, "from_pretrained")
        except ImportError:
            pytest.skip("transformers not installed")
