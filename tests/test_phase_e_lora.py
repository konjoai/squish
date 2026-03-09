"""
tests/test_phase_e_lora.py

Coverage tests for squish/lora_manager.py — Phase E1 (LoRA adapter management).

All tests run without safetensors, MLX, or a real model installed.
"""

from __future__ import annotations

import json
import threading
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from squish.lora_manager import LoRAManager

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_config(tmp_path: Path, rank=8, alpha=16.0, target_modules=None) -> Path:
    """Write a minimal adapter_config.json and return the directory Path."""
    if target_modules is None:
        target_modules = ["q_proj", "v_proj"]
    cfg = {"r": rank, "lora_alpha": alpha, "target_modules": target_modules}
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    (adapter_dir / "adapter_config.json").write_text(json.dumps(cfg))
    return adapter_dir


def _make_safetensors(adapter_dir: Path) -> Path:
    """Write a fake *.safetensors stub (empty file) and return path."""
    st = adapter_dir / "adapter_model.safetensors"
    st.write_bytes(b"fake")
    return st


def _make_model(**attrs):
    """Return a SimpleNamespace-based mock model with settable attributes."""
    return types.SimpleNamespace(**attrs)


# ── Registry ─────────────────────────────────────────────────────────────────


class TestLoRAManagerRegistry:
    def test_register_and_is_registered(self, tmp_path):
        mgr = LoRAManager()
        assert not mgr.is_registered("legal")
        mgr.register("legal", str(tmp_path))
        assert mgr.is_registered("legal")

    def test_registered_domains_sorted(self, tmp_path):
        mgr = LoRAManager()
        mgr.register("code", tmp_path / "c")
        mgr.register("legal", tmp_path / "l")
        mgr.register("medical", tmp_path / "m")
        assert mgr.registered_domains() == ["code", "legal", "medical"]

    def test_registered_domains_empty(self):
        mgr = LoRAManager()
        assert mgr.registered_domains() == []

    def test_register_path_object(self, tmp_path):
        mgr = LoRAManager()
        mgr.register("x", tmp_path)
        assert mgr.is_registered("x")


# ── Config ────────────────────────────────────────────────────────────────────


class TestLoRAManagerConfig:
    def test_read_adapter_config_basic(self, tmp_path):
        adapter_dir = _make_config(tmp_path, rank=16, alpha=32.0)
        mgr = LoRAManager()
        mgr.register("d", adapter_dir)
        cfg = mgr._read_adapter_config("d")
        assert cfg["r"] == 16
        assert cfg["lora_alpha"] == 32.0

    def test_read_adapter_config_cached(self, tmp_path):
        adapter_dir = _make_config(tmp_path, rank=4)
        mgr = LoRAManager()
        mgr.register("d", adapter_dir)
        # First call reads JSON; second should return cached copy
        cfg1 = mgr._read_adapter_config("d")
        cfg2 = mgr._read_adapter_config("d")
        assert cfg1 is cfg2  # same object from cache

    def test_read_adapter_config_unregistered(self):
        mgr = LoRAManager()
        with pytest.raises(KeyError, match="Domain not registered"):
            mgr._read_adapter_config("ghost")

    def test_read_adapter_config_missing_file(self, tmp_path):
        mgr = LoRAManager()
        mgr.register("d", tmp_path / "nonexistent")
        with pytest.raises(FileNotFoundError):
            mgr._read_adapter_config("d")

    def test_invalidate_config_cache(self, tmp_path):
        adapter_dir = _make_config(tmp_path, rank=4)
        mgr = LoRAManager()
        mgr.register("d", adapter_dir)
        mgr._read_adapter_config("d")
        assert "d" in mgr._config_cache
        mgr.invalidate_config_cache("d")
        assert "d" not in mgr._config_cache

    def test_invalidate_config_cache_noop_if_missing(self):
        mgr = LoRAManager()
        # Should not raise
        mgr.invalidate_config_cache("ghost")


# ── Loading ───────────────────────────────────────────────────────────────────


class TestLoRAManagerLoad:
    def test_load_unregistered_raises(self):
        mgr = LoRAManager()
        with pytest.raises(KeyError, match="Domain not registered"):
            mgr.load("ghost")

    def test_load_safetensors_import_error(self, tmp_path):
        """When safetensors is not installed, _load_safetensors raises ImportError."""
        adapter_dir = _make_config(tmp_path, rank=4)
        mgr = LoRAManager()
        mgr.register("d", adapter_dir)
        with patch.dict("sys.modules", {"safetensors": None}):
            with pytest.raises(ImportError, match="safetensors package required"):
                mgr._load_safetensors(adapter_dir)

    def test_load_safetensors_success(self, tmp_path):
        """Happy-path: safetensors.safe_open is available and returns weights."""
        adapter_dir = _make_config(tmp_path)
        _make_safetensors(adapter_dir)

        dummy_weight = np.ones((4, 4), dtype=np.float32)

        mock_fh = MagicMock()
        mock_fh.__enter__ = lambda s: s
        mock_fh.__exit__ = MagicMock(return_value=False)
        mock_fh.keys.return_value = ["layer.weight"]
        mock_fh.get_tensor.return_value = dummy_weight

        mock_safe_open = MagicMock(return_value=mock_fh)
        mock_safetensors = MagicMock()
        mock_safetensors.safe_open = mock_safe_open

        with patch.dict("sys.modules", {"safetensors": mock_safetensors}):
            weights = mgr = LoRAManager()
            weights = mgr._load_safetensors(adapter_dir)

        assert "layer.weight" in weights
        assert (weights["layer.weight"] == dummy_weight).all()

    def test_load_caches_result(self, tmp_path):
        """Second load() call returns same dict from cache (no extra io)."""
        adapter_dir = _make_config(tmp_path, rank=4)
        mgr = LoRAManager()
        mgr.register("d", adapter_dir)

        dummy_weights = {"k": np.zeros((2, 2))}
        with patch.object(mgr, "_load_safetensors", return_value=dummy_weights):
            w1 = mgr.load("d")
            w2 = mgr.load("d")
        assert w1 is w2

    def test_load_lru_eviction(self, tmp_path):
        """When cache capacity is reached, the oldest entry is evicted."""
        mgr = LoRAManager(max_cache_size=2)
        for i in range(3):
            d = tmp_path / f"a{i}"
            d.mkdir()
            (d / "adapter_config.json").write_text(json.dumps({"r": 4}))
            mgr.register(f"d{i}", d)

        dummy = {"k": np.zeros((1,))}
        with patch.object(mgr, "_load_safetensors", return_value=dummy):
            mgr.load("d0")
            mgr.load("d1")
            assert mgr.cache_size() == 2
            mgr.load("d2")  # evicts d0
            assert mgr.cache_size() == 2
            assert "d0" not in mgr._cache
            assert "d1" in mgr._cache
            assert "d2" in mgr._cache

    def test_evict_cached(self, tmp_path):
        adapter_dir = _make_config(tmp_path)
        mgr = LoRAManager()
        mgr.register("d", adapter_dir)
        dummy = {"k": np.zeros((1,))}
        with patch.object(mgr, "_load_safetensors", return_value=dummy):
            mgr.load("d")
        assert mgr.cache_size() == 1
        removed = mgr.evict("d")
        assert removed is True
        assert mgr.cache_size() == 0

    def test_evict_not_cached(self):
        mgr = LoRAManager()
        removed = mgr.evict("ghost")
        assert removed is False

    def test_cache_size(self, tmp_path):
        mgr = LoRAManager(max_cache_size=10)
        for i in range(3):
            d = tmp_path / f"b{i}"
            d.mkdir()
            (d / "adapter_config.json").write_text(json.dumps({"r": 4}))
            mgr.register(f"d{i}", d)
        dummy = {"k": np.zeros((1,))}
        with patch.object(mgr, "_load_safetensors", return_value=dummy):
            for i in range(3):
                mgr.load(f"d{i}")
        assert mgr.cache_size() == 3


# ── Apply / Unapply ───────────────────────────────────────────────────────────


class TestLoRAManagerApplyUnapply:
    def _setup(self, tmp_path):
        adapter_dir = _make_config(tmp_path, rank=4, alpha=8.0)
        mgr = LoRAManager()
        mgr.register("d", adapter_dir)
        return mgr, adapter_dir

    def test_apply_basic(self, tmp_path):
        """apply() patches model attributes with scaled delta weights."""
        mgr, adapter_dir = self._setup(tmp_path)

        orig_w = np.ones((2, 2), dtype=np.float32)
        model = _make_model(layer=types.SimpleNamespace(weight=orig_w.copy()))

        weights = {"layer.weight": np.full((2, 2), 0.1, dtype=np.float32)}
        with patch.object(mgr, "_load_safetensors", return_value=weights):
            mgr.apply(model, "d")

        # scale = alpha/rank = 8/4 = 2.0
        expected = orig_w + 2.0 * weights["layer.weight"]
        assert np.allclose(model.layer.weight, expected)

    def test_unapply_restores(self, tmp_path):
        """unapply() restores the original weights after apply()."""
        mgr, adapter_dir = self._setup(tmp_path)

        orig_w = np.ones((2, 2), dtype=np.float32)
        model = _make_model(layer=types.SimpleNamespace(weight=orig_w.copy()))

        weights = {"layer.weight": np.full((2, 2), 0.5, dtype=np.float32)}
        with patch.object(mgr, "_load_safetensors", return_value=weights):
            mgr.apply(model, "d")

        mgr.unapply(model)
        assert np.allclose(model.layer.weight, orig_w)
        assert mgr._patched_layers == []
        assert mgr._original_weights == {}

    def test_apply_missing_layer_skipped(self, tmp_path):
        """apply() silently skips layers not present on the model."""
        mgr, _ad = self._setup(tmp_path)
        model = _make_model()  # no attributes

        weights = {"ghost.weight": np.ones((2, 2), dtype=np.float32)}
        with patch.object(mgr, "_load_safetensors", return_value=weights):
            mgr.apply(model, "d")  # should not raise

        assert mgr._patched_layers == []

    def test_apply_none_param_skipped(self, tmp_path):
        """apply() skips layers where the param value is None."""
        mgr, _ad = self._setup(tmp_path)
        model = _make_model(layer=types.SimpleNamespace(weight=None))

        weights = {"layer.weight": np.ones((2, 2), dtype=np.float32)}
        with patch.object(mgr, "_load_safetensors", return_value=weights):
            mgr.apply(model, "d")

        # weight stayed None, nothing patched
        assert mgr._patched_layers == []

    def test_unapply_missing_orig_continue(self, tmp_path):
        """unapply() skips layers whose original weight is not in ~_original_weights."""
        adapter_dir = _make_config(tmp_path, rank=2, alpha=2.0)
        mgr = LoRAManager()
        mgr.register("d", adapter_dir)
        model = _make_model(layer=types.SimpleNamespace(weight=np.zeros((3,))))

        # Manually set _patched_layers to a key not in _original_weights
        mgr._patched_layers = ["layer.weight"]
        mgr._original_weights = {}  # empty → .get returns None
        # Should not raise, should silently skip
        mgr.unapply(model)
        assert mgr._patched_layers == []

    def test_unapply_obj_none_continue(self, tmp_path):
        """unapply() skips layers where _resolve_param returns None."""
        adapter_dir = _make_config(tmp_path, rank=2, alpha=2.0)
        mgr = LoRAManager()
        mgr.register("d", adapter_dir)
        orig_w = np.ones((2,))
        model = _make_model()  # no 'ghost' attribute

        # layer was applied on a model with 'ghost.weight', now unapply on model without it
        mgr._patched_layers = ["ghost.weight"]
        mgr._original_weights = {"ghost.weight": orig_w}
        mgr.unapply(model)  # resolve_param returns (None, "") → continue
        assert mgr._patched_layers == []

    def test_apply_is_thread_safe(self, tmp_path):
        """apply() and unapply() can be called from multiple threads without error."""
        adapter_dir = _make_config(tmp_path, rank=2, alpha=2.0)
        mgr = LoRAManager()
        mgr.register("d", adapter_dir)

        orig_w = np.zeros((4, 4), dtype=np.float32)
        model = _make_model(layer=types.SimpleNamespace(weight=orig_w.copy()))

        weights = {"layer.weight": np.ones((4, 4), dtype=np.float32)}
        errors = []

        def worker():
            try:
                with patch.object(mgr, "_load_safetensors", return_value=weights):
                    mgr.apply(model, "d")
                mgr.unapply(model)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Thread errors: {errors}"


# ── _resolve_param ────────────────────────────────────────────────────────────


class TestResolveParam:
    def test_top_level_attr(self):
        mgr = LoRAManager()
        model = _make_model(weight=np.ones((3,)))
        obj, attr = mgr._resolve_param(model, "weight")
        assert obj is model
        assert attr == "weight"

    def test_nested_attr(self):
        mgr = LoRAManager()
        inner = types.SimpleNamespace(w=np.zeros((2,)))
        model = _make_model(layer=inner)
        obj, attr = mgr._resolve_param(model, "layer.w")
        assert obj is inner
        assert attr == "w"

    def test_missing_intermediate(self):
        mgr = LoRAManager()
        model = _make_model()
        obj, attr = mgr._resolve_param(model, "missing.w")
        assert obj is None
        assert attr == ""


# ── adapter_info ──────────────────────────────────────────────────────────────


class TestAdapterInfo:
    def test_adapter_info_with_cached_weights(self, tmp_path):
        adapter_dir = _make_config(tmp_path, rank=8, alpha=16.0, target_modules=["q"])
        _make_safetensors(adapter_dir)
        mgr = LoRAManager()
        mgr.register("d", adapter_dir)

        # Populate cache with known weights
        arr = np.ones((4, 4), dtype=np.float32)
        mgr._cache["d"] = {"layer.weight": arr}

        info = mgr.adapter_info("d")
        assert info["rank"] == 8
        assert info["alpha"] == 16.0
        assert info["target_modules"] == ["q"]
        assert info["total_params"] == 16
        assert info["size_mb"] >= 0

    def test_adapter_info_no_cache(self, tmp_path):
        """adapter_info works even when no weights are in cache."""
        adapter_dir = _make_config(tmp_path, rank=4, alpha=4.0)
        mgr = LoRAManager()
        mgr.register("d", adapter_dir)
        info = mgr.adapter_info("d")
        assert info["total_params"] == 0

    def test_adapter_info_unregistered_raises(self):
        mgr = LoRAManager()
        with pytest.raises(KeyError):
            mgr.adapter_info("ghost")

    def test_adapter_info_defaults_when_keys_missing(self, tmp_path):
        """Config without r/lora_alpha/target_modules uses defaults."""
        adapter_dir = tmp_path / "ad"
        adapter_dir.mkdir()
        (adapter_dir / "adapter_config.json").write_text("{}")
        mgr = LoRAManager()
        mgr.register("d", adapter_dir)
        info = mgr.adapter_info("d")
        assert info["rank"] == 8
        assert info["alpha"] == 8.0
        assert info["target_modules"] == []

    def test_adapter_info_none_adapter_path(self, tmp_path):
        """adapter_info gracefully handles a None registry entry (skips file size loop)."""
        adapter_dir = _make_config(tmp_path, rank=4, alpha=4.0)
        mgr = LoRAManager()
        mgr.register("d", adapter_dir)
        # Force adapter_path to None by overriding the registry entry
        mgr._registry["d"] = None  # type: ignore[assignment]
        # _read_adapter_config already cached so it won't re-read from path
        mgr._config_cache["d"] = {"r": 4, "lora_alpha": 4.0, "target_modules": []}
        info = mgr.adapter_info("d")
        assert info["size_mb"] == 0.0
