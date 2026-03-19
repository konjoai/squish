"""tests/test_compressed_loader_torch_unit.py

Unit tests for squish/compressed_loader_torch.py.

These tests mock out torch & transformers so they run in the MLX-only
CI environment (no CUDA required).
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_transformers_mock():
    """Minimal transformers mock."""
    mock = MagicMock(name="transformers")

    class _FakeModel:
        def __init__(self):
            self._state = {
                "model.embed_tokens.weight": _fake_tensor(np.ones((10, 16), dtype=np.float32)),
                "model.layers.0.mlp.gate_proj.weight": _fake_tensor(np.ones((8, 16), dtype=np.float32)),
            }

        def eval(self): return self

        def state_dict(self):
            return dict(self._state)

        def load_state_dict(self, new_state, strict=False):
            result = MagicMock()
            result.missing_keys = []
            return result

    class _fake_tensor:
        def __init__(self, arr): self._arr = arr

    mock.AutoModelForCausalLM.from_pretrained.return_value = _FakeModel()
    mock.AutoTokenizer.from_pretrained.return_value = MagicMock()
    return mock


def _make_torch_mock():
    mock = MagicMock(name="torch")
    mock.float32  = "float32"
    mock.float16  = "float16"
    mock.bfloat16 = "bfloat16"

    class _FakeTensor:
        def __init__(self, arr):
            self._arr = arr

        def to(self, device=None, dtype=None):
            return self

    mock.from_numpy.side_effect = lambda arr: _FakeTensor(np.ascontiguousarray(arr))
    mock.no_grad.return_value.__enter__ = lambda s: None
    mock.no_grad.return_value.__exit__ = MagicMock(return_value=False)
    return mock


def _build_npy_dir(tmp_path: Path, include_q4a: bool = True) -> Path:
    """Create a minimal npy-dir with tensor files for testing."""
    tensors = tmp_path / "tensors"
    tensors.mkdir()
    # manifest
    (tmp_path / "manifest.json").write_text(json.dumps({"version": 1}))

    if include_q4a:
        # model_layers_0_mlp_gate_proj_weight with asymmetric INT4
        n_rows, half_cols = 8, 8
        packed = np.zeros((n_rows, half_cols), dtype=np.uint8)
        scales = np.ones((n_rows, 4), dtype=np.float32)
        zeros  = np.zeros((n_rows, 4), dtype=np.float32)
        np.save(tensors / "model_layers_0_mlp_gate_proj_weight__q4a.npy", packed)
        np.save(tensors / "model_layers_0_mlp_gate_proj_weight__s4a.npy", scales)
        np.save(tensors / "model_layers_0_mlp_gate_proj_weight__z4a.npy", zeros)
    else:
        # passthrough tensor
        arr = np.ones((8, 16), dtype=np.float16)
        np.save(tensors / "model_layers_0_mlp_gate_proj_weight__pt.npy", arr)

    return tmp_path


# ---------------------------------------------------------------------------
# _collect_npy_keys
# ---------------------------------------------------------------------------

class TestCollectNpyKeys:
    def test_q4a_key_detected(self, tmp_path):
        tensors = tmp_path / "tensors"
        tensors.mkdir()
        np.save(tensors / "model_layers_0_self_attn_q_proj_weight__q4a.npy",
                np.zeros((4, 4), dtype=np.uint8))
        np.save(tensors / "model_layers_0_self_attn_q_proj_weight__s4a.npy",
                np.ones((4, 2), dtype=np.float32))
        np.save(tensors / "model_layers_0_self_attn_q_proj_weight__z4a.npy",
                np.zeros((4, 2), dtype=np.float32))

        from squish.compressed_loader_torch import _collect_npy_keys
        keys = _collect_npy_keys(tensors)
        assert "model_layers_0_self_attn_q_proj_weight" in keys

    def test_passthrough_key_detected(self, tmp_path):
        tensors = tmp_path / "tensors"
        tensors.mkdir()
        np.save(tensors / "model_embed_tokens_weight__pt.npy",
                np.ones((10, 16), dtype=np.float16))
        from squish.compressed_loader_torch import _collect_npy_keys
        keys = _collect_npy_keys(tensors)
        assert "model_embed_tokens_weight" in keys

    def test_empty_dir(self, tmp_path):
        tensors = tmp_path / "tensors"
        tensors.mkdir()
        from squish.compressed_loader_torch import _collect_npy_keys
        keys = _collect_npy_keys(tensors)
        assert len(keys) == 0

    def test_unrecognised_files_ignored(self, tmp_path):
        tensors = tmp_path / "tensors"
        tensors.mkdir()
        (tensors / "README.txt").write_text("hello")
        (tensors / "config.json").write_text("{}")
        from squish.compressed_loader_torch import _collect_npy_keys
        keys = _collect_npy_keys(tensors)
        assert len(keys) == 0


# ---------------------------------------------------------------------------
# load_compressed_model_torch
# ---------------------------------------------------------------------------

class TestLoadCompressedModelTorch:
    def _patch_and_load(self, tmp_path, include_q4a=True, verbose=False):
        npy_dir = _build_npy_dir(tmp_path, include_q4a=include_q4a)
        mock_torch       = _make_torch_mock()
        mock_transformers = _make_transformers_mock()

        with patch.dict(sys.modules, {
            "torch": mock_torch,
            "transformers": mock_transformers,
        }):
            import importlib
            import squish.compressed_loader_torch as mod
            importlib.reload(mod)
            model, tokenizer = mod.load_compressed_model_torch(
                npy_dir   = str(npy_dir),
                model_dir = str(tmp_path / "fake_model"),
                device    = "cpu",
                verbose   = verbose,
            )
        return model, tokenizer

    def test_returns_model_and_tokenizer(self, tmp_path):
        model, tokenizer = self._patch_and_load(tmp_path)
        assert model is not None
        assert tokenizer is not None

    def test_accepts_tensors_subdir_directly(self, tmp_path):
        """Passing the tensors/ dir directly also works."""
        npy_dir = _build_npy_dir(tmp_path)
        tensor_dir = npy_dir / "tensors"
        mock_torch        = _make_torch_mock()
        mock_transformers = _make_transformers_mock()

        with patch.dict(sys.modules, {
            "torch": mock_torch,
            "transformers": mock_transformers,
        }):
            import importlib
            import squish.compressed_loader_torch as mod
            importlib.reload(mod)
            model, tokenizer = mod.load_compressed_model_torch(
                npy_dir   = str(tensor_dir),
                model_dir = str(tmp_path / "fake_model"),
                device    = "cpu",
                verbose   = False,
            )
        assert model is not None

    def test_missing_tensors_dir_raises(self, tmp_path):
        mock_torch        = _make_torch_mock()
        mock_transformers = _make_transformers_mock()
        with patch.dict(sys.modules, {
            "torch": mock_torch,
            "transformers": mock_transformers,
        }):
            import importlib
            import squish.compressed_loader_torch as mod
            importlib.reload(mod)
            with pytest.raises(FileNotFoundError):
                mod.load_compressed_model_torch(
                    npy_dir   = str(tmp_path / "nonexistent"),
                    model_dir = str(tmp_path),
                    device    = "cpu",
                    verbose   = False,
                )

    def test_passthrough_tensors_loaded(self, tmp_path):
        """Test with passthrough FP16 tensors instead of INT4."""
        model, tokenizer = self._patch_and_load(tmp_path, include_q4a=False)
        assert model is not None

    def test_verbose_output(self, tmp_path, capsys):
        self._patch_and_load(tmp_path, verbose=True)
        captured = capsys.readouterr()
        assert "compressed_loader_torch" in captured.out
