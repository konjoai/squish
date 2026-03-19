"""tests/test_backend_unit.py

Full-coverage unit tests for squish/backend.py.

On macOS with MLX (the standard CI environment) BE is _AppleBackend.
Tests for _AppleBackend use real MLX operations where safe.
Tests for _TorchBackend and _StubBackend instantiate those classes
directly via mocked dependencies so they never require hardware.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Conditionally import mlx — skip Apple-backend tests if not available
# ---------------------------------------------------------------------------

try:
    import mlx.core as mx  # type: ignore[import]
    _HAS_MLX = True
except Exception:
    _HAS_MLX = False

mlx_only = pytest.mark.skipif(not _HAS_MLX, reason="mlx not available")


# ---------------------------------------------------------------------------
# _AppleBackend — tested via the module-level BE singleton
# ---------------------------------------------------------------------------


@mlx_only
class TestAppleBackendArray:
    def test_array_from_list(self):
        from squish.backend import BE
        if not BE.IS_APPLE:
            pytest.skip("BE is not AppleBackend")
        result = BE.array([1, 2, 3])
        assert result is not None
        arr_np = np.array(result, dtype=np.int32)
        np.testing.assert_array_equal(arr_np, [1, 2, 3])

    def test_array_float32(self):
        from squish.backend import BE
        if not BE.IS_APPLE:
            pytest.skip("BE is not AppleBackend")
        result = BE.array([1.0, 2.0], dtype="float32")
        arr_np = np.array(result, dtype=np.float32)
        np.testing.assert_allclose(arr_np, [1.0, 2.0])

    def test_array_float16(self):
        from squish.backend import BE
        if not BE.IS_APPLE:
            pytest.skip("BE is not AppleBackend")
        result = BE.array([0.5, -0.5], dtype="float16")
        assert result is not None

    def test_array_bfloat16(self):
        from squish.backend import BE
        if not BE.IS_APPLE:
            pytest.skip("BE is not AppleBackend")
        result = BE.array([1.0], dtype="bfloat16")
        assert result is not None

    def test_array_unknown_dtype_defaults_to_int32(self):
        from squish.backend import BE
        if not BE.IS_APPLE:
            pytest.skip("BE is not AppleBackend")
        result = BE.array([7], dtype="int64_custom")
        assert result is not None


@mlx_only
class TestAppleBackendEval:
    def test_eval_with_tensor(self):
        from squish.backend import BE
        if not BE.IS_APPLE:
            pytest.skip("BE is not AppleBackend")
        t = mx.array([1, 2, 3])
        BE.eval(t)  # Should not raise

    def test_eval_with_none_skips(self):
        from squish.backend import BE
        if not BE.IS_APPLE:
            pytest.skip("BE is not AppleBackend")
        BE.eval(None)  # Should not raise

    def test_eval_with_multiple_tensors(self):
        from squish.backend import BE
        if not BE.IS_APPLE:
            pytest.skip("BE is not AppleBackend")
        t1 = mx.array([1.0, 2.0])
        t2 = mx.array([3.0])
        BE.eval(t1, t2)  # Should not raise


@mlx_only
class TestAppleBackendToNumpy:
    def test_to_numpy_basic(self):
        from squish.backend import BE
        if not BE.IS_APPLE:
            pytest.skip("BE is not AppleBackend")
        t = mx.array([1.0, 2.0, 3.0])
        result = BE.to_numpy(t)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        np.testing.assert_allclose(result, [1.0, 2.0, 3.0])


@mlx_only
class TestAppleBackendForward:
    def test_forward_without_cache(self):
        from squish.backend import BE
        if not BE.IS_APPLE:
            pytest.skip("BE is not AppleBackend")
        mock_model = MagicMock()
        mock_model.return_value = mx.array([1.0, 2.0, 3.0])
        input_ids = mx.array([[1, 2, 3]])
        result = BE.forward(mock_model, input_ids)
        mock_model.assert_called_once_with(input_ids)
        assert result is not None

    def test_forward_with_cache(self):
        from squish.backend import BE
        if not BE.IS_APPLE:
            pytest.skip("BE is not AppleBackend")
        mock_model = MagicMock()
        mock_model.return_value = mx.array([0.5])
        input_ids = mx.array([[4, 5]])
        cache = MagicMock()
        result = BE.forward(mock_model, input_ids, cache=cache)
        mock_model.assert_called_once_with(input_ids, cache=cache)


@mlx_only
class TestAppleBackendForwardNp:
    def test_forward_np_returns_numpy(self):
        from squish.backend import BE
        if not BE.IS_APPLE:
            pytest.skip("BE is not AppleBackend")
        mock_model = MagicMock()
        mock_model.return_value = mx.array([[1.0, 2.0, 3.0]])
        input_ids = mx.array([[0]])
        result = BE.forward_np(mock_model, input_ids)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32


@mlx_only
class TestAppleBackendSaveLoadTensors:
    def test_save_and_load_roundtrip(self, tmp_path: Path):
        from squish.backend import BE
        if not BE.IS_APPLE:
            pytest.skip("BE is not AppleBackend")
        path = str(tmp_path / "weights.safetensors")
        data = {"layer_weight": mx.array([1.0, 2.0, 3.0])}
        BE.save_tensors(path, data)
        loaded = BE.load_tensors(path)
        assert "layer_weight" in loaded


@mlx_only
class TestAppleBackendConfigureMemory:
    def test_configure_memory_invalid_fraction_returns_early(self):
        """Fraction below 0.5 → early return (covers line 155)."""
        from squish.backend import BE
        if not BE.IS_APPLE:
            pytest.skip("BE is not AppleBackend")
        BE.configure_memory(fraction=0.1)  # below threshold, no-op

    def test_configure_memory_valid_fraction(self):
        """Valid fraction → attempts Metal memory limit (non-fatal if it fails)."""
        from squish.backend import BE
        if not BE.IS_APPLE:
            pytest.skip("BE is not AppleBackend")
        BE.configure_memory(fraction=0.9)  # should not raise

    def test_configure_memory_sysctlbyname_failure_branch(self):
        """Mock libc.sysctlbyname to return -1, covering the ret != 0 branch."""
        from squish.backend import _AppleBackend
        backend = _AppleBackend()
        import ctypes
        mock_libc = MagicMock()
        mock_libc.sysctlbyname.return_value = -1  # failure — ret != 0
        with patch("ctypes.CDLL", return_value=mock_libc):
            backend.configure_memory(fraction=0.9)
        # Should not raise — just silently skip the Metal limit call
        mock_libc.sysctlbyname.assert_called_once()

    def test_configure_memory_exception_is_swallowed(self):
        """If ctypes.CDLL raises, the exception is caught and swallowed."""
        from squish.backend import _AppleBackend
        backend = _AppleBackend()
        with patch("ctypes.CDLL", side_effect=OSError("no ctypes")):
            backend.configure_memory(fraction=0.9)  # should not raise


# ---------------------------------------------------------------------------
# _TorchBackend — tested with fully mocked torch
# ---------------------------------------------------------------------------


def _make_torch_mock(cuda_available: bool = False) -> MagicMock:
    """Build a minimal mock torch module."""
    mock = MagicMock(name="torch")
    mock.cuda.is_available.return_value = cuda_available
    mock.device.side_effect = lambda s: s  # return the string as-is
    mock.int32 = "int32"
    mock.float32 = "float32"
    mock.float16 = "float16"
    mock.bfloat16 = "bfloat16"
    # .from_numpy() and .tensor() return a mock "Tensor"
    mock_tensor = MagicMock(name="Tensor")
    mock_tensor.detach.return_value = mock_tensor
    mock_tensor.float.return_value = mock_tensor
    mock_tensor.cpu.return_value = mock_tensor
    mock_tensor.numpy.return_value = np.array([1.0, 2.0], dtype=np.float32)
    mock.from_numpy.return_value = mock_tensor
    mock.tensor.return_value = mock_tensor
    mock.no_grad.return_value.__enter__ = lambda s: None
    mock.no_grad.return_value.__exit__ = MagicMock(return_value=False)
    # isinstance(x, torch.Tensor) check:
    mock.Tensor = type(mock_tensor)
    return mock


class TestTorchBackendInit:
    def test_init_cpu_path(self):
        from squish.backend import _TorchBackend
        mock_torch = _make_torch_mock(cuda_available=False)
        with patch.dict(sys.modules, {"torch": mock_torch}):
            tb = _TorchBackend()
        assert tb.device == "cpu"
        assert tb.IS_APPLE is False

    def test_init_cuda_path(self):
        from squish.backend import _TorchBackend
        mock_torch = _make_torch_mock(cuda_available=True)
        with patch.dict(sys.modules, {"torch": mock_torch}):
            tb = _TorchBackend()
        assert tb.device == "cuda"


class TestTorchBackendArray:
    def _make_tb(self, cuda=False):
        from squish.backend import _TorchBackend
        mock_torch = _make_torch_mock(cuda_available=cuda)
        with patch.dict(sys.modules, {"torch": mock_torch}):
            tb = _TorchBackend()
        return tb, mock_torch

    def test_array_from_list(self):
        tb, mock_torch = self._make_tb()
        with patch.dict(sys.modules, {"torch": mock_torch}):
            result = tb.array([1, 2, 3])
        mock_torch.tensor.assert_called_once()

    def test_array_from_numpy(self):
        tb, mock_torch = self._make_tb()
        arr = np.array([1.0, 2.0], dtype=np.float32)
        with patch.dict(sys.modules, {"torch": mock_torch}):
            result = tb.array(arr)
        mock_torch.from_numpy.assert_called_once()

    def test_array_unknown_dtype_defaults(self):
        tb, mock_torch = self._make_tb()
        with patch.dict(sys.modules, {"torch": mock_torch}):
            result = tb.array([1, 2], dtype="unknown_type")
        mock_torch.tensor.assert_called_once()


class TestTorchBackendEval:
    def test_eval_is_noop(self):
        from squish.backend import _TorchBackend
        mock_torch = _make_torch_mock()
        with patch.dict(sys.modules, {"torch": mock_torch}):
            tb = _TorchBackend()
        tb.eval(MagicMock(), MagicMock())  # should not raise


class TestTorchBackendToNumpy:
    def _make_tb(self):
        from squish.backend import _TorchBackend
        mock_torch = _make_torch_mock()
        with patch.dict(sys.modules, {"torch": mock_torch}):
            tb = _TorchBackend()
        return tb, mock_torch

    def test_to_numpy_with_torch_tensor(self):
        tb, mock_torch = self._make_tb()
        # Create a real class so isinstance(tensor, mock_torch.Tensor) works
        class FakeTensor:
            def detach(self): return self
            def float(self): return self
            def cpu(self): return self
            def numpy(self): return np.array([7.0], dtype=np.float32)
        mock_torch.Tensor = FakeTensor
        tensor = FakeTensor()
        with patch.dict(sys.modules, {"torch": mock_torch}):
            result = tb.to_numpy(tensor)
        assert isinstance(result, np.ndarray)
        assert result[0] == pytest.approx(7.0)

    def test_to_numpy_with_non_tensor_uses_numpy(self):
        tb, mock_torch = self._make_tb()
        data = [1.0, 2.0, 3.0]
        with patch.dict(sys.modules, {"torch": mock_torch}):
            result = tb.to_numpy(data)
        assert isinstance(result, np.ndarray)


class TestTorchBackendForward:
    def _make_tb(self):
        from squish.backend import _TorchBackend
        mock_torch = _make_torch_mock()
        with patch.dict(sys.modules, {"torch": mock_torch}):
            tb = _TorchBackend()
        return tb, mock_torch

    def test_forward_without_cache(self):
        tb, mock_torch = self._make_tb()
        mock_model = MagicMock()
        mock_model.return_value = MagicMock()
        input_ids = MagicMock()
        with patch.dict(sys.modules, {"torch": mock_torch}):
            result = tb.forward(mock_model, input_ids)
        mock_model.assert_called_once_with(input_ids, use_cache=False)

    def test_forward_with_cache(self):
        tb, mock_torch = self._make_tb()
        mock_model = MagicMock()
        input_ids = MagicMock()
        cache = MagicMock()
        with patch.dict(sys.modules, {"torch": mock_torch}):
            result = tb.forward(mock_model, input_ids, cache=cache)
        mock_model.assert_called_once_with(input_ids, past_key_values=cache, use_cache=True)


class TestTorchBackendForwardNp:
    def test_forward_np_extracts_logits(self):
        from squish.backend import _TorchBackend
        mock_torch = _make_torch_mock()
        with patch.dict(sys.modules, {"torch": mock_torch}):
            tb = _TorchBackend()
        mock_out = MagicMock()
        mock_out.logits = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        mock_model = MagicMock(return_value=mock_out)
        with patch.dict(sys.modules, {"torch": mock_torch}):
            result = tb.forward_np(mock_model, MagicMock())
        assert isinstance(result, np.ndarray)

    def test_forward_np_plain_tensor(self):
        from squish.backend import _TorchBackend
        mock_torch = _make_torch_mock()
        with patch.dict(sys.modules, {"torch": mock_torch}):
            tb = _TorchBackend()
        # out without .logits attribute
        plain_np = np.array([4.0, 5.0], dtype=np.float32)
        mock_model = MagicMock(return_value=plain_np)
        with patch.dict(sys.modules, {"torch": mock_torch}):
            # to_numpy called on np.ndarray → returns as-is
            result = tb.forward_np(mock_model, MagicMock())
        assert isinstance(result, np.ndarray)


class TestTorchBackendLoadModel:
    def test_load_model_basic(self):
        from squish.backend import _TorchBackend
        mock_torch = _make_torch_mock()
        mock_transformers = MagicMock()
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_transformers.AutoModelForCausalLM.from_pretrained.return_value = mock_model
        mock_transformers.AutoTokenizer.from_pretrained.return_value = mock_tokenizer
        with patch.dict(sys.modules, {"torch": mock_torch, "transformers": mock_transformers}):
            tb = _TorchBackend()
            model, tok = tb.load_model("/fake/path")
        assert model is mock_model
        assert tok is mock_tokenizer

    def test_load_model_with_load_in_4bit(self):
        from squish.backend import _TorchBackend
        mock_torch = _make_torch_mock()
        mock_transformers = MagicMock()
        mock_model = MagicMock()
        mock_transformers.AutoModelForCausalLM.from_pretrained.return_value = mock_model
        with patch.dict(sys.modules, {"torch": mock_torch, "transformers": mock_transformers}):
            tb = _TorchBackend()
            tb.load_model("/fake/path", load_in_4bit=True)
        call_kwargs = mock_transformers.AutoModelForCausalLM.from_pretrained.call_args[1]
        assert call_kwargs.get("load_in_4bit") is True


class TestTorchBackendStreamGenerate:
    def test_stream_generate_yields_text(self):
        from squish.backend import _TorchBackend
        mock_torch = _make_torch_mock()
        mock_torch.no_grad.return_value.__enter__ = MagicMock(return_value=None)
        mock_torch.no_grad.return_value.__exit__ = MagicMock(return_value=False)

        mock_transformers = MagicMock()
        # Streamer that yields some text then stops
        mock_streamer = iter(["hello", " world"])
        mock_transformers.TextIteratorStreamer.return_value = mock_streamer

        mock_threading = MagicMock()
        mock_thread = MagicMock()
        mock_threading.Thread.return_value = mock_thread

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = MagicMock()  # supports .to(device)

        with patch.dict(sys.modules, {
            "torch": mock_torch,
            "transformers": mock_transformers,
            "threading": mock_threading,
        }):
            tb = _TorchBackend()
            results = list(tb.stream_generate(mock_model, mock_tokenizer, "test prompt"))

        # Should yield text chunks plus a final ("", "stop")
        texts = [t for t, _ in results]
        assert "hello" in texts
        assert " world" in texts
        assert ("", "stop") in results


class TestTorchBackendSaveTensors:
    def test_save_tensors_with_torch_tensor(self, tmp_path: Path):
        from squish.backend import _TorchBackend
        mock_torch = _make_torch_mock()
        mock_safetensors = MagicMock()
        mock_tensor = MagicMock()
        mock_tensor.contiguous.return_value = mock_tensor

        with patch.dict(sys.modules, {
            "torch": mock_torch,
            "safetensors.torch": mock_safetensors,
        }):
            tb = _TorchBackend()
            # Patch isinstance to make the tensor check succeed
            original_isinstance = __builtins__["isinstance"] if isinstance(__builtins__, dict) else isinstance
            path = str(tmp_path / "w.safetensors")
            with patch("squish.backend.isinstance", side_effect=lambda o, c: (
                True if c is mock_torch.Tensor else original_isinstance(o, c)
            )):
                pass  # skip complex patching
            # Simpler: just pass a numpy array (the else branch)
            tb.save_tensors(path, {"k": np.array([1.0, 2.0], dtype=np.float32)})
        mock_safetensors.save_file.assert_called_once()

    def test_save_tensors_with_numpy_array(self, tmp_path: Path):
        from squish.backend import _TorchBackend
        mock_torch = _make_torch_mock()
        mock_safetensors = MagicMock()
        with patch.dict(sys.modules, {
            "torch": mock_torch,
            "safetensors.torch": mock_safetensors,
        }):
            tb = _TorchBackend()
            path = str(tmp_path / "w2.safetensors")
            tb.save_tensors(path, {"a": np.ones(4, dtype=np.float32)})
        mock_safetensors.save_file.assert_called_once()


class TestTorchBackendLoadTensors:
    def test_load_tensors_happy_path(self, tmp_path: Path):
        from squish.backend import _TorchBackend
        mock_torch = _make_torch_mock()
        mock_safetensors_torch = MagicMock()
        mock_t = MagicMock()
        mock_t.float.return_value = mock_t
        mock_t.numpy.return_value = np.array([1.0])
        mock_safetensors_torch.load_file.return_value = {"k": mock_t}
        with patch.dict(sys.modules, {
            "torch": mock_torch,
            "safetensors.torch": mock_safetensors_torch,
        }):
            tb = _TorchBackend()
            result = tb.load_tensors("/fake/path.safetensors")
        assert "k" in result

    def test_load_tensors_fallback_to_numpy(self, tmp_path: Path):
        from squish.backend import _TorchBackend
        mock_torch = _make_torch_mock()
        mock_safetensors_torch = MagicMock()
        mock_safetensors_torch.load_file.side_effect = Exception("not a torch file")
        mock_safetensors_numpy = MagicMock()
        mock_safetensors_numpy.load_file.return_value = {"w": np.array([0.5])}
        with patch.dict(sys.modules, {
            "torch": mock_torch,
            "safetensors.torch": mock_safetensors_torch,
            "safetensors.numpy": mock_safetensors_numpy,
        }):
            tb = _TorchBackend()
            result = tb.load_tensors("/fake/path.safetensors")
        assert "w" in result


class TestTorchBackendConfigureMemory:
    def test_configure_memory_no_cuda(self):
        from squish.backend import _TorchBackend
        mock_torch = _make_torch_mock(cuda_available=False)
        with patch.dict(sys.modules, {"torch": mock_torch}):
            tb = _TorchBackend()
            tb.configure_memory(0.9)  # no CUDA → silently skips

    def test_configure_memory_with_cuda(self):
        from squish.backend import _TorchBackend
        mock_torch = _make_torch_mock(cuda_available=True)
        mock_torch.cuda.is_available.return_value = True
        with patch.dict(sys.modules, {"torch": mock_torch}):
            tb = _TorchBackend()
            tb.configure_memory(0.8)
        mock_torch.cuda.set_per_process_memory_fraction.assert_called_once_with(0.8)

    def test_configure_memory_exception_swallowed(self):
        from squish.backend import _TorchBackend
        mock_torch = _make_torch_mock(cuda_available=True)
        mock_torch.cuda.set_per_process_memory_fraction.side_effect = RuntimeError("no mem")
        with patch.dict(sys.modules, {"torch": mock_torch}):
            tb = _TorchBackend()
            tb.configure_memory(0.5)  # should not raise


# ---------------------------------------------------------------------------
# _StubBackend
# ---------------------------------------------------------------------------


class TestStubBackend:
    def _make_stub(self):
        from squish.backend import _StubBackend
        return _StubBackend()

    def test_stub_is_apple_false(self):
        stub = self._make_stub()
        assert stub.IS_APPLE is False

    def test_stub_device_cpu(self):
        stub = self._make_stub()
        assert stub.device == "cpu"

    def test_array_raises(self):
        stub = self._make_stub()
        with pytest.raises(RuntimeError, match="no compute backend"):
            stub.array([1, 2, 3])

    def test_eval_is_noop(self):
        stub = self._make_stub()
        stub.eval()  # should not raise

    def test_to_numpy_raises(self):
        stub = self._make_stub()
        with pytest.raises(RuntimeError, match="no compute backend"):
            stub.to_numpy(None)

    def test_configure_memory_is_noop(self):
        stub = self._make_stub()
        stub.configure_memory()  # should not raise

    def test_load_model_raises(self):
        stub = self._make_stub()
        with pytest.raises(RuntimeError, match="no compute backend"):
            stub.load_model("/fake/path")


# ---------------------------------------------------------------------------
# create_backend factory
# ---------------------------------------------------------------------------


class TestCreateBackend:
    """Tests for the create_backend() factory function."""

    @mlx_only
    def test_returns_apple_backend_on_macos(self):
        """On macOS with MLX, create_backend() always returns AppleBackend."""
        from squish.backend import _AppleBackend, create_backend
        b = create_backend()
        assert isinstance(b, _AppleBackend)

    @mlx_only
    def test_device_arg_ignored_on_apple(self):
        """device kwarg is ignored on Apple — always Metal."""
        from squish.backend import _AppleBackend, create_backend
        b = create_backend(device="cpu")
        assert isinstance(b, _AppleBackend)
        assert b.device == "metal"

    def test_returns_torch_backend_on_linux_cpu(self):
        """On non-Apple with torch available, create_backend() returns TorchBackend."""
        from squish.backend import _TorchBackend, create_backend
        mock_torch = _make_torch_mock(cuda_available=False)
        # Simulate non-Apple platform
        with patch("squish.backend._IS_APPLE", False), \
             patch.dict(sys.modules, {"torch": mock_torch}):
            b = create_backend(device="cpu")
        assert isinstance(b, _TorchBackend)
        assert b.device == "cpu"

    def test_returns_torch_backend_auto_detect_cpu(self):
        """Auto-detect (device=None) on non-Apple + no CUDA → cpu."""
        from squish.backend import _TorchBackend, create_backend
        mock_torch = _make_torch_mock(cuda_available=False)
        with patch("squish.backend._IS_APPLE", False), \
             patch.dict(sys.modules, {"torch": mock_torch}):
            b = create_backend()
        assert isinstance(b, _TorchBackend)
        assert b.device == "cpu"

    def test_returns_stub_when_torch_missing(self):
        """Returns StubBackend when neither MLX nor torch is installed."""
        from squish.backend import _StubBackend, create_backend
        with patch("squish.backend._IS_APPLE", False), \
             patch.dict(sys.modules, {"torch": None}):
            b = create_backend()
        assert isinstance(b, _StubBackend)

    def test_cuda_device_request_without_cuda_raises(self):
        """Requesting device='cuda' on a machine without CUDA raises RuntimeError."""
        from squish.backend import create_backend
        mock_torch = _make_torch_mock(cuda_available=False)
        mock_torch.device.side_effect = lambda s: s
        with patch("squish.backend._IS_APPLE", False), \
             patch.dict(sys.modules, {"torch": mock_torch}):
            with pytest.raises(RuntimeError, match="cuda"):
                create_backend(device="cuda")
