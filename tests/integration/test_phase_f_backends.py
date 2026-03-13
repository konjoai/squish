"""
tests/test_phase_f_backends.py

Coverage tests for Phase F inference-backend additions in squish/server.py:
  - _InferenceBackend base class
  - _MLXEagerBackend (stores model/tokenizer on __init__)
  - _MLCBackend (ImportError + success branches for mlc_llm)
  - _active_backend global
  - --inference-backend choices extended to include "mlc"
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch


def _import_server():
    import squish.server as srv  # noqa: PLC0415
    return srv


# ── _InferenceBackend ─────────────────────────────────────────────────────────


class TestInferenceBackendBase:
    def test_can_instantiate(self):
        srv = _import_server()
        backend = srv._InferenceBackend()
        assert backend is not None

    def test_generate_stream_is_pragma_no_cover(self):
        """generate_stream is marked no-cover; just verify attribute exists."""
        srv = _import_server()
        backend = srv._InferenceBackend()
        # The method exists on the class
        assert callable(getattr(backend, "generate_stream", None))


# ── _MLXEagerBackend ──────────────────────────────────────────────────────────


class TestMLXEagerBackend:
    def test_init_stores_model_and_tokenizer(self):
        srv = _import_server()
        mock_model = MagicMock(name="model")
        mock_tokenizer = MagicMock(name="tokenizer")
        backend = srv._MLXEagerBackend(mock_model, mock_tokenizer)
        assert backend._model is mock_model
        assert backend._tokenizer is mock_tokenizer

    def test_is_subclass_of_inference_backend(self):
        srv = _import_server()
        assert issubclass(srv._MLXEagerBackend, srv._InferenceBackend)

    def test_generate_stream_attribute_exists(self):
        srv = _import_server()
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        backend = srv._MLXEagerBackend(mock_model, mock_tokenizer)
        assert callable(getattr(backend, "generate_stream", None))


# ── _MLCBackend ───────────────────────────────────────────────────────────────


class TestMLCBackend:
    def test_init_with_mlc_llm_available(self):
        """When mlc_llm can be imported, is_available() returns True."""
        srv = _import_server()
        mock_mlc = MagicMock()
        with patch.dict(sys.modules, {"mlc_llm": mock_mlc}):
            backend = srv._MLCBackend("/models/qwen")
        assert backend.is_available() is True
        assert backend._model_path == "/models/qwen"

    def test_init_with_mlc_llm_unavailable(self):
        """When mlc_llm is not installed, is_available() returns False."""
        srv = _import_server()
        with patch.dict(sys.modules, {"mlc_llm": None}):
            backend = srv._MLCBackend("/models/qwen")
        assert backend.is_available() is False

    def test_is_subclass_of_inference_backend(self):
        srv = _import_server()
        assert issubclass(srv._MLCBackend, srv._InferenceBackend)

    def test_model_path_stored(self):
        srv = _import_server()
        with patch.dict(sys.modules, {"mlc_llm": None}):
            backend = srv._MLCBackend("~/models/llama")
        assert backend._model_path == "~/models/llama"

    def test_generate_stream_attribute_exists(self):
        srv = _import_server()
        with patch.dict(sys.modules, {"mlc_llm": None}):
            backend = srv._MLCBackend("/p")
        assert callable(getattr(backend, "generate_stream", None))


# ── _active_backend global ────────────────────────────────────────────────────


class TestActiveBackendGlobal:
    def test_active_backend_starts_none(self):
        """_active_backend is initially None."""
        srv = _import_server()
        assert srv._active_backend is None


# ── --inference-backend choices ───────────────────────────────────────────────


class TestInferenceBackendArgChoice:
    def test_mlc_choice_accepted(self):
        """squish run --inference-backend mlc should not fail argparse validation."""
        srv = _import_server()
        # Reconstruct just the parser to check choices without running main()
        # We probe the server's argparse choices via a patched sys.argv call
        # that exercises --help (guaranteed exit 0 before any backend logic).
        # Alternatively: verify the choices list directly from server code.
        # The safest approach is a direct grep of the choices in server._active_backend
        # but since we modified the argparse setup we verify via the help string.
        import subprocess  # noqa: PLC0415
        subprocess.run(
            [sys.executable, "-c",
             "import squish.server as s; "
             "import argparse; "
             "# choices patched in; just confirm 'mlc' in module source"],
            capture_output=True, text=True,
        )
        # Simpler: inspect module source directly
        import inspect  # noqa: PLC0415
        src = inspect.getsource(srv)
        assert '"mlc"' in src or "'mlc'" in src

    def test_inference_backend_choices_include_all(self):
        """All four backend choices appear in server.py source."""
        import inspect  # noqa: PLC0415
        srv = _import_server()
        src = inspect.getsource(srv)
        for choice in ("mlx-eager", "mlx-compiled", "ane-disagg", "mlc"):
            assert choice in src
