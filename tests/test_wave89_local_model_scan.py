"""tests/test_wave89_local_model_scan.py

Wave 89 — Local Model Scanner + squish pull URI schemes

Tests for:
  - LocalModel dataclass fields
  - LocalModelScanner.scan_squish() with temp dirs
  - LocalModelScanner.scan_ollama() with mock manifest files
  - LocalModelScanner.scan_lm_studio() with GGUF files
  - LocalModelScanner.find_all() deduplication
  - Nonexistent directories return []
  - /api/tags reflects scanner-discovered models
  - _dir_to_canonical() name normalisation
  - squish import subparser registered in CLI
  - squish pull URI dispatch (ollama: / hf: prefixes)
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)


# ============================================================================
# TestLocalModelDataclass
# ============================================================================

class TestLocalModelDataclass(unittest.TestCase):

    def _make(self, **kw):
        from squish.serving.local_model_scanner import LocalModel
        defaults = dict(name="qwen3:8b", path="/tmp/test", source="squish")
        defaults.update(kw)
        return LocalModel(**defaults)

    def test_required_fields(self):
        m = self._make()
        assert m.name == "qwen3:8b"
        assert m.source == "squish"
        assert isinstance(m.path, Path)

    def test_default_optional_fields(self):
        m = self._make()
        assert m.size_bytes == 0
        assert m.family == ""
        assert m.params == ""

    def test_path_is_coerced_to_Path(self):
        m = self._make(path="/tmp/some/model")
        assert isinstance(m.path, Path)

    def test_explicit_optional_fields(self):
        m = self._make(size_bytes=4_000_000_000, family="qwen", params="8B")
        assert m.size_bytes == 4_000_000_000
        assert m.family == "qwen"
        assert m.params == "8B"


# ============================================================================
# TestScanSquish
# ============================================================================

class TestScanSquish(unittest.TestCase):

    def test_nonexistent_dir_returns_empty(self):
        from squish.serving.local_model_scanner import LocalModelScanner
        scanner = LocalModelScanner(squish_models_dir=Path("/nonexistent/dir"))
        assert scanner.scan_squish() == []

    def test_empty_dir_returns_empty(self):
        from squish.serving.local_model_scanner import LocalModelScanner
        with tempfile.TemporaryDirectory() as tmp:
            scanner = LocalModelScanner(squish_models_dir=Path(tmp))
            assert scanner.scan_squish() == []

    def test_dot_dirs_skipped(self):
        from squish.serving.local_model_scanner import LocalModelScanner
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / ".hidden").mkdir()
            scanner = LocalModelScanner(squish_models_dir=Path(tmp))
            assert scanner.scan_squish() == []

    def test_single_model_dir(self):
        from squish.serving.local_model_scanner import LocalModelScanner
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "Qwen3-8B-bf16").mkdir()
            scanner = LocalModelScanner(squish_models_dir=Path(tmp))
            models = scanner.scan_squish()
            assert len(models) == 1
            assert models[0].source == "squish"

    def test_multiple_model_dirs(self):
        from squish.serving.local_model_scanner import LocalModelScanner
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "Qwen3-8B-bf16").mkdir()
            (Path(tmp) / "Llama-3.1-8B-Instruct-bf16").mkdir()
            scanner = LocalModelScanner(squish_models_dir=Path(tmp))
            models = scanner.scan_squish()
            assert len(models) == 2

    def test_size_bytes_computed(self):
        from squish.serving.local_model_scanner import LocalModelScanner
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "TestModel-7B-bf16"
            model_dir.mkdir()
            (model_dir / "weights.npz").write_bytes(b"x" * 1000)
            scanner = LocalModelScanner(squish_models_dir=Path(tmp))
            models = scanner.scan_squish()
            assert models[0].size_bytes == 1000


# ============================================================================
# TestScanOllama
# ============================================================================

class TestScanOllama(unittest.TestCase):

    def _make_manifest_dir(self, tmp, model_name, tag, size=4_200_000_000):
        model_dir = Path(tmp) / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "schemaVersion": 2,
            "layers": [
                {"digest": "sha256:001", "size": size,
                 "mediaType": "application/vnd.ollama.image.model"},
            ],
        }
        (model_dir / tag).write_text(json.dumps(manifest))
        return model_dir

    def test_nonexistent_dir_returns_empty(self):
        from squish.serving.local_model_scanner import LocalModelScanner
        scanner = LocalModelScanner(ollama_manifests_dir=Path("/nonexistent"))
        assert scanner.scan_ollama() == []

    def test_parses_name_and_tag(self):
        from squish.serving.local_model_scanner import LocalModelScanner
        with tempfile.TemporaryDirectory() as tmp:
            self._make_manifest_dir(tmp, "qwen3", "8b")
            scanner = LocalModelScanner(ollama_manifests_dir=Path(tmp))
            models = scanner.scan_ollama()
            assert len(models) == 1
            assert models[0].name == "qwen3:8b"
            assert models[0].source == "ollama"

    def test_parses_size_from_layers(self):
        from squish.serving.local_model_scanner import LocalModelScanner
        with tempfile.TemporaryDirectory() as tmp:
            self._make_manifest_dir(tmp, "llama3.1", "8b", size=5_000_000_000)
            scanner = LocalModelScanner(ollama_manifests_dir=Path(tmp))
            models = scanner.scan_ollama()
            assert models[0].size_bytes == 5_000_000_000

    def test_multiple_models(self):
        from squish.serving.local_model_scanner import LocalModelScanner
        with tempfile.TemporaryDirectory() as tmp:
            self._make_manifest_dir(tmp, "qwen3", "8b")
            self._make_manifest_dir(tmp, "gemma3", "4b")
            scanner = LocalModelScanner(ollama_manifests_dir=Path(tmp))
            models = scanner.scan_ollama()
            assert len(models) == 2

    def test_malformed_manifest_handled(self):
        from squish.serving.local_model_scanner import LocalModelScanner
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "badmodel"
            model_dir.mkdir()
            (model_dir / "latest").write_text("NOT JSON {{{")
            scanner = LocalModelScanner(ollama_manifests_dir=Path(tmp))
            models = scanner.scan_ollama()
            assert len(models) == 1
            assert models[0].size_bytes == 0


# ============================================================================
# TestScanLmStudio
# ============================================================================

class TestScanLmStudio(unittest.TestCase):

    def test_nonexistent_dir_returns_empty(self):
        from squish.serving.local_model_scanner import LocalModelScanner
        scanner = LocalModelScanner(lm_studio_dir=Path("/nonexistent"))
        assert scanner.scan_lm_studio() == []

    def test_finds_gguf_file(self):
        from squish.serving.local_model_scanner import LocalModelScanner
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "model.gguf").write_bytes(b"GGUFxxx")
            scanner = LocalModelScanner(lm_studio_dir=Path(tmp))
            models = scanner.scan_lm_studio()
            assert len(models) == 1
            assert models[0].source == "gguf"

    def test_finds_nested_gguf(self):
        from squish.serving.local_model_scanner import LocalModelScanner
        with tempfile.TemporaryDirectory() as tmp:
            sub = Path(tmp) / "publisher" / "model"
            sub.mkdir(parents=True)
            (sub / "Q4_K_M.gguf").write_bytes(b"GGUFxxx")
            scanner = LocalModelScanner(lm_studio_dir=Path(tmp))
            models = scanner.scan_lm_studio()
            assert len(models) == 1

    def test_non_gguf_files_skipped(self):
        from squish.serving.local_model_scanner import LocalModelScanner
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "README.md").write_text("readme")
            (Path(tmp) / "config.json").write_text("{}")
            scanner = LocalModelScanner(lm_studio_dir=Path(tmp))
            assert scanner.scan_lm_studio() == []


# ============================================================================
# TestFindAll
# ============================================================================

class TestFindAll(unittest.TestCase):

    def _empty_scanner(self):
        from squish.serving.local_model_scanner import LocalModelScanner
        return LocalModelScanner(
            squish_models_dir=Path("/nonexistent"),
            ollama_manifests_dir=Path("/nonexistent"),
            lm_studio_dir=Path("/nonexistent"),
        )

    def test_empty_sources_returns_empty(self):
        scanner = self._empty_scanner()
        assert scanner.find_all() == []

    def test_deduplicates_same_name(self):
        from squish.serving.local_model_scanner import LocalModel
        scanner = self._empty_scanner()
        scanner.scan_squish   = lambda: [LocalModel("qwen3:8b", "/tmp/a", "squish")]
        scanner.scan_ollama   = lambda: [LocalModel("qwen3:8b", "/tmp/b", "ollama")]
        scanner.scan_lm_studio = lambda: []
        result = scanner.find_all()
        assert len(result) == 1

    def test_squish_wins_dedup_order(self):
        from squish.serving.local_model_scanner import LocalModel
        scanner = self._empty_scanner()
        scanner.scan_squish   = lambda: [LocalModel("qwen3:8b", "/tmp/s", "squish")]
        scanner.scan_ollama   = lambda: [LocalModel("qwen3:8b", "/tmp/o", "ollama")]
        scanner.scan_lm_studio = lambda: []
        result = scanner.find_all()
        assert result[0].source == "squish"

    def test_distinct_names_all_included(self):
        from squish.serving.local_model_scanner import LocalModel
        scanner = self._empty_scanner()
        scanner.scan_squish   = lambda: [LocalModel("qwen3:8b",  "/a", "squish")]
        scanner.scan_ollama   = lambda: [LocalModel("llama3:8b", "/b", "ollama")]
        scanner.scan_lm_studio = lambda: [LocalModel("gemma3:4b", "/c", "gguf")]
        result = scanner.find_all()
        assert len(result) == 3


# ============================================================================
# TestDirToCanonical
# ============================================================================

class TestDirToCanonical(unittest.TestCase):

    def _canon(self, name):
        from squish.serving.local_model_scanner import _dir_to_canonical
        return _dir_to_canonical(name)

    def test_qwen3_8b(self):
        result = self._canon("Qwen3-8B-bf16")
        assert "qwen3" in result.lower()
        assert "8b" in result.lower()

    def test_llama_model(self):
        result = self._canon("Llama-3.1-8B-Instruct-bf16")
        assert "llama" in result.lower()
        assert "8b" in result.lower()

    def test_strips_bf16(self):
        result = self._canon("TestModel-7B-bf16")
        assert "bf16" not in result.lower()

    def test_strips_instruct(self):
        result = self._canon("Mistral-7B-Instruct-bf16")
        assert "instruct" not in result.lower()


# ============================================================================
# TestOllamaCompatTags
# ============================================================================

class TestOllamaCompatTags(unittest.TestCase):

    def test_api_tags_returns_200(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from squish.serving.ollama_compat import mount_ollama

        app = FastAPI()

        class _FakeState:
            model = None
            model_name = "squish"

        mount_ollama(app,
                     get_state=lambda: _FakeState(),
                     get_generate=lambda: None,
                     get_tokenizer=lambda: None)
        client = TestClient(app)
        resp = client.get("/api/tags")
        assert resp.status_code == 200
        assert "models" in resp.json()

    def test_api_tags_includes_ollama_models(self):
        """When scanner returns Ollama models, /api/tags lists them."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from squish.serving.local_model_scanner import LocalModelScanner, LocalModel
        from squish.serving.ollama_compat import mount_ollama

        class _FakeState:
            model = None
            model_name = "squish"

        app = FastAPI()
        with patch.object(LocalModelScanner, "find_all", return_value=[
            LocalModel("llama3.1:8b", Path("/tmp/ol"), "ollama",
                       size_bytes=5_000_000_000, family="llama", params="8B"),
        ]):
            mount_ollama(app,
                         get_state=lambda: _FakeState(),
                         get_generate=lambda: None,
                         get_tokenizer=lambda: None)
            client = TestClient(app)
            resp = client.get("/api/tags")
            assert resp.status_code == 200
            models = resp.json()["models"]
            names = [m["name"] for m in models]
            assert any("llama3.1" in n for n in names)


# ============================================================================
# TestCmdImportRegistered
# ============================================================================

class TestCmdImportRegistered(unittest.TestCase):

    def test_cmd_import_callable(self):
        import squish.cli as cli
        assert callable(cli.cmd_import)

    def test_cmd_import_has_docstring(self):
        import squish.cli as cli
        assert cli.cmd_import.__doc__ is not None

    def test_import_subparser_registered(self):
        """squish import subcommand must dispatch to cmd_import."""
        import squish.cli as cli
        import argparse

        # Verify by calling main() with patched argv and intercepting dispatch
        called_with = {}
        def _spy_import(args):
            called_with["args"] = args

        with patch.object(cli, "cmd_import", side_effect=_spy_import):
            with patch("sys.argv", ["squish", "import", "ollama:qwen3:8b"]):
                try:
                    cli.main()
                except SystemExit:
                    pass
        assert "args" in called_with, "cmd_import was not called"
        assert called_with["args"].import_source == "ollama:qwen3:8b"


# ============================================================================
# TestCmdPullUriDispatch
# ============================================================================

class TestCmdPullUriDispatch(unittest.TestCase):

    def _make_args(self, model):
        import types
        return types.SimpleNamespace(
            model=model, models_dir="", token="",
            int2=False, int3=False, int8=False,
            verbose=False, force=False, refresh_catalog=False,
        )

    def test_ollama_uri_calls_pull_from_ollama(self):
        import squish.cli as cli
        args = self._make_args("ollama:qwen3:8b")
        with patch.object(cli, "_pull_from_ollama") as mock_fn:
            mock_fn.return_value = None
            try:
                cli.cmd_pull(args)
            except SystemExit:
                pass
            mock_fn.assert_called_once()
            call_args = mock_fn.call_args[0]
            assert "qwen3" in str(call_args[0])

    def test_hf_uri_calls_pull_from_hf(self):
        import squish.cli as cli
        args = self._make_args("hf:mlx-community/Qwen3-8B-bf16")
        with patch.object(cli, "_pull_from_hf") as mock_fn:
            mock_fn.return_value = None
            try:
                cli.cmd_pull(args)
            except SystemExit:
                pass
            mock_fn.assert_called_once()

    def test_huggingface_url_dispatches_to_hf(self):
        import squish.cli as cli
        args = self._make_args("https://huggingface.co/mlx-community/Qwen3-8B-bf16")
        with patch.object(cli, "_pull_from_hf") as mock_fn:
            mock_fn.return_value = None
            try:
                cli.cmd_pull(args)
            except SystemExit:
                pass
            mock_fn.assert_called_once()

    def test_pull_from_ollama_no_server_prints_friendly_message(self):
        """_pull_from_ollama prints friendly message when Ollama is not running."""
        import squish.cli as cli
        import io
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            cli._pull_from_ollama("qwen3:8b", Path("/tmp"), None)
        output = buf.getvalue()
        assert len(output) > 0
        assert any(word in output.lower() for word in
                   ("ollama", "not running", "start", "install", "squish pull"))


if __name__ == "__main__":
    unittest.main()
