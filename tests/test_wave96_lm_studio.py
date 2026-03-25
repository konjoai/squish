"""tests/test_wave96_lm_studio.py — Wave 96: LM Studio auto-detection tests.

Covers:
- LocalModelScanner.scan_lm_studio(): source tag, publisher/repo naming,
  GGUF discovery, safetensors discovery, LMSTUDIO_MODELS_DIR env override
- probe_lm_studio(): not running, running+models, LMSTUDIO_BASE_URL override
- LMStudioStatus: str(), model_count
- LMStudioClient.models(): not running returns []
- LMStudioClient.chat_completions(): SSE token iteration
- cmd_models: external section appears when LM Studio models present
- find_all() deduplication with lm_studio source
"""
from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ── helpers ────────────────────────────────────────────────────────────────────

def _capture(fn, *a, **kw) -> str:
    buf = io.StringIO()
    old, sys.stdout = sys.stdout, buf
    try:
        fn(*a, **kw)
    finally:
        sys.stdout = old
    return buf.getvalue()


def _make_lms_tree(root: Path) -> dict[str, Path]:
    """Create a minimal LM Studio directory tree for tests.

    Structure:
        root/
          lmstudio-ai/
            gemma-2-2b-it-GGUF/
              gemma-2-2b-it-Q4_K_M.gguf   (fake GGUF)
          meta-llama/
            Llama-3.2-3B-Instruct/
              model.safetensors            (fake safetensors)
    """
    gguf_dir  = root / "lmstudio-ai" / "gemma-2-2b-it-GGUF"
    gguf_dir.mkdir(parents=True)
    gguf_file = gguf_dir / "gemma-2-2b-it-Q4_K_M.gguf"
    gguf_file.write_bytes(b"\x00" * 1024)   # 1 KB stub

    st_dir = root / "meta-llama" / "Llama-3.2-3B-Instruct"
    st_dir.mkdir(parents=True)
    st_file = st_dir / "model.safetensors"
    st_file.write_bytes(b"\x00" * 2048)    # 2 KB stub

    return {"gguf": gguf_file, "st_dir": st_dir, "st_file": st_file}


# ══════════════════════════════════════════════════════════════════════════════
# 1. LocalModelScanner.scan_lm_studio()
# ══════════════════════════════════════════════════════════════════════════════

class TestScanLmStudio(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._root = Path(self._tmp.name)
        self._files = _make_lms_tree(self._root)

    def tearDown(self):
        self._tmp.cleanup()

    def _scanner(self):
        from squish.serving.local_model_scanner import LocalModelScanner
        return LocalModelScanner(lm_studio_dir=self._root)

    # Source tag
    def test_gguf_source_is_lm_studio(self):
        models = self._scanner().scan_lm_studio()
        sources = {m.source for m in models}
        self.assertIn("lm_studio", sources)
        self.assertNotIn("gguf", sources, "source='gguf' bug should be fixed")

    def test_safetensors_source_is_lm_studio(self):
        models = self._scanner().scan_lm_studio()
        self.assertTrue(all(m.source == "lm_studio" for m in models))

    # GGUF discovery
    def test_finds_gguf_file(self):
        names = [m.name for m in self._scanner().scan_lm_studio()]
        # name should be "publisher/repo" = "lmstudio-ai/gemma-2-2b-it-GGUF"
        self.assertTrue(any("lmstudio-ai" in n for n in names),
                        f"No lmstudio-ai model in: {names}")

    def test_gguf_publisher_repo_naming(self):
        models = self._scanner().scan_lm_studio()
        gguf_m = next(m for m in models if m.path.suffix == ".gguf")
        self.assertIn("/", gguf_m.name, "Expected publisher/repo format")
        self.assertEqual(gguf_m.name, "lmstudio-ai/gemma-2-2b-it-GGUF")

    def test_gguf_size_populated(self):
        models = self._scanner().scan_lm_studio()
        gguf_m = next(m for m in models if m.path.suffix == ".gguf")
        self.assertGreater(gguf_m.size_bytes, 0)

    # Safetensors discovery
    def test_finds_safetensors_repo(self):
        names = [m.name for m in self._scanner().scan_lm_studio()]
        self.assertTrue(any("meta-llama" in n for n in names),
                        f"No meta-llama safetensors in: {names}")

    def test_safetensors_publisher_repo_naming(self):
        models = self._scanner().scan_lm_studio()
        st_m = next(m for m in models if m.path.is_dir())
        self.assertIn("/", st_m.name)
        self.assertEqual(st_m.name, "meta-llama/Llama-3.2-3B-Instruct")

    def test_safetensors_size_populated(self):
        models = self._scanner().scan_lm_studio()
        st_m = next(m for m in models if m.path.is_dir())
        self.assertGreater(st_m.size_bytes, 0)

    # Empty dir
    def test_missing_dir_returns_empty(self):
        from squish.serving.local_model_scanner import LocalModelScanner
        scn = LocalModelScanner(lm_studio_dir=Path("/nonexistent/lm-studio/models"))
        self.assertEqual(scn.scan_lm_studio(), [])

    # Two models found total
    def test_returns_two_models(self):
        models = self._scanner().scan_lm_studio()
        self.assertEqual(len(models), 2, f"Expected 2, got {len(models)}: {[m.name for m in models]}")

    # LMSTUDIO_MODELS_DIR env var override
    def test_env_var_override(self):
        from squish.serving.local_model_scanner import LocalModelScanner
        # Scanner with default dir (nonexistent), but env var points to our tree
        with patch.dict(os.environ, {"LMSTUDIO_MODELS_DIR": str(self._root)}):
            scn = LocalModelScanner()  # default dir likely doesn't exist
            models = scn.scan_lm_studio()
        self.assertEqual(len(models), 2)

    def test_env_var_override_takes_precedence(self):
        from squish.serving.local_model_scanner import LocalModelScanner
        other_root = Path(self._tmp.name + "_other")
        other_root.mkdir()
        with patch.dict(os.environ, {"LMSTUDIO_MODELS_DIR": str(other_root)}):
            scn = LocalModelScanner(lm_studio_dir=self._root)  # explicit dir ignored
            models = scn.scan_lm_studio()
        self.assertEqual(models, [], "Env override should point to empty dir")

    # find_all() integration
    def test_find_all_includes_lm_studio(self):
        from squish.serving.local_model_scanner import LocalModelScanner
        scn = LocalModelScanner(lm_studio_dir=self._root)
        all_models = scn.find_all()
        sources = {m.source for m in all_models}
        self.assertIn("lm_studio", sources)

    # family / params inference
    def test_family_guessed_from_gguf_stem(self):
        models = self._scanner().scan_lm_studio()
        gguf_m = next(m for m in models if m.path.suffix == ".gguf")
        self.assertEqual(gguf_m.family, "gemma")


# ══════════════════════════════════════════════════════════════════════════════
# 2. probe_lm_studio()
# ══════════════════════════════════════════════════════════════════════════════

class TestProbeLmStudio(unittest.TestCase):

    def test_not_running_returns_false(self):
        import urllib.error
        from squish.serving.lm_studio_bridge import probe_lm_studio
        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("refused")):
            status = probe_lm_studio()
        self.assertFalse(status.running)

    def test_not_running_never_raises(self):
        from squish.serving.lm_studio_bridge import probe_lm_studio
        with patch("urllib.request.urlopen", side_effect=OSError("no route")):
            try:
                probe_lm_studio()
            except Exception as exc:
                self.fail(f"probe_lm_studio raised {exc!r}")

    def test_running_returns_true(self):
        from squish.serving.lm_studio_bridge import probe_lm_studio
        payload = json.dumps({"data": [{"id": "lmstudio-community/gemma-2-2b-it-GGUF"}]}).encode()
        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.read.return_value = payload
        mock_resp.headers = {}
        with patch("urllib.request.urlopen", return_value=mock_resp):
            status = probe_lm_studio()
        self.assertTrue(status.running)

    def test_running_populates_loaded_models(self):
        from squish.serving.lm_studio_bridge import probe_lm_studio
        payload = json.dumps({"data": [
            {"id": "lmstudio-community/gemma-2-2b-it-GGUF"},
            {"id": "meta-llama/Llama-3.2-3B-Instruct"},
        ]}).encode()
        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.read.return_value = payload
        mock_resp.headers = {}
        with patch("urllib.request.urlopen", return_value=mock_resp):
            status = probe_lm_studio()
        self.assertEqual(len(status.loaded_models), 2)
        self.assertIn("lmstudio-community/gemma-2-2b-it-GGUF", status.loaded_models)

    def test_base_url_env_override(self):
        import urllib.error
        from squish.serving.lm_studio_bridge import probe_lm_studio
        captured_urls: list[str] = []

        def _fake(req, timeout=0.8):
            captured_urls.append(req.get_full_url())
            raise urllib.error.URLError("test")

        with patch.dict(os.environ, {"LMSTUDIO_BASE_URL": "http://10.0.0.1:5678"}):
            with patch("urllib.request.urlopen", side_effect=_fake):
                probe_lm_studio()

        self.assertTrue(any("10.0.0.1:5678" in u for u in captured_urls),
                        f"Expected custom URL, got: {captured_urls}")

    def test_base_url_in_status(self):
        import urllib.error
        from squish.serving.lm_studio_bridge import probe_lm_studio
        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("x")):
            status = probe_lm_studio()
        self.assertIn("1234", status.base_url)


# ══════════════════════════════════════════════════════════════════════════════
# 3. LMStudioStatus
# ══════════════════════════════════════════════════════════════════════════════

class TestLMStudioStatus(unittest.TestCase):

    def test_model_count_zero_when_not_running(self):
        from squish.serving.lm_studio_bridge import LMStudioStatus
        s = LMStudioStatus(running=False, base_url="http://127.0.0.1:1234")
        self.assertEqual(s.model_count, 0)

    def test_model_count_when_running(self):
        from squish.serving.lm_studio_bridge import LMStudioStatus
        s = LMStudioStatus(
            running=True,
            base_url="http://127.0.0.1:1234",
            loaded_models=["modelA", "modelB"],
        )
        self.assertEqual(s.model_count, 2)

    def test_str_not_running(self):
        from squish.serving.lm_studio_bridge import LMStudioStatus
        s = LMStudioStatus(running=False, base_url="http://127.0.0.1:1234")
        self.assertIn("not running", str(s).lower())

    def test_str_running_with_model(self):
        from squish.serving.lm_studio_bridge import LMStudioStatus
        s = LMStudioStatus(running=True, base_url="http://127.0.0.1:1234",
                           loaded_models=["gemma:2b"])
        out = str(s)
        self.assertIn("running", out.lower())
        self.assertIn("gemma:2b", out)


# ══════════════════════════════════════════════════════════════════════════════
# 4. LMStudioClient
# ══════════════════════════════════════════════════════════════════════════════

class TestLMStudioClient(unittest.TestCase):

    def test_models_returns_empty_when_not_running(self):
        import urllib.error
        from squish.serving.lm_studio_bridge import LMStudioClient
        client = LMStudioClient()
        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("x")):
            result = client.models()
        self.assertEqual(result, [])

    def test_models_returns_list_when_running(self):
        from squish.serving.lm_studio_bridge import LMStudioClient
        payload = json.dumps({"data": [{"id": "gemma:2b"}, {"id": "llama3:8b"}]}).encode()
        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.read.return_value = payload
        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = LMStudioClient().models()
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["id"], "gemma:2b")

    def test_chat_completions_sse_yields_tokens(self):
        from squish.serving.lm_studio_bridge import LMStudioClient

        sse_lines = [
            b'data: {"choices":[{"delta":{"content":"Hello"}}]}\n',
            b'data: {"choices":[{"delta":{"content":" world"}}]}\n',
            b'data: [DONE]\n',
        ]
        mock_resp = MagicMock()
        mock_resp.__iter__ = lambda s: iter(sse_lines)
        mock_resp.status = 200
        with patch("urllib.request.urlopen", return_value=mock_resp):
            tokens = list(LMStudioClient().chat_completions(
                messages=[{"role": "user", "content": "hi"}],
                model="gemma:2b",
                stream=True,
            ))
        self.assertEqual(tokens, ["Hello", " world"])

    def test_chat_completions_not_running_raises(self):
        import urllib.error
        from squish.serving.lm_studio_bridge import LMStudioClient
        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("refused")):
            with self.assertRaises(ConnectionError):
                list(LMStudioClient().chat_completions(
                    messages=[{"role": "user", "content": "hi"}],
                    stream=True,
                ))


# ══════════════════════════════════════════════════════════════════════════════
# 5. cmd_models integration
# ══════════════════════════════════════════════════════════════════════════════

class TestCmdModelsLmStudio(unittest.TestCase):
    """cmd_models should show LM Studio models in the External section."""

    def _run_models(self, tmp_lms_root: Path) -> str:
        import squish.cli as cli
        from squish.serving.lm_studio_bridge import LMStudioStatus
        args = types.SimpleNamespace()

        # Make ~/models/ exist and be empty so the Squish section runs
        with tempfile.TemporaryDirectory() as squish_models_tmp:
            squish_models = Path(squish_models_tmp)
            # Patch _MODELS_DIR and LocalModelScanner to use our tmp dirs
            with patch.object(cli, "_MODELS_DIR", squish_models):
                with patch(
                    "squish.serving.local_model_scanner.LocalModelScanner.__init__",
                    lambda self, **kw: None,
                ):
                    with patch(
                        "squish.serving.local_model_scanner.LocalModelScanner.scan_ollama",
                        return_value=[],
                    ):
                        with patch(
                            "squish.serving.local_model_scanner.LocalModelScanner.scan_lm_studio",
                            return_value=self._fake_lms_models(),
                        ):
                            with patch(
                                "squish.serving.lm_studio_bridge.probe_lm_studio",
                                return_value=LMStudioStatus(running=False, base_url="http://127.0.0.1:1234"),
                            ):
                                return _capture(cli.cmd_models, args)

    def _fake_lms_models(self):
        from squish.serving.local_model_scanner import LocalModel
        return [
            LocalModel(
                name="lmstudio-ai/gemma-2-2b-it-GGUF",
                path=Path("/fake/gemma.gguf"),
                source="lm_studio",
                size_bytes=1_500_000_000,
            ),
        ]

    def test_external_section_present(self):
        with tempfile.TemporaryDirectory() as d:
            out = self._run_models(Path(d))
        self.assertTrue(
            "External" in out or "lm_studio" in out or "gemma" in out.lower(),
            f"LM Studio section missing. Output:\n{out[:800]}",
        )

    def test_lm_studio_model_name_shown(self):
        with tempfile.TemporaryDirectory() as d:
            out = self._run_models(Path(d))
        self.assertIn("lmstudio-ai", out)

    def test_lm_studio_source_tag_shown(self):
        with tempfile.TemporaryDirectory() as d:
            out = self._run_models(Path(d))
        self.assertIn("lm_studio", out)


if __name__ == "__main__":
    unittest.main()
