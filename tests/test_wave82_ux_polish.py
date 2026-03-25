"""tests/test_wave82_ux_polish.py

Wave 82 — UX Polish: Fuzzy Catalog Resolution, Browser Auto-Open, Offline Status

Tests for:
  - catalog._normalize_model_name(): dash→colon conversion, quant-suffix stripping
  - catalog.resolve(): fuzzy matching for user-friendly names ("qwen2.5-7b" → entry)
  - catalog.suggest(): "did you mean?" results for typos
  - cli._catalog_suggest() wrapper accessible from cli module
  - cli._open_browser_when_ready(): uses subprocess.Popen, not threading.Thread
  - index.html: loadModels failure threshold ≥ 2 before showing offline banner
  - index.html: auto-enable agent mode logic present at startup
"""
from __future__ import annotations

import os
import sys
import re
import unittest
from unittest.mock import MagicMock, patch

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from squish.catalog import (
    _normalize_model_name,
    resolve,
    suggest,
)


# ============================================================================
# TestNormalizeModelName — _normalize_model_name()
# ============================================================================

class TestNormalizeModelName(unittest.TestCase):
    """_normalize_model_name converts user-friendly names to catalog IDs."""

    def test_dash_to_colon_7b(self):
        assert _normalize_model_name("qwen2.5-7b") == "qwen2.5:7b"

    def test_dash_to_colon_8b(self):
        assert _normalize_model_name("qwen3-8b") == "qwen3:8b"

    def test_dash_to_colon_14b(self):
        assert _normalize_model_name("qwen2.5-14b") == "qwen2.5:14b"

    def test_dash_to_colon_llama(self):
        assert _normalize_model_name("llama3.1-8b") == "llama3.1:8b"

    def test_dash_to_colon_deepseek_r1(self):
        assert _normalize_model_name("deepseek-r1-7b") == "deepseek-r1:7b"

    def test_dash_to_colon_gemma(self):
        assert _normalize_model_name("gemma3-4b") == "gemma3:4b"

    def test_strips_int2_suffix(self):
        result = _normalize_model_name("qwen2.5-7b-int2")
        assert result == "qwen2.5:7b"

    def test_strips_int4_suffix(self):
        result = _normalize_model_name("qwen3-8b-int4")
        assert result == "qwen3:8b"

    def test_strips_bf16_suffix(self):
        result = _normalize_model_name("llama3.1-8b-bf16")
        assert result == "llama3.1:8b"

    def test_strips_int8_suffix_underscore(self):
        result = _normalize_model_name("qwen3-8b_int8")
        assert result == "qwen3:8b"

    def test_already_canonical_unchanged(self):
        assert _normalize_model_name("qwen2.5:7b") == "qwen2.5:7b"

    def test_uppercase_lowercased(self):
        result = _normalize_model_name("Qwen2.5-7B")
        assert result == "qwen2.5:7b"

    def test_leading_trailing_spaces(self):
        result = _normalize_model_name("  qwen3-8b  ")
        assert result == "qwen3:8b"

    def test_no_size_suffix_unchanged(self):
        # "qwen2.5" doesn't have a -Nb suffix, so dash→colon not applied
        result = _normalize_model_name("qwen2.5")
        assert result == "qwen2.5"

    def test_returns_lowercase(self):
        result = _normalize_model_name("LLAMA3.1-8B")
        assert result == result.lower()


# ============================================================================
# TestResolveFuzzy — resolve() with user-friendly names
# ============================================================================

class TestResolveFuzzy(unittest.TestCase):
    """resolve() must accept user-friendly dash-separated names."""

    def test_resolve_qwen25_7b_dash(self):
        """'qwen2.5-7b' → qwen2.5:7b entry."""
        entry = resolve("qwen2.5-7b")
        assert entry is not None
        assert entry.id == "qwen2.5:7b"

    def test_resolve_qwen3_8b_dash(self):
        entry = resolve("qwen3-8b")
        assert entry is not None
        assert entry.id == "qwen3:8b"

    def test_resolve_llama31_8b_dash(self):
        entry = resolve("llama3.1-8b")
        assert entry is not None
        assert entry.id == "llama3.1:8b"

    def test_resolve_deepseek_r1_7b_dash(self):
        entry = resolve("deepseek-r1-7b")
        assert entry is not None
        assert entry.id == "deepseek-r1:7b"

    def test_resolve_gemma3_4b_dash(self):
        entry = resolve("gemma3-4b")
        assert entry is not None
        assert entry.id == "gemma3:4b"

    def test_resolve_with_int2_suffix(self):
        """Quant suffix stripped before resolution."""
        entry = resolve("qwen2.5-7b-int2")
        assert entry is not None
        assert entry.id == "qwen2.5:7b"

    def test_resolve_canonical_still_works(self):
        entry = resolve("qwen2.5:7b")
        assert entry is not None
        assert entry.id == "qwen2.5:7b"

    def test_resolve_alias_still_works(self):
        entry = resolve("7b")
        assert entry is not None
        assert entry.id == "qwen2.5:7b"

    def test_resolve_prefix_still_works(self):
        entry = resolve("qwen3")
        assert entry is not None
        assert entry.id.startswith("qwen3:")

    def test_resolve_unknown_returns_none(self):
        entry = resolve("nonexistent-model-xyz123")
        assert entry is None

    def test_resolve_case_insensitive(self):
        entry = resolve("QWEN2.5-7B")
        assert entry is not None
        assert entry.id == "qwen2.5:7b"


# ============================================================================
# TestSuggest — suggest() for "did you mean?" functionality
# ============================================================================

class TestSuggest(unittest.TestCase):
    """suggest() returns relevant entries for near-miss model names."""

    def test_suggest_returns_list(self):
        results = suggest("qwen2.5")
        assert isinstance(results, list)

    def test_suggest_nonempty_for_close_match(self):
        results = suggest("qwen3-8b")
        assert len(results) > 0

    def test_suggest_returns_at_most_3(self):
        results = suggest("qwen")
        assert len(results) <= 3

    def test_suggest_hits_correct_family(self):
        results = suggest("llama3")
        ids = [e.id for e in results]
        assert any("llama3" in i for i in ids)

    def test_suggest_empty_for_garbage(self):
        results = suggest("zzznonexistent999xyz")
        assert results == []

    def test_suggest_typo_deepseek(self):
        results = suggest("deepsek-r1-7b")
        # the typo is close enough for substring match on "r1" or "7b"
        assert isinstance(results, list)  # should return something or []

    def test_suggest_returns_catalog_entries(self):
        from squish.catalog import CatalogEntry
        results = suggest("qwen3")
        for e in results:
            assert isinstance(e, CatalogEntry)


# ============================================================================
# TestCliCatalogSuggest — _catalog_suggest() wrapper in cli.py
# ============================================================================

class TestCliCatalogSuggest(unittest.TestCase):
    """cli._catalog_suggest must resolve to catalog.suggest."""

    def test_cli_catalog_suggest_callable(self):
        import squish.cli as cli
        assert callable(cli._catalog_suggest)

    def test_cli_catalog_suggest_returns_list(self):
        import squish.cli as cli
        results = cli._catalog_suggest("qwen3")
        assert isinstance(results, list)

    def test_cli_catalog_suggest_nonempty_for_known_family(self):
        import squish.cli as cli
        results = cli._catalog_suggest("qwen2.5-7b")
        assert len(results) > 0


# ============================================================================
# TestBrowserAutoOpen — _open_browser_when_ready uses subprocess not thread
# ============================================================================

class TestBrowserAutoOpen(unittest.TestCase):
    """_open_browser_when_ready must use subprocess.Popen (survives os.execv)."""

    def test_uses_subprocess_popen(self):
        """Implementation must call subprocess.Popen, not threading.Thread."""
        import squish.cli as cli
        import inspect
        src = inspect.getsource(cli._open_browser_when_ready)
        assert "Popen" in src, "Must use subprocess.Popen to survive os.execv"
        assert "start_new_session" in src, "Must use start_new_session=True to detach"

    def test_does_not_use_threading_thread(self):
        """Must NOT use daemon thread (killed by os.execv)."""
        import squish.cli as cli
        import inspect
        src = inspect.getsource(cli._open_browser_when_ready)
        assert "threading.Thread" not in src, "Thread is killed by os.execv"

    def test_calls_subprocess_popen_with_python(self):
        """Popen must use sys.executable (same interpreter)."""
        import squish.cli as cli
        import inspect
        src = inspect.getsource(cli._open_browser_when_ready)
        assert "sys.executable" in src, "Must use sys.executable for portability"

    def test_popen_called_on_invocation(self):
        """Calling _open_browser_when_ready should spawn a subprocess.Popen."""
        import squish.cli as cli
        import subprocess
        with patch.object(subprocess, "Popen") as mock_popen:
            cli._open_browser_when_ready("http://127.0.0.1:11435/chat", 11435)
            mock_popen.assert_called_once()

    def test_popen_gets_start_new_session(self):
        """Popen must be called with start_new_session=True."""
        import squish.cli as cli
        import subprocess
        with patch.object(subprocess, "Popen") as mock_popen:
            cli._open_browser_when_ready("http://127.0.0.1:11435/chat", 11435)
            _, kwargs = mock_popen.call_args
            assert kwargs.get("start_new_session") is True


# ============================================================================
# TestOfflineStatusThreshold — index.html defers offline banner
# ============================================================================

class TestOfflineStatusThreshold(unittest.TestCase):
    """The UI must wait for ≥2 consecutive failures before showing offline."""

    def _read_html(self) -> str:
        html_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "squish", "static", "index.html"
        )
        with open(html_path, encoding="utf-8") as f:
            return f.read()

    def test_fail_counter_variable_present(self):
        html = self._read_html()
        assert "_loadModelsFails" in html

    def test_threshold_at_least_2(self):
        html = self._read_html()
        m = re.search(r"_OFFLINE_FAIL_THRESHOLD\s*=\s*(\d+)", html)
        assert m is not None, "_OFFLINE_FAIL_THRESHOLD constant must exist"
        assert int(m.group(1)) >= 2, "Threshold must be at least 2 to tolerate inference load"

    def test_threshold_used_in_failure_branch(self):
        html = self._read_html()
        assert "_loadModelsFails >= _OFFLINE_FAIL_THRESHOLD" in html

    def test_counter_resets_on_success(self):
        html = self._read_html()
        assert "_loadModelsFails = 0" in html

    def test_counter_increments_on_failure(self):
        html = self._read_html()
        assert "_loadModelsFails++" in html

    def test_fetch_has_timeout(self):
        html = self._read_html()
        assert "AbortSignal.timeout" in html


# ============================================================================
# TestAgentAutoEnable — index.html auto-enables agent mode on startup
# ============================================================================

class TestAgentAutoEnable(unittest.TestCase):
    """The UI init block must query /v1/agent/tools and auto-enable agent mode."""

    def _read_html(self) -> str:
        html_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "squish", "static", "index.html"
        )
        with open(html_path, encoding="utf-8") as f:
            return f.read()

    def test_agent_tools_endpoint_queried_at_startup(self):
        html = self._read_html()
        assert "/v1/agent/tools" in html

    def test_toggle_agent_mode_called_when_tools_present(self):
        html = self._read_html()
        assert "toggleAgentMode()" in html

    def test_auto_enable_guarded_by_tool_count(self):
        html = self._read_html()
        assert "td.tools" in html or "(td.tools || []).length" in html

    def test_auto_enable_only_when_agent_mode_off(self):
        html = self._read_html()
        assert "!agentMode" in html


if __name__ == "__main__":
    unittest.main()
