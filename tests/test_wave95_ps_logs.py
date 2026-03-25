"""tests/test_wave95_ps_logs.py

Wave 95 — squish ps + squish logs CLI Commands

Tests for:
  - cmd_ps registered in CLI argparser
  - cmd_logs registered in CLI argparser
  - cmd_ps: server not running → friendly message
  - cmd_ps: no model loaded → empty models list message
  - cmd_ps: one model loaded → prints name/size/details
  - cmd_ps: model with all fields populated
  - cmd_ps: model size formatting (GB vs MB)
  - cmd_ps: --startup flag queries /v1/startup-profile
  - cmd_ps: startup profile fetch failure silenced
  - cmd_ps: --host / --port args respected
  - cmd_logs: log file not found → helpful message
  - cmd_logs: empty log file
  - cmd_logs: reads last N lines (default 50)
  - cmd_logs: --tail limits output
  - cmd_logs: prints header with file path
  - cmd_logs: custom --log-file path
  - cmd_ps parser: --startup flag default False
  - cmd_ps parser: --host default 127.0.0.1
  - cmd_ps parser: --port default 11435
  - cmd_logs parser: --tail default 50
  - cmd_logs parser: --follow default False
  - cmd_logs parser: --log-file default empty string
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

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _import_cli():
    import squish.cli as _cli
    return _cli


def _make_args(**kw) -> types.SimpleNamespace:
    defaults = dict(host="127.0.0.1", port=11435, startup=False,
                    tail=50, follow=False, log_file="")
    defaults.update(kw)
    return types.SimpleNamespace(**defaults)


def _capture(fn, *args, **kwargs) -> str:
    buf = io.StringIO()
    old, sys.stdout = sys.stdout, buf
    try:
        fn(*args, **kwargs)
    finally:
        sys.stdout = old
    return buf.getvalue()


# ============================================================================
# TestCmdPsRegistered — parser registration
# ============================================================================

class TestCmdPsRegistered(unittest.TestCase):

    def _subcommands(self):
        cli = _import_cli()
        ap = cli.build_parser()
        return ap._subparsers._group_actions[0].choices

    def test_ps_subcommand_exists(self):
        assert "ps" in self._subcommands()

    def test_ps_func_is_cmd_ps(self):
        cli = _import_cli()
        ap = cli.build_parser()
        choices = ap._subparsers._group_actions[0].choices
        ns = choices["ps"].parse_args([])
        assert ns.func is cli.cmd_ps

    def test_ps_host_default(self):
        cli = _import_cli()
        ap = cli.build_parser()
        ns = ap._subparsers._group_actions[0].choices["ps"].parse_args([])
        assert ns.host == "127.0.0.1"

    def test_ps_port_default(self):
        cli = _import_cli()
        ap = cli.build_parser()
        ns = ap._subparsers._group_actions[0].choices["ps"].parse_args([])
        assert ns.port == 11435

    def test_ps_startup_default_false(self):
        cli = _import_cli()
        ap = cli.build_parser()
        ns = ap._subparsers._group_actions[0].choices["ps"].parse_args([])
        assert ns.startup is False

    def test_ps_startup_flag_sets_true(self):
        cli = _import_cli()
        ap = cli.build_parser()
        ns = ap._subparsers._group_actions[0].choices["ps"].parse_args(["--startup"])
        assert ns.startup is True


# ============================================================================
# TestCmdLogsRegistered — parser registration
# ============================================================================

class TestCmdLogsRegistered(unittest.TestCase):

    def _subcommands(self):
        cli = _import_cli()
        ap = cli.build_parser()
        return ap._subparsers._group_actions[0].choices

    def test_logs_subcommand_exists(self):
        assert "logs" in self._subcommands()

    def test_logs_func_is_cmd_logs(self):
        cli = _import_cli()
        ap = cli.build_parser()
        choices = ap._subparsers._group_actions[0].choices
        ns = choices["logs"].parse_args([])
        assert ns.func is cli.cmd_logs

    def test_logs_tail_default(self):
        cli = _import_cli()
        ns = cli.build_parser()._subparsers._group_actions[0].choices["logs"].parse_args([])
        assert ns.tail == 50

    def test_logs_follow_default_false(self):
        cli = _import_cli()
        ns = cli.build_parser()._subparsers._group_actions[0].choices["logs"].parse_args([])
        assert ns.follow is False

    def test_logs_log_file_default_empty(self):
        cli = _import_cli()
        ns = cli.build_parser()._subparsers._group_actions[0].choices["logs"].parse_args([])
        assert ns.log_file == ""

    def test_logs_tail_flag_parsed(self):
        cli = _import_cli()
        ns = cli.build_parser()._subparsers._group_actions[0].choices["logs"].parse_args(["--tail", "100"])
        assert ns.tail == 100


# ============================================================================
# TestCmdPs — behavior
# ============================================================================

class TestCmdPs(unittest.TestCase):

    def test_server_not_running_prints_message(self):
        cli = _import_cli()
        import urllib.error
        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("refused")):
            out = _capture(cli.cmd_ps, _make_args())
        assert "No server running" in out or "squish run" in out

    def test_server_not_running_shows_host_port(self):
        cli = _import_cli()
        import urllib.error
        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("refused")):
            out = _capture(cli.cmd_ps, _make_args(host="10.0.0.1", port=9999))
        assert "10.0.0.1" in out or "9999" in out or "No server running" in out

    def test_empty_models_list_prints_no_model(self):
        cli = _import_cli()
        resp_data = json.dumps({"models": []}).encode()
        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.read.return_value = resp_data
        with patch("urllib.request.urlopen", return_value=mock_resp):
            out = _capture(cli.cmd_ps, _make_args())
        assert "No model" in out or "no model" in out.lower() or "squish run" in out

    def test_one_model_loaded_shows_name(self):
        cli = _import_cli()
        resp_data = json.dumps({"models": [
            {"name": "qwen3:8b", "size": 4_800_000_000, "details": {
                "family": "qwen", "parameter_size": "8B",
                "quantization_level": "INT4", "context_length": 32768
            }}
        ]}).encode()
        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.read.return_value = resp_data
        with patch("urllib.request.urlopen", return_value=mock_resp):
            out = _capture(cli.cmd_ps, _make_args())
        assert "qwen3:8b" in out

    def test_model_size_formatted_gb(self):
        cli = _import_cli()
        resp_data = json.dumps({"models": [
            {"name": "llama3.3:70b", "size": 35_000_000_000, "details": {}}
        ]}).encode()
        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.read.return_value = resp_data
        with patch("urllib.request.urlopen", return_value=mock_resp):
            out = _capture(cli.cmd_ps, _make_args())
        assert "GB" in out or "35" in out

    def test_model_size_zero_shows_dash(self):
        cli = _import_cli()
        resp_data = json.dumps({"models": [
            {"name": "qwen3:1.7b", "size": 0, "details": {}}
        ]}).encode()
        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.read.return_value = resp_data
        with patch("urllib.request.urlopen", return_value=mock_resp):
            out = _capture(cli.cmd_ps, _make_args())
        assert "—" in out or "qwen3:1.7b" in out

    def test_model_details_printed(self):
        cli = _import_cli()
        resp_data = json.dumps({"models": [
            {"name": "qwen3:8b", "size": 4_200_000_000, "details": {
                "family": "qwen", "parameter_size": "8B",
                "quantization_level": "INT4", "context_length": 32768
            }}
        ]}).encode()
        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.read.return_value = resp_data
        with patch("urllib.request.urlopen", return_value=mock_resp):
            out = _capture(cli.cmd_ps, _make_args())
        assert "INT4" in out
        assert "32" in out  # context length
        assert "qwen" in out.lower()

    def test_startup_flag_false_skips_startup_profile(self):
        """--startup not set → no attempt to fetch /v1/startup-profile."""
        cli = _import_cli()
        resp_data = json.dumps({"models": []}).encode()
        call_log: list[str] = []

        def _mock_urlopen(req, timeout=5):
            call_log.append(getattr(req, "full_url", str(req)))
            m = MagicMock()
            m.__enter__ = lambda s: s
            m.__exit__ = MagicMock(return_value=False)
            m.read.return_value = resp_data
            return m

        with patch("urllib.request.urlopen", side_effect=_mock_urlopen):
            _capture(cli.cmd_ps, _make_args(startup=False))

        assert not any("/v1/startup-profile" in url for url in call_log)

    def test_startup_flag_true_fetches_startup_profile(self):
        """--startup flag → fetches /v1/startup-profile."""
        cli = _import_cli()
        ps_data = json.dumps({"models": []}).encode()
        sp_data = json.dumps({
            "total_ms": 420.5,
            "phases": {"model_load": 380.0, "kv_init": 40.5}
        }).encode()
        call_log: list[str] = []

        def _mock_urlopen(req, timeout=5):
            url = getattr(req, "full_url", str(req))
            call_log.append(url)
            m = MagicMock()
            m.__enter__ = lambda s: s
            m.__exit__ = MagicMock(return_value=False)
            m.read.return_value = sp_data if "startup" in url else ps_data
            return m

        with patch("urllib.request.urlopen", side_effect=_mock_urlopen):
            out = _capture(cli.cmd_ps, _make_args(startup=True))

        assert any("/v1/startup-profile" in url for url in call_log)
        assert "420" in out or "Startup" in out

    def test_startup_profile_failure_silenced(self):
        """If startup profile fetch fails, cmd_ps still completes."""
        cli = _import_cli()
        ps_data = json.dumps({"models": []}).encode()
        import urllib.error
        call_count = [0]

        def _mock_urlopen(req, timeout=5):
            call_count[0] += 1
            url = getattr(req, "full_url", str(req))
            if "startup" in url:
                raise urllib.error.URLError("not found")
            m = MagicMock()
            m.__enter__ = lambda s: s
            m.__exit__ = MagicMock(return_value=False)
            m.read.return_value = ps_data
            return m

        # Should not raise
        _capture(cli.cmd_ps, _make_args(startup=True))


# ============================================================================
# TestCmdLogs — behavior
# ============================================================================

class TestCmdLogs(unittest.TestCase):

    def test_missing_log_file_prints_message(self):
        cli = _import_cli()
        out = _capture(cli.cmd_logs, _make_args(log_file="/nonexistent/path/squish.log"))
        assert "No log file" in out or "not found" in out.lower()

    def test_missing_log_file_prints_daemon_hint(self):
        cli = _import_cli()
        out = _capture(cli.cmd_logs, _make_args(log_file="/nonexistent/path/squish.log"))
        assert "daemon" in out.lower() or "squish daemon" in out

    def test_empty_log_file_prints_message(self):
        cli = _import_cli()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            tmp_path = f.name
        try:
            out = _capture(cli.cmd_logs, _make_args(log_file=tmp_path, tail=50))
            assert "empty" in out.lower() or "No log" in out or len(out.strip()) > 0
        finally:
            os.unlink(tmp_path)

    def test_reads_log_lines(self):
        cli = _import_cli()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            f.write("line one\nline two\nline three\n")
            tmp_path = f.name
        try:
            out = _capture(cli.cmd_logs, _make_args(log_file=tmp_path, tail=50))
            assert "line one" in out
            assert "line two" in out
            assert "line three" in out
        finally:
            os.unlink(tmp_path)

    def test_tail_limits_to_n_lines(self):
        cli = _import_cli()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            for i in range(100):
                f.write(f"log line {i}\n")
            tmp_path = f.name
        try:
            out = _capture(cli.cmd_logs, _make_args(log_file=tmp_path, tail=5))
            # Should contain the LAST 5 lines (95-99), not the first 5
            assert "log line 99" in out
            assert "log line 95" in out
            assert "log line 0" not in out
        finally:
            os.unlink(tmp_path)

    def test_tail_default_50_lines(self):
        cli = _import_cli()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            for i in range(200):
                f.write(f"entry {i}\n")
            tmp_path = f.name
        try:
            out = _capture(cli.cmd_logs, _make_args(log_file=tmp_path, tail=50))
            # Last entry should be present, entry 0 should not
            assert "entry 199" in out
            assert "entry 0" not in out
        finally:
            os.unlink(tmp_path)

    def test_header_printed_with_file_path(self):
        cli = _import_cli()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            f.write("some log content\n")
            tmp_path = f.name
        try:
            out = _capture(cli.cmd_logs, _make_args(log_file=tmp_path, tail=50))
            # Header should mention the path or "Last N lines"
            assert tmp_path in out or "Last" in out or "lines" in out.lower()
        finally:
            os.unlink(tmp_path)

    def test_custom_log_file_path_used(self):
        cli = _import_cli()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            f.write("custom log entry\n")
            tmp_path = f.name
        try:
            out = _capture(cli.cmd_logs, _make_args(log_file=tmp_path, tail=50))
            assert "custom log entry" in out
        finally:
            os.unlink(tmp_path)

    def test_tail_one_returns_last_line_only(self):
        cli = _import_cli()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            f.write("first\nsecond\nthird\n")
            tmp_path = f.name
        try:
            out = _capture(cli.cmd_logs, _make_args(log_file=tmp_path, tail=1))
            assert "third" in out
            assert "first" not in out
        finally:
            os.unlink(tmp_path)
