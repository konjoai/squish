"""Behavioral coverage for ``squish.agent.builtin_tools`` — file, shell, REPL,
fetch, web-search, and document-extraction tools, plus registration. Pure-
Python: tmp_path files, mocked urllib, in-process + faked-isolation REPL.
"""
from __future__ import annotations

import urllib.error

import pytest

from squish.agent import builtin_tools as bt
from squish.agent.tool_registry import ToolRegistry


def test_safe_path_rejects_null_bytes():
    with pytest.raises(ValueError, match="null bytes"):
        bt._safe_path("a\x00b")
    assert bt._safe_path("a//b/../c") == "a/c"


def test_read_file_window_and_header(tmp_path):
    p = tmp_path / "f.txt"
    p.write_text("l1\nl2\nl3\nl4\n")
    out = bt.squish_read_file(str(p), start_line=2, end_line=3)
    assert "l2\nl3" in out and "of 4 total" in out


def test_read_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        bt.squish_read_file(str(tmp_path / "nope.txt"))


def test_write_file_creates_parents(tmp_path):
    p = tmp_path / "sub" / "f.txt"
    msg = bt.squish_write_file(str(p), "hello")
    assert p.read_text() == "hello" and "5 bytes" in msg.replace(",", "")


def test_list_dir_entries(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "a.txt").write_text("xx")
    out = bt.squish_list_dir(str(tmp_path))
    assert "[DIR]  sub/" in out and "[FILE] a.txt" in out and "items total" in out


def test_list_dir_not_a_directory(tmp_path):
    f = tmp_path / "f"
    f.write_text("x")
    with pytest.raises(NotADirectoryError):
        bt.squish_list_dir(str(f))


def test_list_dir_getsize_error(tmp_path, monkeypatch):
    (tmp_path / "a.txt").write_text("x")
    monkeypatch.setattr(bt.os.path, "getsize",
                        lambda p: (_ for _ in ()).throw(OSError("stat fail")))
    out = bt.squish_list_dir(str(tmp_path))
    assert "(unknown size)" in out  # getsize OSError fallback (214-215)


def test_parse_ddg_snippet_outside_region():
    # Snippet appears BEFORE the link → not in the link's region → left empty.
    page = ('<td class="result-snippet">orphan</td>'
            '<a class="result-link" href="https://x.com">Title</a>')
    out = bt._parse_ddg_lite_results(page, 5)
    assert out == [("https://x.com", "Title", "")]  # 113→118 (no snippet found)


def test_run_shell_empty_raises():
    with pytest.raises(ValueError, match="non-empty"):
        bt.squish_run_shell("   ")


def test_run_shell_stdout_and_exit():
    out = bt.squish_run_shell("echo hello")
    assert "hello" in out and "[exit 0]" in out


def test_run_shell_stderr_and_nonzero():
    out = bt.squish_run_shell("echo oops >&2; exit 3")
    assert "[stderr]" in out and "oops" in out and "[exit 3]" in out


def test_run_shell_timeout(monkeypatch):
    import subprocess

    def _boom(*a, **k):
        raise subprocess.TimeoutExpired(cmd="x", timeout=1)

    monkeypatch.setattr(bt.subprocess, "run", _boom)
    assert "[TIMEOUT]" in bt.squish_run_shell("sleep 5", timeout=1)


def test_repl_empty_raises():
    with pytest.raises(ValueError, match="non-empty"):
        bt.squish_python_repl("  ")


def test_repl_in_process_paths():
    assert bt._repl_run_in_process("print('hi')", timeout=5).strip() == "hi"
    assert bt._repl_run_in_process("x = 1 + 1", timeout=5) == "[no output]"
    assert "[ERROR]" in bt._repl_run_in_process("raise ValueError('boom')", timeout=5)


def test_repl_falls_back_to_in_process(monkeypatch):
    monkeypatch.setattr(bt, "_repl_run_isolated", lambda *a, **k: None)
    assert bt.squish_python_repl("print(6*7)").strip() == "42"


def test_repl_isolated_runs_in_child():
    # Real spawned-child execution (slow but exercises the isolation path).
    out = bt.squish_python_repl("print('child-out')", timeout=10)
    assert "child-out" in out


def test_repl_isolated_child_error_status():
    # Child raises → worker sends ("err", traceback) → "[ERROR]" (363-364).
    out = bt.squish_python_repl("raise ValueError('child boom')", timeout=10)
    assert "[ERROR]" in out and "child boom" in out


def test_repl_isolated_unavailable_returns_none(monkeypatch):
    import multiprocessing
    monkeypatch.setattr(multiprocessing, "get_context",
                        lambda kind: (_ for _ in ()).throw(ValueError("no spawn")))
    assert bt._repl_run_isolated("print(1)", timeout=5, max_memory_mb=64) is None  # 330-332


def test_repl_isolated_spawn_failure_returns_none(monkeypatch):
    import multiprocessing
    real_ctx = multiprocessing.get_context("spawn")

    class _Ctx:
        def Pipe(self, duplex=False):
            return real_ctx.Pipe(duplex=duplex)

        def Process(self, **kw):
            class _P:
                def start(self_inner):
                    raise OSError("fork bomb blocked")
            return _P()

    monkeypatch.setattr(multiprocessing, "get_context", lambda kind: _Ctx())
    assert bt._repl_run_isolated("print(1)", timeout=5, max_memory_mb=64) is None  # 343-345


def test_repl_in_process_timeout():
    # Busy loop trips the SIGALRM-based timeout in the in-process path (381, 389).
    assert "[TIMEOUT]" in bt._repl_run_in_process("\nwhile True:\n    pass\n", timeout=1)


def test_repl_in_process_without_sigalrm(monkeypatch):
    import signal
    monkeypatch.delattr(signal, "SIGALRM", raising=False)
    # No SIGALRM → the alarm setup/teardown is skipped (378→385, 394→398).
    assert bt._repl_run_in_process("print('no-alarm')", timeout=5).strip() == "no-alarm"


def _fake_ctx(monkeypatch, *, alive=False, poll=False, recv=None, recv_raises=False):
    """Drive _repl_run_isolated's parent-side branches deterministically by
    faking the multiprocessing context (the real child outcomes — timeout,
    EOF, RLIMIT — are non-deterministic, especially on macOS)."""
    import multiprocessing

    class _Conn:
        def close(self):
            pass

        def poll(self):
            return poll

        def recv(self):
            if recv_raises:
                raise EOFError()
            return recv

    class _Proc:
        def __init__(self):
            self._alive = alive

        def start(self):
            pass

        def join(self, t=None):
            pass

        def is_alive(self):
            return self._alive

        def terminate(self):
            self._alive = False

    class _Ctx:
        def Pipe(self, duplex=False):
            return _Conn(), _Conn()

        def Process(self, **kw):
            return _Proc()

    monkeypatch.setattr(multiprocessing, "get_context", lambda kind: _Ctx())


def test_repl_isolated_timeout(monkeypatch):
    _fake_ctx(monkeypatch, alive=True)  # still alive after join → terminate (349-352)
    assert "[TIMEOUT]" in bt._repl_run_isolated("x", timeout=1, max_memory_mb=64)


def test_repl_isolated_eof_on_recv(monkeypatch):
    _fake_ctx(monkeypatch, poll=True, recv_raises=True)  # poll true but recv EOF (358-360)
    assert "terminated" in bt._repl_run_isolated("x", timeout=1, max_memory_mb=64)


def test_repl_isolated_memory_status(monkeypatch):
    _fake_ctx(monkeypatch, poll=True, recv=("mem", "partial"))  # 361-362
    out = bt._repl_run_isolated("x", timeout=1, max_memory_mb=64)
    assert "[MEMORY LIMIT EXCEEDED]" in out and "partial" in out


def test_repl_isolated_ok_and_empty(monkeypatch):
    _fake_ctx(monkeypatch, poll=True, recv=("ok", "result"))
    assert bt._repl_run_isolated("x", timeout=1, max_memory_mb=64) == "result"
    _fake_ctx(monkeypatch, poll=True, recv=("ok", ""))
    assert bt._repl_run_isolated("x", timeout=1, max_memory_mb=64) == "[no output]"


def test_repl_isolated_no_message_terminated(monkeypatch):
    _fake_ctx(monkeypatch, poll=False)  # child exited without sending (367-368)
    assert "terminated" in bt._repl_run_isolated("x", timeout=1, max_memory_mb=64)


class _Resp:
    def __init__(self, data: bytes):
        self._data = data

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self, n=None):
        return self._data[:n] if n is not None else self._data


def test_fetch_url_bad_scheme():
    with pytest.raises(ValueError, match="http/https"):
        bt.squish_fetch_url("file:///etc/passwd")


def test_fetch_url_no_host():
    with pytest.raises(ValueError, match="must include a host"):
        bt.squish_fetch_url("http://")


def test_fetch_url_success_and_truncation(monkeypatch):
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: _Resp(b"hello world"))
    assert bt.squish_fetch_url("https://example.com") == "hello world"
    # max_bytes smaller than content → truncation notice.
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: _Resp(b"abcdef"))
    out = bt.squish_fetch_url("https://example.com", max_bytes=3)
    assert out.startswith("abc") and "TRUNCATED" in out


def test_fetch_url_http_and_url_errors(monkeypatch):
    monkeypatch.setattr("urllib.request.urlopen",
                        lambda *a, **k: (_ for _ in ()).throw(
                            urllib.error.HTTPError("u", 404, "Not Found", {}, None)))
    assert "[HTTP 404]" in bt.squish_fetch_url("https://example.com")
    monkeypatch.setattr("urllib.request.urlopen",
                        lambda *a, **k: (_ for _ in ()).throw(urllib.error.URLError("dns")))
    assert "[URLError]" in bt.squish_fetch_url("https://example.com")


_DDG_HTML = (
    '<a class="result-link" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com">'
    "Example &amp; Title</a>\n"
    '<td class="result-snippet">A <b>snippet</b> here</td>'
)


def test_web_search_empty_raises():
    with pytest.raises(ValueError, match="non-empty"):
        bt.squish_web_search("  ")


def test_web_search_parses_results(monkeypatch):
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: _Resp(_DDG_HTML.encode()))
    out = bt.squish_web_search("python", max_results=3)
    assert "Example & Title" in out
    assert "https://example.com" in out
    assert "A snippet here" in out


def test_web_search_fallback_links(monkeypatch):
    # No result-link rows → fall back to scraping bare hrefs (excluding DDG's own).
    page = '<a href="https://realsite.com/page">x</a><a href="https://duckduckgo.com/y">z</a>'
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: _Resp(page.encode()))
    out = bt.squish_web_search("q")
    assert "https://realsite.com/page" in out and "duckduckgo.com/y" not in out


def test_web_search_no_results(monkeypatch):
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: _Resp(b"<html>empty</html>"))
    assert "No results found" in bt.squish_web_search("q")


def test_web_search_http_and_url_errors(monkeypatch):
    monkeypatch.setattr("urllib.request.urlopen",
                        lambda *a, **k: (_ for _ in ()).throw(
                            urllib.error.HTTPError("u", 503, "Down", {}, None)))
    assert "[HTTP 503]" in bt.squish_web_search("q")
    monkeypatch.setattr("urllib.request.urlopen",
                        lambda *a, **k: (_ for _ in ()).throw(urllib.error.URLError("x")))
    assert "[URLError]" in bt.squish_web_search("q")


def test_create_file_and_exists_guard(tmp_path):
    p = tmp_path / "n.txt"
    assert "Created" in bt.squish_create_file(str(p), "data")
    with pytest.raises(FileExistsError):
        bt.squish_create_file(str(p), "again")


def test_create_directory_and_file_conflict(tmp_path):
    d = tmp_path / "d"
    assert "Created directory" in bt.squish_create_directory(str(d))
    f = tmp_path / "f"
    f.write_text("x")
    with pytest.raises(FileExistsError):
        bt.squish_create_directory(str(f))


def test_delete_file(tmp_path):
    p = tmp_path / "d.txt"
    p.write_text("x")
    assert "Deleted" in bt.squish_delete_file(str(p))
    with pytest.raises(FileNotFoundError):
        bt.squish_delete_file(str(p))


def test_move_file(tmp_path):
    src = tmp_path / "s.txt"
    src.write_text("x")
    dst = tmp_path / "sub" / "d.txt"
    assert "Moved" in bt.squish_move_file(str(src), str(dst))
    assert dst.read_text() == "x"
    with pytest.raises(FileNotFoundError):
        bt.squish_move_file(str(tmp_path / "gone"), str(tmp_path / "x"))
    # dst exists guard
    src2 = tmp_path / "s2.txt"
    src2.write_text("y")
    with pytest.raises(FileExistsError):
        bt.squish_move_file(str(src2), str(dst))


def test_apply_edit(tmp_path):
    p = tmp_path / "e.txt"
    p.write_text("alpha beta gamma")
    assert "Applied edit" in bt.squish_apply_edit(str(p), "beta", "BETA")
    assert p.read_text() == "alpha BETA gamma"
    with pytest.raises(ValueError, match="not found"):
        bt.squish_apply_edit(str(p), "missing", "x")
    p.write_text("dup dup")
    with pytest.raises(ValueError, match="ambiguous"):
        bt.squish_apply_edit(str(p), "dup", "x")
    with pytest.raises(FileNotFoundError):
        bt.squish_apply_edit(str(tmp_path / "no.txt"), "a", "b")


def test_read_document_text(tmp_path):
    p = tmp_path / "doc.md"
    p.write_text("# Title\nbody")
    out = bt.squish_read_document(str(p))
    assert "# Document: doc.md" in out and "body" in out


def test_read_document_empty_text(tmp_path):
    p = tmp_path / "blank.txt"
    p.write_text("   ")
    assert "[no extractable text found]" in bt.squish_read_document(str(p))


def test_read_document_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        bt.squish_read_document(str(tmp_path / "no.txt"))
    d = tmp_path / "adir"
    d.mkdir()
    with pytest.raises(ValueError, match="Not a file"):
        bt.squish_read_document(str(d))


def test_read_document_docx(tmp_path):
    import zipfile
    p = tmp_path / "d.docx"
    body = '<w:p><w:r><w:t>Hello docx</w:t></w:r></w:p>'
    with zipfile.ZipFile(p, "w") as zf:
        zf.writestr("word/document.xml", f"<w:document>{body}</w:document>")
    out = bt.squish_read_document(str(p))
    assert "Hello docx" in out


def test_read_document_pdf_via_pypdf(tmp_path, monkeypatch):
    import sys
    import types
    fake = types.ModuleType("pypdf")

    class _Page:
        def extract_text(self):
            return "pdf page text"

    class PdfReader:
        def __init__(self, path):
            self.pages = [_Page()]

    fake.PdfReader = PdfReader
    monkeypatch.setitem(sys.modules, "pypdf", fake)
    p = tmp_path / "doc.pdf"
    p.write_bytes(b"%PDF-1.4")
    assert "pdf page text" in bt.squish_read_document(str(p))


def test_read_document_pdf_via_pdfplumber(tmp_path, monkeypatch):
    import builtins
    import sys
    import types
    real_import = builtins.__import__

    def _no_pypdf(name, *a, **k):
        if name == "pypdf":
            raise ImportError("pypdf")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", _no_pypdf)

    fake = types.ModuleType("pdfplumber")

    class _Page:
        def extract_text(self):
            return "plumber text"

    class _PDF:
        pages = [_Page()]

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    fake.open = lambda path: _PDF()
    monkeypatch.setitem(sys.modules, "pdfplumber", fake)
    p = tmp_path / "doc.pdf"
    p.write_bytes(b"%PDF-1.4")
    assert "plumber text" in bt.squish_read_document(str(p))


def test_read_document_pdf_without_parser(tmp_path, monkeypatch):
    import builtins
    real_import = builtins.__import__

    def _block(name, *a, **k):
        if name in ("pypdf", "pdfplumber"):
            raise ImportError(name)
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", _block)
    p = tmp_path / "doc.pdf"
    p.write_bytes(b"%PDF-1.4 fake")
    out = bt.squish_read_document(str(p))
    assert "requires `pypdf` or `pdfplumber`" in out


def test_register_builtin_tools():
    reg = ToolRegistry()
    bt.register_builtin_tools(reg)
    assert "squish_read_file" in reg and "squish_web_search" in reg
