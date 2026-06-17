"""
tests/agent/test_read_document.py

squish_read_document lets the agent analyse uploaded files across every UI.
It must handle text-like formats and .docx (stdlib-only) directly, degrade
gracefully when an optional PDF parser is missing, and never crash on bad input.
"""

from __future__ import annotations

import io
import zipfile

import pytest

from squish.agent.builtin_tools import (
    register_builtin_tools,
    squish_read_document,
)
from squish.agent.tool_registry import ToolRegistry


def test_reads_plain_text(tmp_path):
    f = tmp_path / "note.txt"
    f.write_text("built with grit\nline two\n")
    out = squish_read_document(str(f))
    assert "built with grit" in out
    assert "line two" in out
    assert "note.txt" in out  # header names the document


def test_reads_code_and_json(tmp_path):
    f = tmp_path / "data.json"
    f.write_text('{"konjo": true, "tier": 1}')
    out = squish_read_document(str(f))
    assert '"konjo": true' in out


def test_max_chars_caps_output(tmp_path):
    f = tmp_path / "big.txt"
    f.write_text("x" * 10_000)
    out = squish_read_document(str(f), max_chars=100)
    # Body (after the header) is capped at 100 chars, well under the 10k input.
    body = out.split("\n\n", 1)[1]
    assert len(body) <= 100
    assert "x" * 100 in body


def test_extracts_docx_without_python_docx(tmp_path):
    # Build a minimal .docx (a zip with word/document.xml) in-memory.
    doc_xml = (
        '<?xml version="1.0"?>'
        '<w:document xmlns:w="x"><w:body>'
        "<w:p><w:r><w:t>Hello from Konjo</w:t></w:r></w:p>"
        "<w:p><w:r><w:t>second paragraph</w:t></w:r></w:p>"
        "</w:body></w:document>"
    )
    f = tmp_path / "doc.docx"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("word/document.xml", doc_xml)
    f.write_bytes(buf.getvalue())

    out = squish_read_document(str(f))
    assert "Hello from Konjo" in out
    assert "second paragraph" in out
    # Paragraph boundary became a newline.
    assert "Hello from Konjo\n" in out


def test_pdf_without_parser_degrades_gracefully(tmp_path, monkeypatch):
    # Simulate neither pypdf nor pdfplumber being importable.
    import builtins

    real_import = builtins.__import__

    def _no_pdf(name, *args, **kwargs):
        if name in ("pypdf", "pdfplumber") or name.startswith(("pypdf.", "pdfplumber.")):
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_pdf)
    f = tmp_path / "report.pdf"
    f.write_bytes(b"%PDF-1.4 fake")
    out = squish_read_document(str(f))
    assert "pypdf" in out or "pdfplumber" in out  # actionable guidance, no crash


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        squish_read_document(str(tmp_path / "nope.txt"))


def test_registered_in_builtin_registry():
    reg = ToolRegistry()
    register_builtin_tools(reg)
    assert "squish_read_document" in reg.names()
