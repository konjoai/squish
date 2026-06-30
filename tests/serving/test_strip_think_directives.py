"""
tests/serving/test_strip_think_directives.py

The reasoning soft-switch (``/think`` / ``/no_think`` / ``/nothink``) is a
prompt-control token consumed by the chat template — it must never surface in
a model's user-visible reply. ``strip_think_directives`` removes any directive
a model echoes back, on every API surface, without mangling legitimate prose.
"""

from __future__ import annotations

import pytest

from squish.serving.tool_calling import strip_think_directives


class TestRemovesDirectives:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("Hello /no_think world", "Hello world"),
            ("Here is the answer. /nothink", "Here is the answer."),
            ("/think Let me reason", "Let me reason"),
            ("/no-think variant", "variant"),
            ("done./no_think", "done."),
            ("a /think b /no_think c", "a b c"),
            ("/NoThink mixed case", "mixed case"),
        ],
    )
    def test_directive_stripped(self, text: str, expected: str):
        assert strip_think_directives(text) == expected


class TestPreservesLegitimateText:
    @pytest.mark.parametrize(
        "text",
        [
            "plain text no directive",
            "use TCP/IP and think about it",
            "path is /think/foo",  # path segment, not a directive
            "saved to foo/think",  # trailing path segment
            "rethink the approach",  # substring, not a directive
            "",
        ],
    )
    def test_unchanged(self, text: str):
        assert strip_think_directives(text) == text

    def test_no_slash_is_fast_path(self):
        # Identity for any text without a slash (no allocation/regex work).
        s = "a perfectly normal answer with no directives"
        assert strip_think_directives(s) is s
