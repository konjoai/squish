"""
tests/serving/test_tool_call_stream_filter.py

The agent loop must stream genuine reasoning text but never leak the
``<tool_call>`` syntax into a chat bubble (clients render a structured tool
card instead). ToolCallStreamFilter enforces that, token by token.
"""

from __future__ import annotations

from squish.serving.tool_calling import ToolCallStreamFilter


def _run(tokens: list[str]) -> str:
    """Feed tokens through the filter and return the concatenated visible text."""
    f = ToolCallStreamFilter()
    out = []
    for i, tok in enumerate(tokens):
        out.append(f.feed(tok, final=(i == len(tokens) - 1)))
    return "".join(out)


class TestPlainText:
    def test_plain_reasoning_passes_through(self):
        assert _run(["Hello", " ", "world", "."]) == "Hello world."

    def test_single_final_token(self):
        assert _run(["done"]) == "done"


class TestToolCallSuppression:
    def test_reasoning_before_tool_call_is_kept_syntax_dropped(self):
        toks = ["Let me ", "check.", "<tool_call>", '{"name": "x"}', "</tool_call>"]
        assert _run(toks) == "Let me check."

    def test_tool_call_with_no_reasoning_emits_nothing(self):
        toks = ["<tool_call>", '{"name": "squish_list_dir"}', "</tool_call>"]
        assert _run(toks) == ""

    def test_marker_split_across_tokens_does_not_leak(self):
        # "<tool_call>" arrives as "<tool" + "_call>" — the holdback must keep
        # the partial marker from being emitted as visible text.
        toks = ["Reasoning ", "<tool", "_call>", '{"name": "x"}']
        assert _run(toks) == "Reasoning "

    def test_everything_after_a_call_stays_suppressed(self):
        toks = ["a", "<tool_call>", "junk", "more junk", "trailing"]
        assert _run(toks) == "a"


class TestSuppressingFlag:
    def test_flag_flips_on_tool_call(self):
        f = ToolCallStreamFilter()
        f.feed("thinking ")
        assert f.suppressing is False
        f.feed("<tool_call>")
        assert f.suppressing is True


class TestHoldback:
    def test_no_text_lost_when_no_tool_call(self):
        # Trailing chars shorter than the marker are held back mid-stream, then
        # flushed on the final token — nothing is lost when no call appears.
        assert _run(["hel", "lo", "!"]) == "hello!"

    def test_holdback_does_not_emit_partial_marker_early(self):
        # Mid-stream a lone "<" must be held (could begin "<tool_call>").
        f = ToolCallStreamFilter()
        assert "<tool_call>"[:1] not in f.feed("<", final=False)
