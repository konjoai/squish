"""tests/test_wave128_ui_rep_penalty.py

Wave 128 — Web-UI repetition-penalty control (small-model loop prevention).

Small models (llama3.2:1b, qwen2.5:1.5b) degenerate into verbatim repetition on
some prompts (e.g. summarising a structured doc). The decode-time `_LoopGuard`
is a safety net — it *stops* a loop, but only after several repeats leak. The
real prevention is a repetition penalty applied during sampling.

The server already honours `repetition_penalty` in the request body (wired into
`mlx_lm`'s `make_logits_processors`), but the chat UI never sent it, so the
default was 1.0 (off). This pins the UI control in place: a slider defaulting to
1.1 (the Ollama/llama.cpp default) whose value is sent on every chat request.

Source-structure assertions over index.html, matching the other UI wave tests.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _index_html() -> str:
    return (ROOT / "squish" / "static" / "index.html").read_text(encoding="utf-8")


class TestRepetitionPenaltyControl:
    """The chat UI must expose a repetition-penalty control and send it."""

    def test_slider_present_with_default_1_1(self):
        html = _index_html()
        m = re.search(r'id="s-reppen"[^>]*value="([0-9.]+)"', html)
        assert m, "repetition-penalty slider (#s-reppen) missing from index.html"
        assert float(m.group(1)) == 1.1, (
            f"default repetition penalty should be 1.1, got {m.group(1)}"
        )

    def test_slider_min_is_one(self):
        # 1.0 must be reachable so users can disable the penalty entirely.
        html = _index_html()
        m = re.search(r'id="s-reppen"[^>]*min="([0-9.]+)"', html)
        assert m and float(m.group(1)) <= 1.0

    def test_generate_reads_the_slider(self):
        html = _index_html()
        assert "s-reppen" in html
        assert "repetition_penalty" in html, (
            "_doGenerate must read the slider into a repetition_penalty variable"
        )

    def test_request_body_includes_penalty(self):
        # The value must actually be attached to the chat request body.
        html = _index_html()
        assert re.search(r"body\.repetition_penalty\s*=\s*repetition_penalty", html), (
            "repetition_penalty must be added to the request body"
        )

    def test_penalty_only_sent_when_above_one(self):
        # Sending 1.0 is a no-op; guard the assignment so the API default (1.0)
        # is preserved when the user disables it.
        html = _index_html()
        assert re.search(r"repetition_penalty\s*>\s*1(\.0)?\b", html), (
            "the body should only carry repetition_penalty when > 1.0"
        )


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__, "-q"]))
