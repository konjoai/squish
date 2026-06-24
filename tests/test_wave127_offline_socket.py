"""tests/test_wave127_offline_socket.py

Wave 127 — Web-UI offline banner during generation + socket.send() log spam.

Two release-blocking annoyances reported while testing small models:

  1. The web UI flashes an "offline" banner while a model is *thinking* — a
     /v1/models poll times out because prefill briefly starves the event loop.
     The chat is actively streaming from the server, so it is NOT offline; the
     banner must be suppressed while ``isStreaming`` is set.

  2. ``socket.send() raised exception.`` floods the log when a client closes the
     tab mid-stream — the streaming endpoints kept decoding and writing to a
     dead socket. They must bail on ``request.is_disconnected()``.

These are source-structure assertions (matching the style of the other wave
tests): they pin the guard in place so a future refactor can't silently drop it.
"""

from __future__ import annotations

import inspect
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ==============================================================================
# 1.  Offline banner suppressed while streaming
# ==============================================================================


def _index_html() -> str:
    return (ROOT / "squish" / "static" / "index.html").read_text(encoding="utf-8")


class TestOfflineBannerSuppressedWhileStreaming:
    """loadModels() must not flip to offline while a request is in flight."""

    def test_loadmodels_checks_isstreaming(self):
        html = _index_html()
        # Isolate the loadModels function body.
        m = re.search(r"async function loadModels\(\).*?\n}", html, re.DOTALL)
        assert m, "loadModels() not found in index.html"
        body = m.group(0)
        assert "isStreaming" in body, (
            "loadModels() must consult isStreaming so a poll timeout during "
            "generation does not show the offline banner"
        )

    def test_isstreaming_guard_precedes_offline_banner(self):
        html = _index_html()
        m = re.search(r"async function loadModels\(\).*?\n}", html, re.DOTALL)
        body = m.group(0)
        guard_pos = body.find("isStreaming")
        banner_pos = body.find("showOfflineBanner")
        assert guard_pos != -1 and banner_pos != -1
        assert guard_pos < banner_pos, (
            "the isStreaming early-return must come before showOfflineBanner"
        )

    def test_threshold_still_at_least_two(self):
        # The 2-failure tolerance is the secondary mitigation; keep it.
        html = _index_html()
        m = re.search(r"_OFFLINE_FAIL_THRESHOLD\s*=\s*(\d+)", html)
        assert m and int(m.group(1)) >= 2


# ==============================================================================
# 2.  Streaming endpoints bail on client disconnect
# ==============================================================================


@pytest.fixture(scope="module")
def _server_src():
    import squish.server as srv

    return srv


class TestDisconnectHandling:
    """Every long-lived streaming loop must check request.is_disconnected()."""

    def test_completions_stream_checks_disconnect(self, _server_src):
        src = inspect.getsource(_server_src.completions)
        assert "is_disconnected" in src, (
            "/v1/completions stream must stop decoding when the client drops"
        )

    def test_chat_completions_has_stop_mechanism(self, _server_src):
        # The chat path uses a producer/consumer _stop_evt rather than an inline
        # is_disconnected() check — assert that halt mechanism is still present.
        src = inspect.getsource(_server_src.chat_completions)
        assert "_stop_evt" in src

    def test_ollama_streams_check_disconnect(self):
        src = (ROOT / "squish" / "serving" / "ollama_compat.py").read_text(encoding="utf-8")
        # Both /api/generate and /api/chat streamers must guard the socket.
        assert src.count("is_disconnected") >= 2, (
            "both Ollama streaming generators must check request.is_disconnected()"
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
