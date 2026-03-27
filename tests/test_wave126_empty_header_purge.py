"""Wave 126: purge 4 consecutive empty section-header comments inside main().

The following four headers had zero code between them and were deleted:
  # ── Attention and KV kernels ─────────────────────────────────────────────
  # ── KV cache strategies ──────────────────────────────────────────────────
  # ── Speculative decoding variants ─────────────────────────────────────────
  # ── Token-importance / adaptive-layer strategies ──────────────────────────
"""

from pathlib import Path

SERVER = Path(__file__).parent.parent / "squish" / "server.py"
SRC = SERVER.read_text()
LINES = SRC.splitlines()


def test_attention_kv_header_removed():
    assert "# ── Attention and KV kernels" not in SRC


def test_kv_cache_strategies_header_removed():
    assert "# ── KV cache strategies" not in SRC


def test_speculative_decoding_header_removed():
    assert "# ── Speculative decoding variants" not in SRC


def test_token_importance_header_removed():
    assert "# ── Token-importance / adaptive-layer strategies" not in SRC


def test_wave37_wire_header_preserved():
    """The Wave 37 header that followed the deleted stubs must still be present."""
    assert "# ── Wave 37: Wire Everything In" in SRC


def test_line_count():
    assert len(LINES) == 4698, f"Expected 4698 lines, got {len(LINES)}"
