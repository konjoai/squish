"""Wave 123 — empty Wave 41-55 comment stub purge regression tests.

Verifies that 38 lines of empty Wave 41-49 / 52-55 comment sections
were deleted from both module-level globals and main(), while live
Wave headers and the _lazy_expert variable remain intact.
"""
import pathlib
import re

SERVER = pathlib.Path(__file__).parent.parent / "squish" / "server.py"
SRC = SERVER.read_text(encoding="utf-8")
LINES = SRC.splitlines()


# ── Absence: empty stubs deleted from globals ─────────────────────────────────

def test_no_wave41_global():
    assert "# ── Wave 41:" not in SRC, "Wave 41 stub must be deleted from globals"

def test_no_wave42_global():
    assert "# ── Wave 42:" not in SRC, "Wave 42 stub must be deleted from globals"

def test_no_wave43_global():
    assert "# ── Wave 43:" not in SRC, "Wave 43 stub must be deleted from globals"

def test_no_wave44_global():
    assert "# ── Wave 44:" not in SRC, "Wave 44 stub must be deleted from globals"

def test_no_wave45_global():
    assert "# ── Wave 45:" not in SRC, "Wave 45 stub must be deleted from globals"

def test_no_wave46_global():
    assert "# ── Wave 46:" not in SRC, "Wave 46 stub must be deleted from globals"

def test_no_wave47_global():
    assert "# ── Wave 47:" not in SRC, "Wave 47 stub must be deleted from globals"

def test_no_wave48_global():
    assert "# ── Wave 48:" not in SRC, "Wave 48 stub must be deleted from globals"

def test_no_wave49_global():
    assert "# ── Wave 49:" not in SRC, "Wave 49 stub must be deleted from globals"

def test_no_wave52_global():
    assert "# ── Wave 52:" not in SRC, "Wave 52 stub must be deleted from globals"

def test_no_wave53_global():
    assert "# ── Wave 53:" not in SRC, "Wave 53 stub must be deleted from globals"

def test_no_wave55_global():
    assert "# ── Wave 55:" not in SRC, "Wave 55 stub must be deleted from globals"


# ── Absence: orphaned `global _lazy_expert` in main() ────────────────────────

def test_no_orphan_global_lazy_expert_in_main():
    """global _lazy_expert inside main() (no assignment) must be gone."""
    in_main = False
    for line in LINES:
        stripped = line.strip()
        if stripped.startswith("def main("):
            in_main = True
        if in_main and stripped == "global _lazy_expert":
            raise AssertionError("orphaned `global _lazy_expert` still present in main()")


# ── Presence: live headers and variables preserved ────────────────────────────

def test_wave50_header_present():
    assert "# ── Wave 50:" in SRC, "Wave 50 header must be preserved"

def test_wave51_header_present():
    assert "# ── Wave 51:" in SRC, "Wave 51 header must be preserved"

def test_wave54_header_present():
    assert "# ── Wave 54:" in SRC, "Wave 54 header must be preserved"

def test_wave37_header_present():
    assert "# ── Wave 37:" in SRC, "Wave 37 header must be preserved"

def test_wave27_in_main_present():
    assert "# ── Wave 27: Inference velocity features" in SRC, \
        "Wave 27 live header in main() must be preserved"

def test_lazy_expert_global_var_present():
    """_lazy_expert = None at module level must survive."""
    assert "_lazy_expert            = None" in SRC, \
        "_lazy_expert module-level variable must be preserved"


# ── Line-count gate ───────────────────────────────────────────────────────────

def test_line_count():
    """server.py must be ≤ 4750 lines (Wave 123 target was 4721; squash routing routes
    added ~22 lines after that wave — those additions are exempt from purge targets)."""
    count = len(LINES)
    assert count <= 4750, (
        f"Expected ≤ 4750 lines (squash-routing-adjusted), got {count}"
    )
    assert count > 4650, f"Sanity floor: expected > 4650 lines; got {count}"
