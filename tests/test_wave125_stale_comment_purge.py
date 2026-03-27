"""Wave 125 — stale Wave 98 commented-out code purge (-11 lines).

Verifies that the commented-out Wave 98 FFN-sparsity-patching block
(referencing _SPARSITY_TRIM_AVAILABLE and _n98) is removed, and that
the surrounding live code is intact.
"""
import pathlib

SERVER = pathlib.Path(__file__).parent.parent / "squish" / "server.py"
SRC = SERVER.read_text(encoding="utf-8")
LINES = SRC.splitlines()


def test_sparsity_trim_commented_code_absent():
    assert "_SPARSITY_TRIM_AVAILABLE" not in SRC


def test_ffn_mask_patch_import_absent():
    assert "ffn_mask_patch" not in SRC


def test_n98_var_absent():
    """_n98 was the patched-layer count variable for the disabled block."""
    assert "_n98" not in SRC


def test_wave98_disabled_comment_absent():
    assert "Wave 98: FFN layer patching is disabled" not in SRC


def test_wave107_plan_comment_absent():
    assert "see plan Wave 107" not in SRC


def test_sparse_ffn_live_code_present():
    """The surrounding sparse-ffn feature (_sfn, _e82b, _w82_prof) must still work."""
    assert "_e82b" in SRC
    assert "_sfn" in SRC
    assert "_w82_prof" in SRC


def test_line_count():
    """server.py must be exactly 4702 lines after Wave 125 deletions."""
    count = len(LINES)
    assert count == 4702, (
        f"Expected 4702 lines after Wave 125 (-11 from 4713), got {count}"
    )
