"""
tests/test_wave119_dead_stub_purge.py

Wave 119 regression: confirm all 19 Wave-13/14 lazy stubs, the ToMe dead
globals/flags, and the lookahead-k dead flag are permanently deleted from
server.py and that no new dead try/except-import-pass stubs creep back in.

These are pure-unit, read-only tests — no I/O, no network, no MLX.
"""
import ast
import re
import sys
from pathlib import Path

import pytest

_SERVER_PY = Path(__file__).parent.parent / "squish" / "server.py"
_SERVER_TEXT = _SERVER_PY.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _server_lines() -> list[str]:
    return _SERVER_TEXT.splitlines()


# ---------------------------------------------------------------------------
# 1. Confirm the 19 deleted module paths are absent
# ---------------------------------------------------------------------------

_DELETED_MODULE_PATHS = [
    "squish.attention.duo_attention",
    "squish.attention.duo_decoding",
    "squish.kv.shadow_kv",
    "squish.kv.pq_cache",
    "squish.kv.spe_cache",
    "squish.token.knapspec",
    "squish.token.token_merging",
    "squish.token.token_swift",
    "squish.token.c2t",
    "squish.speculative.sub_spec",
    "squish.quant.dfloat11",
    "squish.quant.rans_codec",
    "squish.speculative.qspec",
    "squish.speculative.quant_spec",
    "squish.speculative.copy_spec",
    "squish.quant.squeeze_llm",
    "squish.quant.nf4_quant",
    "squish.quant.spin_quant",
    "squish.speculative.head_infer",
]


@pytest.mark.parametrize("mod_path", _DELETED_MODULE_PATHS)
def test_deleted_module_path_absent(mod_path: str) -> None:
    """Each deleted Wave-13/14 module path must not appear anywhere in server.py."""
    assert mod_path not in _SERVER_TEXT, (
        f"Dead import stub for '{mod_path}' was re-introduced into server.py"
    )


# ---------------------------------------------------------------------------
# 2. Confirm dead ToMe globals absent
# ---------------------------------------------------------------------------


def test_tome_config_global_absent() -> None:
    assert "_tome_config" not in _SERVER_TEXT, (
        "_tome_config was re-added; TokenMerging module does not exist"
    )


def test_tome_state_global_absent() -> None:
    assert "_tome_state" not in _SERVER_TEXT, (
        "_tome_state was re-added; TokenMerging module does not exist"
    )


# ---------------------------------------------------------------------------
# 3. Confirm dead ToMe flags absent
# ---------------------------------------------------------------------------


def test_tome_r_flag_absent() -> None:
    assert "--tome-r" not in _SERVER_TEXT


def test_tome_start_layer_flag_absent() -> None:
    assert "--tome-start-layer" not in _SERVER_TEXT


def test_tome_end_layer_flag_absent() -> None:
    assert "--tome-end-layer" not in _SERVER_TEXT


# ---------------------------------------------------------------------------
# 4. Confirm dead lookahead-k flag absent
# ---------------------------------------------------------------------------


def test_lookahead_k_flag_absent() -> None:
    assert "--lookahead-k" not in _SERVER_TEXT


# ---------------------------------------------------------------------------
# 5. No new always-failing try/except-import-pass stubs
#    A "dead stub" pattern is:  try: from X import Y # noqa: F401 \n except ImportError: pass
#    and X is not findable via importlib.  We scan for the pattern and check
#    that every such block targets a reachable module.
# ---------------------------------------------------------------------------


def _collect_try_import_pass_blocks(source: str) -> list[str]:
    """
    Scan source for the pattern:
        try:
            from <module_path> import <symbol>  # noqa: F401
        except ImportError:
            pass

    Returns a list of module paths found in such blocks.
    """
    # Match: try:\n<ws>from MODULE import SYMBOL  # noqa: F401\n<ws>except ImportError:\n<ws>pass
    pattern = re.compile(
        r"try:\s*\n\s*from\s+([\w.]+)\s+import\s+\w+\s*#\s*noqa:\s*F401\s*\n\s*except\s+ImportError:\s*\n\s*pass",
        re.MULTILINE,
    )
    return [m.group(1) for m in pattern.finditer(source)]


def test_no_new_dead_import_stubs() -> None:
    """Every remaining try/except-import-pass (noqa F401) block must resolve."""
    import importlib.util

    stub_modules = _collect_try_import_pass_blocks(_SERVER_TEXT)
    missing = []
    for mod in stub_modules:
        spec = importlib.util.find_spec(mod)
        if spec is None:
            missing.append(mod)
    assert not missing, (
        f"Dead try/except import stubs found in server.py for non-existent modules: {missing}\n"
        "Delete these stubs (the modules don't exist so they always silently fail)."
    )


# ---------------------------------------------------------------------------
# 6. Version is 9.6.0
# ---------------------------------------------------------------------------


def test_version_bumped() -> None:
    from squish import __version__
    assert __version__ == "9.6.0", f"Expected 9.6.0, got {__version__}"
