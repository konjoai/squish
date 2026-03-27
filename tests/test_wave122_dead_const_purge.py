"""Wave 122 regression tests — dead module-level constant purge.

Verifies that the four dead module-level constants removed in Wave 122
are absent from server.py, and that all live constants / infrastructure
they were adjacent to remain intact and untouched.

Dead constants removed (13 lines):
  1. _agent_kv_config definition  (L146 pre-wave)  + orphaned global decl
  2. _SEMANTIC_CACHE_CONFIG dict   (L244 pre-wave)  — 7 lines
  3. _compress_threshold           (L331 pre-wave)  — 1 line
  4. _TTY_ERR                      (L405 pre-wave)  — 1 line

Nothing else changed; test_version.py asserts v9.9.0.
"""

from __future__ import annotations

import ast
import pathlib
import re
import subprocess
import sys

# ── Fixtures ──────────────────────────────────────────────────────────────────
_ROOT = pathlib.Path(__file__).parent.parent
_SERVER = _ROOT / "squish" / "server.py"
_SRC = _SERVER.read_text(encoding="utf-8")
_LINES = _SRC.splitlines()


# ── 1. Absence assertions (deleted constants must not appear) ─────────────────

def test_agent_kv_config_definition_absent():
    """_agent_kv_config global definition was removed."""
    # Match the bare assignment form (not a comment or string literal)
    assigns = [l for l in _LINES if re.match(r"^_agent_kv_config\s*[=:]", l)]
    assert assigns == [], f"Found _agent_kv_config assignment(s): {assigns}"


def test_agent_kv_config_global_decl_absent():
    """Orphaned `global _agent_kv_config` in main() was removed."""
    global_decls = [l for l in _LINES if re.search(r"\bglobal\s+_agent_kv_config\b", l)]
    assert global_decls == [], f"Found global _agent_kv_config decl(s): {global_decls}"


def test_semantic_cache_config_absent():
    """_SEMANTIC_CACHE_CONFIG dict was removed; _semantic_cache instance is alive."""
    assigns = [l for l in _LINES if re.match(r"^_SEMANTIC_CACHE_CONFIG\s*[=:]", l)]
    assert assigns == [], f"Found _SEMANTIC_CACHE_CONFIG assignment(s): {assigns}"


def test_compress_threshold_absent():
    """_compress_threshold constant was removed."""
    assigns = [l for l in _LINES if re.match(r"^_compress_threshold\s*=", l)]
    assert assigns == [], f"Found _compress_threshold assignment(s): {assigns}"


def test_tty_err_absent():
    """_TTY_ERR constant was removed."""
    assigns = [l for l in _LINES if re.match(r"^_TTY_ERR\s*[=:]", l)]
    assert assigns == [], f"Found _TTY_ERR assignment(s): {assigns}"


def test_phase13a_agentKV_comment_in_main_absent():
    """Orphaned Phase 13A comment inside main() was removed."""
    inner = [l.strip() for l in _LINES if "Phase 13A: Asymmetric INT2 KV cache" in l]
    assert inner == [], f"Orphaned Phase 13A comment still present: {inner}"


# ── 2. Preserved live constants (must still exist) ────────────────────────────

def test_semantic_cache_instance_present():
    """_semantic_cache = None (the live instance var) is still present."""
    present = any(re.match(r"^_semantic_cache\s*=\s*None", l) for l in _LINES)
    assert present, "_semantic_cache = None missing — live variable accidentally removed"


def test_compress_enabled_present():
    """_compress_enabled is still present (adjacent to deleted _compress_threshold)."""
    present = any(re.match(r"^_compress_enabled\s*=", l) for l in _LINES)
    assert present, "_compress_enabled missing"


def test_tty_present():
    """_TTY (stdout) is still present; only _TTY_ERR (stderr) was dead."""
    present = any(re.match(r"^_TTY\s*:", l) for l in _LINES)
    assert present, "_TTY missing — live sibling accidentally removed"


def test_true_color_err_present():
    """_TRUE_COLOR_ERR (adjacent to deleted _TTY_ERR) is still present."""
    present = any(re.match(r"^_TRUE_COLOR_ERR\s*:", l) for l in _LINES)
    assert present, "_TRUE_COLOR_ERR missing"


def test_active_backend_present():
    """_active_backend is still present (was NOT dead — tested by phase_f tests)."""
    present = any(re.match(r"^_active_backend\s*[=:]", l) for l in _LINES)
    assert present, "_active_backend missing — live global accidentally removed"


def test_inference_backend_present():
    """_inference_backend is still present (live, read by COMPRESS_PATH routing)."""
    present = any(re.match(r"^_inference_backend\s*=", l) for l in _LINES)
    assert present, "_inference_backend missing"


def test_semantic_cache_feature_code_intact():
    """Semantic cache lookup/store code is still present (only config dict deleted)."""
    assert "_semantic_cache.lookup(" in _SRC, "semantic cache lookup removed"
    assert "_semantic_cache.store(" in _SRC, "semantic cache store removed"


def test_compress_path_comment_intact():
    """COMPRESS_PATH routing comment block still present (only the threshold var deleted)."""
    assert "COMPRESS_PATH" in _SRC, "COMPRESS_PATH routing comment removed"


# ── 3. Line-count gate ────────────────────────────────────────────────────────

def test_line_count_reduced_by_wave122():
    """server.py is smaller than v9.8.0 (4772) and larger than 4700 (sanity floor)."""
    n = len(_LINES)
    assert n < 4772, f"Expected < 4772 lines post-Wave-122; got {n}"
    assert n > 4700, f"Expected > 4700 lines (sanity floor); got {n}"


def test_line_count_wave122_delta():
    """Exactly 13 lines were removed in Wave 122 (4772 → 4759)."""
    n = len(_LINES)
    assert n == 4759, f"Expected 4759 lines; got {n} (delta from 4772 should be 13)"


# ── 4. No new dead constants introduced ───────────────────────────────────────

def test_no_new_dead_consts():
    """dead_consts_analysis sentinel: at most 1 false-positive (_active_backend)."""
    result = subprocess.run(
        [sys.executable, "dev/dead_consts_analysis.py"],
        capture_output=True, text=True,
        cwd=str(_ROOT),
    )
    assert result.returncode == 0, f"dead_consts_analysis failed: {result.stderr}"
    # Expect exactly the known 1 false-positive (_active_backend)
    lines = [l for l in result.stdout.splitlines() if l.startswith("  L")]
    assert len(lines) <= 1, (
        f"Expected ≤ 1 dead constant (known false-positive); found {len(lines)}: {lines}"
    )
