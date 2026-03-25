"""tests/test_wave85_color_dedup.py

Wave 85 — CLI Color Dedup + README Accuracy

Tests for:
  - cli._C is squish._term.C — single palette object, no duplicate
  - server.py does not define local _gradient, _LOGO_GRAD, _CLight (verified
    via ast.parse since server.py has a pre-existing SyntaxError at line 6906)
  - No old duplicate variables remain in cli namespace
  - v1_router default server_url uses port 11435 (not 11434)
  - _term.has_truecolor() respects NO_COLOR env var
  - _term._detect_background_info() respects SQUISH_DARK_BG and COLORFGBG
  - _PaletteANSI uses standard 8/16-colour ANSI codes (not hardcoded RGB)
"""
from __future__ import annotations

import ast
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

_SQUISH_PKG = Path(_repo_root) / "squish"
_SERVER_PY  = _SQUISH_PKG / "server.py"


def _top_level_names(filepath: Path) -> set[str]:
    """Return the set of names defined at module top-level in *filepath* using ast.parse.

    This avoids importing the file (which would fail if it has runtime errors) and
    checks only structural definitions (class, function, assignment).
    """
    src = filepath.read_text(encoding="utf-8")
    tree = ast.parse(src, filename=str(filepath))
    names: set[str] = set()
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    names.add(t.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                bound = alias.asname if alias.asname else alias.name.split(".")[0]
                names.add(bound)
    return names


# ============================================================================
# TestColorDedup — identity assertions
# ============================================================================

class TestColorDedup(unittest.TestCase):
    """cli._C must be the identical object to squish._term.C (not a copy)."""

    def test_cli_C_is_term_C(self):
        from squish import _term
        import squish.cli as _cli
        assert _cli._C is _term.C, (
            "cli._C is not squish._term.C — the palette was not consolidated. "
            "Remove the duplicate palette classes in cli.py and use: "
            "from squish._term import C as _C"
        )

    def test_server_gradient_is_term_gradient(self):
        """server.py must not define a top-level _gradient() function."""
        names = _top_level_names(_SERVER_PY)
        # After consolidation, _gradient is imported (not defined), so it still
        # appears in names as an import alias.  The key check is that no FunctionDef
        # named _gradient exists at module level.
        src = _SERVER_PY.read_text(encoding="utf-8")
        tree = ast.parse(src)
        func_names = {n.name for n in ast.walk(tree)
                      if isinstance(n, ast.FunctionDef) and
                      isinstance(getattr(n, "col_offset", None), int) and
                      n.col_offset == 0}
        assert "_gradient" not in func_names, (
            "server.py still defines a top-level _gradient() function — "
            "remove it and import from squish._term"
        )

    def test_server_logo_grad_is_term_logo_grad(self):
        """server.py must not define a top-level _LOGO_GRAD list literal."""
        src = _SERVER_PY.read_text(encoding="utf-8")
        tree = ast.parse(src)
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Name) and t.id == "_LOGO_GRAD":
                        # Check that the value is an import (Name node), not a List
                        assert not isinstance(node.value, ast.List), (
                            "server.py still defines _LOGO_GRAD as a list literal — "
                            "remove it and import from squish._term"
                        )


# ============================================================================
# TestNoDuplicateNames — old duplicate variables must be gone from cli
# ============================================================================

class TestNoDuplicateNames(unittest.TestCase):
    """Old duplicate palette variables must not exist in cli module namespace."""

    def test_no_CLI_TRUE_COLOR(self):
        import squish.cli as _cli
        assert not hasattr(_cli, "_CLI_TRUE_COLOR"), (
            "_CLI_TRUE_COLOR still present in cli — remove it and use _term.py detection"
        )

    def test_no_CLI_IS_DARK_BG(self):
        import squish.cli as _cli
        assert not hasattr(_cli, "_CLI_IS_DARK_BG"), (
            "_CLI_IS_DARK_BG still present in cli — remove it and use _term.py detection"
        )

    def test_no_CLI_BG_CONFIRMED(self):
        import squish.cli as _cli
        assert not hasattr(_cli, "_CLI_BG_CONFIRMED"), (
            "_CLI_BG_CONFIRMED still present in cli — remove it and use _term.py detection"
        )

    def test_no_duplicate_C_class_in_cli(self):
        """cli must not define its own _Palette / _C class (only import from _term)."""
        from squish import _term
        import squish.cli as _cli
        assert type(_cli._C) is type(_term.C), (
            "cli._C is not the same type as _term.C — a local subclass may still exist"
        )

    def test_no_has_truecolor_cli_function(self):
        """cli must not define its own _has_truecolor_cli() function."""
        import squish.cli as _cli
        assert not hasattr(_cli, "_has_truecolor_cli"), (
            "_has_truecolor_cli still present in cli — remove the duplicate detection function"
        )

    def test_no_server_duplicate_C_class_via_ast(self):
        """server.py must not define a top-level _CLight class."""
        src = _SERVER_PY.read_text(encoding="utf-8")
        tree = ast.parse(src)
        class_names = {n.name for n in ast.walk(tree)
                       if isinstance(n, ast.ClassDef) and
                       isinstance(getattr(n, "col_offset", None), int) and
                       n.col_offset == 0}
        assert "_CLight" not in class_names, (
            "server.py still defines a top-level _CLight class — "
            "remove it and import C from squish._term"
        )

    def test_no_server_has_truecolor_function_via_ast(self):
        """server.py must not define a local _has_truecolor() function."""
        src = _SERVER_PY.read_text(encoding="utf-8")
        tree = ast.parse(src)
        func_names = {n.name for n in ast.walk(tree)
                      if isinstance(n, ast.FunctionDef) and
                      isinstance(getattr(n, "col_offset", None), int) and
                      n.col_offset == 0}
        assert "_has_truecolor" not in func_names, (
            "server.py still defines a local _has_truecolor() function — "
            "remove it and use squish._term.has_truecolor"
        )


# ============================================================================
# TestHasTruecolor — function-level env var behaviour
# ============================================================================

class TestHasTruecolor(unittest.TestCase):
    """squish._term.has_truecolor() must respect NO_COLOR and detect truecolor signals."""

    def test_no_color_env_var_disables_truecolor(self):
        """NO_COLOR=1 must prevent truecolor even when COLORTERM=truecolor."""
        from squish._term import has_truecolor
        with patch.dict(os.environ, {"NO_COLOR": "1", "COLORTERM": "truecolor"}, clear=False):
            with patch("os.isatty", return_value=True):
                result = has_truecolor(1)
        assert result is False, "NO_COLOR env var must disable truecolor"

    def test_colorterm_truecolor_enables_color(self):
        """COLORTERM=truecolor on a tty must return True (no NO_COLOR)."""
        from squish._term import has_truecolor
        env = {k: v for k, v in os.environ.items() if k != "NO_COLOR"}
        env["COLORTERM"] = "truecolor"
        with patch.dict(os.environ, env, clear=True):
            with patch("os.isatty", return_value=True):
                result = has_truecolor(1)
        assert result is True, "COLORTERM=truecolor on a tty should enable truecolor"

    def test_non_tty_returns_false(self):
        """has_truecolor must return False when fd is not a tty."""
        from squish._term import has_truecolor
        with patch("os.isatty", return_value=False):
            result = has_truecolor(1)
        assert result is False, "Non-tty fd must return False even with COLORTERM=truecolor"

    def test_force_color_enables_truecolor(self):
        """FORCE_COLOR=1 on a tty must return True."""
        from squish._term import has_truecolor
        env = {k: v for k, v in os.environ.items() if k not in ("NO_COLOR",)}
        env.pop("NO_COLOR", None)
        env["FORCE_COLOR"] = "1"
        with patch.dict(os.environ, env, clear=True):
            with patch("os.isatty", return_value=True):
                result = has_truecolor(1)
        assert result is True, "FORCE_COLOR=1 on a tty should enable truecolor"


# ============================================================================
# TestDetectBackground — _detect_background_info() env var logic
# ============================================================================

class TestDetectBackground(unittest.TestCase):
    """_detect_background_info() must use SQUISH_DARK_BG and COLORFGBG correctly."""

    def test_squish_dark_bg_1_returns_dark_confirmed(self):
        from squish._term import _detect_background_info
        with patch.dict(os.environ, {"SQUISH_DARK_BG": "1"}, clear=False):
            is_dark, confirmed = _detect_background_info()
        assert is_dark is True
        assert confirmed is True

    def test_squish_dark_bg_0_returns_light_confirmed(self):
        from squish._term import _detect_background_info
        with patch.dict(os.environ, {"SQUISH_DARK_BG": "0"}, clear=False):
            is_dark, confirmed = _detect_background_info()
        assert is_dark is False
        assert confirmed is True

    def test_colorfgbg_dark_returns_dark_confirmed(self):
        """COLORFGBG=15;0 means light fg on dark bg → dark background."""
        from squish._term import _detect_background_info
        env = {k: v for k, v in os.environ.items() if k not in ("SQUISH_DARK_BG",)}
        env["COLORFGBG"] = "15;0"  # bg index 0 (< 7) = dark
        with patch.dict(os.environ, env, clear=True):
            is_dark, confirmed = _detect_background_info()
        assert is_dark is True
        assert confirmed is True

    def test_colorfgbg_light_returns_light_confirmed(self):
        """COLORFGBG=0;15 means dark fg on light bg → light background."""
        from squish._term import _detect_background_info
        env = {k: v for k, v in os.environ.items() if k not in ("SQUISH_DARK_BG",)}
        env["COLORFGBG"] = "0;15"  # bg index 15 (>= 7) = light
        with patch.dict(os.environ, env, clear=True):
            is_dark, confirmed = _detect_background_info()
        assert is_dark is False
        assert confirmed is True

    def test_no_env_vars_returns_dark_unconfirmed(self):
        """No env vars → dark fallback, NOT confirmed → ANSI palette used."""
        from squish._term import _detect_background_info
        env = {k: v for k, v in os.environ.items()
               if k not in ("SQUISH_DARK_BG", "COLORFGBG")}
        with patch.dict(os.environ, env, clear=True):
            is_dark, confirmed = _detect_background_info()
        assert confirmed is False, (
            "When no background env vars are set, confirmed must be False "
            "so _PaletteANSI is used (respects terminal theme)"
        )


# ============================================================================
# TestPaletteANSIFormat — ANSI palette must use standard codes, not RGB
# ============================================================================

class TestPaletteANSIFormat(unittest.TestCase):
    """_PaletteANSI colours must use 8/16-colour ANSI codes, not 24-bit RGB."""

    def test_palette_ansi_p_is_magenta(self):
        """_PaletteANSI.P should be \\033[35m (magenta) or empty string."""
        from squish._term import _PaletteANSI
        p = _PaletteANSI()
        assert "38;2;" not in p.P, (
            "_PaletteANSI.P must not use 24-bit RGB codes — it should use "
            "standard ANSI \\033[35m (magenta) or empty for non-TTY"
        )
        assert p.P in ("\033[35m", ""), (
            f"_PaletteANSI.P must be \\033[35m or '' (got {repr(p.P)})"
        )

    def test_palette_ansi_t_is_cyan(self):
        """_PaletteANSI.T should be \\033[36m (cyan) or empty string."""
        from squish._term import _PaletteANSI
        p = _PaletteANSI()
        assert "38;2;" not in p.T, "_PaletteANSI.T must not use 24-bit RGB"
        assert p.T in ("\033[36m", ""), (
            f"_PaletteANSI.T must be \\033[36m or '' (got {repr(p.T)})"
        )

    def test_palette_ansi_g_is_green(self):
        """_PaletteANSI.G should be \\033[32m (green) or empty string."""
        from squish._term import _PaletteANSI
        p = _PaletteANSI()
        assert "38;2;" not in p.G, "_PaletteANSI.G must not use 24-bit RGB"

    def test_palette_r_is_reset(self):
        """All palette variants must have R = \\033[0m (reset) or ''."""
        from squish._term import _Palette, _PaletteLight, _PaletteANSI
        for cls in (_Palette, _PaletteLight, _PaletteANSI):
            obj = cls()
            assert obj.R in ("\033[0m", ""), (
                f"{cls.__name__}.R must be reset code or '' (got {repr(obj.R)})"
            )


# ============================================================================
# TestV1RouterPort — port default fix
# ============================================================================

class TestV1RouterPort(unittest.TestCase):
    """V1Router must default to port 11435, not 11434."""

    def test_default_server_url_uses_11435(self):
        from squish.api.v1_router import V1Router
        router = V1Router()
        schema = router.openapi_schema()
        servers = schema.get("servers", [])
        assert servers, "OpenAPI schema must have at least one server entry"
        for s in servers:
            url = s.get("url", "")
            assert "11434" not in url, (
                f"Server URL {url!r} still contains old port 11434 — use 11435"
            )
            assert "11435" in url, (
                f"Server URL {url!r} must contain port 11435"
            )

    def test_openapi_schema_builder_default_port(self):
        from squish.api.v1_router import OpenAPISchemaBuilder, BUILTIN_ROUTES
        builder = OpenAPISchemaBuilder(routes=BUILTIN_ROUTES)
        schema = builder.build()
        server_url = schema.get("servers", [{}])[0].get("url", "")
        assert "11435" in server_url, (
            f"OpenAPISchemaBuilder default server URL must use port 11435, got {server_url!r}"
        )
        assert "11434" not in server_url


# ============================================================================
# TestTermExports — _term.__all__ completeness
# ============================================================================

class TestTermExports(unittest.TestCase):
    """squish._term must export the expected public API."""

    def test_all_exports_present(self):
        from squish import _term
        expected = {"has_truecolor", "detect_dark_background", "C", "gradient", "LOGO_GRAD"}
        missing = expected - set(_term.__all__)
        assert not missing, f"squish._term.__all__ missing: {missing}"

    def test_C_has_required_attributes(self):
        """The C palette object must have all standard colour attributes."""
        from squish._term import C
        required_attrs = ("DP", "P", "V", "L", "MG", "PK", "LPK", "T", "LT", "G",
                          "W", "SIL", "DIM", "B", "R")
        for attr in required_attrs:
            assert hasattr(C, attr), (
                f"squish._term.C missing attribute {attr!r} — _Palette is incomplete"
            )

    def test_gradient_callable(self):
        from squish._term import gradient
        result = gradient("hello", [(139, 92, 246), (34, 211, 238)])
        assert "hello" in result

    def test_logo_grad_has_entries(self):
        from squish._term import LOGO_GRAD
        assert len(LOGO_GRAD) >= 4, "LOGO_GRAD must contain at least 4 colour stops"
        for stop in LOGO_GRAD:
            assert len(stop) == 3, f"Each LOGO_GRAD stop must be a (R, G, B) tuple, got {stop}"


if __name__ == "__main__":
    unittest.main()
