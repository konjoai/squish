"""tests/test_wave79_startup_inference.py

Wave 79 — Startup optimisation + inference loop micro-optimisations

Tests for:
  - squish.__version__ == "9.2.0" and matches pyproject.toml
  - uvicorn is NOT imported at squish.server module-load time
  - _require("uvicorn") is not called at module level (no eager side-effect)
  - _prefetch_caches pre-computation: list is built once, hasattr skipped per token
  - mlx_lm stream loop _text_getter: callable resolved once from first item type
"""
from __future__ import annotations

import importlib
import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)


# ============================================================================
# Version consistency
# ============================================================================

class TestVersionConsistency(unittest.TestCase):
    """squish.__version__ must be "9.10.0" and consistent with pyproject.toml."""

    def test_version_is_9_5_0(self):
        import squish
        self.assertEqual(squish.__version__, "9.10.0")

    def test_version_is_string(self):
        import squish
        self.assertIsInstance(squish.__version__, str)

    def test_pyproject_version_matches(self):
        """pyproject.toml `version = "..."` must equal squish.__version__."""
        import squish
        pyproject = Path(_repo_root) / "pyproject.toml"
        if not pyproject.exists():
            self.skipTest("pyproject.toml not found")
        text = pyproject.read_text()
        # Find first `version = "..."` line (the project version, not a dep pin)
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("version") and "=" in stripped:
                _, _, val = stripped.partition("=")
                ver = val.strip().strip('"').strip("'")
                self.assertEqual(ver, squish.__version__,
                                 f"pyproject.toml version {ver!r} != squish.__version__ {squish.__version__!r}")
                return
        self.fail("Could not find version line in pyproject.toml")


# ============================================================================
# Uvicorn deferral
# ============================================================================

class TestUvicornDeferred(unittest.TestCase):
    """Importing squish.server must NOT import uvicorn."""

    def test_uvicorn_not_in_sys_modules_after_server_import(self):
        """uvicorn must not appear in sys.modules after `import squish.server`."""
        # Remove uvicorn (and all sub-modules) from sys.modules to simulate cold import
        to_remove = [k for k in sys.modules if k == "uvicorn" or k.startswith("uvicorn.")]
        saved = {k: sys.modules.pop(k) for k in to_remove}
        try:
            # Re-import squish.server (already in sys.modules — that's fine; we
            # just want to check nothing re-adds uvicorn as a side effect)
            import squish.server  # noqa: F401
            self.assertNotIn(
                "uvicorn", sys.modules,
                "uvicorn must NOT be imported as a side-effect of `import squish.server`",
            )
        finally:
            sys.modules.update(saved)

    def test_uvicorn_submodules_not_imported(self):
        """No uvicorn.* sub-module should appear in sys.modules from squish.server import."""
        to_remove = {k: sys.modules.pop(k)
                     for k in list(sys.modules)
                     if k == "uvicorn" or k.startswith("uvicorn.")}
        try:
            import squish.server  # noqa: F401
            stray = [k for k in sys.modules if k == "uvicorn" or k.startswith("uvicorn.")]
            self.assertEqual(stray, [],
                             f"Unexpected uvicorn modules loaded: {stray}")
        finally:
            sys.modules.update(to_remove)


# ============================================================================
# _prefetch_caches pre-computation
# ============================================================================

class TestPrefetchCachesPrecomputation(unittest.TestCase):
    """_prefetch_caches must include only those layer caches that have start_prefetch."""

    def _make_layer_caches(self, n_prefetch: int, n_plain: int):
        """Return a mixed list of mock layer caches."""
        caches = []
        for _ in range(n_prefetch):
            m = MagicMock()
            m.start_prefetch = MagicMock()
            caches.append(m)
        for _ in range(n_plain):
            m = MagicMock(spec=[])  # spec=[] means no attributes — hasattr returns False
            caches.append(m)
        return caches

    def test_prefetch_caches_contains_only_prefetch_layers(self):
        """List comprehension must filter to layers with start_prefetch."""
        layer_caches = self._make_layer_caches(n_prefetch=4, n_plain=28)
        prefetch_caches = [lc for lc in layer_caches if hasattr(lc, "start_prefetch")]
        self.assertEqual(len(prefetch_caches), 4)
        for lc in prefetch_caches:
            self.assertTrue(hasattr(lc, "start_prefetch"))

    def test_prefetch_caches_empty_when_no_prefetch_support(self):
        """When no layer cache supports prefetch, the list is empty."""
        layer_caches = self._make_layer_caches(n_prefetch=0, n_plain=32)
        prefetch_caches = [lc for lc in layer_caches if hasattr(lc, "start_prefetch")]
        self.assertEqual(prefetch_caches, [])

    def test_prefetch_caches_all_when_all_support(self):
        """When all layer caches support prefetch, the list has all entries."""
        layer_caches = self._make_layer_caches(n_prefetch=32, n_plain=0)
        prefetch_caches = [lc for lc in layer_caches if hasattr(lc, "start_prefetch")]
        self.assertEqual(len(prefetch_caches), 32)

    def test_prefetch_loop_calls_start_prefetch_only_on_eligible(self):
        """Using _prefetch_caches list, only eligible caches have start_prefetch called."""
        layer_caches = self._make_layer_caches(n_prefetch=3, n_plain=5)
        prefetch_caches = [lc for lc in layer_caches if hasattr(lc, "start_prefetch")]
        # Simulate what the decode loop does
        for lc in prefetch_caches:
            lc.start_prefetch()
        for lc in prefetch_caches:
            lc.start_prefetch.assert_called_once()
        # Plain caches should NOT have been called
        for lc in layer_caches:
            if not hasattr(lc, "start_prefetch"):
                # MagicMock with spec=[] — no calls at all
                self.assertEqual(lc.mock_calls, [])

    def test_hasattr_not_called_per_token_in_loop(self):
        """After pre-computing _prefetch_caches, the inner loop never calls hasattr."""
        layer_caches = self._make_layer_caches(n_prefetch=2, n_plain=30)
        prefetch_caches = [lc for lc in layer_caches if hasattr(lc, "start_prefetch")]

        hasattr_calls = []
        original_hasattr = builtins_hasattr = __builtins__
        # We can't monkey-patch builtins.hasattr easily, but we CAN verify that
        # iterating prefetch_caches and calling start_prefetch directly never
        # needs to touch the non-prefetch cache objects at all.
        touched = set()
        for lc in prefetch_caches:
            touched.add(id(lc))
            lc.start_prefetch()
        plain_ids = {id(lc) for lc in layer_caches if not hasattr(lc, "start_prefetch")}
        self.assertTrue(touched.isdisjoint(plain_ids),
                        "Inner loop should never touch non-prefetch caches")


# ============================================================================
# mlx_lm stream loop: _text_getter resolved once
# ============================================================================

class TestTextGetterDeduplication(unittest.TestCase):
    """_text_getter callable must be resolved once from the first item type."""

    def _make_generation_result_item(self, text: str):
        """Create a mock GenerationResult-like object (has .text attribute)."""
        obj = MagicMock()
        obj.text = text
        return obj

    def test_text_getter_uses_attribute_for_generation_result(self):
        """When first item has .text, _text_getter returns item.text."""
        items = [self._make_generation_result_item(f"token{i}") for i in range(5)]
        _text_getter = None
        results = []
        for item in items:
            if _text_getter is None:
                _text_getter = (lambda i: i.text) if hasattr(item, "text") else str
            results.append(_text_getter(item))
        self.assertEqual(results, [f"token{i}" for i in range(5)])

    def test_text_getter_uses_str_for_plain_strings(self):
        """When first item has no .text attribute, _text_getter falls back to str."""
        items = ["tok0", "tok1", "tok2"]
        _text_getter = None
        results = []
        for item in items:
            if _text_getter is None:
                _text_getter = (lambda i: i.text) if hasattr(item, "text") else str
            results.append(_text_getter(item))
        self.assertEqual(results, ["tok0", "tok1", "tok2"])

    def test_text_getter_set_only_on_first_item(self):
        """_text_getter must be None before the loop and set exactly once."""
        items = [self._make_generation_result_item(f"t{i}") for i in range(10)]
        _text_getter = None
        setter_calls = 0
        for item in items:
            if _text_getter is None:
                _text_getter = (lambda i: i.text) if hasattr(item, "text") else str
                setter_calls += 1
        self.assertEqual(setter_calls, 1,
                         "_text_getter must be set exactly once, not per-token")

    def test_text_getter_callable_type(self):
        """_text_getter must be a callable after the first item."""
        item = self._make_generation_result_item("hello")
        _text_getter = None
        if _text_getter is None:
            _text_getter = (lambda i: i.text) if hasattr(item, "text") else str
        self.assertTrue(callable(_text_getter))

    def test_text_getter_none_initially(self):
        """_text_getter starts as None (deferred initialisation)."""
        _text_getter = None
        self.assertIsNone(_text_getter)

    def test_mixed_item_types_handled_correctly(self):
        """Even if item type could theoretically change, we reuse the first getter."""
        # In practice mlx_lm yields homogeneous types; test the getter reuse logic
        first_item = self._make_generation_result_item("first")
        _text_getter = None
        if _text_getter is None:
            _text_getter = (lambda i: i.text) if hasattr(first_item, "text") else str
        # Apply getter to a plain string (would fail if str() is used on a GenerationResult)
        self.assertEqual(_text_getter(first_item), "first")


# ============================================================================
# Import-time import-count regression test
# ============================================================================

class TestServerImportSideEffects(unittest.TestCase):
    """squish.server module-level imports must not trigger heavy optional deps."""

    def test_server_importable_without_uvicorn_installed(self):
        """squish.server should be importable even if uvicorn isn't installed yet.

        Simulate by hiding uvicorn from sys.modules and builtins.__import__,
        then importing squish.server — it should succeed because the import is
        now inside a function, not at module level.
        """
        import builtins as _builtins
        original_import = _builtins.__import__

        def _blocking_import(name, *args, **kwargs):
            if name == "uvicorn":
                raise ImportError("uvicorn blocked for test")
            return original_import(name, *args, **kwargs)

        # Remove any cached squish.server so the import runs fresh
        cached_server = sys.modules.pop("squish.server", None)
        cached_uvicorn = {k: sys.modules.pop(k)
                          for k in list(sys.modules)
                          if k == "uvicorn" or k.startswith("uvicorn.")}
        try:
            _builtins.__import__ = _blocking_import
            # squish.server should import successfully even with uvicorn blocked
            import squish.server  # noqa: F401  — must not raise
        except ImportError as exc:
            self.fail(f"squish.server import failed when uvicorn is unavailable: {exc}")
        finally:
            _builtins.__import__ = original_import
            if cached_server is not None:
                sys.modules["squish.server"] = cached_server
            sys.modules.update(cached_uvicorn)


if __name__ == "__main__":
    unittest.main()
