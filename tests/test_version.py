"""
tests/test_version.py

Verify that squish.__version__ is consistent with the installed package
metadata and the pinned expected version.

Phase 5A Bug 3 requirement: add a CI test asserting:
    squish.__version__ == importlib.metadata.version("squish")
"""
from __future__ import annotations

import importlib
import importlib.metadata

import pytest

# Pinned expected version — update this whenever pyproject.toml version changes.
EXPECTED_VERSION = "9.0.0"


class TestVersionConsistency:
    def test_version_attribute_exists(self):
        """squish must expose a __version__ string attribute."""
        import squish
        assert hasattr(squish, "__version__")
        assert isinstance(squish.__version__, str)

    def test_version_is_expected(self):
        """squish.__version__ must match the pinned release string."""
        import squish
        assert squish.__version__ == EXPECTED_VERSION, (
            f"squish.__version__ is {squish.__version__!r}, "
            f"expected {EXPECTED_VERSION!r}.  Update __init__.py or EXPECTED_VERSION."
        )

    def test_version_matches_package_metadata(self):
        """
        When *squish* is installed (pip install -e . or pip install squish),
        squish.__version__ must equal importlib.metadata.version("squish").

        Skipped if the package is not installed (e.g. raw source checkout
        without an editable install).
        """
        try:
            meta_version = importlib.metadata.version("squish")
        except importlib.metadata.PackageNotFoundError:
            pytest.skip("squish is not installed; skipping metadata version check")
        import squish
        assert squish.__version__ == meta_version, (
            f"squish.__version__ ({squish.__version__!r}) disagrees with "
            f"importlib.metadata ({meta_version!r}).  "
            f"Re-run `pip install -e .` or update pyproject.toml."
        )

    def test_version_is_semver_like(self):
        """__version__ must be in MAJOR.MINOR.PATCH format."""
        import squish
        parts = squish.__version__.split(".")
        assert len(parts) == 3, f"Expected 3 version components, got: {squish.__version__!r}"
        for part in parts:
            assert part.isdigit(), (
                f"Version component {part!r} is not an integer in {squish.__version__!r}"
            )


# ---------------------------------------------------------------------------
# squish.__getattr__ — lazy import machinery (Phase 5A Bug 3 / __init__.py)
# ---------------------------------------------------------------------------

class TestLazyImport:
    """Exercise the module-level __getattr__ in squish/__init__.py."""

    def test_lazy_load_returns_class(self):
        """Accessing a registered name triggers lazy import and returns the class."""
        import squish
        cls = squish.QuantizedKVCache
        assert cls is not None
        assert hasattr(cls, "__init__")

    def test_lazy_load_caches_result(self):
        """Second access should return the same cached object (no re-import)."""
        import squish
        a = squish.QuantizedKVCache
        b = squish.QuantizedKVCache
        assert a is b

    def test_lazy_load_pull_model_alias(self):
        """pull_model is the catalog.pull alias — exercises the special-case path."""
        import squish
        fn = squish.pull_model
        # Should resolve to a callable (the pull function from squish.catalog)
        assert callable(fn)

    def test_lazy_load_resolve_model_alias(self):
        """resolve_model is the catalog.resolve alias."""
        import squish
        fn = squish.resolve_model
        assert callable(fn)

    def test_attribute_error_for_unknown_name(self):
        """Accessing a name not in _LAZY_IMPORTS raises AttributeError."""
        import squish
        with pytest.raises(AttributeError, match="has no attribute"):
            _ = squish._totally_nonexistent_squish_attr_xyz

    def test_all_list_includes_version(self):
        """__all__ must include __version__."""
        import squish
        assert "__version__" in squish.__all__

    def test_all_list_includes_lazy_names(self):
        """__all__ must include all lazily-importable names."""
        import squish
        from squish import _LAZY_IMPORTS
        for name in _LAZY_IMPORTS:
            assert name in squish.__all__, f"{name!r} missing from __all__"

    def test_kv_budget_broker_exported(self):
        """KVBudgetBroker must be reachable via squish.KVBudgetBroker."""
        import squish
        cls = squish.KVBudgetBroker
        assert hasattr(cls, "instance")

    def test_disk_kv_cache_exported(self):
        """DiskKVCache must be reachable via squish.DiskKVCache."""
        import squish
        cls = squish.DiskKVCache
        assert hasattr(cls, "__init__")
