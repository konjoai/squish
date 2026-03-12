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
