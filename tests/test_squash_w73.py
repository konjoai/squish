"""W73: Version bump to 9.14.0 — release seal tests."""
import importlib.metadata
import pytest


class TestSquishVersion:
    def test_version_is_9_14_0(self):
        version = importlib.metadata.version("squish")
        assert version == "9.14.0", f"Expected version 9.14.0, got {version}"

    def test_version_follows_semver(self):
        version = importlib.metadata.version("squish")
        parts = version.split(".")
        assert len(parts) == 3
        assert all(p.isdigit() for p in parts)

    def test_no_prerelease_suffix(self):
        version = importlib.metadata.version("squish")
        for suffix in ("dev", "rc", "alpha", "beta", "a0", "b0"):
            assert suffix not in version, f"Unexpected prerelease suffix in {version}"

    def test_version_major_is_9(self):
        version = importlib.metadata.version("squish")
        assert version.split(".")[0] == "9"
