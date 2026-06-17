"""Unit tests for the bounded retry/backoff in the background catalog fetch.

Covers ``squish.catalog._fetch_catalog_bytes`` — transient network errors are
retried with backoff, non-transient errors are not, and success short-circuits.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from squish import catalog


def _resp(data: bytes):
    """Build a urlopen-style context-manager mock returning *data*."""
    cm = MagicMock()
    cm.__enter__.return_value.read.return_value = data
    cm.__exit__.return_value = False
    return cm


class TestFetchCatalogBytes:
    def test_success_first_try(self):
        with patch("squish.catalog.urllib.request.urlopen", return_value=_resp(b"{}")) as uo, \
             patch("squish.catalog.time.sleep") as slp:
            out = catalog._fetch_catalog_bytes()
        assert out == b"{}"
        assert uo.call_count == 1
        slp.assert_not_called()  # no backoff when the first attempt works

    def test_retries_then_succeeds(self):
        # Two transient OSErrors, then success on the third attempt.
        side = [OSError("dns"), OSError("timeout"), _resp(b'{"ok": 1}')]
        with patch("squish.catalog.urllib.request.urlopen", side_effect=side) as uo, \
             patch("squish.catalog.time.sleep") as slp:
            out = catalog._fetch_catalog_bytes(max_attempts=3)
        assert out == b'{"ok": 1}'
        assert uo.call_count == 3
        assert slp.call_count == 2  # backoff slept between the three attempts

    def test_exhausts_and_returns_none(self):
        with patch("squish.catalog.urllib.request.urlopen", side_effect=OSError("down")) as uo, \
             patch("squish.catalog.time.sleep") as slp:
            out = catalog._fetch_catalog_bytes(max_attempts=3)
        assert out is None
        assert uo.call_count == 3
        assert slp.call_count == 2  # no sleep after the final failed attempt

    def test_value_error_not_retried(self):
        with patch("squish.catalog.urllib.request.urlopen", side_effect=ValueError("bad url")) as uo, \
             patch("squish.catalog.time.sleep") as slp:
            out = catalog._fetch_catalog_bytes(max_attempts=3)
        assert out is None
        assert uo.call_count == 1  # non-transient: no retry
        slp.assert_not_called()

    def test_backoff_is_exponential(self):
        side = [OSError("a"), OSError("b"), _resp(b"{}")]
        with patch("squish.catalog.urllib.request.urlopen", side_effect=side), \
             patch("squish.catalog.time.sleep") as slp:
            catalog._fetch_catalog_bytes(max_attempts=3)
        base = catalog._CATALOG_FETCH_BACKOFF_BASE
        waited = [c.args[0] for c in slp.call_args_list]
        assert waited == [base, base * 2]
