"""Unit tests for the bounded retry/backoff in the background catalog fetch.

Covers ``squish.catalog._fetch_catalog_bytes`` — transient network errors are
retried with backoff, non-transient errors are not, and success short-circuits.

Both ``opener`` and ``sleeper`` are injected directly (rather than patching the
shared globals ``urllib.request.urlopen`` / ``time.sleep``) so these tests are
fully isolated from any in-flight background catalog-refresh daemon thread
spawned by other tests.
"""
from __future__ import annotations

from unittest.mock import MagicMock

from squish import catalog


def _resp(data: bytes):
    """Build a urlopen-style context-manager mock returning *data*."""
    cm = MagicMock()
    cm.__enter__.return_value.read.return_value = data
    cm.__exit__.return_value = False
    return cm


class TestFetchCatalogBytes:
    def test_success_first_try(self):
        opener, sleeper = MagicMock(return_value=_resp(b"{}")), MagicMock()
        out = catalog._fetch_catalog_bytes(opener=opener, sleeper=sleeper)
        assert out == b"{}"
        assert opener.call_count == 1
        sleeper.assert_not_called()  # no backoff when the first attempt works

    def test_retries_then_succeeds(self):
        # Two transient OSErrors, then success on the third attempt.
        opener = MagicMock(side_effect=[OSError("dns"), OSError("timeout"), _resp(b'{"ok": 1}')])
        sleeper = MagicMock()
        out = catalog._fetch_catalog_bytes(max_attempts=3, opener=opener, sleeper=sleeper)
        assert out == b'{"ok": 1}'
        assert opener.call_count == 3
        assert sleeper.call_count == 2  # backoff slept between the three attempts

    def test_exhausts_and_returns_none(self):
        opener, sleeper = MagicMock(side_effect=OSError("down")), MagicMock()
        out = catalog._fetch_catalog_bytes(max_attempts=3, opener=opener, sleeper=sleeper)
        assert out is None
        assert opener.call_count == 3
        assert sleeper.call_count == 2  # no sleep after the final failed attempt

    def test_value_error_not_retried(self):
        opener, sleeper = MagicMock(side_effect=ValueError("bad url")), MagicMock()
        out = catalog._fetch_catalog_bytes(max_attempts=3, opener=opener, sleeper=sleeper)
        assert out is None
        assert opener.call_count == 1  # non-transient: no retry
        sleeper.assert_not_called()

    def test_backoff_is_exponential(self):
        opener = MagicMock(side_effect=[OSError("a"), OSError("b"), _resp(b"{}")])
        sleeper = MagicMock()
        catalog._fetch_catalog_bytes(max_attempts=3, opener=opener, sleeper=sleeper)
        base = catalog._CATALOG_FETCH_BACKOFF_BASE
        waited = [c.args[0] for c in sleeper.call_args_list]
        assert waited == [base, base * 2]
