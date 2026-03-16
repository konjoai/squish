"""
tests/test_catalog_ssl.py

Tests for SSL verification helpers in squish/catalog.py.
Covers _ssl_verify(), _apply_ssl_env(), _is_ssl_error(), and the
_SSLError raised by _hf_download on SSL failures.
"""
from __future__ import annotations

import os
import ssl
from unittest.mock import MagicMock, patch

import pytest

from squish.catalog import (
    _apply_ssl_env,
    _is_ssl_error,
    _ssl_verify,
    _SSLError,
)

# ── _ssl_verify ───────────────────────────────────────────────────────────────

class TestSslVerify:
    def _call(self, env: dict[str, str]) -> bool | str:
        """Call _ssl_verify with a specific env, cleaning up after."""
        relevant = (
            "SQUISH_VERIFY_SSL",
            "HF_HUB_DISABLE_SSL_VERIFICATION",
            "REQUESTS_CA_BUNDLE",
            "CURL_CA_BUNDLE",
        )
        saved = {k: os.environ.pop(k, None) for k in relevant}
        try:
            os.environ.update({k: v for k, v in env.items() if v is not None})
            return _ssl_verify()
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def test_default_is_true(self):
        assert self._call({}) is True

    def test_squish_false(self):
        for val in ("false", "False", "FALSE", "0", "no", "off"):
            assert self._call({"SQUISH_VERIFY_SSL": val}) is False

    def test_squish_true_explicit(self):
        assert self._call({"SQUISH_VERIFY_SSL": "true"}) is True
        assert self._call({"SQUISH_VERIFY_SSL": "1"}) is True

    def test_hf_hub_disable_flag(self):
        assert self._call({"HF_HUB_DISABLE_SSL_VERIFICATION": "1"}) is False
        assert self._call({"HF_HUB_DISABLE_SSL_VERIFICATION": "true"}) is False

    def test_requests_ca_bundle(self):
        result = self._call({"REQUESTS_CA_BUNDLE": "/etc/ssl/cert.pem"})
        assert result == "/etc/ssl/cert.pem"

    def test_curl_ca_bundle(self):
        result = self._call({"CURL_CA_BUNDLE": "/usr/local/cert.pem"})
        assert result == "/usr/local/cert.pem"

    def test_requests_ca_bundle_takes_priority_over_curl(self):
        result = self._call({
            "REQUESTS_CA_BUNDLE": "/a/cert.pem",
            "CURL_CA_BUNDLE": "/b/cert.pem",
        })
        assert result == "/a/cert.pem"

    def test_squish_false_overrides_ca_bundle(self):
        # SQUISH_VERIFY_SSL=false disables verification even if CA bundle is set
        result = self._call({
            "SQUISH_VERIFY_SSL": "false",
            "REQUESTS_CA_BUNDLE": "/some/cert.pem",
        })
        assert result is False


# ── _apply_ssl_env ────────────────────────────────────────────────────────────

class TestApplySslEnv:
    def test_sets_hf_hub_disable_when_false(self):
        to_clear = (
            "SQUISH_VERIFY_SSL", "HF_HUB_DISABLE_SSL_VERIFICATION",
            "HTTPX_VERIFY", "REQUESTS_CA_BUNDLE", "SSL_CERT_FILE",
        )
        saved = {k: os.environ.pop(k, None) for k in to_clear}
        try:
            os.environ["SQUISH_VERIFY_SSL"] = "false"
            _apply_ssl_env()
            assert os.environ.get("HF_HUB_DISABLE_SSL_VERIFICATION") == "1"
            assert os.environ.get("HTTPX_VERIFY") == "0"
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def test_sets_ca_bundle_when_path(self):
        to_clear = (
            "SQUISH_VERIFY_SSL", "HF_HUB_DISABLE_SSL_VERIFICATION",
            "HTTPX_VERIFY", "REQUESTS_CA_BUNDLE", "SSL_CERT_FILE",
        )
        saved = {k: os.environ.pop(k, None) for k in to_clear}
        try:
            os.environ["REQUESTS_CA_BUNDLE"] = "/path/to/ca.pem"
            _apply_ssl_env()
            assert os.environ.get("SSL_CERT_FILE") == "/path/to/ca.pem"
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def test_no_side_effects_when_default(self):
        to_clear = (
            "SQUISH_VERIFY_SSL", "HF_HUB_DISABLE_SSL_VERIFICATION",
            "HTTPX_VERIFY", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE", "SSL_CERT_FILE",
        )
        saved = {k: os.environ.pop(k, None) for k in to_clear}
        try:
            _apply_ssl_env()
            # verify=True → no env vars injected
            assert "HF_HUB_DISABLE_SSL_VERIFICATION" not in os.environ
            assert "HTTPX_VERIFY" not in os.environ
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v


# ── _is_ssl_error ─────────────────────────────────────────────────────────────

class TestIsSslError:
    def test_detects_certificate_verify_failed(self):
        exc = ssl.SSLCertVerificationError(
            "[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed"
        )
        assert _is_ssl_error(exc) is True

    def test_detects_via_cause_chain(self):
        inner = ssl.SSLCertVerificationError("CERTIFICATE_VERIFY_FAILED")
        outer = RuntimeError("connection failed")
        outer.__cause__ = inner
        assert _is_ssl_error(outer) is True

    def test_false_for_unrelated_error(self):
        exc = ValueError("Model not found")
        assert _is_ssl_error(exc) is False

    def test_false_for_none_like_error(self):
        exc = RuntimeError("something went wrong")
        assert _is_ssl_error(exc) is False

    def test_detects_ssl_error_name(self):
        exc = Exception("SSLError: handshake failed")
        assert _is_ssl_error(exc) is True


# ── _SSLError ─────────────────────────────────────────────────────────────────

class TestSSLError:
    def test_is_runtime_error(self):
        # Must be a RuntimeError so cli.py `except RuntimeError` catches it
        err = _SSLError("something")
        assert isinstance(err, RuntimeError)

    def test_message_preserved(self):
        msg = "SSL verify failed for repo xyz"
        err = _SSLError(msg)
        assert msg in str(err)
