"""
tests/test_catalog_ssl.py

Tests for SSL verification helpers in squish/catalog.py.
Covers _ssl_verify(), _apply_ssl_env(), _is_ssl_error(), and the
_SSLError raised by _hf_download on SSL failures.

Squish never disables SSL verification. The only supported override is
a custom CA bundle via REQUESTS_CA_BUNDLE / CURL_CA_BUNDLE.
"""
from __future__ import annotations

import os
import ssl
import sys
import types
from pathlib import Path

import pytest

from squish.catalog import (
    _hf_download,
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


# ── _apply_ssl_env ────────────────────────────────────────────────────────────

class TestApplySslEnv:
    def test_sets_ca_bundle_when_path(self):
        to_clear = (
            "REQUESTS_CA_BUNDLE", "SSL_CERT_FILE", "CURL_CA_BUNDLE",
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
            "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE", "SSL_CERT_FILE",
        )
        saved = {k: os.environ.pop(k, None) for k in to_clear}
        try:
            _apply_ssl_env()
            # verify=True → no env vars injected
            assert "SSL_CERT_FILE" not in os.environ
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


# ── _hf_download SSL behaviour ────────────────────────────────────────────────

class TestHfDownloadSslBehaviour:
    def test_raises_ssl_error_on_ssl_failure(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """SSL failures surface as _SSLError with a CA-bundle-only fix message."""
        calls = {"n": 0}

        def _snapshot_download(**kwargs):
            calls["n"] += 1
            raise ssl.SSLCertVerificationError("CERTIFICATE_VERIFY_FAILED")

        fake_hf = types.SimpleNamespace(snapshot_download=_snapshot_download)
        monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf)

        with pytest.raises(_SSLError) as exc_info:
            _hf_download("owner/model", tmp_path, token=None)

        # Single strict attempt, no insecure retry.
        assert calls["n"] == 1

        # Error message must propose ONLY the secure fix (CA bundle).
        msg = str(exc_info.value)
        assert "REQUESTS_CA_BUNDLE" in msg
        # And must NEVER mention an insecure fallback.
        assert "SQUISH_VERIFY_SSL" not in msg
        assert "HF_HUB_DISABLE_SSL_VERIFICATION" not in msg
        assert "insecure" not in msg.lower() or "does not disable" in msg.lower()

    def test_uses_complete_local_copy_when_network_unavailable(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        (tmp_path / "config.json").write_text("{}", encoding="utf-8")
        (tmp_path / "model.safetensors").write_bytes(b"ok")

        def _snapshot_download(**kwargs):
            raise OSError("network unreachable")

        fake_hf = types.SimpleNamespace(snapshot_download=_snapshot_download)
        monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf)

        # Should not raise because a complete local model already exists.
        _hf_download("owner/model", tmp_path, token=None)

    def test_non_ssl_failure_raises_when_local_copy_is_incomplete(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        # Incomplete local copy: config exists but no weights.
        (tmp_path / "config.json").write_text("{}", encoding="utf-8")

        def _snapshot_download(**kwargs):
            raise OSError("network unreachable")

        fake_hf = types.SimpleNamespace(snapshot_download=_snapshot_download)
        monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf)

        with pytest.raises(OSError):
            _hf_download("owner/model", tmp_path, token=None)
