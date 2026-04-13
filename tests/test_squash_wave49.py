"""tests/test_squash_wave49.py — Wave 49: air-gapped / offline mode.

Tests for:
  - _is_offline() env-var helper
  - OmsSigner.keygen() — Ed25519 keypair generation
  - OmsSigner.sign_local() — offline Ed25519 signing
  - OmsVerifier.verify_local() — offline Ed25519 verification
  - OmsSigner.sign() offline guard (SQUASH_OFFLINE=1 → returns None)
  - OmsSigner.pack_offline() — tarball bundling
  - AttestConfig offline fields
  - CLI: squash keygen, squash verify-local, squash pack-offline, squash attest --offline
  - API: POST /keygen, POST /attest/verify-local, POST /pack/offline
"""

from __future__ import annotations

import json
import os
import tarfile
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _write_bom(directory: Path) -> Path:
    """Write a minimal fake CycloneDX BOM for signing tests."""
    bom = directory / "cyclonedx-mlbom.json"
    bom.write_text(
        json.dumps({"bomFormat": "CycloneDX", "specVersion": "1.4", "components": []}),
        encoding="utf-8",
    )
    return bom


# ──────────────────────────────────────────────────────────────────────────────
# _is_offline
# ──────────────────────────────────────────────────────────────────────────────

class TestIsOffline:
    def test_offline_unset(self, monkeypatch):
        monkeypatch.delenv("SQUASH_OFFLINE", raising=False)
        from squish.squash.oms_signer import _is_offline
        assert _is_offline() is False

    def test_offline_set_1(self, monkeypatch):
        monkeypatch.setenv("SQUASH_OFFLINE", "1")
        from squish.squash.oms_signer import _is_offline
        assert _is_offline() is True

    def test_offline_set_true(self, monkeypatch):
        monkeypatch.setenv("SQUASH_OFFLINE", "true")
        from squish.squash.oms_signer import _is_offline
        assert _is_offline() is True

    def test_offline_set_yes(self, monkeypatch):
        monkeypatch.setenv("SQUASH_OFFLINE", "yes")
        from squish.squash.oms_signer import _is_offline
        assert _is_offline() is True

    def test_offline_set_0_is_false(self, monkeypatch):
        monkeypatch.setenv("SQUASH_OFFLINE", "0")
        from squish.squash.oms_signer import _is_offline
        assert _is_offline() is False

    def test_offline_set_false_is_false(self, monkeypatch):
        monkeypatch.setenv("SQUASH_OFFLINE", "false")
        from squish.squash.oms_signer import _is_offline
        assert _is_offline() is False

    def test_offline_empty_is_false(self, monkeypatch):
        monkeypatch.setenv("SQUASH_OFFLINE", "")
        from squish.squash.oms_signer import _is_offline
        assert _is_offline() is False


# ──────────────────────────────────────────────────────────────────────────────
# OmsSigner.keygen
# ──────────────────────────────────────────────────────────────────────────────

class TestOmsSignerKeygen:
    def test_generates_two_pem_files(self, tmp_path):
        from squish.squash.oms_signer import OmsSigner
        priv, pub = OmsSigner.keygen("test-key", tmp_path)
        assert priv.exists()
        assert pub.exists()

    def test_priv_filename(self, tmp_path):
        from squish.squash.oms_signer import OmsSigner
        priv, _ = OmsSigner.keygen("mykey", tmp_path)
        assert priv.name == "mykey.priv.pem"

    def test_pub_filename(self, tmp_path):
        from squish.squash.oms_signer import OmsSigner
        _, pub = OmsSigner.keygen("mykey", tmp_path)
        assert pub.name == "mykey.pub.pem"

    def test_priv_pem_header(self, tmp_path):
        from squish.squash.oms_signer import OmsSigner
        priv, _ = OmsSigner.keygen("k", tmp_path)
        assert b"PRIVATE KEY" in priv.read_bytes()

    def test_pub_pem_header(self, tmp_path):
        from squish.squash.oms_signer import OmsSigner
        _, pub = OmsSigner.keygen("k", tmp_path)
        assert b"PUBLIC KEY" in pub.read_bytes()

    def test_creates_key_dir(self, tmp_path):
        from squish.squash.oms_signer import OmsSigner
        key_dir = tmp_path / "nested" / "keys"
        assert not key_dir.exists()
        OmsSigner.keygen("k", key_dir)
        assert key_dir.exists()

    def test_returns_path_objects(self, tmp_path):
        from squish.squash.oms_signer import OmsSigner
        priv, pub = OmsSigner.keygen("k", tmp_path)
        assert isinstance(priv, Path)
        assert isinstance(pub, Path)

    def test_priv_not_same_as_pub(self, tmp_path):
        from squish.squash.oms_signer import OmsSigner
        priv, pub = OmsSigner.keygen("k", tmp_path)
        assert priv.read_bytes() != pub.read_bytes()


# ──────────────────────────────────────────────────────────────────────────────
# OmsSigner.sign_local
# ──────────────────────────────────────────────────────────────────────────────

class TestOmsSignerSignLocal:
    def test_writes_sig_file(self, tmp_path):
        from squish.squash.oms_signer import OmsSigner
        priv, _ = OmsSigner.keygen("k", tmp_path)
        bom = _write_bom(tmp_path)
        sig = OmsSigner.sign_local(bom, priv)
        assert sig.exists()

    def test_sig_file_suffix(self, tmp_path):
        from squish.squash.oms_signer import OmsSigner
        priv, _ = OmsSigner.keygen("k", tmp_path)
        bom = _write_bom(tmp_path)
        sig = OmsSigner.sign_local(bom, priv)
        assert sig.suffix == ".sig"

    def test_sig_returns_path(self, tmp_path):
        from squish.squash.oms_signer import OmsSigner
        priv, _ = OmsSigner.keygen("k", tmp_path)
        bom = _write_bom(tmp_path)
        sig = OmsSigner.sign_local(bom, priv)
        assert isinstance(sig, Path)

    def test_sig_content_is_hex(self, tmp_path):
        from squish.squash.oms_signer import OmsSigner
        priv, _ = OmsSigner.keygen("k", tmp_path)
        bom = _write_bom(tmp_path)
        sig = OmsSigner.sign_local(bom, priv)
        hex_str = sig.read_text(encoding="utf-8").strip()
        # Must be valid hex; Ed25519 sig = 64 bytes = 128 hex chars
        assert bytes.fromhex(hex_str)
        assert len(hex_str) == 128

    def test_missing_bom_raises(self, tmp_path):
        from squish.squash.oms_signer import OmsSigner
        priv, _ = OmsSigner.keygen("k", tmp_path)
        with pytest.raises(Exception):
            OmsSigner.sign_local(tmp_path / "nonexistent.json", priv)

    def test_missing_key_raises(self, tmp_path):
        from squish.squash.oms_signer import OmsSigner
        bom = _write_bom(tmp_path)
        with pytest.raises(Exception):
            OmsSigner.sign_local(bom, tmp_path / "no.priv.pem")


# ──────────────────────────────────────────────────────────────────────────────
# OmsVerifier.verify_local
# ──────────────────────────────────────────────────────────────────────────────

class TestOmsVerifierVerifyLocal:
    def _make_signed_bom(self, tmp_path):
        """Return (bom, priv, pub, sig)."""
        from squish.squash.oms_signer import OmsSigner
        priv, pub = OmsSigner.keygen("k", tmp_path)
        bom = _write_bom(tmp_path)
        sig = OmsSigner.sign_local(bom, priv)
        return bom, priv, pub, sig

    def test_valid_sig_returns_true(self, tmp_path):
        from squish.squash.oms_signer import OmsVerifier
        bom, _, pub, sig = self._make_signed_bom(tmp_path)
        assert OmsVerifier.verify_local(bom, pub, sig) is True

    def test_default_sig_path(self, tmp_path):
        from squish.squash.oms_signer import OmsVerifier
        bom, _, pub, _ = self._make_signed_bom(tmp_path)
        # sig_path defaults to bom.with_suffix('.sig')
        assert OmsVerifier.verify_local(bom, pub) is True

    def test_tampered_bom_returns_false(self, tmp_path):
        from squish.squash.oms_signer import OmsVerifier
        bom, _, pub, sig = self._make_signed_bom(tmp_path)
        bom.write_text('{"tampered": true}', encoding="utf-8")
        assert OmsVerifier.verify_local(bom, pub, sig) is False

    def test_wrong_key_returns_false(self, tmp_path):
        from squish.squash.oms_signer import OmsSigner, OmsVerifier
        bom, _, pub, sig = self._make_signed_bom(tmp_path)
        # Generate a different key
        _, wrong_pub = OmsSigner.keygen("wrong", tmp_path)
        assert OmsVerifier.verify_local(bom, wrong_pub, sig) is False

    def test_missing_sig_returns_false(self, tmp_path):
        from squish.squash.oms_signer import OmsVerifier, OmsSigner
        _, pub = OmsSigner.keygen("k2", tmp_path)
        bom = _write_bom(tmp_path)
        # No sig file written
        assert OmsVerifier.verify_local(bom, pub) is False

    def test_corrupted_sig_returns_false(self, tmp_path):
        from squish.squash.oms_signer import OmsVerifier
        bom, _, pub, sig = self._make_signed_bom(tmp_path)
        sig.write_text("0" * 128, encoding="utf-8")
        assert OmsVerifier.verify_local(bom, pub, sig) is False

    def test_roundtrip_two_keys(self, tmp_path):
        """Two independent keypairs both produce valid round-trips."""
        from squish.squash.oms_signer import OmsSigner, OmsVerifier
        priv1, pub1 = OmsSigner.keygen("k1", tmp_path)
        priv2, pub2 = OmsSigner.keygen("k2", tmp_path)
        bom = _write_bom(tmp_path)
        sig1 = OmsSigner.sign_local(bom, priv1)
        sig2 = tmp_path / "bom2.sig"
        # sign_local always writes to <bom>.sig; make a copy for second sig
        sig1_bytes = sig1.read_text(encoding="utf-8")
        from squish.squash.oms_signer import OmsSigner as _OMS
        priv2_loaded_sig = _OMS.sign_local(bom, priv2)
        priv2_loaded_sig.rename(sig2)
        sig1.write_text(sig1_bytes, encoding="utf-8")
        assert OmsVerifier.verify_local(bom, pub1, sig1) is True
        assert OmsVerifier.verify_local(bom, pub2, sig2) is True
        assert OmsVerifier.verify_local(bom, pub1, sig2) is False


# ──────────────────────────────────────────────────────────────────────────────
# OmsSigner.sign() offline guard
# ──────────────────────────────────────────────────────────────────────────────

class TestOmsSignerOfflineGuard:
    def test_sign_returns_none_when_offline(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SQUASH_OFFLINE", "1")
        from squish.squash.oms_signer import OmsSigner
        bom = _write_bom(tmp_path)
        result = OmsSigner.sign(bom)
        assert result is None

    def test_sign_does_not_call_sigstore_when_offline(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SQUASH_OFFLINE", "1")
        # patch sigstore import to raise to confirm it was never called
        import sys
        fake_sigstore = type(sys)("sigstore")
        fake_sigstore.sign = None
        with patch.dict("sys.modules", {"sigstore": None, "sigstore.sign": None}):
            from squish.squash.oms_signer import OmsSigner
            bom = _write_bom(tmp_path)
            # Should not raise even though sigstore is absent; offline guard fires first
            result = OmsSigner.sign(bom)
        assert result is None

    def test_sign_proceeds_when_not_offline(self, tmp_path, monkeypatch):
        monkeypatch.delenv("SQUASH_OFFLINE", raising=False)
        # sigstore not installed → returns None but does NOT hit the offline guard
        from squish.squash.oms_signer import OmsSigner
        bom = _write_bom(tmp_path)
        # Will return None (sigstore not installed) but must not raise
        result = OmsSigner.sign(bom)
        # None is acceptable here (sigstore absent in test env)
        assert result is None or isinstance(result, Path)


# ──────────────────────────────────────────────────────────────────────────────
# OmsSigner.pack_offline
# ──────────────────────────────────────────────────────────────────────────────

class TestPackOffline:
    def _make_model_dir(self, tmp_path: Path) -> Path:
        model_dir = tmp_path / "mymodel"
        model_dir.mkdir()
        _write_bom(model_dir)
        (model_dir / "weights.bin").write_bytes(b"\x00" * 64)
        return model_dir

    def test_creates_tar_gz(self, tmp_path):
        from squish.squash.oms_signer import OmsSigner
        model_dir = self._make_model_dir(tmp_path)
        bundle = OmsSigner.pack_offline(model_dir)
        assert bundle.exists()
        assert bundle.name.endswith(".squash-bundle.tar.gz")

    def test_bundle_is_valid_tar_gz(self, tmp_path):
        from squish.squash.oms_signer import OmsSigner
        model_dir = self._make_model_dir(tmp_path)
        bundle = OmsSigner.pack_offline(model_dir)
        assert tarfile.is_tarfile(bundle)

    def test_bundle_contains_model_files(self, tmp_path):
        from squish.squash.oms_signer import OmsSigner
        model_dir = self._make_model_dir(tmp_path)
        bundle = OmsSigner.pack_offline(model_dir)
        with tarfile.open(bundle, "r:gz") as tar:
            names = tar.getnames()
        assert any("cyclonedx-mlbom.json" in n for n in names)
        assert any("weights.bin" in n for n in names)

    def test_custom_output_path(self, tmp_path):
        from squish.squash.oms_signer import OmsSigner
        model_dir = self._make_model_dir(tmp_path)
        out = tmp_path / "custom.squash-bundle.tar.gz"
        bundle = OmsSigner.pack_offline(model_dir, out)
        assert bundle == out
        assert out.exists()

    def test_missing_model_dir_raises(self, tmp_path):
        from squish.squash.oms_signer import OmsSigner
        with pytest.raises(FileNotFoundError):
            OmsSigner.pack_offline(tmp_path / "nonexistent")

    def test_returns_path_object(self, tmp_path):
        from squish.squash.oms_signer import OmsSigner
        model_dir = self._make_model_dir(tmp_path)
        bundle = OmsSigner.pack_offline(model_dir)
        assert isinstance(bundle, Path)

    def test_auto_timestamp_in_name(self, tmp_path):
        from squish.squash.oms_signer import OmsSigner
        model_dir = self._make_model_dir(tmp_path)
        bundle = OmsSigner.pack_offline(model_dir)
        # Name should include model dir stem + timestamp pattern
        assert model_dir.name in bundle.name


# ──────────────────────────────────────────────────────────────────────────────
# AttestConfig offline fields
# ──────────────────────────────────────────────────────────────────────────────

class TestAttestConfigOfflineFields:
    def test_offline_defaults_false(self, tmp_path):
        from squish.squash.attest import AttestConfig
        cfg = AttestConfig(model_path=tmp_path)
        assert cfg.offline is False

    def test_offline_can_be_set(self, tmp_path):
        from squish.squash.attest import AttestConfig
        cfg = AttestConfig(model_path=tmp_path, offline=True)
        assert cfg.offline is True

    def test_local_signing_key_defaults_none(self, tmp_path):
        from squish.squash.attest import AttestConfig
        cfg = AttestConfig(model_path=tmp_path)
        assert cfg.local_signing_key is None

    def test_local_signing_key_can_be_set(self, tmp_path):
        from squish.squash.attest import AttestConfig
        key = tmp_path / "k.priv.pem"
        cfg = AttestConfig(model_path=tmp_path, local_signing_key=key)
        assert cfg.local_signing_key == key


# ──────────────────────────────────────────────────────────────────────────────
# CLI — squash keygen
# ──────────────────────────────────────────────────────────────────────────────

class TestCliKeygen:
    def _run(self, *args):
        """Run squash CLI via _build_parser and call the handler."""
        import sys
        from squish.squash.cli import _build_parser, _cmd_keygen
        parser = _build_parser()
        ns = parser.parse_args(list(args))
        return _cmd_keygen(ns, quiet=True)

    def test_keygen_exit_0(self, tmp_path):
        rc = self._run("keygen", "ci-key", "--key-dir", str(tmp_path))
        assert rc == 0

    def test_keygen_creates_files(self, tmp_path):
        self._run("keygen", "ci-key", "--key-dir", str(tmp_path))
        assert (tmp_path / "ci-key.priv.pem").exists()
        assert (tmp_path / "ci-key.pub.pem").exists()

    def test_keygen_priv_pem_content(self, tmp_path):
        self._run("keygen", "ci-key", "--key-dir", str(tmp_path))
        assert b"PRIVATE KEY" in (tmp_path / "ci-key.priv.pem").read_bytes()

    def test_keygen_pub_pem_content(self, tmp_path):
        self._run("keygen", "ci-key", "--key-dir", str(tmp_path))
        assert b"PUBLIC KEY" in (tmp_path / "ci-key.pub.pem").read_bytes()


# ──────────────────────────────────────────────────────────────────────────────
# CLI — squash verify-local
# ──────────────────────────────────────────────────────────────────────────────

class TestCliVerifyLocal:
    def _run(self, *args):
        from squish.squash.cli import _build_parser, _cmd_verify_local
        parser = _build_parser()
        ns = parser.parse_args(list(args))
        return _cmd_verify_local(ns, quiet=True)

    def _setup(self, tmp_path):
        from squish.squash.oms_signer import OmsSigner
        priv, pub = OmsSigner.keygen("k", tmp_path)
        bom = _write_bom(tmp_path)
        OmsSigner.sign_local(bom, priv)
        return bom, pub

    def test_valid_sig_exit_0(self, tmp_path):
        bom, pub = self._setup(tmp_path)
        rc = self._run("verify-local", str(bom), "--key", str(pub))
        assert rc == 0

    def test_missing_bom_exit_1(self, tmp_path):
        _, pub = self._setup(tmp_path)
        rc = self._run("verify-local", str(tmp_path / "no.json"), "--key", str(pub))
        assert rc == 1

    def test_missing_key_exit_1(self, tmp_path):
        bom, _ = self._setup(tmp_path)
        rc = self._run("verify-local", str(bom), "--key", str(tmp_path / "no.pub.pem"))
        assert rc == 1

    def test_tampered_bom_exit_2(self, tmp_path):
        bom, pub = self._setup(tmp_path)
        bom.write_text('{"tampered":true}', encoding="utf-8")
        rc = self._run("verify-local", str(bom), "--key", str(pub))
        assert rc == 2


# ──────────────────────────────────────────────────────────────────────────────
# CLI — squash pack-offline
# ──────────────────────────────────────────────────────────────────────────────

class TestCliPackOffline:
    def _run(self, *args):
        from squish.squash.cli import _build_parser, _cmd_pack_offline
        parser = _build_parser()
        ns = parser.parse_args(list(args))
        return _cmd_pack_offline(ns, quiet=True)

    def test_exit_0(self, tmp_path):
        model_dir = tmp_path / "m"
        model_dir.mkdir()
        _write_bom(model_dir)
        rc = self._run("pack-offline", str(model_dir))
        assert rc == 0

    def test_creates_bundle(self, tmp_path):
        model_dir = tmp_path / "m"
        model_dir.mkdir()
        _write_bom(model_dir)
        out = tmp_path / "out.squash-bundle.tar.gz"
        self._run("pack-offline", str(model_dir), "--output", str(out))
        assert out.exists()

    def test_missing_model_dir_exit_1(self, tmp_path):
        rc = self._run("pack-offline", str(tmp_path / "nowhere"))
        assert rc == 1


# ──────────────────────────────────────────────────────────────────────────────
# CLI — squash attest --offline parser
# ──────────────────────────────────────────────────────────────────────────────

class TestCliAttestOfflineArgs:
    def test_offline_flag_parsed(self, tmp_path):
        from squish.squash.cli import _build_parser
        parser = _build_parser()
        ns = parser.parse_args(["attest", str(tmp_path), "--offline"])
        assert ns.offline is True

    def test_offline_key_parsed(self, tmp_path):
        from squish.squash.cli import _build_parser
        parser = _build_parser()
        ns = parser.parse_args([
            "attest", str(tmp_path),
            "--offline", "--sign",
            "--offline-key", str(tmp_path / "k.priv.pem"),
        ])
        assert ns.offline_key == str(tmp_path / "k.priv.pem")

    def test_offline_defaults_false(self, tmp_path):
        from squish.squash.cli import _build_parser
        parser = _build_parser()
        ns = parser.parse_args(["attest", str(tmp_path)])
        assert ns.offline is False

    def test_offline_key_defaults_none(self, tmp_path):
        from squish.squash.cli import _build_parser
        parser = _build_parser()
        ns = parser.parse_args(["attest", str(tmp_path)])
        assert ns.offline_key is None


# ──────────────────────────────────────────────────────────────────────────────
# API — POST /keygen
# ──────────────────────────────────────────────────────────────────────────────

class TestApiKeygen:
    def test_keygen_creates_keys(self, tmp_path):
        from squish.squash.api import app
        from fastapi.testclient import TestClient
        client = TestClient(app)
        resp = client.post("/keygen", json={"key_name": "ci", "key_dir": str(tmp_path)})
        assert resp.status_code == 200
        data = resp.json()
        assert "priv_path" in data
        assert "pub_path" in data
        assert Path(data["priv_path"]).exists()
        assert Path(data["pub_path"]).exists()

    def test_keygen_response_shape(self, tmp_path):
        from squish.squash.api import app
        from fastapi.testclient import TestClient
        client = TestClient(app)
        resp = client.post("/keygen", json={"key_name": "k", "key_dir": str(tmp_path)})
        assert resp.status_code == 200
        keys = set(resp.json().keys())
        assert keys == {"priv_path", "pub_path"}


# ──────────────────────────────────────────────────────────────────────────────
# API — POST /attest/verify-local
# ──────────────────────────────────────────────────────────────────────────────

class TestApiVerifyLocal:
    def _setup(self, tmp_path):
        from squish.squash.oms_signer import OmsSigner
        priv, pub = OmsSigner.keygen("k", tmp_path)
        bom = _write_bom(tmp_path)
        sig = OmsSigner.sign_local(bom, priv)
        return bom, priv, pub, sig

    def test_valid_sig_ok_true(self, tmp_path):
        from squish.squash.api import app
        from fastapi.testclient import TestClient
        bom, _, pub, sig = self._setup(tmp_path)
        client = TestClient(app)
        resp = client.post("/attest/verify-local", json={
            "bom_path": str(bom),
            "pub_key_path": str(pub),
            "sig_path": str(sig),
        })
        assert resp.status_code == 200
        assert resp.json()["ok"] is True

    def test_missing_bom_404(self, tmp_path):
        from squish.squash.api import app
        from fastapi.testclient import TestClient
        _, _, pub, _ = self._setup(tmp_path)
        client = TestClient(app)
        resp = client.post("/attest/verify-local", json={
            "bom_path": str(tmp_path / "no.json"),
            "pub_key_path": str(pub),
        })
        assert resp.status_code == 404

    def test_tampered_bom_ok_false(self, tmp_path):
        from squish.squash.api import app
        from fastapi.testclient import TestClient
        bom, _, pub, sig = self._setup(tmp_path)
        bom.write_text('{"hacked": 1}', encoding="utf-8")
        client = TestClient(app)
        resp = client.post("/attest/verify-local", json={
            "bom_path": str(bom),
            "pub_key_path": str(pub),
            "sig_path": str(sig),
        })
        assert resp.status_code == 200
        assert resp.json()["ok"] is False


# ──────────────────────────────────────────────────────────────────────────────
# API — POST /pack/offline
# ──────────────────────────────────────────────────────────────────────────────

class TestApiPackOffline:
    def _make_model_dir(self, tmp_path):
        model_dir = tmp_path / "mymodel"
        model_dir.mkdir()
        _write_bom(model_dir)
        return model_dir

    def test_pack_offline_200(self, tmp_path):
        from squish.squash.api import app
        from fastapi.testclient import TestClient
        model_dir = self._make_model_dir(tmp_path)
        out = tmp_path / "out.squash-bundle.tar.gz"
        client = TestClient(app)
        resp = client.post("/pack/offline", json={
            "model_dir": str(model_dir),
            "output_path": str(out),
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["bundle_path"] == str(out)
        assert out.exists()

    def test_pack_offline_response_shape(self, tmp_path):
        from squish.squash.api import app
        from fastapi.testclient import TestClient
        model_dir = self._make_model_dir(tmp_path)
        out = tmp_path / "bundle.squash-bundle.tar.gz"
        client = TestClient(app)
        resp = client.post("/pack/offline", json={
            "model_dir": str(model_dir),
            "output_path": str(out),
        })
        keys = set(resp.json().keys())
        assert {"bundle_path", "size_bytes", "model_dir"} <= keys

    def test_pack_offline_size_nonzero(self, tmp_path):
        from squish.squash.api import app
        from fastapi.testclient import TestClient
        model_dir = self._make_model_dir(tmp_path)
        out = tmp_path / "b.squash-bundle.tar.gz"
        client = TestClient(app)
        resp = client.post("/pack/offline", json={
            "model_dir": str(model_dir),
            "output_path": str(out),
        })
        assert resp.json()["size_bytes"] > 0

    def test_missing_model_dir_404(self, tmp_path):
        from squish.squash.api import app
        from fastapi.testclient import TestClient
        client = TestClient(app)
        resp = client.post("/pack/offline", json={
            "model_dir": str(tmp_path / "nodir"),
        })
        assert resp.status_code == 404

    def test_bundle_is_valid_tar(self, tmp_path):
        from squish.squash.api import app
        from fastapi.testclient import TestClient
        model_dir = self._make_model_dir(tmp_path)
        out = tmp_path / "v.squash-bundle.tar.gz"
        client = TestClient(app)
        resp = client.post("/pack/offline", json={
            "model_dir": str(model_dir),
            "output_path": str(out),
        })
        assert resp.status_code == 200
        assert tarfile.is_tarfile(out)


# ──────────────────────────────────────────────────────────────────────────────
# Module count gate — no new squish/ Python files
# ──────────────────────────────────────────────────────────────────────────────

class TestModuleCount:
    def test_squish_module_count_unchanged(self):
        """squish/ must still have exactly 125 Python files (W51 adds drift.py — new security domain)."""
        squish_dir = Path(__file__).parent.parent / "squish"
        count = len(list(squish_dir.rglob("*.py")))
        assert count == 132, (
            f"Module count changed: expected 132, got {count}. "
            "W49-52 added oms_signer.py + 1; W54-56 added remediate.py, evaluator.py, edge_formats.py, chat.py; "
            "W57 added model_card.py. "
            "Check squish/ for unexpected additions or deletions."
        )
