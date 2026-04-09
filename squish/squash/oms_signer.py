"""squish/squash/oms_signer.py — OpenSSF Model Signing via Sigstore + offline Ed25519.

Phase 2 optional extra.  When ``sigstore`` is available,
:meth:`OmsSigner.sign` produces a Sigstore bundle file alongside the
CycloneDX BOM sidecar.

Deliberately *not* auto-called by Phase 1 — signing is an explicit opt-in
that requires OIDC ambient credentials (GitHub Actions, Workload Identity, or
an interactive browser flow).

Install sigstore separately after the squash extra::

    pip install "squish[squash]" sigstore

**Air-gapped / offline mode (W49)**

Set ``SQUASH_OFFLINE=1`` *or* pass ``--offline`` to ``squash attest`` to
disable all sigstore/OIDC network calls.  In offline mode, use the local
Ed25519 helpers instead:

* :func:`OmsSigner.keygen` — generate a keypair: ``<name>.priv.pem`` + ``<name>.pub.pem``
* :func:`OmsSigner.sign_local` — sign a BOM with a local private key
* :func:`OmsVerifier.verify_local` — verify a BOM against a local public key
* :func:`OmsSigner.pack_offline` — bundle a model dir + squash artefacts into a tarball
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

log = logging.getLogger(__name__)


def _is_offline() -> bool:
    """Return ``True`` when air-gapped mode is active.

    Activated by setting the environment variable ``SQUASH_OFFLINE`` to any
    non-empty, non-zero value (e.g. ``SQUASH_OFFLINE=1``).
    """
    val = os.environ.get("SQUASH_OFFLINE", "").strip()
    return bool(val) and val.lower() not in {"0", "false", "no", "off"}


class OmsSigner:
    """Sign a CycloneDX BOM sidecar using Sigstore.

    All methods are static — the class is a namespace, not a stateful object.
    """

    @staticmethod
    def sign(bom_path: Path) -> Path | None:
        """Sign *bom_path* and write ``<bom_path>.sig.json`` alongside it.

        Parameters
        ----------
        bom_path:
            Path to the ``cyclonedx-mlbom.json`` to sign.

        Returns
        -------
        Path
            ``<bom_path>.sig.json`` on success.
        None
            When sigstore is not installed, offline mode is active, or signing
            fails for any reason.  Never raises.
        """
        # Refuse network operations in air-gapped mode.
        if _is_offline():
            log.warning(
                "OmsSigner.sign: skipped — SQUASH_OFFLINE=1 (air-gapped mode). "
                "Use OmsSigner.sign_local() with a local Ed25519 key instead."
            )
            return None

        # Fast-fail when the optional dependency is absent.
        try:
            from sigstore.sign import Signer  # noqa: F401
        except ImportError:
            log.debug(
                "sigstore not installed — skipping OMS signing "
                "(install separately: pip install sigstore)"
            )
            return None

        # Attempt to sign; any error is non-fatal.
        try:
            from sigstore.sign import Signer, SigningContext  # noqa: F811

            bom_bytes = bom_path.read_bytes()
            with SigningContext.production().signer() as signer:
                result = signer.sign_artifact(input_=bom_bytes)

            sig_path = bom_path.with_name(bom_path.name + ".sig.json")
            sig_path.write_text(result.to_json())
            log.debug("OmsSigner: wrote bundle to %s", sig_path)
            return sig_path

        except Exception as exc:
            log.warning("OMS signing failed (non-fatal): %s", exc)
            return None

    @staticmethod
    def keygen(key_name: str, key_dir: str | Path = ".") -> tuple[Path, Path]:
        """Generate an Ed25519 keypair for offline signing.

        Writes ``<key_dir>/<key_name>.priv.pem`` (PKCS8, no passphrase) and
        ``<key_dir>/<key_name>.pub.pem`` (SubjectPublicKeyInfo).

        Parameters
        ----------
        key_name:
            Base filename for the keypair (no extension).
        key_dir:
            Directory to write the key files.  Created if absent.

        Returns
        -------
        tuple[Path, Path]
            ``(priv_pem_path, pub_pem_path)``

        Raises
        ------
        ImportError
            When the ``cryptography`` package is not installed.
        """
        try:
            from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
            from cryptography.hazmat.primitives import serialization
        except ImportError as exc:
            raise ImportError(
                "cryptography package required for offline keygen: "
                "pip install cryptography"
            ) from exc

        key_dir = Path(key_dir)
        key_dir.mkdir(parents=True, exist_ok=True)

        priv = Ed25519PrivateKey.generate()
        priv_path = key_dir / f"{key_name}.priv.pem"
        pub_path = key_dir / f"{key_name}.pub.pem"

        priv_path.write_bytes(
            priv.private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.PKCS8,
                serialization.NoEncryption(),
            )
        )
        pub_path.write_bytes(
            priv.public_key().public_bytes(
                serialization.Encoding.PEM,
                serialization.PublicFormat.SubjectPublicKeyInfo,
            )
        )
        log.debug("OmsSigner.keygen: wrote keypair to %s / %s", priv_path, pub_path)
        return priv_path, pub_path

    @staticmethod
    def sign_local(bom_path: Path, priv_key_path: Path) -> Path:
        """Sign *bom_path* with a local Ed25519 private key.

        The signature (raw 64 bytes, lower-hex encoded) is written to
        ``<bom_stem>.sig`` alongside the BOM.

        Parameters
        ----------
        bom_path:
            Path to the file to sign (typically ``cyclonedx-mlbom.json``).
        priv_key_path:
            Path to the ``*.priv.pem`` file generated by :meth:`keygen`.

        Returns
        -------
        Path
            Path to the written ``.sig`` file.

        Raises
        ------
        ImportError
            When the ``cryptography`` package is not installed.
        FileNotFoundError
            When *bom_path* or *priv_key_path* does not exist.
        """
        try:
            from cryptography.hazmat.primitives.serialization import load_pem_private_key
        except ImportError as exc:
            raise ImportError(
                "cryptography package required for offline signing: "
                "pip install cryptography"
            ) from exc

        bom_path = Path(bom_path)
        priv_key_path = Path(priv_key_path)

        priv = load_pem_private_key(priv_key_path.read_bytes(), password=None)
        raw_sig = priv.sign(bom_path.read_bytes())  # type: ignore[attr-defined]

        sig_path = bom_path.with_suffix(".sig")
        sig_path.write_text(raw_sig.hex(), encoding="utf-8")
        log.debug("OmsSigner.sign_local: wrote sig to %s", sig_path)
        return sig_path

    @staticmethod
    def pack_offline(
        model_dir: str | Path,
        output_path: str | Path | None = None,
    ) -> Path:
        """Bundle a model directory and its squash artefacts into a tarball.

        Creates a ``.squash-bundle.tar.gz`` archive containing the entire
        *model_dir* tree (weights, BOMs, SPDX, signatures, chain files, …).

        Parameters
        ----------
        model_dir:
            Path to the model directory to bundle.
        output_path:
            Destination ``.squash-bundle.tar.gz`` path.  When ``None``,
            defaults to ``<model_dir.parent>/<model_dir.name>-<timestamp>.squash-bundle.tar.gz``.

        Returns
        -------
        Path
            Path to the created archive.

        Raises
        ------
        FileNotFoundError
            When *model_dir* does not exist.
        """
        import datetime
        import tarfile

        model_dir = Path(model_dir)
        if not model_dir.exists():
            raise FileNotFoundError(f"model_dir not found: {model_dir}")

        if output_path is None:
            ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            output_path = (
                model_dir.parent / f"{model_dir.name}-{ts}.squash-bundle.tar.gz"
            )
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with tarfile.open(output_path, "w:gz") as tar:
            tar.add(model_dir, arcname=model_dir.name)

        log.debug(
            "OmsSigner.pack_offline: wrote bundle %s (%d bytes)",
            output_path,
            output_path.stat().st_size,
        )
        return output_path


class OmsVerifier:
    """Verify a Sigstore bundle against a CycloneDX BOM sidecar.

    All methods are static — the class is a namespace, not a stateful object.
    """

    @staticmethod
    def verify(bom_path: Path, bundle_path: Path | None = None) -> bool | None:
        """Verify the Sigstore bundle for *bom_path*.

        Parameters
        ----------
        bom_path:
            Path to the ``cyclonedx-mlbom.json`` whose signature to verify.
        bundle_path:
            Optional explicit path to the ``*.sig.json`` bundle.  When
            ``None``, defaults to ``<bom_path>.sig.json``.

        Returns
        -------
        True
            Bundle found and cryptographic verification passed.
        False
            Bundle found but verification FAILED (BOM or bundle was tampered).
        None
            No bundle file found — verification skipped gracefully.  This is
            *not* a failure; signing is entirely optional.
        """
        resolved = bundle_path or bom_path.with_name(bom_path.name + ".sig.json")
        if not resolved.exists():
            log.debug(
                "OmsVerifier: no bundle at %s — verification skipped", resolved
            )
            return None

        try:
            from sigstore.verify import Verifier  # noqa: F401
        except ImportError:
            log.debug(
                "sigstore not installed — cannot verify bundle "
                "(install separately: pip install sigstore)"
            )
            return None

        try:
            from sigstore.models import Bundle
            from sigstore.verify import Verifier

            bom_bytes = bom_path.read_bytes()
            bundle = Bundle.from_json(resolved.read_text())
            verifier = Verifier.production()
            verifier.verify_artifact(input_=bom_bytes, bundle=bundle)
            log.debug("OmsVerifier: verification PASSED for %s", bom_path)
            return True

        except Exception as exc:
            log.warning(
                "OmsVerifier: verification FAILED for %s — %s", bom_path, exc
            )
            return False

    @staticmethod
    def verify_local(
        bom_path: Path,
        pub_key_path: Path,
        sig_path: Path | None = None,
    ) -> bool:
        """Verify *bom_path* against a local Ed25519 signature.

        Parameters
        ----------
        bom_path:
            Path to the file whose signature to verify.
        pub_key_path:
            Path to the ``*.pub.pem`` file generated by :meth:`OmsSigner.keygen`.
        sig_path:
            Explicit path to the ``.sig`` file.  When ``None``, defaults to
            ``<bom_path>.sig`` (replacing the BOM's suffix with ``.sig``).

        Returns
        -------
        True
            Signature is cryptographically valid.
        False
            Signature is invalid, file was tampered, or required files are missing.

        Raises
        ------
        ImportError
            When the ``cryptography`` package is not installed.
        """
        try:
            from cryptography.hazmat.primitives.serialization import load_pem_public_key
            from cryptography.exceptions import InvalidSignature
        except ImportError as exc:
            raise ImportError(
                "cryptography package required for offline verification: "
                "pip install cryptography"
            ) from exc

        bom_path = Path(bom_path)
        pub_key_path = Path(pub_key_path)

        if sig_path is None:
            sig_path = bom_path.with_suffix(".sig")
        sig_path = Path(sig_path)

        if not sig_path.exists():
            log.debug("OmsVerifier.verify_local: no sig file at %s", sig_path)
            return False

        try:
            raw_sig = bytes.fromhex(sig_path.read_text(encoding="utf-8").strip())
            pub = load_pem_public_key(pub_key_path.read_bytes())
            pub.verify(raw_sig, bom_path.read_bytes())  # type: ignore[attr-defined]
            log.debug("OmsVerifier.verify_local: PASSED for %s", bom_path)
            return True
        except InvalidSignature:
            log.warning("OmsVerifier.verify_local: FAILED (invalid signature) for %s", bom_path)
            return False
        except Exception as exc:
            log.warning("OmsVerifier.verify_local: error for %s — %s", bom_path, exc)
            return False
