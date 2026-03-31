"""squish/squash/oms_signer.py — OpenSSF Model Signing via Sigstore.

Phase 2 optional extra.  When ``sigstore`` is available,
:meth:`OmsSigner.sign` produces a Sigstore bundle file alongside the
CycloneDX BOM sidecar.

Deliberately *not* auto-called by Phase 1 — signing is an explicit opt-in
that requires OIDC ambient credentials (GitHub Actions, Workload Identity, or
an interactive browser flow).

Install sigstore separately after the squash extra::

    pip install "squish[squash]" sigstore
"""

from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger(__name__)


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
            When sigstore is not installed or signing fails for any reason.
            Never raises.
        """
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
