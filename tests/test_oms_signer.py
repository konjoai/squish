"""tests/test_oms_signer.py — Unit tests for squish.squash.oms_signer.

Test taxonomy: Pure unit — no I/O, no real sigstore calls.  Tests that
OmsSigner.sign() behaves correctly when sigstore is absent or raises.

The real sigstore signing path is not tested here (requires OIDC ambient
credentials that are never available in CI).  The contract being enforced:
    1. ImportError for sigstore  → sign() returns None (never raises).
    2. Any exception from signing → sign() returns None (never raises).
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from squish.squash.oms_signer import OmsSigner


class TestNoSignstore:
    def test_returns_none_when_sigstore_not_importable(self, tmp_path: Path) -> None:
        bom_path = tmp_path / "cyclonedx-mlbom.json"
        bom_path.write_text('{"bomFormat": "CycloneDX"}')

        # Remove sigstore from sys.modules, block import with a None sentinel.
        with patch.dict(sys.modules, {"sigstore": None, "sigstore.sign": None}):
            result = OmsSigner.sign(bom_path)

        assert result is None

    def test_no_exception_propagated_when_sigstore_absent(self, tmp_path: Path) -> None:
        bom_path = tmp_path / "cyclonedx-mlbom.json"
        bom_path.write_text('{"bomFormat": "CycloneDX"}')

        with patch.dict(sys.modules, {"sigstore": None, "sigstore.sign": None}):
            # Must not raise.
            try:
                OmsSigner.sign(bom_path)
            except Exception as exc:  # pragma: no cover
                pytest.fail(f"OmsSigner.sign raised unexpectedly: {exc}")


class TestExceptionSwallow:
    def test_sign_exception_returns_none(self, tmp_path: Path) -> None:
        bom_path = tmp_path / "cyclonedx-mlbom.json"
        bom_path.write_text('{"bomFormat": "CycloneDX"}')

        # Sigstore importable but signing raises RuntimeError.
        mock_signer_module = MagicMock()
        mock_signer_module.Signer = MagicMock()
        mock_signer_module.SigningContext = MagicMock()
        mock_signer_module.SigningContext.production.return_value.signer.side_effect = (
            RuntimeError("OIDC token unavailable")
        )

        with patch.dict(sys.modules, {"sigstore": MagicMock(), "sigstore.sign": mock_signer_module}):
            result = OmsSigner.sign(bom_path)

        assert result is None

    def test_exception_not_propagated(self, tmp_path: Path) -> None:
        bom_path = tmp_path / "cyclonedx-mlbom.json"
        bom_path.write_text('{"bomFormat": "CycloneDX"}')

        mock_signer_module = MagicMock()
        mock_signer_module.Signer = MagicMock()
        mock_signer_module.SigningContext = MagicMock()
        mock_signer_module.SigningContext.production.return_value.signer.side_effect = (
            Exception("network timeout")
        )

        with patch.dict(sys.modules, {"sigstore": MagicMock(), "sigstore.sign": mock_signer_module}):
            try:
                OmsSigner.sign(bom_path)
            except Exception as exc:  # pragma: no cover
                pytest.fail(f"OmsSigner.sign raised unexpectedly: {exc}")
