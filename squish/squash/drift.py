"""squish/squash/drift.py — SBOM drift detection for squish compressed models.

Compares SHA-256 digests recorded in a CycloneDX BOM sidecar against the
actual files on disk.  Allows security and compliance teams to detect
post-deployment file tampering or silent model swaps.

BOM format
----------
The reference BOM is the ``cyclonedx-mlbom.json`` sidecar produced by
:mod:`squish.squash.sbom_builder`.  Per-file weight hashes are stored as
component ``properties`` entries whose ``name`` starts with the prefix
``squish:weight_hash:`` and whose ``value`` is the lowercase hex SHA-256
digest of that file.  For example::

    {
      "name": "squish:weight_hash:tensors/model.0.weight.npy",
      "value": "e3b0c44298fc1c14..."
    }

The function :func:`check_drift` reads these properties, computes fresh
digests from disk, and returns a :class:`DriftResult` summarising any
divergences.

Regulatory drivers
------------------
- CMMC Level 2/3: AC.2.006, CM.2.061 — software integrity verification.
- EU AI Act Art. 9: risk management; Art. 12: technical documentation.
- DoD IL4/IL5: model artefact integrity attestation at inference time.

Stdlib only — ``hashlib``, ``json``, ``pathlib``.  No optional extras.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# Property-name prefix written by sbom_builder.py for per-file weight hashes.
_WEIGHT_HASH_PREFIX = "squish:weight_hash:"

_CHUNK_SIZE = 1 << 20  # 1 MiB read buffer


# ──────────────────────────────────────────────────────────────────────────────
# Public data classes
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class DriftConfig:
    """Configuration for :func:`check_drift`.

    Parameters
    ----------
    bom_path:
        Path to the CycloneDX BOM JSON file (typically
        ``<model_dir>/cyclonedx-mlbom.json``).
    model_dir:
        Directory containing the compressed model files to verify.
    tolerance:
        Reserved for future use (e.g. fuzzy / approximate matching).
        Currently unused — any digest mismatch is treated as drift.
    """

    bom_path: Path
    model_dir: Path
    tolerance: float = 0.0


@dataclass
class DriftHit:
    """A single file that diverged from its BOM-attested digest.

    Parameters
    ----------
    path:
        Relative path of the file within ``model_dir`` (as recorded in
        the BOM property name).
    expected_digest:
        Lowercase hex SHA-256 string recorded in the BOM.
    actual_digest:
        Lowercase hex SHA-256 string computed from disk, or ``""`` when
        the file is missing entirely.
    """

    path: str
    expected_digest: str
    actual_digest: str

    @property
    def missing(self) -> bool:
        """True when the file was listed in the BOM but not found on disk."""
        return self.actual_digest == ""

    @property
    def tampered(self) -> bool:
        """True when the file exists but its digest differs from the BOM."""
        return self.actual_digest != "" and self.actual_digest != self.expected_digest


@dataclass
class DriftResult:
    """Result produced by :func:`check_drift`.

    Parameters
    ----------
    hits:
        Files whose digest diverged from the BOM (missing or tampered).
    files_checked:
        Number of BOM-attested files that were compared.
    ok:
        ``True`` when ``len(hits) == 0`` (no drift detected).
    summary:
        One-line human-readable summary.  Auto-built by ``__post_init__``
        when the caller passes the default empty string.
    """

    hits: list[DriftHit] = field(default_factory=list)
    files_checked: int = 0
    ok: bool = True
    summary: str = ""

    def __post_init__(self) -> None:
        if not self.summary:
            if self.ok:
                self.summary = f"✓ No drift — {self.files_checked} file(s) verified"
            else:
                missing = sum(1 for h in self.hits if h.missing)
                tampered = sum(1 for h in self.hits if h.tampered)
                parts: list[str] = []
                if tampered:
                    parts.append(f"{tampered} tampered")
                if missing:
                    parts.append(f"{missing} missing")
                detail = ", ".join(parts) if parts else f"{len(self.hits)} hit(s)"
                self.summary = (
                    f"✗ Drift detected — {detail} "
                    f"({self.files_checked} checked)"
                )


# ──────────────────────────────────────────────────────────────────────────────
# Private helpers
# ──────────────────────────────────────────────────────────────────────────────


def _sha256_file(path: Path) -> str:
    """Return the lowercase hex SHA-256 digest of *path*."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK_SIZE), b""):
            h.update(chunk)
    return h.hexdigest()


def _parse_bom_hashes(bom: dict[str, Any]) -> dict[str, str]:
    """Extract per-file SHA-256 hashes from a squish CycloneDX BOM dict.

    Reads ``squish:weight_hash:<rel_path>`` entries from the first
    component's ``properties[]`` array.

    Returns a mapping ``{rel_path: hex_digest}`` — empty when the BOM
    contains no recognisable per-file hash properties.
    """
    result: dict[str, str] = {}
    components = bom.get("components", [])
    if not components:
        return result
    for prop in components[0].get("properties", []):
        name: str = prop.get("name", "")
        if name.startswith(_WEIGHT_HASH_PREFIX):
            rel_path = name[len(_WEIGHT_HASH_PREFIX):]
            value: str = prop.get("value", "")
            if rel_path and value:
                result[rel_path] = value.lower()
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────


def check_drift(config: DriftConfig) -> DriftResult:
    """Compare BOM-attested file digests against the model directory on disk.

    Parameters
    ----------
    config:
        Drift configuration specifying BOM path, model directory, and
        options.

    Returns
    -------
    DriftResult
        Container with a list of :class:`DriftHit` objects, file count,
        ``ok`` flag, and a one-line summary string.

    Raises
    ------
    OSError
        If the BOM file cannot be read.
    json.JSONDecodeError
        If the BOM file is not valid JSON.
    ValueError
        If the BOM contains no recognisable per-file ``squish:weight_hash:``
        property entries (i.e. there is nothing to verify against).
    """
    bom_text = config.bom_path.read_text(encoding="utf-8")
    bom: dict[str, Any] = json.loads(bom_text)

    bom_hashes = _parse_bom_hashes(bom)
    if not bom_hashes:
        raise ValueError(
            f"BOM at {config.bom_path} contains no per-file hash entries "
            f"(expected 'squish:weight_hash:*' component properties)"
        )

    hits: list[DriftHit] = []
    for rel_path, expected in sorted(bom_hashes.items()):
        full_path = config.model_dir / rel_path
        if not full_path.exists():
            hits.append(
                DriftHit(path=rel_path, expected_digest=expected, actual_digest="")
            )
            log.warning("drift: missing file %s", rel_path)
        else:
            actual = _sha256_file(full_path)
            if actual != expected:
                hits.append(
                    DriftHit(
                        path=rel_path,
                        expected_digest=expected,
                        actual_digest=actual,
                    )
                )
                log.warning(
                    "drift: digest mismatch for %s (expected %.8s…, got %.8s…)",
                    rel_path,
                    expected,
                    actual,
                )

    ok = len(hits) == 0
    return DriftResult(hits=hits, files_checked=len(bom_hashes), ok=ok)
