"""squish/squash/vex.py — Vulnerability Exploitability eXchange (VEX) engine.

VEX documents describe whether known CVEs actually affect specific model
artifacts.  A VEX feed turns Squash from a build-time tool into an ongoing
operational necessity — you subscribe once and the fleet stays current.

Architecture
-----------
- :class:`VexDocument` : a single VEX document (one CVE → N model statements)
- :class:`VexStatement` : CVE exploitability assessment for one model artifact
- :class:`VexEvaluator` : evaluates a VEX feed against a deployed model inventory
- :class:`VexFeed` : thin wrapper around a local directory of VEX docs or a
  remote HTTPS feed endpoint (fetched via standard library urllib — no requests dep)

VEX format follows CSAF 2.0 / OpenVEX conventions:

    https://www.cisa.gov/sites/default/files/2023-04/minimum-requirements-for-vex-508c.pdf
    https://github.com/openvex/spec

Usage::

    feed = VexFeed.from_directory(Path("./vex-feed"))
    inv = ModelInventory.from_sbom_dir(Path("~/.squish/models"))
    report = VexEvaluator.evaluate(feed, inv)
    for affected in report.affected_models:
        print(affected)
"""

from __future__ import annotations

import json
import logging
import os
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# VEX status values (OpenVEX / CSAF 2.0)
VEX_STATUS_NOT_AFFECTED = "not_affected"
VEX_STATUS_AFFECTED = "affected"
VEX_STATUS_FIXED = "fixed"
VEX_STATUS_UNDER_INVESTIGATION = "under_investigation"


@dataclass
class VexStatement:
    """Exploitability statement for a single CVE × model artifact pair.

    Parameters
    ----------
    vulnerability_id:
        CVE or advisory ID, e.g. ``"CVE-2025-12345"``.
    status:
        One of the :data:`VEX_STATUS_*` constants.
    justification:
        Human-readable explanation.  Required when status is ``not_affected``.
    affected_model_purl:
        Package URL matching the model artifact.  ``None`` = applies to all
        models in the SBOM (broad advisory).
    action_statement:
        Remediation action when status is ``affected``.
    timestamp:
        ISO-8601 UTC string when this statement was published.
    """

    vulnerability_id: str
    status: str
    justification: str = ""
    affected_model_purl: str | None = None
    action_statement: str = ""
    timestamp: str = ""

    @property
    def is_affected(self) -> bool:
        return self.status == VEX_STATUS_AFFECTED

    @property
    def is_under_investigation(self) -> bool:
        return self.status == VEX_STATUS_UNDER_INVESTIGATION


@dataclass
class VexDocument:
    """A VEX document: one or more exploitability statements.

    Parameters
    ----------
    document_id:
        Unique document identifier (URI or UUID).
    issuer:
        Organization or tool that published this VEX document.
    statements:
        Ordered list of :class:`VexStatement` entries.
    """

    document_id: str
    issuer: str = "squash-vex-feed"
    statements: list[VexStatement] = field(default_factory=list)
    last_updated: str = ""

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "VexDocument":
        """Parse a VEX document from a dict (OpenVEX JSON format)."""
        stmts = []
        for s in d.get("statements", []):
            stmts.append(
                VexStatement(
                    vulnerability_id=s.get("vulnerability", {}).get(
                        "name", s.get("vulnerability_id", "UNKNOWN")
                    ),
                    status=s.get("status", VEX_STATUS_UNDER_INVESTIGATION),
                    justification=s.get("justification", ""),
                    affected_model_purl=s.get("products", [None])[0]
                    if s.get("products")
                    else None,
                    action_statement=s.get("action_statement", ""),
                    timestamp=s.get("timestamp", ""),
                )
            )
        return cls(
            document_id=d.get("id", ""),
            issuer=d.get("author", "squash-vex-feed"),
            statements=stmts,
            last_updated=d.get("timestamp", ""),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to OpenVEX-compatible JSON dict."""
        return {
            "@context": "https://openvex.dev/ns/v0.2.0",
            "@type": "OpenVEX",
            "id": self.document_id,
            "author": self.issuer,
            "timestamp": self.last_updated
            or datetime.now(timezone.utc).isoformat(),
            "version": 1,
            "statements": [
                {
                    "vulnerability": {"name": s.vulnerability_id},
                    "status": s.status,
                    "justification": s.justification,
                    "products": (
                        [s.affected_model_purl] if s.affected_model_purl else []
                    ),
                    "action_statement": s.action_statement,
                    "timestamp": s.timestamp,
                }
                for s in self.statements
            ],
        }


@dataclass
class ModelInventoryEntry:
    """One deployed model known to Squash."""

    model_id: str
    purl: str
    sbom_path: Path
    composite_sha256: str = ""


@dataclass
class ModelInventory:
    """Collection of deployed models for fleet-level VEX evaluation."""

    entries: list[ModelInventoryEntry] = field(default_factory=list)

    @classmethod
    def from_sbom_dir(cls, models_root: Path) -> "ModelInventory":
        """Build an inventory by scanning a directory tree for CycloneDX sidecars.

        Parameters
        ----------
        models_root:
            Directory to recursively search for ``cyclonedx-mlbom.json`` files.
        """
        inv = cls()
        for sidecar in models_root.rglob("cyclonedx-mlbom.json"):
            try:
                bom: dict = json.loads(sidecar.read_text())
                components = bom.get("components", [])
                if not components:
                    continue
                comp = components[0]
                purl = comp.get("purl", "")
                hashes = comp.get("hashes", [])
                sha256 = next(
                    (h["content"] for h in hashes if h.get("alg") == "SHA-256"),
                    "",
                )
                inv.entries.append(
                    ModelInventoryEntry(
                        model_id=comp.get("name", str(sidecar.parent.name)),
                        purl=purl,
                        sbom_path=sidecar,
                        composite_sha256=sha256,
                    )
                )
            except (json.JSONDecodeError, OSError, KeyError) as e:
                log.debug("Skipping malformed sidecar %s: %s", sidecar, e)
        return inv

    @classmethod
    def from_list(cls, entries: list[ModelInventoryEntry]) -> "ModelInventory":
        return cls(entries=entries)


@dataclass 
class VexAffectedModel:
    """One affected (model, CVE) pair from a fleet evaluation."""

    model: ModelInventoryEntry
    cve: str
    status: str
    action_statement: str
    justification: str


@dataclass
class VexReport:
    """Result of evaluating a VEX feed against a model inventory."""

    evaluated_at: str
    total_models: int
    total_cves_evaluated: int
    affected_models: list[VexAffectedModel] = field(default_factory=list)
    under_investigation: list[VexAffectedModel] = field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        return len(self.affected_models) == 0

    def summary(self) -> str:
        n_affected = len(self.affected_models)
        n_invest = len(self.under_investigation)
        status = "CLEAN" if self.is_clean else "AFFECTED"
        return (
            f"[{status}] {self.total_models} models × "
            f"{self.total_cves_evaluated} CVEs: "
            f"{n_affected} affected, {n_invest} under investigation"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "evaluated_at": self.evaluated_at,
            "total_models": self.total_models,
            "total_cves_evaluated": self.total_cves_evaluated,
            "is_clean": self.is_clean,
            "affected": [
                {
                    "model_id": a.model.model_id,
                    "purl": a.model.purl,
                    "cve": a.cve,
                    "status": a.status,
                    "action": a.action_statement,
                }
                for a in self.affected_models
            ],
            "under_investigation": [
                {
                    "model_id": u.model.model_id,
                    "cve": u.cve,
                }
                for u in self.under_investigation
            ],
        }


class VexFeed:
    """Collection of VEX documents loaded from a directory or remote URL."""

    def __init__(self, documents: list[VexDocument]) -> None:
        self._documents = documents

    @property
    def documents(self) -> list[VexDocument]:
        return list(self._documents)

    @classmethod
    def from_directory(cls, vex_dir: Path) -> "VexFeed":
        """Load all ``*.vex.json`` / ``*.json`` files from *vex_dir*."""
        docs: list[VexDocument] = []
        for fp in sorted(vex_dir.glob("*.json")):
            try:
                raw = json.loads(fp.read_text())
                docs.append(VexDocument.from_dict(raw))
            except (json.JSONDecodeError, OSError) as e:
                log.warning("Skipping unparseable VEX file %s: %s", fp, e)
        return cls(docs)

    @classmethod
    def from_url(
        cls,
        url: str,
        timeout: int = 30,
        ca_bundle: str | None = None,
    ) -> "VexFeed":
        """Fetch a VEX feed from an HTTPS URL.

        Parameters
        ----------
        url:
            HTTPS endpoint returning a JSON array of VEX documents or a single
            newline-delimited JSON stream.
        timeout:
            Socket timeout in seconds.
        ca_bundle:
            Optional path to a CA certificate bundle for enterprise environments
            with custom PKI.  Passed to a custom SSL context.
        """
        import ssl

        ssl_ctx = ssl.create_default_context()
        if ca_bundle:
            ssl_ctx.load_verify_locations(cafile=ca_bundle)

        try:
            req = urllib.request.Request(
                url,
                headers={"Accept": "application/json", "User-Agent": "squash/1.0"},
            )
            handler = urllib.request.HTTPSHandler(context=ssl_ctx)
            opener = urllib.request.build_opener(handler)
            with opener.open(req, timeout=timeout) as resp:
                raw_text = resp.read().decode("utf-8")
        except OSError as e:
            raise OSError(f"Failed to fetch VEX feed from {url}: {e}") from e

        docs: list[VexDocument] = []
        raw_text = raw_text.strip()

        # Support both a JSON array and NDJSON (one document per line)
        if raw_text.startswith("["):
            items = json.loads(raw_text)
            for item in items:
                docs.append(VexDocument.from_dict(item))
        else:
            for line in raw_text.splitlines():
                line = line.strip()
                if line:
                    try:
                        docs.append(VexDocument.from_dict(json.loads(line)))
                    except json.JSONDecodeError:
                        log.debug("Skipping non-JSON VEX line: %.80s", line)

        return cls(docs)

    @classmethod
    def empty(cls) -> "VexFeed":
        return cls([])


class VexEvaluator:
    """Evaluate a :class:`VexFeed` against a :class:`ModelInventory`.

    The evaluation logic:
    1. For each :class:`VexStatement` in the feed, determine which models in
       the inventory match (by PURL prefix or broadcast).
    2. Emit :class:`VexAffectedModel` entries for ``affected`` and
       ``under_investigation`` statuses.
    3. ``not_affected`` and ``fixed`` statements are recorded as clean.
    """

    @staticmethod
    def evaluate(feed: VexFeed, inventory: ModelInventory) -> VexReport:
        """Evaluate *feed* against *inventory* and return a :class:`VexReport`."""
        now = datetime.now(timezone.utc).isoformat()

        all_statements: list[VexStatement] = []
        for doc in feed.documents:
            all_statements.extend(doc.statements)

        cve_ids = {s.vulnerability_id for s in all_statements}

        affected: list[VexAffectedModel] = []
        under_inv: list[VexAffectedModel] = []

        for model in inventory.entries:
            for stmt in all_statements:
                if not _purl_matches(model.purl, stmt.affected_model_purl):
                    continue
                entry = VexAffectedModel(
                    model=model,
                    cve=stmt.vulnerability_id,
                    status=stmt.status,
                    action_statement=stmt.action_statement,
                    justification=stmt.justification,
                )
                if stmt.is_affected:
                    affected.append(entry)
                elif stmt.is_under_investigation:
                    under_inv.append(entry)

        return VexReport(
            evaluated_at=now,
            total_models=len(inventory.entries),
            total_cves_evaluated=len(cve_ids),
            affected_models=affected,
            under_investigation=under_inv,
        )


def _purl_matches(model_purl: str, stmt_purl: str | None) -> bool:
    """Return True if statement applies to this model.

    - ``None`` / empty → broadcast (applies to all models)
    - Exact match → direct hit
    - Model PURL starts with statement PURL prefix → family match
      (e.g. ``pkg:huggingface/meta-llama`` matches any Llama model)
    """
    if not stmt_purl:
        return True  # broadcast advisory
    if not model_purl:
        return False
    return model_purl == stmt_purl or model_purl.startswith(stmt_purl)


# ─────────────────────────────────────────────────────────────────────────────
# Wave 16 — VexCache: persistent local cache with If-Modified-Since
# ─────────────────────────────────────────────────────────────────────────────

class VexCache:
    """Local disk cache for a remote VEX feed.

    Documents are stored under *cache_dir* (default ``~/.squish/vex-cache/``).
    A ``cache-manifest.json`` records the last fetch URL, timestamp, and
    statement count so :meth:`is_stale` can make fast freshness decisions
    without re-parsing every document.

    Example::

        cache = VexCache()
        feed = cache.load_or_fetch("https://vex.example.com/feed.json")
        if cache.is_stale():
            feed = cache.load_or_fetch(..., force=True)
    """

    DEFAULT_URL: str = "https://raw.githubusercontent.com/squishai/vex-feed/main/feed.json"
    DEFAULT_MAX_AGE_HOURS: int = 24
    _MANIFEST_FILE: str = "cache-manifest.json"

    def __init__(self, cache_dir: Path | None = None) -> None:
        self._cache_dir = cache_dir or (Path.home() / ".squish" / "vex-cache")
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    # ── public API ────────────────────────────────────────────────────────────

    def load_or_fetch(
        self,
        url: str,
        *,
        timeout: float = 10.0,
        ca_bundle: str | None = None,
        force: bool = False,
    ) -> "VexFeed":
        """Return a :class:`VexFeed` — from cache if fresh, else re-fetched.

        Parameters
        ----------
        url:
            Remote VEX feed endpoint (JSON or a ZIP of JSON files).
        timeout:
            HTTP connect+read timeout in seconds.
        ca_bundle:
            Optional path to a PEM CA bundle for TLS verification.
        force:
            If *True*, always re-fetch even if cache is fresh.
        """
        manifest = self._read_manifest()
        feed_file = self._cache_dir / "feed.json"

        last_modified: str | None = manifest.get("last_modified")
        if force or not feed_file.exists() or self.is_stale():
            self._fetch(url, feed_file, last_modified, timeout, ca_bundle)

        if feed_file.exists():
            feed = VexFeed.from_url(url, timeout=timeout, ca_bundle=ca_bundle)
            # Update manifest even if 304 (server reported not modified)
            self._write_manifest(url, feed)
            return feed

        return VexFeed(documents=[])  # cache miss + fetch failed gracefully

    def is_stale(self, max_age_hours: int | None = None) -> bool:
        """Return True if the cache is older than *max_age_hours* or empty."""
        manifest = self._read_manifest()
        if not manifest or "last_fetched" not in manifest:
            return True
        max_age = max_age_hours or self.DEFAULT_MAX_AGE_HOURS
        try:
            fetched_at = datetime.fromisoformat(manifest["last_fetched"])
            age = datetime.now(timezone.utc) - fetched_at
            return age.total_seconds() > max_age * 3600
        except (ValueError, TypeError):
            return True

    def manifest(self) -> dict[str, Any]:
        """Return the cached manifest dict (empty dict if cache is empty)."""
        return self._read_manifest()

    def clear(self) -> None:
        """Delete all cached documents and the manifest."""
        import shutil
        if self._cache_dir.exists():
            shutil.rmtree(self._cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    # ── internals ─────────────────────────────────────────────────────────────

    def _manifest_path(self) -> Path:
        return self._cache_dir / self._MANIFEST_FILE

    def _read_manifest(self) -> dict[str, Any]:
        mp = self._manifest_path()
        if not mp.exists():
            return {}
        try:
            return json.loads(mp.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    def _write_manifest(self, url: str, feed: "VexFeed") -> None:
        manifest = self._read_manifest()
        manifest["url"] = url
        manifest["last_fetched"] = datetime.now(timezone.utc).isoformat()
        manifest["statement_count"] = sum(len(d.statements) for d in feed.documents)
        self._manifest_path().write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )

    def _fetch(
        self,
        url: str,
        dest: Path,
        last_modified: str | None,
        timeout: float,
        ca_bundle: str | None,
    ) -> None:
        """Fetch *url* to *dest*; respects ``If-Modified-Since`` (304 = no-op)."""
        try:
            req = urllib.request.Request(url)
            if last_modified:
                req.add_header("If-Modified-Since", last_modified)

            kwargs: dict[str, Any] = {"timeout": timeout}
            if ca_bundle:
                import ssl
                ctx = ssl.create_default_context(cafile=ca_bundle)
                kwargs["context"] = ctx

            with urllib.request.urlopen(req, **kwargs) as resp:  # noqa: S310
                if resp.status == 304:
                    log.debug("VexCache: 304 Not Modified — reusing cached %s", dest)
                    return
                data = resp.read()
                # Atomic write via temp file
                tmp = dest.with_suffix(".tmp")
                tmp.write_bytes(data)
                tmp.replace(dest)
                # Record Last-Modified header for next conditional GET
                last_mod = resp.headers.get("Last-Modified")
                if last_mod:
                    manifest = self._read_manifest()
                    manifest["last_modified"] = last_mod
                    self._manifest_path().write_text(
                        json.dumps(manifest, indent=2), encoding="utf-8"
                    )
        except Exception as e:
            log.warning("VexCache: fetch failed for %s — %s", url, e)

