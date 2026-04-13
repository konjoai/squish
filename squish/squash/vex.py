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

    @property
    def statements(self) -> list[VexStatement]:
        """Flat list of all :class:`VexStatement` entries across all documents."""
        result: list[VexStatement] = []
        for doc in self._documents:
            result.extend(doc.statements)
        return result

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
        api_key: str | None = None,
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
        api_key:
            Bearer token for authenticated VEX feed endpoints.  Sent as
            ``Authorization: Bearer <key>``.  Falls back to the
            ``SQUASH_VEX_API_KEY`` environment variable when *None*.
        """
        import ssl

        _key = api_key or os.environ.get("SQUASH_VEX_API_KEY") or None

        ssl_ctx = ssl.create_default_context()
        if ca_bundle:
            ssl_ctx.load_verify_locations(cafile=ca_bundle)

        headers: dict[str, str] = {"Accept": "application/json", "User-Agent": "squash/1.0"}
        if _key:
            headers["Authorization"] = f"Bearer {_key}"

        try:
            req = urllib.request.Request(
                url,
                headers=headers,
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

    DEFAULT_URL: str = "https://raw.githubusercontent.com/squishai/vex-feed/main/feed.openvex.json"
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
        api_key: str | None = None,
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
        api_key:
            Bearer token for authenticated feed endpoints.  Defaults to the
            ``SQUASH_VEX_API_KEY`` environment variable when *None*.
        """
        _key = api_key or os.environ.get("SQUASH_VEX_API_KEY") or None

        manifest = self._read_manifest()
        feed_file = self._cache_dir / "feed.json"

        last_modified: str | None = manifest.get("last_modified")
        if force or not feed_file.exists() or self.is_stale():
            self._fetch(url, feed_file, last_modified, timeout, ca_bundle, _key)

        if feed_file.exists():
            feed = VexFeed.from_url(url, timeout=timeout, ca_bundle=ca_bundle, api_key=_key)
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
            raw = manifest["last_fetched"]
            # Python 3.10 fromisoformat() doesn't accept the 'Z' suffix.
            # Normalise it to '+00:00' before parsing.
            if isinstance(raw, str) and raw.endswith("Z"):
                raw = raw[:-1] + "+00:00"
            fetched_at = datetime.fromisoformat(raw)
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
        api_key: str | None = None,
    ) -> None:
        """Fetch *url* to *dest*; respects ``If-Modified-Since`` (304 = no-op)."""
        try:
            req = urllib.request.Request(url)
            if last_modified:
                req.add_header("If-Modified-Since", last_modified)
            if api_key:
                req.add_header("Authorization", f"Bearer {api_key}")

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

    def fetch_squash_feed(self, *, force: bool = False) -> "VexFeed":
        """Convenience shortcut: fetch the canonical Squash community VEX feed.

        Parameters
        ----------
        force:
            If *True*, bypass the ``If-Modified-Since`` cache check and always
            download a fresh copy.

        Returns
        -------
        VexFeed
            Loaded feed from the Squash community endpoint.
        """
        return self.load_or_fetch(SQUASH_VEX_FEED_URL, force=force)

    @classmethod
    def load_bundled(cls) -> "VexFeed":
        """Return a :class:`VexFeed` loaded from the bundled community seed feed.

        No network I/O — reads the OpenVEX document embedded in the squish
        package at ``squish/squash/data/community_vex_feed.openvex.json``.
        Use this as a fallback when :attr:`DEFAULT_URL` is unreachable.

        Returns
        -------
        VexFeed
            Feed containing the bundled community VEX statements.
        """
        data_path = Path(__file__).parent / "data" / "community_vex_feed.openvex.json"
        try:
            raw = json.loads(data_path.read_text(encoding="utf-8"))
        except OSError:
            return VexFeed(documents=[])
        if isinstance(raw, dict) and "statements" in raw:
            return VexFeed(documents=[VexDocument.from_dict(raw)])
        return VexFeed(documents=[])


# ── VEX feed constants ─────────────────────────────────────────────────────────

#: Canonical URL for the Squash community ML-model VEX feed.
SQUASH_VEX_FEED_URL = "https://vex.squish.ai/ml-models/feed.openvex.json"

#: Fallback URL (GitHub raw) used when the primary endpoint is unreachable.
SQUASH_VEX_FEED_FALLBACK_URL = (
    "https://raw.githubusercontent.com/squishai/vex-feed/main/feed.openvex.json"
)


# ─────────────────────────────────────────────────────────────────────────────
# Wave 52 — VEX subscription management
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class VexSubscription:
    """Persisted VEX feed subscription entry.

    The API key is **never** stored on disk.  Only the name of the environment
    variable that holds it is persisted so that key rotation does not require
    re-subscribing.

    Parameters
    ----------
    url:
        Remote VEX feed endpoint to poll.
    alias:
        Short human-readable identifier, e.g. ``"nist-nvd"``.
    api_key_env_var:
        Name of the environment variable that supplies the API key.
    polling_hours:
        Refresh interval used by ``squash vex update --all``.
    last_polled:
        ISO-8601 UTC timestamp of last successful poll (empty = never).
    """

    url: str
    alias: str = ""
    api_key_env_var: str = "SQUASH_VEX_API_KEY"
    polling_hours: int = 24
    last_polled: str = ""


class VexSubscriptionStore:
    """Persistent registry of VEX feed subscriptions.

    Subscriptions are written to ``~/.squish/vex-subscriptions.json`` as a
    JSON array.  The API key itself is never written to disk — only the name
    of the environment variable that holds it.

    Usage::

        store = VexSubscriptionStore()
        store.add(VexSubscription(url="https://vex.example.com/feed.json", alias="example"))
        for sub in store.list():
            print(sub.url, sub.last_polled or "(never polled)")
    """

    _FILENAME: str = "vex-subscriptions.json"

    def __init__(self, store_dir: Path | None = None) -> None:
        self._store_dir = store_dir or (Path.home() / ".squish")
        self._store_dir.mkdir(parents=True, exist_ok=True)

    @property
    def _path(self) -> Path:
        return self._store_dir / self._FILENAME

    # ── public API ────────────────────────────────────────────────────────────

    def add(self, subscription: VexSubscription) -> None:
        """Add or update a subscription (keyed by URL)."""
        subs = self.list()
        subs = [s for s in subs if s.url != subscription.url]
        subs.append(subscription)
        self._write(subs)

    def remove(self, url_or_alias: str) -> bool:
        """Remove a subscription by URL or alias.

        Returns
        -------
        bool
            *True* if a subscription was removed, *False* if not found.
        """
        subs = self.list()
        before = len(subs)
        subs = [s for s in subs if s.url != url_or_alias and s.alias != url_or_alias]
        if len(subs) == before:
            return False
        self._write(subs)
        return True

    def list(self) -> list[VexSubscription]:
        """Return all registered subscriptions."""
        if not self._path.exists():
            return []
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            return [
                VexSubscription(
                    url=item["url"],
                    alias=item.get("alias", ""),
                    api_key_env_var=item.get("api_key_env_var", "SQUASH_VEX_API_KEY"),
                    polling_hours=item.get("polling_hours", 24),
                    last_polled=item.get("last_polled", ""),
                )
                for item in raw
                if "url" in item
            ]
        except (OSError, json.JSONDecodeError, KeyError):
            return []

    def get(self, url_or_alias: str) -> "VexSubscription | None":
        """Return a subscription by URL or alias, or *None* if not found."""
        for sub in self.list():
            if sub.url == url_or_alias or sub.alias == url_or_alias:
                return sub
        return None

    def mark_polled(self, url: str) -> None:
        """Update :attr:`~VexSubscription.last_polled` for *url* to now."""
        subs = self.list()
        now = datetime.now(timezone.utc).isoformat()
        for sub in subs:
            if sub.url == url:
                sub.last_polled = now
        self._write(subs)

    # ── internals ─────────────────────────────────────────────────────────────

    def _write(self, subscriptions: list[VexSubscription]) -> None:
        data = [
            {
                "url": s.url,
                "alias": s.alias,
                "api_key_env_var": s.api_key_env_var,
                "polling_hours": s.polling_hours,
                "last_polled": s.last_polled,
            }
            for s in subscriptions
        ]
        self._path.write_text(json.dumps(data, indent=2), encoding="utf-8")


class VexFeedManifest:
    """Generate and validate OpenVEX 0.2.0 feed manifests.

    This class provides two entry-points for working with the Squash community
    VEX feed format:

    * :meth:`generate` — build a valid OpenVEX document dict from a list of
      statement entries.
    * :meth:`validate` — check that a document dict conforms to the minimum
      required structure.

    Usage::

        doc = VexFeedManifest.generate([
            {
                "vulnerability": {"name": "CVE-2024-12345"},
                "products": [{"@id": "pkg:pypi/numpy@1.24.0"}],
                "status": "not_affected",
                "justification": "vulnerable_code_not_in_execute_path",
            }
        ])
        errors = VexFeedManifest.validate(doc)
        assert not errors
    """

    #: OpenVEX JSON-LD context URL
    OPENVEX_CONTEXT = "https://openvex.dev/ns/v0.2.0"
    #: Document type identifier
    OPENVEX_TYPE = "OpenVEXDocument"
    #: Canonical spec version tag
    SPEC_VERSION = "0.2.0"

    @staticmethod
    def generate(
        entries: list[dict],
        *,
        author: str = "squash",
        doc_id: str | None = None,
        timestamp: str | None = None,
    ) -> dict:
        """Generate a valid OpenVEX 0.2.0 document from a list of statement entries.

        Parameters
        ----------
        entries:
            List of statement dicts.  Each must contain at minimum
            ``"vulnerability"``, ``"products"``, and ``"status"`` keys.
        author:
            Author field in the document metadata; defaults to ``"squash"``.
        doc_id:
            Optional ``@id`` for the document.  A random UUID URN is generated
            if not provided.
        timestamp:
            ISO-8601 timestamp string; defaults to ``datetime.utcnow()``.

        Returns
        -------
        dict
            A fully-populated OpenVEX 0.2.0 document.
        """
        import uuid
        from datetime import datetime, timezone

        if doc_id is None:
            doc_id = f"https://squish.ai/vex/{uuid.uuid4()}"
        if timestamp is None:
            timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        statements: list[dict] = []
        for entry in entries:
            stmt: dict = {
                "vulnerability": entry.get("vulnerability", {}),
                "products": entry.get("products", []),
                "status": entry.get("status", "under_investigation"),
            }
            if "justification" in entry:
                stmt["justification"] = entry["justification"]
            if "impact_statement" in entry:
                stmt["impact_statement"] = entry["impact_statement"]
            if "action_statement" in entry:
                stmt["action_statement"] = entry["action_statement"]
            statements.append(stmt)

        return {
            "@context": VexFeedManifest.OPENVEX_CONTEXT,
            "@type": VexFeedManifest.OPENVEX_TYPE,
            "specVersion": VexFeedManifest.SPEC_VERSION,
            "@id": doc_id,
            "author": author,
            "timestamp": timestamp,
            "statements": statements,
        }

    @staticmethod
    def validate(doc: dict) -> list[str]:
        """Validate an OpenVEX document dict and return a list of error strings.

        An empty list means the document is structurally valid.

        Parameters
        ----------
        doc:
            The document dict to validate.

        Returns
        -------
        list[str]
            Validation errors.  Empty if document is valid.
        """
        errors: list[str] = []

        if doc.get("@context") != VexFeedManifest.OPENVEX_CONTEXT:
            errors.append(
                f"@context must be '{VexFeedManifest.OPENVEX_CONTEXT}', got: {doc.get('@context')!r}"
            )
        if doc.get("@type") != VexFeedManifest.OPENVEX_TYPE:
            errors.append(
                f"@type must be '{VexFeedManifest.OPENVEX_TYPE}', got: {doc.get('@type')!r}"
            )
        if "statements" not in doc:
            errors.append("'statements' key is required")
        elif not isinstance(doc["statements"], list):
            errors.append("'statements' must be a list")
        else:
            for i, stmt in enumerate(doc["statements"]):
                if "vulnerability" not in stmt:
                    errors.append(f"statements[{i}]: 'vulnerability' key is required")
                if "products" not in stmt:
                    errors.append(f"statements[{i}]: 'products' key is required")
                if "status" not in stmt:
                    errors.append(f"statements[{i}]: 'status' key is required")
                valid_statuses = {
                    "not_affected",
                    "affected",
                    "fixed",
                    "under_investigation",
                }
                status = stmt.get("status", "")
                if status and status not in valid_statuses:
                    errors.append(
                        f"statements[{i}]: status {status!r} is not a recognised OpenVEX status"
                    )
        if "@id" not in doc:
            errors.append("'@id' key is required")
        if "author" not in doc:
            errors.append("'author' key is required")
        return errors

