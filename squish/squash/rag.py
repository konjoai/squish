"""squish/squash/rag.py — RAG knowledge base integrity scanner.

RagScanner.index() walks a corpus directory, SHA-256 hashes every document,
and writes a tamper-evident .rag_manifest.json.  RagScanner.verify()
re-hashes the live corpus and reports drift (added / removed / modified
documents).

Addresses the #1 enterprise RAG failure: silently poisoned or drifted
knowledge bases.  The manifest_hash provides a deterministic, reproducible
fingerprint of the entire corpus — suitable for CI/CD gating.

Wave 47.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# ── Data models ──────────────────────────────────────────────────────────────


@dataclass
class RagFileEntry:
    """Hash record for a single document in the corpus."""

    path: str        # relative to corpus_dir, POSIX separators
    sha256: str      # hex digest
    size_bytes: int


@dataclass
class RagManifest:
    """Tamper-evident manifest for a RAG corpus directory.

    ``manifest_hash`` covers only the ``files`` list (sorted by path, rendered
    as canonical JSON).  Metadata fields (corpus_dir, indexed_at, …) are
    intentionally excluded so the hash is a pure content fingerprint.
    """

    version: int                 # always 1
    corpus_dir: str              # absolute path at index time
    indexed_at: str              # ISO-8601 UTC
    file_count: int
    files: list[dict]            # RagFileEntry rendered as dicts, sorted by path
    manifest_hash: str           # sha256(json.dumps(files, sort_keys=True))

    # ── serialisation ────────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def write(self, path: Path) -> None:
        """Write manifest to *path* as pretty-printed JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=False), encoding="utf-8")
        log.debug("Manifest written → %s", path)

    @staticmethod
    def load(path: Path) -> "RagManifest":
        """Load a manifest from *path*.

        Raises
        ------
        FileNotFoundError
            If *path* does not exist.
        ValueError
            If the JSON is missing required fields or the version is unsupported.
        """
        raw = json.loads(path.read_text(encoding="utf-8"))
        for key in ("version", "corpus_dir", "indexed_at", "file_count", "files", "manifest_hash"):
            if key not in raw:
                raise ValueError(f"Manifest at {path!r} is missing required key: {key!r}")
        if raw["version"] != 1:
            raise ValueError(f"Unsupported manifest version: {raw['version']}")
        return RagManifest(**raw)


@dataclass
class RagDriftItem:
    """A single detected divergence between manifest and live corpus."""

    path: str         # corpus-relative POSIX path
    status: str       # "added" | "removed" | "modified"
    old_hash: str     # "" for added files (no prior entry)
    new_hash: str     # "" for removed files (no longer on disk)


@dataclass
class RagVerifyResult:
    """Full verification result for a corpus directory."""

    ok: bool
    corpus_dir: str
    verified_at: str   # ISO-8601 UTC
    total_files: int   # total files found on disk
    drift_count: int
    drift: list[RagDriftItem]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "corpus_dir": self.corpus_dir,
            "verified_at": self.verified_at,
            "total_files": self.total_files,
            "drift_count": self.drift_count,
            "drift": [asdict(d) for d in self.drift],
        }


# ── Scanner ───────────────────────────────────────────────────────────────────


class RagScanner:
    """Corpus integrity scanner.

    All methods are pure static — no instance state required.
    """

    MANIFEST_FILENAME: str = ".rag_manifest.json"
    _CHUNK: int = 65_536  # 64 KB read chunks for _hash_file

    # ── Public API ───────────────────────────────────────────────────────────

    @staticmethod
    def index(
        corpus_dir: str | Path,
        *,
        glob: str = "**/*",
        exclude: list[str] | None = None,
    ) -> RagManifest:
        """Hash every file in *corpus_dir* and write ``.rag_manifest.json``.

        Parameters
        ----------
        corpus_dir:
            Root directory of the RAG corpus.
        glob:
            Glob pattern relative to *corpus_dir* (default: ``"**/*"``).
        exclude:
            Optional list of filename patterns to exclude (e.g. ``["*.pyc"]``).
            The manifest file itself is always excluded.

        Returns
        -------
        RagManifest
            The manifest that was written to disk.
        """
        corpus = Path(corpus_dir).resolve()
        if not corpus.is_dir():
            raise NotADirectoryError(f"corpus_dir is not a directory: {corpus!r}")

        exclude_set: set[str] = set(exclude or [])
        exclude_set.add(RagScanner.MANIFEST_FILENAME)

        log.info("Indexing corpus %s (glob=%r)", corpus, glob)

        entries: list[RagFileEntry] = []
        for p in sorted(corpus.glob(glob)):
            if not p.is_file():
                continue
            rel = p.relative_to(corpus).as_posix()
            if p.name in exclude_set:
                continue
            sha = RagScanner._hash_file(p)
            size = p.stat().st_size
            entries.append(RagFileEntry(path=rel, sha256=sha, size_bytes=size))

        files_as_dicts = [asdict(e) for e in entries]
        mhash = RagScanner._manifest_hash(files_as_dicts)

        manifest = RagManifest(
            version=1,
            corpus_dir=str(corpus),
            indexed_at=_utc_now(),
            file_count=len(entries),
            files=files_as_dicts,
            manifest_hash=mhash,
        )
        manifest.write(corpus / RagScanner.MANIFEST_FILENAME)
        log.info(
            "Indexed %d files → manifest_hash=%s…",
            len(entries),
            mhash[:12],
        )
        return manifest

    @staticmethod
    def verify(corpus_dir: str | Path) -> RagVerifyResult:
        """Compare the live corpus against the stored manifest.

        Parameters
        ----------
        corpus_dir:
            Root directory of the RAG corpus.  Must contain a valid
            ``.rag_manifest.json`` written by :meth:`index`.

        Returns
        -------
        RagVerifyResult
            ``ok=True`` iff no drift was detected.  ``ok=False`` when the
            manifest is missing — :attr:`RagVerifyResult.drift` will contain a
            single entry with ``status="missing_manifest"``.
        """
        corpus = Path(corpus_dir).resolve()
        manifest_path = corpus / RagScanner.MANIFEST_FILENAME
        verified_at = _utc_now()

        if not manifest_path.exists():
            log.warning("No manifest found at %s", manifest_path)
            return RagVerifyResult(
                ok=False,
                corpus_dir=str(corpus),
                verified_at=verified_at,
                total_files=0,
                drift_count=1,
                drift=[
                    RagDriftItem(
                        path=RagScanner.MANIFEST_FILENAME,
                        status="missing_manifest",
                        old_hash="",
                        new_hash="",
                    )
                ],
            )

        manifest = RagManifest.load(manifest_path)
        log.info("Verifying corpus %s against manifest (hash=%s…)", corpus, manifest.manifest_hash[:12])

        # Build {rel_path → sha256} from the stored manifest
        stored: dict[str, str] = {e["path"]: e["sha256"] for e in manifest.files}

        # Scan live corpus (same exclusion logic as index)
        live: dict[str, str] = {}
        for p in sorted(corpus.glob("**/*")):
            if not p.is_file():
                continue
            if p.name == RagScanner.MANIFEST_FILENAME:
                continue
            rel = p.relative_to(corpus).as_posix()
            live[rel] = RagScanner._hash_file(p)

        drift: list[RagDriftItem] = []

        # Modified or removed
        for rel, old_hash in stored.items():
            new_hash = live.get(rel)
            if new_hash is None:
                drift.append(RagDriftItem(path=rel, status="removed", old_hash=old_hash, new_hash=""))
            elif new_hash != old_hash:
                drift.append(RagDriftItem(path=rel, status="modified", old_hash=old_hash, new_hash=new_hash))

        # Added
        for rel, new_hash in live.items():
            if rel not in stored:
                drift.append(RagDriftItem(path=rel, status="added", old_hash="", new_hash=new_hash))

        drift.sort(key=lambda d: d.path)

        result = RagVerifyResult(
            ok=len(drift) == 0,
            corpus_dir=str(corpus),
            verified_at=verified_at,
            total_files=len(live),
            drift_count=len(drift),
            drift=drift,
        )
        log.info(
            "Verify complete — ok=%s, drift_count=%d",
            result.ok,
            result.drift_count,
        )
        return result

    # ── Internals ─────────────────────────────────────────────────────────────

    @staticmethod
    def _hash_file(path: Path) -> str:
        """Return the SHA-256 hex digest of a file, read in 64 KB chunks."""
        h = hashlib.sha256()
        with path.open("rb") as fh:
            while chunk := fh.read(RagScanner._CHUNK):
                h.update(chunk)
        return h.hexdigest()

    @staticmethod
    def _manifest_hash(files: list[dict]) -> str:
        """Return SHA-256 of the canonical JSON representation of *files*.

        Uses ``sort_keys=True`` and no whitespace so the hash is stable
        across Python versions and platforms.
        """
        payload = json.dumps(files, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ── Helpers ───────────────────────────────────────────────────────────────────


def _utc_now() -> str:
    """Return the current UTC time as an ISO-8601 string (seconds precision)."""
    return datetime.datetime.now(tz=datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
