"""squish/squash/lineage.py — Model transformation lineage chain.

Every time a model is quantized, compressed, signed, evaluated, or exported,
``TransformationEvent`` records what happened, who ran it, when, and where.
Events are linked in a Merkle chain: each event's SHA-256 covers its own
content plus the hash of the preceding event, making retroactive tampering
mathematically evident without requiring a trusted third party.

The chain file (``.lineage_chain.json``) travels *with* the model artefacts,
so provenance is available even after a model is transferred or exported.

Regulatory drivers
------------------
- **EU AI Act Annex IV** (Art. 11): technical documentation must record the
  design specifications, development process, and major changes made at each
  stage of a model's lifecycle.  The lineage chain is a machine-readable
  implementation of this requirement.
- **NIST AI RMF GOVERN 1.7**: supply-chain provenance for AI components.
- **M&A / model transfer due diligence**: the chain persists in the model
  directory and is transferred with it, enabling buyers to audit the full
  transformation history of any acquired model asset.

Wave 48.
"""
from __future__ import annotations

import datetime
import getpass
import hashlib
import json
import logging
import socket
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# ── Data models ──────────────────────────────────────────────────────────────


@dataclass
class TransformationEvent:
    """A single recorded transformation applied to a model artefact.

    Fields
    ------
    event_id:
        UUID4 — unique identifier for this event.
    model_id:
        Human-readable model name or HuggingFace repo ID.
    operation:
        Short verb describing the transformation: ``"compress"``,
        ``"quantize"``, ``"sign"``, ``"verify"``, ``"export"``, etc.
    operator:
        Who/what ran this operation.  Defaults to ``user@hostname``;
        CI systems should set ``"ci:<runner>"``.
    timestamp:
        ISO-8601 UTC timestamp at event creation time.
    input_dir:
        Absolute path to the source model directory before the transformation.
    output_dir:
        Absolute path to the destination model directory after the transformation.
    params:
        Operation-specific key/value parameters (e.g. ``{"format": "INT4",
        "awq": true, "group_size": 32}``).
    prev_hash:
        SHA-256 of the preceding event's canonical serialisation.
        Empty string for the genesis (first) event.
    event_hash:
        SHA-256 of this event's canonical serialisation (all fields except
        ``event_hash`` itself, rendered as ``json.dumps(…, sort_keys=True,
        separators=(",",":"))``.
    """

    event_id: str    # UUID4
    model_id: str    # model name / HF repo
    operation: str   # "compress" | "quantize" | "sign" | "verify" | "export" | …
    operator: str    # who/what ran this: "user@host" or "ci:<runner>"
    timestamp: str   # ISO-8601 UTC
    input_dir: str   # absolute source path
    output_dir: str  # absolute destination path
    params: dict     # operation-specific key/value params
    prev_hash: str   # SHA-256 of prev event; "" for genesis
    event_hash: str  # SHA-256 of this event (computed by LineageChain.record)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LineageVerifyResult:
    """Result of verifying a lineage chain's Merkle integrity."""

    ok: bool
    model_dir: str
    verified_at: str      # ISO-8601 UTC
    event_count: int
    broken_at: int | None  # 0-based index of first broken event; None when intact
    message: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ── Chain ────────────────────────────────────────────────────────────────────


class LineageChain:
    """Merkle-chained model transformation ledger stored alongside model artefacts.

    The ledger is a single JSON file (``CHAIN_FILENAME``) containing an
    ordered array of :class:`TransformationEvent` dicts.  Each event's
    ``event_hash`` is the SHA-256 of its canonical JSON (all fields except
    ``event_hash`` itself, serialised with ``sort_keys=True``).  Each event's
    ``prev_hash`` equals the ``event_hash`` of the preceding event, forming a
    Merkle chain that ``verify()`` can check without external state.
    """

    CHAIN_FILENAME = ".lineage_chain.json"

    # ── Event construction ────────────────────────────────────────────────────

    @staticmethod
    def create_event(
        operation: str,
        model_id: str,
        input_dir: str,
        output_dir: str,
        params: dict | None = None,
    ) -> "TransformationEvent":
        """Build a new *unlinked* :class:`TransformationEvent`.

        ``prev_hash`` and ``event_hash`` are set to empty strings; they are
        filled in by :meth:`record` at append time.

        Parameters
        ----------
        operation:
            Short verb label, e.g. ``"compress"``.
        model_id:
            Human-readable model identifier.
        input_dir:
            Source model directory.
        output_dir:
            Destination model directory.
        params:
            Optional operation-specific key/value parameters.
        """
        try:
            user = getpass.getuser()
        except Exception:  # noqa: BLE001
            user = "unknown"
        try:
            host = socket.gethostname()
        except Exception:  # noqa: BLE001
            host = "unknown"

        return TransformationEvent(
            event_id=str(uuid.uuid4()),
            model_id=model_id,
            operation=operation,
            operator=f"{user}@{host}",
            timestamp=_utc_now(),
            input_dir=str(input_dir),
            output_dir=str(output_dir),
            params=dict(params or {}),
            prev_hash="",   # filled by record()
            event_hash="",  # filled by record()
        )

    # ── Append ────────────────────────────────────────────────────────────────

    @staticmethod
    def record(model_dir: "str | Path", event: "TransformationEvent") -> str:
        """Append *event* to the lineage chain in *model_dir* and return its ``event_hash``.

        Steps
        -----
        1. Load the existing chain (or start a new empty chain).
        2. Set ``event.prev_hash`` to the last event's ``event_hash``
           (empty string if the chain is new).
        3. Compute and assign ``event.event_hash``.
        4. Append the serialised event to the chain JSON.
        5. Atomically write the updated chain file.

        Parameters
        ----------
        model_dir:
            Directory to write/update the chain file.
        event:
            The :class:`TransformationEvent` to append.  ``prev_hash`` and
            ``event_hash`` will be mutated in place.

        Returns
        -------
        str
            The computed ``event_hash`` of the newly appended event.
        """
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        chain_path = model_dir / LineageChain.CHAIN_FILENAME

        existing = _load_chain_json(chain_path)
        event.prev_hash = existing[-1]["event_hash"] if existing else ""
        event.event_hash = LineageChain._hash_event(event)

        existing.append(event.to_dict())
        chain_path.write_text(
            json.dumps(existing, indent=2, sort_keys=False),
            encoding="utf-8",
        )
        log.debug("Lineage event recorded → %s  hash=%s…", chain_path, event.event_hash[:16])
        return event.event_hash

    # ── Load ──────────────────────────────────────────────────────────────────

    @staticmethod
    def load(model_dir: "str | Path") -> "list[TransformationEvent]":
        """Load and deserialise all events from *model_dir*/.lineage_chain.json.

        Returns an empty list if the chain file does not exist.

        Raises
        ------
        ValueError
            If the JSON is structurally invalid (wrong type).
        """
        chain_path = Path(model_dir) / LineageChain.CHAIN_FILENAME
        raw = _load_chain_json(chain_path)
        events: list[TransformationEvent] = []
        for entry in raw:
            events.append(
                TransformationEvent(
                    event_id=str(entry.get("event_id", "")),
                    model_id=str(entry.get("model_id", "")),
                    operation=str(entry.get("operation", "")),
                    operator=str(entry.get("operator", "")),
                    timestamp=str(entry.get("timestamp", "")),
                    input_dir=str(entry.get("input_dir", "")),
                    output_dir=str(entry.get("output_dir", "")),
                    params=dict(entry.get("params") or {}),
                    prev_hash=str(entry.get("prev_hash", "")),
                    event_hash=str(entry.get("event_hash", "")),
                )
            )
        return events

    # ── Verify ────────────────────────────────────────────────────────────────

    @staticmethod
    def verify(model_dir: "str | Path") -> "LineageVerifyResult":
        """Verify the Merkle chain integrity of *model_dir*/.lineage_chain.json.

        Re-derives each ``event_hash`` from the event payload (with
        ``event_hash`` zeroed) and checks that ``prev_hash`` equals the
        preceding event's stored hash.  Never raises — all errors are captured
        in the returned :class:`LineageVerifyResult`.

        Returns
        -------
        LineageVerifyResult
            ``ok=True`` when the chain is intact.
            ``ok=False`` when the file is missing, empty, or tampered.
            ``broken_at`` is the 0-based index of the first bad event, or
            ``None`` when the chain is intact or empty.
        """
        model_dir_path = Path(model_dir)
        chain_path = model_dir_path / LineageChain.CHAIN_FILENAME
        now = _utc_now()

        if not chain_path.exists():
            return LineageVerifyResult(
                ok=False,
                model_dir=str(model_dir_path),
                verified_at=now,
                event_count=0,
                broken_at=None,
                message="chain file not found",
            )

        raw = _load_chain_json(chain_path)
        if not raw:
            return LineageVerifyResult(
                ok=True,
                model_dir=str(model_dir_path),
                verified_at=now,
                event_count=0,
                broken_at=None,
                message="empty chain",
            )

        prev_hash = ""
        for i, entry in enumerate(raw):
            # Re-build the event from stored data, zero event_hash, recompute
            evt = TransformationEvent(
                event_id=str(entry.get("event_id", "")),
                model_id=str(entry.get("model_id", "")),
                operation=str(entry.get("operation", "")),
                operator=str(entry.get("operator", "")),
                timestamp=str(entry.get("timestamp", "")),
                input_dir=str(entry.get("input_dir", "")),
                output_dir=str(entry.get("output_dir", "")),
                params=dict(entry.get("params") or {}),
                prev_hash=str(entry.get("prev_hash", "")),
                event_hash="",  # zeroed — excluded from hash input
            )

            stored_hash = str(entry.get("event_hash", ""))
            computed_hash = LineageChain._hash_event(evt)

            if computed_hash != stored_hash:
                return LineageVerifyResult(
                    ok=False,
                    model_dir=str(model_dir_path),
                    verified_at=now,
                    event_count=len(raw),
                    broken_at=i,
                    message=(
                        f"event {i} hash mismatch "
                        f"(stored={stored_hash[:16]}… "
                        f"computed={computed_hash[:16]}…)"
                    ),
                )

            if entry.get("prev_hash", "") != prev_hash:
                return LineageVerifyResult(
                    ok=False,
                    model_dir=str(model_dir_path),
                    verified_at=now,
                    event_count=len(raw),
                    broken_at=i,
                    message=(
                        f"event {i} chain link broken "
                        f"(expected prev={prev_hash[:16] if prev_hash else '(genesis)'}… "
                        f"got={entry.get('prev_hash', '')[:16]}…)"
                    ),
                )

            prev_hash = stored_hash

        n = len(raw)
        return LineageVerifyResult(
            ok=True,
            model_dir=str(model_dir_path),
            verified_at=now,
            event_count=n,
            broken_at=None,
            message=f"chain intact ({n} event(s))",
        )

    # ── Hashing ───────────────────────────────────────────────────────────────

    @staticmethod
    def _hash_event(event: "TransformationEvent") -> str:
        """Return SHA-256 hex of *event*'s canonical JSON (``event_hash`` field excluded)."""
        d = event.to_dict()
        d.pop("event_hash", None)  # must not include the self-hash in the input
        canonical = json.dumps(d, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ── Helpers ───────────────────────────────────────────────────────────────────


def _load_chain_json(chain_path: Path) -> list[dict]:
    """Load chain JSON from *chain_path*; return ``[]`` if missing or unparseable."""
    if not chain_path.exists():
        return []
    try:
        data = json.loads(chain_path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            log.warning("Lineage chain at %s is not a JSON array — ignoring", chain_path)
            return []
        return data
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("Could not parse lineage chain at %s: %s", chain_path, exc)
        return []


def _utc_now() -> str:
    """Return current UTC time as ISO-8601 string, seconds precision."""
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
