"""squish/squash/governor.py — Runtime compliance governor middleware.

Phase 3.  Registers as Starlette ``BaseHTTPMiddleware`` on the FastAPI app.

Boot gate (runs exactly once after the first request, memoised):
1. Locate ``cyclonedx-mlbom.json`` in the loaded model dir.
   If absent → skip all checks silently (non-fatal always).
2. Re-hash all weight files and compare the composite SHA-256 in the sidecar.
   Mismatch → WARNING log; in strict mode → flag ``_integrity_ok = False``.
3. Parse ``performanceMetrics``: find the ``arc_easy`` entry, read its
   ``deltaFromBaseline`` field.  If the implied accuracy ratio is below
   ``min_accuracy_ratio`` → WARNING log; strict → ``_accuracy_ok = False``.

Request handling:
- If strict and any check failed → HTTP 503 JSON body.
- Otherwise pass through.

All state is per-instance so tests can mount multiple independent governors.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

log = logging.getLogger(__name__)

# Module-level reference to the server _state object.
# governor.py imports it lazily so the module can be imported standalone
# (e.g. in tests without loading the full server).
_state    = None  # set to server._state by server.py at middleware registration
_INSTANCE: "SquashGovernor | None" = None  # set in __init__; read by /v1/health/model route


def _get_model_dir() -> Path | None:
    """Return the loaded model directory or None if no model is loaded."""
    if _state is None:
        return None
    d = getattr(_state, "model_dir", "") or ""
    return Path(d) if d else None


class SquashGovernor(BaseHTTPMiddleware):
    """Starlette middleware that enforces ML-BOM compliance at boot time.

    Parameters
    ----------
    app:
        The ASGI application to wrap.
    strict:
        When *True* and a compliance check fails, respond with HTTP 503 to
        every subsequent request.  When *False* (default), log warnings but
        never block traffic.
    min_accuracy_ratio:
        Minimum allowed ratio of quantised ``arc_easy`` score to baseline.
        Default 0.92 (8 pp drop).  Only consulted when
        ``performanceMetrics`` contains a ``deltaFromBaseline`` for ``arc_easy``.
    """

    def __init__(
        self,
        app,
        *,
        strict: bool = False,
        min_accuracy_ratio: float = 0.92,
    ) -> None:
        super().__init__(app)
        self.strict             = strict
        self.min_accuracy_ratio = min_accuracy_ratio

        self._boot_done:    bool        = False
        self._integrity_ok: bool | None = None   # None = not yet checked
        self._accuracy_ok:  bool | None = None
        self._detail:       str         = ""
        self._lock                      = asyncio.Lock()

        # Register as the module-level singleton so routes can access boot_state.
        import squish.squash.governor as _self_mod
        _self_mod._INSTANCE = self

    # ------------------------------------------------------------------
    # Middleware entry point
    # ------------------------------------------------------------------

    async def dispatch(self, request: Request, call_next) -> Response:
        # Skip compliance endpoints themselves to avoid recursion.
        if request.url.path in ("/v1/sbom", "/v1/health/model"):
            return await call_next(request)

        async with self._lock:
            if not self._boot_done:
                self._run_boot_checks()

        if (
            self.strict
            and self._boot_done
            and (self._integrity_ok is False or self._accuracy_ok is False)
        ):
            return JSONResponse(
                {
                    "error":  "model compliance check failed",
                    "detail": self._detail,
                },
                status_code=503,
            )

        return await call_next(request)

    # ------------------------------------------------------------------
    # Boot checks (synchronous — called under asyncio.Lock)
    # ------------------------------------------------------------------

    def _run_boot_checks(self) -> None:
        """Run integrity + accuracy checks exactly once."""
        try:
            self._do_boot_checks()
        except Exception as exc:
            log.warning("SquashGovernor: boot check raised unexpectedly: %s", exc)
        finally:
            self._boot_done = True

    def _do_boot_checks(self) -> None:
        model_dir = _get_model_dir()
        if model_dir is None or not model_dir.is_dir():
            log.debug("SquashGovernor: model not loaded yet — skipping checks")
            self._integrity_ok = True
            self._accuracy_ok  = True
            return

        sidecar = model_dir / "cyclonedx-mlbom.json"
        if not sidecar.exists():
            log.debug(
                "SquashGovernor: no sidecar at %s — compliance checks skipped", sidecar
            )
            self._integrity_ok = True
            self._accuracy_ok  = True
            return

        bom: dict = json.loads(sidecar.read_text())
        component: dict = bom.get("components", [{}])[0]

        # ── 1. Hash integrity ─────────────────────────────────────────────
        self._integrity_ok = self._check_integrity(model_dir, component)

        # ── 2. Accuracy ratio ─────────────────────────────────────────────
        self._accuracy_ok  = self._check_accuracy(component)

    def _check_integrity(self, model_dir: Path, component: dict) -> bool:
        stored_hashes = component.get("hashes", [])
        if not stored_hashes:
            log.debug("SquashGovernor: sidecar has no composite hash — integrity unchecked")
            return True

        stored_composite = stored_hashes[0].get("content", "")
        if not stored_composite:
            return True

        # Import the hashing helpers from sbom_builder (never duplicate logic).
        try:
            from squish.squash.sbom_builder import CycloneDXBuilder
        except ImportError:
            log.debug("SquashGovernor: sbom_builder unavailable — integrity unchecked")
            return True

        file_hashes  = CycloneDXBuilder._hash_weight_files(model_dir)
        live_composite = CycloneDXBuilder._composite_hash(file_hashes) if file_hashes else ""

        if live_composite != stored_composite:
            msg = (
                f"SquashGovernor: composite hash mismatch — "
                f"sidecar={stored_composite[:12]}… live={live_composite[:12]}…"
            )
            log.warning(msg)
            self._detail = msg
            return False

        log.debug("SquashGovernor: composite hash verified OK")
        return True

    def _check_accuracy(self, component: dict) -> bool:
        metrics: list[dict] = (
            component
            .get("modelCard", {})
            .get("quantitativeAnalysis", {})
            .get("performanceMetrics", [])
        )
        if not metrics:
            log.debug("SquashGovernor: no performanceMetrics — accuracy check skipped")
            return True

        arc = next((m for m in metrics if m.get("slice") == "arc_easy"), None)
        if arc is None:
            log.debug("SquashGovernor: no arc_easy metric — accuracy check skipped")
            return True

        delta_str = arc.get("deltaFromBaseline")
        if delta_str is None:
            log.debug("SquashGovernor: arc_easy has no deltaFromBaseline — accuracy check skipped")
            return True

        try:
            delta = float(delta_str)
        except ValueError:
            log.warning("SquashGovernor: unparseable deltaFromBaseline=%r", delta_str)
            return True

        # Ratio check: delta = quant_score - baseline_score.
        # delta >= -8pp  →  ratio >= 0.92 for a typical 70-point baseline.
        # We use the threshold directly: delta < -8 means ratio < 0.92.
        threshold_pp = (self.min_accuracy_ratio - 1.0) * 100  # e.g. -8.0 for 0.92
        if delta < threshold_pp:
            msg = (
                f"SquashGovernor: arc_easy deltaFromBaseline={delta_str} "
                f"below threshold {threshold_pp:.1f}pp"
            )
            log.warning(msg)
            self._detail = msg
            return False

        log.debug("SquashGovernor: arc_easy accuracy ratio OK (delta=%s)", delta_str)
        return True

    # ------------------------------------------------------------------
    # Accessor for routes
    # ------------------------------------------------------------------

    @property
    def boot_state(self) -> dict:
        """Return a dict suitable for /v1/health/model."""
        return {
            "integrity_ok":       self._integrity_ok,
            "accuracy_ok":        self._accuracy_ok,
            "strict_compliance":  self.strict,
        }


# ── Wave 24 — Drift Detection & Continuous Monitoring ─────────────────────────

import datetime as _dt  # noqa: E402
import hashlib as _hashlib  # noqa: E402
import json as _json_drift  # noqa: E402
import threading as _threading  # noqa: E402
from dataclasses import dataclass as _dataclass, field as _field  # noqa: E402
from pathlib import Path as _Path  # noqa: E402
from typing import Callable as _Callable  # noqa: E402


@_dataclass
class DriftEvent:
    """A single detected drift event.

    Attributes
    ----------
    event_type:
        One of ``"BOM_CHANGED"``, ``"CVE_APPEARED"``, ``"POLICY_REGRESSED"``.
    component:
        Human-readable component / artifact name that changed.
    old_value:
        Previous value (string representation).
    new_value:
        Current value (string representation).
    detected_at:
        ISO-8601 UTC timestamp.
    """

    event_type: str
    component: str
    old_value: str
    new_value: str
    detected_at: str = _field(
        default_factory=lambda: _dt.datetime.now(_dt.timezone.utc).isoformat()
    )


class DriftMonitor:
    """Compare model directory state snapshots and emit :class:`DriftEvent` objects.

    Typical CI usage::

        snap = DriftMonitor.snapshot(model_dir)
        # … later …
        events = DriftMonitor.compare(model_dir, snap)
        for e in events:
            print(e.event_type, e.component)

    Long-running monitoring::

        DriftMonitor.watch(model_dir, interval_s=3600,
                           callback=lambda evts: alert(evts))
    """

    @staticmethod
    def snapshot(model_dir: _Path) -> str:
        """Return a SHA-256 fingerprint of the attestation state in *model_dir*.

        Covers: CycloneDX BOM, scan results, attestation record, VEX report,
        and all per-policy JSON files.  Missing files skipped silently.
        """
        model_dir = _Path(model_dir)
        parts: list[str] = []
        for name in sorted([
            "cyclonedx-mlbom.json",
            "squash-scan.json",
            "squash-attest.json",
            "squash-vex-report.json",
        ]):
            p = model_dir / name
            parts.append(p.read_text(encoding="utf-8") if p.exists() else "")
        for p in sorted(model_dir.glob("squash-policy-*.json")):
            parts.append(p.read_text(encoding="utf-8"))
        return _hashlib.sha256("\n".join(parts).encode()).hexdigest()

    @staticmethod
    def compare(
        model_dir: _Path,
        baseline_snapshot: str,
    ) -> list[DriftEvent]:
        """Compare current state of *model_dir* against *baseline_snapshot*.

        Returns a list of :class:`DriftEvent` describing what changed.
        An empty list means no drift.
        """
        model_dir = _Path(model_dir)
        events: list[DriftEvent] = []
        now = _dt.datetime.now(_dt.timezone.utc).isoformat()

        current_snap = DriftMonitor.snapshot(model_dir)
        if current_snap == baseline_snapshot:
            return events

        # BOM identity changed
        bom_path = model_dir / "cyclonedx-mlbom.json"
        if not bom_path.exists():
            events.append(DriftEvent(
                event_type="BOM_CHANGED",
                component="cyclonedx-mlbom.json",
                old_value=baseline_snapshot[:16] + "…",
                new_value="(missing)",
                detected_at=now,
            ))
            return events

        events.append(DriftEvent(
            event_type="BOM_CHANGED",
            component="cyclonedx-mlbom.json",
            old_value=baseline_snapshot[:16] + "…",
            new_value=current_snap[:16] + "…",
            detected_at=now,
        ))

        # CVE check — report any non-fixed/non-affected vulnerabilities
        vex_path = model_dir / "squash-vex-report.json"
        if vex_path.exists():
            try:
                vex = _json_drift.loads(vex_path.read_text(encoding="utf-8"))
                for stmt in vex.get("statements", []):
                    state = stmt.get("analysis", {}).get("state", "")
                    if state not in ("not_affected", "fixed"):
                        cve_id = stmt.get("vulnerability", {}).get("id", "unknown")
                        events.append(DriftEvent(
                            event_type="CVE_APPEARED",
                            component=cve_id,
                            old_value="not_present",
                            new_value="present",
                            detected_at=now,
                        ))
            except Exception:
                pass

        # Policy regression check
        for policy_path in sorted(model_dir.glob("squash-policy-*.json")):
            try:
                pr = _json_drift.loads(policy_path.read_text(encoding="utf-8"))
                if not pr.get("passed", True) and pr.get("error_count", 0) > 0:
                    events.append(DriftEvent(
                        event_type="POLICY_REGRESSED",
                        component=policy_path.stem,
                        old_value="passed",
                        new_value=f"failed ({pr.get('error_count')} errors)",
                        detected_at=now,
                    ))
            except Exception:
                pass

        return events

    @staticmethod
    def watch(
        model_dir: _Path,
        interval_s: float,
        callback: _Callable[[list[DriftEvent]], None],
    ) -> "_threading.Event":
        """Poll *model_dir* every *interval_s* seconds; call *callback* on drift.

        Returns a :class:`threading.Event` that callers can ``set()`` to stop
        the background thread.

        Example::

            stop = DriftMonitor.watch(model_dir, 3600, lambda evts: print(evts))
            # … later …
            stop.set()
        """
        model_dir = _Path(model_dir)
        baseline = DriftMonitor.snapshot(model_dir)
        stop_event = _threading.Event()

        def _loop() -> None:
            nonlocal baseline
            while not stop_event.wait(timeout=interval_s):
                events = DriftMonitor.compare(model_dir, baseline)
                if events:
                    try:
                        callback(events)
                    except Exception:
                        pass
                    baseline = DriftMonitor.snapshot(model_dir)

        t = _threading.Thread(target=_loop, daemon=True)
        t.start()
        return stop_event


# ── Wave 46 — Agent Audit Trail ───────────────────────────────────────────────

import hashlib as _hashlib_audit  # noqa: E402
import os as _os_audit  # noqa: E402
from dataclasses import asdict as _asdict, dataclass as _dc_audit  # noqa: E402


@_dc_audit
class AuditEntry:
    """Immutable audit log entry; entry_hash = sha256(prev|seq|type|ts|in|out). EU AI Act Art. 12."""

    seq: int          # monotonically increasing sequence number
    ts: str           # ISO-8601 UTC timestamp
    session_id: str   # caller-supplied session / request identifier
    event_type: str   # "llm_start", "llm_end", "attestation", "mcp_scan", …
    model_id: str     # model name/path/hash
    input_hash: str   # SHA-256 of raw input (hex), or ""
    output_hash: str  # SHA-256 of raw output (hex), or ""
    latency_ms: float # elapsed ms, or -1
    metadata: dict    # JSON-serialisable extras
    prev_hash: str    # entry_hash of preceding entry ("" for seq 0)
    entry_hash: str   # forward-chain hash — see class docstring

    @staticmethod
    def _compute_hash(prev_hash: str, seq: int, event_type: str, ts: str,
                      input_hash: str, output_hash: str) -> str:
        raw = f"{prev_hash}|{seq}|{event_type}|{ts}|{input_hash}|{output_hash}"
        return _hashlib_audit.sha256(raw.encode()).hexdigest()


def _hash_text(text: str) -> str:
    return _hashlib_audit.sha256(text.encode("utf-8", errors="replace")).hexdigest()


class AgentAuditLogger:
    """Append-only JSONL audit logger (SHA-256 hash chain, EU AI Act Art. 12).
    Default: $SQUASH_AUDIT_LOG or ~/.squash/audit.jsonl. Thread-safe.
    """

    def __init__(self, log_path: "_Path | str | None" = None) -> None:
        if log_path is None:
            env = _os_audit.environ.get("SQUASH_AUDIT_LOG", "")
            log_path = _Path(env) if env else _Path.home() / ".squash" / "audit.jsonl"
        self._path = _Path(log_path)
        self._lock = _threading.Lock()
        self._seq: int = -1  # lazily initialised on first append

    @property
    def path(self) -> _Path:
        return self._path

    def _ensure_dir(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def _last_entry(self) -> "dict | None":
        if not self._path.exists():
            return None
        with self._path.open("r", encoding="utf-8") as fh:
            last: "dict | None" = None
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        last = _json_drift.loads(line)
                    except Exception:
                        pass
        return last

    def append(
        self,
        *,
        session_id: str = "",
        event_type: str,
        model_id: str = "",
        input_hash: str = "",
        output_hash: str = "",
        latency_ms: float = -1.0,
        metadata: "dict | None" = None,
    ) -> AuditEntry:
        """Append one entry to the audit log and return it (thread-safe)."""
        with self._lock:
            self._ensure_dir()
            if self._seq < 0:
                last = self._last_entry()
                self._seq = (last["seq"] + 1) if last else 0
                self._prev_hash = last["entry_hash"] if last else ""
            else:
                self._seq += 1

            ts = _dt.datetime.now(_dt.timezone.utc).isoformat()
            entry_hash = AuditEntry._compute_hash(
                self._prev_hash, self._seq, event_type, ts, input_hash, output_hash
            )
            entry = AuditEntry(
                seq=self._seq,
                ts=ts,
                session_id=session_id,
                event_type=event_type,
                model_id=model_id,
                input_hash=input_hash,
                output_hash=output_hash,
                latency_ms=latency_ms,
                metadata=metadata or {},
                prev_hash=self._prev_hash,
                entry_hash=entry_hash,
            )
            self._prev_hash = entry_hash

            line = _json_drift.dumps(_asdict(entry), separators=(",", ":")) + "\n"
            with self._path.open("a", encoding="utf-8") as fh:
                fh.write(line)

        return entry

    def read_tail(self, n: int = 100) -> "list[dict]":
        """Return the last *n* entries as plain dicts (oldest first)."""
        if not self._path.exists():
            return []
        entries: list[dict] = []
        with self._path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        entries.append(_json_drift.loads(line))
                    except Exception:
                        pass
        return entries[-n:] if len(entries) > n else entries

    def verify_chain(self) -> "tuple[bool, str]":
        """Verify the forward hash chain; return ``(True, "")`` or ``(False, reason)``."""
        if not self._path.exists():
            return True, ""
        prev = ""
        with self._path.open("r", encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    e = _json_drift.loads(line)
                except Exception:
                    return False, f"line {lineno}: invalid JSON"
                expected = AuditEntry._compute_hash(
                    prev, e["seq"], e["event_type"], e["ts"],
                    e["input_hash"], e["output_hash"],
                )
                if e["entry_hash"] != expected:
                    return False, (
                        f"line {lineno} seq={e['seq']}: entry_hash mismatch "
                        f"(expected {expected[:12]}… got {e['entry_hash'][:12]}…)"
                    )
                if e["prev_hash"] != prev:
                    return False, (
                        f"line {lineno} seq={e['seq']}: prev_hash mismatch"
                    )
                prev = e["entry_hash"]
        return True, ""


_AUDIT_LOGGER: "AgentAuditLogger | None" = None


def get_audit_logger() -> AgentAuditLogger:
    """Return the process-level :class:`AgentAuditLogger` singleton."""
    global _AUDIT_LOGGER
    if _AUDIT_LOGGER is None:
        _AUDIT_LOGGER = AgentAuditLogger()
    return _AUDIT_LOGGER
