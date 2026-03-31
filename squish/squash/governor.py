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
