"""squish.squash.integrations.kubernetes — Kubernetes Admission Webhook handler.

Provides a Validating Admission Webhook that enforces Squash attestation policy
at cluster admission time.  Pods that declare ``squash.ai/attestation-required:
"true"`` must carry a ``squash.ai/bom-digest`` annotation whose digest is
present in the local policy store and has passed all configured policies.

Pods without the ``squash.ai/attestation-required`` annotation are allowed
through with no check (opt-in model).

Architecture
------------
- :class:`WebhookConfig` — tunable parameters (policy store path, default_allow,
  annotation keys, policies list).
- :class:`KubernetesWebhookHandler` — stateless admission reviewer; testable
  without a running server.
- :func:`serve_webhook` — stdlib HTTPS server shim; used by the CLI.

Usage::

    from squish.squash.integrations.kubernetes import (
        KubernetesWebhookHandler,
        WebhookConfig,
        serve_webhook,
    )

    config = WebhookConfig(
        policy_store_path=Path("/var/squash/policy-store.json"),
        policies=["eu-ai-act", "enterprise-strict"],
    )
    handler = KubernetesWebhookHandler(config)

    # In tests: call handle() directly
    review = {"request": {"uid": "abc", "kind": {"kind": "Pod"}, ...}}
    response = handler.handle(review)
    assert response["response"]["allowed"] is True

    # In production: start the server
    serve_webhook(handler, port=8443, tls_cert="/tls/tls.crt", tls_key="/tls/tls.key")
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# ── Public annotation keys ─────────────────────────────────────────────────────

#: Pod annotation that opts the pod into Squash policy enforcement.
#: Value must be the string ``"true"`` for enforcement to activate.
ANNOTATION_REQUIRED = "squash.ai/attestation-required"

#: Pod annotation carrying the CycloneDX BOM SHA-256 digest.
#: Must be present when :data:`ANNOTATION_REQUIRED` is ``"true"``.
ANNOTATION_BOM_DIGEST = "squash.ai/bom-digest"

#: Pod annotation used to attach a human-readable policy summary (set by
#: a mutating webhook or CI pipeline for observability).
ANNOTATION_POLICY_SUMMARY = "squash.ai/policy-summary"


# ── Configuration ──────────────────────────────────────────────────────────────


@dataclass
class WebhookConfig:
    """Configuration for :class:`KubernetesWebhookHandler`.

    Attributes
    ----------
    policy_store_path:
        Path to a JSON file mapping BOM digest strings to ``bool``
        (``true`` = passed all policies, ``false`` = failed).  When *None*,
        the handler allows any pod that has the BOM digest annotation set
        (useful for mutation-only admission configs).
    default_allow:
        If *True* (the default), pods without the attestation-required
        annotation are allowed through without inspection.  Set to *False*
        to deny all unlabelled pods in high-security namespaces.
    required_annotation:
        Annotation key that opts a pod into enforcement.
    bom_digest_annotation:
        Annotation key carrying the BOM SHA-256 digest.
    policies:
        Policy names used when generating denials (informational only in
        the default implementation; extend for active policy re-evaluation).
    namespaces_exclude:
        Namespace names to skip; admission always allowed from these
        namespaces (e.g. ``["kube-system"]``).
    """

    policy_store_path: Path | None = None
    default_allow: bool = True
    required_annotation: str = ANNOTATION_REQUIRED
    bom_digest_annotation: str = ANNOTATION_BOM_DIGEST
    policies: list[str] = field(default_factory=lambda: ["enterprise-strict"])
    namespaces_exclude: list[str] = field(default_factory=lambda: ["kube-system"])


# ── Handler ────────────────────────────────────────────────────────────────────


class KubernetesWebhookHandler:
    """Stateless Kubernetes Validating Admission Webhook handler.

    Processes an ``AdmissionReview`` request dict and returns a valid
    ``AdmissionReview`` response dict suitable for serialising as JSON.

    The handler is intentionally free of Kubernetes SDK dependencies — it works
    with plain dicts so it can be tested and used without a cluster.
    """

    def __init__(self, config: WebhookConfig | None = None) -> None:
        self.config = config or WebhookConfig()
        self._store: dict[str, bool] | None = None  # lazy-loaded policy store

    def handle(self, admission_review: dict) -> dict:
        """Process an ``AdmissionReview`` request and return a response.

        Parameters
        ----------
        admission_review:
            The decoded JSON body of the admission webhook POST request,
            as specified in the Kubernetes Admission Review API v1.

        Returns
        -------
        dict
            A valid ``AdmissionReview`` response with ``apiVersion``,
            ``kind``, and ``response`` keys.
        """
        request: dict[str, Any] = admission_review.get("request") or {}
        uid: str = request.get("uid") or ""
        api_version: str = admission_review.get("apiVersion", "admission.k8s.io/v1")

        # Only enforce on Pod resources
        kind_obj: dict = request.get("kind") or {}
        resource_kind: str = kind_obj.get("kind", "")
        if resource_kind != "Pod":
            log.debug("Webhook: allowing non-Pod resource %s (uid=%s)", resource_kind, uid)
            return self._build_response(uid, allowed=True, api_version=api_version)

        # Skip excluded namespaces
        namespace: str = request.get("namespace") or ""
        if namespace in self.config.namespaces_exclude:
            log.debug("Webhook: allowing pod in excluded namespace %s (uid=%s)", namespace, uid)
            return self._build_response(uid, allowed=True, api_version=api_version)

        # Extract annotations from the pod object
        pod_obj: dict = request.get("object") or {}
        metadata: dict = pod_obj.get("metadata") or {}
        annotations: dict = metadata.get("annotations") or {}

        # Opt-in check
        if not _truthy(annotations.get(self.config.required_annotation)):
            if self.config.default_allow:
                log.debug("Webhook: allowing pod without attestation annotation (uid=%s)", uid)
                return self._build_response(uid, allowed=True, api_version=api_version)
            else:
                return self._build_response(
                    uid,
                    allowed=False,
                    status_message=(
                        f"Pod is missing required annotation '{self.config.required_annotation}'. "
                        "Set it to 'true' and provide a BOM digest to be admitted."
                    ),
                    status_code=403,
                    api_version=api_version,
                )

        # BOM digest must be present
        bom_digest: str = annotations.get(self.config.bom_digest_annotation, "").strip()
        if not bom_digest:
            return self._build_response(
                uid,
                allowed=False,
                status_message=(
                    f"Pod requires Squash attestation but annotation "
                    f"'{self.config.bom_digest_annotation}' is missing or empty."
                ),
                status_code=403,
                api_version=api_version,
            )

        # Policy store check (when a store is configured)
        if self.config.policy_store_path:
            store = self._load_store()
            if bom_digest not in store:
                return self._build_response(
                    uid,
                    allowed=False,
                    status_message=(
                        f"BOM digest '{bom_digest}' is not in the Squash policy store. "
                        "Run 'squash attest' and update the store before deploying."
                    ),
                    status_code=403,
                    api_version=api_version,
                )
            if not store[bom_digest]:
                policy_list = ", ".join(self.config.policies)
                return self._build_response(
                    uid,
                    allowed=False,
                    status_message=(
                        f"BOM digest '{bom_digest}' failed policy checks "
                        f"[{policy_list}]. Review the attestation report."
                    ),
                    status_code=403,
                    api_version=api_version,
                )

        log.info(
            "Webhook: allowing pod with bom-digest=%s (uid=%s, namespace=%s)",
            bom_digest,
            uid,
            namespace,
        )
        return self._build_response(uid, allowed=True, api_version=api_version)

    def reload_store(self) -> None:
        """Force-reload the policy store from disk on the next :meth:`handle` call."""
        self._store = None

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _load_store(self) -> dict[str, bool]:
        """Load (or return cached) policy store dict."""
        if self._store is not None:
            return self._store
        if not self.config.policy_store_path:
            self._store = {}
            return self._store
        try:
            raw = Path(self.config.policy_store_path).read_text(encoding="utf-8")
            self._store = json.loads(raw)
        except (OSError, json.JSONDecodeError) as e:
            log.warning("Webhook: could not load policy store %s — %s", self.config.policy_store_path, e)
            self._store = {}
        return self._store

    @staticmethod
    def _build_response(
        uid: str,
        *,
        allowed: bool,
        status_message: str | None = None,
        status_code: int = 200,
        api_version: str = "admission.k8s.io/v1",
    ) -> dict:
        """Build an ``AdmissionReview`` response dict."""
        response: dict[str, Any] = {
            "uid": uid,
            "allowed": allowed,
        }
        if not allowed and status_message:
            response["status"] = {
                "code": status_code,
                "message": status_message,
            }
        return {
            "apiVersion": api_version,
            "kind": "AdmissionReview",
            "response": response,
        }


# ── Server ─────────────────────────────────────────────────────────────────────


def serve_webhook(
    handler: KubernetesWebhookHandler,
    *,
    port: int = 8443,
    tls_cert: str | None = None,
    tls_key: str | None = None,
) -> None:  # pragma: no cover — integration-tested separately
    """Start the admission webhook HTTPS server.

    Blocks until interrupted (``KeyboardInterrupt`` / ``SIGTERM``).

    Parameters
    ----------
    handler:
        A configured :class:`KubernetesWebhookHandler` instance.
    port:
        TCP port to listen on (default: 8443, the Kubernetes standard for
        admission webhooks).
    tls_cert:
        Path to the PEM-encoded TLS certificate.  If *None*, the server
        listens over plain HTTP (useful for development only).
    tls_key:
        Path to the PEM-encoded TLS private key.
    """
    import http.server
    import ssl

    _handler_ref = handler  # capture for closure

    class _AdmissionHandler(http.server.BaseHTTPRequestHandler):
        def do_POST(self_inner) -> None:  # noqa: N805
            try:
                length = int(self_inner.headers.get("Content-Length", 0))
                body = self_inner.rfile.read(length)
                review = json.loads(body)
                response_dict = _handler_ref.handle(review)
                encoded = json.dumps(response_dict).encode()
                self_inner.send_response(200)
                self_inner.send_header("Content-Type", "application/json")
                self_inner.send_header("Content-Length", str(len(encoded)))
                self_inner.end_headers()
                self_inner.wfile.write(encoded)
            except Exception as exc:  # noqa: BLE001
                log.exception("Webhook: unhandled error processing request: %s", exc)
                self_inner.send_response(500)
                self_inner.end_headers()

        def log_message(self_inner, fmt: str, *args: object) -> None:  # noqa: N805
            log.debug("Webhook HTTP: " + fmt, *args)

    httpd = http.server.HTTPServer(("", port), _AdmissionHandler)
    if tls_cert and tls_key:
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(tls_cert, tls_key)
        httpd.socket = ctx.wrap_socket(httpd.socket, server_side=True)
        log.info("Webhook: listening on https://0.0.0.0:%d (TLS enabled)", port)
    else:
        log.warning("Webhook: listening on http://0.0.0.0:%d (no TLS — dev mode only)", port)

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        log.info("Webhook: shutting down")
    finally:
        httpd.server_close()


# ── Convenience ────────────────────────────────────────────────────────────────


def _truthy(value: object) -> bool:
    """Return True when *value* is the string 'true' (case-insensitive) or bool True."""
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"
