"""squish/serving/cors_config.py — CORS Configuration for the Squish API Server.

Wave 72: provides a :class:`CORSConfig` dataclass and helpers for applying
Cross-Origin Resource Sharing headers to HTTP responses. Suitable for use with
any WSGI/ASGI-style handler that exposes response headers as a mutable dict.

Usage::

    from squish.serving.cors_config import CORSConfig, apply_cors_headers

    config = CORSConfig(allowed_origins=["https://app.example.com"])

    # In your request handler:
    headers = {}
    apply_cors_headers(headers, request_origin="https://app.example.com", config=config)
    # headers now contains the appropriate CORS entries
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


__all__ = [
    "CORSConfig",
    "apply_cors_headers",
    "is_origin_allowed",
    "DEFAULT_CORS",
]


@dataclass
class CORSConfig:
    """Declarative CORS policy for the Squish REST API.

    Attributes:
        allowed_origins: List of permitted origins. Use ``["*"]`` to allow any
            origin (open / development mode).  Supports exact matches and prefix
            matches (e.g. ``"https://*.example.com"`` is matched with
            ``startswith`` after stripping the wildcard prefix).
        allowed_methods: HTTP methods to advertise in the preflight response.
        allowed_headers: Request headers to permit. ``*`` means all.
        expose_headers: Response headers the browser may access.
        max_age: Seconds for browsers to cache the preflight result.
        allow_credentials: Whether to include ``Access-Control-Allow-Credentials``.
            Must be ``False`` when ``allowed_origins`` contains ``"*"`` — browsers
            reject credentialled wildcard responses.
    """

    allowed_origins: list[str] = field(
        default_factory=lambda: ["*"]
    )
    allowed_methods: list[str] = field(
        default_factory=lambda: ["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD"]
    )
    allowed_headers: list[str] = field(
        default_factory=lambda: ["Content-Type", "Authorization", "X-Requested-With"]
    )
    expose_headers: list[str] = field(default_factory=list)
    max_age: int = 86_400
    allow_credentials: bool = False

    def __post_init__(self) -> None:
        if self.allow_credentials and "*" in self.allowed_origins:
            raise ValueError(
                "CORSConfig: allow_credentials=True is incompatible with "
                "allowed_origins=['*'] — browsers block credentialled wildcard responses. "
                "Specify explicit origins instead."
            )


# Shared default instance for the Squish server (open / development mode)
DEFAULT_CORS = CORSConfig(allowed_origins=["*"])


def is_origin_allowed(origin: str, allowed_origins: list[str]) -> bool:
    """Test whether *origin* is permitted by *allowed_origins*.

    Rules:
    - ``"*"`` matches any origin.
    - An exact string match is checked first.
    - A pattern starting with ``"*."`` is treated as a subdomain wildcard:
      ``"*.example.com"`` matches ``"https://foo.example.com"``.

    Args:
        origin: The ``Origin`` header value from the request.
        allowed_origins: List of allowed origins from :class:`CORSConfig`.

    Returns:
        ``True`` if *origin* is permitted.
    """
    if not origin:
        return False
    for allowed in allowed_origins:
        if allowed == "*":
            return True
        if allowed == origin:
            return True
        # Wildcard subdomain: *.example.com
        if allowed.startswith("*."):
            suffix = allowed[1:]  # e.g. ".example.com"
            # Strip scheme from origin for comparison, then re-add
            scheme_end = origin.find("://")
            if scheme_end != -1:
                host_part = origin[scheme_end + 3:]  # strip "https://"
                if ("." + host_part).endswith(suffix) or host_part.endswith(suffix):
                    # Ensure it IS a subdomain (at least one extra label)
                    if host_part.endswith(suffix.lstrip(".")):
                        remaining = host_part[: -len(suffix.lstrip("."))]
                        if remaining.endswith("."):
                            return True
    return False


def apply_cors_headers(
    headers: dict,
    request_origin: Optional[str],
    config: CORSConfig,
    *,
    is_preflight: bool = False,
) -> None:
    """Mutate *headers* by adding the appropriate CORS entries.

    Args:
        headers: Mutable dict of response headers (modified in-place).
        request_origin: Value of the ``Origin`` request header; may be ``None``.
        config: Active :class:`CORSConfig` policy.
        is_preflight: If ``True``, include preflight-specific headers
            (``Access-Control-Allow-Methods``, ``Access-Control-Allow-Headers``,
            ``Access-Control-Max-Age``).
    """
    if not request_origin:
        return

    origin_ok = is_origin_allowed(request_origin, config.allowed_origins)
    if not origin_ok:
        # Omit CORS headers — browser will block the cross-origin request
        return

    # Decide which origin to echo back
    if "*" in config.allowed_origins:
        headers["Access-Control-Allow-Origin"] = "*"
    else:
        headers["Access-Control-Allow-Origin"] = request_origin
        # Tell caches that the response varies by Origin
        vary = headers.get("Vary", "")
        if "Origin" not in vary:
            headers["Vary"] = f"{vary}, Origin".lstrip(", ")

    if config.allow_credentials:
        headers["Access-Control-Allow-Credentials"] = "true"

    if config.expose_headers:
        headers["Access-Control-Expose-Headers"] = ", ".join(config.expose_headers)

    if is_preflight:
        headers["Access-Control-Allow-Methods"] = ", ".join(config.allowed_methods)
        if config.allowed_headers == ["*"]:
            headers["Access-Control-Allow-Headers"] = "*"
        else:
            headers["Access-Control-Allow-Headers"] = ", ".join(config.allowed_headers)
        headers["Access-Control-Max-Age"] = str(config.max_age)
