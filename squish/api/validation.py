"""squish/api/validation.py — Shared request-parameter validators.

Every public API endpoint that accepts a JSON body must coerce caller-supplied
sampling parameters (``max_tokens``, ``temperature``, ``top_p`` …) into the
correct numeric type *and* range-check them.  Doing the raw cast inline
(``int(body.get("max_tokens"))``) turns a bad payload into an uncaught
``ValueError``/``TypeError`` which FastAPI renders as a **500 + traceback** —
the wrong answer for what is really a client mistake.

These helpers convert every malformed value into a clean
``HTTPException(400, ...)`` with a human-readable message naming the offending
parameter, so the same defensive behaviour is reused everywhere (DRY) instead
of being re-implemented per endpoint.

The module is import-safe on every platform — it pulls in no MLX / Metal code.
"""

from __future__ import annotations

import json
import math
from typing import Any

from fastapi import HTTPException, Request

# Inclusive sampling ranges mirrored from the JSON schema in
# ``squish/api/v1_router.py`` (_CHAT_COMPLETIONS_REQUEST).
_TEMPERATURE_MIN = 0.0
_TEMPERATURE_MAX = 2.0
_TOP_P_MIN = 0.0  # exclusive lower bound — top_p must be > 0
_TOP_P_MAX = 1.0  # inclusive upper bound

# Absolute upper bounds. Without these, a caller can pass an astronomically
# large ``max_tokens`` (e.g. "99999999999999999999999999") which ``int()``
# happily parses — and the server then attempts a near-infinite generation that
# saturates the backend.  The ceiling is far above any real context window, so
# legitimate requests are unaffected; only abuse is rejected with a 400.
_MAX_TOKENS_CEILING = 131_072
_MAX_STEPS_CEILING = 100


def _bad(param: str, message: str) -> HTTPException:
    """Build a uniform 400 error for a rejected parameter."""
    return HTTPException(status_code=400, detail=f"'{param}' {message}")


def _is_bool(value: Any) -> bool:
    """True only for genuine bools (``bool`` is an ``int`` subclass in Python)."""
    return isinstance(value, bool)


async def parse_json_body(request: Request) -> dict[str, Any]:
    """Return the request body as a dict, or raise 400 on malformed JSON.

    ``Request.json()`` raises :class:`json.JSONDecodeError` (a
    ``ValueError`` subclass) on malformed input; left unguarded that surfaces
    as a 500.  A non-object top-level JSON value (list, number, string) is also
    rejected because every endpoint expects an object body.
    """
    try:
        body = await request.json()
    except (json.JSONDecodeError, ValueError, UnicodeDecodeError) as exc:
        raise HTTPException(status_code=400, detail="request body must be valid JSON") from exc
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="request body must be a JSON object")
    return body


def parse_max_tokens(value: Any, default: int) -> int:
    """Coerce ``max_tokens`` to a finite int ≥ 0, else raise 400.

    ``0`` is permitted (some callers use it to request prompt-only handling);
    the schema's lower bound of 1 is enforced by callers that need it.
    """
    if value is None:
        return default
    if _is_bool(value) or not isinstance(value, (int, float, str)):
        raise _bad("max_tokens", "must be an integer")
    try:
        ivalue = int(value)
    except (ValueError, TypeError) as exc:
        raise _bad("max_tokens", "must be an integer") from exc
    if ivalue < 0:
        raise _bad("max_tokens", "must be >= 0")
    if ivalue > _MAX_TOKENS_CEILING:
        raise _bad("max_tokens", f"must be <= {_MAX_TOKENS_CEILING}")
    return ivalue


def parse_max_steps(value: Any, default: int) -> int:
    """Coerce ``max_steps`` to an int ≥ 1, else raise 400."""
    if value is None:
        return default
    if _is_bool(value) or not isinstance(value, (int, float, str)):
        raise _bad("max_steps", "must be an integer")
    try:
        ivalue = int(value)
    except (ValueError, TypeError) as exc:
        raise _bad("max_steps", "must be an integer") from exc
    if ivalue < 1:
        raise _bad("max_steps", "must be >= 1")
    if ivalue > _MAX_STEPS_CEILING:
        raise _bad("max_steps", f"must be <= {_MAX_STEPS_CEILING}")
    return ivalue


def _parse_finite_float(value: Any, param: str, default: float) -> float:
    """Coerce *value* to a finite float, raising 400 on NaN/inf/garbage."""
    if value is None:
        return default
    if _is_bool(value) or not isinstance(value, (int, float, str)):
        raise _bad(param, "must be a number")
    try:
        fvalue = float(value)
    except (ValueError, TypeError) as exc:
        raise _bad(param, "must be a number") from exc
    if not math.isfinite(fvalue):
        raise _bad(param, "must be a finite number")
    return fvalue


def parse_temperature(value: Any, default: float = 0.7) -> float:
    """Coerce ``temperature`` to a finite float in [0.0, 2.0], else raise 400."""
    fvalue = _parse_finite_float(value, "temperature", default)
    if not (_TEMPERATURE_MIN <= fvalue <= _TEMPERATURE_MAX):
        raise _bad(
            "temperature",
            f"must be between {_TEMPERATURE_MIN} and {_TEMPERATURE_MAX}",
        )
    return fvalue


def parse_top_p(value: Any, default: float = 0.9) -> float:
    """Coerce ``top_p`` to a finite float in (0.0, 1.0], else raise 400."""
    fvalue = _parse_finite_float(value, "top_p", default)
    if not (_TOP_P_MIN < fvalue <= _TOP_P_MAX):
        raise _bad(
            "top_p",
            f"must be > {_TOP_P_MIN} and <= {_TOP_P_MAX}",
        )
    return fvalue


def parse_embedding_input(value: Any) -> list[str]:
    """Validate ``/v1/embeddings`` ``input`` → non-empty list[str], else 400.

    Accepts a single non-empty string or a non-empty list of non-empty
    strings (mirroring the OpenAI embeddings contract).  Anything else —
    ``None``, empty string, empty list, a list containing non-strings or
    blank strings — is a client error.
    """
    if isinstance(value, str):
        if not value.strip():
            raise _bad("input", "must be a non-empty string")
        return [value]
    if isinstance(value, list):
        if not value:
            raise _bad("input", "must be a non-empty list")
        for item in value:
            if not isinstance(item, str) or not item.strip():
                raise _bad("input", "must be a list of non-empty strings")
        return list(value)
    raise _bad("input", "must be a string or a list of strings")
