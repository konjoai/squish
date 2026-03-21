"""squish/logging_config.py — Central structured logging configuration for Squish.

Provides a single entry point to configure the ``squish.*`` logger hierarchy
for both human-readable (rich or plain) and machine-readable (JSON) output.

Usage
─────
    from squish.logging_config import configure_logging, get_squish_logger

    # Rich pretty output at DEBUG level
    configure_logging(level="debug", use_rich=True)

    # JSON lines to stderr (e.g. for log aggregation)
    configure_logging(level="info", json_format=True)

    # Get a module-level logger
    log = get_squish_logger(__name__)
    log.info("Model loaded in %.2fs", elapsed)
"""

from __future__ import annotations

import logging
import sys
from typing import Any

__all__ = ["configure_logging", "get_squish_logger"]

# Root logger for the entire squish package hierarchy.
_ROOT_LOGGER: logging.Logger = logging.getLogger("squish")


class _JsonFormatter(logging.Formatter):
    """Format :class:`logging.LogRecord` objects as single-line JSON strings."""

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        import json as _json

        _STDLIB_KEYS = frozenset(logging.LogRecord(
            "", 0, "", 0, "", (), None,
        ).__dict__.keys()) | {"message", "asctime"}

        payload: dict[str, Any] = {
            "ts":      self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level":   record.levelname,
            "logger":  record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        # Include any extra fields the caller passed via ``extra={}``
        for key, val in record.__dict__.items():
            if key not in _STDLIB_KEYS and not key.startswith("_"):
                payload[key] = val
        return _json.dumps(payload, default=str)


def configure_logging(
    level:       str  = "warning",
    use_rich:    bool = False,
    json_format: bool = False,
) -> None:
    """Configure the ``squish.*`` logger hierarchy.

    Safe to call multiple times; each call replaces previous handlers so
    duplicate log lines are never emitted (important in unit tests that call
    this function more than once).

    Parameters
    ----------
    level       : Log level name — ``"debug"``, ``"info"``, ``"warning"``,
                  ``"error"``, or ``"critical"``.  Unknown values fall back to
                  WARNING.
    use_rich    : Attach a :class:`rich.logging.RichHandler` for pretty
                  terminal output.  Ignored when ``json_format=True``.
    json_format : Emit each log record as a JSON object on *stderr*.
                  Takes precedence over ``use_rich``.
    """
    numeric_level = getattr(logging, level.upper(), logging.WARNING)
    _ROOT_LOGGER.setLevel(numeric_level)

    # Remove any handlers accumulated by previous calls to prevent duplicates.
    _ROOT_LOGGER.handlers.clear()

    if json_format:
        handler: logging.Handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(_JsonFormatter())
    elif use_rich:
        try:
            from rich.logging import RichHandler
            handler = RichHandler(show_time=True, show_path=False)
        except ImportError:  # pragma: no cover
            handler = logging.StreamHandler(sys.stderr)
    else:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter("%(asctime)s  %(levelname)-8s  %(name)s  %(message)s")
        )

    _ROOT_LOGGER.addHandler(handler)
    # Prevent messages from escalating to the root (stdlib) logger, which
    # would cause duplicate output when the calling application also
    # configures logging.basicConfig.
    _ROOT_LOGGER.propagate = False


def get_squish_logger(name: str) -> logging.Logger:
    """Return a child logger in the ``squish.*`` namespace.

    Parameters
    ----------
    name : Typically ``__name__`` of the calling module.  Names that already
           start with ``"squish"`` are used as-is; others are prefixed with
           ``"squish."``.
    """
    if name.startswith("squish"):
        return logging.getLogger(name)
    return logging.getLogger(f"squish.{name}")
