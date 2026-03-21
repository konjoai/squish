"""tests/test_logging_config_unit.py

Full unit tests for squish/logging_config.py.

Coverage targets
────────────────
 • _JsonFormatter.format
     – basic log record (exc_info=False, no extra fields)
     – exc_info=True branch
     – extra fields branch (key not in _STDLIB_KEYS and not starting with _)
     – stdlib-key / private-key False branches (both outcomes of the compound
       condition in the for-loop)
 • configure_logging
     – json_format=True  (JSON handler)
     – use_rich=True     (RichHandler)
     – neither           (plain StreamHandler, the default)
     – idempotent: calling twice must not double up handlers
     – propagate=False always set
 • get_squish_logger
     – name already starts with "squish" → used as-is
     – name does NOT start with "squish" → "squish." prefix added
"""

from __future__ import annotations

import json
import logging
import sys

import pytest

from squish.logging_config import (
    _JsonFormatter,
    _ROOT_LOGGER,
    configure_logging,
    get_squish_logger,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _clean_root_logger():
    """Restore the squish root logger to a clean state after every test."""
    original_level     = _ROOT_LOGGER.level
    original_handlers  = list(_ROOT_LOGGER.handlers)
    original_propagate = _ROOT_LOGGER.propagate
    yield
    _ROOT_LOGGER.handlers  = original_handlers
    _ROOT_LOGGER.level     = original_level
    _ROOT_LOGGER.propagate = original_propagate


# ═══════════════════════════════════════════════════════════════════════════════
# _JsonFormatter
# ═══════════════════════════════════════════════════════════════════════════════

class TestJsonFormatter:

    def _make_record(
        self,
        msg:       str  = "hello",
        level:     int  = logging.INFO,
        exc_info:  bool = False,
        **extra,
    ) -> logging.LogRecord:
        record = logging.LogRecord(
            name    = "squish.test",
            level   = level,
            pathname= "",
            lineno  = 0,
            msg     = msg,
            args    = (),
            exc_info= None,
        )
        if exc_info:
            try:
                raise ValueError("test-exception")
            except ValueError:
                import sys
                record.exc_info = sys.exc_info()
        for k, v in extra.items():
            setattr(record, k, v)
        return record

    def test_basic_record(self):
        fmt    = _JsonFormatter()
        record = self._make_record("world")
        payload = json.loads(fmt.format(record))
        assert payload["level"]   == "INFO"
        assert payload["message"] == "world"
        assert payload["logger"]  == "squish.test"
        assert "ts" in payload

    def test_exc_info_included(self):
        fmt    = _JsonFormatter()
        record = self._make_record("boom", exc_info=True)
        payload = json.loads(fmt.format(record))
        assert "exc" in payload
        assert "ValueError" in payload["exc"]

    def test_extra_field_included(self):
        """A custom attribute added to the record appears in the JSON output."""
        fmt    = _JsonFormatter()
        record = self._make_record("extra-test", custom_key="my_value")
        payload = json.loads(fmt.format(record))
        assert payload["custom_key"] == "my_value"

    def test_stdlib_keys_excluded(self):
        """Standard LogRecord attributes must NOT be duplicated in the JSON."""
        fmt    = _JsonFormatter()
        record = self._make_record("stdlib-test")
        payload = json.loads(fmt.format(record))
        # 'levelname' is a stdlib key — it must not appear as a raw key again
        # (it IS present as 'level' via the explicit mapping)
        assert "levelname" not in payload

    def test_private_key_excluded(self):
        """Attributes starting with '_' must be excluded from extra fields."""
        fmt    = _JsonFormatter()
        record = self._make_record("private-test")
        record._private_attr = "secret"
        payload = json.loads(fmt.format(record))
        assert "_private_attr" not in payload

    def test_no_exc_info(self):
        """When no exception, 'exc' key must be absent."""
        fmt    = _JsonFormatter()
        record = self._make_record("no-exc")
        payload = json.loads(fmt.format(record))
        assert "exc" not in payload


# ═══════════════════════════════════════════════════════════════════════════════
# configure_logging
# ═══════════════════════════════════════════════════════════════════════════════

class TestConfigureLogging:

    def test_json_format_attaches_json_formatter(self, capsys):
        configure_logging(level="info", json_format=True)
        logger = get_squish_logger("test.json")
        logger.info("json-output")
        captured = capsys.readouterr().err
        payload  = json.loads(captured.strip())
        assert payload["message"] == "json-output"

    def test_use_rich_attaches_rich_handler(self):
        configure_logging(level="debug", use_rich=True)
        assert len(_ROOT_LOGGER.handlers) == 1
        from rich.logging import RichHandler
        assert isinstance(_ROOT_LOGGER.handlers[0], RichHandler)

    def test_plain_handler_by_default(self):
        configure_logging(level="warning")
        assert len(_ROOT_LOGGER.handlers) == 1
        assert isinstance(_ROOT_LOGGER.handlers[0], logging.StreamHandler)

    def test_json_takes_precedence_over_rich(self, capsys):
        """json_format=True must take precedence over use_rich=True."""
        configure_logging(level="info", use_rich=True, json_format=True)
        # Handler must be a plain StreamHandler with a _JsonFormatter, not RichHandler
        handler = _ROOT_LOGGER.handlers[0]
        assert isinstance(handler.formatter, _JsonFormatter)

    def test_idempotent_no_duplicate_handlers(self):
        """Calling configure_logging twice must not accumulate handlers."""
        configure_logging(level="info")
        configure_logging(level="debug")
        assert len(_ROOT_LOGGER.handlers) == 1

    def test_propagate_false(self):
        configure_logging()
        assert _ROOT_LOGGER.propagate is False

    def test_level_debug_set(self):
        configure_logging(level="debug")
        assert _ROOT_LOGGER.level == logging.DEBUG

    def test_level_info_set(self):
        configure_logging(level="info")
        assert _ROOT_LOGGER.level == logging.INFO

    def test_level_warning_default(self):
        configure_logging()
        assert _ROOT_LOGGER.level == logging.WARNING

    def test_unknown_level_falls_back_to_warning(self):
        configure_logging(level="verbosely_chatty")
        assert _ROOT_LOGGER.level == logging.WARNING


# ═══════════════════════════════════════════════════════════════════════════════
# get_squish_logger
# ═══════════════════════════════════════════════════════════════════════════════

class TestGetSquishLogger:

    def test_non_squish_name_gets_prefix(self):
        log = get_squish_logger("mymodule")
        assert log.name == "squish.mymodule"

    def test_squish_prefixed_name_unchanged(self):
        log = get_squish_logger("squish.server")
        assert log.name == "squish.server"

    def test_squish_exact_unchanged(self):
        log = get_squish_logger("squish")
        assert log.name == "squish"

    def test_returns_logger_instance(self):
        log = get_squish_logger("test.thing")
        assert isinstance(log, logging.Logger)
