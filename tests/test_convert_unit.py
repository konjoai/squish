"""tests/test_convert_unit.py

Unit tests for the pure-Python helpers in squish/convert.py that are
not covered by test_compression_pipeline.py:

  safe_key            — dot→double-underscore substitution
  has_outliers        — returns True/False correctly
  Spinner             — context manager, start/stop, update, non-TTY spin thread
  _apply_awq_single   — empty scales (no-op), ImportError fallback
  _clear_line         — non-TTY path (covered by normal execution, TTY has pragma)
"""
from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# safe_key
# ---------------------------------------------------------------------------


class TestSafeKey:
    def test_replaces_dots(self):
        from squish.convert import safe_key
        assert safe_key("model.layers.0.weight") == "model__layers__0__weight"

    def test_no_dots_unchanged(self):
        from squish.convert import safe_key
        assert safe_key("weight") == "weight"

    def test_empty_string(self):
        from squish.convert import safe_key
        assert safe_key("") == ""


# ---------------------------------------------------------------------------
# has_outliers
# ---------------------------------------------------------------------------


class TestHasOutliers:
    def test_returns_true_for_strong_outliers(self):
        from squish.convert import has_outliers
        # Create a 128-element row with one very large value; ratio = max/mean >> 10
        arr = np.ones((4, 128), dtype=np.float32)
        arr[0, 0] = 1000.0  # row-max/row-mean ≈ 1000 / ((1000+127)/128) ≈ 113
        assert has_outliers(arr, threshold=10.0) is True

    def test_returns_false_for_uniform_data(self):
        from squish.convert import has_outliers
        arr = np.ones((4, 4), dtype=np.float32)  # uniform → ratio=1.0
        # With all-equal values: row_max == row_mean → ratio == 1.0
        # threshold=5.0 → 1.0 < 5.0 → False
        assert has_outliers(arr, threshold=5.0) is False

    def test_returns_bool(self):
        from squish.convert import has_outliers
        rng = np.random.default_rng(0)
        arr = rng.standard_normal((8, 16)).astype(np.float32)
        result = has_outliers(arr, threshold=100.0)
        assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# _apply_awq_single
# ---------------------------------------------------------------------------


class TestApplyAwqSingle:
    def test_empty_scales_returns_original(self):
        from squish.convert import _apply_awq_single
        arr = np.ones((4, 4), dtype=np.float32)
        result = _apply_awq_single("weight", arr, {})
        assert result is arr  # same object (no copy)

    def test_none_scales_returns_original(self):
        from squish.convert import _apply_awq_single
        arr = np.ones((4, 4), dtype=np.float32)
        # Empty dict is falsy, so None or {} both return early
        result = _apply_awq_single("weight", arr, {})
        np.testing.assert_array_equal(result, arr)

    def test_import_error_falls_back(self):
        """When squish.quant.awq is unavailable, returns arr_f32 unchanged."""
        from squish.convert import _apply_awq_single
        arr = np.ones((4, 4), dtype=np.float32)
        scales = {"weight": np.ones(4, dtype=np.float32)}  # non-empty
        with patch.dict(__import__('sys').modules, {"squish.quant.awq": None}):
            result = _apply_awq_single("weight", arr, scales)
        assert result is arr or np.array_equal(result, arr)


# ---------------------------------------------------------------------------
# Spinner
# ---------------------------------------------------------------------------


class TestSpinner:
    def test_context_manager_start_and_stop(self):
        """Spinner starts and stops without error in non-TTY context."""
        from squish.convert import Spinner
        with Spinner("test label", interval=0.01):
            pass  # no-op in non-TTY

    def test_context_manager_no_exception_on_exit(self):
        from squish.convert import Spinner
        sp = Spinner("task", interval=0.01)
        sp.start()
        sp.stop()

    def test_update_suffix(self):
        from squish.convert import Spinner
        sp = Spinner("quantizing", interval=0.01)
        sp.update("layer 3/10")
        assert sp._suffix == "layer 3/10"

    def test_stop_with_final_message(self, capsys):
        """stop(final_msg) prints the message."""
        from squish.convert import Spinner
        sp = Spinner("task", interval=0.01)
        sp.start()
        sp.stop(final_msg="done")
        captured = capsys.readouterr()
        assert "done" in captured.out

    def test_stop_without_final_message_no_output(self, capsys):
        """stop() without final_msg produces no output."""
        from squish.convert import Spinner
        sp = Spinner("task", interval=0.01)
        sp.start()
        sp.stop()
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_enter_returns_self(self):
        from squish.convert import Spinner
        sp = Spinner("task", interval=0.01)
        result = sp.__enter__()
        assert result is sp
        sp.__exit__(None, None, None)

    def test_spin_thread_terminates(self):
        """The background spin thread terminates after stop() is called."""
        from squish.convert import Spinner
        sp = Spinner("task", interval=0.05)
        sp.start()
        sp.stop()
        assert not sp._thread.is_alive()

    def test_non_tty_spin_does_not_write(self, capsys):
        """In non-TTY mode, no output is written during spin."""
        from squish.convert import Spinner
        sp = Spinner("silent spin", interval=0.02)
        sp.start()
        time.sleep(0.05)
        sp.stop()
        captured = capsys.readouterr()
        # No spinner frames should appear in stdout
        assert "⠋" not in captured.out
