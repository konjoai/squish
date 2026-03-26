"""tests/test_wave114_rep_loop.py — Wave 114: repetition penalty + loop detection.

Pure-unit tests — no I/O, no process-state mutation, deterministic.

Covers:
- _detect_loop: known looping strings correctly detected
- _detect_loop: clean strings not falsely flagged
- _detect_loop: edge cases (short strings, single char repeats)
- _LOOP_* constants are present with sane values
- _generate_tokens signature accepts repetition_penalty kwarg
- server.py: repetition_penalty parsed from chat_completions request body
- server.py: repetition_penalty parsed from completions request body
"""
from __future__ import annotations

import importlib
import inspect
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import squish.server as _srv


# ==============================================================================
# 1.  _detect_loop — detection correctness
# ==============================================================================

class TestDetectLoopPositive(unittest.TestCase):
    """_detect_loop must return True for clearly looping strings."""

    def _hit(self, text: str) -> None:
        self.assertTrue(_srv._detect_loop(text), f"Expected loop in: {text!r}")

    def test_short_period_repeated_many_times(self):
        # 12-char period × 5 reps
        unit = "animals:cats,"
        self._hit(unit * 5)

    def test_space_padded_unit(self):
        unit = " animals:cats "
        self._hit(unit * 5)

    def test_word_boundary_period(self):
        unit = "duckduckgo."
        self._hit(unit * 5)

    def test_typical_model_runon(self):
        # Replicates the observed failure: ", animals:cats" repeated ~100+×
        unit = ", animals:cats"
        self._hit(unit * 6)

    def test_longer_period(self):
        # 40-char period × 4 reps — at boundary of _LOOP_MIN_REPS
        unit = "term:dog:cat:dog:dog:cat:dog:dogs:cats,h"
        self._hit(unit * 4)

    def test_exact_min_reps(self):
        # Exactly _LOOP_MIN_REPS repetitions of a _LOOP_MIN_PERIOD-char unit
        unit = "a" * _srv._LOOP_MIN_PERIOD
        self._hit(unit * _srv._LOOP_MIN_REPS)

    def test_loop_at_tail_with_clean_prefix(self):
        # Loop must be detectable even when preceded by normal text
        prefix = "Here is a function that uses DuckDuckGo to find a dog picture. "
        unit = "duckduckgo."  # 11 chars, above _LOOP_MIN_PERIOD
        self._hit(prefix + unit * 6)


class TestDetectLoopNegative(unittest.TestCase):
    """_detect_loop must return False for normal (non-looping) text."""

    def _miss(self, text: str) -> None:
        preview = repr(text)[:80]
        self.assertFalse(_srv._detect_loop(text), f"False positive on: {preview}")

    def test_normal_sentence(self):
        self._miss(
            "I can provide a simple code example demonstrating how to use "
            "squish_web_search to find a picture of a dog."
        )

    def test_unique_words(self):
        self._miss("The quick brown fox jumps over the lazy dog near the riverbank.")

    def test_string_too_short_for_detection(self):
        # Less than min_period * min_reps chars — nothing to detect
        self._miss("abc" * 2)

    def test_near_miss_three_reps(self):
        # _LOOP_MIN_REPS - 1 repetitions should NOT trigger
        unit = "animals:cats,"
        text = unit * (_srv._LOOP_MIN_REPS - 1)
        self._miss(text)

    def test_python_code_snippet(self):
        self._miss(
            "def web_search(query):\n"
            "    import requests\n"
            "    url = f'https://duckduckgo.com/?q={query}'\n"
            "    return requests.get(url).text\n"
        )


# ==============================================================================
# 2.  _LOOP_* constants sanity
# ==============================================================================

class TestLoopConstants(unittest.TestCase):

    def test_loop_win_positive(self):
        self.assertGreater(_srv._LOOP_WIN, 0)

    def test_loop_min_period_lt_max(self):
        self.assertLess(_srv._LOOP_MIN_PERIOD, _srv._LOOP_MAX_PERIOD)

    def test_loop_min_reps_at_least_3(self):
        self.assertGreaterEqual(_srv._LOOP_MIN_REPS, 3)

    def test_loop_check_every_positive(self):
        self.assertGreater(_srv._LOOP_CHECK_EVERY, 0)

    def test_loop_window_covers_min_detection(self):
        # _LOOP_WIN must be large enough to hold _LOOP_MAX_PERIOD * _LOOP_MIN_REPS chars
        self.assertGreaterEqual(
            _srv._LOOP_WIN,
            _srv._LOOP_MAX_PERIOD * _srv._LOOP_MIN_REPS,
        )


# ==============================================================================
# 3.  _generate_tokens signature
# ==============================================================================

class TestGenerateTokensSignature(unittest.TestCase):

    def test_repetition_penalty_param_exists(self):
        sig = inspect.signature(_srv._generate_tokens)
        self.assertIn("repetition_penalty", sig.parameters)

    def test_repetition_penalty_default_is_1(self):
        sig = inspect.signature(_srv._generate_tokens)
        default = sig.parameters["repetition_penalty"].default
        self.assertEqual(default, 1.0)

    def test_existing_params_unchanged(self):
        sig = inspect.signature(_srv._generate_tokens)
        for name in ("prompt", "max_tokens", "temperature", "top_p", "stop", "seed"):
            self.assertIn(name, sig.parameters, f"missing param: {name}")


# ==============================================================================
# 4.  API body parsing — repetition_penalty extracted correctly
# ==============================================================================

class TestChatCompletionsBodyParsing(unittest.TestCase):
    """Verify the chat_completions handler reads repetition_penalty from body."""

    def test_repetition_penalty_parsed(self):
        """Source of chat_completions must contain repetition_penalty body.get."""
        src = inspect.getsource(_srv.chat_completions)
        self.assertIn('body.get("repetition_penalty"', src)

    def test_repetition_penalty_default_1_in_chat(self):
        src = inspect.getsource(_srv.chat_completions)
        # Default must be 1.0 (no penalty)
        self.assertIn('"repetition_penalty", 1.0', src)


class TestCompletionsBodyParsing(unittest.TestCase):
    """Verify the completions handler reads repetition_penalty from body."""

    def test_repetition_penalty_parsed(self):
        src = inspect.getsource(_srv.completions)
        self.assertIn('body.get("repetition_penalty"', src)

    def test_repetition_penalty_default_1_in_completions(self):
        src = inspect.getsource(_srv.completions)
        self.assertIn('"repetition_penalty", 1.0', src)


if __name__ == "__main__":
    unittest.main()
