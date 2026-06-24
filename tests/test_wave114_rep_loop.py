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


# ==============================================================================
# 5.  Paragraph / block-level loops (regression: KV-store run-on bug)
# ==============================================================================

# The exact shape a 1B/1.5B model emitted when it ran off the rails: a whole
# ~220-char paragraph repeated verbatim. The original 80-char period cap could
# never see this, so the loop ran to max_tokens.
_KV_PARAGRAPH = (
    "In the K-V store, the simplicity part is the flat file store, which is used "
    "to store the past interactions. The efficiency part is the hash table, which "
    "is used to store the necessary information for the model's response.\n\n"
)


class TestDetectLoopParagraph(unittest.TestCase):
    """Block-sized verbatim repeats must be detected (period well past 80)."""

    def test_paragraph_period_exceeds_old_cap(self):
        # Guards the assumption behind this whole fix.
        self.assertGreater(len(_KV_PARAGRAPH), 80)

    def test_paragraph_repeated_is_a_loop(self):
        self.assertTrue(_srv._detect_loop(_KV_PARAGRAPH * 5))

    def test_paragraph_detection_is_phase_robust(self):
        # Check may land mid-paragraph — detection must not depend on the tail
        # ending exactly on a repeat boundary.
        text = (_KV_PARAGRAPH * 5)[:-37]
        self.assertTrue(_srv._detect_loop(text))

    def test_single_paragraph_is_not_a_loop(self):
        self.assertFalse(_srv._detect_loop(_KV_PARAGRAPH))

    def test_block_reps_lower_than_short_reps(self):
        # Blocks need fewer reps than short fragments to be trusted.
        self.assertLess(_srv._LOOP_BLOCK_REPS, _srv._LOOP_MIN_REPS)

    def test_reps_for_period_scales(self):
        self.assertEqual(_srv._reps_for_period(_srv._LOOP_MIN_PERIOD), _srv._LOOP_MIN_REPS)
        self.assertEqual(_srv._reps_for_period(_srv._LOOP_BLOCK_PERIOD), _srv._LOOP_BLOCK_REPS)


# ==============================================================================
# 6.  _LoopGuard — the shared rolling-window detector
# ==============================================================================

class TestLoopGuard(unittest.TestCase):
    """The guard used by every decode path (stream, KV-cache, fallback)."""

    def _feed(self, guard, text, chunk=5):
        for i in range(0, len(text), chunk):
            if guard.feed(text[i:i + chunk]):
                return i
        return None

    def test_fires_on_repeating_block(self):
        guard = _srv._LoopGuard()
        fired = self._feed(guard, _KV_PARAGRAPH * 6)
        self.assertIsNotNone(fired, "guard never fired on a clear paragraph loop")

    def test_fires_on_short_runon(self):
        guard = _srv._LoopGuard()
        fired = self._feed(guard, ", animals:cats" * 40, chunk=3)
        self.assertIsNotNone(fired)

    def test_quiet_on_unique_prose(self):
        guard = _srv._LoopGuard()
        prose = (
            "The river wound slowly through the valley as the sun dipped below the "
            "distant hills, casting amber shadows across the quiet meadow. A heron "
            "lifted from the reeds, wings creaking, and traced a slow arc toward the "
            "far bank where willows leaned over the current. Somewhere a dog barked "
            "twice, then thought better of it, and the evening settled into the kind "
            "of stillness that makes a person remember names they had tried to forget."
        )
        self.assertIsNone(self._feed(guard, prose))

    def test_empty_tokens_do_not_advance_cadence(self):
        # Empty/whitespace-stripped tokens must not count toward the check cadence
        # nor toward the buffer.
        guard = _srv._LoopGuard()
        for _ in range(100):
            self.assertFalse(guard.feed(""))

    def test_check_cadence_respected(self):
        # The guard only inspects every _LOOP_CHECK_EVERY non-empty tokens, so a
        # loop fed one char at a time is detected at a multiple of the cadence.
        guard = _srv._LoopGuard()
        fired_count = None
        n = 0
        for ch in _KV_PARAGRAPH * 6:
            n += 1
            if guard.feed(ch):
                fired_count = n
                break
        self.assertIsNotNone(fired_count)
        self.assertEqual(fired_count % _srv._LOOP_CHECK_EVERY, 0)


if __name__ == "__main__":
    unittest.main()
