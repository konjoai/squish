"""Repetition-loop detection for the decode loop (platform-neutral, pure-Python).

A degenerate small model loops in two distinct ways:

  1. short n-gram run-ons   ("animals:cats, animals:cats, ...")
  2. whole-sentence / paragraph repeats (the model re-emits a 150-300 char
     block verbatim over and over)

The detector below catches BOTH.  Wave 114 only scanned periods up to 80 chars
with an exact ``unit * reps == tail`` test, so a repeating paragraph (period
well past 80) sailed straight through.  We now scan up to a full paragraph and
compare per-character against the text one period earlier: once enough of the
loop has accumulated, ``text[i] == text[i - period]`` holds across the whole
span regardless of where the 20-token check happens to land, making detection
phase-robust (and tolerant of a single drifting char such as a counter).

Extracted from ``squish/server.py`` (Wave 127) so the stream, KV-cache and
fallback decode paths share one detector — and to keep server.py under its
line-count ceiling.  No MLX / platform dependencies: safe to import anywhere.
"""

from __future__ import annotations

_LOOP_WIN          = 1200  # trailing chars of generated text to inspect
_LOOP_CHECK_EVERY  = 20    # analyse every N emitted tokens
_LOOP_MIN_PERIOD   = 10    # smallest repeating unit (chars)
_LOOP_MAX_PERIOD   = 300   # largest repeating unit (chars) — covers a paragraph
_LOOP_MIN_REPS     = 4     # reps required for SHORT units (< _LOOP_BLOCK_PERIOD)
_LOOP_BLOCK_PERIOD = 120   # at/above this period a unit counts as a "block"
_LOOP_BLOCK_REPS   = 2     # reps required for block-sized units
_LOOP_MATCH_RATIO  = 0.95  # fraction of the span that must repeat verbatim


def _reps_for_period(period: int) -> int:
    """Reps required before *period* counts as a loop.

    Short fragments occur in legitimate prose and need many verbatim repetitions
    before we trust them; whole-sentence blocks practically never repeat by
    chance, so two is enough.
    """
    return _LOOP_BLOCK_REPS if period >= _LOOP_BLOCK_PERIOD else _LOOP_MIN_REPS


def _detect_loop(text: str) -> bool:
    """Return True when the tail of *text* is a verbatim repeating substring.

    For each candidate period the trailing ``period * reps`` characters are
    compared against the characters one period earlier; a loop is declared when
    at least ``_LOOP_MATCH_RATIO`` of them match.  The early-exit keeps the scan
    cheap on normal, non-looping text (it bails after a handful of mismatches).
    """
    n = len(text)
    for period in range(_LOOP_MIN_PERIOD, _LOOP_MAX_PERIOD + 1):
        span = period * _reps_for_period(period)
        if span > n:
            continue
        seg = text[-span:]
        allowed = (span - period) - int((span - period) * _LOOP_MATCH_RATIO)
        mism = 0
        for i in range(period, span):
            if seg[i] != seg[i - period]:
                mism += 1
                if mism > allowed:
                    break
        else:
            return True
    return False


class _LoopGuard:
    """Rolling-window repetition detector shared by every decode path.

    ``feed`` accumulates emitted text and, every ``_LOOP_CHECK_EVERY`` non-empty
    tokens, returns True when ``_detect_loop`` fires on the trailing window.
    Centralising this means the stream, KV-cache and fallback decode loops all
    get identical loop protection — Wave 114 guarded only the stream path, so
    KV-cache decodes ran degenerate loops all the way to max_tokens.
    """

    __slots__ = ("_buf", "_emitted")

    def __init__(self) -> None:
        self._buf = ""
        self._emitted = 0

    def feed(self, tok_text: str) -> bool:
        if not tok_text:
            return False
        self._emitted += 1
        self._buf += tok_text
        if len(self._buf) > _LOOP_WIN:
            self._buf = self._buf[-_LOOP_WIN:]
        if self._emitted % _LOOP_CHECK_EVERY != 0:
            return False
        return _detect_loop(self._buf)
