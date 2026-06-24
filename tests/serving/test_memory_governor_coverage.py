"""Remaining-branch coverage for memory_governor: vm_stat parse edge cases and
the daemon thread lifecycle. Pure-Python; ``_run`` (the subprocess wrapper) is
patched, so no real ``vm_stat`` is invoked."""

from unittest.mock import patch

from squish.serving.memory_governor import MemoryGovernor, _read_vm_stat


def test_read_vm_stat_handles_bad_pagesize_nonint_and_missing_keywords():
    # Non-numeric page size → ValueError swallowed (page_size stays 4096);
    # non-integer "Pages free" value → that page count parses to 0;
    # omitted keywords (speculative/purgeable/wired) → their _pages() return 0.
    raw = (
        "Mach Virtual Memory Statistics: (page size of abc bytes)\n"
        "Pages free: xyz.\n"
        "Pages inactive: 50.\n"
    )
    with patch("squish.serving.memory_governor._run", return_value=raw):
        available_gb, wired_gb = _read_vm_stat()
    assert available_gb >= 0.0
    assert wired_gb == 0.0  # no "Pages wired down" line


def test_stop_without_start_is_a_noop():
    gov = MemoryGovernor()
    gov.stop()  # _thread is None → the join branch is skipped
    assert gov._thread is None


def test_loop_runs_one_poll_then_exits(monkeypatch):
    gov = MemoryGovernor()
    gov._interval = 0.01  # keep the wait() short
    # First iteration polls; the poll sets the stop event so the loop then exits.
    monkeypatch.setattr(gov, "_poll_once", gov._stop_event.set)
    gov._loop()
    assert gov._stop_event.is_set()
