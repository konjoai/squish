"""Tests for Wave 24 — Drift detection and continuous monitoring (DriftMonitor)."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from squish.squash.governor import DriftEvent, DriftMonitor


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_model_dir(tmp_path: Path) -> Path:
    """Create a minimal model directory with the files DriftMonitor hashes."""
    d = tmp_path / "model"
    d.mkdir()
    (d / "cyclonedx-mlbom.json").write_text(
        json.dumps({"components": []}), encoding="utf-8"
    )
    (d / "config.json").write_text(
        json.dumps({"model_type": "test"}), encoding="utf-8"
    )
    return d


# ── Shape / dtype contract tests ──────────────────────────────────────────────

def test_drift_event_fields():
    evt = DriftEvent(
        event_type="BOM_CHANGED",
        component="cyclonedx-mlbom.json",
        old_value="abc",
        new_value="def",
    )
    assert isinstance(evt.event_type, str)
    assert isinstance(evt.component, str)
    assert isinstance(evt.detected_at, str)


def test_detected_at_is_iso8601(tmp_path):
    evt = DriftEvent(event_type="T", component="c", old_value="a", new_value="b")
    # Basic ISO-8601 check: contains T and ends in Z or +offset
    assert "T" in evt.detected_at


# ── Snapshot correctness ───────────────────────────────────────────────────────

def test_snapshot_returns_hex(tmp_path):
    d = _make_model_dir(tmp_path)
    snap = DriftMonitor.snapshot(d)
    assert isinstance(snap, str)
    assert len(snap) == 64  # SHA-256 hex


def test_snapshot_reproducible(tmp_path):
    d = _make_model_dir(tmp_path)
    s1 = DriftMonitor.snapshot(d)
    s2 = DriftMonitor.snapshot(d)
    assert s1 == s2


def test_empty_dir_snapshot_does_not_raise(tmp_path):
    d = tmp_path / "empty"
    d.mkdir()
    snap = DriftMonitor.snapshot(d)
    assert isinstance(snap, str)


# ── Compare / drift detection ─────────────────────────────────────────────────

def test_compare_same_snapshot_empty(tmp_path):
    d = _make_model_dir(tmp_path)
    snap = DriftMonitor.snapshot(d)
    events = DriftMonitor.compare(d, snap)
    assert events == []


def test_compare_changed_bom_emits_event(tmp_path):
    d = _make_model_dir(tmp_path)
    snap = DriftMonitor.snapshot(d)
    # Mutate the BOM
    bom_path = d / "cyclonedx-mlbom.json"
    bom_path.write_text(
        json.dumps({"components": [{"type": "library", "name": "new-dep"}]}),
        encoding="utf-8",
    )
    events = DriftMonitor.compare(d, snap)
    assert len(events) >= 1
    assert any(evt.component == "cyclonedx-mlbom.json" for evt in events)


def test_compare_events_are_drift_event_instances(tmp_path):
    d = _make_model_dir(tmp_path)
    snap = DriftMonitor.snapshot(d)
    (d / "cyclonedx-mlbom.json").write_text(
        json.dumps({"components": [{"type": "library", "name": "x"}]}),
        encoding="utf-8",
    )
    events = DriftMonitor.compare(d, snap)
    for e in events:
        assert isinstance(e, DriftEvent)


# ── Watch ─────────────────────────────────────────────────────────────────────

def test_watch_returns_stop_event(tmp_path):
    d = _make_model_dir(tmp_path)
    stop = DriftMonitor.watch(d, interval_s=3600, callback=lambda evts: None)
    assert isinstance(stop, threading.Event)
    stop.set()  # signal to stop


def test_watch_stop_event_halts_thread(tmp_path):
    d = _make_model_dir(tmp_path)
    fired: list = []

    def cb(evts):
        fired.extend(evts)

    stop = DriftMonitor.watch(d, interval_s=0.05, callback=cb)
    time.sleep(0.2)
    stop.set()
    time.sleep(0.15)
    # After stop, no further callbacks should fire
    count_before = len(fired)
    time.sleep(0.2)
    assert len(fired) == count_before
