"""Behavioral coverage for ``squish.platform.ane_router`` — ANE/GPU routing
policy, availability, caps caching, and chip-generation detection.

A fake chip detector is injected and ``sys.platform`` is monkeypatched, so the
tests are host-agnostic (no real Apple hardware / CoreML). Pure-Python.
"""
from __future__ import annotations

import json
import types

import pytest

from squish.platform import ane_router as ar
from squish.platform.ane_router import ANERouter


def _detector(gen=3, raises=False):
    class _D:
        def detect(self):
            if raises:
                raise RuntimeError("detect boom")
            return types.SimpleNamespace(generation=gen)

    return _D()


def _router(tmp_path, gen=3, monkeypatch=None, platform="darwin"):
    if monkeypatch is not None:
        monkeypatch.setattr(ar.sys, "platform", platform)
    return ANERouter(_detector_override=_detector(gen=gen),
                     _caps_path=tmp_path / "caps.json")


# ── routing policy branches ─────────────────────────────────────────────────


def test_route_non_macos_uses_gpu(tmp_path, monkeypatch):
    r = _router(tmp_path, gen=3, monkeypatch=monkeypatch, platform="linux")
    pol = r.get_policy(1_000_000_000)
    assert pol.preferred_backend == "gpu" and not pol.enabled
    assert "non-macOS" in pol.reason


def test_route_env_disabled(tmp_path, monkeypatch):
    monkeypatch.setenv("SQUISH_ANE_ENABLED", "0")
    r = _router(tmp_path, gen=3, monkeypatch=monkeypatch)
    pol = r.get_policy(1_000_000_000)
    assert pol.preferred_backend == "gpu" and "disabled by environment" in pol.reason


def test_route_unknown_chip(tmp_path, monkeypatch):
    monkeypatch.delenv("SQUISH_ANE_ENABLED", raising=False)
    r = _router(tmp_path, gen=0, monkeypatch=monkeypatch)
    pol = r.get_policy(1_000_000_000)
    assert pol.preferred_backend == "gpu" and "unknown chip" in pol.reason


def test_route_model_too_large(tmp_path, monkeypatch):
    monkeypatch.delenv("SQUISH_ANE_ENABLED", raising=False)
    r = _router(tmp_path, gen=3, monkeypatch=monkeypatch)
    pol = r.get_policy(13_000_000_000)  # > 8B → GPU
    assert pol.preferred_backend == "gpu" and "8B" in pol.reason


def test_route_env_forced(tmp_path, monkeypatch):
    monkeypatch.setenv("SQUISH_ANE_ENABLED", "1")
    r = _router(tmp_path, gen=3, monkeypatch=monkeypatch)
    pol = r.get_policy(3_000_000_000)
    assert pol.preferred_backend == "ane" and pol.enabled and "forced" in pol.reason


def test_route_default_ane(tmp_path, monkeypatch):
    monkeypatch.delenv("SQUISH_ANE_ENABLED", raising=False)
    r = _router(tmp_path, gen=3, monkeypatch=monkeypatch)
    assert r.route(3_800_000_000) == "ane"
    pol = r.get_policy(3_800_000_000)
    assert pol.enabled and "ANE" in pol.reason and pol.ane_memory_budget_gb == 4.0


# ── is_ane_available ────────────────────────────────────────────────────────


def test_is_ane_available_true(tmp_path, monkeypatch):
    monkeypatch.delenv("SQUISH_ANE_ENABLED", raising=False)
    assert _router(tmp_path, gen=3, monkeypatch=monkeypatch).is_ane_available() is True


def test_is_ane_available_false_non_macos(tmp_path, monkeypatch):
    assert _router(tmp_path, gen=3, monkeypatch=monkeypatch, platform="linux").is_ane_available() is False


def test_is_ane_available_false_env_and_unknown(tmp_path, monkeypatch):
    monkeypatch.setenv("SQUISH_ANE_ENABLED", "0")
    assert _router(tmp_path, gen=3, monkeypatch=monkeypatch).is_ane_available() is False
    monkeypatch.delenv("SQUISH_ANE_ENABLED", raising=False)
    assert _router(tmp_path, gen=0, monkeypatch=monkeypatch).is_ane_available() is False


# ── caps caching ────────────────────────────────────────────────────────────


def test_cache_and_load_caps_roundtrip(tmp_path, monkeypatch):
    r = _router(tmp_path, gen=3, monkeypatch=monkeypatch)
    r.cache_caps()
    loaded = r.load_caps()
    assert loaded["chip_generation"] == 3 and loaded["ane_budget_gb"] == 4.0


def test_cache_caps_explicit_path(tmp_path, monkeypatch):
    r = _router(tmp_path, gen=3, monkeypatch=monkeypatch)
    target = tmp_path / "sub" / "c.json"
    r.cache_caps(path=target)
    assert json.loads(target.read_text())["chip_generation"] == 3


def test_load_caps_missing_returns_none(tmp_path, monkeypatch):
    assert _router(tmp_path, gen=3, monkeypatch=monkeypatch).load_caps(path=tmp_path / "nope.json") is None


def test_load_caps_corrupt_returns_none(tmp_path, monkeypatch):
    r = _router(tmp_path, gen=3, monkeypatch=monkeypatch)
    bad = tmp_path / "bad.json"
    bad.write_text("{ not json")
    assert r.load_caps(path=bad) is None


# ── chip-generation detection ───────────────────────────────────────────────


def test_detect_chip_generation_detect_failure_returns_zero(tmp_path, monkeypatch):
    monkeypatch.setattr(ar.sys, "platform", "darwin")
    r = ANERouter(_detector_override=_detector(raises=True), _caps_path=tmp_path / "c.json")
    assert r._chip_generation == 0  # detect() raised → 0 (232-234)


def test_detect_chip_generation_none_detector(tmp_path, monkeypatch):
    # No chip detector available → _detector is None → generation 0 (221-222).
    monkeypatch.setattr(ar, "_CHIP_DETECTOR_AVAILABLE", False)
    r = ANERouter(_caps_path=tmp_path / "c.json")
    assert r._chip_generation == 0


def test_detect_chip_generation_non_macos_without_override(tmp_path, monkeypatch):
    # Real-detector path on non-macOS without an override → guarded to 0 (225-226).
    monkeypatch.setattr(ar.sys, "platform", "linux")
    monkeypatch.setattr(ar, "_CHIP_DETECTOR_AVAILABLE", True)
    monkeypatch.setattr(ar, "ChipDetector", lambda: _detector(gen=3))
    r = ANERouter(_caps_path=tmp_path / "c.json")
    assert r._chip_generation == 0


# ── singleton ───────────────────────────────────────────────────────────────


def test_singleton_get_and_reset():
    ar.reset_ane_router()
    first = ar.get_ane_router()
    assert ar.get_ane_router() is first  # reused
    ar.reset_ane_router()
    assert ar.get_ane_router() is not first  # fresh after reset
