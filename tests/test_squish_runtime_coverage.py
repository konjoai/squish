"""Behavioral coverage for the generate_stream / _sample / header edge paths of
``squish.runtime.squish_runtime`` left untested by the baseline suite.
Pure-Python numpy; no MLX.
"""
from __future__ import annotations

import numpy as np
import pytest

from squish.runtime.squish_runtime import SquishRuntime, SquizdFlags


def _rt(layer_count=4, vocab_size=128):
    return SquishRuntime.from_flags(SquizdFlags.NONE, layer_count=layer_count, vocab_size=vocab_size)


# ── generate_stream ──────────────────────────────────────────────────────────


def test_generate_stream_zero_tokens_yields_nothing():
    rt = _rt()
    out = list(rt.generate_stream("hello", max_new_tokens=0, seed=0))
    assert out == []  # range(0) → loop never runs (338→exit)


def test_generate_stream_hits_length_budget():
    rt = _rt()
    out = list(rt.generate_stream("hello", max_new_tokens=3, seed=1))
    assert len(out) == 3
    assert out[-1][1] == "length"  # final token tagged length
    assert all(fin is None for _, fin in out[:-1])


def test_generate_stream_stops_on_eos(monkeypatch):
    rt = _rt()
    # Force the sampler to emit the EOS id (2) so the stop path runs (344-345).
    monkeypatch.setattr(SquishRuntime, "_sample", staticmethod(lambda *a, **k: 2))
    out = list(rt.generate_stream("hello", max_new_tokens=10, seed=0))
    assert len(out) == 1 and out[0][1] == "stop"


def test_generate_non_stream():
    rt = _rt()
    text = rt.generate("hello", max_new_tokens=4, seed=2)
    assert isinstance(text, str) and text


# ── _sample ──────────────────────────────────────────────────────────────────


def test_sample_greedy_when_temp_zero():
    logits = np.array([[0.1, 5.0, 0.2]], dtype=np.float32)  # (1, vocab)
    rng = np.random.default_rng(0)
    assert SquishRuntime._sample(logits, rng=rng, temperature=0.0) == 1  # argmax


def test_sample_temperature_path():
    # A near one-hot distribution → softmax sampling almost surely picks index 1.
    logits = np.array([[0.0, 50.0, 0.0, 0.0]], dtype=np.float32)
    rng = np.random.default_rng(0)
    tok = SquishRuntime._sample(logits, rng=rng, temperature=0.8)  # 420 (softmax path)
    assert tok == 1


# ── header / from_file ───────────────────────────────────────────────────────


def test_from_file_roundtrip(tmp_path):
    rt = _rt(layer_count=8)
    path = tmp_path / "model.squizd"
    path.write_bytes(rt.header.raw_bytes)
    loaded = SquishRuntime.from_file(path)
    assert loaded.layer_count == 8


def test_from_file_missing_uses_null_header(tmp_path):
    loaded = SquishRuntime.from_file(tmp_path / "absent.squizd")
    assert loaded.layer_count == 0  # null header


def test_from_file_truncated_uses_null_header(tmp_path):
    path = tmp_path / "short.squizd"
    path.write_bytes(b"SQZ")  # < header size → null header
    loaded = SquishRuntime.from_file(path)
    assert loaded.layer_count == 0
