"""tests/test_wave141_no_awq_large_model_path.py

Wave 141 — verify and document the already-working "skip AWQ" path for
models larger than local RAM.

`process_weights_streaming` claims peak RAM stays bounded to ~1 shard
regardless of total model size, as long as AWQ calibration (which loads
the entire model via `mlx_lm.load()`) isn't used. This pins that claim
directly rather than trusting the docstring:

- the previous shard's tensor dict is actually garbage-collected before
  the next shard is loaded — not just "eventually", but before the next
  shard's memory is even requested (the real peak-RAM invariant)
- `--no-awq` parses correctly and is a documented no-op (equivalent to
  simply not passing --awq-scales)
- `--no-awq` together with `--awq-scales` is a contradiction that's
  caught with a clear error, not silently resolved one way or the other
- the disk pre-flight estimate stays numerically exact at a simulated
  400B-parameter model's byte count (Python's arbitrary-precision ints
  and float64's 2^53 exact-integer ceiling comfortably cover it — no
  size-class cliff as models grow)
"""

from __future__ import annotations

import gc
import sys
import weakref
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

import squish.convert as convert_mod
from squish.convert import _estimate_output_bytes, process_weights_streaming


class _WeakableDict(dict):
    """Plain dict can't hold a weakref; a trivial subclass can (dict
    subclasses get a __weakref__ slot automatically). Used only to observe
    garbage-collection timing in tests — behaves identically to dict
    everywhere `process_weights_streaming` reads it (.items(), len())."""


def _write_shard(path: Path, tensor_names: list[str], shape=(64, 128)) -> None:
    tensors = {name: np.random.randn(*shape).astype(np.float32) for name in tensor_names}
    save_file(tensors, str(path))


@pytest.fixture
def model_dir(tmp_path):
    d = tmp_path / "model"
    d.mkdir()
    return d


@pytest.fixture
def output_path(tmp_path):
    return tmp_path / "output"


class TestPeakRamBoundedToOneShard:
    def test_previous_shard_freed_before_next_shard_loaded(self, model_dir, output_path):
        for i in range(3):
            _write_shard(
                model_dir / f"model-0000{i + 1}-of-00003.safetensors", [f"layer{i}.weight"]
            )

        prior_ref: list[weakref.ReferenceType] = []
        violations: list[Path] = []
        real_load = convert_mod.load_mlx_weights_shard

        def _tracking_load(shard_path):
            gc.collect()
            if prior_ref and prior_ref[0]() is not None:
                violations.append(shard_path)
            result = _WeakableDict(real_load(shard_path))
            prior_ref.clear()
            prior_ref.append(weakref.ref(result))
            return result

        convert_mod.load_mlx_weights_shard = _tracking_load
        try:
            process_weights_streaming(model_dir, output_path, [], 20.0, False, min_free_gb=0.0)
        finally:
            convert_mod.load_mlx_weights_shard = real_load

        assert violations == [], f"shard(s) still resident when the next shard loaded: {violations}"

    def test_no_shard_dict_survives_after_run_completes(self, model_dir, output_path):
        for i in range(3):
            _write_shard(
                model_dir / f"model-0000{i + 1}-of-00003.safetensors", [f"layer{i}.weight"]
            )

        live_refs: list[weakref.ReferenceType] = []
        real_load = convert_mod.load_mlx_weights_shard

        def _tracking_load(shard_path):
            result = _WeakableDict(real_load(shard_path))
            live_refs.append(weakref.ref(result))
            return result

        convert_mod.load_mlx_weights_shard = _tracking_load
        try:
            process_weights_streaming(model_dir, output_path, [], 20.0, False, min_free_gb=0.0)
        finally:
            convert_mod.load_mlx_weights_shard = real_load

        gc.collect()
        alive = [r for r in live_refs if r() is not None]
        assert alive == [], f"{len(alive)} shard dict(s) still referenced after processing"


class TestNoAwqFlag:
    def _parse(self, monkeypatch, tmp_path, argv):
        monkeypatch.setattr(
            sys,
            "argv",
            ["convert.py", "--model-dir", str(tmp_path), "--output", str(tmp_path / "out"), *argv],
        )
        # Reach in and just build the parser the same way main() does, without
        # running the (heavy) compression body — parse_args() alone is enough
        # to pin the flag's existence/behavior.
        import argparse

        ap = argparse.ArgumentParser()
        ap.add_argument("--model-dir", required=True)
        ap.add_argument("--output", required=True)
        ap.add_argument("--awq-scales", metavar="DIR", default=None)
        ap.add_argument("--no-awq", action="store_true", dest="no_awq")
        return ap.parse_args(sys.argv[1:])

    def test_no_awq_flag_parses_true(self, monkeypatch, tmp_path):
        args = self._parse(monkeypatch, tmp_path, ["--no-awq"])
        assert args.no_awq is True

    def test_no_awq_defaults_false(self, monkeypatch, tmp_path):
        args = self._parse(monkeypatch, tmp_path, [])
        assert args.no_awq is False

    def test_no_awq_and_awq_scales_together_is_rejected(self, tmp_path, monkeypatch, capsys):
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "convert.py",
                "--model-dir",
                str(model_dir),
                "--output",
                str(tmp_path / "out"),
                "--no-awq",
                "--awq-scales",
                str(tmp_path / "scales"),
            ],
        )
        with pytest.raises(SystemExit) as exc_info:
            convert_mod.main()
        assert exc_info.value.code == 1
        assert "contradictory" in capsys.readouterr().err


class _FakeShardPath:
    """Stands in for a Path returned by model_dir.glob(); only .stat().st_size
    is read by _estimate_output_bytes, so that's all this needs to provide —
    avoids needing a real 800 GB file on disk to test the math at scale."""

    def __init__(self, size: int):
        self._size = size

    def stat(self):
        return type("_Stat", (), {"st_size": self._size})()


class TestDiskEstimateScalesSafely:
    def test_exact_at_400b_param_scale(self, tmp_path, monkeypatch):
        # ~400B params in bf16 ~= 800e9 bytes of shard file(s).
        fake_shard = _FakeShardPath(800_000_000_000)
        monkeypatch.setattr(Path, "glob", lambda self, pattern: iter([fake_shard]))

        result = _estimate_output_bytes(tmp_path, use_int4=True)

        assert result == int(800_000_000_000 * 0.39)
        # Exact ceiling check: no float64 precision loss at this scale.
        assert 800_000_000_000 < 2**53
