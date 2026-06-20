"""Coverage for squish.cli cmd_benchmark — KV-cache compression benchmark.
Uses the real numpy QuantizedKVCache with tiny synthetic streams; baseline
save/compare driven via tmp files. Host-agnostic.
"""

from __future__ import annotations

import argparse
import json

import pytest

from squish import cli


def _ns(**kw):
    kw.setdefault("ctx", 8)
    kw.setdefault("head_dim", 8)
    kw.setdefault("n_heads", 2)
    kw.setdefault("seed", 0)
    kw.setdefault("threshold", 0.05)
    kw.setdefault("modes", "int8")
    kw.setdefault("save", None)
    kw.setdefault("compare", None)
    return argparse.Namespace(**kw)


def test_benchmark_runs_all_modes(capsys):
    cli.cmd_benchmark(_ns(modes="int8,int4,int2"))
    out = capsys.readouterr().out
    assert "squish benchmark" in out and "SNR (dB)" in out


def test_benchmark_skip_int2_non_divisible(capsys):
    cli.cmd_benchmark(_ns(modes="int2", head_dim=6))  # 6 % 4 != 0 → skip
    assert "skip int2" in capsys.readouterr().out


def test_benchmark_skip_int4_odd_head_dim(capsys):
    cli.cmd_benchmark(_ns(modes="int4", head_dim=3))  # 3 % 2 != 0 → skip
    assert "skip int4" in capsys.readouterr().out


def test_benchmark_save_baseline(tmp_path, capsys):
    out_file = tmp_path / "baseline.json"
    cli.cmd_benchmark(_ns(modes="int8", save=str(out_file)))
    assert out_file.exists()
    payload = json.loads(out_file.read_text())
    assert payload["version"] == 1 and "int8" in payload["results"]
    assert "Baseline saved" in capsys.readouterr().out


def test_benchmark_compare_missing_baseline(tmp_path):
    with pytest.raises(SystemExit) as exc:
        cli.cmd_benchmark(_ns(modes="int8", compare=str(tmp_path / "nope.json")))
    assert exc.value.code == 2


def test_benchmark_compare_no_regression(tmp_path, capsys):
    base = tmp_path / "b.json"
    cli.cmd_benchmark(_ns(modes="int8", save=str(base)))
    capsys.readouterr()
    # Compare same params against the just-saved baseline → deltas ~0 → pass.
    cli.cmd_benchmark(_ns(modes="int8", compare=str(base)))
    assert "within" in capsys.readouterr().out


def test_benchmark_compare_regression_exits_1(tmp_path, capsys):
    base = tmp_path / "b.json"
    base.write_text(
        json.dumps(
            {"version": 1, "results": {"int8": {"snr_db": 999.0, "compression_ratio": 999.0}}}
        )
    )
    with pytest.raises(SystemExit) as exc:
        cli.cmd_benchmark(_ns(modes="int8", compare=str(base)))
    assert exc.value.code == 1
    assert "REGRESSION" in capsys.readouterr().out


def test_benchmark_compare_mode_not_in_baseline(tmp_path, capsys):
    base = tmp_path / "b.json"
    base.write_text(json.dumps({"version": 1, "results": {}}))
    cli.cmd_benchmark(_ns(modes="int8", compare=str(base)))
    assert "not in baseline" in capsys.readouterr().out


def test_benchmark_compare_zero_baseline_metric_skipped(tmp_path, capsys):
    base = tmp_path / "b.json"
    base.write_text(
        json.dumps({"version": 1, "results": {"int8": {"snr_db": 0.0, "compression_ratio": 0.0}}})
    )
    cli.cmd_benchmark(_ns(modes="int8", compare=str(base)))
    assert "within" in capsys.readouterr().out  # both metrics skipped (base 0)
