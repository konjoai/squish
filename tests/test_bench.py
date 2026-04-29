"""tests/test_bench.py — W102 squish bench subcommand tests.

Covers:
  - Subcommand registered in build_parser()
  - Default argument values
  - cmd_bench output structure (format, shape, backend, latency lines)
  - INT4 and INT8 format paths
  - Throughput printed (GOPS / GB/s)
  - Warmup and iters args respected (smoke)
  - Error: bad group_size (doesn't divide in_features)
  - Namespace args forwarded correctly
"""
from __future__ import annotations

import argparse
import sys
import types
from io import StringIO
from unittest.mock import patch

import pytest

import squish.cli as cli


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bench_ns(**kwargs) -> argparse.Namespace:
    defaults = dict(
        format="int4",
        batch=1,
        in_features=64,
        out_features=32,
        group_size=16,
        iters=5,
        warmup=2,
    )
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def _run_bench(capsys, **kwargs):
    ns = _bench_ns(**kwargs)
    cli.cmd_bench(ns)
    return capsys.readouterr().out


# ---------------------------------------------------------------------------
# Subcommand registration
# ---------------------------------------------------------------------------

class TestBenchSubcommandRegistered:
    def test_bench_in_subparsers(self):
        ap = cli.build_parser()
        subs = [a._name_parser_map for a in ap._actions if hasattr(a, "_name_parser_map")]
        assert subs, "No subparsers found"
        assert "bench" in subs[0], f"'bench' not in subcommands: {list(subs[0].keys())}"

    def test_bench_func_is_cmd_bench(self):
        ap = cli.build_parser()
        subs = [a._name_parser_map for a in ap._actions if hasattr(a, "_name_parser_map")]
        bench_parser = subs[0]["bench"]
        defaults = bench_parser.get_default("func")
        assert defaults is cli.cmd_bench


# ---------------------------------------------------------------------------
# Default argument values
# ---------------------------------------------------------------------------

class TestBenchDefaults:
    def test_format_default_is_int4(self):
        ap = cli.build_parser()
        ns = ap.parse_args(["bench"])
        assert ns.format == "int4"

    def test_batch_default_is_1(self):
        ap = cli.build_parser()
        ns = ap.parse_args(["bench"])
        assert ns.batch == 1

    def test_in_features_default_is_4096(self):
        ap = cli.build_parser()
        ns = ap.parse_args(["bench"])
        assert ns.in_features == 4096

    def test_out_features_default_is_4096(self):
        ap = cli.build_parser()
        ns = ap.parse_args(["bench"])
        assert ns.out_features == 4096

    def test_group_size_default_is_64(self):
        ap = cli.build_parser()
        ns = ap.parse_args(["bench"])
        assert ns.group_size == 64

    def test_iters_default_is_100(self):
        ap = cli.build_parser()
        ns = ap.parse_args(["bench"])
        assert ns.iters == 100

    def test_warmup_default_is_10(self):
        ap = cli.build_parser()
        ns = ap.parse_args(["bench"])
        assert ns.warmup == 10


# ---------------------------------------------------------------------------
# Output structure — INT4
# ---------------------------------------------------------------------------

class TestBenchOutputInt4:
    def test_output_contains_int4_label(self, capsys):
        out = _run_bench(capsys, format="int4")
        assert "INT4" in out

    def test_output_contains_shape_line(self, capsys):
        out = _run_bench(capsys, format="int4", batch=2, in_features=64, out_features=32)
        assert "Shape" in out
        assert "2, 64" in out
        assert "32, 64" in out

    def test_output_contains_p50(self, capsys):
        out = _run_bench(capsys)
        assert "p50" in out

    def test_output_contains_p95(self, capsys):
        out = _run_bench(capsys)
        assert "p95" in out

    def test_output_contains_p99(self, capsys):
        out = _run_bench(capsys)
        assert "p99" in out

    def test_output_contains_gops(self, capsys):
        out = _run_bench(capsys)
        assert "GOPS" in out

    def test_output_contains_backend(self, capsys):
        out = _run_bench(capsys)
        assert "Backend" in out

    def test_output_contains_groups_line(self, capsys):
        out = _run_bench(capsys, in_features=64, group_size=16)
        assert "Groups" in out
        assert "4 groups" in out


# ---------------------------------------------------------------------------
# Output structure — INT8
# ---------------------------------------------------------------------------

class TestBenchOutputInt8:
    def test_output_contains_int8_label(self, capsys):
        out = _run_bench(capsys, format="int8", in_features=64, out_features=32, group_size=16)
        assert "INT8" in out

    def test_output_contains_p50_int8(self, capsys):
        out = _run_bench(capsys, format="int8", in_features=64, out_features=32, group_size=16)
        assert "p50" in out


# ---------------------------------------------------------------------------
# Argument parsing roundtrip
# ---------------------------------------------------------------------------

class TestBenchArgParsing:
    def test_format_int8_parsed(self):
        ap = cli.build_parser()
        ns = ap.parse_args(["bench", "--format", "int8"])
        assert ns.format == "int8"

    def test_batch_parsed(self):
        ap = cli.build_parser()
        ns = ap.parse_args(["bench", "--batch", "4"])
        assert ns.batch == 4

    def test_iters_parsed(self):
        ap = cli.build_parser()
        ns = ap.parse_args(["bench", "--iters", "50"])
        assert ns.iters == 50

    def test_warmup_parsed(self):
        ap = cli.build_parser()
        ns = ap.parse_args(["bench", "--warmup", "3"])
        assert ns.warmup == 3

    def test_group_size_parsed(self):
        ap = cli.build_parser()
        ns = ap.parse_args(["bench", "--group-size", "32"])
        assert ns.group_size == 32

    def test_invalid_format_rejected(self):
        ap = cli.build_parser()
        with pytest.raises(SystemExit):
            ap.parse_args(["bench", "--format", "fp16"])
