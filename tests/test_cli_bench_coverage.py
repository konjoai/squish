"""Coverage for squish.cli cmd_bench — quantized GEMV micro-benchmark.
Exercises the int4 (numpy fallback) and int8 paths with tiny shapes.
Host-agnostic (no Rust ext required; the Rust matmul branch is pragma'd).
"""

from __future__ import annotations

import argparse

from squish import cli


def _ns(fmt):
    return argparse.Namespace(
        format=fmt,
        batch=2,
        in_features=64,
        out_features=64,
        group_size=32,
        iters=2,
        warmup=1,
    )


def test_bench_int4_numpy(capsys):
    cli.cmd_bench(_ns("int4"))
    out = capsys.readouterr().out
    assert "INT4 GEMV" in out and "throughput" in out


def test_bench_int8(capsys):
    cli.cmd_bench(_ns("int8"))
    out = capsys.readouterr().out
    assert "INT8 GEMV" in out and "p50  latency" in out
