#!/usr/bin/env python3
"""Isolated before/after for PR-B item 4 — npy-dir passthrough (``__pt.npy``).

Measures the exact operation the change touches, on a real model's float16
passthrough tensors, with no server/thermal noise:

  OLD: mx.array(np.array(arr, dtype=np.float32)).astype(mx.bfloat16)
  NEW: mx.array(np.asarray(arr)).astype(mx.bfloat16)            # f16, no upcast

Per tensor (no accumulation — mirrors the load loop's transient, which is what
this change touches): asserts the bf16 result is BIT-IDENTICAL between the two
paths, and tracks the transient peak RSS. The float32 upcast allocates a host
buffer 2× the float16 size *per tensor*; that transient is the saving.

Run:
  ~/squish/.venv/bin/python benchmarks/perf/npy_pt_passthrough_bench.py \
      [~/models/Qwen3-8B-int4/tensors]
"""
from __future__ import annotations

import os
import sys
import time

import mlx.core as mx
import numpy as np
import psutil

_PROC = psutil.Process()


def _rss_mb() -> float:
    return _PROC.memory_info().rss / 1e6


def _convert(path: str, upcast: bool) -> tuple[mx.array, float]:
    """One tensor through the old/new path; returns (bf16, transient_peak_mb)."""
    arr = np.load(path, mmap_mode="r")               # float16 on disk
    if upcast:
        host = np.array(arr, dtype=np.float32)       # OLD: 2× host buffer
    else:
        host = np.asarray(arr)                        # NEW: f16, no upcast
    peak = _rss_mb()                                  # sample with host buffer live
    bf16 = mx.array(host).astype(mx.bfloat16)
    mx.eval(bf16)
    return bf16, peak


def _pass(pt_files: list[str], upcast: bool) -> tuple[float, float]:
    """Time the full pass and capture the max transient RSS across tensors."""
    base = _rss_mb()
    peak = base
    t0 = time.perf_counter()
    for p in pt_files:
        bf16, tr = _convert(p, upcast)
        peak = max(peak, tr)
        del bf16
    return time.perf_counter() - t0, peak - base


def _check_bit_identical(pt_files: list[str]) -> None:
    for p in pt_files:
        old, _ = _convert(p, upcast=True)
        new, _ = _convert(p, upcast=False)
        if not bool(mx.all(old == new).item()):
            raise AssertionError(f"bf16 mismatch (f16→f32→bf16 != f16→bf16): {p}")
        del old, new


def main() -> int:
    tensor_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser(
        "~/models/Qwen3-8B-int4/tensors")
    pt_files = sorted(
        os.path.join(tensor_dir, f)
        for f in os.listdir(tensor_dir)
        if f.endswith("__pt.npy")
    )
    if not pt_files:
        print(f"No __pt.npy tensors in {tensor_dir}")
        return 1
    total_mb = sum(os.path.getsize(p) for p in pt_files) / 1e6
    print(f"{len(pt_files)} passthrough tensors, {total_mb:.0f} MB float16 on disk")

    _check_bit_identical(pt_files)
    print(f"correctness: bf16 bit-identical across all {len(pt_files)} tensors ✓")

    old_t, old_r, new_t, new_r = [], [], [], []
    for _ in range(3):  # interleaved → shared thermal conditions
        dt, dr = _pass(pt_files, upcast=True);  old_t.append(dt); old_r.append(dr)
        dt, dr = _pass(pt_files, upcast=False); new_t.append(dt); new_r.append(dr)
    old_med, new_med = sorted(old_t)[1], sorted(new_t)[1]
    print(f"\n{'path':<24}{'time (median)':>16}{'transient RSS':>16}")
    print(f"{'OLD (f16→f32→bf16)':<24}{old_med * 1000:>14.0f}ms{max(old_r):>14.0f}MB")
    print(f"{'NEW (f16→bf16)':<24}{new_med * 1000:>14.0f}ms{max(new_r):>14.0f}MB")
    print(f"\nspeedup: {old_med / new_med:.2f}×  "
          f"({(1 - new_med / old_med) * 100:.0f}% faster)  "
          f"peak transient saved: {max(old_r) - max(new_r):.0f}MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
