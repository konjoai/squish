"""Regression tests for grouped INT8 reconstruction with non-aligned dims.

The internal `_quantize_numpy*` helpers pad the last partial group when the
column count `d` is not a multiple of `group_size`. Reconstruction must honour
the *real* group size — which is not recoverable from `(dims, n_groups)` alone —
so QuantizationResult carries `group_size`. Before the fix, reconstruction
re-derived `group_size = dims // n_groups`, applying scales to the wrong columns
(max error ~0.45 instead of ~0.01).
"""
from __future__ import annotations

import numpy as np
import pytest

from squish.quant.quantizer import (
    QuantizationResult,
    _quantize_numpy,
    _quantize_numpy_asymmetric,
    _reconstruct_numpy,
)


@pytest.mark.parametrize("qfn", [_quantize_numpy, _quantize_numpy_asymmetric])
@pytest.mark.parametrize(
    "d,group_size",
    [
        (100, 64),  # non-aligned: pad 28
        (130, 64),  # non-aligned: pad 62
        (96, 32),   # non-aligned to 32
        (128, 64),  # aligned (control)
        (64, 64),   # group_size == d → per-row branch
    ],
)
def test_grouped_roundtrip_accurate(qfn, d, group_size):
    rng = np.random.default_rng(1234)
    x = rng.standard_normal((4, d)).astype(np.float32)
    res = qfn(x, group_size=group_size)
    recon = _reconstruct_numpy(res)
    assert recon.shape == x.shape
    # INT8 grouped quant keeps max abs error well under 0.05 for unit-variance data.
    assert np.abs(recon - x).max() < 0.05


def test_group_size_field_is_stored_on_grouped_path():
    x = np.random.default_rng(0).standard_normal((2, 100)).astype(np.float32)
    res = _quantize_numpy(x, group_size=64)
    assert res.group_size == 64


def test_legacy_result_without_group_size_still_reconstructs_aligned():
    # Old results predate the group_size field (defaults to 0). On aligned data
    # the dims // n_groups fallback is exact, so reconstruction stays correct.
    rng = np.random.default_rng(7)
    x = rng.standard_normal((3, 128)).astype(np.float32)
    res = _quantize_numpy(x, group_size=64)
    legacy = QuantizationResult(
        quantized=res.quantized, scales=res.scales, dims=res.dims, n=res.n
    )  # group_size omitted → 0
    assert legacy.group_size == 0
    recon = _reconstruct_numpy(legacy)
    assert np.abs(recon - x).max() < 0.05


def test_nonaligned_reconstruction_was_broken_without_field():
    # Documents the original defect: a non-aligned result with group_size unset
    # reconstructs with large error (scales land on the wrong columns).
    rng = np.random.default_rng(3)
    x = rng.standard_normal((4, 100)).astype(np.float32)
    res = _quantize_numpy(x, group_size=64)
    broken = res._replace(group_size=0)  # simulate the pre-fix derive-from-dims path
    recon_broken = _reconstruct_numpy(broken)
    recon_fixed = _reconstruct_numpy(res)
    assert np.abs(recon_broken - x).max() > 0.2   # wrong
    assert np.abs(recon_fixed - x).max() < 0.05   # correct
