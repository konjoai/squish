"""Regression: HQQ decode must use the stored config group size.

decode() recomputed group_size as ceil(dim_size / n_groups), which differs from
the real group size whenever dim_size is not an exact multiple of it — every
group then misaligned against its scale/zero and reconstruction error blew up.
"""
from __future__ import annotations

import numpy as np
import pytest

from squish.quant.hqq import HQQConfig, HQQQuantizer


def _rel_err(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / np.linalg.norm(a))


@pytest.mark.parametrize("dim,group_size", [(100, 30), (130, 64), (96, 32), (100, 25)])
@pytest.mark.parametrize("axis", [0, 1])
def test_decode_roundtrip_non_divisible_group(dim, group_size, axis):
    rng = np.random.default_rng(0)
    w = (rng.standard_normal((dim, 4)) if axis == 1
         else rng.standard_normal((4, dim))).astype(np.float32)
    q = HQQQuantizer(HQQConfig(bits=4, group_size=group_size, axis=axis))
    recon = q.decode(q.encode(w))
    assert recon.shape == w.shape
    # 4-bit HQQ on unit Gaussian keeps relative error well under 0.15.
    assert _rel_err(w, recon) < 0.15


def test_axis1_roundtrip_was_broken_before_fix():
    # decode() never transposed the stored axis-1 codes back, so axis=1 raised
    # a broadcast error end-to-end (even on aligned dims).
    rng = np.random.default_rng(5)
    w = rng.standard_normal((96, 4)).astype(np.float32)
    q = HQQQuantizer(HQQConfig(bits=4, group_size=32, axis=1))
    recon = q.decode(q.encode(w))
    assert recon.shape == w.shape
    assert _rel_err(w, recon) < 0.15


def test_non_divisible_was_broken_before_fix():
    # Sharpened guard: the non-aligned case used to be ~0.30 rel error.
    rng = np.random.default_rng(1)
    w = rng.standard_normal((4, 100)).astype(np.float32)
    q = HQQQuantizer(HQQConfig(bits=4, group_size=30, axis=0))
    assert _rel_err(w, q.decode(q.encode(w))) < 0.15


def test_full_row_group_size_minus_one():
    rng = np.random.default_rng(2)
    w = rng.standard_normal((4, 100)).astype(np.float32)
    q = HQQQuantizer(HQQConfig(bits=4, group_size=-1, axis=0))
    recon = q.decode(q.encode(w))
    assert recon.shape == w.shape
    assert np.isfinite(recon).all()
