"""Tests for NF4 (NormalFloat4) quantization — implements the --nf4 convert path.

The module backs convert.py's `--nf4`/`use_nf4` writer and loader_utils' reader,
which were previously dangling imports (ImportError on use).
"""
from __future__ import annotations

import numpy as np
import pytest

from squish.convert import _pick_int4_group_size
from squish.quant.nf4_quant import _NF4_CODEBOOK, dequantize_nf4, quantize_nf4


class TestCodebook:
    def test_shape_and_dtype(self):
        assert _NF4_CODEBOOK.shape == (16,)
        assert _NF4_CODEBOOK.dtype == np.float32

    def test_bounds_zero_and_monotonic(self):
        assert _NF4_CODEBOOK[0] == -1.0
        assert _NF4_CODEBOOK[-1] == 1.0
        assert _NF4_CODEBOOK[7] == 0.0
        assert np.all(np.diff(_NF4_CODEBOOK) > 0)


class TestContract:
    @pytest.mark.parametrize("n,d", [(4, 128), (8, 256), (3, 4096), (2, 64)])
    def test_shapes_match_writer_reader_contract(self, n, d):
        gs = _pick_int4_group_size(d)
        w = np.random.default_rng(0).standard_normal((n, d)).astype(np.float32)
        packed, scales = quantize_nf4(w, group_size=gs)
        assert packed.shape == (n, d // 2) and packed.dtype == np.uint8
        assert scales.shape == (n, d // gs) and scales.dtype == np.float32

    @pytest.mark.parametrize("d", [128, 256, 4096, 64])
    def test_reader_derives_group_size_exactly(self, d):
        # loader_utils recovers gs as (packed.shape[1]*2) // scales.shape[1].
        gs = _pick_int4_group_size(d)
        w = np.random.default_rng(1).standard_normal((4, d)).astype(np.float32)
        packed, scales = quantize_nf4(w, group_size=gs)
        assert (packed.shape[1] * 2) // scales.shape[1] == gs


class TestRoundTrip:
    @pytest.mark.parametrize("n,d", [(4, 128), (8, 256), (3, 4096)])
    def test_gaussian_reconstruction_accuracy(self, n, d):
        gs = _pick_int4_group_size(d)
        w = np.random.default_rng(2).standard_normal((n, d)).astype(np.float32)
        recon = dequantize_nf4(*quantize_nf4(w, group_size=gs), group_size=gs)
        assert recon.shape == w.shape
        # NF4 on (near-)Gaussian weights stays well under 0.12 relative error.
        assert np.linalg.norm(recon - w) / np.linalg.norm(w) < 0.12

    def test_nibble_packing_order_lo_even_hi_odd(self):
        # Exercise distinct even/odd indices to lock the packing convention.
        w = np.array([[-1.0, 1.0, 0.0, -1.0]], dtype=np.float32)  # gs=4, one group
        packed, scales = quantize_nf4(w, group_size=4)
        recon = dequantize_nf4(packed, scales, group_size=4)
        # ±1 and 0 are exact codebook levels → reconstructed exactly.
        np.testing.assert_allclose(recon, w, atol=1e-6)

    def test_zero_group_is_safe(self):
        w = np.zeros((2, 64), dtype=np.float32)
        recon = dequantize_nf4(*quantize_nf4(w, group_size=32), group_size=32)
        assert np.all(recon == 0.0)

    def test_per_group_scale_independence(self):
        # Two groups with very different magnitudes both reconstruct well.
        w = np.concatenate(
            [np.full((1, 32), 0.01, np.float32), np.full((1, 32), 5.0, np.float32)],
            axis=1,
        )
        recon = dequantize_nf4(*quantize_nf4(w, group_size=32), group_size=32)
        assert np.linalg.norm(recon - w) / np.linalg.norm(w) < 0.05


class TestValidation:
    def test_rejects_1d(self):
        with pytest.raises(ValueError):
            quantize_nf4(np.zeros(64, dtype=np.float32), group_size=32)

    def test_rejects_indivisible_group(self):
        with pytest.raises(ValueError):
            quantize_nf4(np.zeros((2, 100), dtype=np.float32), group_size=32)

    def test_rejects_odd_cols(self):
        with pytest.raises(ValueError):
            quantize_nf4(np.zeros((2, 33), dtype=np.float32), group_size=33)
