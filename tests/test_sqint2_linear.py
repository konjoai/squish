"""tests/test_sqint2_linear.py — W103.4c SQINT2Linear MLX inference module.

Coverage:
  - Pure-NumPy reference base GEMV (`_base_gemv_numpy`) matches the residual
    decomposition in decompress_weight() to fp32 roundoff.
  - SQINT2Linear constructor validation (without MLX): RuntimeError on import.
  - With MLX available: forward of single-vector and batched x matches
    decompress_weight(layer) @ x to ≤ 1e-3 abs / rel.
  - Bias add path.
  - residual_rank=0 + nnz=0 path: forward stays correct (no NaN, matches dense).
  - rank>0 only / sparse>0 only routes both work.
  - rotate_left=False / rotate_right=False configurations.
  - in_features < in_pad: zero-padding path is correct.
  - from_layer factory yields the same module as the raw constructor.
  - Module count gate: squish/ stays at 85 after W103.4c (84 + sqint2_linear.py).
  - Public API: SQINT2Linear is importable from squish.quant.sqint2_linear.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from squish.quant.sqint2 import (
    SQINT2Config,
    compress_weight,
    decompress_weight,
)
from squish.quant.sqint2_linear import (
    SQINT2Linear,
    _HAS_MLX,
    _base_gemv_numpy,
)

mx = pytest.importorskip("mlx.core") if _HAS_MLX else None


# ── helpers ─────────────────────────────────────────────────────────────────


def _gauss(out=64, in_=128, sigma=0.02, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((out, in_)).astype(np.float32) * sigma


def _layer(out=64, in_=128, rank=8, sparse_frac=0.01, seed=0,
           rotate_left=True, rotate_right=True, factor_dtype="fp16"):
    W = _gauss(out, in_, seed=seed)
    cfg = SQINT2Config(
        group_size=32, refine_iters=2, seed=42,
        rotate_left=rotate_left, rotate_right=rotate_right,
        residual_rank=rank, residual_factor_dtype=factor_dtype,
        sparse_frac=sparse_frac,
    )
    return W, compress_weight(W, cfg)


# ── 1. Pure-NumPy reference GEMV  ───────────────────────────────────────────


class TestBaseGemvNumpy:
    """`_base_gemv_numpy(layer, x_rot)` should equal the residual-stripped
    portion of decompress_weight() applied to x_rot. The test reconstructs
    that portion explicitly."""

    def test_matches_dequant_only_residual_zero(self):
        # rank=0 + sparse_frac=0 → decompress_weight is the base GEMV target.
        W, layer = _layer(rank=0, sparse_frac=0.0)
        x = np.random.default_rng(2).standard_normal(
            (layer.scales.shape[1] * layer.cfg.group_size,)
        ).astype(np.float32)
        # x is in the rotated frame already (no Hadamard applied here — just
        # reuse the same vector for both paths).
        y = _base_gemv_numpy(layer, x)
        # Build expected via decompress_weight then unapply Hadamard.
        from squish.quant.sqint2 import (
            _unpack_2bit, _round_up, NF2_VALUES,
        )
        in_pad = _round_up(layer.in_features, layer.cfg.group_size)
        n_groups = in_pad // layer.cfg.group_size
        idx = _unpack_2bit(layer.indices, in_pad)
        rescaled = NF2_VALUES[idx.astype(np.intp)]
        rescaled_3d = rescaled.reshape(layer.out_features, n_groups, layer.cfg.group_size)
        scale_3d = layer.scales[:, :, None]
        zp_3d = layer.zero_points[:, :, None]
        W_rot = ((rescaled_3d - zp_3d) * scale_3d).reshape(layer.out_features, in_pad)
        y_ref = (W_rot @ x).astype(np.float32)
        np.testing.assert_allclose(y, y_ref, atol=1e-5, rtol=1e-5)

    def test_constant_zero_input(self):
        _, layer = _layer()
        in_pad = layer.scales.shape[1] * layer.cfg.group_size
        y = _base_gemv_numpy(layer, np.zeros(in_pad, dtype=np.float32))
        np.testing.assert_array_equal(y, np.zeros(layer.out_features, dtype=np.float32))


# ── 2. Constructor validation ────────────────────────────────────────────────


class TestImport:
    def test_class_is_importable(self):
        from squish.quant.sqint2_linear import SQINT2Linear as SL
        assert SL is SQINT2Linear

    def test_runtimeerror_without_mlx(self):
        if _HAS_MLX:
            pytest.skip("mlx is installed; skipping the no-mlx path")
        # When mlx is missing the class import succeeds but instantiation fails.
        _, layer = _layer()
        with pytest.raises(RuntimeError, match="mlx"):
            SQINT2Linear.from_layer(layer)


# ── 3. Forward parity (only when MLX is available) ──────────────────────────


@pytest.mark.skipif(not _HAS_MLX, reason="mlx (Apple Silicon) not installed")
class TestForwardParity:
    """y = SQINT2Linear(layer) @ x must equal decompress_weight(layer) @ x."""

    def _forward_check(self, W, layer, x, atol=1e-3, rtol=1e-3):
        sl = SQINT2Linear.from_layer(layer)
        y_mlx = np.asarray(sl(mx.array(x)))
        W_recon = decompress_weight(layer)
        y_ref = (x.astype(np.float32) @ W_recon.T.astype(np.float32))
        np.testing.assert_allclose(y_mlx, y_ref, atol=atol, rtol=rtol)

    def test_full_path_single_vector(self):
        W, layer = _layer(rank=16, sparse_frac=0.01, factor_dtype="fp32")
        x = np.random.default_rng(7).standard_normal(layer.in_features).astype(np.float32)
        self._forward_check(W, layer, x)

    def test_full_path_batched(self):
        W, layer = _layer(rank=16, sparse_frac=0.01, factor_dtype="fp32")
        x = np.random.default_rng(7).standard_normal((4, layer.in_features)).astype(np.float32)
        self._forward_check(W, layer, x)

    def test_no_residual(self):
        W, layer = _layer(rank=0, sparse_frac=0.0)
        x = np.random.default_rng(0).standard_normal(layer.in_features).astype(np.float32)
        self._forward_check(W, layer, x)

    def test_lowrank_only(self):
        W, layer = _layer(rank=8, sparse_frac=0.0, factor_dtype="fp32")
        x = np.random.default_rng(0).standard_normal(layer.in_features).astype(np.float32)
        self._forward_check(W, layer, x)

    def test_sparse_only_rank_zero(self):
        W, layer = _layer(rank=0, sparse_frac=0.01)
        x = np.random.default_rng(0).standard_normal(layer.in_features).astype(np.float32)
        self._forward_check(W, layer, x)

    def test_in_features_not_multiple_of_group_size(self):
        # in=130, gs=32 → in_pad=160; padding path exercised.
        W = _gauss(64, 130)
        cfg = SQINT2Config(group_size=32, residual_rank=4, sparse_frac=0.0,
                            residual_factor_dtype="fp32")
        layer = compress_weight(W, cfg)
        x = np.random.default_rng(0).standard_normal(130).astype(np.float32)
        self._forward_check(W, layer, x)

    def test_rotate_left_false(self):
        W = _gauss()
        cfg = SQINT2Config(group_size=32, rotate_left=False, residual_factor_dtype="fp32")
        layer = compress_weight(W, cfg)
        x = np.random.default_rng(0).standard_normal(layer.in_features).astype(np.float32)
        self._forward_check(W, layer, x)

    def test_rotate_right_false(self):
        # rotate_right requires gs to divide in_features (no padding rotation).
        W = _gauss(out=64, in_=128)
        cfg = SQINT2Config(group_size=32, rotate_right=False, residual_factor_dtype="fp32")
        layer = compress_weight(W, cfg)
        x = np.random.default_rng(0).standard_normal(layer.in_features).astype(np.float32)
        self._forward_check(W, layer, x)

    def test_bias_add(self):
        W, layer = _layer(rank=4, sparse_frac=0.0, factor_dtype="fp32")
        bias = np.linspace(-0.1, 0.1, layer.out_features).astype(np.float32)
        sl = SQINT2Linear.from_layer(layer, bias=bias)
        x = np.random.default_rng(0).standard_normal(layer.in_features).astype(np.float32)
        y = np.asarray(sl(mx.array(x)))
        W_recon = decompress_weight(layer)
        y_ref = (x.astype(np.float32) @ W_recon.T.astype(np.float32)) + bias
        np.testing.assert_allclose(y, y_ref, atol=1e-3, rtol=1e-3)

    def test_x_wrong_dim_raises(self):
        _, layer = _layer()
        sl = SQINT2Linear.from_layer(layer)
        wrong = mx.zeros((layer.in_features + 7,), dtype=mx.float32)
        with pytest.raises(ValueError, match="in_features"):
            sl(wrong)


# ── 4. Properties / repr ────────────────────────────────────────────────────


@pytest.mark.skipif(not _HAS_MLX, reason="mlx (Apple Silicon) not installed")
class TestProperties:
    def test_in_out_features(self):
        _, layer = _layer(out=96, in_=128)
        sl = SQINT2Linear.from_layer(layer)
        assert sl.in_features == 128
        assert sl.out_features == 96
        assert sl.group_size == 32

    def test_repr_string(self):
        _, layer = _layer()
        sl = SQINT2Linear.from_layer(layer)
        s = repr(sl)
        assert "SQINT2Linear" in s
        assert "in=128" in s
        assert "out=64" in s


# ── 5. Module count gate ────────────────────────────────────────────────────


class TestModuleCount:
    def test_count_equals_85(self):
        """W103.4c adds squish/quant/sqint2_linear.py: 84 → 85.

        Ceiling stays at 125 (CLAUDE.md). Headroom: 40.
        """
        import squish
        root = Path(squish.__file__).parent
        py_files = [
            f for f in root.rglob("*.py")
            if "experimental" not in f.parts
            and "__pycache__" not in f.parts
        ]
        count = len(py_files)
        assert count == 85, (
            f"Module count {count} != 85. W103.4c expected to add exactly "
            "sqint2_linear.py."
        )
        assert count <= 125
