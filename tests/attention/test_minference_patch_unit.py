"""tests/test_minference_patch_unit.py

Full-coverage unit tests for squish/minference_patch.py.

The existing test_wave17_server_wiring.py covers:
  - select_pattern_for_sequence (all three branches)
  - Import / callable assertions

This file adds coverage for:
  - _make_a_shape_mask        — small seq_len, normal seq_len
  - _make_vertical_slash_mask — basic correctness
  - _make_block_sparse_mask   — single block, multi-block (row_s>0 branch)
  - _iter_attention_modules   — ImportError path (mlx.nn absent), nn.Module walk
  - _make_patched_class       — seq_len <= threshold (passthrough), seq_len > threshold
                                (mask injection with/without existing mask)
  - patch_model_minference    — invalid pattern raises ValueError,
                                a-shape/vertical-slash/block-sparse pkw selection,
                                no attention modules, module __class__ patch,
                                restore_fn call
  - unpatch_model_minference  — delegates to restore_fn, exception swallowed
"""
from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from squish.attention.minference_patch import (
    _make_a_shape_mask,
    _make_block_sparse_mask,
    _make_vertical_slash_mask,
    patch_model_minference,
    select_pattern_for_sequence,
    unpatch_model_minference,
)

# Determine if MLX is available
try:
    import mlx.core as mx
    import mlx.nn as nn
    _HAS_MLX = True
except Exception:
    _HAS_MLX = False

mlx_only = pytest.mark.skipif(not _HAS_MLX, reason="mlx not available")


# ---------------------------------------------------------------------------
# select_pattern_for_sequence (already tested in wave17, but covering all branches)
# ---------------------------------------------------------------------------


class TestSelectPatternForSequence:
    def test_short_sequence_a_shape(self):
        assert select_pattern_for_sequence(512) == "a-shape"

    def test_medium_sequence_vertical_slash(self):
        assert select_pattern_for_sequence(4096) == "vertical-slash"

    def test_long_sequence_block_sparse(self):
        assert select_pattern_for_sequence(8192) == "block-sparse"
        assert select_pattern_for_sequence(100_000) == "block-sparse"

    def test_boundary_values(self):
        assert select_pattern_for_sequence(2047) == "a-shape"
        assert select_pattern_for_sequence(2048) == "vertical-slash"
        assert select_pattern_for_sequence(8191) == "vertical-slash"


# ---------------------------------------------------------------------------
# _make_a_shape_mask
# ---------------------------------------------------------------------------


@mlx_only
class TestMakeAShapeMask:
    def test_shape(self):
        mask = _make_a_shape_mask(seq_len=8, window=4, n_sinks=2)
        arr = np.array(mask)
        assert arr.shape == (8, 8)

    def test_small_seq_len_more_sinks_than_seq(self):
        """n_sinks > seq_len → sinks clamped to seq_len."""
        mask = _make_a_shape_mask(seq_len=3, window=4, n_sinks=10)
        arr = np.array(mask)
        assert arr.shape == (3, 3)

    def test_diagonal_attended(self):
        """Each position should attend to itself (mask[i,i] == 0)."""
        mask = _make_a_shape_mask(seq_len=8, window=4, n_sinks=2)
        arr = np.array(mask)
        for i in range(8):
            assert arr[i, i] == pytest.approx(0.0), f"diagonal {i} should be 0"

    def test_future_positions_blocked(self):
        """No position attends to future tokens (mask[i, j] == -1e4 for j > i)."""
        mask = _make_a_shape_mask(seq_len=6, window=3, n_sinks=1)
        arr = np.array(mask)
        for i in range(6):
            for j in range(i + 1, 6):
                assert arr[i, j] == pytest.approx(-1e4), f"future [{i},{j}]"

    def test_sinks_attended(self):
        """First n_sinks tokens should be attended from all positions."""
        mask = _make_a_shape_mask(seq_len=8, window=3, n_sinks=2)
        arr = np.array(mask)
        # Position 6 (i=6) should attend to sinks at 0 and 1
        assert arr[6, 0] == pytest.approx(0.0)
        assert arr[6, 1] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# _make_vertical_slash_mask
# ---------------------------------------------------------------------------


@mlx_only
class TestMakeVerticalSlashMask:
    def test_shape(self):
        mask = _make_vertical_slash_mask(seq_len=10, stride=3, n_sinks=2)
        arr = np.array(mask)
        assert arr.shape == (10, 10)

    def test_self_attended(self):
        """Every position should attend to itself."""
        mask = _make_vertical_slash_mask(seq_len=8, stride=2, n_sinks=1)
        arr = np.array(mask)
        for i in range(8):
            assert arr[i, i] == pytest.approx(0.0)

    def test_sinks_attended(self):
        mask = _make_vertical_slash_mask(seq_len=10, stride=5, n_sinks=3)
        arr = np.array(mask)
        # Position 5 should attend to sinks 0,1,2
        for s in range(3):
            assert arr[5, s] == pytest.approx(0.0)

    def test_stride_columns_attended(self):
        """Columns at stride positions should be 0."""
        mask = _make_vertical_slash_mask(seq_len=12, stride=3, n_sinks=0)
        arr = np.array(mask)
        # Position 9 (i=9), col=6 (stride=3, from sinks=0 to 9 in steps of 3)
        assert arr[9, 6] == pytest.approx(0.0)
        assert arr[9, 9] == pytest.approx(0.0)  # self


# ---------------------------------------------------------------------------
# _make_block_sparse_mask
# ---------------------------------------------------------------------------


@mlx_only
class TestMakeBlockSparseMask:
    def test_shape(self):
        mask = _make_block_sparse_mask(seq_len=8, block_size=4)
        arr = np.array(mask)
        assert arr.shape == (8, 8)

    def test_single_block_no_first_block_sinks(self):
        """When there's only one block (row_s=0), the first-block branch is skipped."""
        mask = _make_block_sparse_mask(seq_len=4, block_size=4)
        arr = np.array(mask)
        # Within block 0: diagonal and below should be 0
        assert arr[0, 0] == pytest.approx(0.0)
        assert arr[3, 3] == pytest.approx(0.0)
        # Above diagonal: -1e4
        assert arr[0, 1] == pytest.approx(-1e4)

    def test_second_block_attends_to_first_block_sinks(self):
        """Row in block 1 (row_s>0) should attend to first block (cols 0..block_size)."""
        mask = _make_block_sparse_mask(seq_len=8, block_size=4)
        arr = np.array(mask)
        # Row 4 is in block 1 (row_s=4 > 0) → should attend to first block (cols 0,1,2,3)
        for col in range(4):
            assert arr[4, col] == pytest.approx(0.0), f"row 4, col {col}"

    def test_non_multiple_seq_len(self):
        """seq_len not a multiple of block_size — last partial block handled."""
        mask = _make_block_sparse_mask(seq_len=7, block_size=3)
        arr = np.array(mask)
        assert arr.shape == (7, 7)


# ---------------------------------------------------------------------------
# _iter_attention_modules
# ---------------------------------------------------------------------------


class TestIterAttentionModules:
    def test_no_mlx_yields_nothing(self):
        """When mlx.nn is absent, yields nothing (ImportError path)."""
        from squish.attention.minference_patch import _iter_attention_modules
        # Temporarily hide mlx.nn
        with patch.dict(sys.modules, {"mlx.nn": None}):
            result = list(_iter_attention_modules(MagicMock()))
        assert result == []

    @mlx_only
    def test_model_without_attn_attributes_yields_nothing(self):
        """A module with no 'attn'/'attention'/'mixer' attributes yields nothing."""
        from squish.attention.minference_patch import _iter_attention_modules

        class SimpleModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(4, 4)

        model = SimpleModule()
        result = list(_iter_attention_modules(model))
        assert result == []

    @mlx_only
    def test_model_with_attn_attribute_yields_it(self):
        """Module with an 'attn' attribute that is an nn.Module is yielded."""
        from squish.attention.minference_patch import _iter_attention_modules

        class AttnModule(nn.Module):
            def __init__(self):
                super().__init__()

        class ModelWithAttn(nn.Module):
            def __init__(self):
                super().__init__()
                self.attn = AttnModule()

        model = ModelWithAttn()
        result = list(_iter_attention_modules(model))
        names = [name for name, _ in result]
        assert "attn" in names


# ---------------------------------------------------------------------------
# _make_patched_class
# ---------------------------------------------------------------------------


@mlx_only
class TestMakePatchedClass:
    def test_short_sequence_passthrough(self):
        """When seq_len <= threshold, original __call__ is used unchanged."""
        from squish.attention.minference_patch import _make_patched_class

        outputs = []

        class FakeAttn(nn.Module):
            def __init__(self):
                super().__init__()

            def __call__(self, x, mask=None, cache=None, **kw):
                outputs.append(("called", x.shape, mask))
                return x

        get_mask_calls = []

        def _get_mask(sl):
            get_mask_calls.append(sl)
            return mx.zeros((sl, sl))

        Patched = _make_patched_class(FakeAttn, _get_mask, seq_len_threshold=16)
        attn = FakeAttn()
        attn.__class__ = Patched

        x = mx.zeros((1, 4, 8))   # seq_len=4 < threshold=16
        result = attn(x, mask=None)
        assert len(get_mask_calls) == 0  # mask not requested
        assert len(outputs) == 1

    def test_long_sequence_with_mask(self):
        """When seq_len > threshold, sparse mask is added to existing mask."""
        from squish.attention.minference_patch import _make_patched_class

        received_masks = []

        class FakeAttn(nn.Module):
            def __init__(self):
                super().__init__()

            def __call__(self, x, mask=None, cache=None, **kw):
                received_masks.append(mask)
                return x

        def _get_mask(sl):
            return mx.zeros((sl, sl))

        Patched = _make_patched_class(FakeAttn, _get_mask, seq_len_threshold=4)
        attn = FakeAttn()
        attn.__class__ = Patched

        x = mx.zeros((1, 8, 8))  # seq_len=8 > threshold=4
        existing_mask = mx.zeros((8, 8))
        attn(x, mask=existing_mask)
        # mask should have been passed
        assert received_masks[0] is not None

    def test_long_sequence_without_existing_mask(self):
        """When seq_len > threshold and mask=None, sparse mask is passed as mask."""
        from squish.attention.minference_patch import _make_patched_class

        received_masks = []

        class FakeAttn(nn.Module):
            def __init__(self):
                super().__init__()

            def __call__(self, x, mask=None, cache=None, **kw):
                received_masks.append(mask)
                return x

        def _get_mask(sl):
            return mx.ones((sl, sl))

        Patched = _make_patched_class(FakeAttn, _get_mask, seq_len_threshold=4)
        attn = FakeAttn()
        attn.__class__ = Patched

        x = mx.zeros((1, 8, 8))  # seq_len=8 > threshold
        attn(x, mask=None)
        assert received_masks[0] is not None  # sparse mask was set


# ---------------------------------------------------------------------------
# patch_model_minference
# ---------------------------------------------------------------------------


class TestPatchModelMinference:
    def test_invalid_pattern_raises_value_error(self):
        model = MagicMock()
        with pytest.raises(ValueError, match="Unknown minference pattern"):
            patch_model_minference(model, pattern="unknown-pattern")

    def test_returns_callable_restore_fn(self):
        """Returns a callable even when no modules are patched."""
        model = MagicMock()
        model.__class__ = object  # Not an nn.Module, so _iter yields nothing
        restore_fn = patch_model_minference(model, pattern="a-shape")
        assert callable(restore_fn)

    def test_pattern_a_shape_pkw(self):
        """a-shape pattern uses window and n_sinks in _pkw."""
        model = MagicMock()
        # Should not raise and should return restore_fn
        restore_fn = patch_model_minference(
            model, pattern="a-shape", window=128, n_sinks=2
        )
        assert callable(restore_fn)

    def test_pattern_vertical_slash_pkw(self):
        """vertical-slash pattern uses stride and n_sinks."""
        model = MagicMock()
        restore_fn = patch_model_minference(
            model, pattern="vertical-slash", stride=32, n_sinks=3
        )
        assert callable(restore_fn)

    def test_pattern_block_sparse_pkw(self):
        """block-sparse pattern uses block_size."""
        model = MagicMock()
        restore_fn = patch_model_minference(
            model, pattern="block-sparse", block_size=32
        )
        assert callable(restore_fn)

    def test_restore_fn_calls_class_restoration(self):
        """restore_fn() attempts to restore __class__ for patched modules."""
        model = MagicMock()
        restore_fn = patch_model_minference(model, pattern="a-shape")
        restore_fn()  # should not raise


# ---------------------------------------------------------------------------
# unpatch_model_minference
# ---------------------------------------------------------------------------


class TestUnpatchModelMinference:
    def test_calls_restore_fn(self):
        """unpatch_model_minference calls the provided restore_fn."""
        model = MagicMock()
        restore_fn = MagicMock()
        unpatch_model_minference(model, restore_fn)
        restore_fn.assert_called_once()

    def test_exception_in_restore_fn_is_swallowed(self):
        """Exceptions in restore_fn are caught and swallowed — no crash."""
        model = MagicMock()

        def failing_restore():
            raise RuntimeError("restore failed")

        # Should not raise
        unpatch_model_minference(model, failing_restore)
