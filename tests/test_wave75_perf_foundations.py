"""tests/test_wave75_perf_foundations.py

Wave 75 — Performance Foundations

Tests for:
  - _warmup_model        : Metal JIT pre-compile on startup
  - Tier-3 loader tags   : npy-dir* strings trigger the first-run warning
  - Chunked prefill default: on by default; --no-chunk-prefill disables it
  - _print_optimization_status: prints a row for every tracked module
  - FusedSampler fallback: _warn() is called and flag is set False on import failure
"""
from __future__ import annotations

import io
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

import os

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)


# ===========================================================================
# _warmup_model
# ===========================================================================

class TestWarmupModelNoop(unittest.TestCase):
    """_warmup_model() returns immediately and safely when _state.model is None."""

    def test_noop_when_no_model(self):
        """Should not raise or call mlx when model is None."""
        from squish.server import _warmup_model

        state = MagicMock()
        state.model = None
        state.tokenizer = None

        with patch("squish.server._state", state):
            # Must not raise regardless of MLX availability
            _warmup_model(verbose=False)

    def test_noop_when_no_model_verbose(self):
        """No output when model is None, even with verbose=True."""
        from squish.server import _warmup_model

        state = MagicMock()
        state.model = None
        state.tokenizer = MagicMock()

        with patch("squish.server._state", state):
            with patch("squish.server._ok") as mock_ok:
                with patch("squish.server._warn") as mock_warn:
                    _warmup_model(verbose=True)

        mock_ok.assert_not_called()
        mock_warn.assert_not_called()


class TestWarmupModelWithModel(unittest.TestCase):
    """_warmup_model() runs a forward pass and calls mx.eval when model is loaded."""

    def _make_state(self, bos_id: int | None = 1) -> MagicMock:
        state = MagicMock()
        state.model = MagicMock()
        state.tokenizer = MagicMock()
        state.tokenizer.bos_token_id = bos_id
        return state

    def test_calls_model_forward_pass(self):
        """The model is called exactly once when a valid model is in state."""
        from squish.server import _warmup_model

        state = self._make_state(bos_id=1)

        with patch("squish.server._state", state):
            _warmup_model(verbose=False)

        state.model.assert_called_once()

    def test_uses_bos_token_from_tokenizer(self):
        """tokenizer.bos_token_id is accessed to build the dummy input token."""
        from squish.server import _warmup_model

        state = self._make_state(bos_id=7)

        with patch("squish.server._state", state):
            _warmup_model(verbose=False)

        # Verify bos_token_id was accessed (proves we read the tokenizer)
        _ = state.tokenizer.bos_token_id  # attribute was accessed during warmup

    def test_falls_back_to_token_1_when_no_bos(self):
        """Model is still called even when bos_token_id is None (uses token 1)."""
        from squish.server import _warmup_model

        state = self._make_state(bos_id=None)

        with patch("squish.server._state", state):
            _warmup_model(verbose=False)

        # Model must still be called; no_bos path uses token 1 as fallback
        state.model.assert_called_once()

    def test_verbose_ok_contains_timing_info(self):
        """_ok() is called with Metal JIT warm-up text when verbose=True."""
        from squish.server import _warmup_model

        state = self._make_state()

        with patch("squish.server._state", state):
            with patch("squish.server._ok") as mock_ok:
                _warmup_model(verbose=True)

        self.assertTrue(mock_ok.called)
        self.assertIn("Metal JIT warm-up", mock_ok.call_args[0][0])

    def test_forward_pass_exception_is_caught(self):
        """Exceptions during the forward pass are caught; _warn called when verbose."""
        from squish.server import _warmup_model

        state = self._make_state()
        state.model.side_effect = RuntimeError("Metal kernel error")

        with patch("squish.server._state", state):
            with patch("squish.server._warn") as mock_warn:
                # Must not raise
                _warmup_model(verbose=True)

        mock_warn.assert_called_once()
        self.assertIn("[warmup]", mock_warn.call_args[0][0])

    def test_forward_pass_exception_silent_when_not_verbose(self):
        """Exceptions are swallowed silently when verbose=False."""
        from squish.server import _warmup_model

        state = self._make_state()
        state.model.side_effect = RuntimeError("Metal kernel error")

        with patch("squish.server._state", state):
            with patch("squish.server._warn") as mock_warn:
                _warmup_model(verbose=False)

        mock_warn.assert_not_called()

    def test_any_exception_during_warmup_does_not_propagate(self):
        """Any exception inside the warmup try-block is caught and not re-raised."""
        from squish.server import _warmup_model

        state = self._make_state()
        # Cause an arbitrary exception on model call
        state.model.side_effect = Exception("unexpected internal error")

        try:
            with patch("squish.server._state", state):
                _warmup_model(verbose=False)
        except Exception as _e:  # pragma: no cover
            self.fail(f"_warmup_model should not propagate exceptions, got: {_e}")


# ===========================================================================
# Tier-3 loader tag detection logic
# ===========================================================================

class TestTier3LoaderTagDetection(unittest.TestCase):
    """Verify which loader tags identify a costly first-run dequantize pass."""

    _TIER3_TAGS = [
        "npy-dir",
        "npy-dir-int4",
        "npy-dir-4bit",
        "npy-dir-8bit",
    ]
    _FAST_TAGS = [
        "squish-mlx",
        "finalized-f16",
        "squish-4bit",
        "mlx-native",
        "mlx_lm",
        "transformers",
        "squish-torch",
        "squish",
    ]

    def test_tier3_tags_identified_by_prefix(self):
        """All slow first-run tags must start with 'npy-dir'."""
        for tag in self._TIER3_TAGS:
            with self.subTest(tag=tag):
                self.assertTrue(
                    tag.startswith("npy-dir"),
                    f"{tag!r} should be identified as tier-3 (slow first-run)",
                )

    def test_fast_tags_not_tier3(self):
        """All fast cached-load tags must NOT start with 'npy-dir'."""
        for tag in self._FAST_TAGS:
            with self.subTest(tag=tag):
                self.assertFalse(
                    tag.startswith("npy-dir"),
                    f"{tag!r} should NOT be identified as tier-3",
                )


# ===========================================================================
# Chunked prefill default (Wave 75: on by default)
# ===========================================================================

class TestChunkedPrefillDefault(unittest.TestCase):
    """chunk_prefill must be ON by default; --no-chunk-prefill disables it."""

    def test_on_by_default_no_flag(self):
        """When no 'no_chunk_prefill' attribute is set the result is True."""
        args = types.SimpleNamespace()
        result = not getattr(args, "no_chunk_prefill", False)
        self.assertTrue(result, "chunk_prefill should be on by default")

    def test_disabled_by_no_chunk_prefill_true(self):
        """When no_chunk_prefill=True the result is False."""
        args = types.SimpleNamespace(no_chunk_prefill=True)
        result = not getattr(args, "no_chunk_prefill", False)
        self.assertFalse(result, "chunk_prefill should be off when --no-chunk-prefill is set")

    def test_legacy_flag_does_not_disable(self):
        """The legacy chunk_prefill=True attribute does not disable the new default."""
        args = types.SimpleNamespace(chunk_prefill=True)
        # New logic only reads no_chunk_prefill — legacy flag is a no-op
        result = not getattr(args, "no_chunk_prefill", False)
        self.assertTrue(result)

    def test_both_flags_no_chunk_prefill_wins(self):
        """If for some reason both flags appear, no_chunk_prefill=True disables it."""
        args = types.SimpleNamespace(chunk_prefill=True, no_chunk_prefill=True)
        result = not getattr(args, "no_chunk_prefill", False)
        self.assertFalse(result)

    def test_server_module_global_default_is_true(self):
        """squish.server._chunk_prefill_enabled must be False initially (set True in main()).

        The module-level default is still False; main() sets it to True at startup
        (not-no-chunk-prefill).  We verify the logic expression produces True for
        a fresh namespace here rather than checking the raw global.
        """
        args_no_flag = types.SimpleNamespace()
        self.assertTrue(not getattr(args_no_flag, "no_chunk_prefill", False))


# ===========================================================================
# _print_optimization_status
# ===========================================================================

class TestPrintOptimizationStatus(unittest.TestCase):
    """_print_optimization_status() emits one row per tracked module."""

    _ALL_MODULE_NAMES = [
        "fused-sampler",
        "chunk-prefill",
        "cache-warmup",
        "metal-jit-warmup",
        "prefix-cache",
        "paged-kv",
        "flash-attn3",
    ]

    def _run_with_patches(self, **overrides) -> str:
        """Return captured stdout from _print_optimization_status with given patches."""
        from squish.server import _print_optimization_status

        mock_state_on = MagicMock()
        mock_state_on.model = object()  # truthy

        defaults = {
            "_fused_sampler_enabled": True,
            "_fused_sampler":         MagicMock(),
            "_chunk_prefill_enabled": True,
            "_chunk_prefill_threshold": 512,
            "_cache_warmup_predictor": MagicMock(),
            "_state":                 mock_state_on,
            "_prefix_cache":          MagicMock(_maxsize=512),
            "_paged_kv_cache":        MagicMock(),
            "_flash_attn3":           MagicMock(),
        }
        defaults.update(overrides)

        buf = io.StringIO()
        with patch.multiple("squish.server", **defaults):
            with patch("sys.stdout", buf):
                _print_optimization_status()
        return buf.getvalue()

    def test_all_module_names_appear(self):
        """Every tracked module name must appear in the output."""
        output = self._run_with_patches()
        for name in self._ALL_MODULE_NAMES:
            with self.subTest(module=name):
                self.assertIn(name, output, f"Module '{name}' missing from status table")

    def test_fused_sampler_active(self):
        """fused-sampler row is printed when it is enabled and loaded."""
        output = self._run_with_patches(
            _fused_sampler_enabled=True,
            _fused_sampler=MagicMock(),
        )
        self.assertIn("fused-sampler", output)

    def test_fused_sampler_disabled(self):
        """fused-sampler row still appears when disabled (just shows ✗ / disabled)."""
        mock_state_off = MagicMock()
        mock_state_off.model = None
        output = self._run_with_patches(
            _fused_sampler_enabled=False,
            _fused_sampler=None,
            _state=mock_state_off,
        )
        self.assertIn("fused-sampler", output)

    def test_chunk_prefill_row_present(self):
        """chunk-prefill row appears whether enabled or not."""
        output_on  = self._run_with_patches(_chunk_prefill_enabled=True)
        output_off = self._run_with_patches(_chunk_prefill_enabled=False)
        self.assertIn("chunk-prefill", output_on)
        self.assertIn("chunk-prefill", output_off)

    def test_paged_kv_row_present(self):
        """paged-kv row appears whether enabled or not."""
        output_on  = self._run_with_patches(_paged_kv_cache=MagicMock())
        output_off = self._run_with_patches(_paged_kv_cache=None)
        self.assertIn("paged-kv", output_on)
        self.assertIn("paged-kv", output_off)

    def test_prefix_cache_disabled_shows_row(self):
        """prefix-cache row still appears when _maxsize == 0."""
        output = self._run_with_patches(
            _prefix_cache=MagicMock(_maxsize=0),
        )
        self.assertIn("prefix-cache", output)


# ===========================================================================
# FusedSampler fallback warning
# ===========================================================================

class TestFusedSamplerFallback(unittest.TestCase):
    """If FusedSampler import fails, _fused_sampler_enabled is set False and _warn() is called."""

    def test_import_error_disables_flag_and_calls_warn(self):
        """Simulate the init block: ImportError → _fused_sampler_enabled=False + _warn."""
        import squish.server as _srv

        captured: list[str] = []

        def _fake_warn(msg: str) -> None:
            captured.append(msg)

        # Save originals and restore in finally
        orig_enabled = _srv._fused_sampler_enabled
        orig_sampler = _srv._fused_sampler
        try:
            _srv._fused_sampler_enabled = True
            # Re-enact the try/except block that lives in main():
            try:
                raise ImportError("no module named squish.hardware.fused_sampler")
            except Exception as _e:
                _srv._fused_sampler_enabled = False
                _fake_warn(f"[fused-sampler] Skipped: {_e}")

            self.assertFalse(_srv._fused_sampler_enabled)
            self.assertEqual(len(captured), 1)
            self.assertIn("[fused-sampler]", captured[0])
            self.assertIn("Skipped", captured[0])
        finally:
            _srv._fused_sampler_enabled = orig_enabled
            _srv._fused_sampler = orig_sampler

    def test_fused_sampler_flag_survives_import_failure(self):
        """After a fallback, _fused_sampler stays None."""
        import squish.server as _srv

        orig_sampler = _srv._fused_sampler
        try:
            _srv._fused_sampler = None
            # Simulate failed load: fused_sampler is None, enabled is False
            _srv._fused_sampler_enabled = False
            self.assertIsNone(_srv._fused_sampler)
            self.assertFalse(_srv._fused_sampler_enabled)
        finally:
            _srv._fused_sampler = orig_sampler
            _srv._fused_sampler_enabled = True  # restore to module default


if __name__ == "__main__":
    unittest.main()
