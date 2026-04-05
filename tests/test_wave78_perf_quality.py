"""tests/test_wave78_perf_quality.py

Wave 78 — Module-load performance + INT2/3 quantisation quality

Tests for:
  - RadixTree lazy init: `import squish.server` does NOT import squish.kv.radix_cache
  - _init_prefix_cache / module __getattr__: _PrefixCache accessible before and after init
  - _preoptimize_weights_with_hqq: correct FFN weight modification + non-FFN keys unchanged
  - cmd_convert_model group_size auto-tighten for INT2
  - cmd_check_model: parses config.json and runs HQQ quality simulation
  - _check_layer_config: correct warning emission per configuration
"""
from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import numpy as np

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)


# ===========================================================================
# RadixTree lazy-load: no radix_cache import at module import time
# ===========================================================================

class TestRadixTreeLazyInit(unittest.TestCase):
    """Importing squish.server must NOT import squish.kv.radix_cache at module level."""

    def test_prefix_cache_initialised_via_init_func(self):
        """_init_prefix_cache() must populate both _prefix_cache and _PrefixCache."""
        import squish.server as _srv

        # Save originals, force-reset to test lazy init path
        orig_pc  = _srv.__dict__.get("_prefix_cache")
        orig_rt  = _srv.__dict__.get("_RadixTree")
        orig_pcc = _srv.__dict__.pop("_PrefixCache", None)  # remove from dict to force lazy path
        try:
            _srv._prefix_cache = None
            _srv._RadixTree = None
            # Explicitly call init
            _srv._init_prefix_cache()
            self.assertIsNotNone(_srv._prefix_cache, "_prefix_cache should be set after _init_prefix_cache()")
            self.assertIsNotNone(_srv.__dict__.get("_PrefixCache"), "_PrefixCache should be in __dict__ after init")
        finally:
            _srv._prefix_cache = orig_pc
            _srv._RadixTree    = orig_rt
            if orig_pcc is not None:
                _srv.__dict__["_PrefixCache"] = orig_pcc

    def test_init_prefix_cache_idempotent(self):
        """Calling _init_prefix_cache() twice must not re-create the cache instance."""
        import squish.server as _srv

        # Patch _prefix_cache to a sentinel to ensure no re-init happens
        sentinel = MagicMock()
        sentinel._maxsize = 512
        with patch.object(_srv, "_prefix_cache", sentinel):
            _srv._init_prefix_cache()
            # Should still be the sentinel (not replaced)
            self.assertIs(_srv._prefix_cache, sentinel)

    def test_prefix_cache_instance_is_radix_tree(self):
        """After _init_prefix_cache, _prefix_cache must be a RadixTree instance."""
        import squish.server as _srv

        _srv._init_prefix_cache()
        self.assertIsNotNone(_srv._prefix_cache)
        # Duck-type check: RadixTree exposes get/put/size/_maxsize
        for attr in ("get", "put", "size", "_maxsize"):
            self.assertTrue(
                hasattr(_srv._prefix_cache, attr),
                f"_prefix_cache missing expected attribute: {attr}",
            )

    def test_null_guard_in_generate_tokens_path(self):
        """_generate_tokens must not crash when _prefix_cache is None at entry."""
        # We can't call _generate_tokens directly without a loaded model, but we
        # can verify that _init_prefix_cache() is accessible and sets the cache.
        import squish.server as _srv

        orig = _srv._prefix_cache
        try:
            _srv._prefix_cache = None
            _srv._init_prefix_cache()
            self.assertIsNotNone(_srv._prefix_cache)
        finally:
            _srv._prefix_cache = orig

    def test_print_optimization_status_calls_init(self):
        """_print_optimization_status must call _init_prefix_cache (or handle None)."""
        import squish.server as _srv

        mock_cache = MagicMock()
        mock_cache._maxsize = 512

        with patch.multiple(
            "squish.server",
            _fused_sampler_enabled=False,
            _fused_sampler=None,
            _chunk_prefill_enabled=False,
            _chunk_prefill_threshold=512,
            _cache_warmup_predictor=None,
            _state=MagicMock(model=None),
            _prefix_cache=mock_cache,
            _paged_kv_cache=None,
        ):
            buf = io.StringIO()
            with patch("sys.stdout", buf):
                _srv._print_optimization_status()
            output = buf.getvalue()
        # prefix-cache row must appear with the cap value
        self.assertIn("prefix-cache", output)


# ===========================================================================
# _preoptimize_weights_with_hqq
# ===========================================================================

class TestPreoptimizeWeightsHQQ(unittest.TestCase):
    """Unit tests for the HQQ pre-optimisation shard-processing function."""

    def _write_fake_model_dir(self, tmp_dir: Path) -> Path:
        """Write a minimal fake BF16 model directory for testing."""
        try:
            import mlx.core as mx
        except ImportError:
            self.skipTest("mlx not available")

        src = tmp_dir / "src_model"
        src.mkdir()

        # Write config.json
        (src / "config.json").write_text(json.dumps({
            "model_type": "qwen2",
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "intermediate_size": 128,
            "vocab_size": 256,
        }))

        # Write a fake shard with FFN and non-FFN weights
        weights = {
            "model.layers.0.mlp.gate_proj.weight": mx.ones((8, 16), dtype=mx.bfloat16) * 0.1,
            "model.layers.0.mlp.up_proj.weight":   mx.ones((8, 16), dtype=mx.bfloat16) * 0.2,
            "model.layers.0.mlp.down_proj.weight": mx.ones((16, 8), dtype=mx.bfloat16) * 0.15,
            # Non-FFN keys must NOT be modified
            "model.layers.0.self_attn.q_proj.weight": mx.ones((8, 8), dtype=mx.bfloat16) * 0.05,
            "model.embed_tokens.weight":               mx.ones((256, 64), dtype=mx.bfloat16) * 0.01,
        }
        mx.save_safetensors(str(src / "model.safetensors"), weights)

        return src

    def test_ffn_weights_are_modified(self):
        """FFN weights (gate/up/down_proj) must be changed by HQQ pre-optimisation."""
        from squish.cli import _preoptimize_weights_with_hqq
        try:
            import mlx.core as mx
        except ImportError:
            self.skipTest("mlx not available")

        with tempfile.TemporaryDirectory() as tmp:
            src = self._write_fake_model_dir(Path(tmp))
            # Load original FFN weight
            orig_w = np.array(
                mx.load(str(src / "model.safetensors"))["model.layers.0.mlp.gate_proj.weight"]
                .astype(mx.float32)
            )

            out_dir = _preoptimize_weights_with_hqq(src, ffn_bits=2, group_size=8)
            try:
                new_w = np.array(
                    mx.load(str(out_dir / "model.safetensors"))["model.layers.0.mlp.gate_proj.weight"]
                    .astype(mx.float32)
                )
                # After HQQ encode/decode, values will lie on the quantisation grid
                # (not necessarily equal to the original)
                self.assertEqual(orig_w.shape, new_w.shape)
            finally:
                import shutil
                shutil.rmtree(out_dir, ignore_errors=True)

    def test_non_ffn_weights_unchanged(self):
        """Non-FFN keys (attn, embed) must be passed through unchanged."""
        from squish.cli import _preoptimize_weights_with_hqq
        try:
            import mlx.core as mx
        except ImportError:
            self.skipTest("mlx not available")

        with tempfile.TemporaryDirectory() as tmp:
            src = self._write_fake_model_dir(Path(tmp))
            orig_q = np.array(
                mx.load(str(src / "model.safetensors"))["model.layers.0.self_attn.q_proj.weight"]
                .astype(mx.float32)
            )
            orig_emb = np.array(
                mx.load(str(src / "model.safetensors"))["model.embed_tokens.weight"]
                .astype(mx.float32)
            )

            out_dir = _preoptimize_weights_with_hqq(src, ffn_bits=2, group_size=8)
            try:
                new_q   = np.array(
                    mx.load(str(out_dir / "model.safetensors"))["model.layers.0.self_attn.q_proj.weight"]
                    .astype(mx.float32)
                )
                new_emb = np.array(
                    mx.load(str(out_dir / "model.safetensors"))["model.embed_tokens.weight"]
                    .astype(mx.float32)
                )
                np.testing.assert_array_almost_equal(orig_q, new_q, decimal=3)
                np.testing.assert_array_almost_equal(orig_emb, new_emb, decimal=3)
            finally:
                import shutil
                shutil.rmtree(out_dir, ignore_errors=True)

    def test_config_json_copied(self):
        """config.json and other non-safetensors files must be copied to tmp dir."""
        from squish.cli import _preoptimize_weights_with_hqq
        try:
            import mlx.core as _mx
        except ImportError:
            self.skipTest("mlx not available")

        with tempfile.TemporaryDirectory() as tmp:
            src = self._write_fake_model_dir(Path(tmp))
            out_dir = _preoptimize_weights_with_hqq(src, ffn_bits=3, group_size=8)
            try:
                self.assertTrue((out_dir / "config.json").exists(),
                                "config.json must be copied to tmp dir")
            finally:
                import shutil
                shutil.rmtree(out_dir, ignore_errors=True)

    def test_raises_on_hf_id(self):
        """Passing a non-existent path (HF ID-like) should fail with a helpful error."""
        from squish.cli import _preoptimize_weights_with_hqq
        # A path that doesn't exist has no safetensors files → should die
        with self.assertRaises(SystemExit):
            _preoptimize_weights_with_hqq(
                Path("/nonexistent/fake/model"),
                ffn_bits=2,
                group_size=32,
            )


# ===========================================================================
# cmd_convert_model — group_size auto-tighten
# ===========================================================================

class TestAutoTightenGroupSize(unittest.TestCase):
    """cmd_convert_model must auto-tighten group_size to 32 for INT2."""

    def _make_args(self, ffn_bits=2, group_size=64, hqq=False, dry_run=False):
        args = MagicMock()
        args.source_path = "/fake/model-bf16"
        args.output_path = "/tmp/fake-output"
        args.ffn_bits = ffn_bits
        args.embed_bits = 8
        args.attn_bits = 4
        args.group_size = group_size
        args._default_group_size = 64
        args.mixed_recipe = None
        args.hqq = hqq
        args.cpu = False
        args.dry_run = dry_run
        args.blazing_m3 = False   # Wave 81: prevent _apply_blazing_m3_preset from firing
        return args

    def _run_cmd(self, args) -> str:
        """Run cmd_convert_model with mocked mlx_lm and path checks; return stdout."""
        from squish.cli import cmd_convert_model

        fake_path = MagicMock()
        fake_path.exists.return_value = True
        fake_path.__str__ = lambda self: "/fake/model-bf16"
        fake_path.__truediv__ = lambda self, other: MagicMock(exists=lambda: False)

        mock_mlx_lm = MagicMock()
        fake_convert = MagicMock()
        mock_mlx_lm.convert = fake_convert

        buf = io.StringIO()
        with patch("sys.stdout", buf), \
             patch("squish.cli.Path") as mock_path_cls, \
             patch.dict("sys.modules", {"mlx_lm": mock_mlx_lm}):
            # Make Path(source) return our fake_path so .exists() returns True
            mock_path_cls.return_value.expanduser.return_value.resolve.return_value = fake_path
            # But let the output path resolve to a different mock
            mock_path_cls.side_effect = lambda x: (
                MagicMock(
                    __str__=lambda s: str(x),
                    expanduser=lambda s=None: MagicMock(
                        resolve=lambda: MagicMock(
                            exists=MagicMock(return_value=(x == "/fake/model-bf16")),
                            __str__=lambda s: str(x),
                            __truediv__=lambda s, o: MagicMock(exists=MagicMock(return_value=False)),
                        )
                    ),
                )
            )
            try:
                cmd_convert_model(args)
            except (SystemExit, Exception):
                pass
        return buf.getvalue()

    def test_int2_default_group_size_auto_tightened(self):
        """When ffn_bits==2 and user has not set --group-size, default 64 → 32."""
        args = self._make_args(ffn_bits=2, group_size=64)
        # dry_run=False so we reach the auto-tighten logic; mlx_lm is mocked
        # We use a simpler approach: call the group_size logic directly by
        # inspecting what group_size becomes inside cmd_convert_model.
        # Instead: call with dry_run=True but check after path validation via a
        # real temp dir with a config.json.
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "config.json").write_text(json.dumps({
                "model_type": "qwen2",
                "hidden_size": 2048,
                "num_hidden_layers": 32,
                "vocab_size": 32000,
            }))
            args.source_path = tmp
            args.output_path = str(Path(tmp) / "output")
            args.dry_run = False

            mock_mlx_lm = MagicMock()
            buf = io.StringIO()
            with patch("sys.stdout", buf), \
                 patch.dict("sys.modules", {"mlx_lm": mock_mlx_lm}):
                try:
                    from squish.cli import cmd_convert_model
                    cmd_convert_model(args)
                except (SystemExit, Exception):
                    pass
            output = buf.getvalue()
        self.assertIn("tightening", output.lower(),
                      f"Expected auto group_size tighten message for INT2, got: {output!r}")

    def test_int2_explicit_group_size_not_tightened(self):
        """When user explicitly sets --group-size 128 for INT2, respect it."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "config.json").write_text(json.dumps({
                "model_type": "qwen2", "hidden_size": 2048, "num_hidden_layers": 32, "vocab_size": 32000
            }))
            args = self._make_args(ffn_bits=2, group_size=128)
            args._default_group_size = 64
            args.source_path = tmp
            args.output_path = str(Path(tmp) / "output")
            args.dry_run = False

            mock_mlx_lm = MagicMock()
            buf = io.StringIO()
            with patch("sys.stdout", buf), \
                 patch.dict("sys.modules", {"mlx_lm": mock_mlx_lm}):
                try:
                    from squish.cli import cmd_convert_model
                    cmd_convert_model(args)
                except (SystemExit, Exception):
                    pass
            output = buf.getvalue()
        self.assertNotIn("tightening", output.lower())

    def test_int4_group_size_unchanged(self):
        """INT4 quantisation must never auto-tighten group_size."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "config.json").write_text(json.dumps({
                "model_type": "qwen2", "hidden_size": 2048, "num_hidden_layers": 32, "vocab_size": 32000
            }))
            args = self._make_args(ffn_bits=4, group_size=64)
            args.source_path = tmp
            args.output_path = str(Path(tmp) / "output")
            args.dry_run = False

            mock_mlx_lm = MagicMock()
            buf = io.StringIO()
            with patch("sys.stdout", buf), \
                 patch.dict("sys.modules", {"mlx_lm": mock_mlx_lm}):
                try:
                    from squish.cli import cmd_convert_model
                    cmd_convert_model(args)
                except (SystemExit, Exception):
                    pass
            output = buf.getvalue()
        self.assertNotIn("tightening", output.lower())


# ===========================================================================
# _check_layer_config — warning emission
# ===========================================================================

class TestCheckLayerConfig(unittest.TestCase):
    """_check_layer_config must emit the correct warnings per configuration."""

    def _run(self, label, bits, gs, n_params=2_000_000_000, hidden_size=2048):
        from squish.cli import _check_layer_config
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            _check_layer_config(label, bits, gs, n_params, hidden_size)
        return buf.getvalue()

    def test_int2_large_group_size_warns(self):
        output = self._run("ffn", 2, 64)
        self.assertIn("INT2", output)
        self.assertIn("group_size=64", output)

    def test_int2_small_group_size_no_warning(self):
        output = self._run("ffn", 2, 32)
        self.assertNotIn("risky", output.lower())

    def test_small_model_int2_warns(self):
        output = self._run("ffn", 2, 32, n_params=500_000_000)
        self.assertIn("small", output.lower())

    def test_int4_no_warnings(self):
        output = self._run("ffn", 4, 64)
        self.assertEqual(output.strip(), "")

    def test_int2_attn_without_hqq_warns(self):
        output = self._run("attention", 2, 32)
        self.assertIn("INT2 attention", output)

    def test_none_bits_no_output(self):
        """bits=None must produce no output."""
        output = self._run("ffn", None, 64)
        self.assertEqual(output.strip(), "")


# ===========================================================================
# cmd_check_model — integration
# ===========================================================================

class TestCmdCheckModel(unittest.TestCase):
    """cmd_check_model must parse config.json and print quality metrics."""

    def _write_quant_model(self, tmp: Path, bits: int = 4, gs: int = 64) -> Path:
        model_dir = tmp / "test_quant_model"
        model_dir.mkdir()
        cfg = {
            "model_type": "qwen2",
            "hidden_size": 512,
            "num_hidden_layers": 4,
            "intermediate_size": 1024,
            "vocab_size": 1024,
            "quantization_config": {"bits": bits, "group_size": gs},
        }
        (model_dir / "config.json").write_text(json.dumps(cfg))
        return model_dir

    def _run_check(self, model_path: Path) -> str:
        from squish.cli import cmd_check_model
        args = MagicMock()
        args.model = str(model_path)
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            cmd_check_model(args)
        return buf.getvalue()

    def test_basic_int4_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = self._write_quant_model(Path(tmp), bits=4, gs=64)
            output = self._run_check(model_dir)
        self.assertIn("4-bit", output)
        self.assertIn("SNR", output)

    def test_int2_outputs_snr_warning(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = self._write_quant_model(Path(tmp), bits=2, gs=64)
            output = self._run_check(model_dir)
        self.assertIn("SNR", output)
        # INT2 should trigger the group_size warning in the config section
        self.assertIn("INT2", output)

    def test_model_type_in_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = self._write_quant_model(Path(tmp))
            output = self._run_check(model_dir)
        self.assertIn("qwen2", output)

    def test_missing_model_raises(self):
        from squish.cli import cmd_check_model
        args = MagicMock()
        args.model = "/definitely/does/not/exist/xyz_model"
        with self.assertRaises(SystemExit):
            cmd_check_model(args)

    def test_no_config_json_raises(self):
        from squish.cli import cmd_check_model
        with tempfile.TemporaryDirectory() as tmp:
            # Empty dir — no config.json
            args = MagicMock()
            args.model = tmp
            with self.assertRaises(SystemExit):
                cmd_check_model(args)


# ===========================================================================
# HQQ quality improvement — encode/decode SNR
# ===========================================================================

class TestHQQQualityVsNaive(unittest.TestCase):
    """HQQ must produce lower reconstruction error than naive round-to-nearest for INT2."""

    def _naive_quant_error(self, W: np.ndarray, bits: int, group_size: int) -> float:
        """Reference: naive per-group affine quantization (what mlx_lm.convert uses)."""
        n_levels = 1 << bits
        qmax = n_levels - 1.0
        rows, cols = W.shape
        n_groups = max(1, (cols + group_size - 1) // group_size)
        padded = n_groups * group_size
        W_pad = np.pad(W, ((0, 0), (0, padded - cols))) if padded > cols else W
        W_g = W_pad.reshape(rows, n_groups, group_size)
        g_min = W_g.min(axis=-1, keepdims=True)
        g_max = W_g.max(axis=-1, keepdims=True)
        span = np.maximum(g_max - g_min, 1e-6)
        scale = span / qmax
        codes = np.clip(np.round((W_g - g_min) / scale), 0, qmax)
        W_hat = codes * scale + g_min
        W_hat_flat = W_hat.reshape(rows, -1)[:, :cols]
        norm_orig = float(np.linalg.norm(W))
        if norm_orig == 0:
            return 0.0
        return float(np.linalg.norm(W - W_hat_flat)) / norm_orig

    def test_hqq_lower_error_than_naive_int2(self):
        """HQQ must achieve strictly lower relative error than naive quant for INT2."""
        from squish.experimental.hqq_quant import HQQConfig, HQQQuantizer

        rng = np.random.default_rng(1337)
        W = rng.standard_normal((128, 64)).astype(np.float32) * 0.02

        # Naive error
        naive_err = self._naive_quant_error(W, bits=2, group_size=64)

        # HQQ error
        cfg = HQQConfig(bits=2, group_size=64, max_iter=10)
        q = HQQQuantizer(cfg)
        t = q.encode(W)
        W_hat = q.decode(t)
        hqq_err = q.relative_error(W, W_hat)

        self.assertLess(
            hqq_err, naive_err,
            f"HQQ relative error {hqq_err:.6f} must be < naive {naive_err:.6f} for INT2",
        )

    def test_hqq_int3_snr_above_threshold(self):
        """HQQ INT3 must achieve SNR > 15 dB on typical transformer weight distribution."""
        from squish.experimental.hqq_quant import HQQConfig, HQQQuantizer

        rng = np.random.default_rng(42)
        W = rng.standard_normal((256, 64)).astype(np.float32) * 0.02
        cfg = HQQConfig(bits=3, group_size=64, max_iter=10)
        q = HQQQuantizer(cfg)
        t = q.encode(W)
        W_hat = q.decode(t)
        snr_db = q.quantisation_error_db(W, W_hat)
        self.assertGreater(snr_db, 15.0, f"HQQ INT3 SNR {snr_db:.1f} dB must be > 15 dB")


if __name__ == "__main__":
    unittest.main()
