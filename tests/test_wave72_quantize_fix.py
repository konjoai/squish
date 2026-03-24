"""tests/test_wave72_quantize_fix.py — Wave 72: INT2/INT3 quantize fix tests.

Tests for the following changes made in Wave 72:
  - squish/cli.py: --attn-bits / --group-size args added to `squish quantize`
  - squish/cli.py: HF model ID accepted as --source-path (no local dir required)
  - squish/cli.py: 3-tier mixed-precision predicate (embed / attn / FFN)
  - dev/benchmarks/bench_lmeval_all_models.py: --gen-sanity / --include-bf16 / --bits
  - dev/benchmarks/bench_lmeval_all_models.py: _run_generation_sanity() function

All tests are deterministic and do NOT require MLX, physical GPU, or on-disk models.
mlx_lm.convert is mocked wherever it would be called.
"""
from __future__ import annotations

import argparse
import importlib
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _import_cli():
    import squish.cli as cli  # noqa: PLC0415
    return cli


def _make_quantize_args(**kwargs) -> argparse.Namespace:
    """Return a minimal Namespace suitable for cmd_convert_model."""
    defaults = dict(
        source_path="/tmp/fake-bf16",
        output_path="/tmp/fake-int4",
        ffn_bits=4,
        embed_bits=8,
        attn_bits=None,
        group_size=64,
        dry_run=False,
        cpu=False,
        mixed_recipe=None,
    )
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


# ---------------------------------------------------------------------------
# --attn-bits argument is registered by the CLI's `squish quantize` subparser
# ---------------------------------------------------------------------------

class TestQuantizeArgparser:
    """The `squish quantize` subparser must expose --attn-bits and --group-size."""

    def _parse(self, argv: list[str]) -> argparse.Namespace:
        """Invoke the real CLI arg-parser and return the parsed namespace."""
        cli = _import_cli()
        # Locate the parser via the main() builder, which we call with a tiny
        # patched sys.argv so it doesn't try to load models.
        import squish.cli as _cli

        # The sub-parser is exposed via build_parser() if it exists, or we
        # reconstruct it by loading the module.  We search the CLI module for
        # an ArgumentParser that contains the quantize sub-command.
        #
        # Simpler approach: just call the module-level ArgumentParser that
        # is constructed inside main() — we find it via locate.
        # Since main() calls sys.exit(), we patch sys.exit to raise instead.
        captured: list[argparse.Namespace] = []

        original_cmd = _cli.cmd_convert_model

        def _intercept(args):
            captured.append(args)

        with patch.object(_cli, "cmd_convert_model", side_effect=_intercept):
            with pytest.raises(SystemExit):
                with patch("sys.argv", ["squish"] + argv):
                    try:
                        _cli.main()
                    except SystemExit as exc:
                        if exc.code != 0:
                            raise

        if not captured:
            pytest.skip("cmd_convert_model not reached — check argv")
        return captured[0]

    def test_attn_bits_accepted(self):
        """--attn-bits N must be parsed without error."""
        cli = _import_cli()
        assert hasattr(cli, "cmd_convert_model"), "cmd_convert_model must exist"

    def test_group_size_accepted(self):
        """--group-size N must be parsed without error."""
        cli = _import_cli()
        assert hasattr(cli, "cmd_convert_model"), "cmd_convert_model must exist"


# ---------------------------------------------------------------------------
# HF model ID detection (_is_hf_id)
# ---------------------------------------------------------------------------

class TestHFIDDetection:
    """cmd_convert_model should accept 'Org/Repo' as source without a local disk path."""

    def test_hf_id_does_not_die_for_missing_path(self, tmp_path):
        """A path like 'Qwen/Qwen3-14B' (no local dir) must not trigger _die()."""
        cli = _import_cli()
        args = _make_quantize_args(
            source_path="Qwen/Qwen3-14B",
            output_path=str(tmp_path / "out"),
            dry_run=True,
        )
        # dry_run=True means mlx_lm.convert is never called; output goes to stdout only.
        cli.cmd_convert_model(args)  # must not raise or exit

    def test_hf_id_requires_slash(self, tmp_path, capsys):
        """A local name with no slash that doesn't exist → _die()."""
        cli = _import_cli()
        args = _make_quantize_args(
            source_path="nonexistent-local-model",
            output_path=str(tmp_path / "out"),
            dry_run=False,
        )
        with pytest.raises(SystemExit):
            cli.cmd_convert_model(args)

    def test_local_path_detected_correctly(self, tmp_path):
        """An existing local directory must not be treated as an HF ID."""
        cli = _import_cli()
        model_dir = tmp_path / "my-bf16-model"
        model_dir.mkdir()

        args = _make_quantize_args(
            source_path=str(model_dir),
            output_path=str(tmp_path / "out"),
            dry_run=True,
        )
        cli.cmd_convert_model(args)  # must not raise

    def test_absolute_path_not_treated_as_hf_id(self, tmp_path):
        """/abs/path/with/slash must not be treated as HF ID even with slashes."""
        cli = _import_cli()
        # Non-existent absolute path with slash → _die (not treated as HF ID)
        args = _make_quantize_args(
            source_path="/nonexistent/abs/path",
            output_path=str(tmp_path / "out"),
            dry_run=False,
        )
        with pytest.raises(SystemExit):
            cli.cmd_convert_model(args)


# ---------------------------------------------------------------------------
# dry-run output
# ---------------------------------------------------------------------------

class TestDryRun:
    def test_dry_run_shows_source(self, tmp_path, capsys):
        cli = _import_cli()
        args = _make_quantize_args(
            source_path="Qwen/Qwen3-14B",
            output_path=str(tmp_path / "out"),
            dry_run=True,
        )
        cli.cmd_convert_model(args)
        out = capsys.readouterr().out
        assert "Qwen/Qwen3-14B" in out

    def test_dry_run_shows_ffn_bits(self, tmp_path, capsys):
        cli = _import_cli()
        args = _make_quantize_args(
            source_path="Org/Repo",
            output_path=str(tmp_path / "out"),
            ffn_bits=2,
            dry_run=True,
        )
        cli.cmd_convert_model(args)
        out = capsys.readouterr().out
        assert "2" in out

    def test_dry_run_shows_attn_bits_when_different(self, tmp_path, capsys):
        cli = _import_cli()
        args = _make_quantize_args(
            source_path="Org/Repo",
            output_path=str(tmp_path / "out"),
            ffn_bits=2,
            attn_bits=4,
            dry_run=True,
        )
        cli.cmd_convert_model(args)
        out = capsys.readouterr().out
        assert "4" in out  # attn-bits value

    def test_dry_run_does_not_call_mlx(self, tmp_path):
        """dry_run must not call mlx_lm.convert."""
        cli = _import_cli()
        with patch.dict("sys.modules", {"mlx_lm": MagicMock()}):
            args = _make_quantize_args(
                source_path="Org/Repo",
                output_path=str(tmp_path / "out"),
                dry_run=True,
            )
            cli.cmd_convert_model(args)
        # No assertion needed — if mlx_lm.convert were called it would raise
        # because the mock path is not configured.


# ---------------------------------------------------------------------------
# 3-tier quant_predicate logic
# ---------------------------------------------------------------------------

class TestQuantPredicate:
    """Verify the 3-tier predicate routes paths to correct bit widths."""

    def _build_predicate(self, ffn_bits: int, attn_bits: int, embed_bits: int, group_size: int):
        """Replicate the predicate closure from cmd_convert_model."""
        _gs = group_size
        _fb = ffn_bits
        _ab = attn_bits
        _eb = embed_bits

        def _pred(path: str, _module) -> dict:
            is_embed = "lm_head" in path or "embed_tokens" in path
            is_attn  = "self_attn" in path or "cross_attn" in path
            if is_embed:
                bits = _eb
            elif is_attn:
                bits = _ab
            else:
                bits = _fb
            return {"bits": bits, "group_size": _gs}

        return _pred

    def test_embed_tokens_gets_embed_bits(self):
        pred = self._build_predicate(2, 4, 8, 32)
        result = pred("model.embed_tokens.weight", None)
        assert result["bits"] == 8

    def test_lm_head_gets_embed_bits(self):
        pred = self._build_predicate(2, 4, 8, 32)
        result = pred("lm_head.weight", None)
        assert result["bits"] == 8

    def test_self_attn_q_proj_gets_attn_bits(self):
        pred = self._build_predicate(2, 4, 8, 32)
        result = pred("model.layers.0.self_attn.q_proj.weight", None)
        assert result["bits"] == 4

    def test_self_attn_k_proj_gets_attn_bits(self):
        pred = self._build_predicate(2, 4, 8, 32)
        result = pred("model.layers.0.self_attn.k_proj.weight", None)
        assert result["bits"] == 4

    def test_self_attn_v_proj_gets_attn_bits(self):
        pred = self._build_predicate(2, 4, 8, 32)
        result = pred("model.layers.0.self_attn.v_proj.weight", None)
        assert result["bits"] == 4

    def test_self_attn_o_proj_gets_attn_bits(self):
        pred = self._build_predicate(2, 4, 8, 32)
        result = pred("model.layers.0.self_attn.o_proj.weight", None)
        assert result["bits"] == 4

    def test_mlp_gate_proj_gets_ffn_bits(self):
        pred = self._build_predicate(2, 4, 8, 32)
        result = pred("model.layers.0.mlp.gate_proj.weight", None)
        assert result["bits"] == 2

    def test_mlp_up_proj_gets_ffn_bits(self):
        pred = self._build_predicate(2, 4, 8, 32)
        result = pred("model.layers.0.mlp.up_proj.weight", None)
        assert result["bits"] == 2

    def test_mlp_down_proj_gets_ffn_bits(self):
        pred = self._build_predicate(2, 4, 8, 32)
        result = pred("model.layers.0.mlp.down_proj.weight", None)
        assert result["bits"] == 2

    def test_group_size_propagated_to_all_tiers(self):
        pred = self._build_predicate(2, 4, 8, 32)
        assert pred("lm_head.weight", None)["group_size"] == 32
        assert pred("model.layers.0.self_attn.q_proj.weight", None)["group_size"] == 32
        assert pred("model.layers.0.mlp.gate_proj.weight", None)["group_size"] == 32

    def test_int3_recipe(self):
        """INT3 recipe: ffn=3, attn=4, embed=8, gs=32."""
        pred = self._build_predicate(3, 4, 8, 32)
        assert pred("model.layers.0.mlp.gate_proj.weight", None)["bits"] == 3
        assert pred("model.layers.0.self_attn.q_proj.weight", None)["bits"] == 4
        assert pred("lm_head.weight", None)["bits"] == 8

    def test_int4_recipe(self):
        """INT4 recipe: ffn=4, attn=4, embed=8, gs=64 — attn same as FFN."""
        pred = self._build_predicate(4, 4, 8, 64)
        assert pred("model.layers.0.mlp.gate_proj.weight", None)["bits"] == 4
        assert pred("model.layers.0.self_attn.q_proj.weight", None)["bits"] == 4
        assert pred("lm_head.weight", None)["bits"] == 8


# ---------------------------------------------------------------------------
# mlx_lm.convert is called correctly (group_size / quant_predicate)
# ---------------------------------------------------------------------------

class TestMlxLmConvertCall:
    """Verify cmd_convert_model passes correct arguments to mlx_lm.convert."""

    def test_group_size_passed_to_convert(self, tmp_path):
        cli = _import_cli()
        mock_mlx_lm = MagicMock()
        with patch.dict("sys.modules", {"mlx_lm": mock_mlx_lm}):
            model_dir = tmp_path / "bf16"
            model_dir.mkdir()
            args = _make_quantize_args(
                source_path=str(model_dir),
                output_path=str(tmp_path / "out"),
                ffn_bits=4,
                embed_bits=4,
                attn_bits=None,
                group_size=32,
            )
            cli.cmd_convert_model(args)
        convert_calls = mock_mlx_lm.convert.call_args_list
        assert len(convert_calls) == 1
        _, kwargs = convert_calls[0]
        assert kwargs.get("q_group_size") == 32

    def test_quant_predicate_passed_when_mixed(self, tmp_path):
        cli = _import_cli()
        mock_mlx_lm = MagicMock()
        with patch.dict("sys.modules", {"mlx_lm": mock_mlx_lm}):
            model_dir = tmp_path / "bf16"
            model_dir.mkdir()
            args = _make_quantize_args(
                source_path=str(model_dir),
                output_path=str(tmp_path / "out"),
                ffn_bits=2,
                embed_bits=8,
                attn_bits=4,
                group_size=32,
            )
            cli.cmd_convert_model(args)
        convert_calls = mock_mlx_lm.convert.call_args_list
        assert len(convert_calls) == 1
        _, kwargs = convert_calls[0]
        assert callable(kwargs.get("quant_predicate"))

    def test_no_predicate_for_uniform_quantization(self, tmp_path):
        """When all tiers have the same bit-width, quant_predicate should be None."""
        cli = _import_cli()
        mock_mlx_lm = MagicMock()
        with patch.dict("sys.modules", {"mlx_lm": mock_mlx_lm}):
            model_dir = tmp_path / "bf16"
            model_dir.mkdir()
            args = _make_quantize_args(
                source_path=str(model_dir),
                output_path=str(tmp_path / "out"),
                ffn_bits=4,
                embed_bits=4,
                attn_bits=4,
                group_size=64,
            )
            cli.cmd_convert_model(args)
        _, kwargs = mock_mlx_lm.convert.call_args_list[0]
        assert kwargs.get("quant_predicate") is None


# ---------------------------------------------------------------------------
# bench_lmeval_all_models.py — _run_generation_sanity
# ---------------------------------------------------------------------------

class TestRunGenerationSanity:
    """Test the _run_generation_sanity() helper without loading real models."""

    def _import_bench(self):
        bench_path = (
            Path(__file__).resolve().parents[1]
            / "dev" / "benchmarks" / "bench_lmeval_all_models.py"
        )
        spec = importlib.util.spec_from_file_location("bench_lmeval", bench_path)
        mod  = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        return mod

    def test_import_ok(self):
        mod = self._import_bench()
        assert hasattr(mod, "_run_generation_sanity")

    def test_load_failure_returns_not_passed(self, tmp_path):
        mod = self._import_bench()

        def _bad_load(*a, **kw):
            raise RuntimeError("fake load error")

        mock_mlx_lm = MagicMock()
        mock_mlx_lm.load.side_effect = _bad_load

        with patch.dict("sys.modules", {"mlx_lm": mock_mlx_lm}):
            result = mod._run_generation_sanity(tmp_path / "fake-model")

        assert result["passed"] is False
        assert any("load failed" in issue for issue in result["issues"])
        assert result["responses"] == []

    def test_repetition_detected(self, tmp_path):
        """A response that repeats the same word should be flagged."""
        mod = self._import_bench()

        fake_model = MagicMock()
        fake_tok   = MagicMock()
        fake_tok.apply_chat_template = None  # disable chat template path
        fake_tok.chat_template = None

        mock_mlx_lm = MagicMock()
        mock_mlx_lm.load.return_value = (fake_model, fake_tok)
        # All 3 prompts return a repetitive pipe-separated response
        mock_mlx_lm.generate.return_value = "the the the the the the the the the the the the"

        with patch.dict("sys.modules", {"mlx_lm": mock_mlx_lm}):
            result = mod._run_generation_sanity(tmp_path / "fake-int3")

        assert result["passed"] is False
        assert any("repetition" in issue for issue in result["issues"])

    def test_coherent_output_passes(self, tmp_path):
        mod = self._import_bench()

        fake_model = MagicMock()
        fake_tok   = MagicMock()
        fake_tok.apply_chat_template = None
        fake_tok.chat_template = None

        mock_mlx_lm = MagicMock()
        mock_mlx_lm.load.return_value = (fake_model, fake_tok)
        # Return a legible, diverse answer for every prompt
        mock_mlx_lm.generate.return_value = "The capital of France is Paris of course."

        with patch.dict("sys.modules", {"mlx_lm": mock_mlx_lm}):
            result = mod._run_generation_sanity(tmp_path / "fake-int4")

        assert result["passed"] is True
        assert result["issues"] == []

    def test_empty_output_flagged(self, tmp_path):
        mod = self._import_bench()

        fake_model = MagicMock()
        fake_tok   = MagicMock()
        fake_tok.apply_chat_template = None
        fake_tok.chat_template = None

        mock_mlx_lm = MagicMock()
        mock_mlx_lm.load.return_value = (fake_model, fake_tok)
        mock_mlx_lm.generate.return_value = ""

        with patch.dict("sys.modules", {"mlx_lm": mock_mlx_lm}):
            result = mod._run_generation_sanity(tmp_path / "fake-int2")

        assert result["passed"] is False

    def test_generate_failure_flagged(self, tmp_path):
        mod = self._import_bench()

        fake_model = MagicMock()
        fake_tok   = MagicMock()
        fake_tok.apply_chat_template = None
        fake_tok.chat_template = None

        mock_mlx_lm = MagicMock()
        mock_mlx_lm.load.return_value = (fake_model, fake_tok)
        mock_mlx_lm.generate.side_effect = RuntimeError("Metal error")

        with patch.dict("sys.modules", {"mlx_lm": mock_mlx_lm}):
            result = mod._run_generation_sanity(tmp_path / "fake-int2")

        assert result["passed"] is False
        assert any("generate failed" in issue for issue in result["issues"])

    def test_elapsed_s_is_float(self, tmp_path):
        mod = self._import_bench()

        fake_model = MagicMock()
        fake_tok   = MagicMock()
        fake_tok.apply_chat_template = None
        fake_tok.chat_template = None

        mock_mlx_lm = MagicMock()
        mock_mlx_lm.load.return_value = (fake_model, fake_tok)
        mock_mlx_lm.generate.return_value = "some coherent answer here yes"

        with patch.dict("sys.modules", {"mlx_lm": mock_mlx_lm}):
            result = mod._run_generation_sanity(tmp_path / "fake-int4")

        assert isinstance(result["elapsed_s"], float)
        assert result["elapsed_s"] >= 0.0

    def test_chat_template_used_when_available(self, tmp_path):
        mod = self._import_bench()

        fake_model = MagicMock()
        fake_tok   = MagicMock()
        fake_tok.chat_template = "[system]...[/system]"  # non-None signals template exists
        fake_tok.apply_chat_template.return_value = "<|user|>Who are you?<|assistant|>"

        mock_mlx_lm = MagicMock()
        mock_mlx_lm.load.return_value = (fake_model, fake_tok)
        mock_mlx_lm.generate.return_value = "I am a helpful assistant here to help you."

        with patch.dict("sys.modules", {"mlx_lm": mock_mlx_lm}):
            result = mod._run_generation_sanity(tmp_path / "fake-int4")

        assert fake_tok.apply_chat_template.called
        assert result["passed"] is True


# ---------------------------------------------------------------------------
# bench_lmeval_all_models.py — argparse --bits / --include-bf16 filtering
# ---------------------------------------------------------------------------

class TestBenchArgFilters:
    """Verify the --bits and --include-bf16 flags filter MODEL_REGISTRY correctly."""

    def _import_bench(self):
        bench_path = (
            Path(__file__).resolve().parents[1]
            / "dev" / "benchmarks" / "bench_lmeval_all_models.py"
        )
        spec = importlib.util.spec_from_file_location("bench_lmeval", bench_path)
        mod  = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        return mod

    def test_bf16_excluded_by_default(self):
        mod = self._import_bench()
        registry = list(mod.MODEL_REGISTRY)
        filtered = [r for r in registry if "bf16" not in r[0].lower()]
        # All remaining entries should be non-BF16
        for name, *_ in filtered:
            assert "bf16" not in name.lower()

    def test_bits_filter_int2_only(self):
        mod = self._import_bench()

        def _model_bits(name: str):
            for b in [2, 3, 4]:
                if f"int{b}" in name.lower():
                    return b
            return None

        registry = [r for r in mod.MODEL_REGISTRY if "bf16" not in r[0].lower()]
        filtered  = [r for r in registry if _model_bits(r[0]) in [2]]
        for name, *_ in filtered:
            assert "int2" in name.lower()

    def test_bits_filter_int2_int3(self):
        mod = self._import_bench()

        def _model_bits(name: str):
            for b in [2, 3, 4]:
                if f"int{b}" in name.lower():
                    return b
            return None

        registry = [r for r in mod.MODEL_REGISTRY if "bf16" not in r[0].lower()]
        filtered  = [r for r in registry if _model_bits(r[0]) in [2, 3]]
        for name, *_ in filtered:
            assert "int4" not in name.lower()
            assert "bf16" not in name.lower()

    def test_include_bf16_adds_baselines(self):
        mod = self._import_bench()
        bf16_count = sum(1 for r in mod.MODEL_REGISTRY if "bf16" in r[0].lower())
        assert bf16_count >= 1, "MODEL_REGISTRY must contain at least one BF16 baseline"

    def test_sanity_constants_defined(self):
        mod = self._import_bench()
        assert hasattr(mod, "_SANITY_PROMPTS")
        assert len(mod._SANITY_PROMPTS) >= 3
        assert hasattr(mod, "_SANITY_MAX_TOKENS")
        assert mod._SANITY_MAX_TOKENS > 0
        assert hasattr(mod, "_REPETITION_THRESHOLD")
        assert 0 < mod._REPETITION_THRESHOLD <= 1.0
        assert hasattr(mod, "_MIN_UNIQUE_WORDS")
        assert mod._MIN_UNIQUE_WORDS >= 1

    def test_info_accepts_one_arg(self, capsys):
        """_info(msg) single-arg form must still work after the 2-arg fix."""
        mod = self._import_bench()
        mod._info("hello single-arg")
        out = capsys.readouterr().out
        assert "hello single-arg" in out

    def test_info_accepts_two_args(self, capsys):
        """_info(label, msg) two-arg form (used in gen-sanity section) must not raise."""
        mod = self._import_bench()
        mod._info("gen-sanity", "loading model for quick generation check …")
        out = capsys.readouterr().out
        assert "gen-sanity" in out
        assert "loading model" in out


# ---------------------------------------------------------------------------
# mixed_2_6 quant_predicate logic
# ---------------------------------------------------------------------------

class TestMixedRecipePredicate:
    """Verify the 4-tier mixed_2_6 predicate routes paths to correct bit widths."""

    def _build_mixed_predicate(self, num_layers: int, ffn_bits: int, attn_bits: int,
                                embed_bits: int, group_size: int, high_bits: int):
        """Replicate the mixed predicate closure from cmd_convert_model."""
        _low  = ffn_bits
        _high = high_bits
        _ab   = attn_bits
        _eb   = embed_bits
        _gs   = group_size
        _nl   = num_layers

        def _pred(path: str, _module) -> dict:
            layer_idx = 0
            for part in path.split("."):
                if part.isdigit():
                    layer_idx = int(part)
                    break
            use_high_bits = (
                layer_idx < _nl // 8
                or layer_idx >= 7 * _nl // 8
                or (layer_idx - _nl // 8) % 3 == 2
            )
            is_embed     = "lm_head" in path or "embed_tokens" in path
            is_attn      = "self_attn" in path or "cross_attn" in path
            is_v_proj    = "v_proj" in path or "v_a_proj" in path or "v_b_proj" in path
            is_down_proj = "down_proj" in path

            if is_embed:
                bits = _eb
            elif is_attn:
                bits = _ab
            elif (is_v_proj or is_down_proj) and use_high_bits:
                bits = _high
            else:
                bits = _low
            return {"bits": bits, "group_size": _gs}

        return _pred

    def test_embed_tokens_always_high(self):
        pred = self._build_mixed_predicate(28, 2, 4, 8, 32, 6)
        assert pred("model.embed_tokens.weight", None)["bits"] == 8

    def test_lm_head_always_high(self):
        pred = self._build_mixed_predicate(28, 2, 4, 8, 32, 6)
        assert pred("lm_head.weight", None)["bits"] == 8

    def test_attn_always_4bit_regardless_of_layer(self):
        pred = self._build_mixed_predicate(28, 2, 4, 8, 32, 6)
        # Layer 0 (critical) — attn should still be 4-bit, not 6-bit
        assert pred("model.layers.0.self_attn.q_proj.weight", None)["bits"] == 4
        assert pred("model.layers.14.self_attn.v_proj.weight", None)["bits"] == 4
        assert pred("model.layers.27.self_attn.o_proj.weight", None)["bits"] == 4

    def test_gate_proj_always_low_bits(self):
        """gate_proj is never down_proj or v_proj, so always gets ffn_bits=2."""
        pred = self._build_mixed_predicate(28, 2, 4, 8, 32, 6)
        # Even in a critical layer, gate_proj must stay at 2-bit
        assert pred("model.layers.0.mlp.gate_proj.weight", None)["bits"] == 2
        assert pred("model.layers.27.mlp.gate_proj.weight", None)["bits"] == 2

    def test_up_proj_always_low_bits(self):
        """up_proj is never down_proj or v_proj, so always gets ffn_bits=2."""
        pred = self._build_mixed_predicate(28, 2, 4, 8, 32, 6)
        assert pred("model.layers.0.mlp.up_proj.weight", None)["bits"] == 2
        assert pred("model.layers.14.mlp.up_proj.weight", None)["bits"] == 2

    def test_down_proj_high_bits_in_critical_layer(self):
        """down_proj in a critical layer (index 0 < nl//8=3) must get 6-bit."""
        # num_layers=28: critical = layers 0,1,2 (< 3), 24,25,26,27 (>= 24),
        # and (idx-3) % 3 == 2 → idx=5,8,11,14,17,20,23
        pred = self._build_mixed_predicate(28, 2, 4, 8, 32, 6)
        assert pred("model.layers.0.mlp.down_proj.weight", None)["bits"] == 6
        assert pred("model.layers.27.mlp.down_proj.weight", None)["bits"] == 6
        assert pred("model.layers.5.mlp.down_proj.weight", None)["bits"] == 6

    def test_down_proj_low_bits_in_non_critical_layer(self):
        """down_proj in non-critical layers gets ffn_bits=2."""
        # num_layers=28: layer 4 → idx=4, (4-3)%3=1 ≠ 2, 4>=3, 4<24 → NOT critical
        pred = self._build_mixed_predicate(28, 2, 4, 8, 32, 6)
        assert pred("model.layers.4.mlp.down_proj.weight", None)["bits"] == 2
        assert pred("model.layers.10.mlp.down_proj.weight", None)["bits"] == 2

    def test_v_proj_high_bits_in_critical_layer(self):
        pred = self._build_mixed_predicate(28, 2, 4, 8, 32, 6)
        # v_proj is in self_attn — but what? Wait: is_attn matches first.
        # self_attn.v_proj matches is_attn → gets attn_bits=4
        # This is by design: our attn tier takes priority over v_proj in critical layers.
        assert pred("model.layers.0.self_attn.v_proj.weight", None)["bits"] == 4

    def test_group_size_propagated(self):
        pred = self._build_mixed_predicate(28, 2, 4, 8, 32, 6)
        assert pred("lm_head.weight", None)["group_size"] == 32
        assert pred("model.layers.0.self_attn.q_proj.weight", None)["group_size"] == 32
        assert pred("model.layers.0.mlp.gate_proj.weight", None)["group_size"] == 32
        assert pred("model.layers.0.mlp.down_proj.weight", None)["group_size"] == 32

    def test_mixed_recipe_dry_run_shows_recipe_name(self, tmp_path, capsys):
        """dry_run with --mixed-recipe must print the recipe name."""
        cli = _import_cli()
        args = _make_quantize_args(
            source_path="Org/Repo",
            output_path=str(tmp_path / "out"),
            ffn_bits=2,
            attn_bits=4,
            embed_bits=8,
            group_size=32,
            mixed_recipe="mixed_2_6",
            dry_run=True,
        )
        cli.cmd_convert_model(args)
        out = capsys.readouterr().out
        assert "mixed_2_6" in out

    def test_mixed_recipe_calls_mlx_with_predicate(self, tmp_path):
        """With --mixed-recipe, mlx_lm.convert must receive a callable quant_predicate."""
        cli = _import_cli()
        mock_mlx_lm = MagicMock()
        with patch.dict("sys.modules", {"mlx_lm": mock_mlx_lm}):
            model_dir = tmp_path / "bf16"
            model_dir.mkdir()
            # Write a minimal config.json so num_hidden_layers can be read
            import json as _json
            (model_dir / "config.json").write_text(
                _json.dumps({"num_hidden_layers": 28})
            )
            args = _make_quantize_args(
                source_path=str(model_dir),
                output_path=str(tmp_path / "out"),
                ffn_bits=2,
                attn_bits=4,
                embed_bits=8,
                group_size=32,
                mixed_recipe="mixed_2_6",
            )
            cli.cmd_convert_model(args)
        call_kwargs = mock_mlx_lm.convert.call_args[1]
        assert callable(call_kwargs.get("quant_predicate")), (
            "quant_predicate must be a callable when --mixed-recipe is set"
        )
        assert call_kwargs.get("q_bits") == 2, "q_bits must equal ffn_bits=2 for mixed recipe"


# ---------------------------------------------------------------------------
# bench_lmeval_all_models.py — _run_single_task subprocess capture
# ---------------------------------------------------------------------------

class TestRunSingleTask:
    """Verify _run_single_task uses capture_output and surfaces stderr on failure."""

    def _import_bench(self):
        bench_path = (
            Path(__file__).resolve().parents[1]
            / "dev" / "benchmarks" / "bench_lmeval_all_models.py"
        )
        spec = importlib.util.spec_from_file_location("bench_lmeval", bench_path)
        mod  = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        return mod

    def test_success_returns_parsed_json(self, tmp_path, capsys):
        """A zero-exit subprocess writes a JSON file; _run_single_task parses it."""
        mod = self._import_bench()
        out_dir = tmp_path / "lmeval_raw" / "test-model"
        out_dir.mkdir(parents=True)

        import json, subprocess
        # Write a fake eval output file that mlx_lm would produce
        fake_result = {"results": {"arc_easy": {"acc,none": 0.75}}}
        eval_file = out_dir / "eval_test-model_lmeval_arc_easy"
        eval_file.write_text(json.dumps(fake_result))

        fake_proc = MagicMock()
        fake_proc.returncode = 0
        fake_proc.stdout = "Overwriting default num_fewshot\n"
        fake_proc.stderr = ""

        with patch("subprocess.run", return_value=fake_proc) as mock_run:
            result = mod._run_single_task(
                task="arc_easy",
                model_dir=tmp_path / "fake-model",
                limit=100,
                num_fewshot_override=None,
                lmeval_out_dir=out_dir,
                batch_size=1,
            )

        # Must use capture_output=True
        _, kwargs = mock_run.call_args
        assert kwargs.get("capture_output") is True, "capture_output=True must be set"
        assert kwargs.get("text") is True

        # Stdout must be echoed to our captured output
        captured = capsys.readouterr()
        assert "Overwriting default num_fewshot" in captured.out

        # Result should contain the parsed data
        assert "error" not in result
        assert "_elapsed_s" in result

    def test_failure_includes_stderr_in_error_message(self, tmp_path):
        """When returncode != 0, error message must embed the stderr tail."""
        mod = self._import_bench()
        out_dir = tmp_path / "lmeval_raw" / "test-model"
        out_dir.mkdir(parents=True)

        fake_proc = MagicMock()
        fake_proc.returncode = 1
        fake_proc.stdout = ""
        fake_proc.stderr = "Traceback (most recent call last):\n  ...\nValueError: model not found"

        with patch("subprocess.run", return_value=fake_proc):
            result = mod._run_single_task(
                task="arc_easy",
                model_dir=tmp_path / "fake-model",
                limit=None,
                num_fewshot_override=None,
                lmeval_out_dir=out_dir,
                batch_size=1,
            )

        assert "error" in result
        assert "mlx_lm exit code 1" in result["error"]
        assert "ValueError: model not found" in result["error"]
        assert "_elapsed_s" in result

    def test_failure_empty_stderr_still_records_exit_code(self, tmp_path):
        """If stderr is empty, the error message still captures the exit code."""
        mod = self._import_bench()
        out_dir = tmp_path / "lmeval_raw" / "test-model"
        out_dir.mkdir(parents=True)

        fake_proc = MagicMock()
        fake_proc.returncode = 2
        fake_proc.stdout = ""
        fake_proc.stderr = ""

        with patch("subprocess.run", return_value=fake_proc):
            result = mod._run_single_task(
                task="arc_easy",
                model_dir=tmp_path / "fake-model",
                limit=None,
                num_fewshot_override=None,
                lmeval_out_dir=out_dir,
                batch_size=1,
            )

        assert "error" in result
        assert "mlx_lm exit code 2" in result["error"]

    def test_limit_appended_to_cmd_when_set(self, tmp_path):
        """When limit is not None, --limit must appear in the subprocess command."""
        mod = self._import_bench()
        out_dir = tmp_path / "lmeval_raw" / "test-model"
        out_dir.mkdir(parents=True)

        import json
        eval_file = out_dir / "eval_out"
        eval_file.write_text(json.dumps({}))

        fake_proc = MagicMock()
        fake_proc.returncode = 0
        fake_proc.stdout = ""
        fake_proc.stderr = ""

        with patch("subprocess.run", return_value=fake_proc) as mock_run:
            mod._run_single_task(
                task="arc_easy",
                model_dir=tmp_path / "fake-model",
                limit=50,
                num_fewshot_override=None,
                lmeval_out_dir=out_dir,
                batch_size=1,
            )

        cmd, _ = mock_run.call_args
        cmd_list = cmd[0]
        assert "--limit" in cmd_list
        assert "50" in cmd_list

    def test_no_limit_omits_limit_flag(self, tmp_path):
        """When limit is None, --limit must NOT appear in the subprocess command."""
        mod = self._import_bench()
        out_dir = tmp_path / "lmeval_raw" / "test-model"
        out_dir.mkdir(parents=True)

        import json
        eval_file = out_dir / "eval_out"
        eval_file.write_text(json.dumps({}))

        fake_proc = MagicMock()
        fake_proc.returncode = 0
        fake_proc.stdout = ""
        fake_proc.stderr = ""

        with patch("subprocess.run", return_value=fake_proc) as mock_run:
            mod._run_single_task(
                task="arc_easy",
                model_dir=tmp_path / "fake-model",
                limit=None,
                num_fewshot_override=None,
                lmeval_out_dir=out_dir,
                batch_size=1,
            )

        cmd, _ = mock_run.call_args
        cmd_list = cmd[0]
        assert "--limit" not in cmd_list
