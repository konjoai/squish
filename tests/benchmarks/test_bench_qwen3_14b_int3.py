"""
tests/benchmarks/test_bench_qwen3_14b_int3.py

Unit tests for bench_qwen3_14b_int3.py — covers all helper functions, result
dataclasses and the main() flag-parsing path WITHOUT requiring actual model
files on disk.

All heavy I/O (directory stat, safetensors load, mlx_lm) is monkey-patched.
"""
from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import types
from dataclasses import asdict
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ── repo root on path ──────────────────────────────────────────────────────────
_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))

from dev.benchmarks.bench_qwen3_14b_int3 import (
    BenchResults,
    CompressionMetrics,
    PerplexityMetrics,
    SNRLayerResult,
    SNRMetrics,
    ThroughputMetrics,
    _detect_ram_gb,
    _dir_gb,
    _dequantize_int3,
    _ensure_mlx_int4,
    _platform_info,
    _safe_key,
    _snr_db,
    main,
    measure_compression,
    measure_perplexity,
    measure_snr,
    measure_throughput,
    print_markdown,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture()
def tmp_bf16_dir(tmp_path: Path) -> Path:
    """Minimal BF16 model directory with a ~5 MB file (rounds to non-zero GB)."""
    d = tmp_path / "Qwen3-14B-bf16"
    d.mkdir()
    (d / "config.json").write_text("{}")
    # 5 MB so _dir_gb > 0.004 GB → round(gb, 2) > 0.0
    (d / "model-00001.safetensors").write_bytes(b"\x00" * 5_000_000)
    return d


@pytest.fixture()
def tmp_int3_dir(tmp_path: Path) -> Path:
    """Minimal INT3 npy-dir with a manifest and one INT3 layer."""
    d = tmp_path / "Qwen3-14B-int3"
    td = d / "tensors"
    td.mkdir(parents=True)

    from squish.quant.milo_quant import MiLoConfig, MiLoQuantizer

    # Fake a 64×64 weight tensor
    sk = "model__layers__0__self_attn__q_proj__weight"
    shape = (64, 64)
    group_size = 64

    rng = np.random.default_rng(0)
    w = rng.standard_normal(shape).astype(np.float32) * 0.02

    milo = MiLoQuantizer(MiLoConfig(group_size=group_size, max_rank=2))
    q3, s3, z3, comp = milo.quantize(w)
    shape_arr = np.array(shape, dtype=np.int64)

    np.save(td / f"{sk}__q3.npy", q3)
    np.save(td / f"{sk}__s3.npy", s3)
    np.save(td / f"{sk}__z3.npy", z3)
    np.save(td / f"{sk}__lra.npy", comp.a)
    np.save(td / f"{sk}__lrb.npy", comp.b)
    np.save(td / f"{sk}__shape.npy", shape_arr)

    manifest = {"model.layers.0.self_attn.q_proj.weight": sk}
    (td / "manifest.json").write_text(json.dumps(manifest))
    (td / ".manifest_ready").touch()

    # Add a small passthrough file
    np.save(td / f"{sk}__pt.npy", np.zeros((8,), dtype=np.float16))

    return d


# ─────────────────────────────────────────────────────────────────────────────
# _safe_key
# ─────────────────────────────────────────────────────────────────────────────

class TestSafeKey:
    def test_dot_replacement(self):
        assert _safe_key("model.layers.0.weight") == "model__layers__0__weight"

    def test_no_dots(self):
        assert _safe_key("weight") == "weight"

    def test_multiple_dots(self):
        assert _safe_key("a.b.c.d") == "a__b__c__d"

    def test_already_safe(self):
        assert _safe_key("model__layers__0") == "model__layers__0"


# ─────────────────────────────────────────────────────────────────────────────
# _snr_db
# ─────────────────────────────────────────────────────────────────────────────

class TestSnrDb:
    def test_perfect_reconstruction_returns_inf(self):
        w = np.ones((4, 4), dtype=np.float32)
        assert _snr_db(w, w) == float("inf")

    def test_zero_error_returns_inf(self):
        w = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        snr = _snr_db(w, w.copy())
        assert snr == float("inf")

    def test_high_snr_for_small_error(self):
        rng = np.random.default_rng(42)
        w = rng.standard_normal((32, 32)).astype(np.float32)
        noise = w * 0.001
        snr = _snr_db(w, w + noise)
        assert snr > 30.0

    def test_low_snr_for_large_error(self):
        rng = np.random.default_rng(42)
        w = rng.standard_normal((32, 32)).astype(np.float32)
        noise = rng.standard_normal((32, 32)).astype(np.float32) * 10
        snr = _snr_db(w, w + noise)
        assert snr < 0.0

    def test_returns_float(self):
        w = np.ones((8,), dtype=np.float32)
        assert isinstance(_snr_db(w, w * 0.9), float)

    def test_handles_bfloat16_inputs(self):
        """Should work with float16 / bf16 converted to float32 internally."""
        w = np.array([1.0, 2.0, 3.0], dtype=np.float16)
        r = np.array([1.1, 2.1, 3.1], dtype=np.float16)
        snr = _snr_db(w, r)
        assert snr < float("inf")
        assert snr > 0.0

    @pytest.mark.parametrize("scale", [1e-6, 1.0, 1e6])
    def test_scale_invariance(self, scale: float):
        """SNR should be the same regardless of the absolute magnitude."""
        w = np.array([1.0, -1.0, 0.5], dtype=np.float32) * scale
        err = np.array([0.01, -0.01, 0.005], dtype=np.float32) * scale
        snr1 = _snr_db(w, w + err)
        snr2 = _snr_db(w * 2, (w + err) * 2)
        assert abs(snr1 - snr2) < 1e-3


# ─────────────────────────────────────────────────────────────────────────────
# _dir_gb
# ─────────────────────────────────────────────────────────────────────────────

class TestDirGb:
    def test_empty_dir(self, tmp_path: Path):
        assert _dir_gb(tmp_path) == 0.0

    def test_single_file(self, tmp_path: Path):
        f = tmp_path / "a.bin"
        f.write_bytes(b"\x00" * 1_000_000)
        gb = _dir_gb(tmp_path)
        assert abs(gb - 0.001) < 1e-6

    def test_nested_files(self, tmp_path: Path):
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "a.npy").write_bytes(b"\x00" * 500_000)
        (tmp_path / "b.npy").write_bytes(b"\x00" * 500_000)
        gb = _dir_gb(tmp_path)
        assert abs(gb - 0.001) < 1e-6

    def test_returns_float(self, tmp_path: Path):
        assert isinstance(_dir_gb(tmp_path), float)


# ─────────────────────────────────────────────────────────────────────────────
# _detect_ram_gb
# ─────────────────────────────────────────────────────────────────────────────

class TestDetectRamGb:
    def test_returns_positive_float(self):
        ram = _detect_ram_gb()
        assert isinstance(ram, float)
        assert ram > 0.0

    def test_fallback_on_sysctl_failure(self):
        with patch("subprocess.run", side_effect=Exception("no sysctl")):
            ram = _detect_ram_gb()
        assert ram == 16.0


# ─────────────────────────────────────────────────────────────────────────────
# _platform_info
# ─────────────────────────────────────────────────────────────────────────────

class TestPlatformInfo:
    def test_returns_dict_with_required_keys(self):
        info = _platform_info()
        assert isinstance(info, dict)
        assert "platform" in info
        assert "python" in info
        assert "ram_gb" in info

    def test_ram_gb_positive(self):
        info = _platform_info()
        assert info["ram_gb"] > 0.0


# ─────────────────────────────────────────────────────────────────────────────
# _dequantize_int3
# ─────────────────────────────────────────────────────────────────────────────

class TestDequantizeInt3:
    def _make_int3_data(self, shape=(64, 64), group_size=64):
        """Create valid INT3 data using the actual MiLo quantizer."""
        from squish.quant.milo_quant import MiLoConfig, MiLoQuantizer
        rng = np.random.default_rng(7)
        w = rng.standard_normal(shape).astype(np.float32) * 0.02
        milo = MiLoQuantizer(MiLoConfig(group_size=group_size, max_rank=2))
        q3, s3, z3, comp = milo.quantize(w)
        return q3, s3, z3, comp.a, comp.b, shape

    def test_output_shape(self):
        shape = (64, 64)
        q3, s3, z3, lra, lrb, shape = self._make_int3_data(shape=shape, group_size=64)
        out = _dequantize_int3(q3, s3, z3, lra, lrb, shape, group_size=64)
        assert out.shape == shape

    def test_output_dtype_float32(self):
        shape = (64, 64)
        q3, s3, z3, lra, lrb, shape = self._make_int3_data(shape=shape, group_size=64)
        out = _dequantize_int3(q3, s3, z3, lra, lrb, shape, group_size=64)
        assert out.dtype == np.float32

    def test_finite_output(self):
        shape = (64, 64)
        q3, s3, z3, lra, lrb, shape = self._make_int3_data(shape=shape, group_size=64)
        out = _dequantize_int3(q3, s3, z3, lra, lrb, shape, group_size=64)
        assert np.all(np.isfinite(out))

    def test_low_rank_compensator_applied(self):
        """Compensation should change the output vs zero compensator."""
        shape = (64, 64)
        q3, s3, z3, lra, lrb, shape = self._make_int3_data(shape=shape, group_size=64)
        # Zero-out compensator
        lra_zero = np.zeros_like(lra)
        lrb_zero = np.zeros_like(lrb)
        out_with = _dequantize_int3(q3, s3, z3, lra, lrb, shape, group_size=64)
        out_without = _dequantize_int3(q3, s3, z3, lra_zero, lrb_zero, shape, group_size=64)
        assert not np.allclose(out_with, out_without)


# ─────────────────────────────────────────────────────────────────────────────
# measure_compression
# ─────────────────────────────────────────────────────────────────────────────

class TestMeasureCompression:
    def test_returns_dataclass(self, tmp_bf16_dir: Path, tmp_int3_dir: Path):
        result = measure_compression(tmp_bf16_dir, tmp_int3_dir)
        assert isinstance(result, CompressionMetrics)

    def test_bf16_gb_positive(self, tmp_bf16_dir: Path, tmp_int3_dir: Path):
        result = measure_compression(tmp_bf16_dir, tmp_int3_dir)
        assert result.bf16_gb > 0.0

    def test_int3_gb_non_negative(self, tmp_bf16_dir: Path, tmp_int3_dir: Path):
        # Tiny 64×64 test tensors generate ~15 KB which rounds to 0.00 GB.
        # The important thing is the value is non-negative and the tensor count is right.
        result = measure_compression(tmp_bf16_dir, tmp_int3_dir)
        assert result.int3_gb >= 0.0
        assert result.n_int3_tensors == 1

    def test_ratio_in_valid_range(self, tmp_bf16_dir: Path, tmp_int3_dir: Path):
        # For tiny test dirs the overhead means INT3 can be > BF16 in size.
        # Only require ratio is a positive finite float.
        result = measure_compression(tmp_bf16_dir, tmp_int3_dir)
        assert result.ratio > 0.0 and math.isfinite(result.ratio)

    def test_size_savings_derived_from_ratio(self, tmp_bf16_dir: Path, tmp_int3_dir: Path):
        result = measure_compression(tmp_bf16_dir, tmp_int3_dir)
        expected = round((1 - result.ratio) * 100, 1)
        assert abs(result.size_savings_pct - expected) < 0.01

    def test_counts_int3_tensors(self, tmp_bf16_dir: Path, tmp_int3_dir: Path):
        result = measure_compression(tmp_bf16_dir, tmp_int3_dir)
        assert result.n_int3_tensors == 1

    def test_counts_passthrough_tensors(self, tmp_bf16_dir: Path, tmp_int3_dir: Path):
        result = measure_compression(tmp_bf16_dir, tmp_int3_dir)
        assert result.n_passthrough_tensors == 1

    def test_no_error_field(self, tmp_bf16_dir: Path, tmp_int3_dir: Path):
        result = measure_compression(tmp_bf16_dir, tmp_int3_dir)
        assert result.error is None

    def test_actual_bpw_positive(self, tmp_bf16_dir: Path, tmp_int3_dir: Path):
        result = measure_compression(tmp_bf16_dir, tmp_int3_dir)
        assert result.actual_bpw > 0.0

    def test_compress_seconds_none_without_manifest_entry(
        self, tmp_bf16_dir: Path, tmp_int3_dir: Path
    ):
        # The test manifest doesn't include compress_seconds
        result = measure_compression(tmp_bf16_dir, tmp_int3_dir)
        assert result.compress_seconds is None


# ─────────────────────────────────────────────────────────────────────────────
# measure_snr
# ─────────────────────────────────────────────────────────────────────────────

class TestMeasureSnr:
    def test_returns_snr_metrics(self, tmp_bf16_dir: Path, tmp_int3_dir: Path):
        """measure_snr needs safetensors+torch — skip if not available."""
        pytest.importorskip("safetensors")
        pytest.importorskip("torch")
        import torch
        from safetensors.torch import save_file

        shape = (64, 64)
        rng = np.random.default_rng(99)
        w = rng.standard_normal(shape).astype(np.float32)
        import torch as _torch
        save_file(
            {"model.layers.0.self_attn.q_proj.weight": _torch.from_numpy(w).to(_torch.bfloat16)},
            str(tmp_bf16_dir / "model-00001.safetensors"),
        )

        result = measure_snr(tmp_bf16_dir, tmp_int3_dir, max_layers=1, group_size=64)
        assert isinstance(result, SNRMetrics)
        assert result.layers_tested >= 0

    def test_error_if_safetensors_missing(self, tmp_bf16_dir, tmp_int3_dir):
        with patch.dict(sys.modules, {"safetensors": None, "safetensors.torch": None}):
            # Remove cached import entries
            for mod in list(sys.modules):
                if "safetensors" in mod:
                    sys.modules.pop(mod, None)
            # Now re-patch builtins.__import__ to raise ImportError for safetensors
            _orig_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__

            def _fake_import(name, *args, **kwargs):
                if name.startswith("safetensors"):
                    raise ImportError("mocked absence")
                return _orig_import(name, *args, **kwargs)

            import builtins
            with patch.object(builtins, "__import__", side_effect=_fake_import):
                result = measure_snr(tmp_bf16_dir, tmp_int3_dir)
        assert result.error is not None

    def test_error_if_manifest_missing(self, tmp_bf16_dir: Path, tmp_path: Path):
        pytest.importorskip("safetensors")
        pytest.importorskip("torch")  # measure_snr imports safetensors.torch before checking manifest
        no_manifest_dir = tmp_path / "int3_no_manifest"
        no_manifest_dir.mkdir()
        result = measure_snr(tmp_bf16_dir, no_manifest_dir)
        assert result.error is not None
        assert "manifest" in result.error.lower()


# ─────────────────────────────────────────────────────────────────────────────
# _ensure_mlx_int4
# ─────────────────────────────────────────────────────────────────────────────

class TestEnsureMlxInt4:
    def test_returns_cached_path_if_safetensors_present(self, tmp_path: Path):
        cache = tmp_path / "mlx_int4"
        cache.mkdir()
        (cache / "model.safetensors").write_bytes(b"\x00" * 16)
        result = _ensure_mlx_int4(tmp_path / "bf16", cache)
        assert result == cache

    def test_returns_bf16_if_mlx_lm_missing(self, tmp_path: Path):
        cache = tmp_path / "mlx_int4"
        bf16 = tmp_path / "bf16"
        bf16.mkdir()
        # subprocess.run raises FileNotFoundError simulating missing mlx_lm
        with patch("subprocess.run", side_effect=FileNotFoundError):
            result = _ensure_mlx_int4(bf16, cache)
        assert result == bf16

    def test_returns_bf16_on_timeout(self, tmp_path: Path):
        import subprocess
        cache = tmp_path / "mlx_int4"
        bf16 = tmp_path / "bf16"
        bf16.mkdir()
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("cmd", 1.0)):
            result = _ensure_mlx_int4(bf16, cache)
        assert result == bf16

    def test_returns_bf16_if_convert_fails(self, tmp_path: Path):
        cache = tmp_path / "mlx_int4"
        bf16 = tmp_path / "bf16"
        bf16.mkdir()
        mock_proc = MagicMock()
        mock_proc.returncode = 1
        with patch("subprocess.run", return_value=mock_proc):
            result = _ensure_mlx_int4(bf16, cache)
        assert result == bf16


# ─────────────────────────────────────────────────────────────────────────────
# measure_throughput
# ─────────────────────────────────────────────────────────────────────────────

class TestMeasureThroughput:
    def test_error_if_mlx_lm_missing(self, tmp_bf16_dir: Path):
        with patch.dict(sys.modules, {"mlx_lm": None}):
            for mod in list(sys.modules):
                if "mlx_lm" in mod:
                    sys.modules.pop(mod, None)
            import builtins
            _orig = builtins.__import__

            def _no_mlx_lm(name, *args, **kwargs):
                if name == "mlx_lm":
                    raise ImportError("mocked")
                return _orig(name, *args, **kwargs)

            with patch.object(builtins, "__import__", side_effect=_no_mlx_lm):
                result = measure_throughput(tmp_bf16_dir, runs=1)
        assert result.error is not None
        assert "mlx_lm" in result.error

    def test_returns_error_if_load_fails(self, tmp_bf16_dir: Path):
        mlx_lm_mock = MagicMock()
        mlx_lm_mock.load.side_effect = RuntimeError("oom")
        with patch.dict(sys.modules, {"mlx_lm": mlx_lm_mock}), \
             patch("dev.benchmarks.bench_qwen3_14b_int3._dir_gb", return_value=1.0), \
             patch("dev.benchmarks.bench_qwen3_14b_int3._detect_ram_gb", return_value=16.0):
            result = measure_throughput(tmp_bf16_dir, runs=1)
        assert result.error is not None

    def test_returns_dataclass_on_success(self, tmp_bf16_dir: Path):
        mlx_lm_mock = MagicMock()
        model_mock = MagicMock()
        tokenizer_mock = MagicMock()
        mlx_lm_mock.load.return_value = (model_mock, tokenizer_mock)
        # stream_generate yields 10 chunks
        mlx_lm_mock.stream_generate.return_value = iter([MagicMock()] * 10)

        with patch.dict(sys.modules, {"mlx_lm": mlx_lm_mock}), \
             patch("dev.benchmarks.bench_qwen3_14b_int3._dir_gb", return_value=1.0), \
             patch("dev.benchmarks.bench_qwen3_14b_int3._detect_ram_gb", return_value=64.0):
            result = measure_throughput(tmp_bf16_dir, runs=1)

        assert isinstance(result, ThroughputMetrics)
        assert result.n_runs == 1
        assert result.tps_mean > 0.0


# ─────────────────────────────────────────────────────────────────────────────
# measure_perplexity
# ─────────────────────────────────────────────────────────────────────────────

class TestMeasurePerplexity:
    def test_error_if_mlx_missing(self, tmp_bf16_dir: Path):
        import builtins
        _orig = builtins.__import__

        def _no_mlx(name, *args, **kwargs):
            if name in ("mlx.core", "mlx_lm"):
                raise ImportError("mocked")
            return _orig(name, *args, **kwargs)

        with patch.object(builtins, "__import__", side_effect=_no_mlx):
            result = measure_perplexity(tmp_bf16_dir)
        assert result.error is not None

    def test_error_if_load_fails(self, tmp_bf16_dir: Path):
        mlx_mock = MagicMock()
        mlx_lm_mock = MagicMock()
        mlx_lm_mock.load.side_effect = RuntimeError("oom")
        with patch.dict(sys.modules, {"mlx": mlx_mock, "mlx.core": MagicMock(), "mlx_lm": mlx_lm_mock}), \
             patch("dev.benchmarks.bench_qwen3_14b_int3._dir_gb", return_value=1.0), \
             patch("dev.benchmarks.bench_qwen3_14b_int3._detect_ram_gb", return_value=64.0):
            result = measure_perplexity(tmp_bf16_dir)
        assert result.error is not None

    def test_returns_dataclass_type(self, tmp_bf16_dir: Path):
        result = PerplexityMetrics(ppl=12.3, n_tokens=511)
        assert isinstance(result, PerplexityMetrics)
        assert result.ppl == 12.3


# ─────────────────────────────────────────────────────────────────────────────
# print_markdown
# ─────────────────────────────────────────────────────────────────────────────

class TestPrintMarkdown:
    def _full_results(self) -> BenchResults:
        return BenchResults(
            compression=CompressionMetrics(
                bf16_gb=28.5, int3_gb=7.7, ratio=0.27,
                size_savings_pct=73.0, actual_bpw=4.37,
                n_int3_tensors=224, n_passthrough_tensors=8,
                compress_seconds=1800.0,
            ),
            snr=SNRMetrics(
                layers_tested=24, snr_mean_db=13.2,
                snr_min_db=8.1, snr_max_db=21.4, snr_p25_db=10.3,
            ),
            throughput=ThroughputMetrics(
                tps_mean=24.5, tps_stdev=0.8,
                tps_min=23.4, tps_max=25.6,
                ttft_mean_ms=120.0, n_runs=3,
                model_path_used="/models/Qwen3-14B-mlx-int4",
            ),
            perplexity=PerplexityMetrics(ppl=9.87, n_tokens=511),
        )

    def test_prints_model_header(self, capsys):
        print_markdown(self._full_results())
        out = capsys.readouterr().out
        assert "Qwen3-14B" in out

    def test_prints_compression_table(self, capsys):
        print_markdown(self._full_results())
        out = capsys.readouterr().out
        assert "28.5" in out  # bf16_gb
        assert "7.7" in out   # int3_gb

    def test_prints_snr_table(self, capsys):
        print_markdown(self._full_results())
        out = capsys.readouterr().out
        assert "13.2" in out   # mean SNR

    def test_prints_throughput_table(self, capsys):
        print_markdown(self._full_results())
        out = capsys.readouterr().out
        assert "24.5" in out   # tps_mean

    def test_prints_perplexity_table(self, capsys):
        print_markdown(self._full_results())
        out = capsys.readouterr().out
        assert "9.87" in out   # ppl

    def test_no_snr_section_on_error(self, capsys):
        results = self._full_results()
        results.snr = SNRMetrics(
            layers_tested=0, snr_mean_db=0, snr_min_db=0,
            snr_max_db=0, snr_p25_db=0, error="import error"
        )
        print_markdown(results)
        out = capsys.readouterr().out
        # SNR section should not appear
        assert "S1" not in out

    def test_compression_error_suppresses_section(self, capsys):
        results = self._full_results()
        results.compression = CompressionMetrics(
            bf16_gb=0, int3_gb=0, ratio=0, size_savings_pct=0,
            actual_bpw=0, n_int3_tensors=0, n_passthrough_tensors=0,
            compress_seconds=None, error="no dir"
        )
        print_markdown(results)
        out = capsys.readouterr().out
        assert "S0" not in out


# ─────────────────────────────────────────────────────────────────────────────
# BenchResults dataclass
# ─────────────────────────────────────────────────────────────────────────────

class TestBenchResults:
    def test_default_model_id(self):
        r = BenchResults()
        assert r.model_id == "Qwen3-14B"

    def test_timestamp_auto_set(self):
        r = BenchResults()
        assert len(r.timestamp) > 0

    def test_platform_info_defaults_empty(self):
        r = BenchResults()
        assert isinstance(r.platform_info, dict)

    def test_asdict_serializable(self):
        r = BenchResults(
            compression=CompressionMetrics(
                bf16_gb=28.0, int3_gb=7.7, ratio=0.275,
                size_savings_pct=72.5, actual_bpw=4.4,
                n_int3_tensors=224, n_passthrough_tensors=8,
                compress_seconds=None,
            )
        )
        d = asdict(r)
        assert d["compression"]["bf16_gb"] == 28.0
        # Should be JSON-serializable
        _ = json.dumps(d)


# ─────────────────────────────────────────────────────────────────────────────
# SNRMetrics
# ─────────────────────────────────────────────────────────────────────────────

class TestSNRMetrics:
    def test_layer_results_default_empty(self):
        m = SNRMetrics(
            layers_tested=0, snr_mean_db=0, snr_min_db=0,
            snr_max_db=0, snr_p25_db=0
        )
        assert m.layer_results == []

    def test_layer_result_fields(self):
        lr = SNRLayerResult(name="w", shape=(32, 64), snr_db=12.5, time_ms=3.2)
        assert lr.snr_db == 12.5
        assert lr.shape == (32, 64)


# ─────────────────────────────────────────────────────────────────────────────
# main() — argument parsing and exit codes
# ─────────────────────────────────────────────────────────────────────────────

class TestMain:
    def test_exits_1_if_bf16_dir_missing(self, tmp_path: Path):
        rc = main([
            "--bf16-dir", str(tmp_path / "nonexistent"),
            "--int3-dir", str(tmp_path / "also_missing"),
            "--no-snr",
        ])
        assert rc == 1

    def test_exits_1_if_int3_dir_missing_no_wait(self, tmp_bf16_dir: Path, tmp_path: Path):
        rc = main([
            "--bf16-dir", str(tmp_bf16_dir),
            "--int3-dir", str(tmp_path / "nonexistent"),
            "--no-snr",
        ])
        assert rc == 1

    def test_compression_only_succeeds(
        self, tmp_bf16_dir: Path, tmp_int3_dir: Path, tmp_path: Path
    ):
        out_dir = tmp_path / "out"
        rc = main([
            "--bf16-dir",    str(tmp_bf16_dir),
            "--int3-dir",    str(tmp_int3_dir),
            "--no-snr",
            "--output-dir",  str(out_dir),
        ])
        assert rc == 0
        assert (out_dir / "qwen3_14b_int3_bench.json").exists()

    def test_output_json_valid(
        self, tmp_bf16_dir: Path, tmp_int3_dir: Path, tmp_path: Path
    ):
        out_dir = tmp_path / "out"
        main([
            "--bf16-dir",   str(tmp_bf16_dir),
            "--int3-dir",   str(tmp_int3_dir),
            "--no-snr",
            "--output-dir", str(out_dir),
        ])
        with open(out_dir / "qwen3_14b_int3_bench.json") as f:
            data = json.load(f)
        assert "compression" in data
        assert data["model_id"] == "Qwen3-14B"

    def test_markdown_flag_produces_output(
        self, tmp_bf16_dir: Path, tmp_int3_dir: Path, tmp_path: Path, capsys
    ):
        out_dir = tmp_path / "out"
        main([
            "--bf16-dir",   str(tmp_bf16_dir),
            "--int3-dir",   str(tmp_int3_dir),
            "--no-snr",
            "--output-dir", str(out_dir),
            "--markdown",
        ])
        out = capsys.readouterr().out
        # Markdown header should be in output
        assert "Qwen3-14B" in out

    def test_warn_if_manifest_missing(
        self, tmp_bf16_dir: Path, tmp_path: Path, capsys
    ):
        no_manifest = tmp_path / "int3_no_manifest"
        no_manifest.mkdir()
        # Create enough files so the dir isn't empty but no manifest
        (no_manifest / "tensors").mkdir()
        out_dir = tmp_path / "out"
        rc = main([
            "--bf16-dir",   str(tmp_bf16_dir),
            "--int3-dir",   str(no_manifest),
            "--no-snr",
            "--output-dir", str(out_dir),
        ])
        assert rc == 0  # should warn but not abort
        out = capsys.readouterr().out
        assert "manifest" in out.lower() or "warn" in out.lower()
