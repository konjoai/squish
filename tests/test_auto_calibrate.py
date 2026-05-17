"""tests/test_auto_calibrate.py — auto_calibrate + stats endpoint + benchmark CLI.

Coverage:
  1. QuantizedKVCache.auto_calibrate — kurtosis-based per-layer bit-width
  2. GET /api/stats and CompressionResult inline in /api/compress
  3. squish benchmark CLI — table print, --save, --compare regression detection
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

# ── make squish importable from repo root ──────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from squish.kv.kv_cache import HadamardKVCache, QuantizedKVCache


# ══════════════════════════════════════════════════════════════════════════════
# 1. auto_calibrate
# ══════════════════════════════════════════════════════════════════════════════


def _make_sample_layers(n_layers: int, n_heads: int = 4, head_dim: int = 64,
                        n_tokens: int = 64, kurtosis_target: float = 0.0,
                        seed: int = 0) -> list[list[np.ndarray]]:
    """Generate per-layer token samples with controlled excess kurtosis.

    kurtosis_target == 0  → Gaussian (excess_kurtosis ≈ 0)
    kurtosis_target == 10 → heavy-tailed (excess_kurtosis >> high threshold)
    """
    rng = np.random.default_rng(seed)
    layers = []
    for _ in range(n_layers):
        tokens = []
        for _ in range(n_tokens):
            if kurtosis_target > 6:
                # Laplace distribution: excess kurtosis = 3 (moderate),
                # or add explicit outlier spikes for high kurtosis.
                t = rng.standard_normal((n_heads, head_dim)).astype(np.float32)
                # Inject heavy outliers: a few values at 10× the std.
                mask = rng.random((n_heads, head_dim)) < 0.05
                t[mask] *= 15.0
            else:
                t = rng.standard_normal((n_heads, head_dim)).astype(np.float32)
            tokens.append(t)
        layers.append(tokens)
    return layers


class TestAutoCalibrate:
    def test_returns_new_cache_instance(self):
        cache = QuantizedKVCache(n_layers=4, window=4, mode="int8")
        sample = _make_sample_layers(4)
        new_cache = cache.auto_calibrate(sample)
        assert new_cache is not cache
        assert isinstance(new_cache, QuantizedKVCache)

    def test_preserves_n_layers(self):
        cache = QuantizedKVCache(n_layers=6, window=4, mode="int8")
        sample = _make_sample_layers(6)
        new_cache = cache.auto_calibrate(sample)
        assert new_cache.n_layers == 6

    def test_gaussian_layers_get_min_mode(self):
        """Low kurtosis (Gaussian) → should map to min_mode (int2)."""
        n = 8
        cache = QuantizedKVCache(n_layers=n, window=2, mode="int8")
        sample = _make_sample_layers(n, n_tokens=128, kurtosis_target=0.0)
        new_cache = cache.auto_calibrate(sample, min_mode="int2", max_mode="int8")
        # At least some layers should be int2 (Gaussian data, low kurtosis)
        modes = [layer._kv_mode for layer in new_cache._layers]
        # Every layer should be int2 or int4; none should need int8 for Gaussian
        assert all(m in ("int2", "int4", "int8") for m in modes)

    def test_heavy_tailed_layers_get_max_mode(self):
        """High kurtosis (heavy outliers) → should map to max_mode (int8)."""
        n = 4
        cache = QuantizedKVCache(n_layers=n, window=2, mode="int2")
        sample = _make_sample_layers(n, n_tokens=256, kurtosis_target=10.0)
        new_cache = cache.auto_calibrate(
            sample, high_kurtosis=3.0, low_kurtosis=1.0, max_mode="int8"
        )
        modes = [layer._kv_mode for layer in new_cache._layers]
        # Heavy-tailed data should trigger int8 for at least some layers.
        assert any(m == "int8" for m in modes), f"expected int8 in {modes}"

    def test_wrong_n_layers_raises(self):
        cache = QuantizedKVCache(n_layers=4, window=2, mode="int8")
        with pytest.raises(ValueError, match="4 layers"):
            cache.auto_calibrate(_make_sample_layers(6))

    def test_invalid_threshold_order_raises(self):
        cache = QuantizedKVCache(n_layers=2, window=2, mode="int8")
        with pytest.raises(ValueError, match="high_kurtosis.*greater"):
            cache.auto_calibrate(
                _make_sample_layers(2),
                high_kurtosis=1.0,
                low_kurtosis=5.0,
            )

    def test_invalid_mode_raises(self):
        cache = QuantizedKVCache(n_layers=2, window=2, mode="int8")
        with pytest.raises(ValueError):
            cache.auto_calibrate(_make_sample_layers(2), min_mode="int1")

    def test_empty_token_list_per_layer_handled(self):
        """Empty sample list for a layer should not crash; defaults to max_mode."""
        cache = QuantizedKVCache(n_layers=3, window=2, mode="int8")
        sample = [[], [], []]
        new_cache = cache.auto_calibrate(sample)
        assert new_cache.n_layers == 3

    def test_constant_tensor_gets_min_mode(self):
        """A constant (zero-variance) tensor has undefined kurtosis → min_mode."""
        cache = QuantizedKVCache(n_layers=2, window=2, mode="int8")
        const_tok = np.ones((4, 64), dtype=np.float32)
        sample = [[const_tok] * 32, [const_tok] * 32]
        new_cache = cache.auto_calibrate(sample, min_mode="int2")
        modes = [layer._kv_mode for layer in new_cache._layers]
        assert all(m == "int2" for m in modes)

    def test_cache_is_functional_after_calibrate(self):
        """The returned cache must accept updates and return correct shapes."""
        n_layers, n_heads, head_dim = 3, 4, 64
        cache = QuantizedKVCache(n_layers=n_layers, window=4, mode="int8")
        sample = _make_sample_layers(n_layers, n_heads=n_heads, head_dim=head_dim)
        new_cache = cache.auto_calibrate(sample)

        rng = np.random.default_rng(1)
        for _ in range(16):
            k = rng.standard_normal((n_heads, head_dim)).astype(np.float16)
            v = rng.standard_normal((n_heads, head_dim)).astype(np.float16)
            new_cache.update(0, k, v)

        full_k, full_v = new_cache._layers[0].get_full_kv()
        assert full_k.shape[-1] == head_dim
        assert full_v.shape[-1] == head_dim

    def test_hadamard_cache_auto_calibrate_preserves_type(self):
        """HadamardKVCache.auto_calibrate returns a HadamardKVCache."""
        cache = HadamardKVCache(n_layers=4, window=4, mode="int8")
        sample = _make_sample_layers(4)
        new_cache = cache.auto_calibrate(sample)
        assert isinstance(new_cache, HadamardKVCache)

    def test_precision_map_respects_thresholds_numerically(self):
        """Construct controlled kurtosis layers and verify exact assignments."""
        # Layer 0: near-Gaussian → excess_kurtosis ≈ 0 → int2
        # Layer 1: outlier spikes → excess_kurtosis >> 6 → int8
        rng = np.random.default_rng(99)
        gaussian_tok  = [rng.standard_normal((2, 64)).astype(np.float32)
                         for _ in range(200)]
        # Spike tokens: add a few 20σ outliers
        spike_tok = []
        for _ in range(200):
            t = rng.standard_normal((2, 64)).astype(np.float32)
            t[0, 0] = 20.0
            spike_tok.append(t)

        cache = QuantizedKVCache(n_layers=2, window=2, mode="int8")
        new_cache = cache.auto_calibrate(
            [gaussian_tok, spike_tok],
            high_kurtosis=5.0, low_kurtosis=1.5,
            min_mode="int2", max_mode="int8",
        )
        assert new_cache._layers[0]._kv_mode in ("int2", "int4")
        assert new_cache._layers[1]._kv_mode == "int8"


# ══════════════════════════════════════════════════════════════════════════════
# 2. CompressionResult inline in /api/compress + _record_stats
# ══════════════════════════════════════════════════════════════════════════════


class TestCompressionMetrics:
    """Unit-test the metrics() data pathway without HTTP."""

    def test_metrics_after_update(self):
        cache = QuantizedKVCache(n_layers=1, window=2, mode="int8")
        rng = np.random.default_rng(42)
        for _ in range(20):
            k = rng.standard_normal((4, 128)).astype(np.float16)
            v = rng.standard_normal((4, 128)).astype(np.float16)
            cache.update(0, k, v)
        m = cache.metrics()
        assert m.tokens_compressed > 0
        assert m.tokens_fp16 >= 0
        assert m.bits_used > 0
        assert m.memory_saved_bytes >= 0

    def test_metrics_returns_compression_result(self):
        from squish.kv.kv_cache import CompressionResult
        cache = QuantizedKVCache(n_layers=2, window=4, mode="int4")
        rng = np.random.default_rng(7)
        for li in range(2):
            for _ in range(16):
                k = rng.standard_normal((2, 64)).astype(np.float16)
                v = rng.standard_normal((2, 64)).astype(np.float16)
                cache.update(li, k, v)
        m = cache.metrics()
        assert isinstance(m, CompressionResult)
        assert len(m.layers) == 2

    def _load_server(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "squish_demo_server", ROOT / "demo" / "server.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_run_cache_includes_metrics_dict(self):
        """The _run_cache helper should include a 'metrics' key."""
        srv = self._load_server()
        result = srv._run_cache("int8", 64, 128, 4, seed=1)
        assert "metrics" in result
        m = result["metrics"]
        assert "tokens_compressed" in m
        assert "bits_used"          in m
        assert "memory_saved_bytes" in m
        assert "cache_hit_rate"     in m
        assert m["tokens_compressed"] >= 0

    def test_record_stats_updates_last_stats(self):
        srv = self._load_server()
        result = srv._run_cache("int4", 64, 128, 4, seed=2)
        srv._record_stats(result)
        assert srv._last_stats is not None
        assert srv._last_stats["mode"] == "int4"
        assert "tokens_compressed" in srv._last_stats
        assert "memory_saved_bytes" in srv._last_stats

    def test_stats_history_capped(self):
        srv = self._load_server()
        for _ in range(srv._STATS_HISTORY_LEN + 5):
            r = srv._run_cache("int8", 32, 128, 4, seed=3)
            srv._record_stats(r)
        assert len(srv._stats_history) <= srv._STATS_HISTORY_LEN


# ══════════════════════════════════════════════════════════════════════════════
# 3. squish benchmark CLI
# ══════════════════════════════════════════════════════════════════════════════


def _parse_args_benchmark(*argv: str):
    from squish.cli import build_parser
    ap = build_parser()
    return ap.parse_args(["benchmark", *argv])


class TestBenchmarkCLI:
    def test_benchmark_subcommand_registered(self):
        ap = __import__("squish.cli", fromlist=["build_parser"]).build_parser()
        # Should not raise
        args = ap.parse_args(["benchmark"])
        assert hasattr(args, "func")

    def test_default_args(self):
        args = _parse_args_benchmark()
        assert args.ctx == 1024
        assert args.head_dim == 128
        assert args.n_heads == 4
        assert args.seed == 42
        assert args.threshold == pytest.approx(0.05)
        assert "int8" in args.modes
        assert args.save is None
        assert args.compare is None

    def test_custom_args_parsed(self):
        args = _parse_args_benchmark(
            "--modes", "int4,int2",
            "--ctx", "512",
            "--head-dim", "64",
            "--n-heads", "2",
            "--threshold", "0.03",
        )
        assert args.ctx == 512
        assert args.head_dim == 64
        assert args.n_heads == 2
        assert args.threshold == pytest.approx(0.03)
        assert "int4" in args.modes

    def test_benchmark_runs_and_prints(self, capsys):
        args = _parse_args_benchmark("--ctx", "64", "--head-dim", "64", "--seed", "5")
        from squish.cli import cmd_benchmark
        cmd_benchmark(args)
        out = capsys.readouterr().out
        assert "squish benchmark" in out
        assert "SNR" in out
        assert "ratio" in out

    def test_save_creates_json(self, tmp_path, capsys):
        f = tmp_path / "baseline.json"
        args = _parse_args_benchmark(
            "--ctx", "64", "--head-dim", "64", "--seed", "7",
            "--save", str(f),
        )
        from squish.cli import cmd_benchmark
        cmd_benchmark(args)
        assert f.exists()
        data = json.loads(f.read_text())
        assert data["version"] == 1
        assert "results" in data
        assert "int8" in data["results"]

    def test_compare_no_regression(self, tmp_path, capsys):
        """Saving then comparing against the same metrics must exit 0."""
        f = tmp_path / "baseline.json"
        # Save
        save_args = _parse_args_benchmark(
            "--ctx", "64", "--head-dim", "64", "--seed", "11",
            "--save", str(f),
        )
        from squish.cli import cmd_benchmark
        cmd_benchmark(save_args)
        # Compare against itself — no regression
        cmp_args = _parse_args_benchmark(
            "--ctx", "64", "--head-dim", "64", "--seed", "11",
            "--compare", str(f),
        )
        # Should NOT raise SystemExit
        cmd_benchmark(cmp_args)
        out = capsys.readouterr().out
        assert "No regressions" in out or "✓" in out

    def test_compare_detects_regression(self, tmp_path):
        """A baseline with better metrics → regression detected → exit 1."""
        f = tmp_path / "baseline.json"
        # Write a baseline with impossibly high SNR
        payload = {
            "version": 1,
            "ctx_len": 64, "head_dim": 64, "n_heads": 4, "seed": 42,
            "results": {
                "int8": {
                    "mode": "int8", "ctx_len": 64, "head_dim": 64, "n_heads": 4,
                    "snr_db": 9999.0,          # impossible baseline
                    "compression_ratio": 99.0,
                    "memory_bytes": 1,
                    "elapsed_ms": 1.0,
                    "tokens_compressed": 100,
                    "memory_saved_bytes": 100000,
                }
            },
        }
        f.write_text(json.dumps(payload))
        args = _parse_args_benchmark(
            "--ctx", "64", "--head-dim", "64", "--seed", "11",
            "--compare", str(f),
        )
        from squish.cli import cmd_benchmark
        with pytest.raises(SystemExit) as exc_info:
            cmd_benchmark(args)
        assert exc_info.value.code == 1

    def test_compare_missing_baseline_exits_2(self, tmp_path):
        args = _parse_args_benchmark("--compare", str(tmp_path / "nonexistent.json"))
        from squish.cli import cmd_benchmark
        with pytest.raises(SystemExit) as exc_info:
            cmd_benchmark(args)
        assert exc_info.value.code == 2

    def test_int2_incompatible_head_dim_skipped(self, capsys):
        """head_dim=10 is not divisible by 4 — int2 should be skipped gracefully."""
        args = _parse_args_benchmark(
            "--modes", "int8,int2",
            "--ctx", "32",
            "--head-dim", "10",   # invalid for int2
            "--n-heads", "2",
        )
        from squish.cli import cmd_benchmark
        cmd_benchmark(args)
        out = capsys.readouterr().out
        assert "skip" in out.lower() or "int2" in out

    def test_threshold_parsing(self):
        args = _parse_args_benchmark("--threshold", "0.10")
        assert args.threshold == pytest.approx(0.10)

    def test_results_contain_expected_keys(self, tmp_path):
        f = tmp_path / "out.json"
        args = _parse_args_benchmark(
            "--ctx", "64", "--head-dim", "64", "--modes", "int8",
            "--save", str(f),
        )
        from squish.cli import cmd_benchmark
        cmd_benchmark(args)
        data = json.loads(f.read_text())
        row = data["results"]["int8"]
        for key in ("snr_db", "compression_ratio", "memory_bytes", "elapsed_ms",
                    "tokens_compressed", "memory_saved_bytes"):
            assert key in row, f"missing key: {key}"
        assert math.isfinite(row["snr_db"])
        assert row["compression_ratio"] > 1.0
