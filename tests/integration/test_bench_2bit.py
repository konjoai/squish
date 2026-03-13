"""tests/test_bench_2bit.py — 100% coverage for dev/benchmarks/bench_2bit.py"""
from __future__ import annotations

import importlib
import importlib.util
import io
import json
import math
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Ensure the module under test is importable even though it lives in dev/.
# Register it in sys.modules *before* exec_module so that Python 3.14+'s
# dataclass() decorator can resolve forward references via sys.modules.
_BENCH_PATH = Path(__file__).resolve().parents[2] / "dev" / "benchmarks" / "bench_2bit.py"
_SPEC = importlib.util.spec_from_file_location("bench_2bit", _BENCH_PATH)
_MOD  = importlib.util.module_from_spec(_SPEC)          # type: ignore[arg-type]
sys.modules["bench_2bit"] = _MOD
_SPEC.loader.exec_module(_MOD)                          # type: ignore[union-attr]
bench_2bit = _MOD

RNG = np.random.default_rng(99)

# Tiny weight matrix reused across tests — keep it fast.
_W_SMALL = RNG.standard_normal((8, 64)).astype(np.float32) * 0.02


# ── _int4_quantize_np / _int4_dequantize_np ──────────────────────────────────

class TestInt4NpRoundTrip:
    def test_output_shapes(self):
        rows, cols = 4, 64
        W = RNG.standard_normal((rows, cols)).astype(np.float32)
        packed, scales, zp = bench_2bit._int4_quantize_np(W, group_size=64)
        assert packed.shape == (rows, cols // 2)
        assert scales.shape == (rows, 1)
        assert zp.shape    == (rows, 1)

    def test_packed_dtype(self):
        W = RNG.standard_normal((4, 64)).astype(np.float32)
        packed, _, _ = bench_2bit._int4_quantize_np(W, group_size=64)
        assert packed.dtype == np.uint8

    def test_values_in_range(self):
        W = RNG.standard_normal((4, 64)).astype(np.float32) * 5.0
        packed, scales, zp = bench_2bit._int4_quantize_np(W, group_size=64)
        # Unpack manually and check nibbles are in [0, 15].
        lo = packed & 0x0F
        hi = (packed >> 4) & 0x0F
        assert lo.max() <= 15 and lo.min() >= 0
        assert hi.max() <= 15 and hi.min() >= 0

    def test_reconstruction_close_on_uniform(self):
        # Uniform weight matrix → quantisation error ≈ step_size / 2.
        W = np.full((4, 64), 0.5, dtype=np.float32)
        packed, scales, zp = bench_2bit._int4_quantize_np(W, group_size=64)
        W_hat = bench_2bit._int4_dequantize_np(packed, scales, zp, 64, 64)
        np.testing.assert_allclose(W_hat, W, atol=1e-5)

    def test_reconstruction_snr_positive(self):
        W = RNG.standard_normal((16, 64)).astype(np.float32)
        packed, scales, zp = bench_2bit._int4_quantize_np(W, group_size=64)
        W_hat = bench_2bit._int4_dequantize_np(packed, scales, zp, 64, 64)
        snr = bench_2bit._snr_db(W, W_hat)
        assert snr > 0.0

    def test_padding_applied_when_cols_not_aligned(self):
        # 70 cols is not divisible by 64 → should pad and trim correctly.
        W = RNG.standard_normal((4, 70)).astype(np.float32)
        packed, scales, zp = bench_2bit._int4_quantize_np(W, group_size=64)
        W_hat = bench_2bit._int4_dequantize_np(packed, scales, zp, 64, 70)
        assert W_hat.shape == (4, 70)

    def test_dequantize_without_trim(self):
        # original_cols=None returns padded matrix.
        W = RNG.standard_normal((4, 64)).astype(np.float32)
        packed, scales, zp = bench_2bit._int4_quantize_np(W, group_size=64)
        W_hat = bench_2bit._int4_dequantize_np(packed, scales, zp, 64)
        # Shape should be (4, 64) — no trimming needed for aligned cols.
        assert W_hat.shape == (4, 64)

    def test_zero_weight_matrix(self):
        W = np.zeros((4, 64), dtype=np.float32)
        packed, scales, zp = bench_2bit._int4_quantize_np(W, group_size=64)
        W_hat = bench_2bit._int4_dequantize_np(packed, scales, zp, 64, 64)
        np.testing.assert_allclose(W_hat, W, atol=1e-6)

    def test_group_size_smaller_than_cols(self):
        W = RNG.standard_normal((2, 16)).astype(np.float32)
        packed, scales, zp = bench_2bit._int4_quantize_np(W, group_size=8)
        W_hat = bench_2bit._int4_dequantize_np(packed, scales, zp, 8, 16)
        assert W_hat.shape == (2, 16)


# ── _int4_bpw ────────────────────────────────────────────────────────────────

class TestInt4Bpw:
    def test_symmetric_group64(self):
        bpw = bench_2bit._int4_bpw(64, asymmetric=False)
        # 4 + 32/64 = 4.5
        assert bpw == pytest.approx(4.5)

    def test_asymmetric_group64(self):
        bpw = bench_2bit._int4_bpw(64, asymmetric=True)
        # 4 + 64/64 = 5.0
        assert bpw == pytest.approx(5.0)

    def test_asymmetric_group32(self):
        bpw = bench_2bit._int4_bpw(32, asymmetric=True)
        # 4 + 64/32 = 6.0
        assert bpw == pytest.approx(6.0)

    def test_symmetric_group128(self):
        bpw = bench_2bit._int4_bpw(128, asymmetric=False)
        # 4 + 32/128 = 4.25
        assert bpw == pytest.approx(4.25)


# ── _snr_db ───────────────────────────────────────────────────────────────────

class TestSnrDb:
    def test_perfect_reconstruction_is_inf(self):
        W = RNG.standard_normal((4, 8)).astype(np.float32)
        snr = bench_2bit._snr_db(W, W)
        assert math.isinf(snr) and snr > 0

    def test_high_noise_gives_low_snr(self):
        original = np.ones((8, 8), dtype=np.float32)
        noisy    = original + np.ones_like(original) * 100
        snr = bench_2bit._snr_db(original, noisy)
        assert snr < 0

    def test_typical_int4_range(self):
        W     = RNG.standard_normal((16, 64)).astype(np.float32)
        p, s, z = bench_2bit._int4_quantize_np(W, 64)
        W_hat = bench_2bit._int4_dequantize_np(p, s, z, 64, 64)
        snr = bench_2bit._snr_db(W, W_hat)
        # INT4 typically gives 15–50 dB depending on the input distribution.
        assert 5.0 < snr < 80.0

    def test_zero_signal_returns_neg_inf(self):
        W     = np.zeros((4, 8), dtype=np.float32)
        W_hat = np.ones((4, 8), dtype=np.float32)
        snr = bench_2bit._snr_db(W, W_hat)
        assert math.isinf(snr) and snr < 0

    def test_length_mismatch_trims_to_shorter(self):
        a = np.ones((2, 4), dtype=np.float32)
        b = np.ones((2, 8), dtype=np.float32)    # longer → trimmed
        snr = bench_2bit._snr_db(a, b)
        assert math.isinf(snr) and snr > 0       # first 8 elements match


# ── MethodResult + serialisation ────────────────────────────────────────────

class TestMethodResult:
    def test_default_fields(self):
        r = bench_2bit.MethodResult(status="ok")
        assert r.status        == "ok"
        assert r.reason        == ""
        assert r.bpw           is None
        assert r.snr_db        is None
        assert r.compress_ms   is None
        assert r.decompress_ms is None
        assert r.perplexity    is None
        assert r.tps           is None
        assert r.backend       == ""

    def test_full_construction(self):
        r = bench_2bit.MethodResult(
            status="ok", bpw=4.5, snr_db=35.0,
            compress_ms=0.5, decompress_ms=0.1,
            perplexity=7.32, tps=150.0, backend="rust",
        )
        assert r.bpw == pytest.approx(4.5)
        assert r.backend == "rust"

    def test_skip_result(self):
        r = bench_2bit.MethodResult(status="skip", reason="not implemented")
        assert r.status == "skip"
        assert "not implemented" in r.reason


class TestResultSerialization:
    def test_result_to_dict_ok(self):
        r = bench_2bit.MethodResult(status="ok", bpw=4.5, snr_db=35.0)
        d = bench_2bit._result_to_dict(r)
        assert d["status"] == "ok"
        assert d["bpw"] == pytest.approx(4.5)
        assert d["snr_db"] == pytest.approx(35.0)

    def test_result_to_dict_inf_becomes_none(self):
        r = bench_2bit.MethodResult(status="ok", snr_db=float("inf"))
        d = bench_2bit._result_to_dict(r)
        assert d["snr_db"] is None

    def test_result_to_dict_neg_inf_becomes_none(self):
        r = bench_2bit.MethodResult(status="ok", snr_db=float("-inf"))
        d = bench_2bit._result_to_dict(r)
        assert d["snr_db"] is None

    def test_result_to_dict_none_stays_none(self):
        r = bench_2bit.MethodResult(status="skip")
        d = bench_2bit._result_to_dict(r)
        assert d["bpw"] is None

    def test_dict_to_result_round_trip(self):
        r = bench_2bit.MethodResult(
            status="ok", bpw=1.75, snr_db=28.5,
            compress_ms=100.0, decompress_ms=0.05,
            backend="vptq-numpy",
        )
        d = bench_2bit._result_to_dict(r)
        r2 = bench_2bit._dict_to_result(d)
        assert r2.status     == r.status
        assert r2.bpw        == r.bpw
        assert r2.snr_db     == r.snr_db
        assert r2.backend    == r.backend

    def test_dict_to_result_missing_keys_use_defaults(self):
        r = bench_2bit._dict_to_result({"status": "skip"})
        assert r.reason == ""
        assert r.bpw    is None

    def test_dict_to_result_missing_status_defaults_to_error(self):
        r = bench_2bit._dict_to_result({})
        assert r.status == "error"


# ── bench_int4 ────────────────────────────────────────────────────────────────

class TestBenchInt4:
    def test_numpy_path_ok(self):
        """When squish_quant is absent the numpy fallback must succeed."""
        with patch.dict(sys.modules, {"squish_quant": None}):
            r = bench_2bit.bench_int4(_W_SMALL)
        assert r.status  == "ok"
        assert r.backend == "numpy"
        assert r.bpw     == pytest.approx(5.0)
        assert r.snr_db  is not None
        assert r.compress_ms   > 0
        assert r.decompress_ms > 0

    def test_numpy_path_snr_positive(self):
        with patch.dict(sys.modules, {"squish_quant": None}):
            r = bench_2bit.bench_int4(_W_SMALL)
        assert r.snr_db > 0

    def test_rust_path_when_available(self):
        """Mock a squish_quant module to exercise the Rust branch."""
        sq_mock = MagicMock()
        rows, cols = _W_SMALL.shape
        packed_mock = np.zeros((rows, cols // 2), dtype=np.uint8)
        scales_mock = np.ones((rows, cols // bench_2bit.INT4_GROUP_SIZE), dtype=np.float32)
        recon_mock  = np.zeros((rows, cols), dtype=np.float32)

        sq_mock.quantize_int4_grouped.return_value = (packed_mock, scales_mock)
        sq_mock.dequantize_int4_grouped.return_value = recon_mock

        with patch.dict(sys.modules, {"squish_quant": sq_mock}):
            r = bench_2bit.bench_int4(_W_SMALL)

        assert r.status  == "ok"
        assert r.backend == "rust"
        # Symmetric: 4 + 32/group_size
        assert r.bpw == pytest.approx(bench_2bit._int4_bpw(
            bench_2bit.INT4_GROUP_SIZE, asymmetric=False
        ))

    def test_exception_during_rust_falls_back_to_numpy(self):
        """If squish_quant raises during quantize, fall through to numpy."""
        sq_mock = MagicMock()
        sq_mock.quantize_int4_grouped.side_effect = RuntimeError("mock failure")

        with patch.dict(sys.modules, {"squish_quant": sq_mock}):
            r = bench_2bit.bench_int4(_W_SMALL)

        assert r.status  == "ok"
        assert r.backend == "numpy"


# ── bench_vptq ────────────────────────────────────────────────────────────────

class TestBenchVptq:
    def test_ok_with_small_weights(self):
        r = bench_2bit.bench_vptq(_W_SMALL)
        assert r.status  == "ok"
        assert r.backend == "vptq-numpy"
        assert r.bpw     is not None
        assert r.bpw     < bench_2bit._int4_bpw(bench_2bit.INT4_GROUP_SIZE)
        assert r.compress_ms   > 0
        assert r.decompress_ms > 0

    def test_bpw_is_sub_int4(self):
        r = bench_2bit.bench_vptq(_W_SMALL)
        # VPTQ at k=16 primary should compress below INT4 bpw.
        assert r.bpw < 5.0

    def test_snr_is_finite(self):
        r = bench_2bit.bench_vptq(_W_SMALL)
        assert r.snr_db is not None
        assert math.isfinite(r.snr_db)

    def test_skip_when_vptq_unavailable(self):
        # Temporarily hide squish.quant.vptq.
        with patch.dict(sys.modules, {"squish.quant.vptq": None}):
            r = bench_2bit.bench_vptq(_W_SMALL)
        assert r.status == "skip"

    def test_vptq_module_import_error_gives_skip(self):
        with patch("builtins.__import__", side_effect=ImportError("vptq gone")):
            r = bench_2bit.bench_vptq(_W_SMALL)
        assert r.status == "skip"


# ── bench_aqlm ────────────────────────────────────────────────────────────────

class TestBenchAqlm:
    def test_skip_when_module_missing(self):
        """squish.aqlm does not exist yet → always SKIP."""
        r = bench_2bit.bench_aqlm(_W_SMALL)
        assert r.status == "skip"
        assert "Phase 9A" in r.reason

    def test_ok_when_module_available(self):
        """Mock the AQLM module to test the execution path."""
        fake_layer = MagicMock()
        fake_layer.compressed_bits = 1024
        fake_quant = MagicMock()
        fake_quant.compress.return_value  = fake_layer
        fake_quant.decompress.return_value = np.zeros_like(_W_SMALL)

        fake_aqlm = types.ModuleType("squish.aqlm")
        fake_aqlm.AQLMConfig = MagicMock(return_value=MagicMock())
        fake_aqlm.AQLMQuantizer = MagicMock(return_value=fake_quant)

        with patch.dict(sys.modules, {"squish.aqlm": fake_aqlm}):
            r = bench_2bit.bench_aqlm(_W_SMALL)

        rows, cols = _W_SMALL.shape
        assert r.status == "ok"
        assert r.bpw    == pytest.approx(1024 / (rows * cols))


# ── bench_quip ────────────────────────────────────────────────────────────────

class TestBenchQuip:
    def test_ok_with_real_module(self):
        """squish.quant.quip_sharp is implemented (Phase 9B done) → result is ok."""
        r = bench_2bit.bench_quip(_W_SMALL)
        assert r.status  == "ok"
        assert r.backend == "quip-numpy"
        assert r.bpw     is not None
        assert r.compress_ms   > 0
        assert r.decompress_ms > 0

    def test_bpw_is_3(self):
        """BPW = (8 + 16) bits per 8-D group = 3.0 bpw (rotation excluded)."""
        r = bench_2bit.bench_quip(_W_SMALL)
        # index (8 bits) + scale (16 bits) per 8 weights = 3.0 bpw
        assert r.bpw == pytest.approx(3.0)

    def test_snr_finite(self):
        r = bench_2bit.bench_quip(_W_SMALL)
        assert r.snr_db is not None
        assert math.isfinite(r.snr_db)

    def test_skip_when_module_import_fails(self):
        """When squish.quant.quip_sharp raises ImportError, bench_quip returns SKIP."""
        with patch.dict(sys.modules, {"squish.quant.quip_sharp": None}):
            r = bench_2bit.bench_quip(_W_SMALL)
        assert r.status == "skip"

    def test_ok_when_module_mocked(self):
        """Exercise the execution path with a deterministic mock."""
        rows, cols = _W_SMALL.shape
        n_groups  = rows * cols // 8
        fake_layer = MagicMock()
        fake_layer.e8_indices     = np.zeros(n_groups, dtype=np.uint8)
        fake_layer.residual_scales = np.ones(n_groups, dtype=np.float16)
        fake_layer.config         = MagicMock()
        fake_layer.config.group_size = 8
        fake_layer.original_shape    = (rows, cols)

        fake_quant = MagicMock()
        fake_quant.quantize.return_value = fake_layer

        # quip_dequantize is a module-level function, not a method.
        fake_quip = types.ModuleType("squish.quant.quip_sharp")
        fake_quip.QuIPSharpConfig    = MagicMock(return_value=MagicMock())
        fake_quip.QuIPSharpQuantizer = MagicMock(return_value=fake_quant)
        fake_quip.quip_dequantize    = MagicMock(
            return_value=np.zeros_like(_W_SMALL)
        )

        with patch.dict(sys.modules, {"squish.quant.quip_sharp": fake_quip}):
            r = bench_2bit.bench_quip(_W_SMALL)

        assert r.status  == "ok"
        expected_bpw = (n_groups * 8 + n_groups * 16) / (rows * cols)
        assert r.bpw    == pytest.approx(expected_bpw)


# ── display helpers ───────────────────────────────────────────────────────────

class TestDisplayHelpers:
    def _capture(self, fn, *args, **kwargs):
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            fn(*args, **kwargs)
        return buf.getvalue()

    def test_hdr_contains_title(self):
        out = self._capture(bench_2bit._hdr, "My Title")
        assert "My Title" in out

    def test_row_contains_label_and_val(self):
        out = self._capture(bench_2bit._row, "Label", "42.0")
        assert "Label" in out
        assert "42.0" in out

    def test_row_with_extra(self):
        out = self._capture(bench_2bit._row, "Label", "val", "extra info")
        assert "extra info" in out

    def test_skip_contains_label(self):
        out = self._capture(bench_2bit._skip, "MyMethod", "reason here")
        assert "SKIP" in out
        assert "MyMethod" in out

    def test_err_contains_label(self):
        out = self._capture(bench_2bit._err, "MyMethod", "boom")
        assert "ERROR" in out
        assert "MyMethod" in out


# ── _print_table ─────────────────────────────────────────────────────────────

class TestPrintTable:
    def _capture_table(self, method_results):
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            bench_2bit._print_table(method_results)
        return buf.getvalue()

    def test_ok_method_appears(self):
        mr = {
            "int4": bench_2bit.MethodResult(
                status="ok", bpw=5.0, snr_db=21.0,
                compress_ms=0.5, decompress_ms=0.1
            ),
            "vptq": bench_2bit.MethodResult(status="skip", reason="nope"),
            "aqlm": bench_2bit.MethodResult(status="skip", reason="nope"),
            "quip": bench_2bit.MethodResult(status="skip", reason="nope"),
        }
        out = self._capture_table(mr)
        assert "INT4" in out and "5.00" in out

    def test_skip_appears_in_table(self):
        mr = {k: bench_2bit.MethodResult(status="skip", reason="x")
              for _, _, k_name in bench_2bit._METHODS
              for k, _, _ in [next((m for m in bench_2bit._METHODS if m[2] == k_name), bench_2bit._METHODS[0])]}
        # simpler: build by method key
        mr = {name: bench_2bit.MethodResult(status="skip", reason="x")
              for name, _, _ in bench_2bit._METHODS}
        out = self._capture_table(mr)
        assert "SKIP" in out

    def test_error_appears_in_table(self):
        mr = {"int4": bench_2bit.MethodResult(status="error", reason="failed"),
              "vptq": bench_2bit.MethodResult(status="skip"),
              "aqlm": bench_2bit.MethodResult(status="skip"),
              "quip": bench_2bit.MethodResult(status="skip")}
        out = self._capture_table(mr)
        assert "ERROR" in out

    def test_none_bpw_displays_dash(self):
        mr = {"int4": bench_2bit.MethodResult(status="ok", bpw=None),
              "vptq": bench_2bit.MethodResult(status="skip"),
              "aqlm": bench_2bit.MethodResult(status="skip"),
              "quip": bench_2bit.MethodResult(status="skip")}
        out = self._capture_table(mr)
        assert "—" in out

    def test_missing_method_key_ignored(self):
        # Only int4 present; no crash for missing keys.
        mr = {"int4": bench_2bit.MethodResult(status="ok", bpw=5.0)}
        out = self._capture_table(mr)
        assert "INT4" in out


class TestPrintMarkdownTable:
    def _capture(self, mr):
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            bench_2bit._print_markdown_table(mr)
        return buf.getvalue()

    def test_header_present(self):
        mr = {name: bench_2bit.MethodResult(status="skip")
              for name, _, _ in bench_2bit._METHODS}
        out = self._capture(mr)
        assert "| Method |" in out
        assert "|--------|" in out

    def test_ok_row_shows_values(self):
        mr = {
            "int4": bench_2bit.MethodResult(status="ok", bpw=5.0, snr_db=21.0,
                                            compress_ms=0.5, decompress_ms=0.1),
            "vptq": bench_2bit.MethodResult(status="skip"),
            "aqlm": bench_2bit.MethodResult(status="skip"),
            "quip": bench_2bit.MethodResult(status="skip"),
        }
        out = self._capture(mr)
        assert "5.00" in out

    def test_skip_shows_dash(self):
        mr = {name: bench_2bit.MethodResult(status="skip", reason="x")
              for name, _, _ in bench_2bit._METHODS}
        out = self._capture(mr)
        assert "| — |" in out

    def test_none_result_handled(self):
        mr = {"int4": None,
              "vptq": bench_2bit.MethodResult(status="skip"),
              "aqlm": bench_2bit.MethodResult(status="skip"),
              "quip": bench_2bit.MethodResult(status="skip")}
        # None result → should print dash row without crashing.
        out = self._capture(mr)
        assert "INT4 nibble" in out


# ── _load_wikitext ────────────────────────────────────────────────────────────

class TestLoadWikitext:
    def test_fallback_when_datasets_unavailable(self):
        with patch.dict(sys.modules, {"datasets": None}):
            text = bench_2bit._load_wikitext(100)
        assert isinstance(text, str)
        assert len(text) > 50

    def test_fallback_when_load_dataset_raises(self):
        mock_ds = MagicMock()
        mock_ds.load_dataset.side_effect = Exception("offline")
        with patch.dict(sys.modules, {"datasets": mock_ds}):
            text = bench_2bit._load_wikitext(100)
        assert text == bench_2bit._WIKITEXT_SAMPLE

    def test_uses_datasets_when_available(self):
        # Build a minimal fake dataset.
        fake_row = {"text": "The quick brown fox jumps over the lazy dog."}
        mock_ds  = MagicMock()
        mock_ds.load_dataset.return_value = [fake_row] * 5
        with patch.dict(sys.modules, {"datasets": mock_ds}):
            text = bench_2bit._load_wikitext(5)
        assert isinstance(text, str)
        assert len(text) > 0


# ── run_benchmark ─────────────────────────────────────────────────────────────

class TestRunBenchmark:
    def test_dry_run_returns_expected_keys(self):
        results, mr = bench_2bit.run_benchmark(dry_run=True)
        assert "meta"           in results
        assert "model_baseline" in results
        assert "methods"        in results

    def test_meta_fields_present(self):
        results, _ = bench_2bit.run_benchmark(dry_run=True)
        meta = results["meta"]
        assert "squish_version"  in meta
        assert "python_version"  in meta
        assert "platform"        in meta
        assert "weight_shape"    in meta
        assert "int4_group_size" in meta
        assert "vptq_config"     in meta

    def test_all_methods_present_in_result(self):
        results, mr = bench_2bit.run_benchmark(dry_run=True)
        for name, _, _ in bench_2bit._METHODS:
            assert name in results["methods"]

    def test_int4_status_ok(self):
        _, mr = bench_2bit.run_benchmark(dry_run=True)
        assert mr["int4"].status == "ok"

    def test_vptq_status_ok(self):
        _, mr = bench_2bit.run_benchmark(dry_run=True)
        assert mr["vptq"].status == "ok"

    def test_aqlm_status_skip(self):
        _, mr = bench_2bit.run_benchmark(dry_run=True)
        assert mr["aqlm"].status == "skip"

    def test_quip_status_ok(self):
        _, mr = bench_2bit.run_benchmark(dry_run=True)
        assert mr["quip"].status == "ok"

    def test_model_baseline_is_null_without_model_dir(self):
        results, _ = bench_2bit.run_benchmark(dry_run=True)
        assert results["model_baseline"]["perplexity"] is None
        assert results["model_baseline"]["tps"] is None

    def test_model_dir_with_dry_run_skips_model(self):
        results, _ = bench_2bit.run_benchmark(
            model_dir="/fake/model", dry_run=True
        )
        assert results["model_baseline"]["perplexity"] is None

    def test_model_dir_no_dry_run_but_mlx_unavailable(self, tmp_path):
        """When mlx_lm is absent, model eval skips gracefully."""
        with patch.dict(sys.modules, {"mlx_lm": None, "mlx": None, "mlx.core": None}):
            results, _ = bench_2bit.run_benchmark(
                model_dir=str(tmp_path), dry_run=False
            )
        assert results["model_baseline"]["perplexity"] is None

    def test_method_exception_gives_error_status(self):
        """If a bench function raises, run_benchmark catches and marks error."""
        def _raising_bench(W):
            raise RuntimeError("deliberate failure")

        saved = bench_2bit._METHODS[:]
        bench_2bit._METHODS[0] = (
            bench_2bit._METHODS[0][0],
            _raising_bench,
            bench_2bit._METHODS[0][2],
        )
        try:
            _, mr = bench_2bit.run_benchmark(dry_run=True)
            assert mr[bench_2bit._METHODS[0][0]].status == "error"
        finally:
            bench_2bit._METHODS[:] = saved

    def test_result_json_is_serialisable(self):
        results, _ = bench_2bit.run_benchmark(dry_run=True)
        serialised = json.dumps(results)
        restored   = json.loads(serialised)
        assert restored["meta"]["weight_shape"] == [
            bench_2bit.BENCH_ROWS, bench_2bit.BENCH_COLS
        ]


# ── main / CLI ────────────────────────────────────────────────────────────────

class TestMain:
    def test_dry_run_exit_zero(self, tmp_path):
        out_file = tmp_path / "results.json"
        rc = bench_2bit.main(["--dry-run", "--output", str(out_file)])
        assert rc == 0

    def test_json_written_to_output(self, tmp_path):
        out_file = tmp_path / "results.json"
        bench_2bit.main(["--dry-run", "--output", str(out_file)])
        assert out_file.exists()
        with out_file.open() as fh:
            data = json.load(fh)
        assert "methods" in data

    def test_markdown_flag_produces_table(self, tmp_path, capsys):
        out_file = tmp_path / "results.json"
        bench_2bit.main(["--dry-run", "--markdown", "--output", str(out_file)])
        captured = capsys.readouterr()
        assert "| Method |" in captured.out

    def test_default_output_path(self, tmp_path, monkeypatch):
        """DEFAULT_OUTPUT must be writable; monkeypatch to tmp so we don't pollute dev/results."""
        monkeypatch.setattr(bench_2bit, "DEFAULT_OUTPUT", tmp_path / "out.json")
        rc = bench_2bit.main(["--dry-run"])
        assert rc == 0

    def test_ppl_tokens_flag(self, tmp_path):
        out_file = tmp_path / "out.json"
        rc = bench_2bit.main(
            ["--dry-run", "--ppl-tokens", "512", "--output", str(out_file)]
        )
        assert rc == 0
        with out_file.open() as fh:
            data = json.load(fh)
        assert data["meta"]["ppl_max_tokens"] == 512

    def test_tps_tokens_flag(self, tmp_path):
        out_file = tmp_path / "out.json"
        rc = bench_2bit.main(
            ["--dry-run", "--tps-tokens", "64", "--output", str(out_file)]
        )
        assert rc == 0
        with out_file.open() as fh:
            data = json.load(fh)
        assert data["meta"]["tps_gen_tokens"] == 64

    def test_output_parent_created(self, tmp_path):
        deep_out = tmp_path / "nested" / "dir" / "out.json"
        rc = bench_2bit.main(["--dry-run", "--output", str(deep_out)])
        assert rc == 0
        assert deep_out.exists()


# ── _build_parser ────────────────────────────────────────────────────────────

class TestBuildParser:
    def test_defaults(self):
        p    = bench_2bit._build_parser()
        args = p.parse_args([])
        assert args.model_dir  is None
        assert args.ppl_tokens == bench_2bit.PPL_MAX_TOKENS
        assert args.tps_tokens == bench_2bit.TPS_GEN_TOKENS
        assert not args.dry_run
        assert not args.markdown

    def test_dry_run_flag(self):
        p    = bench_2bit._build_parser()
        args = p.parse_args(["--dry-run"])
        assert args.dry_run

    def test_markdown_flag(self):
        p    = bench_2bit._build_parser()
        args = p.parse_args(["--markdown"])
        assert args.markdown

    def test_model_dir(self):
        p    = bench_2bit._build_parser()
        args = p.parse_args(["--model-dir", "/some/model"])
        assert args.model_dir == "/some/model"

    def test_custom_ppl_tokens(self):
        p    = bench_2bit._build_parser()
        args = p.parse_args(["--ppl-tokens", "1024"])
        assert args.ppl_tokens == 1024

    def test_custom_tps_tokens(self):
        p    = bench_2bit._build_parser()
        args = p.parse_args(["--tps-tokens", "256"])
        assert args.tps_tokens == 256


# ── eval_model_perplexity_and_tps (mlx absent path) ─────────────────────────

class TestEvalModelPplTps:
    def test_returns_nones_when_mlx_unavailable(self, tmp_path):
        with patch.dict(sys.modules,
                        {"mlx": None, "mlx.core": None, "mlx_lm": None}):
            result = bench_2bit.eval_model_perplexity_and_tps(str(tmp_path))
        assert result["perplexity"] is None
        assert result["tps"]        is None

    def test_return_keys_always_present(self, tmp_path):
        with patch.dict(sys.modules,
                        {"mlx": None, "mlx.core": None, "mlx_lm": None}):
            result = bench_2bit.eval_model_perplexity_and_tps(str(tmp_path))
        assert "perplexity" in result
        assert "tps"        in result
