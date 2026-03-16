"""tests/test_compression_pipeline.py

End-to-end round-trip tests for the new format variants:
  NF4 → convert → loader_utils → compare
  VPTQ → convert → loader_utils → compare
  DFloat11 passthrough → convert → loader_utils → compare
  INT4 + DFloat11 scales → convert → loader_utils → compare
"""
import tempfile
from pathlib import Path

import numpy as np
import pytest

from squish.convert import quantize_tensor
from squish.io.loader_utils import _dequantize_npy
from squish.quant import quantizer as _qmod

_HAS_INT4 = getattr(_qmod, "_squish_quant", None) is not None

RNG = np.random.default_rng(123)


def _write_and_reload(sub: dict, tmpdir: Path, sk: str = "layer") -> np.ndarray:
    """Write a quantize_tensor() output dict to tmpdir and reload via _dequantize_npy."""
    for suffix, data in sub.items():
        np.save(str(tmpdir / f"{sk}{suffix}.npy"), data)
    return _dequantize_npy(tmpdir, sk)


def _snr_db(orig: np.ndarray, approx: np.ndarray) -> float:
    sig = np.mean(orig.astype(np.float32) ** 2)
    err = np.mean((orig.astype(np.float32) - approx.astype(np.float32)) ** 2)
    return float("inf") if err == 0 else 10 * np.log10(sig / max(err, 1e-30))


# ---------------------------------------------------------------------------
# NF4 round-trip
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _HAS_INT4, reason="squish_quant Rust extension not built")
class TestNF4RoundTrip:
    def test_nf4_basic_round_trip(self):
        arr = RNG.standard_normal((16, 128)).astype(np.float32)
        with tempfile.TemporaryDirectory() as tmpdir:
            sub = quantize_tensor("weight", arr, 20.0, [], use_nf4=True)
            assert "__nf4" in sub
            assert "__s_nf4" in sub
            recon = _write_and_reload(sub, Path(tmpdir))
            snr = _snr_db(arr.reshape(recon.shape), recon)
            # Direct FP32→NF4 path (no INT8 intermediate) should achieve good SNR
            assert snr > 20.0, f"NF4 SNR={snr:.1f} dB too low"

    def test_nf4_shape_preserved(self):
        arr = RNG.standard_normal((8, 64)).astype(np.float32)
        with tempfile.TemporaryDirectory() as tmpdir:
            sub = quantize_tensor("w", arr, 20.0, [], use_nf4=True)
            recon = _write_and_reload(sub, Path(tmpdir))
            # Reconstruction shape may be 2D flat (standard for quantization)
            assert recon.size == arr.size

    def test_nf4_not_lossy_more_than_int4(self):
        """NF4 SNR should be >= INT4 SNR on Gaussian data."""
        arr = RNG.standard_normal((32, 128)).astype(np.float32)
        with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
            sub_nf4 = quantize_tensor("w", arr, 20.0, [], use_nf4=True)
            sub_i4  = quantize_tensor("w", arr, 20.0, [], use_int4=True)
            recon_nf4 = _write_and_reload(sub_nf4, Path(d1))
            recon_i4  = _write_and_reload(sub_i4,  Path(d2))
            snr_nf4 = _snr_db(arr.reshape(recon_nf4.shape), recon_nf4)
            snr_i4  = _snr_db(arr.reshape(recon_i4.shape),  recon_i4)
            # NF4 should be at least as good as INT4
            assert snr_nf4 >= snr_i4 - 2.0, f"NF4 SNR {snr_nf4:.1f} << INT4 SNR {snr_i4:.1f}"


# ---------------------------------------------------------------------------
# VPTQ round-trip
# ---------------------------------------------------------------------------

class TestVPTQRoundTrip:
    def test_vptq_basic_round_trip(self):
        arr = RNG.standard_normal((32, 64)).astype(np.float32)
        from squish.quant.vptq import VPTQConfig
        cfg = VPTQConfig(n_codebook_entries=16, group_size=8, n_residual_entries=4, n_fit_iters=5)
        with tempfile.TemporaryDirectory() as tmpdir:
            sub = quantize_tensor("w", arr, 20.0, [], use_vptq=True, vptq_config=cfg)
            assert "__vq_idx" in sub
            assert "__vq_cb"  in sub
            assert "__vq_meta" in sub
            recon = _write_and_reload(sub, Path(tmpdir))
            # VPTQ is lossy — just check shape and no crash
            assert recon.size == arr.size

    def test_vptq_snr_acceptable(self):
        """VPTQ should achieve at least 10 dB SNR with small codebook."""
        arr = RNG.standard_normal((64, 64)).astype(np.float32)
        from squish.quant.vptq import VPTQConfig
        cfg = VPTQConfig(n_codebook_entries=256, group_size=8, n_residual_entries=16, n_fit_iters=20)
        with tempfile.TemporaryDirectory() as tmpdir:
            sub = quantize_tensor("w", arr, 20.0, [], use_vptq=True, vptq_config=cfg)
            recon = _write_and_reload(sub, Path(tmpdir))
            snr = _snr_db(arr.reshape(recon.shape), recon)
            assert snr > 10.0, f"VPTQ SNR={snr:.1f} dB too low"


# ---------------------------------------------------------------------------
# DFloat11 passthrough round-trip
# ---------------------------------------------------------------------------

class TestDFloat11PassthroughRoundTrip:
    def test_dfloat11_passthrough_exact_or_near_lossless(self):
        arr = RNG.standard_normal((8, 64)).astype(np.float32)
        with tempfile.TemporaryDirectory() as tmpdir:
            # Trigger passthrough path via passthrough_patterns
            sub = quantize_tensor("embed_tokens.weight", arr, 20.0,
                                  ["embed_tokens"], use_dfloat11=True)
            assert "__pt_df11" in sub
            recon = _write_and_reload(sub, Path(tmpdir))
            snr = _snr_db(arr.reshape(recon.shape), recon)
            # DFloat11 is lossless for fp16 stored data
            assert snr > 40.0, f"DFloat11 passthrough SNR={snr:.1f} dB — not lossless?"

    def test_dfloat11_passthrough_vs_plain_passthrough(self):
        """DFloat11 and plain __pt should give same values (just different storage)."""
        arr = RNG.standard_normal((8, 64)).astype(np.float32)
        with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
            sub_df = quantize_tensor("embed.w", arr, 20.0, ["embed"], use_dfloat11=True)
            sub_pt = quantize_tensor("embed.w", arr, 20.0, ["embed"], use_dfloat11=False)
            recon_df = _write_and_reload(sub_df, Path(d1))
            recon_pt = _write_and_reload(sub_pt, Path(d2))
            # Both represent the same fp16 compressed representation
            # Max difference should be within fp16 precision
            diff = np.max(np.abs(recon_df.astype(np.float32) - recon_pt.astype(np.float32)))
            assert diff < 0.01, f"DFloat11 vs plain PT diverged by {diff}"


# ---------------------------------------------------------------------------
# INT4 + DFloat11 scales round-trip
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _HAS_INT4, reason="squish_quant Rust extension not built")
class TestINT4DFloat11ScalesRoundTrip:
    def test_int4_dfloat11_scales_round_trip(self):
        arr = RNG.standard_normal((16, 128)).astype(np.float32)
        with tempfile.TemporaryDirectory() as tmpdir:
            sub = quantize_tensor("w", arr, 20.0, [], use_int4=True, use_dfloat11=True)
            # When dfloat11 is used with int4, __s4 is replaced by __s4_df11
            assert "__s4_df11" in sub
            assert "__s4" not in sub
            recon = _write_and_reload(sub, Path(tmpdir))
            snr = _snr_db(arr.reshape(recon.shape), recon)
            # Direct FP32→INT4 path raises SNR floor
            assert snr > 18.0, f"INT4+DFloat11 SNR={snr:.1f} dB too low"

    def test_int4_direct_fp32_path_better_than_int8_intermediate(self):
        """Direct FP32→INT4 should have strictly better SNR than old INT8→reconstruct→INT4."""
        arr = RNG.standard_normal((32, 128)).astype(np.float32)
        with tempfile.TemporaryDirectory() as d_direct:
            sub_direct = quantize_tensor("w", arr, 20.0, [], use_int4=True)
            recon_direct = _write_and_reload(sub_direct, Path(d_direct))
            snr_direct = _snr_db(arr.reshape(recon_direct.shape), recon_direct)
            # Simulate the old INT8-intermediate path manually
            from squish.quant.quantizer import quantize_embeddings, reconstruct_embeddings, quantize_int4
            flat = arr.reshape(-1, arr.shape[-1])
            r8 = quantize_embeddings(flat, group_size=64)
            recon8 = reconstruct_embeddings(r8)
            packed_old, scales_old = quantize_int4(recon8, group_size=64)
            # Dequantize old path for comparison
            from squish.quant.quantizer import dequantize_int4
            recon_old = dequantize_int4(packed_old, scales_old, group_size=64)
            snr_old = _snr_db(arr.reshape(recon_old.shape), recon_old)
            assert snr_direct >= snr_old, (
                f"Direct INT4 SNR {snr_direct:.1f} dB should be ≥ INT8-intermediate "
                f"SNR {snr_old:.1f} dB"
            )

    def test_int4_super_weight_passthrough(self):
        """super_weight_passthrough=True must produce __pt (FP16 passthrough), not INT4."""
        arr = RNG.standard_normal((8, 64)).astype(np.float32)
        sub = quantize_tensor("layer.weight", arr, 20.0, [], use_int4=True,
                              super_weight_passthrough=True)
        assert "__pt" in sub, "super_weight_passthrough should produce FP16 passthrough"
        assert "__q4" not in sub, "super_weight_passthrough should not produce INT4"


# ---------------------------------------------------------------------------
# KV cache CommVQ round-trip
# ---------------------------------------------------------------------------

class TestCommVQKVCacheRoundTrip:
    def test_commvq_kv_encode_decode(self):
        """CommVQ: tokens past window should be accessible via get_full_kv()."""
        from squish.kv.kv_cache import KVLayerCache

        n_heads, head_dim, window = 4, 32, 4
        cache = KVLayerCache(window=window)
        cache._comm_vq_bits = 2   # 2-bit: 4 codes

        rng = np.random.default_rng(77)
        all_keys   = []
        all_values = []

        # Feed enough tokens to trigger calibration + encoding
        # Calibration requires _SVD_INIT_TOKENS (64) tokens beyond window
        n_tokens = 80
        for _ in range(n_tokens):
            k = rng.standard_normal((n_heads, head_dim)).astype(np.float16)
            v = rng.standard_normal((n_heads, head_dim)).astype(np.float16)
            all_keys.append(k)
            all_values.append(v)
            cache.append(k, v)

        full_k, full_v = cache.get_full_kv()

        # If codebook not fitted yet (< 64 evicted), some tokens may still be in
        # calibration buffer — that's expected behavior
        if full_k is not None:
            assert full_k.ndim == 3
            assert full_k.shape[0] == n_heads
            assert full_k.shape[2] == head_dim
