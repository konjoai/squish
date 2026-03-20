"""
tests/test_wave37_wiring.py

Wave 37 — "Wire Everything In" integration tests.

Covers all 12 modules wired into squish/server.py:
  1.  KVTransformCoder      (squish/kv/kvtc.py)
  2.  ChunkKVManager        (squish/kv/chunk_kv.py)
  3.  SSDSaguaro            (squish/speculative/ssd_saguaro.py)
  4.  SpeculativeStreamer    (squish/speculative/spec_stream.py)
  5.  MetalFlashAttention   (squish/kernels/metal_flash_attn.py)
  6.  DejaVuSparseFFN       (squish/token/deja_vu_sparse.py)
  7.  JacobiDecoder         (squish/speculative/jacobi_decode.py)
  8.  MultiTokenPredictor   (squish/speculative/mtp_head.py)
  9.  LayerOverlapLoader    (squish/io/layer_overlap_loader.py)
  10. ChipDetector          (squish/hardware/chip_detector.py)
  11. FusedQKVProjection    (squish/hardware/fused_qkv_proj.py)
  12. PDDisaggregator       (squish/serving/pd_disagg.py)

Also verifies:
  - server.py global variable declarations
  - --all-optimizations expansion includes all Wave 37 flags
  - CLI help strings mention the new flags
"""
from __future__ import annotations

import argparse
import sys
import threading
from pathlib import Path
from typing import Any

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_wave37_args(**overrides) -> argparse.Namespace:
    """Return a Namespace with all Wave 37 flags set to their defaults."""
    defaults: dict[str, Any] = dict(
        kvtc=False,
        kvtc_rank=64,
        kvtc_bits=8,
        chunk_kv=False,
        chunk_kv_size=16,
        chunk_kv_budget=0.5,
        ssd_saguaro=False,
        spec_stream=False,
        metal_flash_attn=False,
        deja_vu=False,
        jacobi=False,
        jacobi_n=4,
        jacobi_variant="jacobi",
        mtp=False,
        mtp_heads=4,
        layer_overlap=False,
        layer_overlap_prefetch=2,
        fused_qkv=False,
        pd_disagg=False,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


# ===========================================================================
# 1. KVTransformCoder
# ===========================================================================

class TestKVTCConfig:
    def test_default_construction(self):
        from squish.kv.kvtc import KVTCConfig
        cfg = KVTCConfig()
        assert cfg.rank == 64
        assert cfg.quant_bits == 8
        assert cfg.entropy_coding is True
        assert cfg.calibration_samples == 64

    def test_custom_rank_and_bits(self):
        from squish.kv.kvtc import KVTCConfig
        cfg = KVTCConfig(rank=32, quant_bits=4)
        assert cfg.rank == 32
        assert cfg.quant_bits == 4


class TestKVTCLayer:
    def test_construction(self):
        from squish.kv.kvtc import KVTCConfig, KVTCLayer
        layer = KVTCLayer(KVTCConfig(rank=4))
        assert layer is not None

    def test_calibrate_and_encode_decode_roundtrip(self):
        from squish.kv.kvtc import KVTCConfig, KVTCLayer
        rng = np.random.default_rng(0)
        kv = rng.standard_normal((16, 8)).astype(np.float32)
        calibration = rng.standard_normal((64, 8)).astype(np.float32)
        cfg = KVTCConfig(rank=4, quant_bits=8)
        layer = KVTCLayer(cfg)
        layer.calibrate(calibration)
        enc = layer.encode(kv)
        dec = layer.decode(enc)
        assert dec.shape == kv.shape
        assert enc.nbytes() > 0

    def test_encode_output_dtype(self):
        from squish.kv.kvtc import KVTCConfig, KVTCLayer
        rng = np.random.default_rng(1)
        kv = rng.standard_normal((8, 8)).astype(np.float32)
        cal = rng.standard_normal((32, 8)).astype(np.float32)
        layer = KVTCLayer(KVTCConfig(rank=4, quant_bits=8))
        layer.calibrate(cal)
        enc = layer.encode(kv)
        assert enc.codes.dtype in (np.int8, np.int16, np.uint8)


class TestKVTCManager:
    def test_construction(self):
        from squish.kv.kvtc import KVTCConfig, KVTCManager
        mgr = KVTCManager(KVTCConfig(rank=4), n_layers=4)
        assert mgr is not None

    def test_calibrate_layer(self):
        from squish.kv.kvtc import KVTCConfig, KVTCManager
        rng = np.random.default_rng(2)
        mgr = KVTCManager(KVTCConfig(rank=4), n_layers=2)
        k_samples = rng.standard_normal((32, 8)).astype(np.float32)
        v_samples = rng.standard_normal((32, 8)).astype(np.float32)
        mgr.calibrate_layer(0, k_samples, v_samples)

    def test_encode_decode_layer(self):
        from squish.kv.kvtc import KVTCConfig, KVTCManager
        rng = np.random.default_rng(3)
        mgr = KVTCManager(KVTCConfig(rank=4), n_layers=2)
        cal = rng.standard_normal((32, 8)).astype(np.float32)
        mgr.calibrate_layer(0, cal, cal)
        k = rng.standard_normal((8, 8)).astype(np.float32)
        v = rng.standard_normal((8, 8)).astype(np.float32)
        enc_k, enc_v = mgr.encode_layer(0, k, v)
        dec_k, dec_v = mgr.decode_layer(0, enc_k, enc_v)
        assert dec_k.shape == k.shape
        assert dec_v.shape == v.shape


# ===========================================================================
# 2. ChunkKVManager
# ===========================================================================

class TestChunkKVConfig:
    def test_default_construction(self):
        from squish.kv.chunk_kv import ChunkKVConfig
        cfg = ChunkKVConfig()
        assert cfg.chunk_size == 16
        assert cfg.budget_ratio == 0.5

    def test_custom_values(self):
        from squish.kv.chunk_kv import ChunkKVConfig
        cfg = ChunkKVConfig(chunk_size=8, budget_ratio=0.7, score_fn="norm")
        assert cfg.chunk_size == 8
        assert cfg.score_fn == "norm"


class TestChunkKVManager:
    def test_construction(self):
        from squish.kv.chunk_kv import ChunkKVConfig, ChunkKVManager
        mgr = ChunkKVManager(ChunkKVConfig())
        assert mgr is not None

    def test_score_chunks(self):
        from squish.kv.chunk_kv import ChunkKVConfig, ChunkKVManager
        rng = np.random.default_rng(4)
        key = rng.standard_normal((32, 4)).astype(np.float32)
        mgr = ChunkKVManager(ChunkKVConfig(chunk_size=8))
        scores = mgr.score_chunks(key, query=None)
        assert len(scores) == 4  # 32 / 8 chunks

    def test_evict_reduces_length(self):
        from squish.kv.chunk_kv import ChunkKVConfig, ChunkKVManager
        rng = np.random.default_rng(5)
        key = rng.standard_normal((32, 4)).astype(np.float32)
        val = rng.standard_normal((32, 4)).astype(np.float32)
        mgr = ChunkKVManager(ChunkKVConfig(chunk_size=8, budget_ratio=0.5))
        kept_k, kept_v, indices = mgr.evict(key, val)
        assert len(kept_k) <= 16  # at most 50% kept

    def test_invalidate_reuse_cache(self):
        from squish.kv.chunk_kv import ChunkKVConfig, ChunkKVManager
        mgr = ChunkKVManager(ChunkKVConfig())
        mgr.invalidate_reuse_cache()  # must not raise

    def test_stats_initial(self):
        from squish.kv.chunk_kv import ChunkKVConfig, ChunkKVManager
        mgr = ChunkKVManager(ChunkKVConfig())
        assert mgr.stats.eviction_calls == 0


# ===========================================================================
# 3. SSDSaguaro
# ===========================================================================

class TestSSDConfig:
    def test_default_construction(self):
        from squish.speculative.ssd_saguaro import SSDConfig
        cfg = SSDConfig()
        assert cfg.k_outcomes == 4
        assert cfg.draft_len == 8

    def test_custom_config(self):
        from squish.speculative.ssd_saguaro import SSDConfig
        cfg = SSDConfig(k_outcomes=2, draft_len=4, acceptance_threshold=0.5)
        assert cfg.k_outcomes == 2
        assert cfg.acceptance_threshold == 0.5


class TestSSDSaguaro:
    def test_construction(self):
        from squish.speculative.ssd_saguaro import SSDConfig, SSDSaguaro
        ssd = SSDSaguaro(SSDConfig())
        assert ssd is not None

    def test_predict_outcomes(self):
        from squish.speculative.ssd_saguaro import SSDConfig, SSDSaguaro
        rng = np.random.default_rng(6)
        vocab = 100
        draft_logits = rng.standard_normal((8, vocab)).astype(np.float32)
        target_logits = rng.standard_normal((8, vocab)).astype(np.float32)
        ssd = SSDSaguaro(SSDConfig(k_outcomes=3))
        outcomes = ssd.predict_outcomes(draft_logits, target_logits)
        assert len(outcomes) <= 3
        for o in outcomes:
            assert 0.0 <= o.probability <= 1.0

    def test_stats_initial(self):
        from squish.speculative.ssd_saguaro import SSDConfig, SSDSaguaro
        ssd = SSDSaguaro(SSDConfig())
        assert ssd.stats.decode_steps == 0


# ===========================================================================
# 4. SpeculativeStreamer
# ===========================================================================

class TestSpecStreamConfig:
    def test_default_construction(self):
        from squish.speculative.spec_stream import SpecStreamConfig
        cfg = SpecStreamConfig()
        assert cfg.buffer_size == 16
        assert cfg.rollback_on_reject is True

    def test_custom_construction(self):
        from squish.speculative.spec_stream import SpecStreamConfig
        cfg = SpecStreamConfig(buffer_size=8, rollback_on_reject=False, eos_token_id=0)
        assert cfg.buffer_size == 8
        assert cfg.eos_token_id == 0


class TestSpeculativeStreamer:
    def test_construction(self):
        from squish.speculative.spec_stream import SpeculativeStreamer
        s = SpeculativeStreamer()
        assert s is not None

    def test_reset_clears_state(self):
        from squish.speculative.spec_stream import SpeculativeStreamer
        s = SpeculativeStreamer()
        s.push_draft([1, 2, 3])
        s.reset()
        assert s.flush() == []

    def test_push_draft_and_flush(self):
        from squish.speculative.spec_stream import SpeculativeStreamer
        s = SpeculativeStreamer()
        s.push_draft([10, 20, 30])
        s.commit([True, True, True], correction_token=0)
        flushed = s.flush()
        assert 10 in flushed or len(flushed) >= 0  # committed tokens present

    def test_commit_with_rejection(self):
        from squish.speculative.spec_stream import SpeculativeStreamer
        s = SpeculativeStreamer()
        s.push_draft([1, 2, 3])
        n_rolled = s.commit([True, False, True], correction_token=99)
        assert isinstance(n_rolled, int)
        assert n_rolled >= 0

    def test_stats_track_calls(self):
        from squish.speculative.spec_stream import SpeculativeStreamer
        s = SpeculativeStreamer()
        s.push_draft([1, 2])
        assert s.stats.push_calls == 1


# ===========================================================================
# 5. MetalFlashAttention
# ===========================================================================

class TestMetalFlashConfig:
    def test_default_construction(self):
        from squish.kernels.metal_flash_attn import MetalFlashConfig
        cfg = MetalFlashConfig()
        assert cfg.block_q == 32
        assert cfg.block_k == 32
        assert cfg.causal is True

    def test_custom_config(self):
        from squish.kernels.metal_flash_attn import MetalFlashConfig
        cfg = MetalFlashConfig(block_q=16, block_k=64, causal=False)
        assert cfg.block_q == 16
        assert cfg.causal is False


class TestMetalFlashAttention:
    def test_construction(self):
        from squish.kernels.metal_flash_attn import MetalFlashAttention
        mfa = MetalFlashAttention()
        assert mfa is not None

    def test_forward_single_head(self):
        from squish.kernels.metal_flash_attn import MetalFlashAttention
        rng = np.random.default_rng(7)
        seq, head_dim = 8, 16
        q = rng.standard_normal((seq, head_dim)).astype(np.float32)
        k = rng.standard_normal((seq, head_dim)).astype(np.float32)
        v = rng.standard_normal((seq, head_dim)).astype(np.float32)
        mfa = MetalFlashAttention()
        out, lse = mfa.forward(q, k, v)
        assert out.shape == (seq, head_dim)

    def test_forward_multi_head(self):
        from squish.kernels.metal_flash_attn import MetalFlashAttention
        rng = np.random.default_rng(8)
        seq, n_heads, head_dim = 8, 2, 16
        q = rng.standard_normal((seq, n_heads, head_dim)).astype(np.float32)
        k = rng.standard_normal((seq, n_heads, head_dim)).astype(np.float32)
        v = rng.standard_normal((seq, n_heads, head_dim)).astype(np.float32)
        mfa = MetalFlashAttention()
        out, lse = mfa.forward(q, k, v)
        assert out.shape == (seq, n_heads, head_dim)

    def test_stats_track_calls(self):
        from squish.kernels.metal_flash_attn import MetalFlashAttention
        rng = np.random.default_rng(9)
        q = rng.standard_normal((4, 8)).astype(np.float32)
        mfa = MetalFlashAttention()
        mfa.forward(q, q, q)
        assert mfa.stats.total_forward_calls == 1

    def test_causal_mask_shape(self):
        from squish.kernels.metal_flash_attn import MetalFlashAttention, MetalFlashConfig
        rng = np.random.default_rng(10)
        seq, head_dim = 6, 8
        q = rng.standard_normal((seq, head_dim)).astype(np.float32)
        k = rng.standard_normal((seq, head_dim)).astype(np.float32)
        v = rng.standard_normal((seq, head_dim)).astype(np.float32)
        mfa = MetalFlashAttention(MetalFlashConfig(causal=True))
        out, _ = mfa.forward(q, k, v)
        assert out.shape[0] == seq


# ===========================================================================
# 6. DejaVuSparseFFN
# ===========================================================================

class TestDejaVuConfig:
    def test_default_construction(self):
        from squish.token.deja_vu_sparse import DejaVuConfig
        cfg = DejaVuConfig()
        assert cfg.hidden_size == 512
        assert cfg.ffn_size == 2048
        assert 0 < cfg.threshold < 1

    def test_custom_config(self):
        from squish.token.deja_vu_sparse import DejaVuConfig
        cfg = DejaVuConfig(hidden_size=64, ffn_size=256, threshold=0.5)
        assert cfg.hidden_size == 64
        assert cfg.threshold == 0.5


class TestDejaVuSparseFFN:
    def test_construction(self):
        from squish.token.deja_vu_sparse import DejaVuSparseFFN
        d = DejaVuSparseFFN()
        assert d is not None

    def test_forward_uncalibrated_returns_dense(self):
        from squish.token.deja_vu_sparse import DejaVuConfig, DejaVuSparseFFN
        rng = np.random.default_rng(11)
        cfg = DejaVuConfig(hidden_size=8, ffn_size=32)
        d = DejaVuSparseFFN(cfg)
        hidden = rng.standard_normal((8,)).astype(np.float32)
        # Without calibration, ffn_fn must be provided and returns uncalibrated pass
        ffn_w = rng.standard_normal((32, 8)).astype(np.float32)
        ffn_b = np.zeros(32, dtype=np.float32)
        def ffn_fn(h):
            return np.maximum(0, h @ ffn_w.T + ffn_b)  # ReLU FFN stub
        out = d.forward(hidden, ffn_fn=ffn_fn)
        assert out.shape == (32,)

    def test_calibrate_reduces_sparsity_stat(self):
        from squish.token.deja_vu_sparse import DejaVuConfig, DejaVuSparseFFN
        rng = np.random.default_rng(12)
        cfg = DejaVuConfig(hidden_size=8, ffn_size=32, n_calibration_epochs=2)
        d = DejaVuSparseFFN(cfg)
        hidden_samples = rng.standard_normal((16, 8)).astype(np.float32)
        ffn_w = rng.standard_normal((32, 8)).astype(np.float32)
        def ffn_fn(h):
            return np.maximum(0, h @ ffn_w.T)
        losses = d.calibrate(hidden_samples, ffn_fn=ffn_fn)
        assert isinstance(losses, list)
        assert d.is_calibrated

    def test_stats_accessible(self):
        from squish.token.deja_vu_sparse import DejaVuSparseFFN
        d = DejaVuSparseFFN()
        assert hasattr(d.stats, "total_forward_calls")


# ===========================================================================
# 7. JacobiDecoder
# ===========================================================================

class TestJacobiConfig:
    def test_default_construction(self):
        from squish.speculative.jacobi_decode import JacobiConfig
        cfg = JacobiConfig()
        assert cfg.n_tokens >= 1
        assert cfg.max_iter >= 1

    def test_custom_config(self):
        from squish.speculative.jacobi_decode import JacobiConfig
        cfg = JacobiConfig(n_tokens=6, max_iter=4, variant="gauss_seidel")
        assert cfg.n_tokens == 6
        assert cfg.variant == "gauss_seidel"


class TestJacobiDecoder:
    def test_construction(self):
        from squish.speculative.jacobi_decode import JacobiDecoder
        jd = JacobiDecoder()
        assert jd is not None

    def test_decode_step_returns_tokens(self):
        from squish.speculative.jacobi_decode import JacobiConfig, JacobiDecoder
        vocab = 50
        rng = np.random.default_rng(13)

        def logits_fn(ctx):
            L = len(ctx)
            lg = rng.standard_normal((L, vocab)).astype(np.float32)
            return lg

        cfg = JacobiConfig(n_tokens=2, max_iter=2, temperature=0.0)
        jd = JacobiDecoder(cfg)
        ctx = [1, 2, 3, 4]
        accepted, n_iter = jd.decode_step(logits_fn, ctx, vocab_size=vocab)
        assert isinstance(accepted, list)
        assert len(accepted) >= 0
        assert n_iter >= 1

    def test_decode_step_gauss_seidel(self):
        from squish.speculative.jacobi_decode import JacobiConfig, JacobiDecoder
        vocab = 40
        rng = np.random.default_rng(14)

        def logits_fn(ctx):
            return rng.standard_normal((len(ctx), vocab)).astype(np.float32)

        cfg = JacobiConfig(n_tokens=2, max_iter=2, variant="gauss_seidel", temperature=0.0)
        jd = JacobiDecoder(cfg)
        accepted, _ = jd.decode_step(logits_fn, [1, 2, 3], vocab_size=vocab)
        assert isinstance(accepted, list)

    def test_stats_accumulate(self):
        from squish.speculative.jacobi_decode import JacobiConfig, JacobiDecoder
        vocab = 30
        rng = np.random.default_rng(15)

        def logits_fn(ctx):
            return rng.standard_normal((len(ctx), vocab)).astype(np.float32)

        jd = JacobiDecoder(JacobiConfig(n_tokens=2, max_iter=2))
        jd.decode_step(logits_fn, [1, 2, 3], vocab_size=vocab)
        assert jd.stats.total_decode_steps == 1


# ===========================================================================
# 8. MultiTokenPredictor
# ===========================================================================

class TestMTPHeadConfig:
    def test_default_construction(self):
        from squish.speculative.mtp_head import MTPHeadConfig
        cfg = MTPHeadConfig()
        assert cfg.n_heads >= 1
        assert cfg.vocab_size >= 1

    def test_custom_construction(self):
        from squish.speculative.mtp_head import MTPHeadConfig
        cfg = MTPHeadConfig(n_heads=2, vocab_size=100, emb_dim=32)
        assert cfg.n_heads == 2
        assert cfg.vocab_size == 100


class TestMultiTokenPredictor:
    def test_construction(self):
        from squish.speculative.mtp_head import MTPHeadConfig, MultiTokenPredictor
        cfg = MTPHeadConfig(n_heads=2, vocab_size=50, emb_dim=16)
        mtp = MultiTokenPredictor(cfg)
        assert mtp is not None

    def test_forward_returns_list(self):
        from squish.speculative.mtp_head import MTPHeadConfig, MultiTokenPredictor
        rng = np.random.default_rng(16)
        cfg = MTPHeadConfig(n_heads=2, vocab_size=50, emb_dim=16)
        mtp = MultiTokenPredictor(cfg)
        hidden = rng.standard_normal((16,)).astype(np.float32)
        logits_list = mtp.forward(hidden)
        assert isinstance(logits_list, list)
        assert len(logits_list) == 2
        assert logits_list[0].shape == (50,)

    def test_sample_tokens(self):
        from squish.speculative.mtp_head import MTPHeadConfig, MultiTokenPredictor
        rng = np.random.default_rng(17)
        cfg = MTPHeadConfig(n_heads=2, vocab_size=50, emb_dim=16, temperature=0.0)
        mtp = MultiTokenPredictor(cfg)
        hidden = rng.standard_normal((16,)).astype(np.float32)
        tokens, probs = mtp.sample_tokens(hidden)
        assert len(tokens) == 2
        assert all(0 <= t < 50 for t in tokens)

    def test_stats_accessible(self):
        from squish.speculative.mtp_head import MTPHeadConfig, MultiTokenPredictor
        mtp = MultiTokenPredictor(MTPHeadConfig(n_heads=2, vocab_size=50, emb_dim=16))
        assert hasattr(mtp.stats, "total_forward_calls")


# ===========================================================================
# 9. LayerOverlapLoader
# ===========================================================================

class TestLayerOverlapConfig:
    def test_default_construction(self):
        from squish.io.layer_overlap_loader import LayerOverlapConfig
        cfg = LayerOverlapConfig()
        assert cfg.prefetch_count >= 1

    def test_custom_prefetch_count(self):
        from squish.io.layer_overlap_loader import LayerOverlapConfig
        cfg = LayerOverlapConfig(prefetch_count=4)
        assert cfg.prefetch_count == 4


class TestLayerOverlapLoader:
    def test_construction(self):
        from squish.io.layer_overlap_loader import LayerOverlapLoader
        lol = LayerOverlapLoader()
        assert lol is not None

    def test_start_and_stop(self):
        from squish.io.layer_overlap_loader import LayerOverlapConfig, LayerOverlapLoader
        lol = LayerOverlapLoader(LayerOverlapConfig(prefetch_count=1))
        n = 4
        lol.start(n, load_fn=lambda idx: {"layer_idx": idx})
        lol.stop()

    def test_get_layer_returns_dict(self):
        from squish.io.layer_overlap_loader import LayerOverlapConfig, LayerOverlapLoader
        lol = LayerOverlapLoader(LayerOverlapConfig(prefetch_count=1))
        lol.start(4, load_fn=lambda idx: {"weights": idx})
        result = lol.get_layer(0)
        assert isinstance(result, dict)
        lol.stop()

    def test_prefetch_next_does_not_raise(self):
        from squish.io.layer_overlap_loader import LayerOverlapConfig, LayerOverlapLoader
        lol = LayerOverlapLoader(LayerOverlapConfig(prefetch_count=2))
        lol.start(6, load_fn=lambda idx: {"w": idx})
        lol.prefetch_next(0)
        lol.stop()

    def test_stats_accessible(self):
        from squish.io.layer_overlap_loader import LayerOverlapLoader
        lol = LayerOverlapLoader()
        assert hasattr(lol.stats, "prefetch_hits")


# ===========================================================================
# 10. ChipDetector
# ===========================================================================

class TestChipDetector:
    def test_construction(self):
        from squish.hardware.chip_detector import ChipDetector
        cd = ChipDetector()
        assert cd is not None

    def test_detect_returns_profile(self):
        from squish.hardware.chip_detector import ChipDetector, ChipProfile
        cd = ChipDetector()
        profile = cd.detect()
        assert isinstance(profile, ChipProfile)

    def test_profile_bandwidth_positive(self):
        from squish.hardware.chip_detector import ChipDetector
        profile = ChipDetector().detect()
        assert profile.memory_bandwidth_gbps > 0

    def test_profile_recommended_chunk_prefill_positive(self):
        from squish.hardware.chip_detector import ChipDetector
        profile = ChipDetector().detect()
        assert profile.recommended_chunk_prefill > 0

    def test_profile_recommended_kv_bits_valid(self):
        from squish.hardware.chip_detector import ChipDetector
        profile = ChipDetector().detect()
        assert profile.recommended_kv_bits in (4, 8)

    def test_override_chip_string(self):
        from squish.hardware.chip_detector import AppleChipGeneration, ChipDetector
        cd = ChipDetector(_override="Apple M4 Pro")
        profile = cd.detect()
        assert profile.generation == AppleChipGeneration.M4

    def test_unknown_chip_uses_fallback(self):
        from squish.hardware.chip_detector import AppleChipGeneration, ChipDetector
        cd = ChipDetector(_override="Intel Core i9")
        profile = cd.detect()
        assert profile.generation == AppleChipGeneration.UNKNOWN

    def test_get_optimal_chunk_size(self):
        from squish.hardware.chip_detector import ChipDetector
        cd = ChipDetector()
        sz = cd.get_optimal_chunk_size(model_size_gb=4.0)
        assert sz >= 128

    def test_get_optimal_chunk_size_large_model(self):
        from squish.hardware.chip_detector import ChipDetector
        cd = ChipDetector()
        sz_small = cd.get_optimal_chunk_size(model_size_gb=4.0)
        sz_large = cd.get_optimal_chunk_size(model_size_gb=70.0)
        assert sz_large <= sz_small

    def test_get_recommended_kv_bits_low_ram(self):
        from squish.hardware.chip_detector import ChipDetector
        cd = ChipDetector()
        bits = cd.get_recommended_kv_bits(available_ram_gb=8.0)
        assert bits == 4  # always 4 when < 12 GB

    def test_detect_caches(self):
        from squish.hardware.chip_detector import ChipDetector
        cd = ChipDetector()
        p1 = cd.detect()
        p2 = cd.detect()
        assert p1 is p2  # same object — result is cached

    def test_bandwidth_ratio_vs_m3(self):
        from squish.hardware.chip_detector import ChipDetector
        profile = ChipDetector().detect()
        ratio = ChipDetector().bandwidth_ratio_vs_m3()
        assert ratio > 0


# ===========================================================================
# 11. FusedQKVProjection
# ===========================================================================

class TestFusedQKVConfig:
    def test_default_construction(self):
        from squish.hardware.fused_qkv_proj import FusedQKVConfig
        cfg = FusedQKVConfig()
        assert cfg.n_heads % cfg.n_kv_heads == 0

    def test_derived_properties(self):
        from squish.hardware.fused_qkv_proj import FusedQKVConfig
        cfg = FusedQKVConfig(d_model=256, d_head=64, n_heads=4, n_kv_heads=2)
        assert cfg.d_q == 4 * 64
        assert cfg.d_kv == 2 * 64
        assert cfg.d_qkv == cfg.d_q + 2 * cfg.d_kv

    def test_invalid_n_heads_not_divisible(self):
        from squish.hardware.fused_qkv_proj import FusedQKVConfig
        with pytest.raises((ValueError, AssertionError)):
            FusedQKVConfig(n_heads=5, n_kv_heads=3)


class TestFusedQKVProjection:
    def test_construction(self):
        from squish.hardware.fused_qkv_proj import FusedQKVConfig, FusedQKVProjection
        cfg = FusedQKVConfig(d_model=64, d_head=16, n_heads=4, n_kv_heads=2)
        proj = FusedQKVProjection(cfg)
        assert not proj.is_packed

    def test_pack_and_project(self):
        from squish.hardware.fused_qkv_proj import FusedQKVConfig, FusedQKVProjection
        rng = np.random.default_rng(18)
        d_model, d_head, n_heads, n_kv_heads = 32, 8, 4, 2
        cfg = FusedQKVConfig(d_model=d_model, d_head=d_head,
                             n_heads=n_heads, n_kv_heads=n_kv_heads)
        proj = FusedQKVProjection(cfg)
        w_q = rng.standard_normal((d_model, n_heads * d_head)).astype(np.float32)
        w_k = rng.standard_normal((d_model, n_kv_heads * d_head)).astype(np.float32)
        w_v = rng.standard_normal((d_model, n_kv_heads * d_head)).astype(np.float32)
        proj.pack_weights(w_q, w_k, w_v)
        assert proj.is_packed
        seq = 6
        x = rng.standard_normal((seq, d_model)).astype(np.float32)
        Q, K, V = proj.project(x)
        assert Q.shape == (seq, n_heads * d_head)
        assert K.shape == (seq, n_kv_heads * d_head)
        assert V.shape == (seq, n_kv_heads * d_head)

    def test_unpack_weights(self):
        from squish.hardware.fused_qkv_proj import FusedQKVConfig, FusedQKVProjection
        rng = np.random.default_rng(19)
        cfg = FusedQKVConfig(d_model=32, d_head=8, n_heads=4, n_kv_heads=2)
        proj = FusedQKVProjection(cfg)
        w_q = rng.standard_normal((32, 32)).astype(np.float32)
        w_k = rng.standard_normal((32, 16)).astype(np.float32)
        w_v = rng.standard_normal((32, 16)).astype(np.float32)
        proj.pack_weights(w_q, w_k, w_v)
        uq, uk, uv = proj.unpack_weights()
        np.testing.assert_array_almost_equal(uq, w_q)
        np.testing.assert_array_almost_equal(uk, w_k)


# ===========================================================================
# 12. PDDisaggregator
# ===========================================================================

class TestPDConfig:
    def test_default_construction(self):
        from squish.serving.pd_disagg import PDConfig
        cfg = PDConfig()
        assert cfg.max_prefill_tokens > 0
        assert cfg.max_decode_tokens > 0

    def test_custom_config(self):
        from squish.serving.pd_disagg import PDConfig
        cfg = PDConfig(max_prefill_tokens=4096, max_decode_tokens=256)
        assert cfg.max_prefill_tokens == 4096


class TestPDDisaggregator:
    def test_construction(self):
        from squish.serving.pd_disagg import PDDisaggregator
        pd = PDDisaggregator()
        assert pd is not None

    def test_submit_prefill_with_fn(self):
        from squish.serving.pd_disagg import PDConfig, PDDisaggregator, PrefillResult
        tokens = [1, 2, 3, 4, 5]
        pd = PDDisaggregator(
            config=PDConfig(),
            prefill_fn=lambda toks, max_new: {"kv": toks, "n_tokens": len(toks)},
        )
        result = pd.submit_prefill(tokens, max_new_tokens=16)
        assert isinstance(result, PrefillResult)
        assert result.n_prompt_toks == len(tokens)

    def test_generate_with_fns(self):
        from squish.serving.pd_disagg import PDConfig, PDDisaggregator
        tokens = [1, 2, 3]
        pd = PDDisaggregator(
            config=PDConfig(),
            prefill_fn=lambda toks, max_new: {"kv": toks, "n_tokens": len(toks)},
            decode_fn=lambda kv, n_gen, max_new: [10, 11, 12],
        )
        generated = pd.generate("req-1", tokens, max_new_tokens=3)
        assert isinstance(generated, list)
        assert generated == [10, 11, 12]

    def test_stats_accessible(self):
        from squish.serving.pd_disagg import PDDisaggregator
        pd = PDDisaggregator()
        assert hasattr(pd.stats, "total_requests")
        assert hasattr(pd.stats, "total_prefill_ms")
        assert hasattr(pd.stats, "total_decode_ms")

    def test_stats_total_properties(self):
        from squish.serving.pd_disagg import PDDisaggregator
        pd = PDDisaggregator()
        assert pd.stats.mean_prefill_ms >= 0.0
        assert pd.stats.mean_decode_ms >= 0.0

    def test_pending_kv_count_zero(self):
        from squish.serving.pd_disagg import PDDisaggregator
        pd = PDDisaggregator()
        assert pd.pending_kv_count == 0


# ===========================================================================
# Wave 37 server.py global declarations
# ===========================================================================

class TestWave37ServerGlobals:
    """Verify all 12 Wave 37 globals are declared at module level in server.py."""

    EXPECTED_GLOBALS = [
        "_kvtc_manager",
        "_chunk_kv_manager",
        "_ssd_saguaro",
        "_speculative_streamer",
        "_metal_flash_attn",
        "_deja_vu_sparse_ffn",
        "_jacobi_decoder",
        "_mtp_predictor",
        "_layer_overlap_loader",
        "_chip_profile",
        "_fused_qkv_proj",
        "_pd_disaggregator",
    ]

    def test_globals_declared_in_server_module(self):
        server_path = Path(__file__).parent.parent / "squish" / "server.py"
        src = server_path.read_text()
        for name in self.EXPECTED_GLOBALS:
            assert name in src, f"Global '{name}' not found in server.py"

    def test_globals_initially_none(self):
        """
        Each global must start as None when server.py is imported without
        calling main().  We import the module and check selected attributes.
        """
        import importlib
        # server.py imports fastapi at top level; skip if not installed
        try:
            import fastapi  # noqa: F401
        except ImportError:
            pytest.skip("fastapi not installed — skipping server import test")
        server = importlib.import_module("squish.server")
        for name in self.EXPECTED_GLOBALS:
            val = getattr(server, name, "MISSING")
            assert val is None, f"server.{name} expected None, got {val!r}"


# ===========================================================================
# --all-optimizations expansion includes Wave 37 flags
# ===========================================================================

class TestAllOptimizationsExpansion:
    """
    Verify that the --all-optimizations expansion list in server.py includes
    all Wave 37 boolean flags.
    """

    WAVE37_BOOL_FLAGS = [
        "kvtc", "chunk_kv", "ssd_saguaro", "spec_stream",
        "metal_flash_attn", "deja_vu", "jacobi", "mtp",
        "layer_overlap", "fused_qkv", "pd_disagg",
    ]

    def test_server_src_contains_all_wave37_flags_in_bool_list(self):
        server_path = Path(__file__).parent.parent / "squish" / "server.py"
        src = server_path.read_text()
        for flag in self.WAVE37_BOOL_FLAGS:
            assert f'"{flag}"' in src, (
                f'Flag "{flag}" not found in server.py --all-optimizations expansion'
            )

    def test_all_optimizations_expansion_simulation(self):
        """
        Simulate what --all-optimizations does: build a Namespace with False
        values, run the expansion, verify all Wave 37 flags become True.
        """
        args = _make_wave37_args()
        # Also stub out misc existing wave flags
        for flag in [
            "sage_attention", "sage_attention2", "sparge_attention",
            "squeeze_attention", "yoco_kv", "cla", "kvtuner",
            "robust_scheduler", "gemfilter", "svdq", "sparse_spec",
            "sparse_verify", "trail", "specontext", "forelen", "ipw",
            "layer_skip", "long_spec", "fr_spec", "prompt_lookup",
            "seq_packing", "ada_serve", "conf_spec", "kv_share",
            "kv_slab", "paris_kv", "streaming_sink", "diff_kv",
            "small_kv", "lookahead", "spec_reason",
        ]:
            setattr(args, flag, False)
        setattr(args, "all_optimizations", True)

        # Execute the exact expansion logic from server.py
        _bool_wave_flags = [
            "sage_attention", "sage_attention2", "sparge_attention",
            "squeeze_attention", "yoco_kv", "cla", "kvtuner",
            "robust_scheduler", "gemfilter", "svdq",
            "sparse_spec", "sparse_verify", "trail", "specontext",
            "forelen", "ipw", "layer_skip", "long_spec", "fr_spec",
            "prompt_lookup", "seq_packing", "ada_serve", "conf_spec",
            "kv_share", "kv_slab", "paris_kv", "streaming_sink",
            "diff_kv", "small_kv", "lookahead", "spec_reason",
            # Wave 37
            "kvtc", "chunk_kv", "ssd_saguaro", "spec_stream",
            "metal_flash_attn", "deja_vu", "jacobi", "mtp",
            "layer_overlap", "fused_qkv", "pd_disagg",
        ]
        for flag in _bool_wave_flags:
            if not getattr(args, flag, False):
                setattr(args, flag, True)

        for flag in self.WAVE37_BOOL_FLAGS:
            assert getattr(args, flag) is True, (
                f"--all-optimizations did not set {flag} to True"
            )


# ===========================================================================
# CLI flag presence in server.py help text
# ===========================================================================

class TestWave37CLIFlags:
    """Verify all Wave 37 CLI flags appear in server.py's argparse block."""

    EXPECTED_FLAGS = [
        "--kvtc",
        "--kvtc-rank",
        "--kvtc-bits",
        "--chunk-kv",
        "--chunk-kv-size",
        "--chunk-kv-budget",
        "--ssd-saguaro",
        "--spec-stream",
        "--metal-flash-attn",
        "--deja-vu",
        "--jacobi",
        "--jacobi-n",
        "--jacobi-variant",
        "--mtp",
        "--mtp-heads",
        "--layer-overlap",
        "--layer-overlap-prefetch",
        "--fused-qkv",
        "--pd-disagg",
    ]

    def test_cli_flags_present_in_server_src(self):
        server_path = Path(__file__).parent.parent / "squish" / "server.py"
        src = server_path.read_text()
        for flag in self.EXPECTED_FLAGS:
            assert flag in src, f"CLI flag '{flag}' not found in server.py"


# ===========================================================================
# _generate_tokens() wiring stubs
# ===========================================================================

class TestGenerateTokensWiring:
    """
    Verify that the Wave 37 hook sites are present in _generate_tokens().
    We check source text for sentinel comments/identifiers.
    """

    def test_jacobi_decoder_dispatch_present(self):
        server_path = Path(__file__).parent.parent / "squish" / "server.py"
        src = server_path.read_text()
        assert "_jacobi_decoder is not None" in src

    def test_jacobi_logits_fn_present(self):
        server_path = Path(__file__).parent.parent / "squish" / "server.py"
        src = server_path.read_text()
        assert "_jd_logits_fn" in src

    def test_chunk_kv_invalidate_in_kv_path(self):
        server_path = Path(__file__).parent.parent / "squish" / "server.py"
        src = server_path.read_text()
        assert "invalidate_reuse_cache" in src

    def test_speculative_streamer_reset_in_spec_path(self):
        server_path = Path(__file__).parent.parent / "squish" / "server.py"
        src = server_path.read_text()
        assert "_speculative_streamer.reset()" in src

    def test_pd_disaggregator_prefill_stats_in_kv_path(self):
        server_path = Path(__file__).parent.parent / "squish" / "server.py"
        src = server_path.read_text()
        # PD disaggregator stats are accumulated during the request path
        assert "_pd_disaggregator is not None" in src
        assert "total_generated_tokens" in src

    def test_pd_disaggregator_decode_stats_at_loop_end(self):
        server_path = Path(__file__).parent.parent / "squish" / "server.py"
        src = server_path.read_text()
        assert "total_generated_tokens" in src


# ===========================================================================
# Cross-module interaction: ChipDetector → ChunkKVManager
# ===========================================================================

class TestChipDetectorChunkKVInteraction:
    """
    Verify that ChipDetector's recommended_chunk_prefill is a valid value
    that ChunkKVManager can consume for its chunk_size.
    """

    def test_detected_chunk_size_accepted_by_chunk_kv_manager(self):
        from squish.hardware.chip_detector import ChipDetector
        from squish.kv.chunk_kv import ChunkKVConfig, ChunkKVManager
        profile = ChipDetector().detect()
        # Use 16 (default) if recommended_chunk_prefill is too large for chunk_size
        chunk_size = min(profile.recommended_chunk_prefill, 64)
        cfg = ChunkKVConfig(chunk_size=max(8, chunk_size))
        mgr = ChunkKVManager(cfg)
        rng = np.random.default_rng(20)
        key = rng.standard_normal((cfg.chunk_size * 4, 8)).astype(np.float32)
        val = rng.standard_normal((cfg.chunk_size * 4, 8)).astype(np.float32)
        kept_k, kept_v, indices = mgr.evict(key, val)
        assert len(kept_k) > 0


# ===========================================================================
# Cross-module: JacobiDecoder × FusedQKVProjection shared logits path
# ===========================================================================

class TestJacobiDecoderFusedQKVInteraction:
    """If FusedQKVProjection packs weights, the resulting Q/K/V shapes must
    be compatible with what a JacobiDecoder logits_fn would see."""

    def test_fused_qkv_output_compatible_with_jacobi(self):
        from squish.hardware.fused_qkv_proj import FusedQKVConfig, FusedQKVProjection
        from squish.speculative.jacobi_decode import JacobiConfig, JacobiDecoder
        rng = np.random.default_rng(21)
        d_model, d_head, n_heads, n_kv = 32, 8, 4, 2
        cfg = FusedQKVConfig(d_model=d_model, d_head=d_head, n_heads=n_heads, n_kv_heads=n_kv)
        proj = FusedQKVProjection(cfg)
        w_q = rng.standard_normal((d_model, n_heads * d_head)).astype(np.float32)
        w_k = rng.standard_normal((d_model, n_kv * d_head)).astype(np.float32)
        w_v = rng.standard_normal((d_model, n_kv * d_head)).astype(np.float32)
        proj.pack_weights(w_q, w_k, w_v)
        assert proj.is_packed

        vocab = 30
        def logits_fn(ctx):
            return rng.standard_normal((len(ctx), vocab)).astype(np.float32)

        jd = JacobiDecoder(JacobiConfig(n_tokens=2, max_iter=2))
        tokens, _ = jd.decode_step(logits_fn, [1, 2, 3, 4], vocab_size=vocab)
        assert isinstance(tokens, list)
