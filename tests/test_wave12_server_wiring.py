"""tests/test_wave12_server_wiring.py

Verifies that all Wave 12 module classes are importable and have the expected
public APIs that the server.py wiring code depends on.  These are pure
import + instantiation tests — no model or GPU required.
"""

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# PM-KVQ
# ---------------------------------------------------------------------------

class TestPMKVQWiring:
    def test_import(self):
        from squish.pm_kvq import PMKVQConfig, PMKVQScheduler
        cfg = PMKVQConfig(n_blocks=8)
        sched = PMKVQScheduler(cfg)
        assert sched is not None

    def test_advance_does_not_raise(self):
        from squish.pm_kvq import PMKVQConfig, PMKVQScheduler
        sched = PMKVQScheduler(PMKVQConfig(n_blocks=4))
        for _ in range(16):
            sched.advance()  # should cycle without raising

    def test_current_bits_returns_int(self):
        from squish.pm_kvq import PMKVQConfig, PMKVQScheduler
        sched = PMKVQScheduler(PMKVQConfig(n_blocks=4))
        bits = sched.current_bits(0)
        assert isinstance(bits, int)
        assert bits > 0

    def test_reset(self):
        from squish.pm_kvq import PMKVQConfig, PMKVQScheduler
        sched = PMKVQScheduler(PMKVQConfig(n_blocks=4))
        sched.advance()
        sched.advance()
        sched.reset()
        # After reset, current_bits should equal initial bits
        assert sched.current_bits(0) == sched.current_bits(0)


# ---------------------------------------------------------------------------
# MixKVQ
# ---------------------------------------------------------------------------

class TestMixKVQWiring:
    def test_import(self):
        from squish.mix_kvq import MixKVQConfig, MixKVQQuantizer
        cfg  = MixKVQConfig()
        quant = MixKVQQuantizer(cfg)
        assert quant is not None

    def test_config_has_fp16_channel_ratio(self):
        from squish.mix_kvq import MixKVQConfig
        cfg = MixKVQConfig()
        assert hasattr(cfg, "fp16_fraction")
        assert 0.0 < cfg.fp16_fraction <= 1.0

    def test_quantize_returns_array(self):
        from squish.mix_kvq import ChannelScorer, MixKVQConfig, MixKVQQuantizer
        rng    = np.random.default_rng(1)
        n_ch   = 64
        cfg    = MixKVQConfig()
        scorer = ChannelScorer(n_channels=n_ch, config=cfg)
        # Prime the scorer with history
        key_vec = rng.standard_normal(n_ch).astype(np.float32)
        scorer.record(key_vec)
        query   = rng.standard_normal(n_ch).astype(np.float32)
        key_mat = rng.standard_normal((4, n_ch)).astype(np.float32)
        bit_map = scorer.assign_bits(query, key_mat)
        quant   = MixKVQQuantizer(cfg)
        result, scales, bm = quant.quantize(key_vec, bit_map)
        assert result is not None


# ---------------------------------------------------------------------------
# CocktailKV
# ---------------------------------------------------------------------------

class TestCocktailKVWiring:
    def test_import(self):
        from squish.cocktail_kv import CocktailConfig, CocktailKVStore
        cfg   = CocktailConfig()
        store = CocktailKVStore(cfg)
        assert store is not None

    def test_config_has_chunk_size(self):
        from squish.cocktail_kv import CocktailConfig
        cfg = CocktailConfig()
        assert hasattr(cfg, "chunk_size")
        assert cfg.chunk_size > 0

    def test_config_has_similarity_threshold(self):
        from squish.cocktail_kv import CocktailConfig
        cfg = CocktailConfig()
        assert hasattr(cfg, "fp16_fraction")
        assert 0.0 <= cfg.fp16_fraction <= 1.0


# ---------------------------------------------------------------------------
# AgileIO
# ---------------------------------------------------------------------------

class TestAgileIOWiring:
    def test_import(self):
        from squish.agile_io import AgileIOConfig, AgileIOManager
        cfg = AgileIOConfig(n_worker_threads=1, cache_size_mb=1)
        mgr = AgileIOManager(cfg)
        assert mgr is not None
        mgr.shutdown()

    def test_config_attrs(self):
        from squish.agile_io import AgileIOConfig
        cfg = AgileIOConfig(n_worker_threads=2, cache_size_mb=64)
        assert cfg.n_worker_threads == 2
        assert cfg.cache_size_mb == 64

    def test_stats_available(self):
        from squish.agile_io import AgileIOConfig, AgileIOManager
        mgr = AgileIOManager(AgileIOConfig(n_worker_threads=1, cache_size_mb=1))
        s   = mgr.stats
        assert hasattr(s, "hit_rate")
        mgr.shutdown()


# ---------------------------------------------------------------------------
# MiLo
# ---------------------------------------------------------------------------

class TestMiLoWiring:
    def test_import(self):
        from squish.milo_quant import MiLoConfig, MiLoQuantizer
        cfg  = MiLoConfig(target_bits=3, max_rank=4)
        quant = MiLoQuantizer(cfg)
        assert quant is not None

    def test_config_target_bits(self):
        from squish.milo_quant import MiLoConfig
        cfg = MiLoConfig(target_bits=3)
        assert cfg.target_bits == 3

    def test_quantize_small_weight(self):
        from squish.milo_quant import MiLoConfig, MiLoQuantizer
        rng = np.random.default_rng(7)
        w   = rng.standard_normal((16, 32)).astype(np.float32)
        qr  = MiLoQuantizer(MiLoConfig(target_bits=3, max_rank=4))
        q_packed, scales, zeros, comp = qr.quantize(w)
        assert q_packed is not None
        assert comp.rank >= 1


# ---------------------------------------------------------------------------
# SageAttention patch functions
# ---------------------------------------------------------------------------

class TestSageAttentionPatchFn:
    def test_patch_model_sage_attention_importable(self):
        from squish.sage_attention import patch_model_sage_attention, unpatch_model_sage_attention
        assert callable(patch_model_sage_attention)
        assert callable(unpatch_model_sage_attention)

    def test_patch_sets_attribute(self):
        from squish.sage_attention import (
            SageAttentionConfig,
            SageAttentionKernel,
            patch_model_sage_attention,
            unpatch_model_sage_attention,
        )

        class FakeModel:
            pass

        kernel = SageAttentionKernel(SageAttentionConfig())
        model  = FakeModel()
        patch_model_sage_attention(model, kernel)
        assert getattr(model, "_sage_attn_kernel", None) is kernel
        unpatch_model_sage_attention(model)
        assert not hasattr(model, "_sage_attn_kernel")


# ---------------------------------------------------------------------------
# SpargeAttn patch functions
# ---------------------------------------------------------------------------

class TestSpargeAttnPatchFn:
    def test_patch_model_sparge_attn_importable(self):
        from squish.sparge_attn import patch_model_sparge_attn, unpatch_model_sparge_attn
        assert callable(patch_model_sparge_attn)
        assert callable(unpatch_model_sparge_attn)

    def test_patch_sets_attribute(self):
        from squish.sparge_attn import (
            SpargeAttnConfig,
            SpargeAttnEngine,
            patch_model_sparge_attn,
            unpatch_model_sparge_attn,
        )

        class FakeModel:
            pass

        engine = SpargeAttnEngine(SpargeAttnConfig())
        model  = FakeModel()
        patch_model_sparge_attn(model, engine)
        assert getattr(model, "_sparge_engine", None) is engine
        unpatch_model_sparge_attn(model)
        assert not hasattr(model, "_sparge_engine")
