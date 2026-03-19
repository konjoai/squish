"""tests/test_wave27_server_wiring.py

Wave 27 Phase 1 wiring tests — verifies that the five modules enabled in
server.py (chunked prefill, FusedSampler, CacheWarmup, TokenMerging,
LayerSkip) import cleanly, have the expected public API, and behave
correctly at the unit level.

Modules under test
------------------
- squish.streaming.chunked_prefill (Step 1A)
- squish.hardware.fused_sampler    (Step 1B)
- squish.kv.cache_warmup           (Step 1C)
- squish.token.token_merging       (Step 1D)
- squish.token.layer_skip          (Step 1E)
"""

from __future__ import annotations

import math

import numpy as np
import pytest

# ============================================================
# Step 1A — ChunkedPrefill (streaming/chunked_prefill.py)
# ============================================================


class TestChunkedPrefillImport:
    def test_import(self):
        from squish.streaming.chunked_prefill import ChunkedPrefillConfig  # noqa: F401

    def test_config_defaults(self):
        from squish.streaming.chunked_prefill import ChunkedPrefillConfig
        cfg = ChunkedPrefillConfig()
        assert cfg.chunk_size > 0

    def test_config_custom(self):
        from squish.streaming.chunked_prefill import ChunkedPrefillConfig
        cfg = ChunkedPrefillConfig(chunk_size=256)
        assert cfg.chunk_size == 256

    def test_chunk_prefill_callable(self):
        from squish.streaming.chunked_prefill import chunk_prefill
        assert callable(chunk_prefill)


# ============================================================
# Step 1B — FusedSampler (hardware/fused_sampler.py)
# ============================================================


class TestFusedSamplerImport:
    def test_import(self):
        from squish.hardware.fused_sampler import FusedSampler, SamplerConfig  # noqa: F401

    def test_config_defaults(self):
        from squish.hardware.fused_sampler import SamplerConfig
        cfg = SamplerConfig()
        assert cfg.temperature >= 0.0
        assert 0.0 < cfg.top_p <= 1.0

    def test_instantiate(self):
        from squish.hardware.fused_sampler import FusedSampler, SamplerConfig
        cfg = SamplerConfig(temperature=1.0, top_p=0.9)
        sampler = FusedSampler(cfg)
        assert sampler is not None

    def test_sample_near_greedy(self):
        from squish.hardware.fused_sampler import FusedSampler, SamplerConfig
        # Use a very low temperature and a dominant logit — should always pick 7
        cfg     = SamplerConfig(temperature=0.01, top_p=1.0)
        sampler = FusedSampler(cfg)
        logits  = np.zeros(64, dtype=np.float32)
        logits[7] = 200.0  # extremely dominant
        toks = {sampler.sample(logits) for _ in range(20)}
        assert toks == {7}

    def test_sample_returns_valid_index(self):
        from squish.hardware.fused_sampler import FusedSampler, SamplerConfig
        vocab   = 100
        cfg     = SamplerConfig(temperature=1.0, top_p=1.0)
        sampler = FusedSampler(cfg)
        rng     = np.random.default_rng(42)
        logits  = rng.standard_normal(vocab).astype(np.float32)
        tok = sampler.sample(logits)
        assert isinstance(tok, int)
        assert 0 <= tok < vocab

    def test_sample_invalid_temperature_raises(self):
        from squish.hardware.fused_sampler import SamplerConfig
        with pytest.raises(Exception):
            SamplerConfig(temperature=-1.0)


# ============================================================
# Step 1C — CacheWarmup (kv/cache_warmup.py)
# ============================================================


class TestCacheWarmupImport:
    def test_import(self):
        from squish.kv.cache_warmup import CacheWarmupPredictor, WarmupConfig  # noqa: F401

    def test_config_defaults(self):
        from squish.kv.cache_warmup import WarmupConfig
        cfg = WarmupConfig()
        assert cfg.top_k > 0

    def test_instantiate(self):
        from squish.kv.cache_warmup import CacheWarmupPredictor, WarmupConfig
        predictor = CacheWarmupPredictor(WarmupConfig())
        assert predictor is not None

    def test_record_access_and_candidates(self):
        import time
        from squish.kv.cache_warmup import CacheWarmupPredictor, WarmupConfig
        predictor = CacheWarmupPredictor(WarmupConfig(top_k=4))
        prefix = list(range(16))
        predictor.record_access(prefix, time.monotonic())
        candidates = predictor.get_warmup_candidates()
        assert isinstance(candidates, list)

    def test_record_access_empty_raises(self):
        import time
        from squish.kv.cache_warmup import CacheWarmupPredictor, WarmupConfig
        predictor = CacheWarmupPredictor(WarmupConfig())
        # CacheWarmupPredictor requires a non-empty prefix
        with pytest.raises(ValueError):
            predictor.record_access([], time.monotonic())

    def test_multiple_accesses(self):
        import time
        from squish.kv.cache_warmup import CacheWarmupPredictor, WarmupConfig
        predictor = CacheWarmupPredictor(WarmupConfig(top_k=8))
        t0 = time.monotonic()
        for i in range(10):
            predictor.record_access(list(range(i, i + 16)), t0 + i)
        candidates = predictor.get_warmup_candidates()
        assert isinstance(candidates, list)


# ============================================================
# Step 1D — TokenMerging (token/token_merging.py)
# ============================================================


class TestTokenMergingImport:
    def test_import(self):
        from squish.token.token_merging import TokenMergingConfig, TokenMergingState  # noqa: F401

    def test_config_defaults(self):
        from squish.token.token_merging import TokenMergingConfig
        cfg = TokenMergingConfig()
        assert cfg.r > 0

    def test_config_custom(self):
        from squish.token.token_merging import TokenMergingConfig
        cfg = TokenMergingConfig(r=8, start_layer=2, end_layer=9)
        assert cfg.r == 8
        assert cfg.start_layer == 2

    def test_state_instantiate(self):
        from squish.token.token_merging import TokenMergingState
        state = TokenMergingState()
        assert state is not None

    def test_state_reset(self):
        from squish.token.token_merging import TokenMergingState
        state = TokenMergingState()
        state.reset()  # should not raise

    def test_patch_callable(self):
        from squish.token.token_merging import patch_model_tome, unpatch_model_tome
        assert callable(patch_model_tome)
        assert callable(unpatch_model_tome)

    def test_bipartite_merge_callable(self):
        from squish.token.token_merging import bipartite_merge
        assert callable(bipartite_merge)

    def test_invalid_r_raises(self):
        from squish.token.token_merging import TokenMergingConfig
        # r=0 is valid (means no merging); r=-1 should raise
        with pytest.raises(ValueError):
            TokenMergingConfig(r=-1)


# ============================================================
# Step 1E — LayerSkip (token/layer_skip.py)
# ============================================================


class TestLayerSkipImport:
    def test_import(self):
        from squish.token.layer_skip import EarlyExitConfig, ConfidenceEstimator  # noqa: F401

    def test_config_defaults(self):
        from squish.token.layer_skip import EarlyExitConfig
        cfg = EarlyExitConfig()
        assert 0.0 < cfg.confidence_threshold <= 1.0
        assert cfg.exit_layer >= 1

    def test_config_invalid_exit_layer(self):
        from squish.token.layer_skip import EarlyExitConfig
        with pytest.raises(ValueError):
            EarlyExitConfig(num_layers=8, exit_layer=8)

    def test_config_invalid_threshold(self):
        from squish.token.layer_skip import EarlyExitConfig
        with pytest.raises(ValueError):
            EarlyExitConfig(confidence_threshold=1.5)

    def test_confidence_estimator_max_prob(self):
        from squish.token.layer_skip import ConfidenceEstimator
        est    = ConfidenceEstimator("max_prob")
        logits = np.array([0.0, 0.0, 10.0, 0.0], dtype=np.float64)
        score  = est.estimate(logits)
        assert 0.0 <= score <= 1.0
        assert score > 0.9  # near-deterministic

    def test_confidence_estimator_margin(self):
        from squish.token.layer_skip import ConfidenceEstimator
        est    = ConfidenceEstimator("margin")
        logits = np.array([10.0, 9.0, 0.0, 0.0], dtype=np.float64)
        score  = est.estimate(logits)
        assert 0.0 <= score <= 1.0

    def test_confidence_estimator_neg_entropy(self):
        from squish.token.layer_skip import ConfidenceEstimator
        est    = ConfidenceEstimator("neg_entropy")
        logits = np.zeros(100, dtype=np.float64)  # uniform → low confidence
        score  = est.estimate(logits)
        assert 0.0 <= score <= 1.0

    def test_confidence_estimator_top_token(self):
        from squish.token.layer_skip import ConfidenceEstimator
        est    = ConfidenceEstimator()
        logits = np.array([0.0, 0.0, 0.0, 100.0], dtype=np.float64)
        assert est.top_token(logits) == 3

    def test_invalid_metric_raises(self):
        from squish.token.layer_skip import ConfidenceEstimator
        with pytest.raises(ValueError):
            ConfidenceEstimator("invalid_metric")
