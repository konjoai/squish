"""
tests/test_wave31_modules.py

Test suite for Wave 31 modules:
  - squish/kv/kvtc.py            (KVTransformCoder)
  - squish/kv/chunk_kv.py        (ChunkKVManager)
  - squish/speculative/ssd_saguaro.py (SSDSaguaro)
  - squish/vision/content_hash_cache.py (ContentHashImageCache)
  - squish/hardware/chip_detector.py   (ChipDetector)
"""

import numpy as np
import pytest

# ============================================================
# KVTC tests
# ============================================================

from squish.kv.kvtc import (
    KVTCConfig,
    KVTCEncoded,
    KVTCLayer,
    KVTCManager,
    KVTCStats,
)


class TestKVTCConfig:
    def test_default_config(self):
        cfg = KVTCConfig()
        assert cfg.rank == 64
        assert cfg.quant_bits == 8
        assert cfg.entropy_coding is True

    def test_invalid_rank(self):
        with pytest.raises(ValueError, match="rank"):
            KVTCConfig(rank=0)

    def test_invalid_quant_bits(self):
        with pytest.raises(ValueError, match="quant_bits"):
            KVTCConfig(quant_bits=16)

    def test_valid_4bit(self):
        cfg = KVTCConfig(quant_bits=4)
        assert cfg.quant_bits == 4


class TestKVTCLayer:
    def setup_method(self):
        self.rng = np.random.default_rng(42)
        self.cfg = KVTCConfig(rank=8, quant_bits=8)
        self.layer = KVTCLayer(self.cfg)
        self.d_kv = 64
        self.n_samples = 128
        self.samples = self.rng.normal(0, 1, (self.n_samples, self.d_kv)).astype(np.float32)

    def test_calibrate_basic(self):
        self.layer.calibrate(self.samples)
        assert self.layer._calibrated is True
        assert self.layer._components is not None
        assert self.layer._components.shape == (8, self.d_kv)
        assert self.layer._mean is not None

    def test_calibrate_requires_2d(self):
        with pytest.raises(ValueError, match="2-D"):
            self.layer.calibrate(np.zeros(10))

    def test_encode_requires_calibration(self):
        with pytest.raises(RuntimeError, match="calibrated"):
            self.layer.encode(self.samples[:4])

    def test_encode_output_shape(self):
        self.layer.calibrate(self.samples)
        seq_len = 32
        kv = self.rng.normal(0, 1, (seq_len, self.d_kv)).astype(np.float32)
        enc = self.layer.encode(kv)
        assert isinstance(enc, KVTCEncoded)
        assert enc.codes.shape[0] == seq_len
        assert enc.original_shape == kv.shape
        assert enc.rank_used == 8

    def test_encode_decode_roundtrip(self):
        # Use rank=16 for reliable reconstruction on d_kv=64
        layer = KVTCLayer(KVTCConfig(rank=16, quant_bits=8))
        layer.calibrate(self.samples)
        kv = self.rng.normal(0, 1, (16, self.d_kv)).astype(np.float32)
        enc = layer.encode(kv)
        rec = layer.decode(enc)
        assert rec.shape == kv.shape
        # Low-rank reconstruction: relative error should be < 1.0
        err = np.linalg.norm(rec - kv) / np.linalg.norm(kv)
        assert err < 1.0

    def test_encode_decode_symmetric(self):
        self.layer.config.symmetric = True
        self.layer.calibrate(self.samples)
        kv = self.rng.normal(0, 1, (8, self.d_kv)).astype(np.float32)
        enc = self.layer.encode(kv)
        rec = self.layer.decode(enc)
        assert rec.shape == kv.shape

    def test_nbytes_smaller_than_original(self):
        self.layer.calibrate(self.samples)
        kv = self.rng.normal(0, 1, (256, self.d_kv)).astype(np.float32)
        enc = self.layer.encode(kv)
        assert enc.nbytes() < kv.nbytes


class TestKVTCManager:
    def setup_method(self):
        self.rng = np.random.default_rng(0)
        self.n_layers = 4
        self.d_kv = 32
        self.n_samples = 64
        self.cfg = KVTCConfig(rank=4, quant_bits=8)
        self.manager = KVTCManager(self.cfg, self.n_layers)

    def _samples(self):
        return self.rng.normal(0, 1, (self.n_samples, self.d_kv)).astype(np.float32)

    def test_init(self):
        assert self.manager.n_layers == self.n_layers
        assert isinstance(self.manager.stats, KVTCStats)

    def test_calibrate_single_layer(self):
        k_samp = self._samples()
        v_samp = self._samples()
        self.manager.calibrate_layer(0, k_samp, v_samp)
        assert self.manager.stats.calibration_calls == 1

    def test_calibrate_all(self):
        all_k = {i: self._samples() for i in range(self.n_layers)}
        all_v = {i: self._samples() for i in range(self.n_layers)}
        self.manager.calibrate_all(all_k, all_v)
        assert self.manager.stats.calibration_calls == self.n_layers

    def test_encode_decode_roundtrip(self):
        all_k = {i: self._samples() for i in range(self.n_layers)}
        all_v = {i: self._samples() for i in range(self.n_layers)}
        self.manager.calibrate_all(all_k, all_v)

        k = self.rng.normal(0, 1, (16, self.d_kv)).astype(np.float32)
        v = self.rng.normal(0, 1, (16, self.d_kv)).astype(np.float32)
        enc_k, enc_v = self.manager.encode_layer(0, k, v)
        k_rec, v_rec = self.manager.decode_layer(0, enc_k, enc_v)

        assert k_rec.shape == k.shape
        assert v_rec.shape == v.shape
        assert self.manager.stats.encode_calls == 1
        assert self.manager.stats.decode_calls == 1

    def test_stats_track_bytes(self):
        all_k = {0: self._samples()}
        all_v = {0: self._samples()}
        self.manager.calibrate_all(all_k, all_v)
        k = self.rng.normal(0, 1, (32, self.d_kv)).astype(np.float32)
        v = self.rng.normal(0, 1, (32, self.d_kv)).astype(np.float32)
        self.manager.encode_layer(0, k, v)
        assert self.manager.stats.total_bytes_in > 0
        assert self.manager.stats.total_bytes_out > 0

    def test_compression_ratio_positive(self):
        all_k = {0: self._samples()}
        all_v = {0: self._samples()}
        self.manager.calibrate_all(all_k, all_v)
        k = self.rng.normal(0, 1, (128, self.d_kv)).astype(np.float32)
        v = self.rng.normal(0, 1, (128, self.d_kv)).astype(np.float32)
        self.manager.encode_layer(0, k, v)
        assert self.manager.compression_ratio() > 1.0

    def test_repr_contains_layers(self):
        assert "4" in repr(self.manager)


# ============================================================
# ChunkKV tests
# ============================================================

from squish.kv.chunk_kv import (
    ChunkKVConfig,
    ChunkKVManager,
    ChunkKVOrchestrator,
    ChunkKVStats,
    ChunkScore,
)


class TestChunkKVConfig:
    def test_defaults(self):
        cfg = ChunkKVConfig()
        assert cfg.chunk_size == 16
        assert 0 < cfg.budget_ratio <= 1.0

    def test_invalid_chunk_size(self):
        with pytest.raises(ValueError, match="chunk_size"):
            ChunkKVConfig(chunk_size=0)

    def test_invalid_budget_ratio(self):
        with pytest.raises(ValueError, match="budget_ratio"):
            ChunkKVConfig(budget_ratio=0.0)

    def test_invalid_score_fn(self):
        with pytest.raises(ValueError, match="score_fn"):
            ChunkKVConfig(score_fn="unknown")


class TestChunkKVManager:
    def setup_method(self):
        self.rng = np.random.default_rng(7)
        self.cfg = ChunkKVConfig(chunk_size=8, budget_ratio=0.5)
        self.mgr = ChunkKVManager(self.cfg)
        self.seq_len = 64
        self.d_k = 32
        self.key = self.rng.normal(0, 1, (self.seq_len, self.d_k)).astype(np.float32)
        self.val = self.rng.normal(0, 1, (self.seq_len, self.d_k)).astype(np.float32)

    def test_score_chunks_count(self):
        import math
        scores = self.mgr.score_chunks(self.key)
        expected = math.ceil(self.seq_len / self.cfg.chunk_size)
        assert len(scores) == expected
        assert all(isinstance(s, ChunkScore) for s in scores)

    def test_score_chunks_with_query(self):
        q = self.rng.normal(0, 1, (self.d_k,)).astype(np.float32)
        scores = self.mgr.score_chunks(self.key, q)
        assert len(scores) > 0

    def test_evict_reduces_sequence(self):
        k2, v2, idx = self.mgr.evict(self.key, self.val)
        assert k2.shape[0] < self.seq_len
        assert v2.shape[0] == k2.shape[0]
        assert len(idx) == k2.shape[0]

    def test_evict_idx_in_bounds(self):
        _, _, idx = self.mgr.evict(self.key, self.val)
        assert idx.max() < self.seq_len
        assert idx.min() >= 0

    def test_stats_update(self):
        self.mgr.evict(self.key, self.val)
        assert self.mgr.stats.eviction_calls == 1
        assert self.mgr.stats.tokens_kept + self.mgr.stats.tokens_evicted == self.seq_len

    def test_reuse_cache_hit(self):
        # First evict populates cache
        _, _, idx1 = self.mgr.evict(self.key, self.val, layer_idx=3)
        # Second evict with adjacent layer should be a reuse hit
        _, _, idx2 = self.mgr.evict(self.key, self.val, layer_idx=4)
        assert self.mgr.stats.reuse_hits >= 1

    def test_invalidate_reuse_cache(self):
        self.mgr.evict(self.key, self.val, layer_idx=0)
        self.mgr.invalidate_reuse_cache()
        assert self.mgr._cached_indices is None

    def test_norm_score_fn(self):
        mgr = ChunkKVManager(ChunkKVConfig(chunk_size=8, score_fn="norm"))
        scores = mgr.score_chunks(self.key)
        assert all(s.score >= 0 for s in scores)

    def test_repr(self):
        r = repr(self.mgr)
        assert "ChunkKVManager" in r


class TestChunkKVOrchestrator:
    def setup_method(self):
        self.rng = np.random.default_rng(3)
        self.cfg = ChunkKVConfig(chunk_size=8, budget_ratio=0.5, layer_reuse=True)
        self.orch = ChunkKVOrchestrator(self.cfg, n_layers=4)
        self.key = self.rng.normal(0, 1, (64, 32)).astype(np.float32)
        self.val = self.rng.normal(0, 1, (64, 32)).astype(np.float32)

    def test_evict_all_layers(self):
        for i in range(4):
            k2, v2, idx = self.orch.evict_layer(i, self.key, self.val)
            assert k2.shape[0] <= self.key.shape[0]

    def test_reset_request(self):
        for i in range(4):
            self.orch.evict_layer(i, self.key, self.val)
        self.orch.reset_request()
        for mgr in self.orch._managers.values():
            assert mgr._cached_indices is None

    def test_aggregate_stats(self):
        for i in range(4):
            self.orch.evict_layer(i, self.key, self.val)
        stats = self.orch.aggregate_stats
        assert stats.eviction_calls == 4


# ============================================================
# SSDSaguaro tests
# ============================================================

from squish.speculative.ssd_saguaro import (
    SSDConfig,
    SSDOutcome,
    SSDSaguaro,
    SSDStats,
    VerifyResult,
)


class TestSSDConfig:
    def test_defaults(self):
        cfg = SSDConfig()
        assert cfg.k_outcomes == 4
        assert cfg.draft_len == 8

    def test_invalid_k_outcomes(self):
        with pytest.raises(ValueError, match="k_outcomes"):
            SSDConfig(k_outcomes=0)

    def test_invalid_draft_len(self):
        with pytest.raises(ValueError, match="draft_len"):
            SSDConfig(draft_len=0)


class TestSSDSaguaro:
    def setup_method(self):
        self.rng = np.random.default_rng(11)
        self.cfg = SSDConfig(k_outcomes=3, draft_len=5)
        self.ssd = SSDSaguaro(self.cfg)
        self.vocab = 200
        self.draft_len = 5

    def _logits(self, focus_tok: int) -> np.ndarray:
        logits = self.rng.normal(0, 0.1, (self.draft_len, self.vocab)).astype(np.float32)
        logits[:, focus_tok] += 5.0
        return logits

    def test_predict_outcomes_returns_list(self):
        d_logits = self._logits(10)
        t_logits = self._logits(10)
        outcomes = self.ssd.predict_outcomes(d_logits, t_logits)
        assert isinstance(outcomes, list)
        assert all(isinstance(o, SSDOutcome) for o in outcomes)
        assert len(outcomes) <= self.cfg.k_outcomes

    def test_predict_outcomes_probs_sum_approx_one(self):
        d_logits = self._logits(10)
        t_logits = self._logits(10)
        outcomes = self.ssd.predict_outcomes(d_logits, t_logits)
        total_prob = sum(o.probability for o in outcomes)
        # Outcomes are top-k so total prob <= 1
        assert 0.0 <= total_prob <= 1.0 + 1e-6

    def test_prefetch_outcomes_builds_table(self):
        context = list(range(10))
        draft = [5, 3, 7, 2, 8]

        def draft_fn(ctx):
            return [1, 2, 3, 4, 5]

        table = self.ssd.prefetch_outcomes(context, draft, draft_fn)
        assert isinstance(table, dict)
        assert len(table) > 0
        assert all(isinstance(v.tokens, list) for v in table.values())

    def test_verify_and_select_basic(self):
        draft = [10, 10, 10, 10, 10]
        # main model strongly agrees on tok 10
        main_logits = self.rng.normal(0, 0.1, (self.draft_len, self.vocab)).astype(np.float32)
        main_logits[:, 10] += 10.0
        prefetches = {i: type("E", (), {"tokens": [1, 2, 3]})() for i in range(6)}
        result = self.ssd.verify_and_select(main_logits, draft, prefetches)
        assert isinstance(result, VerifyResult)
        assert result.accepted_len >= 0
        assert isinstance(result.prefetch_hit, bool)

    def test_stats_update(self):
        draft = [0, 1, 2, 3, 4]
        main_logits = self.rng.normal(0, 1, (self.draft_len, self.vocab)).astype(np.float32)
        self.ssd.verify_and_select(main_logits, draft, {})
        assert self.ssd.stats.decode_steps == 1
        assert self.ssd.stats.total_tokens_generated > 0

    def test_stats_prefetch_hit_rate(self):
        stats = SSDStats(prefetch_hits=3, prefetch_misses=7)
        assert abs(stats.prefetch_hit_rate - 0.3) < 1e-9

    def test_mean_tokens_per_step(self):
        stats = SSDStats(decode_steps=10, total_tokens_generated=35)
        assert abs(stats.mean_tokens_per_step - 3.5) < 1e-9

    def test_repr(self):
        r = repr(self.ssd)
        assert "SSDSaguaro" in r


# ============================================================
# ContentHashImageCache tests
# ============================================================

from squish.vision.content_hash_cache import (
    ContentHashCacheConfig,
    ContentHashCacheStats,
    ContentHashImageCache,
)


class TestContentHashCacheConfig:
    def test_defaults(self):
        cfg = ContentHashCacheConfig()
        assert cfg.max_entries == 256
        assert cfg.hash_fn == "sha256"

    def test_invalid_max_entries(self):
        with pytest.raises(ValueError, match="max_entries"):
            ContentHashCacheConfig(max_entries=0)

    def test_invalid_hash_fn(self):
        with pytest.raises(ValueError, match="hash_fn"):
            ContentHashCacheConfig(hash_fn="md5")


class TestContentHashImageCache:
    def setup_method(self):
        self.cfg = ContentHashCacheConfig(max_entries=10)
        self.cache = ContentHashImageCache(self.cfg)
        self.rng = np.random.default_rng(99)

    def _img(self, n: int = 100) -> bytes:
        return bytes(self.rng.integers(0, 256, n, dtype=np.uint8))

    def _kv(self) -> np.ndarray:
        return self.rng.normal(0, 1, (16, 64)).astype(np.float32)

    def test_miss_on_empty(self):
        result = self.cache.lookup(self._img())
        assert result is None
        assert self.cache.stats.misses == 1

    def test_store_and_hit(self):
        img = self._img()
        kv = self._kv()
        self.cache.store(img, kv)
        result = self.cache.lookup(img)
        assert result is not None
        np.testing.assert_array_equal(result, kv)
        assert self.cache.stats.hits == 1

    def test_different_images_different_keys(self):
        img1 = bytes([1, 2, 3])
        img2 = bytes([4, 5, 6])
        kv1 = self._kv()
        kv2 = self._kv()
        self.cache.store(img1, kv1)
        self.cache.store(img2, kv2)
        r1 = self.cache.lookup(img1)
        r2 = self.cache.lookup(img2)
        assert r1 is not None and r2 is not None
        assert not np.array_equal(r1, r2)

    def test_lru_eviction_at_capacity(self):
        # Fill to capacity
        images = [bytes([i]) for i in range(10)]
        for i, img in enumerate(images):
            self.cache.store(img, self._kv())
        assert self.cache.size == 10
        # One more → eldest evicted
        self.cache.store(bytes([99]), self._kv())
        assert self.cache.size == 10
        # eldest (images[0]) should be gone
        assert self.cache.lookup(images[0]) is None

    def test_hash_deterministic(self):
        img = b"test_image_bytes"
        h1 = self.cache.hash_image(img)
        h2 = self.cache.hash_image(img)
        assert h1 == h2

    def test_evict_lru_manual(self):
        img = self._img()
        kv = self._kv()
        self.cache.store(img, kv)
        freed = self.cache.evict_lru()
        assert freed > 0
        assert self.cache.size == 0

    def test_clear(self):
        for _ in range(5):
            self.cache.store(self._img(), self._kv())
        self.cache.clear()
        assert self.cache.size == 0

    def test_bytes_cached(self):
        self.cache.store(self._img(), self._kv())
        assert self.cache.bytes_cached > 0

    def test_stats_repr(self):
        r = repr(self.cache.stats)
        assert "ContentHashCacheStats" in r

    def test_ttl_expiry(self):
        import time
        cfg = ContentHashCacheConfig(max_entries=10, ttl_seconds=0.001)
        cache = ContentHashImageCache(cfg)
        img = b"expire_test"
        cache.store(img, self._kv())
        time.sleep(0.01)  # let TTL expire
        result = cache.lookup(img)
        assert result is None

    def test_repr(self):
        r = repr(self.cache)
        assert "ContentHashImageCache" in r


# ============================================================
# ChipDetector tests
# ============================================================

from squish.hardware.chip_detector import (
    AppleChipGeneration,
    ChipDetector,
    ChipProfile,
    CHIP_PROFILES,
)


class TestChipDetectorParsing:
    @pytest.mark.parametrize("chip_str,expected", [
        ("Apple M1", AppleChipGeneration.M1),
        ("Apple M2 Pro", AppleChipGeneration.M2),
        ("Apple M3 Max", AppleChipGeneration.M3),
        ("Apple M4", AppleChipGeneration.M4),
        ("Apple M5 Ultra", AppleChipGeneration.M5),
        ("Intel Core i9", AppleChipGeneration.UNKNOWN),
        ("", AppleChipGeneration.UNKNOWN),
    ])
    def test_parse_generation(self, chip_str, expected):
        result = ChipDetector._parse_generation(chip_str)
        assert result == expected

    def test_m5_not_matched_as_m1(self):
        gen = ChipDetector._parse_generation("Apple M5")
        assert gen == AppleChipGeneration.M5


class TestChipProfiles:
    def test_all_generations_present(self):
        for gen in [AppleChipGeneration.M1, AppleChipGeneration.M2,
                    AppleChipGeneration.M3, AppleChipGeneration.M4,
                    AppleChipGeneration.M5]:
            assert gen in CHIP_PROFILES

    def test_bandwidth_increases_with_generation(self):
        m3_bw = CHIP_PROFILES[AppleChipGeneration.M3].memory_bandwidth_gbps
        m4_bw = CHIP_PROFILES[AppleChipGeneration.M4].memory_bandwidth_gbps
        m5_bw = CHIP_PROFILES[AppleChipGeneration.M5].memory_bandwidth_gbps
        assert m5_bw > m4_bw > m3_bw

    def test_m5_bandwidth_is_153(self):
        assert CHIP_PROFILES[AppleChipGeneration.M5].memory_bandwidth_gbps == 153.0


class TestChipDetector:
    @pytest.mark.parametrize("override,expected_gen", [
        ("Apple M1", AppleChipGeneration.M1),
        ("Apple M3 Max", AppleChipGeneration.M3),
        ("Apple M5 Ultra", AppleChipGeneration.M5),
    ])
    def test_detect_override(self, override, expected_gen):
        det = ChipDetector(_override=override)
        profile = det.detect()
        assert profile.generation == expected_gen

    def test_detect_caches_result(self):
        det = ChipDetector(_override="Apple M4")
        p1 = det.detect()
        p2 = det.detect()
        assert p1 is p2

    def test_unknown_platform(self):
        det = ChipDetector(_override="Not an Apple chip")
        profile = det.detect()
        assert profile.generation == AppleChipGeneration.UNKNOWN

    def test_chunk_size_scales_with_model_size(self):
        det = ChipDetector(_override="Apple M3")
        small = det.get_optimal_chunk_size(7.0)
        large = det.get_optimal_chunk_size(30.0)
        assert large <= small

    def test_kv_bits_drop_for_low_ram(self):
        det = ChipDetector(_override="Apple M5")
        bits = det.get_recommended_kv_bits(8.0)   # < 12 GB
        assert bits == 4

    def test_kv_bits_uses_profile_for_normal_ram(self):
        det = ChipDetector(_override="Apple M3")
        bits = det.get_recommended_kv_bits(24.0)
        assert bits == CHIP_PROFILES[AppleChipGeneration.M3].recommended_kv_bits

    def test_metal_dispatch_unrestricted(self):
        det = ChipDetector(_override="Apple M4")
        assert det.should_enable_metal_dispatch() is True

    def test_bandwidth_ratio_vs_m3(self):
        det = ChipDetector(_override="Apple M5")
        ratio = det.bandwidth_ratio_vs_m3()
        assert ratio == pytest.approx(153.0 / 100.0)

    def test_bandwidth_ratio_m3_is_one(self):
        det = ChipDetector(_override="Apple M3")
        assert det.bandwidth_ratio_vs_m3() == pytest.approx(1.0)

    def test_repr(self):
        det = ChipDetector(_override="Apple M3")
        r = repr(det)
        assert "ChipDetector" in r
        assert "M3" in r
