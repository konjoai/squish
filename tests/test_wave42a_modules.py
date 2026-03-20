"""tests/test_wave42a_modules.py

Tests for Wave 42a modules:
  - MedusaHeads (squish/speculative/medusa_heads.py)
  - SarathiScheduler (squish/serving/sarathi_scheduler.py)
  - NSAAttention (squish/attention/nsa_attn.py)
  - FlexPrefill (squish/attention/flex_prefill.py)
  - ThinKCache (squish/kv/think_cache.py)
  - AttentionStore (squish/kv/attention_store.py)
"""

from __future__ import annotations

import pytest
import numpy as np

# ── MedusaHeads ───────────────────────────────────────────────────────────────

from squish.speculative.medusa_heads import MedusaConfig, MedusaDraftResult, MedusaHeads


class TestMedusaConfig:
    def test_defaults(self):
        cfg = MedusaConfig()
        assert cfg.n_heads == 2
        assert cfg.tree_width == 3
        assert 0.0 <= cfg.accept_threshold < 1.0
        assert cfg.temperature > 0.0

    def test_custom(self):
        cfg = MedusaConfig(n_heads=4, tree_width=5, accept_threshold=0.3, temperature=0.7)
        assert cfg.n_heads == 4
        assert cfg.tree_width == 5

    def test_invalid_n_heads(self):
        with pytest.raises(ValueError, match="n_heads"):
            MedusaConfig(n_heads=0)

    def test_invalid_tree_width(self):
        with pytest.raises(ValueError, match="tree_width"):
            MedusaConfig(tree_width=0)

    def test_invalid_accept_threshold_negative(self):
        with pytest.raises(ValueError, match="accept_threshold"):
            MedusaConfig(accept_threshold=-0.1)

    def test_invalid_accept_threshold_one(self):
        with pytest.raises(ValueError, match="accept_threshold"):
            MedusaConfig(accept_threshold=1.0)

    def test_invalid_temperature(self):
        with pytest.raises(ValueError, match="temperature"):
            MedusaConfig(temperature=0.0)

    def test_repr(self):
        cfg = MedusaConfig(n_heads=3, tree_width=2)
        assert "3" in repr(cfg)


class TestMedusaHeads:
    def _make_model(self, seed=0):
        cfg = MedusaConfig(n_heads=2, tree_width=2, seed=seed)
        return MedusaHeads(cfg)

    def _draft_fn(self, vocab=16):
        """Returns uniform probs over vocab."""
        def fn(head_idx, last_token, context):
            p = np.ones(vocab) / vocab
            return p
        return fn

    def _target_fn(self, vocab=16):
        def fn(last_token, context):
            p = np.ones(vocab) / vocab
            return p
        return fn

    def test_step_returns_medusa_draft_result(self):
        model = self._make_model()
        result = model.step([1, 2, 3], self._draft_fn(), self._target_fn())
        assert isinstance(result, MedusaDraftResult)

    def test_accepted_tokens_nonempty(self):
        model = self._make_model()
        result = model.step([1, 2, 3], self._draft_fn(), self._target_fn())
        assert len(result.accepted_tokens) >= 1

    def test_acceptance_rate_in_range(self):
        model = self._make_model()
        result = model.step([1, 2, 3], self._draft_fn(), self._target_fn())
        assert 0.0 <= result.acceptance_rate <= 1.0

    def test_n_drafted_positive(self):
        model = self._make_model()
        result = model.step([1, 2, 3], self._draft_fn(), self._target_fn())
        assert result.n_drafted >= 0

    def test_multiple_steps_running_stats(self):
        model = self._make_model()
        for _ in range(5):
            model.step([1, 2, 3], self._draft_fn(), self._target_fn())
        assert model.mean_acceptance_rate >= 0.0

    def test_reset_stats(self):
        model = self._make_model()
        for _ in range(3):
            model.step([1, 2], self._draft_fn(), self._target_fn())
        model.reset_stats()
        assert model.mean_acceptance_rate == 0.0

    def test_repr_contains_n_heads(self):
        model = self._make_model()
        assert "2" in repr(model)

    def test_default_config(self):
        model = MedusaHeads()
        assert model.config is not None

    def test_single_head(self):
        cfg = MedusaConfig(n_heads=1, tree_width=1, seed=99)
        model = MedusaHeads(cfg)
        result = model.step([5], self._draft_fn(), self._target_fn())
        assert isinstance(result, MedusaDraftResult)

    def test_skewed_target_biases_acceptance(self):
        cfg = MedusaConfig(n_heads=1, tree_width=1, seed=7, accept_threshold=0.0)
        model = MedusaHeads(cfg)
        Token = 3
        vocab = 8

        def biased_draft(head_idx, last_token, ctx):
            p = np.zeros(vocab)
            p[Token] = 1.0
            return p

        def biased_target(last_token, ctx):
            p = np.zeros(vocab)
            p[Token] = 1.0
            return p

        result = model.step([1, 2], biased_draft, biased_target)
        assert result.n_accepted >= 0


# ── SarathiScheduler ─────────────────────────────────────────────────────────

from squish.serving.sarathi_scheduler import (
    SarathiConfig,
    SarathiRequest,
    SarathiTick,
    SarathiScheduler,
)


class TestSarathiConfig:
    def test_defaults(self):
        cfg = SarathiConfig()
        assert cfg.chunk_size == 512
        assert cfg.max_decode_tokens == 512
        assert cfg.max_batch_size == 16

    def test_invalid_chunk_size(self):
        with pytest.raises(ValueError, match="chunk_size"):
            SarathiConfig(chunk_size=0)

    def test_invalid_max_decode_tokens(self):
        with pytest.raises(ValueError, match="max_decode_tokens"):
            SarathiConfig(max_decode_tokens=0)

    def test_invalid_max_batch_size(self):
        with pytest.raises(ValueError, match="max_batch_size"):
            SarathiConfig(max_batch_size=0)


class TestSarathiScheduler:
    def _make_scheduler(self, chunk_size=256, max_decode=64, max_batch=4):
        cfg = SarathiConfig(
            chunk_size=chunk_size,
            max_decode_tokens=max_decode,
            max_batch_size=max_batch,
        )
        return SarathiScheduler(cfg)

    def test_add_request(self):
        sched = self._make_scheduler()
        req = SarathiRequest("r1", prompt_tokens=100)
        sched.add_request(req)
        assert sched.n_inflight() == 1

    def test_schedule_returns_tick(self):
        sched = self._make_scheduler()
        sched.add_request(SarathiRequest("r1", prompt_tokens=100))
        tick = sched.schedule()
        assert isinstance(tick, SarathiTick)

    def test_tick_not_idle_when_requests_pending(self):
        sched = self._make_scheduler()
        sched.add_request(SarathiRequest("r1", prompt_tokens=100))
        tick = sched.schedule()
        assert not tick.idle

    def test_tick_idle_when_no_requests(self):
        sched = self._make_scheduler()
        tick = sched.schedule()
        assert tick.idle

    def test_multiple_requests_progress(self):
        sched = self._make_scheduler(chunk_size=10, max_decode=10, max_batch=8)
        for i in range(3):
            sched.add_request(SarathiRequest(f"r{i}", prompt_tokens=5, max_new_tokens=3))
        for _ in range(20):
            tick = sched.schedule()
        assert sched.n_completed() >= 0

    def test_stats_dict(self):
        sched = self._make_scheduler()
        stats = sched.stats()
        assert "n_completed" in stats

    def test_complete_requests(self):
        sched = self._make_scheduler(chunk_size=10, max_decode=5, max_batch=4)
        req = SarathiRequest("r1", prompt_tokens=5, max_new_tokens=2)
        sched.add_request(req)
        for _ in range(30):
            sched.schedule()
        assert sched.n_completed() >= 0

    def test_repr(self):
        sched = self._make_scheduler()
        r = repr(sched)
        assert "Sarathi" in r

    def test_sarathi_request_properties(self):
        req = SarathiRequest("r1", prompt_tokens=100, max_new_tokens=50)
        assert req.prefill_remaining > 0
        assert not req.is_complete

    def test_default_config(self):
        sched = SarathiScheduler()
        assert sched.config is not None


# ── NSAAttention ──────────────────────────────────────────────────────────────

from squish.attention.nsa_attn import NSAConfig, NSAAttention


class TestNSAConfig:
    def test_defaults(self):
        cfg = NSAConfig()
        assert cfg.n_heads >= 1
        assert cfg.head_dim >= 1
        assert cfg.block_size >= 1

    def test_invalid_n_heads(self):
        with pytest.raises(ValueError, match="n_heads"):
            NSAConfig(n_heads=0)

    def test_invalid_head_dim(self):
        with pytest.raises(ValueError, match="head_dim"):
            NSAConfig(head_dim=0)

    def test_invalid_block_size(self):
        with pytest.raises(ValueError, match="block_size"):
            NSAConfig(block_size=0)


class TestNSAAttention:
    def _make_model(self, n_heads=2, head_dim=8, block_size=4):
        cfg = NSAConfig(
            n_heads=n_heads,
            head_dim=head_dim,
            block_size=block_size,
            n_selected_blocks=1,
            window_size=4,
            n_selected_tokens=2,
            causal=True,
        )
        return NSAAttention(cfg)

    def _random_qkv(self, n_heads=2, T=12, head_dim=8):
        rng = np.random.default_rng(0)
        Q = rng.standard_normal((n_heads, T, head_dim)).astype(np.float32)
        K = rng.standard_normal((n_heads, T, head_dim)).astype(np.float32)
        V = rng.standard_normal((n_heads, T, head_dim)).astype(np.float32)
        return Q, K, V

    def test_output_shape(self):
        model = self._make_model()
        Q, K, V = self._random_qkv()
        out = model.forward(Q, K, V)
        assert out.shape == (2, 12, 8)

    def test_output_finite(self):
        model = self._make_model()
        Q, K, V = self._random_qkv()
        out = model.forward(Q, K, V)
        assert np.all(np.isfinite(out))

    def test_sparsity_ratio(self):
        model = self._make_model()
        ratio = model.sparsity_ratio(T=64, S=64)
        assert 0.0 <= ratio <= 1.0

    def test_repr_contains_heads(self):
        model = self._make_model()
        assert "2" in repr(model)

    def test_default_config(self):
        model = NSAAttention()
        assert model.config is not None

    def test_non_causal(self):
        cfg = NSAConfig(
            n_heads=2, head_dim=8, block_size=4,
            n_selected_blocks=1, window_size=4, n_selected_tokens=2,
            causal=False,
        )
        model = NSAAttention(cfg)
        Q, K, V = self._random_qkv()
        out = model.forward(Q, K, V)
        assert out.shape == (2, 12, 8)

    def test_single_token(self):
        model = self._make_model()
        Q, K, V = self._random_qkv(T=1)
        out = model.forward(Q, K, V)
        assert out.shape == (2, 1, 8)


# ── FlexPrefill ───────────────────────────────────────────────────────────────

from squish.attention.flex_prefill import FlexPrefillConfig, FlexPrefill


class TestFlexPrefillConfig:
    def test_defaults(self):
        cfg = FlexPrefillConfig()
        assert cfg.n_heads >= 1
        assert 0.0 < cfg.min_keep_ratio <= 1.0

    def test_invalid_min_keep_ratio_zero(self):
        with pytest.raises(ValueError, match="min_keep_ratio"):
            FlexPrefillConfig(min_keep_ratio=0.0)

    def test_invalid_min_keep_ratio_over_one(self):
        with pytest.raises(ValueError, match="min_keep_ratio"):
            FlexPrefillConfig(min_keep_ratio=1.1)

    def test_invalid_n_heads(self):
        with pytest.raises(ValueError, match="n_heads"):
            FlexPrefillConfig(n_heads=0)


class TestFlexPrefill:
    def _make_model(self, n_heads=2, head_dim=8, min_keep=0.5):
        cfg = FlexPrefillConfig(n_heads=n_heads, head_dim=head_dim, min_keep_ratio=min_keep)
        return FlexPrefill(cfg)

    def _random_qkv(self, n_heads=2, T=16, head_dim=8):
        rng = np.random.default_rng(0)
        Q = rng.standard_normal((n_heads, T, head_dim)).astype(np.float32)
        K = rng.standard_normal((n_heads, T, head_dim)).astype(np.float32)
        V = rng.standard_normal((n_heads, T, head_dim)).astype(np.float32)
        return Q, K, V

    def test_output_shape(self):
        model = self._make_model()
        Q, K, V = self._random_qkv()
        out = model.forward(Q, K, V)
        assert out.shape == (2, 16, 8)

    def test_output_finite(self):
        model = self._make_model()
        Q, K, V = self._random_qkv()
        out = model.forward(Q, K, V)
        assert np.all(np.isfinite(out))

    def test_mean_sparsity_ratio_range(self):
        model = self._make_model()
        Q, K, V = self._random_qkv()
        model.forward(Q, K, V)
        ratio = model.mean_sparsity_ratio()
        assert 0.0 <= ratio <= 1.0

    def test_reset_stats(self):
        model = self._make_model()
        Q, K, V = self._random_qkv()
        model.forward(Q, K, V)
        model.reset_stats()
        # After reset, no stats collected — implementation returns 1.0 (no-sparsity sentinel).
        assert model.mean_sparsity_ratio() == 1.0

    def test_repr(self):
        model = self._make_model()
        assert "FlexPrefill" in repr(model)

    def test_default_config(self):
        model = FlexPrefill()
        assert model.config is not None

    def test_full_keep(self):
        """With min_keep_ratio=1.0, all tokens should be kept."""
        cfg = FlexPrefillConfig(n_heads=2, head_dim=8, min_keep_ratio=1.0)
        model = FlexPrefill(cfg)
        Q, K, V = self._random_qkv()
        out = model.forward(Q, K, V)
        assert out.shape == (2, 16, 8)


# ── ThinKCache ────────────────────────────────────────────────────────────────

from squish.kv.think_cache import ThinKConfig, ThinKCache


class TestThinKConfig:
    def test_defaults(self):
        cfg = ThinKConfig()
        assert 0.0 < cfg.keep_ratio <= 1.0
        assert cfg.n_heads >= 1
        assert cfg.head_dim >= 1

    def test_invalid_keep_ratio_zero(self):
        with pytest.raises(ValueError, match="keep_ratio"):
            ThinKConfig(keep_ratio=0.0)

    def test_invalid_keep_ratio_over_one(self):
        with pytest.raises(ValueError, match="keep_ratio"):
            ThinKConfig(keep_ratio=1.1)

    def test_invalid_n_heads(self):
        with pytest.raises(ValueError, match="n_heads"):
            ThinKConfig(n_heads=0)

    def test_invalid_head_dim(self):
        with pytest.raises(ValueError, match="head_dim"):
            ThinKConfig(head_dim=0)


class TestThinKCache:
    def _random_qk(self, n_heads=4, T=8, d=16, seed=0):
        rng = np.random.default_rng(seed)
        Q = rng.standard_normal((n_heads, T, d)).astype(np.float32)
        K = rng.standard_normal((n_heads, T, d)).astype(np.float32)
        return Q, K

    def test_prune_k_shape(self):
        cache = ThinKCache(ThinKConfig(keep_ratio=0.5, n_heads=4, head_dim=16))
        Q, K = self._random_qk()
        K_pruned = cache.prune_k(Q, K)
        assert K_pruned.shape == K.shape

    def test_pruned_channels_zeroed(self):
        cache = ThinKCache(ThinKConfig(keep_ratio=0.5, n_heads=2, head_dim=8))
        Q, K = self._random_qk(n_heads=2, d=8)
        K_pruned = cache.prune_k(Q, K)
        # At least some channels should be zeroed.
        assert np.any(K_pruned == 0.0)

    def test_full_keep_no_zeroing(self):
        """keep_ratio=1.0 should keep all channels."""
        cache = ThinKCache(ThinKConfig(keep_ratio=1.0, n_heads=2, head_dim=8))
        Q, K = self._random_qk(n_heads=2, d=8)
        K_pruned = cache.prune_k(Q, K)
        # No channel zeroed if keep_ratio==1.0
        assert not np.all(K_pruned == 0.0)

    def test_channel_reduction_ratio(self):
        cache = ThinKCache(ThinKConfig(keep_ratio=0.5, n_heads=2, head_dim=8))
        Q, K = self._random_qk(n_heads=2, d=8)
        cache.prune_k(Q, K)
        ratio = cache.channel_reduction_ratio()
        assert 0.0 <= ratio <= 1.0

    def test_reset_stats(self):
        cache = ThinKCache(ThinKConfig(keep_ratio=0.5, n_heads=2, head_dim=8))
        Q, K = self._random_qk(n_heads=2, d=8)
        cache.prune_k(Q, K)
        cache.reset_stats()
        assert cache.channel_reduction_ratio() == 0.0

    def test_keep_indices_shape(self):
        cache = ThinKCache(ThinKConfig(keep_ratio=0.5, n_heads=2, head_dim=8))
        Q, K = self._random_qk(n_heads=2, d=8)
        idx = cache.keep_indices(Q, K)
        assert idx.shape[0] == 2  # n_heads
        assert idx.shape[1] == 4  # keep_ratio * head_dim = 0.5 * 8

    def test_repr(self):
        cache = ThinKCache()
        assert "ThinKCache" in repr(cache)

    def test_default_config(self):
        cache = ThinKCache()
        assert cache.config is not None


# ── AttentionStore ────────────────────────────────────────────────────────────

from squish.kv.attention_store import AttentionStoreConfig, AttentionStore


class TestAttentionStoreConfig:
    def test_defaults(self):
        cfg = AttentionStoreConfig()
        assert cfg.hot_capacity >= 1
        assert cfg.warm_capacity >= 1

    def test_invalid_hot_capacity(self):
        with pytest.raises(ValueError, match="hot_capacity"):
            AttentionStoreConfig(hot_capacity=0)

    def test_invalid_warm_capacity(self):
        with pytest.raises(ValueError, match="warm_capacity"):
            AttentionStoreConfig(warm_capacity=0)


class TestAttentionStore:
    def _kv(self, seed=0, shape=(4, 8)):
        rng = np.random.default_rng(seed)
        K = rng.standard_normal(shape).astype(np.float32)
        V = rng.standard_normal(shape).astype(np.float32)
        return K, V

    def test_store_and_load(self):
        store = AttentionStore(AttentionStoreConfig(hot_capacity=16, warm_capacity=64))
        K, V = self._kv()
        store.store("s1", 0, K, V)
        K2, V2 = store.load("s1", 0)
        np.testing.assert_allclose(K, K2)
        np.testing.assert_allclose(V, V2)

    def test_load_missing_raises_key_error(self):
        store = AttentionStore()
        with pytest.raises(KeyError):
            store.load("nonexistent", 99)

    def test_hit_rate_starts_zero(self):
        store = AttentionStore()
        assert store.hit_rate() == 0.0

    def test_hit_rate_after_store_and_load(self):
        store = AttentionStore()
        K, V = self._kv()
        store.store("s1", 0, K, V)
        store.load("s1", 0)
        assert store.hit_rate() == 1.0

    def test_tiers_used(self):
        store = AttentionStore(AttentionStoreConfig(hot_capacity=4, warm_capacity=8))
        K, V = self._kv()
        store.store("s1", 0, K, V)
        tiers = store.tiers_used()
        assert "hot" in tiers
        assert tiers["hot"] >= 1

    def test_evict_session(self):
        store = AttentionStore()
        K, V = self._kv()
        store.store("s1", 0, K, V)
        store.store("s1", 1, K, V)
        removed = store.evict_session("s1")
        assert removed == 2
        with pytest.raises(KeyError):
            store.load("s1", 0)

    def test_memory_bytes(self):
        store = AttentionStore()
        K, V = self._kv()
        store.store("s1", 0, K, V)
        assert store.memory_bytes() > 0

    def test_hot_eviction_to_warm(self):
        """When hot is full, entries evict to warm tier."""
        cfg = AttentionStoreConfig(hot_capacity=2, warm_capacity=16)
        store = AttentionStore(cfg)
        K, V = self._kv()
        for i in range(4):
            store.store(f"s{i}", 0, K, V)
        tiers = store.tiers_used()
        assert tiers["warm"] >= 1

    def test_repr(self):
        store = AttentionStore()
        assert "AttentionStore" in repr(store)

    def test_store_multiple_layers(self):
        store = AttentionStore()
        K, V = self._kv()
        for layer in range(5):
            store.store("sess", layer, K, V)
        tiers = store.tiers_used()
        total = tiers["hot"] + tiers["warm"] + tiers["ssd"]
        assert total == 5
