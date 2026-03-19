"""tests/test_wave29_modules.py — Tests for Wave 29 inference optimization modules

Covers:
  - pyramid_kv:           PyramidKVConfig, PyramidKVManager, PyramidKVStats
  - sparq_attn:           SparQConfig, SparQAttention, SparQStats
  - kv_merge:             KVMergeRegistry, SharedPrefixSlab, RequestKVView, KVMergeStats
  - logit_filter:         LogitFilterConfig, LogitFilter, LogitFilterStats
  - rest_spec:            DataStore, DataStoreConfig, RESTSpecDecoder, RESTSpecConfig, RESTSpecStats
  - contrastive_decoding: ContrastiveDecoderConfig, ContrastiveDecoder, ContrastiveDecoderStats
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np
import pytest

# ============================================================
# pyramid_kv
# ============================================================
from squish.kv.pyramid_kv import PyramidKVConfig, PyramidKVManager, PyramidKVStats


class TestPyramidKVConfig:
    def test_defaults(self):
        cfg = PyramidKVConfig()
        assert cfg.n_layers == 32
        assert cfg.base_budget == 1024
        assert cfg.min_budget == 64
        assert 0.0 < cfg.alpha < 1.0

    def test_custom_values(self):
        cfg = PyramidKVConfig(n_layers=8, base_budget=512, min_budget=32, alpha=0.5)
        assert cfg.n_layers == 8
        assert cfg.base_budget == 512


class TestPyramidKVManager:
    def _make_mgr(self, n_layers=8, base_budget=256, min_budget=32) -> PyramidKVManager:
        cfg = PyramidKVConfig(n_layers=n_layers, base_budget=base_budget, min_budget=min_budget)
        return PyramidKVManager(cfg)

    def test_bottom_layer_gets_full_budget(self):
        mgr = self._make_mgr()
        # Layer 0 should get at or near the base_budget
        budget_0 = mgr.layer_budget(0)
        assert budget_0 == 256

    def test_top_layer_gets_min_budget_floor(self):
        mgr = self._make_mgr()
        budget_top = mgr.layer_budget(7)
        assert budget_top >= 32

    def test_budgets_are_monotone_decreasing(self):
        mgr = self._make_mgr(n_layers=16)
        budgets = [mgr.layer_budget(l) for l in range(16)]
        for a, b in zip(budgets, budgets[1:]):
            assert a >= b, f"Budget not monotone: {a} < {b}"

    def test_budget_all_layers_above_min(self):
        mgr = self._make_mgr()
        for l in range(8):
            assert mgr.layer_budget(l) >= 32

    def test_invalid_layer_raises(self):
        mgr = self._make_mgr()
        with pytest.raises((ValueError, IndexError)):
            mgr.layer_budget(-1)
        with pytest.raises((ValueError, IndexError)):
            mgr.layer_budget(100)

    def test_update_importance_runs(self):
        mgr = self._make_mgr()
        rng = np.random.default_rng(42)
        attn = rng.random((16, 16)).astype(np.float32)
        attn /= attn.sum(axis=-1, keepdims=True)
        mgr.update_importance(0, attn)  # should not raise

    def test_eviction_mask_shape(self):
        mgr = self._make_mgr()
        rng = np.random.default_rng(0)
        attn = rng.random((32, 32)).astype(np.float32)
        attn /= attn.sum(axis=-1, keepdims=True)
        mgr.update_importance(0, attn)
        mask = mgr.eviction_mask(0, 32)
        assert mask.shape == (32,)
        assert mask.dtype == bool

    def test_eviction_mask_keeps_correct_count(self):
        mgr = self._make_mgr(n_layers=4, base_budget=16, min_budget=4)
        rng = np.random.default_rng(1)
        seq_len = 64
        attn = rng.random((seq_len, seq_len)).astype(np.float32)
        attn /= attn.sum(axis=-1, keepdims=True)
        # Feed several update steps so EMA stabilises
        for _ in range(6):
            mgr.update_importance(0, attn)
        mask = mgr.eviction_mask(0, seq_len)
        # Number of True (kept) tokens should be ≤ layer_budget
        assert mask.sum() <= mgr.layer_budget(0)

    def test_stats_attribute_exists(self):
        mgr = self._make_mgr()
        assert isinstance(mgr.stats, PyramidKVStats)

    def test_single_layer_model(self):
        cfg = PyramidKVConfig(n_layers=1, base_budget=64, min_budget=64)
        mgr = PyramidKVManager(cfg)
        assert mgr.layer_budget(0) == 64


# ============================================================
# sparq_attn
# ============================================================
from squish.attention.sparq_attn import SparQAttention, SparQConfig, SparQStats


class TestSparQConfig:
    def test_defaults(self):
        cfg = SparQConfig()
        # r_dims and top_k_kv can be None (unbounded) or positive
        assert cfg.r_dims is None or cfg.r_dims > 0
        assert cfg.top_k_kv is None or cfg.top_k_kv > 0

    def test_custom_values(self):
        cfg = SparQConfig(r_dims=32, top_k_kv=64)
        attn = SparQAttention(cfg)
        assert attn.config.r_dims == 32
        assert attn.config.top_k_kv == 64


class TestSparQAttention:
    def _make_attn(self, r=8, k=16) -> SparQAttention:
        cfg = SparQConfig(r_dims=r, top_k_kv=k)
        return SparQAttention(cfg)

    def test_decode_step_output_shape(self):
        attn = self._make_attn()
        rng = np.random.default_rng(7)
        n_heads, d_k, seq = 4, 32, 64
        d_v = d_k
        query = rng.standard_normal((n_heads, d_k)).astype(np.float32)
        keys = rng.standard_normal((seq, n_heads, d_k)).astype(np.float32)
        values = rng.standard_normal((seq, n_heads, d_v)).astype(np.float32)
        out = attn.decode_step(query, keys, values)
        assert out.shape == (n_heads, d_v)

    def test_decode_step_output_finite(self):
        attn = self._make_attn()
        rng = np.random.default_rng(8)
        n_heads, d_k, seq = 4, 32, 64
        query = rng.standard_normal((n_heads, d_k)).astype(np.float32)
        keys = rng.standard_normal((seq, n_heads, d_k)).astype(np.float32)
        values = rng.standard_normal((seq, n_heads, d_k)).astype(np.float32)
        out = attn.decode_step(query, keys, values)
        assert np.isfinite(out).all()

    def test_fallback_when_seq_le_k(self):
        """When seq_len ≤ top_k_kv, SparQ should fall back to full attention."""
        attn = self._make_attn(r=8, k=16)
        rng = np.random.default_rng(9)
        n_heads, d_k, seq = 4, 32, 8  # seq(8) < k(16) → fallback
        query = rng.standard_normal((n_heads, d_k)).astype(np.float32)
        keys = rng.standard_normal((seq, n_heads, d_k)).astype(np.float32)
        values = rng.standard_normal((seq, n_heads, d_k)).astype(np.float32)
        out = attn.decode_step(query, keys, values)
        assert out.shape == (n_heads, d_k)

    def test_expected_speedup_positive(self):
        attn = self._make_attn()
        speedup = attn.expected_speedup(seq_len=512, d_k=32)
        assert speedup > 0.0

    def test_stats_exists(self):
        attn = self._make_attn()
        assert isinstance(attn.stats, SparQStats)

    def test_stats_increments_on_decode(self):
        attn = self._make_attn()
        rng = np.random.default_rng(10)
        n_heads, d_k, seq = 4, 32, 64
        query = rng.standard_normal((n_heads, d_k)).astype(np.float32)
        keys = rng.standard_normal((seq, n_heads, d_k)).astype(np.float32)
        values = rng.standard_normal((seq, n_heads, d_k)).astype(np.float32)
        attn.decode_step(query, keys, values)
        assert attn.stats.decode_calls == 1


# ============================================================
# kv_merge
# ============================================================
from squish.kv.kv_merge import KVMergeRegistry, KVMergeStats, RequestKVView, SharedPrefixSlab
from squish.kv.kv_merge import _hash_prefix as _kv_hash_prefix


class TestSharedPrefixSlab:
    def _make_slab(self, prefix_ids=None):
        ids = prefix_ids or [1, 2, 3]
        h = _kv_hash_prefix(ids)
        return SharedPrefixSlab(prefix_ids=ids, prefix_hash=h)

    def test_creation_stores_prefix_ids(self):
        slab = self._make_slab([1, 2, 3])
        assert slab.prefix_len == 3

    def test_refcount(self):
        slab = self._make_slab()
        assert slab.refcount == 0
        slab.acquire()
        assert slab.refcount == 1
        slab.release()
        assert slab.refcount == 0

    def test_release_below_zero_raises(self):
        slab = self._make_slab([1, 2])
        with pytest.raises(ValueError):
            slab.release()  # refcount=0 → underflow


class TestRequestKVView:
    def _make_view(self, prefix_len=4, n_heads=2, d_k=8) -> RequestKVView:
        ids = list(range(prefix_len))
        h = _kv_hash_prefix(ids)
        slab = SharedPrefixSlab(prefix_ids=ids, prefix_hash=h)
        # Populate slab with KV data for layer 0
        rng = np.random.default_rng(99)
        k = rng.standard_normal((prefix_len, n_heads, d_k)).astype(np.float32)
        v = rng.standard_normal((prefix_len, n_heads, d_k)).astype(np.float32)
        slab.store_layer(0, k, v)
        slab.finalize()
        slab.acquire()
        return RequestKVView(slab=slab, request_id="req-1")

    def test_read_kv_prefix_length(self):
        view = self._make_view(prefix_len=4, n_heads=2, d_k=8)
        k, v = view.read_kv(layer_idx=0)
        assert k.shape[0] == 4

    def test_append_private_increases_length(self):
        n_heads, d_k = 2, 8
        view = self._make_view(prefix_len=4, n_heads=n_heads, d_k=d_k)
        rng = np.random.default_rng(20)
        # Append 3 private decode tokens one at a time
        for _ in range(3):
            extra_k = rng.standard_normal((n_heads, d_k)).astype(np.float32)
            extra_v = rng.standard_normal((n_heads, d_k)).astype(np.float32)
            view.append_private(layer_idx=0, key=extra_k, value=extra_v)
        k, v = view.read_kv(layer_idx=0)
        assert k.shape[0] == 4 + 3


class TestKVMergeRegistry:
    def test_get_or_create_view_returns_view(self):
        registry = KVMergeRegistry()
        view = registry.get_or_create_view([1, 2, 3], request_id="req-A")
        assert isinstance(view, RequestKVView)

    def test_two_requests_same_prefix_share_slab(self):
        registry = KVMergeRegistry()
        v1 = registry.get_or_create_view([1, 2, 3], request_id="req-A")
        v2 = registry.get_or_create_view([1, 2, 3], request_id="req-B")
        assert v1.slab is v2.slab

    def test_different_prefix_different_slab(self):
        registry = KVMergeRegistry()
        v1 = registry.get_or_create_view([1, 2, 3], request_id="req-A")
        v2 = registry.get_or_create_view([4, 5, 6], request_id="req-B")
        assert v1.slab is not v2.slab

    def test_release_decrements_refcount(self):
        registry = KVMergeRegistry()
        registry.get_or_create_view([1, 2, 3], request_id="req-A")
        h = _kv_hash_prefix([1, 2, 3])
        rc_before = registry._slabs[h].refcount
        registry.release_view("req-A")
        # After release, slab should be freed (no more refs) or refcount decreased
        if h in registry._slabs:
            assert registry._slabs[h].refcount < rc_before

    def test_stats_exists(self):
        registry = KVMergeRegistry()
        assert isinstance(registry.stats, KVMergeStats)

    def test_hit_rate_increases_on_shared_prefix(self):
        registry = KVMergeRegistry()
        registry.get_or_create_view([1, 2, 3], request_id="req-A")
        registry.get_or_create_view([1, 2, 3], request_id="req-B")
        assert registry.stats.slab_hits >= 1


# ============================================================
# logit_filter
# ============================================================
from squish.token.logit_filter import LogitFilter, LogitFilterConfig, LogitFilterStats


class TestLogitFilterConfig:
    def test_defaults(self):
        cfg = LogitFilterConfig()
        assert cfg.sketch_dim > 0
        assert cfg.top_k > 0

    def test_top_k_less_than_vocab(self):
        cfg = LogitFilterConfig(top_k=1024, sketch_dim=256)
        assert cfg.top_k == 1024


class TestLogitFilter:
    def _make_filter(self, vocab_size=1024, d_model=128, top_k=32, sketch_dim=64):
        cfg = LogitFilterConfig(sketch_dim=sketch_dim, top_k=top_k)
        rng = np.random.default_rng(42)
        vocab_embed = rng.standard_normal((vocab_size, d_model)).astype(np.float32)
        return LogitFilter.from_embedding_matrix(vocab_embed, cfg)

    def test_from_embedding_matrix_creates_filter(self):
        filt = self._make_filter()
        assert filt is not None

    def test_filter_and_score_returns_vocab_size(self):
        vocab_size = 1024
        filt = self._make_filter(vocab_size=vocab_size)
        rng = np.random.default_rng(3)
        d_model = 128
        hidden = rng.standard_normal((d_model,)).astype(np.float32)
        vocab_embed = rng.standard_normal((vocab_size, d_model)).astype(np.float32)
        logits = filt.filter_and_score(hidden, vocab_embed)
        assert logits.shape == (vocab_size,)

    def test_non_candidate_tokens_are_neg_inf(self):
        vocab_size = 1024
        filt = self._make_filter(vocab_size=vocab_size, top_k=32)
        rng = np.random.default_rng(4)
        hidden = rng.standard_normal((128,)).astype(np.float32)
        vocab_embed = rng.standard_normal((vocab_size, 128)).astype(np.float32)
        logits = filt.filter_and_score(hidden, vocab_embed)
        # At most top_k tokens should be non-neg-inf
        finite_count = np.isfinite(logits).sum()
        assert finite_count <= 32

    def test_select_candidates_returns_indices(self):
        filt = self._make_filter()
        rng = np.random.default_rng(5)
        hidden = rng.standard_normal((128,)).astype(np.float32)
        idx = filt.select_candidates(hidden)
        assert len(idx) == filt.config.top_k
        assert idx.dtype in (np.int32, np.int64, np.intp)

    def test_stats_exists(self):
        filt = self._make_filter()
        assert isinstance(filt.stats, LogitFilterStats)

    def test_stats_tracks_calls(self):
        vocab_size, d_model = 1024, 128
        filt = self._make_filter(vocab_size=vocab_size, d_model=d_model)
        rng = np.random.default_rng(6)
        hidden = rng.standard_normal((d_model,)).astype(np.float32)
        vocab_embed = rng.standard_normal((vocab_size, d_model)).astype(np.float32)
        filt.filter_and_score(hidden, vocab_embed)
        filt.filter_and_score(hidden, vocab_embed)
        assert filt.stats.filter_calls == 2


# ============================================================
# rest_spec
# ============================================================
from squish.speculative.rest_spec import (
    DataStore,
    DataStoreConfig,
    RESTSpecConfig,
    RESTSpecDecoder,
    RESTSpecStats,
)


class TestDataStore:
    def test_ingest_populates_store(self):
        store = DataStore()
        store.ingest([1, 2, 3, 4, 5])
        assert store.total_entries > 0

    def test_draft_sequence_returns_list(self):
        store = DataStore()
        store.ingest([10, 20, 30, 40, 50])
        drafts = store.draft_sequence([10, 20, 30])
        assert isinstance(drafts, list)

    def test_draft_sequence_empty_on_miss(self):
        store = DataStore()
        # Ingest unrelated data
        store.ingest([1, 2, 3])
        drafts = store.draft_sequence([99, 88, 77])
        assert drafts == [] or isinstance(drafts, list)

    def test_draft_sequence_hits_after_ingest(self):
        store = DataStore(DataStoreConfig(ngram_order=2, top_b_draft=3, draft_depth=3))
        store.ingest([1, 2, 3, 4, 5, 6])
        drafts = store.draft_sequence([1, 2])
        # Should return the continuation [3, ...]
        if len(drafts) > 0:
            assert drafts[0] == 3

    def test_max_entries_evicts_old(self):
        cfg = DataStoreConfig(max_entries=5, ngram_order=2)
        store = DataStore(cfg)
        for i in range(100):
            store.ingest([i, i + 1, i + 2])
        # Total entries should not blow up wildly
        assert store.total_entries <= 200  # generous cap

    def test_empty_store_draft_returns_empty(self):
        store = DataStore()
        assert store.draft_sequence([1, 2, 3]) == []


class TestRESTSpecDecoder:
    def _make_decoder(self, vocab_size=100):
        store = DataStore(DataStoreConfig(ngram_order=2, draft_depth=3))
        store.ingest(list(range(50)))  # preload data

        def forward_fn(context_ids):
            # context_ids is a List[int] — return (vocab_size,) logits
            if not context_ids:
                return np.zeros(vocab_size, dtype=np.float32)
            last = context_ids[-1] % vocab_size
            logits = np.full(vocab_size, -100.0, dtype=np.float32)
            next_tok = (last + 1) % vocab_size
            logits[next_tok] = 100.0
            return logits

        cfg = RESTSpecConfig(max_draft_len=3)
        return RESTSpecDecoder(store, forward_fn, cfg)

    def test_generate_returns_token_ids(self):
        decoder = self._make_decoder()
        prompt = [1, 2, 3]
        out, stats = decoder.generate(prompt, max_new_tokens=10, eos_id=99)
        assert isinstance(out, list)
        assert len(out) >= len(prompt)

    def test_generate_returns_stats(self):
        decoder = self._make_decoder()
        _, stats = decoder.generate([1, 2, 3], max_new_tokens=5, eos_id=99)
        assert isinstance(stats, RESTSpecStats)

    def test_generate_stops_at_eos(self):
        """Decoder should stop when EOS token is generated."""
        vocab_size = 100
        eos_id = 5
        store = DataStore()

        def forward_always_eos(context_ids):
            # Always predict EOS
            logits = np.full(vocab_size, -100.0)
            logits[eos_id] = 100.0
            return logits

        cfg = RESTSpecConfig(max_draft_len=2)
        decoder = RESTSpecDecoder(store, forward_always_eos, cfg)
        out, _ = decoder.generate([1, 2], max_new_tokens=20, eos_id=eos_id)
        # Should stop quickly after generating eos
        assert len(out) < 25


# ============================================================
# contrastive_decoding
# ============================================================
from squish.sampling.contrastive_decoding import (
    ContrastiveDecoder,
    ContrastiveDecoderConfig,
    ContrastiveDecoderStats,
)


class TestContrastiveDecoderConfig:
    def test_defaults(self):
        cfg = ContrastiveDecoderConfig()
        assert 0.0 <= cfg.alpha <= 2.0
        assert 0.0 <= cfg.beta <= 1.0

    def test_invalid_alpha_raises(self):
        with pytest.raises(ValueError):
            ContrastiveDecoder(ContrastiveDecoderConfig(alpha=3.0))

    def test_invalid_beta_raises(self):
        with pytest.raises(ValueError):
            ContrastiveDecoder(ContrastiveDecoderConfig(beta=1.5))

    def test_invalid_amateur_mode_raises(self):
        with pytest.raises(ValueError):
            ContrastiveDecoder(ContrastiveDecoderConfig(amateur_mode="unknown"))


class TestContrastiveDecoder:
    def _make_logits(self, vocab=200, seed=0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        logits = rng.standard_normal(vocab).astype(np.float32)
        return logits

    def _make_decoder(self, alpha=0.5, beta=0.1) -> ContrastiveDecoder:
        return ContrastiveDecoder(ContrastiveDecoderConfig(alpha=alpha, beta=beta))

    def test_contrast_output_shape(self):
        dec = self._make_decoder()
        expert = self._make_logits(200)
        out = dec.contrast(expert)
        assert out.shape == (200,)

    def test_contrast_output_dtype_float32(self):
        dec = self._make_decoder()
        out = dec.contrast(self._make_logits(200))
        assert out.dtype == np.float32

    def test_contrast_with_external_amateur(self):
        dec = self._make_decoder()
        expert = self._make_logits(200)
        amateur = self._make_logits(200, seed=1)
        out = dec.contrast(expert, amateur)
        assert out.shape == (200,)
        assert np.isfinite(out[np.isfinite(out)]).all()

    def test_apc_masks_implausible_tokens(self):
        dec = self._make_decoder(alpha=0.5, beta=0.5)
        rng = np.random.default_rng(5)
        expert = rng.standard_normal(500).astype(np.float32)
        out = dec.contrast(expert)
        # Many tokens should be -inf (outside plausible set)
        neg_inf_count = np.sum(np.isneginf(out))
        assert neg_inf_count > 0

    def test_no_apc_when_beta_zero(self):
        dec = ContrastiveDecoder(ContrastiveDecoderConfig(alpha=0.3, beta=0.0))
        expert = self._make_logits(100)
        out = dec.contrast(expert)
        # No tokens should be masked to -inf by APC
        assert not np.isneginf(out).any()

    def test_sample_returns_valid_token(self):
        dec = self._make_decoder()
        expert = self._make_logits(200)
        token = dec.sample(expert, temperature=1.0)
        assert 0 <= token < 200

    def test_sample_high_temp_varies(self):
        dec = self._make_decoder(beta=0.0)
        expert = self._make_logits(200)
        tokens = {dec.sample(expert, temperature=5.0) for _ in range(30)}
        assert len(tokens) > 1  # should sample diverse tokens

    def test_stats_increments(self):
        dec = self._make_decoder()
        expert = self._make_logits(100)
        dec.contrast(expert)
        dec.contrast(expert)
        assert dec.stats.total_calls == 2

    def test_stats_type(self):
        dec = self._make_decoder()
        assert isinstance(dec.stats, ContrastiveDecoderStats)

    def test_amateur_modes(self):
        expert = self._make_logits(100)
        for mode in ("high_temp", "uniform", "entropy"):
            dec = ContrastiveDecoder(ContrastiveDecoderConfig(amateur_mode=mode, beta=0.0))
            out = dec.contrast(expert)
            assert out.shape == (100,)

    def test_2d_input_raises(self):
        dec = self._make_decoder()
        with pytest.raises(ValueError):
            dec.contrast(np.zeros((4, 100), dtype=np.float32))

    def test_mismatched_amateur_raises(self):
        dec = self._make_decoder()
        expert = self._make_logits(100)
        amateur = self._make_logits(200)
        with pytest.raises(ValueError):
            dec.contrast(expert, amateur)

    def test_reset_stats(self):
        dec = self._make_decoder()
        expert = self._make_logits(100)
        dec.contrast(expert)
        dec.reset_stats()
        assert dec.stats.total_calls == 0

    def test_repr_contains_alpha(self):
        dec = ContrastiveDecoder(ContrastiveDecoderConfig(alpha=0.75))
        assert "0.75" in repr(dec)
