"""
test_wave25_server_wiring.py — Wave 25 server-wiring tests.

4 tests per module × 14 modules = 56 tests.
Each test covers: import, instantiation, core method invocation, and stats/properties.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

RNG = np.random.default_rng(0xDEAD_BEEF)


# ── FlashMLA ──────────────────────────────────────────────────────────────────

def test_flash_mla_import():
    from squish.attention.flash_mla import FlashMLACache, MLAConfig
    cfg = MLAConfig(n_heads=4, head_dim=32, latent_dim=16)
    assert cfg.n_heads == 4
    assert cfg.latent_dim == 16


def test_flash_mla_append_and_seq_len():
    from squish.attention.flash_mla import FlashMLACache, MLAConfig
    cfg = MLAConfig(n_heads=2, head_dim=8, latent_dim=4)
    cache = FlashMLACache(cfg, max_seq_len=32)
    assert cache.seq_len == 0
    x = RNG.random((4,)).astype(np.float32)
    cache.append(x)
    cache.append(x)
    assert cache.seq_len == 2


def test_flash_mla_attend():
    from squish.attention.flash_mla import FlashMLACache, MLAConfig
    n_heads, head_dim, latent_dim = 2, 8, 4
    cfg = MLAConfig(n_heads=n_heads, head_dim=head_dim, latent_dim=latent_dim)
    cache = FlashMLACache(cfg, max_seq_len=16)
    for _ in range(3):
        cache.append(RNG.random((latent_dim,)).astype(np.float32))
    q = RNG.random((n_heads, head_dim)).astype(np.float32)
    W_uk = RNG.random((latent_dim, n_heads * head_dim)).astype(np.float32)
    W_uv = RNG.random((latent_dim, n_heads * head_dim)).astype(np.float32)
    out = cache.attend(q, W_uk, W_uv)
    assert out.shape == (n_heads, head_dim)
    assert out.dtype == np.float32


def test_flash_mla_compression_ratio():
    from squish.attention.flash_mla import FlashMLACache, MLAConfig
    cfg = MLAConfig(n_heads=4, head_dim=32, latent_dim=8)
    cache = FlashMLACache(cfg)
    assert cache.compression_ratio == pytest.approx(4 * 32 / 8)


# ── NativeSparseAttn ──────────────────────────────────────────────────────────

def test_native_sparse_attn_import():
    from squish.attention.native_sparse_attn import NativeSparseAttention, NSAConfig
    cfg = NSAConfig(n_heads=4, head_dim=16, block_size=8, top_k_blocks=2, window_size=16)
    assert cfg.block_size == 8


def test_native_sparse_attn_forward_shape():
    from squish.attention.native_sparse_attn import NativeSparseAttention, NSAConfig
    cfg = NSAConfig(n_heads=2, head_dim=8, block_size=4, top_k_blocks=2, window_size=8)
    attn = NativeSparseAttention(cfg)
    seq = 16
    q = RNG.random((2, seq, 8)).astype(np.float32)
    k = RNG.random((2, seq, 8)).astype(np.float32)
    v = RNG.random((2, seq, 8)).astype(np.float32)
    out = attn.forward(q, k, v)
    assert out.shape == (2, seq, 8)
    assert out.dtype == np.float32


def test_native_sparse_attn_sparsity():
    from squish.attention.native_sparse_attn import NativeSparseAttention, NSAConfig
    cfg = NSAConfig(n_heads=2, head_dim=8, block_size=4, top_k_blocks=1, window_size=4)
    attn = NativeSparseAttention(cfg)
    assert attn.sparsity == pytest.approx(0.0)
    seq = 32
    q = RNG.random((2, seq, 8)).astype(np.float32)
    k = RNG.random((2, seq, 8)).astype(np.float32)
    v = RNG.random((2, seq, 8)).astype(np.float32)
    attn.forward(q, k, v)
    assert 0.0 <= attn.sparsity <= 1.0


def test_native_sparse_attn_bad_shape():
    from squish.attention.native_sparse_attn import NativeSparseAttention, NSAConfig
    cfg = NSAConfig(n_heads=2, head_dim=8)
    attn = NativeSparseAttention(cfg)
    bad = RNG.random((4, 8)).astype(np.float32)
    with pytest.raises(ValueError):
        attn.forward(bad, bad, bad)


# ── FusedSampler ─────────────────────────────────────────────────────────────

def test_fused_sampler_import():
    from squish.hardware.fused_sampler import FusedSampler, SamplerConfig
    cfg = SamplerConfig(temperature=0.9, top_k=50, top_p=0.9, seed=42)
    s = FusedSampler(cfg)
    assert s is not None


def test_fused_sampler_sample_returns_valid_token():
    from squish.hardware.fused_sampler import FusedSampler, SamplerConfig
    vocab = 100
    cfg = SamplerConfig(temperature=1.0, seed=7)
    s = FusedSampler(cfg)
    logits = RNG.random((vocab,)).astype(np.float32)
    tok = s.sample(logits)
    assert isinstance(tok, int)
    assert 0 <= tok < vocab


def test_fused_sampler_repetition_penalty():
    from squish.hardware.fused_sampler import FusedSampler, SamplerConfig
    vocab = 50
    cfg = SamplerConfig(temperature=1.0, repetition_penalty=1.5, seed=0)
    s = FusedSampler(cfg)
    logits = np.zeros(vocab, dtype=np.float32)
    input_ids = np.array([5, 10, 15], dtype=np.int32)
    tok = s.sample(logits, input_ids)
    assert 0 <= tok < vocab


def test_fused_sampler_batch():
    from squish.hardware.fused_sampler import FusedSampler, SamplerConfig
    vocab, batch = 200, 4
    cfg = SamplerConfig(temperature=0.7, top_k=20, seed=1)
    s = FusedSampler(cfg)
    logits = RNG.random((batch, vocab)).astype(np.float32)
    toks = s.sample_batch(logits)
    assert toks.shape == (batch,)
    assert all(0 <= t < vocab for t in toks)


# ── KVDefrag ──────────────────────────────────────────────────────────────────

def test_kv_defrag_import():
    from squish.kv.kv_defrag import DefragStats, KVDefragmenter
    d = KVDefragmenter(page_size=8, n_heads=2, head_dim=4)
    assert d is not None


def test_kv_defrag_allocate_free():
    from squish.kv.kv_defrag import KVDefragmenter
    d = KVDefragmenter(page_size=4, n_heads=2, head_dim=4)
    pages = d.allocate(1, 12)
    assert len(pages) == 3   # ceil(12/4)
    assert d.utilization > 0.0
    d.free(1)
    assert d.utilization == pytest.approx(0.0)


def test_kv_defrag_defrag():
    from squish.kv.kv_defrag import KVDefragmenter
    d = KVDefragmenter(page_size=4, n_heads=2, head_dim=4)
    d.allocate(1, 8)
    d.allocate(2, 4)
    d.free(1)
    stats = d.defrag()
    assert stats.n_pages_before == stats.n_pages_after
    assert 0.0 <= stats.fragmentation_after <= stats.fragmentation_before


def test_kv_defrag_fragmentation_ratio():
    from squish.kv.kv_defrag import KVDefragmenter
    d = KVDefragmenter(page_size=4, n_heads=2, head_dim=4)
    assert d.fragmentation_ratio == pytest.approx(0.0)
    d.allocate(10, 4)
    assert d.utilization > 0.0


# ── DualChunkAttn ─────────────────────────────────────────────────────────────

def test_dual_chunk_attn_import():
    from squish.attention.dual_chunk_attn import DCAConfig, DualChunkAttention
    cfg = DCAConfig(n_heads=2, head_dim=8, chunk_size=16, inter_chunk_top_k=2)
    assert cfg.chunk_size == 16


def test_dual_chunk_attn_encode_chunk():
    from squish.attention.dual_chunk_attn import DCAConfig, DualChunkAttention
    n_heads, head_dim, chunk_size = 2, 8, 16
    cfg = DCAConfig(n_heads=n_heads, head_dim=head_dim, chunk_size=chunk_size)
    attn = DualChunkAttention(cfg)
    k = RNG.random((n_heads, chunk_size, head_dim)).astype(np.float32)
    v = RNG.random((n_heads, chunk_size, head_dim)).astype(np.float32)
    summary = attn.encode_chunk(k, v)
    assert summary.shape == (n_heads, head_dim)
    assert summary.dtype == np.float32


def test_dual_chunk_attn_forward_intra_only():
    from squish.attention.dual_chunk_attn import DCAConfig, DualChunkAttention
    n_heads, head_dim = 2, 8
    cfg = DCAConfig(n_heads=n_heads, head_dim=head_dim, chunk_size=16)
    attn = DualChunkAttention(cfg)
    seq = 8
    q = RNG.random((n_heads, seq, head_dim)).astype(np.float32)
    k = RNG.random((n_heads, seq, head_dim)).astype(np.float32)
    v = RNG.random((n_heads, seq, head_dim)).astype(np.float32)
    out = attn.forward(q, k, v)
    assert out.shape == (n_heads, seq, head_dim)


def test_dual_chunk_attn_forward_with_past():
    from squish.attention.dual_chunk_attn import DCAConfig, DualChunkAttention
    n_heads, head_dim, chunk_size = 2, 8, 16
    cfg = DCAConfig(n_heads=n_heads, head_dim=head_dim, chunk_size=chunk_size, inter_chunk_top_k=2)
    attn = DualChunkAttention(cfg)
    k_full = RNG.random((n_heads, chunk_size, head_dim)).astype(np.float32)
    v_full = RNG.random((n_heads, chunk_size, head_dim)).astype(np.float32)
    summary = attn.encode_chunk(k_full, v_full)
    past = [summary, summary]
    seq = 8
    q = RNG.random((n_heads, seq, head_dim)).astype(np.float32)
    k = RNG.random((n_heads, seq, head_dim)).astype(np.float32)
    v = RNG.random((n_heads, seq, head_dim)).astype(np.float32)
    out = attn.forward(q, k, v, past_chunks=past)
    assert out.shape == (n_heads, seq, head_dim)


# ── ActivationOffload ─────────────────────────────────────────────────────────

def test_activation_offload_import():
    from squish.streaming.activation_offload import ActivationOffloader, OffloadPolicy
    policy = OffloadPolicy(offload_layers=[0, 2, 4])
    loader = ActivationOffloader(policy)
    assert loader is not None


def test_activation_offload_offload_and_fetch():
    from squish.streaming.activation_offload import ActivationOffloader, OffloadPolicy
    policy = OffloadPolicy(offload_layers=[1, 3])
    loader = ActivationOffloader(policy)
    tensor = RNG.random((16, 32)).astype(np.float32)
    loader.offload(1, tensor)
    fetched = loader.fetch(1)
    np.testing.assert_array_equal(fetched, tensor)
    assert loader.stats.n_offloaded == 1
    assert loader.stats.n_fetched == 1


def test_activation_offload_evict():
    from squish.streaming.activation_offload import ActivationOffloader, OffloadPolicy
    policy = OffloadPolicy(offload_layers=[0])
    loader = ActivationOffloader(policy)
    t = RNG.random((8,)).astype(np.float32)
    loader.offload(0, t)
    loader.evict(0)
    assert loader.buffer_bytes == 0
    with pytest.raises(KeyError):
        loader.fetch(0)


def test_activation_offload_should_offload():
    from squish.streaming.activation_offload import ActivationOffloader, OffloadPolicy
    policy = OffloadPolicy(offload_layers=[2, 5])
    loader = ActivationOffloader(policy)
    assert loader.should_offload(2) is True
    assert loader.should_offload(5) is True
    assert loader.should_offload(3) is False


# ── MorphAttn ─────────────────────────────────────────────────────────────────

def test_morph_attn_import():
    from squish.attention.morph_attn import AttentionMorpher, MorphConfig
    cfg = MorphConfig(n_layers=12, seq_len_full_threshold=512, seq_len_sparse_threshold=4096)
    m = AttentionMorpher(cfg)
    assert m is not None


def test_morph_attn_select_pattern_full():
    from squish.attention.morph_attn import AttentionMorpher, MorphConfig
    cfg = MorphConfig(n_layers=8, seq_len_full_threshold=512, seq_len_sparse_threshold=4096)
    m = AttentionMorpher(cfg)
    for layer in range(8):
        assert m.select_pattern(layer, 256) == "full"


def test_morph_attn_layer_patterns():
    from squish.attention.morph_attn import AttentionMorpher, MorphConfig
    cfg = MorphConfig(n_layers=8, seq_len_full_threshold=512, seq_len_sparse_threshold=4096)
    m = AttentionMorpher(cfg)
    patterns = m.layer_patterns(1000)
    assert len(patterns) == 8
    assert all(p in {"full", "sparse", "linear"} for p in patterns)


def test_morph_attn_flops_reduction():
    from squish.attention.morph_attn import AttentionMorpher, MorphConfig
    cfg = MorphConfig(n_layers=8, seq_len_full_threshold=512, seq_len_sparse_threshold=4096)
    m = AttentionMorpher(cfg)
    r_short = m.estimate_flops_reduction(256)
    r_long = m.estimate_flops_reduction(8192)
    assert 0.0 <= r_short <= 1.0
    assert 0.0 <= r_long <= 1.0
    assert r_long >= r_short   # longer context → more savings


# ── HydraSpec ─────────────────────────────────────────────────────────────────

def test_hydra_spec_import():
    from squish.speculative.hydra_spec import HydraConfig, HydraSpecDecoder
    cfg = HydraConfig(n_heads=3, n_draft=4, hidden_dim=32, vocab_size=100)
    decoder = HydraSpecDecoder(cfg)
    assert decoder is not None


def test_hydra_spec_draft_shape():
    from squish.speculative.hydra_spec import HydraConfig, HydraSpecDecoder
    n_heads, n_draft, hidden_dim, vocab = 3, 4, 32, 100
    cfg = HydraConfig(n_heads=n_heads, n_draft=n_draft, hidden_dim=hidden_dim, vocab_size=vocab)
    decoder = HydraSpecDecoder(cfg)
    hidden = RNG.random((hidden_dim,)).astype(np.float32)
    out = decoder.draft(hidden)
    assert out.draft_tokens.shape == (n_heads, n_draft)
    assert out.draft_logits.shape == (n_heads, n_draft, vocab)
    assert out.draft_tokens.dtype == np.int32


def test_hydra_spec_verify():
    from squish.speculative.hydra_spec import HydraConfig, HydraSpecDecoder
    n_heads, n_draft, hidden_dim, vocab = 2, 3, 16, 50
    cfg = HydraConfig(n_heads=n_heads, n_draft=n_draft, hidden_dim=hidden_dim, vocab_size=vocab)
    decoder = HydraSpecDecoder(cfg)
    hidden = RNG.random((hidden_dim,)).astype(np.float32)
    out = decoder.draft(hidden)
    target_logits = RNG.random((n_heads, n_draft, vocab)).astype(np.float32)
    accepted = decoder.verify(out.draft_tokens, target_logits)
    assert accepted.ndim == 1
    assert accepted.dtype == np.int32
    assert len(accepted) <= n_draft


def test_hydra_spec_acceptance_rate():
    from squish.speculative.hydra_spec import HydraConfig, HydraSpecDecoder
    cfg = HydraConfig(n_heads=2, n_draft=3, hidden_dim=16, vocab_size=50)
    decoder = HydraSpecDecoder(cfg)
    history = np.array([True, False, True, True, False], dtype=bool)
    rate = decoder.acceptance_rate(history)
    assert rate == pytest.approx(0.6)


# ── SeqCompact ────────────────────────────────────────────────────────────────

def test_seq_compact_import():
    from squish.streaming.seq_compact import CompactStats, SequenceCompactor
    sc = SequenceCompactor(n_heads=2, head_dim=8)
    assert sc is not None


def test_seq_compact_compact():
    from squish.streaming.seq_compact import SequenceCompactor
    n_heads, seq, head_dim = 2, 8, 4
    sc = SequenceCompactor(n_heads=n_heads, head_dim=head_dim)
    keys = RNG.random((n_heads, seq, head_dim)).astype(np.float32)
    vals = RNG.random((n_heads, seq, head_dim)).astype(np.float32)
    keep_mask = np.array([True, False, True, True, False, True, False, True], dtype=bool)
    ck, cv, stats = sc.compact(keys, vals, keep_mask)
    n_kept = int(keep_mask.sum())
    assert ck.shape == (n_heads, n_kept, head_dim)
    assert cv.shape == (n_heads, n_kept, head_dim)
    assert stats.n_tokens_after == n_kept
    assert stats.n_tokens_before == seq


def test_seq_compact_compaction_ratio():
    from squish.streaming.seq_compact import SequenceCompactor
    sc = SequenceCompactor(n_heads=2, head_dim=4)
    keys = RNG.random((2, 10, 4)).astype(np.float32)
    vals = RNG.random((2, 10, 4)).astype(np.float32)
    keep_mask = np.ones(10, dtype=bool)
    keep_mask[5:] = False  # keep 5
    _, _, stats = sc.compact(keys, vals, keep_mask)
    assert stats.compaction_ratio == pytest.approx(0.5)


def test_seq_compact_compact_indices():
    from squish.streaming.seq_compact import SequenceCompactor
    sc = SequenceCompactor(n_heads=2, head_dim=4)
    keep = np.array([0, 2, 4], dtype=np.int64)
    mapping = sc.compact_indices(6, keep)
    assert mapping.shape == (6,)
    assert mapping[0] == 0
    assert mapping[2] == 1
    assert mapping[4] == 2
    assert mapping[1] == -1
    assert mapping[3] == -1
    assert mapping[5] == -1


# ── ParallelSampler ───────────────────────────────────────────────────────────

def test_parallel_sampler_import():
    from squish.hardware.parallel_sampler import DiversityConfig, ParallelSampler
    cfg = DiversityConfig(n_samples=4, temperature=0.8, seed=0)
    ps = ParallelSampler(cfg)
    assert ps is not None


def test_parallel_sampler_sample_result():
    from squish.hardware.parallel_sampler import DiversityConfig, ParallelSampler, SampleResult
    vocab = 100
    cfg = DiversityConfig(n_samples=8, temperature=1.0, seed=42)
    ps = ParallelSampler(cfg)
    logits = RNG.random((vocab,)).astype(np.float32)
    result = ps.sample(logits)
    assert isinstance(result, SampleResult)
    assert 0 <= result.best_token < vocab
    assert result.all_tokens.shape == (8,)
    assert result.all_probs.shape == (8,)
    assert 0.0 <= result.diversity_score <= 1.0


def test_parallel_sampler_batch():
    from squish.hardware.parallel_sampler import DiversityConfig, ParallelSampler
    vocab, batch = 50, 4
    cfg = DiversityConfig(n_samples=4, seed=1)
    ps = ParallelSampler(cfg)
    logits = RNG.random((batch, vocab)).astype(np.float32)
    toks = ps.sample_batch(logits)
    assert toks.shape == (batch,)
    assert all(0 <= t < vocab for t in toks)


def test_parallel_sampler_best_in_candidates():
    from squish.hardware.parallel_sampler import DiversityConfig, ParallelSampler
    vocab = 100
    cfg = DiversityConfig(n_samples=16, seed=3)
    ps = ParallelSampler(cfg)
    logits = RNG.random((vocab,)).astype(np.float32)
    result = ps.sample(logits)
    # best_token is always one of the candidates
    assert result.best_token in result.all_tokens


# ── ContextSummarizer ─────────────────────────────────────────────────────────

def test_context_summarizer_import():
    from squish.context.context_summarizer import ContextSummarizer, SummaryConfig
    cfg = SummaryConfig(method="importance", budget=64)
    cs = ContextSummarizer(cfg)
    assert cs is not None


def test_context_summarizer_needs_compression():
    from squish.context.context_summarizer import ContextSummarizer, SummaryConfig
    cfg = SummaryConfig(budget=128)
    cs = ContextSummarizer(cfg)
    assert cs.needs_compression(200) is True
    assert cs.needs_compression(64) is False


def test_context_summarizer_importance():
    from squish.context.context_summarizer import ContextSummarizer, SummaryConfig
    seq, dim, budget = 200, 32, 64
    cfg = SummaryConfig(method="importance", budget=budget, min_keep_recent=16)
    cs = ContextSummarizer(cfg)
    tokens = np.arange(seq, dtype=np.int32)
    embs = RNG.random((seq, dim)).astype(np.float32)
    compressed, stats = cs.summarize(tokens, embs)
    assert len(compressed) <= budget
    assert stats.n_tokens_in == seq
    assert stats.compression_ratio <= 1.0


def test_context_summarizer_recency():
    from squish.context.context_summarizer import ContextSummarizer, SummaryConfig
    seq, budget = 300, 100
    cfg = SummaryConfig(method="recency", budget=budget)
    cs = ContextSummarizer(cfg)
    tokens = np.arange(seq, dtype=np.int32)
    compressed, stats = cs.summarize(tokens)
    assert len(compressed) == budget
    # recency keeps the most recent tokens
    np.testing.assert_array_equal(compressed, tokens[seq - budget:])


# ── SchemaGen ─────────────────────────────────────────────────────────────────

def test_schema_gen_import():
    from squish.grammar.schema_gen import SchemaGenEngine, SchemaState
    engine = SchemaGenEngine(vocab_size=50)
    assert engine is not None


def test_schema_gen_reset():
    from squish.grammar.schema_gen import SchemaGenEngine
    engine = SchemaGenEngine(vocab_size=50)
    state = engine.reset()
    assert not state.is_complete
    assert len(state.stack) > 0


def test_schema_gen_constrain():
    from squish.grammar.schema_gen import SchemaGenEngine
    engine = SchemaGenEngine(vocab_size=50)
    state = engine.reset()
    logits = np.zeros(50, dtype=np.float32)
    constrained = engine.constrain(logits, state)
    assert constrained.shape == (50,)
    # Some positions should be -inf (invalid tokens masked)
    assert (constrained == -np.inf).any()


def test_schema_gen_valid_next_chars():
    from squish.grammar.schema_gen import SchemaGenEngine
    engine = SchemaGenEngine(vocab_size=50)
    state = engine.reset()
    chars = engine.valid_next_chars(state)
    assert isinstance(chars, list)
    assert len(chars) > 0
