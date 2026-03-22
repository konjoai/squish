"""tests/test_wave43a_modules.py

Tests for Wave 43a modules:
  - MTPDecode (squish/speculative/mtp_decode.py)
  - CascadeKV (squish/kv/cascade_kv.py)
  - HeadPruner (squish/model/head_pruner.py)
  - PagedAttention (squish/kv/paged_attn.py)
  - LayerCollapse (squish/model/layer_collapse.py)
  - RelayAttention (squish/attention/relay_attn.py)
"""

from __future__ import annotations

import pytest
import numpy as np

# ── MTPDecode ─────────────────────────────────────────────────────────────────

from squish.speculative.mtp_decode import MTPConfig, MTPDraftResult, MTPDecode


class TestMTPConfig:
    def test_defaults(self):
        cfg = MTPConfig()
        assert cfg.n_heads >= 1
        assert cfg.vocab_size > 0
        assert cfg.hidden_size > 0

    def test_custom(self):
        cfg = MTPConfig(n_heads=3, vocab_size=256, hidden_size=64)
        assert cfg.n_heads == 3

    def test_invalid_n_heads(self):
        with pytest.raises((ValueError, Exception)):
            MTPConfig(n_heads=0)

    def test_repr(self):
        cfg = MTPConfig(n_heads=2)
        assert "2" in repr(cfg)


class TestMTPDecode:
    def _make(self, vocab=64, hidden=32, n_heads=2):
        cfg = MTPConfig(n_heads=n_heads, vocab_size=vocab, hidden_size=hidden)
        return MTPDecode(cfg)

    def test_step_returns_draft_result(self):
        model = self._make()
        h = np.random.randn(32).astype(np.float32)
        result = model.step(h)
        assert isinstance(result, MTPDraftResult)

    def test_draft_tokens_length(self):
        model = self._make(n_heads=3)
        h = np.random.randn(32).astype(np.float32)
        result = model.step(h)
        assert len(result.draft_tokens) == 3

    def test_draft_tokens_in_vocab(self):
        model = self._make(vocab=64)
        h = np.random.randn(32).astype(np.float32)
        result = model.step(h)
        for tok in result.draft_tokens:
            assert 0 <= tok < 64

    def test_log_probs_shape(self):
        model = self._make(n_heads=2, vocab=64)
        h = np.random.randn(32).astype(np.float32)
        result = model.step(h)
        assert result.log_probs.shape[0] == 2

    def test_verify_and_accept(self):
        model = self._make(n_heads=2, vocab=64)
        h = np.random.randn(32).astype(np.float32)
        result = model.step(h)
        target_lp = np.log(np.ones((2, 64)) / 64)
        accepted = model.verify_and_accept(result, target_lp)
        assert isinstance(accepted, MTPDraftResult)
        assert len(accepted.draft_tokens) <= len(result.draft_tokens)

    def test_reset(self):
        model = self._make()
        model.step(np.random.randn(32).astype(np.float32))
        model.reset()  # should not raise

    def test_default_config(self):
        model = MTPDecode()
        assert model.config is not None

    def test_single_head(self):
        model = self._make(n_heads=1)
        h = np.random.randn(32).astype(np.float32)
        result = model.step(h)
        assert len(result.draft_tokens) == 1

    def test_accepted_tokens_subset(self):
        model = self._make(n_heads=3, vocab=16)
        h = np.random.randn(32).astype(np.float32)
        result = model.step(h)
        uniform_lp = np.log(np.ones((3, 16)) / 16)
        accepted = model.verify_and_accept(result, uniform_lp)
        assert accepted.n_accepted <= 3

    def test_n_accepted_field(self):
        model = self._make(n_heads=2)
        h = np.random.randn(32).astype(np.float32)
        result = model.step(h)
        uniform_lp = np.log(np.ones((2, 64)) / 64)
        accepted = model.verify_and_accept(result, uniform_lp)
        assert 0 <= accepted.n_accepted <= 2

    def test_repeated_steps(self):
        model = self._make()
        for _ in range(5):
            model.step(np.random.randn(32).astype(np.float32))

    def test_high_dim_hidden(self):
        model = self._make(hidden=128, vocab=32)
        h = np.random.randn(128).astype(np.float32)
        result = model.step(h)
        assert isinstance(result, MTPDraftResult)


# ── CascadeKV ──────────────────────────────────────────────────────────────────

from squish.kv.cascade_kv import CascadeKVConfig, CascadeKVBlock, CascadeKV


class TestCascadeKVConfig:
    def test_defaults(self):
        cfg = CascadeKVConfig()
        assert cfg.n_heads >= 1
        assert cfg.head_dim > 0

    def test_custom(self):
        cfg = CascadeKVConfig(n_heads=4, head_dim=32)
        assert cfg.n_heads == 4
        assert cfg.head_dim == 32


class TestCascadeKV:
    def _make(self, n_heads=2, head_dim=8):
        cfg = CascadeKVConfig(n_heads=n_heads, head_dim=head_dim)
        return CascadeKV(cfg)

    def test_set_shared_prefix(self):
        kv = self._make()
        keys = np.random.randn(4, 2, 8).astype(np.float32)
        values = np.random.randn(4, 2, 8).astype(np.float32)
        kv.set_shared_prefix(keys, values)  # should not raise

    def test_create_request_returns_int(self):
        kv = self._make()
        rid = kv.create_request()
        assert isinstance(rid, int)

    def test_multiple_requests(self):
        kv = self._make()
        r1 = kv.create_request()
        r2 = kv.create_request()
        assert r1 != r2

    def test_append_token(self):
        kv = self._make(n_heads=2, head_dim=8)
        rid = kv.create_request()
        k = np.random.randn(2, 8).astype(np.float32)
        v = np.random.randn(2, 8).astype(np.float32)
        kv.append_token(rid, k, v)

    def test_attend_with_prefix(self):
        kv = self._make(n_heads=2, head_dim=8)
        prefix_k = np.random.randn(4, 2, 8).astype(np.float32)
        prefix_v = np.random.randn(4, 2, 8).astype(np.float32)
        kv.set_shared_prefix(prefix_k, prefix_v)
        rid = kv.create_request()
        k = np.random.randn(2, 8).astype(np.float32)
        v = np.random.randn(2, 8).astype(np.float32)
        kv.append_token(rid, k, v)
        q = np.random.randn(2, 8).astype(np.float32)
        out = kv.attend(rid, q)
        assert out.shape == (2, 8)

    def test_attend_without_prefix(self):
        kv = self._make(n_heads=2, head_dim=8)
        rid = kv.create_request()
        k = np.random.randn(2, 8).astype(np.float32)
        v = np.random.randn(2, 8).astype(np.float32)
        kv.append_token(rid, k, v)
        q = np.random.randn(2, 8).astype(np.float32)
        out = kv.attend(rid, q)
        assert out.shape == (2, 8)

    def test_output_dtype(self):
        kv = self._make(n_heads=2, head_dim=8)
        rid = kv.create_request()
        kv.append_token(rid, np.random.randn(2, 8).astype(np.float32), np.random.randn(2, 8).astype(np.float32))
        out = kv.attend(rid, np.random.randn(2, 8).astype(np.float32))
        assert out.dtype == np.float32

    def test_multi_token_append(self):
        kv = self._make(n_heads=2, head_dim=8)
        rid = kv.create_request()
        for _ in range(5):
            kv.append_token(rid, np.random.randn(2, 8).astype(np.float32), np.random.randn(2, 8).astype(np.float32))
        out = kv.attend(rid, np.random.randn(2, 8).astype(np.float32))
        assert out.shape == (2, 8)


# ── HeadPruner ─────────────────────────────────────────────────────────────────

from squish.model.head_pruner import HeadPrunerConfig, PruningMask, HeadPruner


class TestHeadPrunerConfig:
    def test_defaults(self):
        cfg = HeadPrunerConfig()
        assert cfg.n_heads >= 1
        assert 0.0 < cfg.keep_fraction <= 1.0

    def test_custom(self):
        cfg = HeadPrunerConfig(n_heads=8, keep_fraction=0.5)
        assert cfg.n_heads == 8


class TestHeadPruner:
    def _make(self, n_heads=4, head_dim=8):
        cfg = HeadPrunerConfig(n_heads=n_heads, head_dim=head_dim, keep_fraction=0.75)
        return HeadPruner(cfg)

    def test_score_heads_shape(self):
        pruner = self._make(n_heads=4, head_dim=8)
        W_o = np.random.randn(4, 8, 16).astype(np.float32)
        scores = pruner.score_heads(W_o)
        assert scores.shape == (4,)

    def test_calibrate_and_mask(self):
        pruner = self._make(n_heads=4, head_dim=8)
        hidden = np.random.randn(10, 32).astype(np.float32)
        pruner.calibrate(hidden)
        mask = pruner.compute_mask()
        assert isinstance(mask, PruningMask)

    def test_mask_keep_fraction(self):
        pruner = self._make(n_heads=8, head_dim=8)
        pruner.calibrate(np.random.randn(10, 64).astype(np.float32))
        mask = pruner.compute_mask()
        n_kept = int(mask.head_mask.sum())
        assert n_kept >= 1

    def test_apply_mask(self):
        pruner = self._make(n_heads=4, head_dim=8)
        pruner.calibrate(np.random.randn(10, 32).astype(np.float32))
        mask = pruner.compute_mask()
        hidden = np.random.randn(5, 32).astype(np.float32)
        out = pruner.apply_mask(hidden, mask)
        assert out.shape == hidden.shape

    def test_default_config(self):
        pruner = HeadPruner()
        assert pruner.config is not None

    def test_score_heads_non_negative(self):
        pruner = self._make()
        W_o = np.random.randn(4, 8, 16).astype(np.float32)
        scores = pruner.score_heads(W_o)
        assert (scores >= 0).all()

    def test_mask_is_boolean(self):
        pruner = self._make()
        pruner.calibrate(np.random.randn(5, 32).astype(np.float32))
        mask = pruner.compute_mask()
        assert mask.head_mask.dtype == bool or mask.head_mask.dtype == np.bool_


# ── PagedAttention ─────────────────────────────────────────────────────────────

from squish.kv.paged_attn import PagedAttnConfig, PagedAttnBlock, PagedAttention


class TestPagedAttnConfig:
    def test_defaults(self):
        cfg = PagedAttnConfig()
        assert cfg.block_size > 0
        assert cfg.n_heads >= 1
        assert cfg.head_dim > 0

    def test_custom(self):
        cfg = PagedAttnConfig(block_size=8, n_heads=2, head_dim=16)
        assert cfg.block_size == 8


class TestPagedAttention:
    def _make(self, block_size=4, n_heads=2, head_dim=8, max_blocks=32):
        cfg = PagedAttnConfig(block_size=block_size, n_heads=n_heads, head_dim=head_dim, max_blocks=max_blocks)
        return PagedAttention(cfg)

    def test_create_sequence(self):
        pa = self._make()
        seq_id = pa.create_sequence()
        assert isinstance(seq_id, int)

    def test_append_token(self):
        pa = self._make()
        sid = pa.create_sequence()
        k = np.random.randn(2, 8).astype(np.float32)
        v = np.random.randn(2, 8).astype(np.float32)
        pa.append_token(sid, k, v)

    def test_get_kv_returns_arrays(self):
        pa = self._make()
        sid = pa.create_sequence()
        k = np.random.randn(2, 8).astype(np.float32)
        v = np.random.randn(2, 8).astype(np.float32)
        pa.append_token(sid, k, v)
        K, V = pa.get_kv(sid)
        assert K.shape[-1] == 8
        assert V.shape[-1] == 8

    def test_free_sequence(self):
        pa = self._make()
        sid = pa.create_sequence()
        pa.append_token(sid, np.random.randn(2, 8).astype(np.float32), np.random.randn(2, 8).astype(np.float32))
        pa.free_sequence(sid)

    def test_multiple_sequences(self):
        pa = self._make()
        sids = [pa.create_sequence() for _ in range(3)]
        for sid in sids:
            pa.append_token(sid, np.random.randn(2, 8).astype(np.float32), np.random.randn(2, 8).astype(np.float32))
        for sid in sids:
            K, V = pa.get_kv(sid)
            assert K.shape[0] >= 1

    def test_prefix_sharing(self):
        pa = self._make(block_size=4)
        s1 = pa.create_sequence()
        for _ in range(4):
            pa.append_token(s1, np.random.randn(2, 8).astype(np.float32), np.random.randn(2, 8).astype(np.float32))
        s2 = pa.create_sequence()
        pa.share_prefix(s1, s2, prefix_len=4)
        K2, V2 = pa.get_kv(s2)
        assert K2.shape[0] >= 4

    def test_default_config(self):
        pa = PagedAttention()
        assert pa.config is not None

    def test_block_capacity(self):
        pa = self._make(block_size=4, max_blocks=8)
        sid = pa.create_sequence()
        for _ in range(8):
            pa.append_token(sid, np.random.randn(2, 8).astype(np.float32), np.random.randn(2, 8).astype(np.float32))
        K, V = pa.get_kv(sid)
        assert K.shape[0] == 8


# ── LayerCollapse ──────────────────────────────────────────────────────────────

from squish.model.layer_collapse import LayerCollapseConfig, CollapseSchedule, LayerCollapse


class TestLayerCollapseConfig:
    def test_defaults(self):
        cfg = LayerCollapseConfig()
        assert cfg.n_layers >= 2
        assert 0.0 < cfg.max_prune_fraction < 1.0

    def test_custom(self):
        cfg = LayerCollapseConfig(n_layers=8, max_prune_fraction=0.25)
        assert cfg.n_layers == 8


class TestLayerCollapse:
    def _make(self, n_layers=8, hidden=16):
        cfg = LayerCollapseConfig(n_layers=n_layers, hidden_size=hidden, max_prune_fraction=0.25)
        return LayerCollapse(cfg)

    def test_calibrate(self):
        lc = self._make(n_layers=4, hidden=16)
        hidden_states = [np.random.randn(5, 16).astype(np.float32) for _ in range(4)]
        lc.calibrate(hidden_states)

    def test_mean_similarities_shape(self):
        lc = self._make(n_layers=4, hidden=16)
        h = [np.random.randn(5, 16).astype(np.float32) for _ in range(4)]
        lc.calibrate(h)
        sims = lc.mean_similarities()
        assert len(sims) == 3  # n_layers - 1 adjacent pairs

    def test_similarities_in_range(self):
        lc = self._make(n_layers=4)
        h = [np.random.randn(5, 16).astype(np.float32) for _ in range(4)]
        lc.calibrate(h)
        sims = lc.mean_similarities()
        assert (np.array(sims) >= -1.0).all() and (np.array(sims) <= 1.0).all()

    def test_compute_schedule_returns_collapse_schedule(self):
        lc = self._make(n_layers=6, hidden=16)
        h = [np.random.randn(5, 16).astype(np.float32) for _ in range(6)]
        lc.calibrate(h)
        schedule = lc.compute_schedule()
        assert isinstance(schedule, CollapseSchedule)

    def test_skip_layers_bounded(self):
        lc = self._make(n_layers=8, hidden=16)
        h = [np.random.randn(5, 16).astype(np.float32) for _ in range(8)]
        lc.calibrate(h)
        lc.compute_schedule()
        n_skipped = sum(lc.should_skip(i) for i in range(8))
        assert n_skipped <= 2  # max 25% of 8

    def test_reset_calibration(self):
        lc = self._make()
        h = [np.random.randn(5, 16).astype(np.float32) for _ in range(8)]
        lc.calibrate(h)
        lc.reset_calibration()

    def test_default_config(self):
        lc = LayerCollapse()
        assert lc.config is not None


# ── RelayAttention ─────────────────────────────────────────────────────────────

from squish.attention.relay_attn import RelayAttnConfig, RelayAttention


class TestRelayAttnConfig:
    def test_defaults(self):
        cfg = RelayAttnConfig()
        assert cfg.n_heads >= 1
        assert cfg.head_dim > 0
        assert 0.0 < cfg.bypass_threshold < 1.0

    def test_custom(self):
        cfg = RelayAttnConfig(n_heads=4, head_dim=16, bypass_threshold=0.8)
        assert cfg.n_heads == 4


class TestRelayAttention:
    def _make(self, n_heads=2, head_dim=8):
        cfg = RelayAttnConfig(n_heads=n_heads, head_dim=head_dim, bypass_threshold=0.9)
        return RelayAttention(cfg)

    def test_attend_output_shape(self):
        ra = self._make(n_heads=2, head_dim=8)
        seq = 6
        q = np.random.randn(2, 1, 8).astype(np.float32)
        k = np.random.randn(2, seq, 8).astype(np.float32)
        v = np.random.randn(2, seq, 8).astype(np.float32)
        out, _ = ra.attend(q, k, v)
        assert out.shape == (2, 1, 8)

    def test_attend_returns_skip_mask(self):
        ra = self._make()
        q = np.random.randn(2, 1, 8).astype(np.float32)
        k = np.random.randn(2, 4, 8).astype(np.float32)
        v = np.random.randn(2, 4, 8).astype(np.float32)
        _, skip_mask = ra.attend(q, k, v)
        assert skip_mask.shape == (2,)

    def test_reset(self):
        ra = self._make()
        q = np.random.randn(2, 1, 8).astype(np.float32)
        k = np.random.randn(2, 4, 8).astype(np.float32)
        v = np.random.randn(2, 4, 8).astype(np.float32)
        ra.attend(q, k, v)
        ra.reset()

    def test_adapt_thresholds(self):
        ra = self._make()
        for _ in range(5):
            q = np.random.randn(2, 1, 8).astype(np.float32)
            k = np.random.randn(2, 4, 8).astype(np.float32)
            v = np.random.randn(2, 4, 8).astype(np.float32)
            ra.attend(q, k, v)
        ra.adapt_thresholds()

    def test_default_config(self):
        ra = RelayAttention()
        assert ra.config is not None

    def test_output_dtype_float32(self):
        ra = self._make()
        q = np.random.randn(2, 1, 8).astype(np.float32)
        k = np.random.randn(2, 4, 8).astype(np.float32)
        v = np.random.randn(2, 4, 8).astype(np.float32)
        out, _ = ra.attend(q, k, v)
        assert out.dtype == np.float32

    def test_relay_bank_filled(self):
        ra = self._make()
        for _ in range(3):
            q = np.random.randn(2, 1, 8).astype(np.float32)
            k = np.random.randn(2, 4, 8).astype(np.float32)
            v = np.random.randn(2, 4, 8).astype(np.float32)
            ra.attend(q, k, v)
        # After several steps, relay bank should be populated
        q = np.random.randn(2, 1, 8).astype(np.float32)
        k = np.random.randn(2, 4, 8).astype(np.float32)
        v = np.random.randn(2, 4, 8).astype(np.float32)
        out, skip = ra.attend(q, k, v)
        assert out is not None
