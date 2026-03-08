"""tests/test_online_sd_unit.py — 100 % coverage for squish/online_sd.py"""
import numpy as np
import pytest

from squish.online_sd import (
    OnlineDraftUpdater,
    OnlineSDConfig,
    OnlineSDStats,
    OnlineTraceBuffer,
)

RNG = np.random.default_rng(99)


# ---------------------------------------------------------------------------
# OnlineSDConfig
# ---------------------------------------------------------------------------

class TestOnlineSDConfig:
    def test_defaults(self):
        cfg = OnlineSDConfig()
        assert cfg.buffer_capacity == 512
        assert cfg.update_every == 64
        assert cfg.learning_rate == pytest.approx(1e-4)
        assert cfg.lora_rank == 16
        assert cfg.min_acceptance_rate == pytest.approx(0.85)

    def test_custom(self):
        cfg = OnlineSDConfig(buffer_capacity=32, update_every=8, lora_rank=0)
        assert cfg.buffer_capacity == 32
        assert cfg.lora_rank == 0


# ---------------------------------------------------------------------------
# OnlineTraceBuffer
# ---------------------------------------------------------------------------

class TestOnlineTraceBuffer:
    def test_initial_state(self):
        buf = OnlineTraceBuffer(capacity=8)
        assert len(buf) == 0
        assert buf.capacity == 8
        assert not buf.is_full()

    def test_invalid_capacity(self):
        with pytest.raises(ValueError, match="capacity"):
            OnlineTraceBuffer(capacity=0)

    def test_add_samples(self):
        buf = OnlineTraceBuffer(4)
        h = np.zeros(16, dtype=np.float32)
        buf.add(h, 5)
        assert len(buf) == 1

    def test_eviction_at_capacity(self):
        buf = OnlineTraceBuffer(3)
        for i in range(5):
            buf.add(np.ones(4, dtype=np.float32) * i, i)
        assert len(buf) == 3

    def test_is_full(self):
        buf = OnlineTraceBuffer(2)
        buf.add(np.zeros(4, dtype=np.float32), 0)
        assert not buf.is_full()
        buf.add(np.zeros(4, dtype=np.float32), 1)
        assert buf.is_full()

    def test_add_copies_array(self):
        buf = OnlineTraceBuffer(4)
        h = np.ones(4, dtype=np.float32)
        buf.add(h, 42)
        h[:] = 99.0
        hiddens, _ = buf.get_batch()
        # Should not have been mutated
        np.testing.assert_array_equal(hiddens[0], np.ones(4))

    def test_get_batch_returns_correct_shapes(self):
        buf = OnlineTraceBuffer(8)
        for i in range(3):
            buf.add(np.ones(10, dtype=np.float32), i)
        hiddens, tokens = buf.get_batch()
        assert hiddens.shape == (3, 10)
        assert tokens.shape == (3,)
        assert tokens.dtype == np.int64

    def test_get_batch_raises_when_empty(self):
        buf = OnlineTraceBuffer(4)
        with pytest.raises(RuntimeError, match="empty"):
            buf.get_batch()

    def test_clear_empties_buffer(self):
        buf = OnlineTraceBuffer(4)
        buf.add(np.zeros(4), 0)
        buf.clear()
        assert len(buf) == 0


# ---------------------------------------------------------------------------
# OnlineDraftUpdater
# ---------------------------------------------------------------------------

class TestOnlineDraftUpdater:
    HIDDEN = 8
    VOCAB = 32

    def _make(self, lora_rank=4, update_every=2):
        cfg = OnlineSDConfig(
            buffer_capacity=16,
            update_every=update_every,
            learning_rate=1e-3,
            lora_rank=lora_rank,
        )
        return OnlineDraftUpdater(cfg, hidden_dim=self.HIDDEN, vocab_size=self.VOCAB)

    def _fill(self, updater, n=4):
        for i in range(n):
            h = RNG.standard_normal(self.HIDDEN).astype(np.float32)
            updater.record(h, i % self.VOCAB)

    def test_initial_state(self):
        u = self._make()
        assert u.total_updates == 0
        assert u.hidden_dim == self.HIDDEN
        assert u.vocab_size == self.VOCAB
        assert len(u.buffer) == 0

    def test_default_config_when_none(self):
        u = OnlineDraftUpdater(config=None, hidden_dim=self.HIDDEN, vocab_size=self.VOCAB)
        assert u.total_updates == 0

    def test_record_adds_to_buffer(self):
        u = self._make()
        h = np.zeros(self.HIDDEN, dtype=np.float32)
        u.record(h, 5)
        assert len(u.buffer) == 1

    def test_should_update_false_initially(self):
        u = self._make(update_every=4)
        self._fill(u, 3)
        assert not u.should_update()

    def test_should_update_true_after_enough_samples(self):
        u = self._make(update_every=2)
        self._fill(u, 2)
        assert u.should_update()

    def test_compute_loss_standard_weight(self):
        u = self._make()
        self._fill(u, 4)
        weight = RNG.standard_normal((self.VOCAB, self.HIDDEN)).astype(np.float32)
        loss = u.compute_loss(weight)
        assert isinstance(loss, float)
        assert loss > 0.0

    def test_compute_loss_transposed_weight(self):
        u = self._make()
        self._fill(u, 4)
        # weight shape (hidden, vocab) → transposed convention
        weight_T = RNG.standard_normal((self.HIDDEN, self.VOCAB)).astype(np.float32)
        loss = u.compute_loss(weight_T)
        assert isinstance(loss, float)

    def test_compute_loss_invalid_weight_shape(self):
        u = self._make()
        self._fill(u, 2)
        bad_weight = np.ones((5, 5), dtype=np.float32)
        with pytest.raises(ValueError, match="incompatible"):
            u.compute_loss(bad_weight)

    def test_apply_update_returns_same_shape(self):
        u = self._make(lora_rank=4, update_every=2)
        self._fill(u, 4)
        weight = RNG.standard_normal((self.VOCAB, self.HIDDEN)).astype(np.float32)
        updated = u.apply_update(weight)
        assert updated.shape == weight.shape

    def test_apply_update_changes_weight(self):
        u = self._make(lora_rank=4, update_every=2)
        self._fill(u, 4)
        weight = RNG.standard_normal((self.VOCAB, self.HIDDEN)).astype(np.float32)
        orig = weight.copy()
        updated = u.apply_update(weight)
        assert not np.allclose(updated, orig)

    def test_apply_update_increments_total_updates(self):
        u = self._make(lora_rank=4, update_every=2)
        self._fill(u, 4)
        weight = RNG.standard_normal((self.VOCAB, self.HIDDEN)).astype(np.float32)
        u.apply_update(weight)
        assert u.total_updates == 1

    def test_apply_update_resets_samples_counter(self):
        u = self._make(update_every=2)
        self._fill(u, 4)
        w = RNG.standard_normal((self.VOCAB, self.HIDDEN)).astype(np.float32)
        u.apply_update(w)
        # After reset, should_update becomes False until enough new samples
        assert not u.should_update()

    def test_apply_update_no_lora(self):
        # lora_rank=0 → direct weight update
        u = self._make(lora_rank=0, update_every=2)
        self._fill(u, 4)
        weight = RNG.standard_normal((self.VOCAB, self.HIDDEN)).astype(np.float32)
        orig = weight.copy()
        updated = u.apply_update(weight)
        assert not np.allclose(updated, orig)

    def test_lora_delta_with_lora(self):
        u = self._make(lora_rank=4)
        delta = u.lora_delta()
        assert delta is not None
        assert delta.shape == (self.VOCAB, self.HIDDEN)

    def test_lora_delta_without_lora(self):
        u = self._make(lora_rank=0)
        assert u.lora_delta() is None

    def test_reset_clears_buffer_and_counters(self):
        u = self._make()
        self._fill(u, 4)
        w = RNG.standard_normal((self.VOCAB, self.HIDDEN)).astype(np.float32)
        u.apply_update(w)
        u.reset()
        assert len(u.buffer) == 0
        assert u.total_updates == 0
        assert not u.should_update()

    def test_reset_zeroes_lora(self):
        u = self._make(lora_rank=4, update_every=2)
        self._fill(u, 4)
        w = RNG.standard_normal((self.VOCAB, self.HIDDEN)).astype(np.float32)
        u.apply_update(w)
        u.reset()
        delta = u.lora_delta()
        assert delta is not None
        np.testing.assert_array_equal(delta, np.zeros((self.VOCAB, self.HIDDEN)))

    def test_reset_no_lora_branch(self):
        # lora_rank=0 → lora_a is None → False branch of 'if lora_a is not None'
        u = self._make(lora_rank=0)
        u.reset()
        assert len(u.buffer) == 0
        assert u.total_updates == 0


# ---------------------------------------------------------------------------
# OnlineSDStats
# ---------------------------------------------------------------------------

class TestOnlineSDStats:
    def test_defaults(self):
        s = OnlineSDStats()
        assert s.total_drafted == 0
        assert s.total_accepted == 0
        assert s.update_count == 0
        assert s.pre_update_acceptance == pytest.approx(0.0)

    def test_acceptance_rate_zero_when_no_drafts(self):
        assert OnlineSDStats().acceptance_rate == 0.0

    def test_acceptance_rate_nonzero(self):
        s = OnlineSDStats()
        s.record_step(10, 7)
        assert s.acceptance_rate == pytest.approx(0.7)

    def test_record_step_accumulates(self):
        s = OnlineSDStats()
        s.record_step(5, 3)
        s.record_step(5, 4)
        assert s.total_drafted == 10
        assert s.total_accepted == 7

    def test_record_update_captures_acceptance(self):
        s = OnlineSDStats()
        s.record_step(10, 8)
        s.record_update()
        assert s.update_count == 1
        assert s.pre_update_acceptance == pytest.approx(0.8)

    def test_reset(self):
        s = OnlineSDStats()
        s.record_step(10, 8)
        s.record_update()
        s.reset()
        assert s.total_drafted == 0
        assert s.total_accepted == 0
        assert s.update_count == 0
        assert s.pre_update_acceptance == pytest.approx(0.0)
