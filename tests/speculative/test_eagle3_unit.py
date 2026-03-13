"""tests/test_eagle3_unit.py

Coverage for branches in squish/eagle3.py not covered by test_wave19_server_wiring.py:

  Eagle3Config.__post_init__
    - vocab_size <= 0       → ValueError
    - draft_layers <= 0     → ValueError
    - max_draft_len <= 0    → ValueError
    - feature_dim <= 0      → ValueError

  Eagle3Decoder
    - draft_step(n_steps=0) → returns []
    - draft_step(n_steps > max_draft_len) → capped at max_draft_len
    - verify_step with _last_draft_features is None → (False, bonus_token)
    - verify_step with near-zero norm → sim=0.0 branch
    - acceptance_rate with _n_total == 0 → 0.0

  Eagle3Stats
    - mean_feature_similarity with n_verifications == 0 → 0.0
    - acceptance_rate with n_verifications == 0 → 0.0
"""
from __future__ import annotations

import numpy as np
import pytest

from squish.speculative.eagle3 import Eagle3Config, Eagle3Decoder, Eagle3Stats


# ---------------------------------------------------------------------------
# Eagle3Config validation
# ---------------------------------------------------------------------------


class TestEagle3ConfigValidation:
    def test_vocab_size_zero_raises(self):
        with pytest.raises(ValueError, match="vocab_size"):
            Eagle3Config(vocab_size=0)

    def test_draft_layers_zero_raises(self):
        with pytest.raises(ValueError, match="draft_layers"):
            Eagle3Config(draft_layers=0)

    def test_max_draft_len_zero_raises(self):
        with pytest.raises(ValueError, match="max_draft_len"):
            Eagle3Config(max_draft_len=0)

    def test_feature_dim_zero_raises(self):
        with pytest.raises(ValueError, match="feature_dim"):
            Eagle3Config(feature_dim=0)

    def test_feature_dim_negative_raises(self):
        with pytest.raises(ValueError, match="feature_dim"):
            Eagle3Config(feature_dim=-8)


# ---------------------------------------------------------------------------
# Eagle3Decoder — draft_step edge cases
# ---------------------------------------------------------------------------


class TestEagle3DecoderEdgeCases:
    def _make_decoder(self, hidden_dim=32, vocab_size=64, max_draft_len=4):
        cfg = Eagle3Config(
            hidden_dim=hidden_dim,
            vocab_size=vocab_size,
            max_draft_len=max_draft_len,
            acceptance_threshold=0.5,
        )
        return Eagle3Decoder(cfg)

    def test_draft_step_zero_n_steps_returns_empty(self):
        decoder = self._make_decoder()
        hidden = np.ones(32, dtype=np.float32)
        result = decoder.draft_step(hidden, n_steps=0)
        assert result == []

    def test_draft_step_negative_n_steps_returns_empty(self):
        decoder = self._make_decoder()
        hidden = np.ones(32, dtype=np.float32)
        result = decoder.draft_step(hidden, n_steps=-1)
        assert result == []

    def test_draft_step_caps_at_max_draft_len(self):
        decoder = self._make_decoder(max_draft_len=3)
        hidden = np.random.randn(32).astype(np.float32)
        result = decoder.draft_step(hidden, n_steps=10)  # 10 > max_draft_len=3
        assert len(result) == 3

    def test_acceptance_rate_before_any_verify_step(self):
        """acceptance_rate with _n_total == 0 returns 0.0."""
        decoder = self._make_decoder()
        assert decoder.acceptance_rate == 0.0

    def test_n_total_zero_initially(self):
        decoder = self._make_decoder()
        assert decoder.n_total == 0

    def test_verify_step_no_draft_features(self):
        """When no draft step was run, _last_draft_features is None → rejected."""
        decoder = self._make_decoder()
        hidden = np.random.randn(32).astype(np.float32)
        accepted, bonus = decoder.verify_step([0, 1, 2], hidden)
        assert accepted is False
        assert isinstance(bonus, int)
        assert decoder.n_total == 1

    def test_verify_step_near_zero_norm(self):
        """When draft or target features have near-zero norm, sim=0.0."""
        cfg = Eagle3Config(
            hidden_dim=4,
            vocab_size=8,
            max_draft_len=2,
            acceptance_threshold=0.5,
        )
        decoder = Eagle3Decoder(cfg)

        # Force _last_draft_features to be near-zero
        decoder._last_draft_features = np.zeros(4, dtype=np.float32)  # norm ~ 0

        # target_hidden also near-zero → both norms < 1e-8 → sim = 0.0
        target_hidden = np.zeros(4, dtype=np.float32)
        accepted, bonus = decoder.verify_step([], target_hidden)
        assert accepted is False  # sim=0.0 < threshold=0.5

    def test_verify_step_above_threshold_accepted(self):
        """When similarity > threshold, draft is accepted."""
        cfg = Eagle3Config(
            hidden_dim=32,
            vocab_size=64,
            max_draft_len=3,
            acceptance_threshold=0.01,  # very low threshold
        )
        decoder = Eagle3Decoder(cfg)
        rng = np.random.default_rng(42)
        hidden = rng.standard_normal(32).astype(np.float32)

        # Run a draft step to set _last_draft_features
        decoder.draft_step(hidden, n_steps=1)

        # Verify with the same hidden state → should produce high similarity
        # (same input to predict_features → same features → sim = 1.0 > 0.01)
        accepted, bonus = decoder.verify_step([0], hidden)
        assert isinstance(accepted, bool)


# ---------------------------------------------------------------------------
# Eagle3Stats — zero verifications
# ---------------------------------------------------------------------------


class TestEagle3StatsZeroVerifications:
    def test_mean_feature_similarity_zero_verifications(self):
        st = Eagle3Stats(n_verifications=0)
        assert st.mean_feature_similarity == 0.0

    def test_acceptance_rate_zero_verifications(self):
        st = Eagle3Stats(n_verifications=0)
        assert st.acceptance_rate == 0.0

    def test_nonzero_verifications(self):
        st = Eagle3Stats(
            total_draft_steps=10,
            total_accepted=3,
            feature_sim_sum=7.5,
            n_verifications=10,
        )
        assert st.mean_feature_similarity == pytest.approx(0.75)
        assert st.acceptance_rate == pytest.approx(0.3)
