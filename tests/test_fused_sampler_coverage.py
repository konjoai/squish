"""Behavioral coverage for ``squish.hardware.fused_sampler`` — the fused
temperature / top-k / top-p / min-p / repetition-penalty token sampler.
Pure numpy; no MLX/Metal.
"""

from __future__ import annotations

import numpy as np
import pytest

from squish.hardware.fused_sampler import FusedSampler, SamplerConfig


def _sampler(**kw):
    kw.setdefault("seed", 0)
    return FusedSampler(SamplerConfig(**kw))


# ── SamplerConfig validation ─────────────────────────────────────────────────


@pytest.mark.parametrize(
    "kw,msg",
    [
        ({"temperature": 0.0}, "temperature must be"),
        ({"top_k": -1}, "top_k must be"),
        ({"top_p": 0.0}, "top_p must be"),
        ({"top_p": 1.5}, "top_p must be"),
        ({"min_p": -0.1}, "min_p must be"),
        ({"min_p": 1.0}, "min_p must be"),
        ({"repetition_penalty": 0.0}, "repetition_penalty must be"),
    ],
)
def test_config_validation(kw, msg):
    with pytest.raises(ValueError, match=msg):
        SamplerConfig(**kw)


def test_config_defaults_ok():
    c = SamplerConfig()
    assert c.temperature == 1.0 and c.top_k == 0 and c.top_p == 1.0


# ── sample ───────────────────────────────────────────────────────────────────


def test_sample_requires_1d():
    with pytest.raises(ValueError, match="logits must be 1-D"):
        _sampler().sample(np.zeros((2, 3), np.float32))


def test_sample_sharp_distribution_picks_peak():
    logits = np.array([0.0, 100.0, 0.0, 0.0], dtype=np.float32)
    assert _sampler(temperature=0.5).sample(logits) == 1


def test_sample_deterministic_with_seed():
    logits = np.random.default_rng(0).standard_normal(50).astype(np.float32)
    a = _sampler(seed=42).sample(logits)
    b = _sampler(seed=42).sample(logits)
    assert a == b


# ── sample_batch ─────────────────────────────────────────────────────────────


def test_sample_batch_requires_2d():
    with pytest.raises(ValueError, match="logits must be 2-D"):
        _sampler().sample_batch(np.zeros(3, np.float32))


def test_sample_batch_returns_one_per_row():
    logits = np.stack(
        [
            np.array([100.0, 0.0, 0.0], np.float32),
            np.array([0.0, 0.0, 100.0], np.float32),
        ]
    )
    out = _sampler(temperature=0.5).sample_batch(logits)
    assert out.tolist() == [0, 2]


def test_sample_batch_shared_1d_input_ids():
    logits = np.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]], np.float32)
    out = _sampler(repetition_penalty=1.5).sample_batch(logits, input_ids=np.array([0, 1]))
    assert out.shape == (2,)


def test_sample_batch_per_row_2d_input_ids():
    logits = np.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]], np.float32)
    ids = np.array([[0], [2]])  # 2-D → per-row penalty
    out = _sampler(repetition_penalty=2.0).sample_batch(logits, input_ids=ids)
    assert out.shape == (2,)


# ── reset_rng ────────────────────────────────────────────────────────────────


def test_reset_rng_negative_raises():
    with pytest.raises(ValueError, match="seed must be"):
        _sampler().reset_rng(-1)


def test_reset_rng_makes_sampling_reproducible():
    s = _sampler(seed=1)
    logits = np.random.default_rng(0).standard_normal(40).astype(np.float32)
    s.reset_rng(7)
    a = s.sample(logits)
    s.reset_rng(7)
    b = s.sample(logits)
    assert a == b


# ── filters (via _compute_probs) ─────────────────────────────────────────────


def test_repetition_penalty_lowers_seen_token_prob():
    logits = np.array([5.0, 5.0, 5.0], np.float32)
    s = _sampler(repetition_penalty=2.0)
    probs = s._compute_probs(logits, np.array([0]))
    assert probs[0] < probs[1]  # token 0 penalised (positive logit / penalty)


def test_repetition_penalty_negative_logit_branch():
    logits = np.array([-4.0, 1.0, 1.0], np.float32)
    s = _sampler(repetition_penalty=2.0)
    probs = s._compute_probs(logits, np.array([0]))
    # Negative logit is multiplied by penalty (made more negative) → lower prob.
    assert probs[0] < probs[1]


def test_repetition_penalty_ignores_out_of_range_ids():
    logits = np.array([1.0, 2.0, 3.0], np.float32)
    s = _sampler(repetition_penalty=2.0)
    probs = s._compute_probs(logits, np.array([-5, 99]))  # all out of range → no-op
    np.testing.assert_allclose(probs, s._compute_probs(logits, None), rtol=1e-9)


def test_min_p_filter_zeros_low_prob_tokens():
    logits = np.array([10.0, 0.0, 0.0, 0.0], np.float32)
    p = _sampler(min_p=0.5)._compute_probs(logits, None)
    assert p[1] == 0.0 and p[0] > 0.0  # weak tokens masked


def test_top_k_keeps_only_k_tokens():
    logits = np.array([5.0, 4.0, 3.0, 2.0, 1.0], np.float32)
    p = _sampler(top_k=2)._compute_probs(logits, None)
    assert np.count_nonzero(p) == 2 and abs(p.sum() - 1.0) < 1e-9


def test_top_k_above_vocab_is_noop():
    logits = np.array([5.0, 4.0, 3.0], np.float32)
    p = _sampler(top_k=10)._compute_probs(logits, None)  # k >= vocab → keep all
    assert np.count_nonzero(p) == 3


def test_top_p_nucleus_truncates_tail():
    logits = np.array([3.0, 2.0, 1.0, 0.0, -5.0], np.float32)
    p = _sampler(top_p=0.6)._compute_probs(logits, None)
    assert abs(p.sum() - 1.0) < 1e-9
    assert np.count_nonzero(p) < 5  # tail beyond the nucleus zeroed


def test_top_p_when_remaining_mass_below_threshold():
    # top_k zeros all but the single peak whose softmax prob (~0.38) is below
    # top_p (0.6) → no cumulative entry reaches the threshold, so the nucleus
    # boundary loop is skipped and renormalise still yields a valid pmf.
    logits = np.array([1.0, 0.9, 0.8], np.float32)
    p = _sampler(top_k=1, top_p=0.6)._compute_probs(logits, None)
    assert np.count_nonzero(p) == 1 and abs(p.sum() - 1.0) < 1e-9
