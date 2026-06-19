"""Behavioral coverage for ``squish.experimental.jacobi_decode`` — the pure-numpy
Jacobi / Gauss-Seidel parallel fixed-point decoder. Deterministic ``logits_fn``
factories drive every convergence / timeout / init / variant branch.
"""

from __future__ import annotations

import numpy as np
import pytest

from squish.experimental.jacobi_decode import (
    JacobiConfig,
    JacobiDecoder,
    JacobiStats,
    _sample_token,
    _softmax,
)


def _logits_for(preds, vocab=8):
    """Build a (len, vocab) logit array with a sharp peak at each ``preds[r]``."""
    arr = np.full((len(preds), vocab), -10.0, np.float32)
    for r, t in enumerate(preds):
        arr[r, t] = 10.0
    return arr


def _identity_fn(vocab=8):
    """logits_fn whose argmax at each row equals that row's input token."""
    return lambda ids: _logits_for(ids, vocab)


def _alternating_fn(vocab=8):
    """logits_fn that predicts token 0 then 1 then 0 … on successive calls — never
    reaches a fixed point, forcing the timeout path."""
    state = {"c": 0}

    def fn(ids):
        tok = state["c"] % 2
        state["c"] += 1
        return _logits_for([tok] * len(ids), vocab)

    return fn


# ── JacobiConfig validation ──────────────────────────────────────────────────


@pytest.mark.parametrize(
    "kw,msg",
    [
        ({"n_tokens": 0}, "n_tokens must be"),
        ({"max_iter": 0}, "max_iter must be"),
        ({"variant": "nope"}, "variant must be"),
        ({"temperature": -1.0}, "temperature must be"),
        ({"init": "nope"}, "init must be"),
    ],
)
def test_config_validation(kw, msg):
    with pytest.raises(ValueError, match=msg):
        JacobiConfig(**kw)


def test_config_defaults():
    c = JacobiConfig()
    assert c.n_tokens == 4 and c.variant == "jacobi" and c.init == "uniform"


# ── helpers ──────────────────────────────────────────────────────────────────


def test_softmax_normalises():
    p = _softmax(np.array([1.0, 2.0, 3.0]))
    assert abs(p.sum() - 1.0) < 1e-9 and p[2] > p[0]


def test_sample_token_greedy_when_temp_zero():
    assert _sample_token(np.array([0.0, 9.0, 0.0]), 0.0, np.random.default_rng(0)) == 1


def test_sample_token_stochastic_when_temp_positive():
    rng = np.random.default_rng(0)
    tok = _sample_token(np.array([0.0, 100.0, 0.0]), 0.7, rng)
    assert tok == 1  # near-deterministic peak even with temperature


# ── JacobiStats ──────────────────────────────────────────────────────────────


def test_stats_zero_state():
    s = JacobiStats()
    assert s.mean_tokens_per_step == 0.0
    assert s.mean_iterations_per_step == 0.0
    assert s.fixed_point_rate == 0.0


def test_stats_nonzero_means_and_rate():
    s = JacobiStats(
        total_decode_steps=2,
        total_tokens_generated=6,
        total_iterations=4,
        total_fixed_points=6,
    )
    assert s.mean_tokens_per_step == 3.0
    assert s.mean_iterations_per_step == 2.0
    assert s.fixed_point_rate == 1.0
    assert "steps=2" in repr(s)


# ── decode_step: convergence ─────────────────────────────────────────────────


def test_jacobi_immediate_convergence():
    dec = JacobiDecoder(JacobiConfig(n_tokens=2, max_iter=4, variant="jacobi"))
    accepted, n_iter = dec.decode_step(_identity_fn(), context_ids=[7])
    assert accepted == [7, 7] and n_iter == 1
    assert dec.stats.total_decode_steps == 1


def test_gauss_seidel_convergence_and_check():
    dec = JacobiDecoder(JacobiConfig(n_tokens=2, max_iter=4, variant="gauss_seidel"))
    accepted, n_iter = dec.decode_step(_identity_fn(), context_ids=[3])
    assert accepted == [3, 3] and n_iter == 1


# ── decode_step: timeout paths ───────────────────────────────────────────────


def test_jacobi_timeout_with_correction_break():
    dec = JacobiDecoder(JacobiConfig(n_tokens=2, max_iter=2, variant="jacobi"))
    accepted, n_iter = dec.decode_step(_alternating_fn(), context_ids=[7])
    # never converges → timeout; final verify yields a correction → single token.
    assert n_iter == 2 and len(accepted) == 1


def test_gauss_seidel_timeout_when_never_fixed():
    dec = JacobiDecoder(JacobiConfig(n_tokens=2, max_iter=2, variant="gauss_seidel"))
    accepted, n_iter = dec.decode_step(_alternating_fn(), context_ids=[7])
    assert n_iter == 2 and len(accepted) >= 1


# ── decode_step: init strategies ─────────────────────────────────────────────


def test_random_init_uses_vocab(monkeypatch):
    dec = JacobiDecoder(JacobiConfig(n_tokens=2, max_iter=2, init="random"))
    accepted, _ = dec.decode_step(_identity_fn(), context_ids=[5], vocab_size=8)
    assert len(accepted) >= 1  # random guesses are identity-fixed → converge


def test_fallback_init_when_no_context():
    # init="uniform" but empty context → falls through to the [1]*n fallback.
    dec = JacobiDecoder(JacobiConfig(n_tokens=2, max_iter=2, init="uniform"))
    accepted, n_iter = dec.decode_step(_identity_fn(), context_ids=[])
    assert accepted == [1, 1] and n_iter == 1


# ── reset_stats / __repr__ ───────────────────────────────────────────────────


def test_reset_stats_clears():
    dec = JacobiDecoder(JacobiConfig(n_tokens=1, max_iter=2))
    dec.decode_step(_identity_fn(), context_ids=[2])
    assert dec.stats.total_decode_steps == 1
    dec.reset_stats()
    assert dec.stats.total_decode_steps == 0


def test_decoder_repr():
    dec = JacobiDecoder(JacobiConfig(n_tokens=3, variant="gauss_seidel"))
    assert "JacobiDecoder(n_tokens=3" in repr(dec)


def test_decoder_default_config():
    dec = JacobiDecoder()
    assert dec._cfg.n_tokens == 4
