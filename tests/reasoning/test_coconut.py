"""Tests for CoconutDecoder — continuous-latent chain-of-thought BFS decoder.

Uses the module's documented injection points (projection_head / answer_decoder)
with deterministic NumPy callables — the intended unit-test path, not E2E.
"""
from __future__ import annotations

import numpy as np
import pytest

from squish.reasoning.coconut import (
    CoconutConfig,
    CoconutDecoder,
    CoconutResult,
    LatentThoughtState,
)


class TestConfig:
    def test_defaults(self):
        c = CoconutConfig()
        assert c.max_latent_steps == 8 and c.beam_width == 4 and c.latent_dim == 256

    @pytest.mark.parametrize("kw", [
        {"max_latent_steps": 0}, {"beam_width": 0}, {"latent_dim": 0},
    ])
    def test_invalid(self, kw):
        with pytest.raises(ValueError):
            CoconutConfig(**kw)


class TestDataclasses:
    def test_latent_state_depth(self):
        st = LatentThoughtState(latent=np.zeros(4, np.float32), step=3)
        assert st.depth == 3

    def test_token_reduction_ratio(self):
        assert CoconutResult("a", n_latent_steps=0, used_fallback=True).token_reduction_ratio == 0.0
        assert CoconutResult("a", n_latent_steps=4, used_fallback=False).token_reduction_ratio == 0.25


class TestFallback:
    def test_fallback_used_when_no_head(self):
        d = CoconutDecoder(CoconutConfig(max_latent_steps=5, latent_dim=8))
        res = d.decode("Solve: 2+2=?")
        assert res.used_fallback is True
        assert "Solve" in res.answer
        assert 1 <= res.n_latent_steps <= 5
        assert res.best_beam is None

    def test_no_head_no_fallback_raises(self):
        d = CoconutDecoder(CoconutConfig(fallback_to_token_decode=False, latent_dim=8))
        with pytest.raises(RuntimeError, match="No projection_head"):
            d.decode("x")


class TestLatentSearch:
    def _decoder(self, **kw):
        cfg = CoconutConfig(latent_dim=4, max_latent_steps=3, beam_width=2, **kw)
        d = CoconutDecoder(cfg)
        # Deterministic projection head: decay the latent toward zero.
        d.install_projection_head(lambda init, lat: (lat * 0.9).astype(np.float32))
        return d

    def test_decode_with_head_runs_full_search(self):
        d = self._decoder()
        res = d.decode("prompt")
        assert res.used_fallback is False
        assert res.n_latent_steps == 3            # == max_latent_steps
        assert res.best_beam is not None
        assert res.best_beam.latent.shape == (4,)
        assert res.answer.startswith("latent_answer(")

    def test_injected_answer_decoder_is_used(self):
        d = self._decoder()
        d.install_answer_decoder(lambda latent: "INJECTED")
        assert d.decode("p").answer == "INJECTED"

    def test_bfs_calls_head_once_per_step(self):
        d = self._decoder()
        calls = {"n": 0}

        def head(init, lat):
            calls["n"] += 1
            return (lat * 0.5).astype(np.float32)

        d.install_projection_head(head)
        res = d.decode("p")
        # The head yields one child per beam, so the search is a single chain:
        # one call per step over max_latent_steps=3 (beam_width never binds).
        assert calls["n"] == 3
        assert res.best_beam.step == 3


class TestInitialiseLatent:
    def test_hidden_state_truncated(self):
        d = CoconutDecoder(CoconutConfig(latent_dim=4))
        out = d._initialise_latent("p", np.arange(10, dtype=np.float32))
        assert out.shape == (4,)
        np.testing.assert_array_equal(out, np.arange(4, dtype=np.float32))

    def test_hidden_state_padded(self):
        d = CoconutDecoder(CoconutConfig(latent_dim=6))
        out = d._initialise_latent("p", np.array([1.0, 2.0], np.float32))
        assert out.shape == (6,)
        np.testing.assert_array_equal(out[:2], [1.0, 2.0])
        assert np.count_nonzero(out[2:]) == 0

    def test_hash_stub_is_deterministic_within_run(self):
        d = CoconutDecoder(CoconutConfig(latent_dim=8))
        a = d._initialise_latent("same-prompt", None)
        b = d._initialise_latent("same-prompt", None)
        assert a.shape == (8,)
        np.testing.assert_array_equal(a, b)


class TestAnswerStub:
    def test_stub_with_two_or_more(self):
        d = CoconutDecoder(CoconutConfig(latent_dim=4))
        s = d._latent_to_answer_stub(np.array([1.2345, 6.789, 0.0, 0.0], np.float32))
        assert s.startswith("latent_answer(1.23")

    def test_stub_with_single_element(self):
        d = CoconutDecoder(CoconutConfig(latent_dim=1))
        assert d._latent_to_answer_stub(np.array([1.0], np.float32)) == "latent_answer(?)"
