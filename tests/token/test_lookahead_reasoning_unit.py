"""Unit tests for squish.token.lookahead_reasoning (LookaheadReasoning)."""

import pytest

from squish.token.lookahead_reasoning import (
    LookaheadBatch,
    LookaheadConfig,
    LookaheadReasoningEngine,
    LookaheadStats,
    LookaheadStep,
    _default_batch_verifier,
)


def _step(text="step", confidence=0.8, tokens=10, pos=0, accepted=False):
    return LookaheadStep(
        text=text, source="draft", confidence=confidence,
        tokens_used=tokens, batch_position=pos, accepted=accepted,
    )


def _draft_fn(confidence=0.8, tokens=10):
    def fn(context: str) -> LookaheadStep:
        return LookaheadStep(
            text=f"draft step {len(context) % 9}",
            source="draft",
            confidence=confidence,
            tokens_used=tokens,
        )
    return fn


def _verifier_all_accept():
    def fn(steps, context):
        return [1.0] * len(steps)
    return fn


def _verifier_all_reject():
    def fn(steps, context):
        return [0.0] * len(steps)
    return fn


# ---------------------------------------------------------------------------
# TestLookaheadConfig
# ---------------------------------------------------------------------------


class TestLookaheadConfig:
    def test_defaults(self):
        cfg = LookaheadConfig()
        assert cfg.lookahead_k == 4
        assert cfg.greedy_prefix_accept is True

    def test_invalid_lookahead_k(self):
        with pytest.raises(ValueError):
            LookaheadConfig(lookahead_k=0)

    def test_invalid_acceptance_score(self):
        with pytest.raises(ValueError):
            LookaheadConfig(min_acceptance_score=1.5)

    def test_invalid_max_step_tokens(self):
        with pytest.raises(ValueError):
            LookaheadConfig(max_step_tokens=0)


# ---------------------------------------------------------------------------
# TestLookaheadStep
# ---------------------------------------------------------------------------


class TestLookaheadStep:
    def test_valid(self):
        s = _step()
        assert s.source == "draft"

    def test_invalid_source(self):
        with pytest.raises(ValueError):
            LookaheadStep(text="x", source="unknown", confidence=0.5, tokens_used=5)

    def test_invalid_confidence(self):
        with pytest.raises(ValueError):
            LookaheadStep(text="x", source="draft", confidence=-0.1, tokens_used=5)


# ---------------------------------------------------------------------------
# TestLookaheadBatch
# ---------------------------------------------------------------------------


class TestLookaheadBatch:
    def test_n_steps(self):
        batch = LookaheadBatch(steps=[_step(), _step()], context_at_start="ctx", n_accepted=1)
        assert batch.n_steps == 2

    def test_acceptance_rate(self):
        s1 = _step(accepted=True)
        s2 = _step(accepted=False)
        batch = LookaheadBatch(steps=[s1, s2], context_at_start="ctx", n_accepted=1)
        assert batch.acceptance_rate == 0.5

    def test_accepted_steps_filtered(self):
        s1 = _step(accepted=True)
        s2 = _step(accepted=False)
        s3 = _step(accepted=True)
        batch = LookaheadBatch(steps=[s1, s2, s3], context_at_start="ctx", n_accepted=2)
        assert len(batch.accepted_steps) == 2

    def test_total_tokens(self):
        batch = LookaheadBatch(
            steps=[_step(tokens=10), _step(tokens=15)],
            context_at_start="ctx",
        )
        assert batch.total_tokens == 25


# ---------------------------------------------------------------------------
# TestDefaultBatchVerifier
# ---------------------------------------------------------------------------


class TestDefaultBatchVerifier:
    def test_returns_scores_for_all_steps(self):
        steps = [_step("the quick fox"), _step("the lazy dog")]
        scores = _default_batch_verifier(steps, "the quick brown fox")
        assert len(scores) == 2

    def test_high_overlap_high_score(self):
        steps = [_step("the quick brown fox")]
        scores = _default_batch_verifier(steps, "the quick brown fox")
        assert scores[0] == pytest.approx(1.0)

    def test_no_overlap_zero(self):
        steps = [_step("alpha beta gamma")]
        scores = _default_batch_verifier(steps, "delta epsilon zeta")
        assert scores[0] == 0.0


# ---------------------------------------------------------------------------
# TestLookaheadStats
# ---------------------------------------------------------------------------


class TestLookaheadStats:
    def test_steps_per_cycle_zero(self):
        s = LookaheadStats()
        assert s.steps_per_cycle == 0.0

    def test_acceptance_rate(self):
        s = LookaheadStats(total_draft_steps=10, total_accepted_steps=6)
        assert abs(s.acceptance_rate - 0.6) < 1e-6

    def test_batch_efficiency(self):
        s = LookaheadStats(total_target_batches=5, total_accepted_steps=10)
        assert abs(s.batch_efficiency - 2.0) < 1e-6

    def test_estimated_speedup_with_acceptance(self):
        s = LookaheadStats(total_cycles=5, total_accepted_steps=20)
        assert s.estimated_speedup >= 1.0


# ---------------------------------------------------------------------------
# TestLookaheadReasoningEngine
# ---------------------------------------------------------------------------


class TestLookaheadReasoningEngine:
    def _make_engine(self, k=3, accept_score=0.0, verifier=None):
        cfg = LookaheadConfig(lookahead_k=k, min_acceptance_score=accept_score)
        return LookaheadReasoningEngine(
            config=cfg,
            draft_fn=_draft_fn(),
            batch_verify_fn=verifier or _verifier_all_accept(),
        )

    def test_run_cycle_returns_batch(self):
        engine = self._make_engine()
        batch = engine.run_cycle("initial context")
        assert isinstance(batch, LookaheadBatch)
        assert batch.n_steps == 3

    def test_all_accepted_when_verifier_passes(self):
        engine = self._make_engine(k=3)
        batch = engine.run_cycle("initial context")
        assert batch.n_accepted == 3

    def test_none_accepted_when_verifier_fails(self):
        engine = self._make_engine(k=3, accept_score=0.5, verifier=_verifier_all_reject())
        batch = engine.run_cycle("initial context")
        assert batch.n_accepted == 0

    def test_greedy_prefix_stops_at_first_reject(self):
        # verifier: first accept, second reject, third accept
        def partial_verifier(steps, ctx):
            return [1.0, 0.0, 1.0]
        cfg = LookaheadConfig(lookahead_k=3, min_acceptance_score=0.5, greedy_prefix_accept=True)
        engine = LookaheadReasoningEngine(
            config=cfg,
            draft_fn=_draft_fn(),
            batch_verify_fn=partial_verifier,
        )
        batch = engine.run_cycle("ctx")
        # greedy: accept step0, reject step1, stop -> step2 never accepted
        assert batch.n_accepted == 1

    def test_stats_accumulate_across_cycles(self):
        engine = self._make_engine(k=2)
        engine.run_cycle("ctx one")
        engine.run_cycle("ctx two")
        assert engine.stats.total_cycles == 2
        assert engine.stats.total_draft_steps == 4

    def test_generate_chain_returns_steps(self):
        engine = self._make_engine(k=2)
        steps = engine.generate_chain("start", max_steps=6)
        assert len(steps) >= 1

    def test_reset_clears_state(self):
        engine = self._make_engine()
        engine.run_cycle("ctx")
        engine.reset()
        assert engine.stats.total_cycles == 0
        assert engine.all_accepted_steps == []

    def test_infinite_loop_guard(self):
        # If verifier always rejects, generate_chain should break without looping forever
        engine = self._make_engine(k=2, accept_score=0.5, verifier=_verifier_all_reject())
        steps = engine.generate_chain("ctx", max_steps=10)
        assert len(steps) == 0  # nothing accepted
