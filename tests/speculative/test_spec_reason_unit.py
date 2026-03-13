"""Unit tests for squish.speculative.spec_reason (SpecReason step-level speculation)."""

import pytest

from squish.speculative.spec_reason import (
    ReasoningStep,
    SpecReasonConfig,
    SpecReasonOrchestrator,
    SpecReasonStats,
    StepVerdict,
    _cosine_sim_scorer,
)


def _step(text="test step", source="draft", confidence=0.8, tokens=20, idx=0):
    return ReasoningStep(
        text=text, source=source, confidence=confidence, tokens_used=tokens, step_idx=idx
    )


def _draft_fn(confidence=0.9, tokens=15):
    def fn(context: str) -> ReasoningStep:
        return ReasoningStep(
            text=f"step based on: {context[:10]}",
            source="draft",
            confidence=confidence,
            tokens_used=tokens,
        )
    return fn


def _target_fn():
    def fn(context: str) -> ReasoningStep:
        return ReasoningStep(
            text=f"target step for: {context[:10]}",
            source="target",
            confidence=0.99,
            tokens_used=30,
        )
    return fn


# ---------------------------------------------------------------------------
# TestReasoningStep
# ---------------------------------------------------------------------------


class TestReasoningStep:
    def test_valid_construction(self):
        step = _step()
        assert step.text == "test step"
        assert step.source == "draft"

    def test_invalid_source(self):
        with pytest.raises(ValueError, match="source"):
            ReasoningStep(text="x", source="invalid", confidence=0.5, tokens_used=10)

    def test_invalid_confidence_high(self):
        with pytest.raises(ValueError, match="confidence"):
            ReasoningStep(text="x", source="draft", confidence=1.5, tokens_used=10)

    def test_invalid_confidence_low(self):
        with pytest.raises(ValueError, match="confidence"):
            ReasoningStep(text="x", source="target", confidence=-0.1, tokens_used=10)

    def test_invalid_tokens(self):
        with pytest.raises(ValueError, match="tokens_used"):
            ReasoningStep(text="x", source="draft", confidence=0.5, tokens_used=-1)


# ---------------------------------------------------------------------------
# TestSpecReasonConfig
# ---------------------------------------------------------------------------


class TestSpecReasonConfig:
    def test_defaults(self):
        cfg = SpecReasonConfig()
        assert cfg.min_acceptance_score == 0.75
        assert cfg.max_step_tokens == 256
        assert cfg.domain == "general"

    def test_invalid_acceptance_score(self):
        with pytest.raises(ValueError):
            SpecReasonConfig(min_acceptance_score=1.5)

    def test_invalid_max_draft_steps(self):
        with pytest.raises(ValueError):
            SpecReasonConfig(max_draft_steps=0)

    def test_invalid_confidence_gate(self):
        with pytest.raises(ValueError):
            SpecReasonConfig(confidence_gate=-0.1)


# ---------------------------------------------------------------------------
# TestCosineSimScorer
# ---------------------------------------------------------------------------


class TestCosineSimScorer:
    def test_identical_returns_high(self):
        step = _step(text="the quick brown fox")
        score = _cosine_sim_scorer(step, "the quick brown fox")
        assert score == 1.0

    def test_disjoint_returns_low(self):
        step = _step(text="alpha beta gamma")
        score = _cosine_sim_scorer(step, "delta epsilon zeta")
        assert score == 0.0

    def test_partial_overlap(self):
        step = _step(text="alpha beta")
        score = _cosine_sim_scorer(step, "alpha gamma")
        assert 0.0 < score < 1.0

    def test_empty_context(self):
        step = _step(text="")
        score = _cosine_sim_scorer(step, "")
        assert score == 1.0


# ---------------------------------------------------------------------------
# TestSpecReasonOrchestrator
# ---------------------------------------------------------------------------


class TestSpecReasonOrchestrator:
    def _make_orch(self, acceptance_score=0.0, confidence_gate=0.0, regen=True):
        cfg = SpecReasonConfig(
            min_acceptance_score=acceptance_score,
            confidence_gate=confidence_gate,
            target_regenerates_on_reject=regen,
        )
        orch = SpecReasonOrchestrator(
            config=cfg,
            draft_fn=_draft_fn(confidence=0.8),
            target_fn=_target_fn(),
        )
        return orch

    def test_generate_step_returns_step_and_verdict(self):
        orch = self._make_orch(acceptance_score=0.0)
        step, verdict = orch.generate_step("initial context with enough words")
        assert isinstance(step, ReasoningStep)
        assert isinstance(verdict, StepVerdict)

    def test_low_threshold_accepts_draft(self):
        orch = self._make_orch(acceptance_score=0.0)
        step, verdict = orch.generate_step("shared words in context")
        assert verdict == StepVerdict.ACCEPT

    def test_high_threshold_rejects_draft(self):
        orch = self._make_orch(acceptance_score=1.0, regen=False)
        step, verdict = orch.generate_step("completely different context xyz")
        assert verdict == StepVerdict.REJECT

    def test_target_regenerates_on_reject(self):
        orch = self._make_orch(acceptance_score=1.0, regen=True)
        step, verdict = orch.generate_step("zzz aaa bbb")
        assert orch.stats.target_steps >= 1

    def test_confidence_gate_skips_verification(self):
        cfg = SpecReasonConfig(confidence_gate=0.5)
        orch = SpecReasonOrchestrator(
            config=cfg,
            draft_fn=_draft_fn(confidence=0.95),
            target_fn=_target_fn(),
        )
        _, verdict = orch.generate_step("any context")
        assert verdict == StepVerdict.ACCEPT
        assert orch.stats.conf_gate_skipped == 1

    def test_stats_draft_steps_increment(self):
        orch = self._make_orch(acceptance_score=0.0)
        for _ in range(3):
            orch.generate_step("context words shared")
        assert orch.stats.draft_steps == 3

    def test_generate_chain_returns_steps(self):
        orch = self._make_orch(acceptance_score=0.0)
        steps = orch.generate_chain("initial reasoning context", max_steps=5)
        assert len(steps) >= 1

    def test_reset_clears_stats(self):
        orch = self._make_orch(acceptance_score=0.0)
        orch.generate_step("context")
        orch.reset()
        assert orch.stats.total_steps == 0
        assert orch.accepted_steps == []


# ---------------------------------------------------------------------------
# TestSpecReasonStats
# ---------------------------------------------------------------------------


class TestSpecReasonStats:
    def test_acceptance_rate_no_drafts(self):
        s = SpecReasonStats()
        assert s.draft_acceptance_rate == 0.0

    def test_acceptance_rate(self):
        s = SpecReasonStats(draft_steps=10, accepted_draft_steps=7)
        assert abs(s.draft_acceptance_rate - 0.7) < 1e-6

    def test_estimated_speedup_gt1_with_acceptance(self):
        s = SpecReasonStats(draft_steps=10, accepted_draft_steps=8, total_draft_tokens=100)
        assert s.estimated_speedup > 1.0

    def test_target_tokens_saved(self):
        s = SpecReasonStats(
            draft_steps=5, accepted_draft_steps=3, total_draft_tokens=75
        )
        assert s.target_tokens_saved == 75
