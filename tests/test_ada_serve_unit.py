"""Unit tests for squish.ada_serve (AdaServe SLO-customized speculative decoding)."""

import time

import pytest

from squish.ada_serve import (
    BUILT_IN_SLOS,
    AdaServeConfig,
    AdaServeRequest,
    AdaServeScheduler,
    AdaServeStats,
    SLOTarget,
    select_gamma,
)


def _slo(task="general", ttft=300.0, tpot=60.0, total=10000.0, priority=5):
    return SLOTarget(
        task_type=task,
        time_to_first_token_ms=ttft,
        time_per_output_token_ms=tpot,
        total_latency_ms=total,
        priority=priority,
    )


def _cfg(min_gamma=1, max_gamma=8, base_gamma=4, **kw):
    return AdaServeConfig(min_gamma=min_gamma, max_gamma=max_gamma, base_gamma=base_gamma, **kw)


# ---------------------------------------------------------------------------
# TestSLOTarget
# ---------------------------------------------------------------------------


class TestSLOTarget:
    def test_is_tight_for_short_ttft(self):
        slo = _slo(ttft=100.0)
        assert slo.is_tight is True

    def test_is_not_tight_for_long_ttft(self):
        slo = _slo(ttft=500.0)
        assert slo.is_tight is False

    def test_invalid_ttft(self):
        with pytest.raises(ValueError):
            SLOTarget(task_type="x", time_to_first_token_ms=0.0)

    def test_invalid_priority(self):
        with pytest.raises(ValueError):
            SLOTarget(task_type="x", time_to_first_token_ms=100, time_per_output_token_ms=10, priority=11)

    def test_built_in_git_commit_is_tight(self):
        slo = BUILT_IN_SLOS["git_commit"]
        assert slo.is_tight is True

    def test_built_in_devops_plan_not_tight(self):
        slo = BUILT_IN_SLOS["devops_plan"]
        assert slo.is_tight is False


# ---------------------------------------------------------------------------
# TestAdaServeConfig
# ---------------------------------------------------------------------------


class TestAdaServeConfig:
    def test_defaults(self):
        cfg = AdaServeConfig()
        assert cfg.min_gamma == 1
        assert cfg.max_gamma == 8
        assert cfg.base_gamma == 4

    def test_invalid_min_gamma(self):
        with pytest.raises(ValueError):
            AdaServeConfig(min_gamma=0)

    def test_invalid_max_lt_min(self):
        with pytest.raises(ValueError):
            AdaServeConfig(min_gamma=5, max_gamma=3)

    def test_invalid_tight_scale(self):
        with pytest.raises(ValueError):
            AdaServeConfig(tight_slo_gamma_scale=0.0)

    def test_invalid_relaxed_scale(self):
        with pytest.raises(ValueError):
            AdaServeConfig(relaxed_slo_gamma_scale=0.5)

    def test_invalid_goodput_weight(self):
        with pytest.raises(ValueError):
            AdaServeConfig(goodput_weight=1.5)


# ---------------------------------------------------------------------------
# TestSelectGamma
# ---------------------------------------------------------------------------


class TestSelectGamma:
    def test_tight_slo_returns_low_gamma(self):
        cfg = _cfg(base_gamma=4, tight_slo_gamma_scale=0.5)
        slo = _slo(ttft=100.0)  # tight
        gamma = select_gamma(slo, cfg)
        assert gamma <= cfg.base_gamma

    def test_relaxed_slo_returns_high_gamma(self):
        cfg = _cfg(base_gamma=4, relaxed_slo_gamma_scale=2.0)
        slo = _slo(ttft=1000.0, total=100000.0)  # relaxed, no budget pressure
        gamma = select_gamma(slo, cfg, elapsed_ms=0)
        assert gamma >= cfg.base_gamma

    def test_gamma_within_bounds(self):
        cfg = _cfg(min_gamma=2, max_gamma=6, base_gamma=4)
        for ttft in [50.0, 200.0, 1000.0]:
            slo = _slo(ttft=ttft)
            g = select_gamma(slo, cfg)
            assert cfg.min_gamma <= g <= cfg.max_gamma

    def test_urgent_request_gets_low_gamma(self):
        cfg = _cfg(base_gamma=4)
        slo = _slo(total=1000.0)
        # Almost at budget limit
        gamma = select_gamma(slo, cfg, elapsed_ms=900.0, tokens_generated=10)
        assert gamma <= cfg.base_gamma


# ---------------------------------------------------------------------------
# TestAdaServeRequest
# ---------------------------------------------------------------------------


class TestAdaServeRequest:
    def test_initial_state(self):
        slo = _slo()
        req = AdaServeRequest(request_id="r1", slo=slo)
        assert req.tokens_generated == 0
        assert req.completed is False

    def test_elapsed_ms_positive(self):
        slo = _slo()
        req = AdaServeRequest(request_id="r1", slo=slo)
        time.sleep(0.001)
        assert req.elapsed_ms > 0

    def test_is_slo_at_risk_no_total_budget(self):
        slo = _slo(total=0.0)
        req = AdaServeRequest(request_id="r1", slo=slo)
        assert req.is_slo_at_risk is False


# ---------------------------------------------------------------------------
# TestAdaServeStats
# ---------------------------------------------------------------------------


class TestAdaServeStats:
    def test_slo_violation_rate_initial(self):
        s = AdaServeStats()
        assert s.slo_violation_rate == 0.0

    def test_goodput_rate_no_tokens(self):
        s = AdaServeStats()
        assert s.goodput_rate == 1.0

    def test_record_request_slo_met(self):
        s = AdaServeStats()
        s.record_request(gamma_used=4, tokens_generated=100, slo_met=True)
        assert s.total_requests == 1
        assert s.total_slo_violations == 0
        assert s.total_goodput_tokens == 100

    def test_record_request_slo_violated(self):
        s = AdaServeStats()
        s.record_request(gamma_used=4, tokens_generated=100, slo_met=False)
        assert s.total_slo_violations == 1
        assert s.total_goodput_tokens == 0

    def test_mean_gamma(self):
        s = AdaServeStats()
        s.record_request(gamma_used=2, tokens_generated=50, slo_met=True)
        s.record_request(gamma_used=6, tokens_generated=50, slo_met=True)
        assert s.mean_gamma == 4.0

    def test_goodput_improvement_estimate(self):
        s = AdaServeStats()
        for g in [2, 3, 4, 6]:
            s.record_request(gamma_used=g, tokens_generated=50, slo_met=True)
        assert s.estimated_goodput_improvement_vs_fixed >= 1.0


# ---------------------------------------------------------------------------
# TestAdaServeScheduler
# ---------------------------------------------------------------------------


class TestAdaServeScheduler:
    def test_initial_n_active_zero(self):
        scheduler = AdaServeScheduler(_cfg())
        assert scheduler.n_active == 0

    def test_enqueue_increases_active(self):
        scheduler = AdaServeScheduler(_cfg())
        scheduler.enqueue(AdaServeRequest(request_id="r1", slo=_slo()))
        assert scheduler.n_active == 1

    def test_get_gamma_returns_valid(self):
        scheduler = AdaServeScheduler(_cfg(min_gamma=1, max_gamma=8))
        scheduler.enqueue(AdaServeRequest(request_id="r1", slo=_slo()))
        g = scheduler.get_gamma("r1")
        assert 1 <= g <= 8

    def test_complete_removes_from_active(self):
        scheduler = AdaServeScheduler(_cfg())
        scheduler.enqueue(AdaServeRequest(request_id="r2", slo=_slo()))
        scheduler.complete("r2", tokens_generated=50, slo_met=True)
        assert scheduler.n_active == 0

    def test_complete_updates_stats(self):
        scheduler = AdaServeScheduler(_cfg())
        scheduler.enqueue(AdaServeRequest(request_id="r3", slo=_slo()))
        scheduler.complete("r3", tokens_generated=100, slo_met=False)
        assert scheduler.stats.total_slo_violations == 1

    def test_register_slo(self):
        scheduler = AdaServeScheduler(_cfg())
        custom_slo = _slo(task="custom", ttft=50.0)
        scheduler.register_slo("custom", custom_slo)
        assert "custom" in scheduler._slo_registry

    def test_get_gamma_unknown_request_returns_base(self):
        cfg = _cfg(base_gamma=4)
        scheduler = AdaServeScheduler(cfg)
        assert scheduler.get_gamma("nonexistent") == 4
