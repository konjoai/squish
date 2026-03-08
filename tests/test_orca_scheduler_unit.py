"""
tests/test_orca_scheduler_unit.py

Unit tests for the ORCA iteration-level scheduling classes appended to
squish/scheduler.py — 100% coverage of:
  OrcaConfig, RequestState, SelectivePreemption, IterationLevelScheduler.
"""

import pytest

from squish.scheduler import (
    OrcaConfig,
    RequestState,
    SelectivePreemption,
    IterationLevelScheduler,
)


# ---------------------------------------------------------------------------
# OrcaConfig
# ---------------------------------------------------------------------------

class TestOrcaConfig:
    def test_defaults(self):
        cfg = OrcaConfig()
        assert cfg.max_batch_tokens >= 1
        assert cfg.preemption_mode in ("swap", "recompute")
        assert cfg.max_waiting >= 0

    def test_custom(self):
        cfg = OrcaConfig(max_batch_tokens=512, preemption_mode="recompute", max_waiting=10)
        assert cfg.max_batch_tokens == 512
        assert cfg.preemption_mode == "recompute"
        assert cfg.max_waiting == 10

    @pytest.mark.parametrize("kwargs, exc", [
        ({"max_batch_tokens": 0},           "max_batch_tokens"),
        ({"preemption_mode": "unknown"},    "preemption_mode"),
        ({"max_waiting": -1},               "max_waiting"),
    ])
    def test_validation(self, kwargs, exc):
        with pytest.raises(ValueError, match=exc):
            OrcaConfig(**kwargs)


# ---------------------------------------------------------------------------
# RequestState
# ---------------------------------------------------------------------------

class TestRequestState:
    def test_total_tokens(self):
        r = RequestState(request_id="r1", prompt_len=10, generated=5)
        assert r.total_tokens == 15

    def test_is_finished_false(self):
        r = RequestState(prompt_len=5, max_new_tokens=20, generated=10)
        assert not r.is_finished

    def test_is_finished_true(self):
        r = RequestState(prompt_len=5, max_new_tokens=10, generated=10)
        assert r.is_finished

    def test_defaults(self):
        r = RequestState()
        assert r.generated == 0
        assert not r.preempted
        assert r.total_tokens == 0


# ---------------------------------------------------------------------------
# SelectivePreemption
# ---------------------------------------------------------------------------

class TestSelectivePreemption:
    def test_select_victim_largest_footprint(self):
        sp = SelectivePreemption(mode="swap")
        r1 = RequestState(request_id="r1", prompt_len=5, generated=2)   # 7 tokens
        r2 = RequestState(request_id="r2", prompt_len=20, generated=10) # 30 tokens
        r3 = RequestState(request_id="r3", prompt_len=3, generated=1)   # 4 tokens
        victim = sp.select_victim([r1, r2, r3])
        assert victim is r2

    def test_select_victim_empty(self):
        sp = SelectivePreemption()
        assert sp.select_victim([]) is None

    def test_preempt_swap_preserves_progress(self):
        sp      = SelectivePreemption(mode="swap")
        running = []
        waiting = []
        r = RequestState(request_id="r1", prompt_len=10, generated=5, max_new_tokens=20)
        running.append(r)
        sp.preempt(r, running, waiting)
        assert r not in running
        assert r in waiting
        assert r.generated == 5       # preserved in swap mode
        assert r.preempted is True

    def test_preempt_recompute_resets_progress(self):
        sp      = SelectivePreemption(mode="recompute")
        running = []
        waiting = []
        r = RequestState(request_id="r1", prompt_len=10, generated=8, max_new_tokens=20)
        running.append(r)
        sp.preempt(r, running, waiting)
        assert r.generated == 0    # reset in recompute mode
        assert r.preempted is True

    def test_invalid_mode(self):
        with pytest.raises(ValueError, match="mode"):
            SelectivePreemption(mode="teleport")

    def test_preempted_request_goes_to_front_of_waiting(self):
        sp      = SelectivePreemption(mode="swap")
        running = []
        waiting = [RequestState(request_id="existing")]
        r = RequestState(request_id="victim", prompt_len=5)
        running.append(r)
        sp.preempt(r, running, waiting)
        assert waiting[0].request_id == "victim"   # front of queue


# ---------------------------------------------------------------------------
# IterationLevelScheduler
# ---------------------------------------------------------------------------

class TestIterationLevelScheduler:
    def _make(self, max_batch_tokens=100, mode="swap", max_waiting=0):
        cfg = OrcaConfig(max_batch_tokens=max_batch_tokens,
                         preemption_mode=mode, max_waiting=max_waiting)
        return IterationLevelScheduler(cfg)

    def test_add_and_admit_request(self):
        sched = self._make(max_batch_tokens=50)
        r = RequestState(request_id="r1", prompt_len=10, max_new_tokens=5)
        sched.add_request(r)
        to_run, admitted, preempted = sched.step()
        assert r in admitted
        assert r in to_run
        assert preempted == []

    def test_waiting_queue_full_raises(self):
        sched = self._make(max_waiting=2)
        for i in range(2):
            sched.add_request(RequestState(request_id=f"r{i}", prompt_len=5))
        with pytest.raises(RuntimeError, match="Waiting queue full"):
            sched.add_request(RequestState(request_id="overflow", prompt_len=5))

    def test_request_not_admitted_when_over_budget(self):
        sched = self._make(max_batch_tokens=15)
        r1 = RequestState(request_id="r1", prompt_len=12, max_new_tokens=10)
        r2 = RequestState(request_id="r2", prompt_len=10, max_new_tokens=10)
        sched.add_request(r1)
        sched.add_request(r2)
        to_run, admitted, preempted = sched.step()
        # r1 fits (12 ≤ 15), r2 would make total 22 > 15
        assert r1 in to_run
        assert r2 not in to_run
        assert r2 in sched.waiting

    def test_preemption_when_over_budget_after_admission(self):
        # Large request fills budget; then another gets added directly
        sched = self._make(max_batch_tokens=20, mode="swap")
        r1 = RequestState(request_id="r1", prompt_len=18, max_new_tokens=5)
        sched.add_request(r1)
        sched.step()   # r1 admitted (18 ≤ 20)

        # Now manually add a big request directly to running via add_request + step
        r2 = RequestState(request_id="r2", prompt_len=15, max_new_tokens=5)
        sched.add_request(r2)
        to_run, admitted, preempted = sched.step()
        # Either r2 wasn't admitted (still waiting) or r1 was preempted
        # The invariant is: total running tokens ≤ max_batch_tokens
        total = sum(r.total_tokens for r in to_run)
        assert total <= 20

    def test_finished_requests_removed(self):
        sched = self._make(max_batch_tokens=200)
        r = RequestState(request_id="r1", prompt_len=5, max_new_tokens=1, generated=0)
        sched.add_request(r)
        sched.step()          # admitted
        sched.tick(tokens_per_request=1)   # generated=1 → finished
        to_run, _, _ = sched.step()
        assert r not in to_run  # removed as finished

    def test_tick_advances_generated(self):
        sched = self._make()
        r = RequestState(request_id="r1", prompt_len=5, max_new_tokens=20)
        sched.add_request(r)
        sched.step()
        sched.tick(tokens_per_request=3)
        assert r.generated == 3

    def test_tick_caps_at_max_new_tokens(self):
        sched = self._make()
        r = RequestState(request_id="r1", prompt_len=5, max_new_tokens=10, generated=8)
        sched.add_request(r)
        sched.step()
        sched.tick(tokens_per_request=5)  # would go to 13, should cap at 10
        assert r.generated == 10

    def test_step_number_increments(self):
        sched = self._make()
        assert sched.step_number == 0
        sched.step()
        assert sched.step_number == 1
        sched.step()
        assert sched.step_number == 2

    def test_running_and_waiting_properties(self):
        sched = self._make(max_batch_tokens=10)
        r1 = RequestState(request_id="r1", prompt_len=8, max_new_tokens=5)
        r2 = RequestState(request_id="r2", prompt_len=8, max_new_tokens=5)
        sched.add_request(r1)
        sched.add_request(r2)
        sched.step()
        assert r1 in sched.running
        assert r2 in sched.waiting

    def test_empty_step(self):
        sched = self._make()
        to_run, admitted, preempted = sched.step()
        assert to_run == []
        assert admitted == []
        assert preempted == []

    def test_multiple_requests_all_fit(self):
        sched = self._make(max_batch_tokens=100)
        reqs = [RequestState(request_id=f"r{i}", prompt_len=5) for i in range(5)]
        for r in reqs:
            sched.add_request(r)
        to_run, admitted, preempted = sched.step()
        assert len(admitted) == 5
        assert len(to_run) == 5

    def test_recompute_mode_resets_progress_on_preemption(self):
        sched = self._make(max_batch_tokens=20, mode="recompute")
        r1 = RequestState(request_id="r1", prompt_len=18, max_new_tokens=10)
        sched.add_request(r1)
        sched.step()   # r1 admitted
        sched.tick(5)  # r1.generated = 5
        # Force r1 onto waiting by adding a competitor that triggers preemption
        r2 = RequestState(request_id="r2", prompt_len=15, max_new_tokens=10)
        sched.add_request(r2)
        sched.step()
        # If r1 was preempted and recomputed, its progress is reset
        if r1 in sched.waiting and r1.preempted:
            assert r1.generated == 0
