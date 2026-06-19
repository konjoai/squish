"""Coverage for the pure-Python sections of ``squish.serving.scheduler`` — the
ORCA iteration-level scheduler (OrcaConfig / RequestState / SelectivePreemption /
IterationLevelScheduler), the cached ``_Request.prefix_key``, and the
``BatchScheduler._import_mx`` platform/ImportError guards. No MLX — host-agnostic.
"""

from __future__ import annotations

import sys

import pytest

from squish.serving import scheduler as sch
from squish.serving.scheduler import (
    BatchScheduler,
    IterationLevelScheduler,
    OrcaConfig,
    RequestState,
    SelectivePreemption,
    _Request,
)


# ── _Request.prefix_key caching ──────────────────────────────────────────────


def test_request_prefix_key_is_cached():
    req = _Request(
        request_id="r",
        input_ids=[1, 2, 3],
        max_tokens=8,
        temperature=1.0,
        top_p=1.0,
        stop_ids=[],
        seed=None,
    )
    first = req.prefix_key()
    assert req._prefix_key == first
    assert req.prefix_key() == first  # second call hits the cached branch


# ── BatchScheduler._import_mx ────────────────────────────────────────────────


def test_import_mx_returns_none_off_darwin(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    assert BatchScheduler._import_mx() is None


def test_import_mx_returns_none_on_import_error(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setitem(sys.modules, "mlx", None)
    monkeypatch.setitem(sys.modules, "mlx.core", None)
    assert BatchScheduler._import_mx() is None


# ── OrcaConfig ───────────────────────────────────────────────────────────────


def test_orca_config_defaults():
    c = OrcaConfig()
    assert c.max_batch_tokens == 2048 and c.preemption_mode == "swap"


@pytest.mark.parametrize(
    "kw,msg",
    [
        ({"max_batch_tokens": 0}, "max_batch_tokens must be"),
        ({"preemption_mode": "nope"}, "preemption_mode must be"),
        ({"max_waiting": -1}, "max_waiting must be"),
    ],
)
def test_orca_config_validation(kw, msg):
    with pytest.raises(ValueError, match=msg):
        OrcaConfig(**kw)


# ── RequestState ─────────────────────────────────────────────────────────────


def test_request_state_properties():
    r = RequestState(request_id="a", prompt_len=10, max_new_tokens=4, generated=4)
    assert r.total_tokens == 14 and r.is_finished is True
    assert RequestState(prompt_len=1, max_new_tokens=8).is_finished is False


# ── SelectivePreemption ──────────────────────────────────────────────────────


def test_preemption_rejects_bad_mode():
    with pytest.raises(ValueError, match="mode must be"):
        SelectivePreemption("bogus")


def test_select_victim_empty_returns_none():
    assert SelectivePreemption().select_victim([]) is None


def test_select_victim_picks_largest():
    small = RequestState(request_id="s", prompt_len=2, max_new_tokens=99)
    big = RequestState(request_id="b", prompt_len=50, max_new_tokens=99)
    assert SelectivePreemption().select_victim([small, big]) is big


def test_preempt_swap_preserves_progress():
    p = SelectivePreemption("swap")
    v = RequestState(request_id="v", prompt_len=4, max_new_tokens=99, generated=7)
    running, waiting = [v], []
    p.preempt(v, running, waiting)
    assert running == [] and waiting == [v]
    assert v.preempted is True and v.generated == 7  # swap keeps progress


def test_preempt_recompute_resets_progress():
    p = SelectivePreemption("recompute")
    v = RequestState(request_id="v", prompt_len=4, max_new_tokens=99, generated=7)
    running, waiting = [v], []
    p.preempt(v, running, waiting)
    assert v.generated == 0 and waiting[0] is v  # recompute starts over


# ── IterationLevelScheduler ──────────────────────────────────────────────────


def test_scheduler_add_request_and_queue_full():
    s = IterationLevelScheduler(OrcaConfig(max_waiting=1))
    s.add_request(RequestState(request_id="a", prompt_len=1, max_new_tokens=8))
    assert len(s.waiting) == 1
    with pytest.raises(RuntimeError, match="Waiting queue full"):
        s.add_request(RequestState(request_id="b", prompt_len=1, max_new_tokens=8))


def test_step_admits_fitting_and_skips_too_large():
    s = IterationLevelScheduler(OrcaConfig(max_batch_tokens=10))
    s.add_request(RequestState(request_id="base", prompt_len=8, max_new_tokens=99))
    s.step()  # admit base → budget 8
    s.add_request(RequestState(request_id="big", prompt_len=5, max_new_tokens=99))  # 8+5>10
    s.add_request(RequestState(request_id="small", prompt_len=1, max_new_tokens=99))  # 8+1<=10
    to_run, admitted, preempted = s.step()
    ids = {r.request_id for r in admitted}
    assert ids == {"small"}  # big skipped (else branch), small admitted
    assert any(r.request_id == "big" for r in s.waiting)
    assert preempted == [] and s.step_number == 2


def test_step_preempts_when_over_budget():
    s = IterationLevelScheduler(OrcaConfig(max_batch_tokens=10, preemption_mode="swap"))
    s.add_request(RequestState(request_id="r", prompt_len=8, max_new_tokens=100))
    s.step()  # admit → budget 8
    s.tick(5)  # generated 5 → total 13 > 10
    _to_run, _admitted, preempted = s.step()
    assert [r.request_id for r in preempted] == ["r"]
    assert any(r.request_id == "r" for r in s.waiting)


def test_step_removes_finished_requests():
    s = IterationLevelScheduler(OrcaConfig(max_batch_tokens=100))
    s.add_request(RequestState(request_id="done", prompt_len=1, max_new_tokens=1))
    s.step()  # admit
    s.tick(1)  # generated 1 → finished
    to_run, _admitted, _preempted = s.step()
    assert to_run == [] and s.running == []


def test_tick_clamps_to_max_new_tokens():
    s = IterationLevelScheduler(OrcaConfig(max_batch_tokens=100))
    r = RequestState(request_id="r", prompt_len=1, max_new_tokens=3)
    s.add_request(r)
    s.step()
    s.tick(10)  # would overshoot → clamped
    assert r.generated == 3
    assert list(s.running) and list(s.waiting) == []
