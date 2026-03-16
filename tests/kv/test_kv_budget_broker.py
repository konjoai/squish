"""
tests/test_kv_budget_broker.py

Unit tests for KVBudgetBroker — the centralised KV token-budget arbitrator
added in Phase 5C Opt 7 (squish/kv_cache.py).

All tests are pure-Python; no MLX, model files or hardware required.
"""
from __future__ import annotations

import pytest

from squish.kv.kv_cache import KVBudgetBroker

# ---------------------------------------------------------------------------
# Fixture: fresh broker for every test (via singleton reset)
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_broker():
    """Ensure a clean singleton before and after every test."""
    KVBudgetBroker.reset()
    yield
    KVBudgetBroker.reset()


# ---------------------------------------------------------------------------
# Singleton behaviour
# ---------------------------------------------------------------------------

class TestSingleton:
    def test_instance_returns_same_object(self):
        a = KVBudgetBroker.instance()
        b = KVBudgetBroker.instance()
        assert a is b

    def test_reset_creates_fresh_instance(self):
        a = KVBudgetBroker.instance()
        KVBudgetBroker.reset()
        b = KVBudgetBroker.instance()
        assert a is not b

    def test_reset_then_instance_is_clean(self):
        KVBudgetBroker.instance().register("sys_a", 1024)
        KVBudgetBroker.reset()
        assert KVBudgetBroker.instance().registered_systems == []


# ---------------------------------------------------------------------------
# set_total
# ---------------------------------------------------------------------------

class TestSetTotal:
    def test_default_total_is_zero(self):
        assert KVBudgetBroker.instance().total_tokens == 0

    def test_set_total_stores_value(self):
        KVBudgetBroker.instance().set_total(8_192)
        assert KVBudgetBroker.instance().total_tokens == 8_192

    def test_set_total_zero_is_allowed(self):
        KVBudgetBroker.instance().set_total(0)
        assert KVBudgetBroker.instance().total_tokens == 0

    def test_set_total_negative_raises(self):
        with pytest.raises(ValueError, match="total_tokens must be >= 0"):
            KVBudgetBroker.instance().set_total(-1)


# ---------------------------------------------------------------------------
# register / allocated
# ---------------------------------------------------------------------------

class TestRegister:
    def test_register_returns_requested_when_unconstrained(self):
        broker = KVBudgetBroker.instance()
        alloc = broker.register("snap_kv", 4_096)
        assert alloc == 4_096

    def test_allocated_matches_register_return(self):
        broker = KVBudgetBroker.instance()
        broker.register("sys_a", 2_048)
        assert broker.allocated("sys_a") == 2_048

    def test_allocated_zero_for_unknown_system(self):
        assert KVBudgetBroker.instance().allocated("nonexistent") == 0

    def test_register_zero_requested_raises(self):
        with pytest.raises(ValueError, match="requested must be > 0"):
            KVBudgetBroker.instance().register("bad", 0)

    def test_register_negative_requested_raises(self):
        with pytest.raises(ValueError, match="requested must be > 0"):
            KVBudgetBroker.instance().register("bad", -10)

    def test_re_register_updates_allocation(self):
        broker = KVBudgetBroker.instance()
        broker.register("sys_a", 1_000)
        broker.register("sys_a", 2_000)   # update
        assert broker.allocated("sys_a") == 2_000


# ---------------------------------------------------------------------------
# Unconstrained (total == 0) allocation
# ---------------------------------------------------------------------------

class TestUnconstrainedAllocation:
    def test_multiple_systems_get_full_requested(self):
        broker = KVBudgetBroker.instance()
        broker.set_total(0)    # unconstrained
        broker.register("snap",     4_096)
        broker.register("squeeze",  2_048)
        broker.register("small",    1_024)
        assert broker.allocated("snap")     == 4_096
        assert broker.allocated("squeeze")  == 2_048
        assert broker.allocated("small")    == 1_024

    def test_total_fits_exactly_gives_full_request(self):
        broker = KVBudgetBroker.instance()
        broker.set_total(3_072)
        broker.register("a", 1_024)
        broker.register("b", 2_048)
        assert broker.allocated("a") == 1_024
        assert broker.allocated("b") == 2_048


# ---------------------------------------------------------------------------
# Constrained (scaled) allocation
# ---------------------------------------------------------------------------

class TestConstrainedAllocation:
    def test_two_equal_systems_split_evenly(self):
        broker = KVBudgetBroker.instance()
        broker.set_total(2_000)
        broker.register("a", 2_000)
        broker.register("b", 2_000)
        # Each should get ~1 000 (50%)
        total_alloc = broker.allocated("a") + broker.allocated("b")
        assert total_alloc <= 2_000
        assert broker.allocated("a") >= 1
        assert broker.allocated("b") >= 1

    def test_proportional_allocation(self):
        broker = KVBudgetBroker.instance()
        broker.set_total(3_000)
        broker.register("small",  1_000)   # 25%
        broker.register("large",  3_000)   # 75%
        total = broker.allocated("small") + broker.allocated("large")
        assert total <= 3_000
        assert broker.allocated("large") > broker.allocated("small")

    def test_each_system_gets_at_least_one_token(self):
        broker = KVBudgetBroker.instance()
        broker.set_total(1)   # extremely tight budget
        broker.register("sys_a", 1_000)
        broker.register("sys_b", 1_000)
        assert broker.allocated("sys_a") >= 1
        assert broker.allocated("sys_b") >= 1

    def test_three_systems_constrained(self):
        broker = KVBudgetBroker.instance()
        broker.set_total(6_000)
        broker.register("a", 4_000)
        broker.register("b", 4_000)
        broker.register("c", 4_000)
        total = sum(broker.allocated(n) for n in ("a", "b", "c"))
        assert total <= 6_000

    def test_set_total_after_register_recomputes(self):
        broker = KVBudgetBroker.instance()
        broker.register("a", 4_000)
        broker.register("b", 4_000)
        # With total=0 (unconstrained) both get 4000
        assert broker.allocated("a") == 4_000
        # Now constrain to 4000
        broker.set_total(4_000)
        assert broker.allocated("a") + broker.allocated("b") <= 4_000


# ---------------------------------------------------------------------------
# unregister
# ---------------------------------------------------------------------------

class TestUnregister:
    def test_unregister_removes_from_registered_systems(self):
        broker = KVBudgetBroker.instance()
        broker.register("sys_a", 2_048)
        broker.unregister("sys_a")
        assert "sys_a" not in broker.registered_systems

    def test_unregister_gives_zero_allocated(self):
        broker = KVBudgetBroker.instance()
        broker.register("sys_a", 2_048)
        broker.unregister("sys_a")
        assert broker.allocated("sys_a") == 0

    def test_unregister_nonexistent_is_noop(self):
        # Must not raise
        KVBudgetBroker.instance().unregister("ghost")

    def test_unregister_allows_remaining_system_full_budget(self):
        broker = KVBudgetBroker.instance()
        broker.set_total(4_096)
        broker.register("a", 4_096)
        broker.register("b", 4_096)
        # Both constrained
        assert broker.allocated("a") < 4_096
        # Remove b — a should now get its full request
        broker.unregister("b")
        assert broker.allocated("a") == 4_096


# ---------------------------------------------------------------------------
# registered_systems / summary
# ---------------------------------------------------------------------------

class TestIntrospection:
    def test_registered_systems_empty_by_default(self):
        assert KVBudgetBroker.instance().registered_systems == []

    def test_registered_systems_contains_all_registered(self):
        broker = KVBudgetBroker.instance()
        broker.register("x", 100)
        broker.register("y", 200)
        assert set(broker.registered_systems) == {"x", "y"}

    def test_summary_empty_when_no_registrations(self):
        assert KVBudgetBroker.instance().summary() == {}

    def test_summary_returns_copy(self):
        broker = KVBudgetBroker.instance()
        broker.register("a", 512)
        s = broker.summary()
        s["a"] = 0        # mutate copy
        assert broker.allocated("a") == 512  # original unchanged

    def test_summary_keys_match_registered_systems(self):
        broker = KVBudgetBroker.instance()
        broker.register("p", 100)
        broker.register("q", 200)
        assert set(broker.summary().keys()) == set(broker.registered_systems)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_no_registrations_summary_empty(self):
        broker = KVBudgetBroker.instance()
        broker.set_total(1_024)
        assert broker.summary() == {}

    def test_single_system_gets_full_budget_when_constrained(self):
        broker = KVBudgetBroker.instance()
        broker.set_total(1_000)
        broker.register("only_system", 5_000)
        # Single system — it gets the whole budget as-is (scales to max(1, int(5000 * 0.2)) = 1000)
        assert broker.allocated("only_system") == 1_000

    def test_seven_simultaneous_systems(self):
        """Regression: the plan names 7 budget-consuming modules."""
        systems = [
            "squeeze_kv", "small_kv", "yoco_kv", "diffkv",
            "kvtuner", "kvsharer", "adaptive_budget",
        ]
        broker = KVBudgetBroker.instance()
        broker.set_total(32_768)
        for i, name in enumerate(systems, start=1):
            broker.register(name, i * 1_024)
        total = sum(broker.allocated(n) for n in systems)
        assert total <= 32_768
        for name in systems:
            assert broker.allocated(name) >= 1
