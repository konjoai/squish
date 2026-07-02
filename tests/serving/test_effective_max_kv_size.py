"""
tests/serving/test_effective_max_kv_size.py

Kill-test evidence for Phase 3 of the memory-governor eviction sprint:
proves ``squish.server._effective_max_kv_size`` actually applies
``MemoryGovernor.budget_tokens()`` as a per-request context-size ceiling
when pressure is elevated, and leaves the configured ``_max_kv_size``
untouched at NORMAL pressure or when no governor is present.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

import squish.server as _srv
from squish.serving.memory_governor import LEVEL_CRITICAL, LEVEL_NORMAL, LEVEL_URGENT, LEVEL_WARNING


@pytest.fixture(autouse=True)
def _reset_kv_size_state():
    """Every test gets a clean slate: no governor wired, no configured ceiling."""
    orig_governor   = _srv._memory_governor
    orig_max_kv     = _srv._max_kv_size

    _srv._memory_governor = None
    _srv._max_kv_size     = None

    yield

    _srv._memory_governor = orig_governor
    _srv._max_kv_size     = orig_max_kv


def _mock_governor(level, budget_tokens):
    gov = MagicMock()
    gov.pressure_level = level
    gov.budget_tokens.return_value = budget_tokens
    return gov


class TestNoGovernorOrNormalPressure:
    def test_no_governor_returns_configured_ceiling_unchanged(self):
        _srv._max_kv_size = 8192
        assert _srv._effective_max_kv_size() == 8192

    def test_no_governor_returns_none_when_unconfigured(self):
        _srv._max_kv_size = None
        assert _srv._effective_max_kv_size() is None

    def test_normal_pressure_returns_configured_ceiling_unchanged(self):
        _srv._max_kv_size = 8192
        _srv._memory_governor = _mock_governor(LEVEL_NORMAL, budget_tokens=100)

        assert _srv._effective_max_kv_size() == 8192

    def test_normal_pressure_does_not_even_query_budget_tokens(self):
        """Don't apply the ceiling unconditionally — only degrade when there's
        a reason to. At NORMAL, budget_tokens() shouldn't be called at all."""
        _srv._max_kv_size = 8192
        gov = _mock_governor(LEVEL_NORMAL, budget_tokens=100)
        _srv._memory_governor = gov

        _srv._effective_max_kv_size()

        gov.budget_tokens.assert_not_called()


class TestElevatedPressureCapsContext:
    def test_warning_caps_at_budget_when_budget_is_lower(self):
        _srv._max_kv_size = 32768
        _srv._memory_governor = _mock_governor(LEVEL_WARNING, budget_tokens=4096)

        assert _srv._effective_max_kv_size() == 4096

    def test_urgent_caps_at_budget_when_budget_is_lower(self):
        _srv._max_kv_size = 32768
        _srv._memory_governor = _mock_governor(LEVEL_URGENT, budget_tokens=1024)

        assert _srv._effective_max_kv_size() == 1024

    def test_critical_caps_at_budget_too(self):
        """CRITICAL doesn't reject requests yet (Phase 4, unbuilt) — until it
        does, the context ceiling is the only degradation, so it must still
        apply here rather than silently falling back to unlimited."""
        _srv._max_kv_size = 32768
        _srv._memory_governor = _mock_governor(LEVEL_CRITICAL, budget_tokens=512)

        assert _srv._effective_max_kv_size() == 512

    def test_never_raises_the_configured_ceiling(self):
        """A generous governor budget must not override a stricter operator
        setting — this only ever lowers the ceiling."""
        _srv._max_kv_size = 2048
        _srv._memory_governor = _mock_governor(LEVEL_WARNING, budget_tokens=999_999)

        assert _srv._effective_max_kv_size() == 2048

    def test_unconfigured_ceiling_adopts_budget_under_pressure(self):
        """When no explicit --max-kv-size was set, elevated pressure still
        imposes a ceiling instead of leaving it at mlx_lm's implicit default."""
        _srv._max_kv_size = None
        _srv._memory_governor = _mock_governor(LEVEL_URGENT, budget_tokens=1500)

        assert _srv._effective_max_kv_size() == 1500


class TestBudgetTokensActuallyConsulted:
    def test_low_available_memory_reduces_effective_budget(self):
        """End-to-end with a real MemoryGovernor.budget_tokens() computation
        (not just a mocked return value) — mock only the underlying
        pressure/vm_stat probes, per the existing governor test pattern."""
        from unittest.mock import patch

        from squish.serving.memory_governor import MemoryGovernor

        with patch("squish.serving.memory_governor._read_pressure_level", return_value=LEVEL_WARNING), \
             patch("squish.serving.memory_governor._read_vm_stat", return_value=(1.001, 1.0)):
            gov = MemoryGovernor(poll_interval=60.0).start()
        try:
            _srv._max_kv_size = 32768
            _srv._memory_governor = gov

            # available=1.001GB, headroom=1GB → usable=0.001GB; bytes_per_token=512 default
            expected_budget = int(0.001 * (1 << 30) / 512)
            result = _srv._effective_max_kv_size()

            assert result == expected_budget
            assert result < 32768  # confirms it was actually reduced, not discarded
        finally:
            gov.stop()
