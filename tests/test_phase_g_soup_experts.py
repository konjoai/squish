"""
tests/test_phase_g_soup_experts.py

Coverage tests for squish/soup_experts.py — Phase G2 (Soup-of-Experts).

``apply_mix()`` is hardware-bound and marked ``# pragma: no cover``.
"""

from __future__ import annotations

import pytest

from squish.soup_experts import SoupOfExperts


class TestSoupOfExpertsRegistry:
    def test_register_and_is_registered(self):
        soe = SoupOfExperts()
        assert not soe.is_registered("legal")
        soe.register_expert("legal", "/adapters/legal.safetensors")
        assert soe.is_registered("legal")

    def test_registered_domains_sorted(self):
        soe = SoupOfExperts()
        soe.register_expert("code", "/c.st")
        soe.register_expert("legal", "/l.st")
        soe.register_expert("medical", "/m.st")
        assert soe.registered_domains() == ["code", "legal", "medical"]

    def test_registered_domains_empty(self):
        soe = SoupOfExperts()
        assert soe.registered_domains() == []

    def test_expert_path_registered(self):
        soe = SoupOfExperts()
        soe.register_expert("code", "/path/code.safetensors")
        assert soe.expert_path("code") == "/path/code.safetensors"

    def test_expert_path_unregistered_raises(self):
        soe = SoupOfExperts()
        with pytest.raises(KeyError, match="Expert not registered"):
            soe.expert_path("ghost")

    def test_register_default_weight(self):
        soe = SoupOfExperts()
        soe.register_expert("code", "/c.st", default_weight=0.0)
        assert soe.get_weights()["code"] == 0.0

    def test_register_nonzero_default_weight(self):
        soe = SoupOfExperts()
        soe.register_expert("legal", "/l.st", default_weight=0.5)
        assert soe.get_weights()["legal"] == 0.5


class TestSoupOfExpertsWeights:
    def _soe_with_three(self):
        soe = SoupOfExperts()
        soe.register_expert("code", "/c.st")
        soe.register_expert("legal", "/l.st")
        soe.register_expert("medical", "/m.st")
        return soe

    def test_set_mixing_weights_valid(self):
        soe = self._soe_with_three()
        soe.set_mixing_weights({"code": 0.5, "legal": 0.3, "medical": 0.2})
        w = soe.get_weights()
        assert abs(w["code"] - 0.5) < 1e-9
        assert abs(w["legal"] - 0.3) < 1e-9
        assert abs(w["medical"] - 0.2) < 1e-9

    def test_set_mixing_weights_sum_not_one_raises(self):
        soe = self._soe_with_three()
        with pytest.raises(ValueError, match="sum to"):
            soe.set_mixing_weights({"code": 0.5, "legal": 0.3, "medical": 0.5})

    def test_set_mixing_weights_unknown_domain_raises(self):
        soe = self._soe_with_three()
        with pytest.raises(KeyError, match="Expert not registered"):
            soe.set_mixing_weights({"ghost": 1.0})

    def test_set_mixing_weights_partial_update(self):
        soe = SoupOfExperts()
        soe.register_expert("a", "/a")
        soe.register_expert("b", "/b")
        soe.set_mixing_weights({"a": 0.6, "b": 0.4})
        assert soe.get_weights()["a"] == pytest.approx(0.6)

    def test_get_weights_is_copy(self):
        soe = SoupOfExperts()
        soe.register_expert("a", "/a")
        w1 = soe.get_weights()
        w2 = soe.get_weights()
        assert w1 is not w2

    def test_reset_weights(self):
        soe = SoupOfExperts()
        soe.register_expert("code", "/c.st", default_weight=0.0)
        soe.register_expert("legal", "/l.st", default_weight=0.0)
        soe.set_mixing_weights({"code": 0.7, "legal": 0.3})
        soe.reset_weights()
        assert soe.get_weights()["code"] == 0.0
        assert soe.get_weights()["legal"] == 0.0

    def test_tolerance_boundary_passes(self):
        """Weights summing to 0.999 within default tolerance (0.01) pass."""
        soe = SoupOfExperts()
        soe.register_expert("a", "/a")
        soe.register_expert("b", "/b")
        soe.set_mixing_weights({"a": 0.5, "b": 0.499})  # sum = 0.999

    def test_tolerance_exceeded_raises(self):
        """Weights summing to 0.95 exceed default tolerance."""
        soe = SoupOfExperts()
        soe.register_expert("a", "/a")
        with pytest.raises(ValueError):
            soe.set_mixing_weights({"a": 0.95})

    def test_custom_tolerance(self):
        """Strict tolerance (0.001) rejects sum=0.999."""
        soe = SoupOfExperts(tolerance=0.001)
        soe.register_expert("a", "/a")
        soe.register_expert("b", "/b")
        with pytest.raises(ValueError):
            soe.set_mixing_weights({"a": 0.5, "b": 0.499})


class TestSoupOfExpertsDomainDetection:
    def test_no_experts_returns_empty(self):
        soe = SoupOfExperts()
        result = soe.detect_domain("any prompt")
        assert result == {}

    def test_exact_domain_match(self):
        soe = SoupOfExperts()
        soe.register_expert("code", "/c.st")
        soe.register_expert("legal", "/l.st")
        result = soe.detect_domain("write some code for me")
        assert result["code"] > result["legal"]

    def test_no_overlap_returns_uniform(self):
        soe = SoupOfExperts()
        soe.register_expert("code", "/c.st")
        soe.register_expert("legal", "/l.st")
        result = soe.detect_domain("hello world xyz")
        assert result["code"] == pytest.approx(0.5)
        assert result["legal"] == pytest.approx(0.5)

    def test_weights_sum_to_one(self):
        soe = SoupOfExperts()
        soe.register_expert("code", "/c.st")
        soe.register_expert("legal", "/l.st")
        soe.register_expert("medical", "/m.st")
        result = soe.detect_domain("write a function to diagnose a patient case")
        total = sum(result.values())
        assert total == pytest.approx(1.0, abs=1e-9)

    def test_hyphenated_domain_detection(self):
        """Domain 'legal-review' should match prompt containing 'legal'."""
        soe = SoupOfExperts()
        soe.register_expert("legal-review", "/l.st")
        result = soe.detect_domain("legal document")
        # 'legal' appears in both domain tokens and prompt → non-zero score
        assert result["legal-review"] > 0.0

    def test_single_expert_uniform(self):
        """Single registered expert with no overlap → weight = 1.0."""
        soe = SoupOfExperts()
        soe.register_expert("xray", "/x.st")
        result = soe.detect_domain("completely unrelated text here")
        assert result["xray"] == pytest.approx(1.0)
