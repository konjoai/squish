"""tests/test_squash_w83.py — Wave 83: NIST AI RMF 1.0 controls scanner.

Tests for squish/squash/nist_rmf.py — all pure-unit, no I/O.
``NistRmfScanner.from_dict()`` is used throughout so we never hit the
filesystem; that keeps these as fast deterministic unit tests.
"""

from __future__ import annotations

import json

import pytest

from squish.squash.nist_rmf import (
    NistControlStatus,
    NistRmfControl,
    NistRmfFunction,
    NistRmfPosture,
    NistRmfReport,
    NistRmfScanner,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_EMPTY: dict = {}


def _scan(artifacts: dict) -> NistRmfReport:
    """Convenience wrapper."""
    return NistRmfScanner.from_dict(artifacts)


# ---------------------------------------------------------------------------
# TestNistRmfFunction — enum membership
# ---------------------------------------------------------------------------

class TestNistRmfFunction:
    def test_has_govern(self):
        assert NistRmfFunction.GOVERN == "GOVERN"

    def test_has_map(self):
        assert NistRmfFunction.MAP == "MAP"

    def test_has_measure(self):
        assert NistRmfFunction.MEASURE == "MEASURE"

    def test_has_manage(self):
        assert NistRmfFunction.MANAGE == "MANAGE"

    def test_exactly_four_values(self):
        assert len(NistRmfFunction) == 4


# ---------------------------------------------------------------------------
# TestNistControlStatus — enum membership
# ---------------------------------------------------------------------------

class TestNistControlStatus:
    def test_has_pass(self):
        assert NistControlStatus.PASS == "PASS"

    def test_has_fail(self):
        assert NistControlStatus.FAIL == "FAIL"

    def test_has_na(self):
        assert NistControlStatus.NOT_APPLICABLE == "N/A"

    def test_has_unknown(self):
        assert NistControlStatus.UNKNOWN == "UNKNOWN"


# ---------------------------------------------------------------------------
# TestNistRmfControl — dataclass
# ---------------------------------------------------------------------------

class TestNistRmfControl:
    def test_defaults(self):
        c = NistRmfControl(
            control_id="GOV-1.1",
            function=NistRmfFunction.GOVERN,
            title="Test",
            description="Desc",
        )
        assert c.status == NistControlStatus.UNKNOWN
        assert c.evidence == []
        assert c.recommendations == []

    def test_to_dict_keys(self):
        c = NistRmfControl(
            control_id="MS-2.1",
            function=NistRmfFunction.MEASURE,
            title="Metrics",
            description="...",
            status=NistControlStatus.PASS,
            evidence=["found metrics"],
        )
        d = c.to_dict()
        assert d["control_id"] == "MS-2.1"
        assert d["function"] == "MEASURE"
        assert d["status"] == "PASS"
        assert d["evidence"] == ["found metrics"]

    def test_to_dict_recommendations_empty_by_default(self):
        c = NistRmfControl(
            control_id="MG-1.1",
            function=NistRmfFunction.MANAGE,
            title="T",
            description="D",
        )
        assert c.to_dict()["recommendations"] == []


# ---------------------------------------------------------------------------
# TestNistRmfReport — structure and serialisation
# ---------------------------------------------------------------------------

class TestNistRmfReport:
    def _make_report(self, posture: NistRmfPosture) -> NistRmfReport:
        controls = [
            NistRmfControl(
                "GOV-1.1",
                NistRmfFunction.GOVERN,
                "T",
                "D",
                NistControlStatus.PASS,
            )
        ]
        return NistRmfReport(
            model_path="/tmp/model",
            controls=controls,
            posture=posture,
            pass_count=1,
            fail_count=0,
            na_count=0,
        )

    def test_to_dict_contains_posture(self):
        r = self._make_report(NistRmfPosture.COMPLIANT)
        d = r.to_dict()
        assert d["posture"] == "COMPLIANT"

    def test_to_dict_contains_controls(self):
        r = self._make_report(NistRmfPosture.PARTIAL)
        d = r.to_dict()
        assert len(d["controls"]) == 1

    def test_to_json_is_valid(self):
        r = self._make_report(NistRmfPosture.NON_COMPLIANT)
        parsed = json.loads(r.to_json())
        assert "posture" in parsed

    def test_summary_contains_posture(self):
        r = self._make_report(NistRmfPosture.COMPLIANT)
        summary = r.summary()
        assert "COMPLIANT" in summary

    def test_summary_contains_all_functions(self):
        r = _scan(_EMPTY)
        summary = r.summary()
        for fn in ("GOVERN", "MAP", "MEASURE", "MANAGE"):
            assert fn in summary


# ---------------------------------------------------------------------------
# TestNistRmfScannerPosture — posture calculation
# ---------------------------------------------------------------------------

class TestNistRmfScannerPosture:
    def test_empty_artifacts_non_compliant(self):
        r = _scan(_EMPTY)
        assert r.posture == NistRmfPosture.NON_COMPLIANT

    def test_report_has_13_controls(self):
        r = _scan(_EMPTY)
        assert len(r.controls) == 13

    def test_all_four_functions_represented(self):
        r = _scan(_EMPTY)
        functions_seen = {c.function for c in r.controls}
        assert functions_seen == set(NistRmfFunction)

    def test_full_artifacts_compliant(self):
        """Rich artifact set should yield COMPLIANT posture."""
        artifacts = {
            "sbom": {
                "metadata": {
                    "supplier": {"name": "Acme AI"},
                    "component": {"description": "GPT model for text generation"},
                },
                "formulation": [{"ref": "training_job_123"}],
            },
            "attestation": {
                "signer": "ci-pipeline@acme.ai",
                "accuracy": {"arc_easy": 0.706},
                "risk_assessment": {"nist_rmf": "MODERATE"},
            },
            "scan": {"status": "safe", "findings": []},
            "model_card": {
                "limitations": "Not for medical diagnosis",
                "intended_use": "Enterprise chatbot",
                "bias": "Gender bias audit performed",
                "metrics": {"arc_easy": 70.6},
            },
            "provenance": {"training_runs": [{"id": "run-01"}]},
            "policy": {"template": "eu_ai_act_v1", "status": "pass"},
            "vex": {"statements": [{"vuln": "CVE-2024-0001", "state": "not_affected"}]},
            "drift": {"status": "stable", "p_value": 0.42},
        }
        r = _scan(artifacts)
        assert r.posture == NistRmfPosture.COMPLIANT
        assert r.fail_count == 0

    def test_partial_posture_when_roughly_half_pass(self):
        """Provide enough to pass ~50% of controls but not all."""
        artifacts = {
            "scan": {"status": "safe"},
            "policy": {"status": "pass"},
            "sbom": {
                "metadata": {"supplier": {"name": "Corp"}},
                "formulation": [{"ref": "r1"}],
            },
            "attestation": {"signer": "ci@corp.com"},
            "model_card": {"limitations": "x"},
            # no provenance → MAP-3.1 FAIL
            # no eval metrics → MS-2.1 FAIL
            # no bias → MS-3.1 FAIL
            # no vex → MG-1.1 FAIL
            # no drift → MG-2.1 FAIL
        }
        r = _scan(artifacts)
        assert r.posture in (NistRmfPosture.PARTIAL, NistRmfPosture.COMPLIANT)


# ---------------------------------------------------------------------------
# TestGovControls — GOVERN function
# ---------------------------------------------------------------------------

class TestGovControls:
    def test_gov_1_1_passes_with_policy_template(self):
        r = _scan({"policy": {"template": "nist_rmf_v1"}})
        ctrl = next(c for c in r.controls if c.control_id == "GOV-1.1")
        assert ctrl.status == NistControlStatus.PASS

    def test_gov_1_1_fails_without_policy(self):
        r = _scan(_EMPTY)
        ctrl = next(c for c in r.controls if c.control_id == "GOV-1.1")
        assert ctrl.status == NistControlStatus.FAIL
        assert ctrl.recommendations  # non-empty

    def test_gov_2_1_passes_with_supplier(self):
        r = _scan({"sbom": {"metadata": {"supplier": {"name": "Acme"}}}})
        ctrl = next(c for c in r.controls if c.control_id == "GOV-2.1")
        assert ctrl.status == NistControlStatus.PASS

    def test_gov_3_1_passes_with_signer(self):
        r = _scan({"attestation": {"signer": "ci@example.com"}})
        ctrl = next(c for c in r.controls if c.control_id == "GOV-3.1")
        assert ctrl.status == NistControlStatus.PASS

    def test_gov_4_1_passes_with_limitations(self):
        r = _scan({"model_card": {"limitations": "Not for clinical use"}})
        ctrl = next(c for c in r.controls if c.control_id == "GOV-4.1")
        assert ctrl.status == NistControlStatus.PASS


# ---------------------------------------------------------------------------
# TestMeasureControls — MEASURE function
# ---------------------------------------------------------------------------

class TestMeasureControls:
    def test_ms_1_1_passes_with_scan(self):
        r = _scan({"scan": {"status": "safe"}})
        ctrl = next(c for c in r.controls if c.control_id == "MS-1.1")
        assert ctrl.status == NistControlStatus.PASS

    def test_ms_2_1_passes_with_accuracy_in_attestation(self):
        r = _scan({"attestation": {"accuracy": {"arc_easy": 70.6}}})
        ctrl = next(c for c in r.controls if c.control_id == "MS-2.1")
        assert ctrl.status == NistControlStatus.PASS

    def test_ms_4_1_fails_on_unsafe_scan(self):
        r = _scan({"scan": {"status": "unsafe", "findings": [{"cve": "CVE-x"}]}})
        ctrl = next(c for c in r.controls if c.control_id == "MS-4.1")
        assert ctrl.status == NistControlStatus.FAIL
        assert ctrl.recommendations

    def test_ms_4_1_passes_on_clean_scan(self):
        r = _scan({"scan": {"status": "safe"}})
        ctrl = next(c for c in r.controls if c.control_id == "MS-4.1")
        assert ctrl.status == NistControlStatus.PASS


# ---------------------------------------------------------------------------
# TestManageControls — MANAGE function
# ---------------------------------------------------------------------------

class TestManageControls:
    def test_mg_1_1_passes_with_vex(self):
        r = _scan({"vex": {"statements": []}})
        ctrl = next(c for c in r.controls if c.control_id == "MG-1.1")
        assert ctrl.status == NistControlStatus.PASS

    def test_mg_1_1_fails_without_vex(self):
        r = _scan(_EMPTY)
        ctrl = next(c for c in r.controls if c.control_id == "MG-1.1")
        assert ctrl.status == NistControlStatus.FAIL

    def test_mg_2_1_passes_with_drift(self):
        r = _scan({"drift": {"status": "stable"}})
        ctrl = next(c for c in r.controls if c.control_id == "MG-2.1")
        assert ctrl.status == NistControlStatus.PASS

    def test_mg_2_1_fails_without_drift(self):
        r = _scan(_EMPTY)
        ctrl = next(c for c in r.controls if c.control_id == "MG-2.1")
        assert ctrl.status == NistControlStatus.FAIL


# ---------------------------------------------------------------------------
# TestExports — __init__.py surface
# ---------------------------------------------------------------------------

class TestExports:
    def test_nist_rmf_function_exported(self):
        from squish.squash import NistRmfFunction as F  # noqa: F401
        assert F.GOVERN

    def test_nist_rmf_report_exported(self):
        from squish.squash import NistRmfReport as R  # noqa: F401
        assert R

    def test_nist_rmf_scanner_exported(self):
        from squish.squash import NistRmfScanner as S  # noqa: F401
        assert S

    def test_nist_control_status_exported(self):
        from squish.squash import NistControlStatus as CS  # noqa: F401
        assert CS.PASS

    def test_nist_rmf_posture_exported(self):
        from squish.squash import NistRmfPosture as P  # noqa: F401
        assert P.COMPLIANT
