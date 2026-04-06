"""tests/test_squash_wave35.py — Wave 35: CLI help text surfaces eu-cra / fedramp / cmmc.

Covers:
  Policy registry regression guard:
    - AVAILABLE_POLICIES contains all 9 expected policy names
    - eu-cra, fedramp, cmmc are present

  squash attest --policy help text:
    - Help string mentions "eu-cra"
    - Help string mentions "fedramp"
    - Help string mentions "cmmc"
    - Help string mentions "squash policies" as a discovery hint

  squash attest-composed --policy help text:
    - Same three policy names present in help string
    - Discovery hint present

  Integration shim help text (attest-mlflow, attest-wandb,
  attest-huggingface, attest-langchain):
    - Each --policies argument help string mentions eu-cra, fedramp, cmmc

  squash policies command (subprocess integration):
    - CLI output lists eu-cra, fedramp, cmmc when invoked

Test taxonomy:
  - TestPolicyRegistryRegression: pure unit — imports only, no I/O.
  - TestAttestHelpText: pure unit — inspects argparse action objects directly.
  - TestIntegrationShimHelpText: pure unit — inspects argparse action objects.
  - TestPoliciesCommandOutput: subprocess integration — spawns squash as a child
    process reading stdout; cleans up automatically (no temp files created).
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from squish.squash.policy import AVAILABLE_POLICIES


# ── helpers ───────────────────────────────────────────────────────────────────


def _build_parser():
    """Return the squash CLI argparse parser without side effects."""
    from squish.squash.cli import _build_parser as _bp
    return _bp()


def _find_action(parser, subcommand: str, dest: str):
    """Return the argparse Action object for *dest* inside *subcommand*."""
    sub_actions = [
        a for a in parser._subparsers._group_actions
        if hasattr(a, "choices")
    ]
    assert sub_actions, "no subparsers found"
    sub_parser = sub_actions[0].choices[subcommand]
    for action in sub_parser._actions:
        if action.dest == dest:
            return action
    raise KeyError(f"dest={dest!r} not found in subcommand={subcommand!r}")


# ── policy registry regression guard ─────────────────────────────────────────


class TestPolicyRegistryRegression:
    """Ensure the 9 policy names present after Wave 34 are still registered."""

    EXPECTED = frozenset(
        {
            "eu-ai-act",
            "nist-ai-rmf",
            "owasp-llm-top10",
            "iso-42001",
            "enterprise-strict",
            "strict",
            "eu-cra",
            "fedramp",
            "cmmc",
        }
    )

    def test_all_nine_policies_present(self):
        assert self.EXPECTED <= AVAILABLE_POLICIES, (
            f"Missing: {self.EXPECTED - AVAILABLE_POLICIES}"
        )

    def test_eu_cra_in_available(self):
        assert "eu-cra" in AVAILABLE_POLICIES

    def test_fedramp_in_available(self):
        assert "fedramp" in AVAILABLE_POLICIES

    def test_cmmc_in_available(self):
        assert "cmmc" in AVAILABLE_POLICIES

    def test_available_policies_is_frozenset(self):
        assert isinstance(AVAILABLE_POLICIES, frozenset)


# ── squash attest --policy help text ─────────────────────────────────────────


class TestAttestHelpText:
    """Help text for 'squash attest --policy' must surface the Wave 34 policies."""

    @pytest.fixture(scope="class")
    def attest_policy_help(self):
        action = _find_action(_build_parser(), "attest", "policies")
        return action.help or ""

    def test_eu_cra_in_attest_help(self, attest_policy_help):
        assert "eu-cra" in attest_policy_help, (
            f"'eu-cra' not found in attest --policy help: {attest_policy_help!r}"
        )

    def test_fedramp_in_attest_help(self, attest_policy_help):
        assert "fedramp" in attest_policy_help

    def test_cmmc_in_attest_help(self, attest_policy_help):
        assert "cmmc" in attest_policy_help

    def test_discovery_hint_in_attest_help(self, attest_policy_help):
        assert "squash policies" in attest_policy_help, (
            "Help text should direct users to 'squash policies' for full policy list"
        )

    def test_default_enterprise_strict_still_mentioned(self, attest_policy_help):
        assert "enterprise-strict" in attest_policy_help


# ── squash attest-composed --policy help text ─────────────────────────────────


class TestAttestComposedHelpText:
    """Help text for 'squash attest-composed --policy' must surface Wave 34 policies."""

    @pytest.fixture(scope="class")
    def ac_policy_help(self):
        action = _find_action(_build_parser(), "attest-composed", "policies")
        return action.help or ""

    def test_eu_cra_in_attest_composed_help(self, ac_policy_help):
        assert "eu-cra" in ac_policy_help

    def test_fedramp_in_attest_composed_help(self, ac_policy_help):
        assert "fedramp" in ac_policy_help

    def test_cmmc_in_attest_composed_help(self, ac_policy_help):
        assert "cmmc" in ac_policy_help

    def test_discovery_hint_in_attest_composed_help(self, ac_policy_help):
        assert "squash policies" in ac_policy_help


# ── integration shim help text ───────────────────────────────────────────────


class TestIntegrationShimHelpText:
    """--policies help for attest-mlflow / wandb / huggingface / langchain."""

    SHIMS = ["attest-mlflow", "attest-wandb", "attest-huggingface", "attest-langchain"]

    @pytest.fixture(scope="class")
    def shim_helps(self):
        parser = _build_parser()
        return {
            shim: _find_action(parser, shim, "policies").help or ""
            for shim in self.SHIMS
        }

    @pytest.mark.parametrize("shim", SHIMS)
    def test_eu_cra_in_shim_help(self, shim_helps, shim):
        assert "eu-cra" in shim_helps[shim], (
            f"'eu-cra' not in {shim} --policies help: {shim_helps[shim]!r}"
        )

    @pytest.mark.parametrize("shim", SHIMS)
    def test_fedramp_in_shim_help(self, shim_helps, shim):
        assert "fedramp" in shim_helps[shim]

    @pytest.mark.parametrize("shim", SHIMS)
    def test_cmmc_in_shim_help(self, shim_helps, shim):
        assert "cmmc" in shim_helps[shim]


# ── squash policies command (subprocess integration) ──────────────────────────


class TestPoliciesCommandOutput:
    """Subprocess integration: 'squash policies' stdout lists eu-cra/fedramp/cmmc."""

    @pytest.fixture(scope="class")
    def policies_output(self):
        result = subprocess.run(
            [sys.executable, "-m", "squish.squash.cli", "policies"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        # Accept exit 0 or 1 — the command may warn on missing squash install
        # but must still list built-in policies from the registry.
        return result.stdout + result.stderr

    def test_eu_cra_listed(self, policies_output):
        assert "eu-cra" in policies_output, (
            f"'eu-cra' not found in 'squash policies' output:\n{policies_output}"
        )

    def test_fedramp_listed(self, policies_output):
        assert "fedramp" in policies_output

    def test_cmmc_listed(self, policies_output):
        assert "cmmc" in policies_output

    def test_enterprise_strict_listed(self, policies_output):
        """Regression: existing policy must still appear."""
        assert "enterprise-strict" in policies_output
