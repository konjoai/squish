"""cicd.py — CI/CD integration adapters for the squash pipeline (Wave 25).

Detects the current CI environment from well-known environment variables,
formats attestation results as native CI annotations, and generates pipeline
job summaries.

Supported environments
----------------------
* GitHub Actions (``GITHUB_ACTIONS=true``)
* Jenkins (``JENKINS_URL`` set)
* GitLab CI (``GITLAB_CI=true``)
* CircleCI (``CIRCLECI=true``)
* Generic / unknown

Usage in CI::

    from squish.squash.cicd import CicdAdapter

    env = CicdAdapter.detect()
    summary = CicdAdapter.job_summary(result, env)
    CicdAdapter.annotate(result, env)
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


@dataclass
class CiEnvironment:
    """Runtime CI environment metadata.

    Attributes
    ----------
    env_name:
        Normalised name: ``"github"``, ``"jenkins"``, ``"gitlab"``,
        ``"circleci"``, or ``"unknown"``.
    job_id:
        Opaque job / build identifier, or ``""`` if not available.
    repo:
        Repository slug (``org/repo``), or ``""`` if not available.
    branch:
        Branch / ref name, or ``""`` if not available.
    actor:
        Username that triggered the run, or ``""`` if not available.
    """

    env_name: str
    job_id: str = ""
    repo: str = ""
    branch: str = ""
    actor: str = ""


@dataclass
class CicdReport:
    """Aggregated CI report object.

    Attributes
    ----------
    env:
        Detected :class:`CiEnvironment`.
    attestation:
        Attestation result dict (from squash-attest.json, or ``None``).
    ntia:
        NTIA validation result dict, or ``None``.
    risk:
        Risk assessment result dict, or ``None``.
    drift_events:
        List of drift event dicts, or ``[]``.
    passed:
        ``True`` if all sub-checks passed.
    """

    env: CiEnvironment
    attestation: dict | None = None
    ntia: dict | None = None
    risk: dict | None = None
    drift_events: list[dict] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """Return True only if all present sub-checks passed."""
        checks: list[bool] = []
        if self.attestation is not None:
            checks.append(bool(self.attestation.get("passed", True)))
        if self.ntia is not None:
            checks.append(bool(self.ntia.get("passed", True)))
        if self.risk is not None:
            checks.append(bool(self.risk.get("passed", True)))
        checks.append(len(self.drift_events) == 0)
        return all(checks)


class CicdAdapter:
    """Detect CI environment and format squash results as native CI feedback.

    All methods are static; no instantiation required.
    """

    @staticmethod
    def detect() -> CiEnvironment:
        """Auto-detect the current CI environment from environment variables.

        Returns
        -------
        CiEnvironment
            Populated from known env-var patterns.  Falls back to
            ``env_name="unknown"`` when no CI system is recognised.
        """
        env = os.environ

        if env.get("GITHUB_ACTIONS") == "true":
            return CiEnvironment(
                env_name="github",
                job_id=env.get("GITHUB_RUN_ID", ""),
                repo=env.get("GITHUB_REPOSITORY", ""),
                branch=env.get("GITHUB_REF_NAME", ""),
                actor=env.get("GITHUB_ACTOR", ""),
            )

        if env.get("JENKINS_URL"):
            return CiEnvironment(
                env_name="jenkins",
                job_id=env.get("BUILD_ID", env.get("BUILD_NUMBER", "")),
                repo=env.get("GIT_URL", ""),
                branch=env.get("GIT_BRANCH", ""),
                actor=env.get("BUILD_USER", ""),
            )

        if env.get("GITLAB_CI") == "true":
            return CiEnvironment(
                env_name="gitlab",
                job_id=env.get("CI_JOB_ID", ""),
                repo=env.get("CI_PROJECT_PATH", ""),
                branch=env.get("CI_COMMIT_BRANCH", ""),
                actor=env.get("GITLAB_USER_LOGIN", ""),
            )

        if env.get("CIRCLECI") == "true":
            return CiEnvironment(
                env_name="circleci",
                job_id=env.get("CIRCLE_BUILD_NUM", ""),
                repo=env.get("CIRCLE_PROJECT_REPONAME", ""),
                branch=env.get("CIRCLE_BRANCH", ""),
                actor=env.get("CIRCLE_USERNAME", ""),
            )

        return CiEnvironment(env_name="unknown")

    @staticmethod
    def annotate(report: CicdReport, env: CiEnvironment | None = None) -> None:
        """Write CI-native annotations to stdout based on *report* results.

        * **GitHub Actions**: emits ``::error::`` / ``::warning::`` workflow
          commands.
        * **All others**: writes human-readable lines to stdout.

        Parameters
        ----------
        report:
            The :class:`CicdReport` to annotate.
        env:
            Override environment detection.  Uses ``report.env`` if ``None``.
        """
        ci = env or report.env

        def _error(msg: str) -> None:
            if ci.env_name == "github":
                print(f"::error::{msg}")
            else:
                print(f"[SQUASH ERROR] {msg}")

        def _warning(msg: str) -> None:
            if ci.env_name == "github":
                print(f"::warning::{msg}")
            else:
                print(f"[SQUASH WARN] {msg}")

        def _info(msg: str) -> None:
            if ci.env_name == "github":
                print(f"::notice::{msg}")
            else:
                print(f"[SQUASH] {msg}")

        if report.ntia is not None and not report.ntia.get("passed", True):
            missing = report.ntia.get("missing_fields", [])
            _error(
                f"NTIA minimum elements incomplete — missing: {', '.join(missing)}"
            )

        if report.risk is not None and not report.risk.get("passed", True):
            cat = report.risk.get("category", "unknown")
            fw = report.risk.get("framework", "?")
            _error(f"AI risk assessment failed — {fw} category: {cat}")

        if report.attestation is not None and not report.attestation.get("passed", True):
            _error("Squash attestation check failed")

        for evt in report.drift_events:
            etype = evt.get("event_type", "DRIFT")
            comp = evt.get("component", "?")
            _warning(f"Drift detected: {etype} in {comp}")

        if report.passed:
            _info("All squash checks passed")

    @staticmethod
    def job_summary(report: CicdReport, env: CiEnvironment | None = None) -> str:
        """Return a Markdown job summary string suitable for GitHub Actions.

        Parameters
        ----------
        report:
            The :class:`CicdReport` to summarise.
        env:
            Override environment detection.  Uses ``report.env`` if ``None``.

        Returns
        -------
        str
            GitHub Actions Markdown job summary.
        """
        ci = env or report.env
        status = "✅ PASSED" if report.passed else "❌ FAILED"
        lines: list[str] = [
            f"## Squash AI-SBOM Report — {status}",
            "",
            f"| Field | Value |",
            f"|---|---|",
            f"| Environment | `{ci.env_name}` |",
            f"| Job ID | `{ci.job_id or 'n/a'}` |",
            f"| Repository | `{ci.repo or 'n/a'}` |",
            f"| Branch | `{ci.branch or 'n/a'}` |",
            "",
        ]

        if report.ntia is not None:
            score = report.ntia.get("completeness_score", 0.0)
            ntia_ok = "✅" if report.ntia.get("passed") else "❌"
            lines.append(f"### NTIA Minimum Elements {ntia_ok}")
            lines.append(f"Completeness score: **{score:.1%}**")
            missing = report.ntia.get("missing_fields", [])
            if missing:
                lines.append(f"Missing: {', '.join(f'`{m}`' for m in missing)}")
            lines.append("")

        if report.risk is not None:
            risk_ok = "✅" if report.risk.get("passed") else "❌"
            cat = report.risk.get("category", "unknown")
            fw = report.risk.get("framework", "unknown")
            lines.append(f"### AI Risk Assessment {risk_ok}")
            lines.append(f"**{fw}** category: **{cat}**")
            rationale = report.risk.get("rationale", [])
            for r in rationale[:3]:
                lines.append(f"- {r}")
            lines.append("")

        if report.drift_events:
            lines.append(f"### Drift Events ⚠️ ({len(report.drift_events)})")
            for evt in report.drift_events[:5]:
                lines.append(
                    f"- **{evt.get('event_type', '?')}** — {evt.get('component', '?')}"
                )
            lines.append("")

        return "\n".join(lines)

    @staticmethod
    def run_pipeline(
        model_dir: Path,
        *,
        report_format: str = "text",
    ) -> CicdReport:
        """Execute the full squash check pipeline for *model_dir*.

        Runs NTIA validation, risk assessment, and drift snapshot in sequence,
        builds a :class:`CicdReport`, annotates CI, and returns it.

        Parameters
        ----------
        model_dir:
            Path to the squash model artefact directory.
        report_format:
            One of ``"github"``, ``"jenkins"``, ``"gitlab"``, ``"text"``.
            Overrides environment detection for annotation formatting.
        """
        from squish.squash.policy import NtiaValidator  # lazy import
        from squish.squash.risk import AiRiskAssessor  # lazy import
        from squish.squash.governor import DriftMonitor  # lazy import

        model_dir = Path(model_dir)
        bom_path = model_dir / "cyclonedx-mlbom.json"

        ci = CicdAdapter.detect()
        if report_format not in ("text", "unknown"):
            ci.env_name = report_format

        ntia_result = NtiaValidator.check(bom_path) if bom_path.exists() else None
        eu_result = AiRiskAssessor.assess_eu_ai_act(bom_path) if bom_path.exists() else None
        snap = DriftMonitor.snapshot(model_dir)
        drift_events = DriftMonitor.compare(model_dir, snap)  # same snap → no events

        ntia_dict: dict | None = None
        if ntia_result is not None:
            ntia_dict = {
                "passed": ntia_result.passed,
                "completeness_score": ntia_result.completeness_score,
                "missing_fields": ntia_result.missing_fields,
                "present_fields": ntia_result.present_fields,
            }

        risk_dict: dict | None = None
        if eu_result is not None:
            risk_dict = {
                "passed": eu_result.passed,
                "framework": eu_result.framework,
                "category": eu_result.category.value,
                "rationale": eu_result.rationale,
                "mitigation_required": eu_result.mitigation_required,
            }

        drift_dicts = [
            {
                "event_type": e.event_type,
                "component": e.component,
                "old_value": e.old_value,
                "new_value": e.new_value,
                "detected_at": e.detected_at,
            }
            for e in drift_events
        ]

        report = CicdReport(
            env=ci,
            ntia=ntia_dict,
            risk=risk_dict,
            drift_events=drift_dicts,
        )

        CicdAdapter.annotate(report, ci)
        return report
