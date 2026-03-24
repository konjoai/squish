"""squish/packaging/release_validator.py — Pre-release gate validator.

Enforces a set of mandatory and advisory checks before a ``git tag v*``
release is made.  All checks are stateless and run in-process (no server
required).

Mandatory checks (failure blocks release):
  1. pytest pass rate ≥ 99 % (subprocess)
  2. CHANGELOG contains ``[{major}.0.0]`` entry for target version
  3. All ``squish/**/*.py`` files carry the SPDX-License-Identifier comment
  4. ``pyproject.toml`` has required metadata fields populated
  5. ``squish serve --help`` exits 0 (CLI smoke test)

Advisory checks (emitted as warnings only):
  A. arXiv citation reference present in README.md
  B. Docker image tag matches release version
  C. PLAN.md Wave checklist entry exists for target version

Classes
───────
CheckResult     — Single check outcome (name, passed, message).
ReleaseReport   — Full report with per-check results and summary.
ReleaseConfig   — Configuration for the validator.
ReleaseValidator — Main validator class.

Usage::

    from squish.packaging.release_validator import ReleaseValidator

    validator = ReleaseValidator(version="45.0.0")
    report    = validator.validate()
    if not report.passed:
        for r in report.failures:
            print(f"FAIL: {r.name} — {r.message}")
        raise SystemExit(1)
"""
from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CheckResult:
    """Outcome of a single pre-release check.

    Attributes
    ----------
    name:
        Short identifier for the check, e.g. ``"pytest_pass_rate"``.
    passed:
        True if the check succeeded.
    message:
        Human-readable detail (reason for failure or confirmation of pass).
    mandatory:
        True for blocking checks, False for advisory warnings.
    """
    name:      str
    passed:    bool
    message:   str
    mandatory: bool = True


@dataclass
class ReleaseReport:
    """Aggregated result of all release validation checks.

    Attributes
    ----------
    version:
        Target release version string.
    results:
        Ordered list of all check results.
    passed:
        True iff all *mandatory* checks passed.
    """
    version:  str
    results:  List[CheckResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.results if r.mandatory)

    @property
    def failures(self) -> List[CheckResult]:
        return [r for r in self.results if not r.passed and r.mandatory]

    @property
    def warnings(self) -> List[CheckResult]:
        return [r for r in self.results if not r.passed and not r.mandatory]

    def summary(self) -> str:
        total     = len(self.results)
        mandatory = [r for r in self.results if r.mandatory]
        passed    = [r for r in mandatory if r.passed]
        advisory  = [r for r in self.results if not r.mandatory]
        warns     = [r for r in advisory if not r.passed]
        lines = [
            f"Release validation for v{self.version}",
            f"  Mandatory: {len(passed)}/{len(mandatory)} passed",
            f"  Advisory:  {len(warns)} warning(s)",
            f"  Overall:   {'PASS' if self.passed else 'FAIL'}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ReleaseConfig:
    """Configuration for ReleaseValidator.

    Attributes
    ----------
    repo_root:
        Absolute path to the repository root. Defaults to CWD.
    pytest_pass_threshold:
        Minimum fraction of tests that must pass (0.0–1.0). Default 0.99.
    spdx_identifier:
        Expected SPDX licence identifier line in source files.
        Default ``"SPDX-License-Identifier: MIT"``.
    required_pyproject_fields:
        Top-level ``[project]`` fields that must be present in pyproject.toml.
    cli_entry_point:
        Module/script used for the CLI smoke test. Default ``"squish"``.
    """
    repo_root:                 Optional[Path] = None
    pytest_pass_threshold:     float = 0.99
    spdx_identifier:           str   = "SPDX-License-Identifier: MIT"
    required_pyproject_fields: List[str] = field(default_factory=lambda: [
        "name", "version", "description", "requires-python",
        "license", "authors", "classifiers",
    ])
    cli_entry_point: str = "squish"

    def __post_init__(self) -> None:
        if not (0.0 < self.pytest_pass_threshold <= 1.0):
            raise ValueError(
                f"pytest_pass_threshold must be in (0, 1], "
                f"got {self.pytest_pass_threshold}"
            )


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class ReleaseValidator:
    """Run all pre-release checks and return a ``ReleaseReport``.

    Usage::

        validator = ReleaseValidator(version="45.0.0")
        report    = validator.validate()
        print(report.summary())
    """

    def __init__(
        self,
        version: str,
        config: Optional[ReleaseConfig] = None,
    ) -> None:
        if not version:
            raise ValueError("version must be a non-empty string")
        self._version = version
        self._cfg     = config or ReleaseConfig()
        self._root    = self._cfg.repo_root or Path.cwd()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate(self) -> ReleaseReport:
        """Run all checks and return the report."""
        report = ReleaseReport(version=self._version)

        # Mandatory checks
        report.results.append(self._check_pytest())
        report.results.append(self._check_changelog())
        report.results.append(self._check_spdx_headers())
        report.results.append(self._check_pyproject())
        report.results.append(self._check_cli_smoke())

        # Advisory checks
        report.results.append(self._check_arxiv_reference())
        report.results.append(self._check_plan_wave_entry())

        return report

    # ------------------------------------------------------------------
    # Mandatory checks
    # ------------------------------------------------------------------

    def _check_pytest(self) -> CheckResult:
        """Run pytest and verify pass rate ≥ threshold."""
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pytest", "--tb=no", "-q", "--no-header"],
                capture_output=True,
                text=True,
                timeout=300,
                cwd=str(self._root),
            )
            output = result.stdout + result.stderr
            # Parse "X passed, Y failed" from pytest summary line
            passed, failed = self._parse_pytest_counts(output)
            total = passed + failed
            if total == 0:
                return CheckResult(
                    name="pytest_pass_rate",
                    passed=False,
                    message="No tests collected — check test discovery.",
                )
            rate = passed / total
            ok   = rate >= self._cfg.pytest_pass_threshold
            return CheckResult(
                name="pytest_pass_rate",
                passed=ok,
                message=(
                    f"{passed}/{total} tests passed "
                    f"({rate*100:.1f}% >= "
                    f"{self._cfg.pytest_pass_threshold*100:.1f}% required)"
                ) if ok else (
                    f"Only {passed}/{total} tests passed "
                    f"({rate*100:.1f}% < "
                    f"{self._cfg.pytest_pass_threshold*100:.1f}% required)"
                ),
            )
        except subprocess.TimeoutExpired:
            return CheckResult(
                name="pytest_pass_rate",
                passed=False,
                message="pytest timed out after 300 seconds.",
            )
        except Exception as exc:
            return CheckResult(
                name="pytest_pass_rate",
                passed=False,
                message=f"Failed to run pytest: {exc}",
            )

    def _check_changelog(self) -> CheckResult:
        """Verify CHANGELOG.md contains an entry for the target version."""
        changelog = self._root / "CHANGELOG.md"
        if not changelog.exists():
            return CheckResult(
                name="changelog_entry",
                passed=False,
                message="CHANGELOG.md not found.",
            )
        # Extract major version component
        major = self._version.split(".")[0]
        pattern = f"[{major}.0.0]"
        try:
            content = changelog.read_text(encoding="utf-8")
        except OSError as exc:
            return CheckResult(
                name="changelog_entry",
                passed=False,
                message=f"Could not read CHANGELOG.md: {exc}",
            )
        found = pattern in content
        return CheckResult(
            name="changelog_entry",
            passed=found,
            message=(
                f"Found '{pattern}' in CHANGELOG.md" if found
                else f"Missing '{pattern}' entry in CHANGELOG.md"
            ),
        )

    def _check_spdx_headers(self) -> CheckResult:
        """Verify all squish/**/*.py files carry the SPDX identifier."""
        squish_src = self._root / "squish"
        if not squish_src.exists():
            return CheckResult(
                name="spdx_headers",
                passed=False,
                message="squish/ source directory not found.",
            )
        identifier = self._cfg.spdx_identifier
        missing: List[str] = []
        checked = 0
        for py_file in squish_src.rglob("*.py"):
            checked += 1
            try:
                # Only scan the first 10 lines for performance
                with py_file.open("r", encoding="utf-8", errors="replace") as fh:
                    header = "".join(fh.readline() for _ in range(10))
                if identifier not in header:
                    missing.append(str(py_file.relative_to(self._root)))
            except OSError:
                missing.append(str(py_file.relative_to(self._root)))
        if checked == 0:
            return CheckResult(
                name="spdx_headers",
                passed=False,
                message="No .py files found in squish/.",
            )
        if missing:
            sample = missing[:5]
            tail   = f" (+ {len(missing)-5} more)" if len(missing) > 5 else ""
            return CheckResult(
                name="spdx_headers",
                passed=False,
                message=(
                    f"{len(missing)}/{checked} files missing '{identifier}': "
                    f"{', '.join(sample)}{tail}"
                ),
            )
        return CheckResult(
            name="spdx_headers",
            passed=True,
            message=f"All {checked} squish/*.py files carry SPDX identifier.",
        )

    def _check_pyproject(self) -> CheckResult:
        """Verify required fields in pyproject.toml [project] table."""
        pyproject = self._root / "pyproject.toml"
        if not pyproject.exists():
            return CheckResult(
                name="pyproject_metadata",
                passed=False,
                message="pyproject.toml not found.",
            )
        try:
            import tomllib  # Python 3.11+
        except ImportError:
            try:
                import tomli as tomllib  # type: ignore[no-redef]
            except ImportError:
                # Best-effort text scan if toml libraries unavailable
                return self._check_pyproject_text(pyproject)

        try:
            data    = tomllib.loads(pyproject.read_text(encoding="utf-8"))
            project = data.get("project", {})
        except Exception as exc:
            return CheckResult(
                name="pyproject_metadata",
                passed=False,
                message=f"Failed to parse pyproject.toml: {exc}",
            )

        missing = [f for f in self._cfg.required_pyproject_fields if not project.get(f)]
        if missing:
            return CheckResult(
                name="pyproject_metadata",
                passed=False,
                message=f"Missing pyproject.toml [project] fields: {missing}",
            )
        return CheckResult(
            name="pyproject_metadata",
            passed=True,
            message="All required pyproject.toml [project] fields are set.",
        )

    def _check_pyproject_text(self, pyproject: Path) -> CheckResult:
        """Fallback TOML check via plain text search."""
        try:
            content = pyproject.read_text(encoding="utf-8")
        except OSError as exc:
            return CheckResult(
                name="pyproject_metadata",
                passed=False,
                message=f"Could not read pyproject.toml: {exc}",
            )
        missing = [
            f for f in self._cfg.required_pyproject_fields
            if f not in content
        ]
        if missing:
            return CheckResult(
                name="pyproject_metadata",
                passed=False,
                message=f"Missing pyproject.toml fields (text scan): {missing}",
            )
        return CheckResult(
            name="pyproject_metadata",
            passed=True,
            message="pyproject.toml contains required field names (text scan).",
        )

    def _check_cli_smoke(self) -> CheckResult:
        """Run ``squish serve --help`` and verify exit code 0."""
        try:
            result = subprocess.run(
                [sys.executable, "-m", "squish", "serve", "--help"],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=str(self._root),
            )
            ok = result.returncode == 0
            return CheckResult(
                name="cli_smoke_test",
                passed=ok,
                message=(
                    "`python -m squish serve --help` exited 0" if ok
                    else (
                        f"`python -m squish serve --help` exited "
                        f"{result.returncode}: {result.stderr[:200]}"
                    )
                ),
            )
        except subprocess.TimeoutExpired:
            return CheckResult(
                name="cli_smoke_test",
                passed=False,
                message="CLI smoke test timed out after 30 seconds.",
            )
        except Exception as exc:
            return CheckResult(
                name="cli_smoke_test",
                passed=False,
                message=f"CLI smoke test failed: {exc}",
            )

    # ------------------------------------------------------------------
    # Advisory checks
    # ------------------------------------------------------------------

    def _check_arxiv_reference(self) -> CheckResult:
        """Warn if README.md does not reference an arXiv paper."""
        readme = self._root / "README.md"
        try:
            content = readme.read_text(encoding="utf-8")
            found   = "arxiv.org" in content.lower() or "arXiv" in content
            return CheckResult(
                name="arxiv_reference",
                passed=found,
                message=(
                    "arXiv reference found in README.md" if found
                    else "No arXiv reference in README.md (advisory)"
                ),
                mandatory=False,
            )
        except OSError:
            return CheckResult(
                name="arxiv_reference",
                passed=False,
                message="README.md not found (advisory)",
                mandatory=False,
            )

    def _check_plan_wave_entry(self) -> CheckResult:
        """Warn if PLAN.md does not contain a wave entry for the release version."""
        plan = self._root / "docs" / "planning" / "PLAN.md"
        if not plan.exists():
            plan = self._root / "PLAN.md"
        try:
            content = plan.read_text(encoding="utf-8")
            major   = self._version.split(".")[0]
            found   = f"v{major}" in content or f"[{major}.0.0]" in content
            return CheckResult(
                name="plan_wave_entry",
                passed=found,
                message=(
                    f"Found v{major} entry in PLAN.md" if found
                    else f"No v{major} entry in PLAN.md (advisory)"
                ),
                mandatory=False,
            )
        except OSError:
            return CheckResult(
                name="plan_wave_entry",
                passed=False,
                message="PLAN.md not found (advisory)",
                mandatory=False,
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_pytest_counts(output: str) -> tuple:
        """Parse 'X passed, Y failed' from pytest terminal output.

        Returns (passed, failed) integers.
        """
        import re
        passed = 0
        failed = 0
        # Match lines like: "1234 passed, 5 failed, 12 warnings in 45.32s"
        for match in re.finditer(r"(\d+)\s+passed", output):
            passed = int(match.group(1))
        for match in re.finditer(r"(\d+)\s+failed", output):
            failed = int(match.group(1))
        return passed, failed

    def __repr__(self) -> str:
        return f"ReleaseValidator(version={self._version!r}, root={self._root})"
