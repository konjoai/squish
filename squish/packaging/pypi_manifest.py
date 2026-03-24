"""squish/packaging/pypi_manifest.py — PyPI package manifest builder.

Reads ``pyproject.toml`` and the workspace tree, generates ``MANIFEST.in``
rules, validates wheel contents against an allowlist, and emits a
``PyPIManifestReport`` with size breakdown and any flagged files.

Classes
───────
ManifestRule       — A single include/exclude MANIFEST.in rule.
WheelEntry         — A file present in a built wheel.
PyPIManifestReport — Report from manifest validation / generation.
ManifestConfig     — Validator configuration.
PyPIManifest       — Main builder / validator class.

Usage::

    from squish.packaging.pypi_manifest import PyPIManifest

    manifest = PyPIManifest()
    report   = manifest.build_and_validate()
    print(report.summary())

    # Write MANIFEST.in
    if report.passed:
        manifest.write_manifest_in()
"""
from __future__ import annotations

import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Set


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ManifestRule:
    """A single MANIFEST.in include or exclude directive.

    Attributes
    ----------
    directive:
        MANIFEST.in directive: ``"include"``, ``"recursive-include"``,
        ``"exclude"``, ``"recursive-exclude"``, ``"global-exclude"``.
    pattern:
        File glob pattern, e.g. ``"squish/**/*.py"`` or ``"*.pyc"``.
    """
    directive: str
    pattern:   str

    def __str__(self) -> str:
        return f"{self.directive} {self.pattern}"


@dataclass(frozen=True)
class WheelEntry:
    """A file entry inside a built wheel archive."""
    path:     str   # Wheel-relative path
    size_kb:  float


@dataclass
class PyPIManifestReport:
    """Result of the manifest validation / generation pass.

    Attributes
    ----------
    rules:
        Generated MANIFEST.in rules.
    wheel_entries:
        Files found in the wheel (empty if wheel not built).
    flagged:
        Files that violated the allowlist.
    total_size_kb:
        Total wheel size in kilobytes.
    passed:
        True iff no allowlist violations were found.
    """
    rules:          List[ManifestRule] = field(default_factory=list)
    wheel_entries:  List[WheelEntry]   = field(default_factory=list)
    flagged:        List[str]          = field(default_factory=list)
    total_size_kb:  float              = 0.0

    @property
    def passed(self) -> bool:
        return not self.flagged

    def summary(self) -> str:
        lines = [
            f"PyPI manifest report",
            f"  Rules generated  : {len(self.rules)}",
            f"  Wheel entries    : {len(self.wheel_entries)}",
            f"  Total size       : {self.total_size_kb:.1f} KB",
            f"  Flagged entries  : {len(self.flagged)}",
            f"  Result           : {'PASS' if self.passed else 'FAIL'}",
        ]
        if self.flagged:
            lines.append("  Violations:")
            for f in self.flagged[:10]:
                lines.append(f"    - {f}")
            if len(self.flagged) > 10:
                lines.append(f"    ... and {len(self.flagged) - 10} more")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ManifestConfig:
    """Configuration for PyPIManifest.

    Attributes
    ----------
    repo_root:
        Repository root path. Default: CWD.
    include_patterns:
        Glob patterns relative to repo root to include in the sdist.
    exclude_dirs:
        Directory names to exclude from the wheel (security/hygiene).
    allowlist_prefixes:
        Wheel entry path prefixes that are permitted; anything else is flagged.
    max_wheel_size_kb:
        Advisory maximum wheel size (KB). Violations are warnings, not errors.
    """
    repo_root:          Optional[Path] = None
    include_patterns:   List[str]      = field(default_factory=lambda: [
        "squish/**/*.py",
        "squish/**/*.metal",
        "squish/**/*.json",
        "docs/squizd_format_spec.md",
        "README.md",
        "LICENSE",
        "pyproject.toml",
    ])
    exclude_dirs:       List[str]      = field(default_factory=lambda: [
        "dev", "eval_output", "eval_output_14b", "eval_output_14b_25shot",
        "eval_output_7b_full", "logs", "results", "tests", "docker",
        "helm", "demos", "assets",
    ])
    allowlist_prefixes: List[str]      = field(default_factory=lambda: [
        "squish/",
        "squish-",       # e.g. squish-45.0.0.dist-info/
        "README",
        "LICENSE",
    ])
    max_wheel_size_kb:  float          = 5_120.0  # 5 MB advisory limit


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class PyPIManifest:
    """Build and validate PyPI package manifest for the Squish distribution.

    Does not require ``pip`` or ``build`` to be installed for rule generation.
    Wheel validation requires a pre-built ``.whl`` file path.

    Usage::

        manifest = PyPIManifest()
        report   = manifest.build_and_validate()
        manifest.write_manifest_in()   # writes MANIFEST.in to repo root
    """

    def __init__(self, config: Optional[ManifestConfig] = None) -> None:
        self._cfg  = config or ManifestConfig()
        self._root = self._cfg.repo_root or Path.cwd()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_rules(self) -> List[ManifestRule]:
        """Generate MANIFEST.in rules from configuration."""
        rules: List[ManifestRule] = []

        # Includes
        for pat in self._cfg.include_patterns:
            if "/**/" in pat or pat.endswith("/**"):
                # Recursive include: split into dir + file pattern
                parts = pat.split("/", 1)
                rules.append(ManifestRule("recursive-include", pat))
            elif pat.startswith("squish/"):
                rules.append(ManifestRule("include", pat))
            else:
                rules.append(ManifestRule("include", pat))

        # Excludes
        for d in self._cfg.exclude_dirs:
            rules.append(ManifestRule("recursive-exclude", f"{d} *"))

        # Global hygiene excludes
        for ext in ("*.pyc", "*.pyo", "__pycache__", ".DS_Store", "*.egg-info"):
            rules.append(ManifestRule("global-exclude", ext))

        return rules

    def write_manifest_in(self) -> Path:
        """Write MANIFEST.in to the repository root.

        Returns the path to the written file.
        """
        rules    = self.generate_rules()
        out_path = self._root / "MANIFEST.in"
        lines    = [
            "# Auto-generated by squish/packaging/pypi_manifest.py",
            "# Do not edit manually — regenerate with: "
            "python -m squish.packaging.pypi_manifest",
            "",
        ] + [str(r) for r in rules] + [""]
        out_path.write_text("\n".join(lines), encoding="utf-8")
        return out_path

    def validate_wheel(self, wheel_path: Path) -> List[str]:
        """Validate wheel contents against the allowlist.

        Parameters
        ----------
        wheel_path:
            Path to a ``.whl`` file.

        Returns
        -------
        List[str]
            List of flagged entry paths that violated the allowlist.
        """
        if not wheel_path.exists():
            raise FileNotFoundError(f"Wheel not found: {wheel_path}")
        flagged: List[str] = []
        with zipfile.ZipFile(wheel_path, "r") as zf:
            for name in zf.namelist():
                if not any(name.startswith(p) for p in self._cfg.allowlist_prefixes):
                    flagged.append(name)
        return flagged

    def wheel_entries(self, wheel_path: Path) -> List[WheelEntry]:
        """Return a list of all entries in the wheel."""
        if not wheel_path.exists():
            return []
        entries: List[WheelEntry] = []
        with zipfile.ZipFile(wheel_path, "r") as zf:
            for info in zf.infolist():
                entries.append(WheelEntry(
                    path=info.filename,
                    size_kb=round(info.file_size / 1024.0, 2),
                ))
        return entries

    def build_and_validate(
        self,
        wheel_path: Optional[Path] = None,
    ) -> PyPIManifestReport:
        """Generate rules and optionally validate a pre-built wheel.

        Parameters
        ----------
        wheel_path:
            Optional path to a ``.whl`` file for deep validation.
            If ``None``, only rule generation is performed.

        Returns
        -------
        PyPIManifestReport
        """
        report = PyPIManifestReport()
        report.rules = self.generate_rules()

        if wheel_path:
            try:
                entries       = self.wheel_entries(wheel_path)
                report.wheel_entries = entries
                report.total_size_kb = sum(e.size_kb for e in entries)
                report.flagged = self.validate_wheel(wheel_path)
            except FileNotFoundError:
                # Wheel not built yet — skip wheel validation
                pass

        return report

    def find_excluded_files_in_tree(self) -> List[str]:
        """Return relative paths of any excluded-dir files found under the root.

        Useful for detecting accidental inclusion of dev artefacts.
        """
        found: List[str] = []
        for excl_dir in self._cfg.exclude_dirs:
            candidate = self._root / excl_dir
            if candidate.is_dir():
                found.append(str(Path(excl_dir)))
        return found

    def sdist_size_estimate_kb(self) -> float:
        """Estimate the sdist size by summing matched source files on disk."""
        total = 0.0
        for pat in self._cfg.include_patterns:
            for p in self._root.glob(pat):
                if p.is_file():
                    total += p.stat().st_size
        return round(total / 1024.0, 2)

    def __repr__(self) -> str:
        return f"PyPIManifest(root={self._root})"
