"""squish/squash/cli.py — Standalone ``squash`` CLI entry point.

Provides the ``squash attest`` sub-command that CI/CD integrations call:

    squash attest ./my-model --policy eu-ai-act --policy enterprise-strict

Exit codes (per project CLI standard):
    0  Success — attestation passed
    1  User / input error (bad path, unknown policy, missing flag)
    2  Runtime error (I/O failure, scan error, attestation violation)

Usage::

    squash attest MODEL_PATH [options]

Options::

    --policy, -p     Policy name to evaluate (repeatable, default: enterprise-strict)
    --output-dir     Artifact output directory (default: model dir)
    --sign           Sign the CycloneDX BOM via Sigstore keyless signing
    --fail-on-violation  Exit 2 if any policy error-severity finding is raised
    --skip-scan      Skip the security scanner
    --json-result    Path to write the master attestation record as JSON
    --model-id       Override the model ID in the SBOM
    --hf-repo        HuggingFace repository ID for provenance metadata
    --quant-format   Quantization format label (e.g. INT4, BF16)
    --quiet, -q      Suppress informational output (errors still go to stderr)
    --help           Show this message and exit

"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="squash",
        description="AI-SBOM attestation for ML models (Squish Squash)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  squash attest ./llama-3.1-8b-q4 --policy eu-ai-act\n"
            "  squash attest ./model --sign --fail-on-violation --json-result ./result.json\n"
            "  squash policies              # list available policy templates\n"
        ),
    )
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress info output")

    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    # ── squash attest ──────────────────────────────────────────────────────────
    attest = sub.add_parser("attest", help="Run full attestation pipeline on a model artifact")
    attest.add_argument(
        "model_path",
        help="Path to model directory or file (e.g. ./llama-3.1-8b-q4 or ./model.gguf)",
    )
    attest.add_argument(
        "--policy", "-p",
        dest="policies",
        action="append",
        default=[],
        metavar="POLICY",
        help="Policy name to evaluate (repeatable). Default: enterprise-strict",
    )
    attest.add_argument("--output-dir", default=None, help="Artifact destination directory")
    attest.add_argument("--sign", action="store_true", help="Sign BOM via Sigstore keyless")
    attest.add_argument(
        "--fail-on-violation",
        action="store_true",
        help="Exit 2 if any error-severity policy finding exists or scan is unsafe",
    )
    attest.add_argument("--skip-scan", action="store_true", help="Skip security scanner")
    attest.add_argument(
        "--json-result",
        default=None,
        metavar="PATH",
        help="Write master attestation record JSON to this path",
    )
    attest.add_argument("--model-id", default="", help="Override model ID in SBOM")
    attest.add_argument("--hf-repo", default="", help="HuggingFace repo ID for provenance")
    attest.add_argument(
        "--quant-format",
        default="unknown",
        help="Quantization format label (e.g. INT4, BF16)",
    )

    # ── squash policies ────────────────────────────────────────────────────────
    policies_cmd = sub.add_parser("policies", help="List available built-in policy templates")
    policies_cmd.add_argument(
        "--validate",
        metavar="PATH",
        default=None,
        help="Validate a custom YAML rules file (exit 0 = valid, 1 = user error, 2 = invalid rules)",
    )

    # ── squash scan ────────────────────────────────────────────────────────────
    scan_cmd = sub.add_parser("scan", help="Run security scanner only (no SBOM generation)")
    scan_cmd.add_argument("model_path", help="Path to model directory or file")
    scan_cmd.add_argument("--json-result", default=None, metavar="PATH")
    scan_cmd.add_argument(
        "--sarif",
        default=None,
        metavar="PATH",
        help="Write SARIF 2.1.0 output to PATH",
    )
    scan_cmd.add_argument(
        "--exit-2-on-unsafe",
        action="store_true",
        default=False,
        help="Exit 2 on critical/high findings; exit 1 on other unsafe statuses",
    )

    # ── squash diff ───────────────────────────────────────────────────────────
    diff_cmd = sub.add_parser(
        "diff",
        help="Compare two CycloneDX SBOM snapshots and report differences",
    )
    diff_cmd.add_argument("sbom_a", metavar="SBOM_A", help="Older (baseline) SBOM JSON file")
    diff_cmd.add_argument("sbom_b", metavar="SBOM_B", help="Newer SBOM JSON file")
    diff_cmd.add_argument(
        "--exit-1-on-regression",
        action="store_true",
        default=False,
        help="Exit 1 when new vulnerabilities are introduced or policy status worsens",
    )

    # ── squash verify ──────────────────────────────────────────────────────────
    verify_cmd = sub.add_parser(
        "verify",
        help="Verify the Sigstore bundle for a model's CycloneDX BOM",
    )
    verify_cmd.add_argument(
        "model_path",
        help="Path to model directory (must contain cyclonedx-mlbom.json)",
    )
    verify_cmd.add_argument(
        "--bundle",
        default=None,
        metavar="PATH",
        help="Explicit path to the .sig.json bundle (default: <bom>.sig.json)",
    )
    verify_cmd.add_argument(
        "--strict",
        action="store_true",
        default=False,
        help="Exit 2 when no bundle is found (treat unsigned BOMs as failures)",
    )

    # ── squash report ──────────────────────────────────────────────────────────
    report_cmd = sub.add_parser(
        "report",
        help="Generate an HTML or JSON compliance report from attestation artifacts",
        description="squash report MODEL_DIR  # writes squash-report.html into model dir",
    )
    report_cmd.add_argument(
        "model_path",
        help="Path to model directory containing attestation artifacts",
    )
    report_cmd.add_argument(
        "--output",
        default=None,
        metavar="PATH",
        help="Output file path (default: <model_dir>/squash-report.html)",
    )
    report_cmd.add_argument(
        "--format",
        choices=["html", "json"],
        default="html",
        help="Output format (default: html)",
    )

    # ── squash vex ─────────────────────────────────────────────────────────────
    vex_cmd = sub.add_parser(
        "vex",
        help="VEX feed cache management",
    )
    vex_sub = vex_cmd.add_subparsers(dest="vex_command")
    vex_update = vex_sub.add_parser("update", help="Refresh the local VEX feed cache")
    vex_update.add_argument(
        "--url",
        default=None,
        help="Override VEX feed URL (default: SQUASH_VEX_URL env or built-in)",
    )
    vex_update.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="HTTP timeout in seconds (default: 10)",
    )
    vex_sub.add_parser("status", help="Show VEX cache status and freshness")

    # ── squash attest-composed ────────────────────────────────────────────────
    ac_cmd = sub.add_parser(
        "attest-composed",
        help="Attest multiple models and produce a parent composite BOM",
        description="squash attest-composed MODEL_A MODEL_B ...  [--output-dir DIR]",
    )
    ac_cmd.add_argument(
        "model_paths",
        nargs="+",
        metavar="MODEL_PATH",
        help="Two or more model directories to attest",
    )
    ac_cmd.add_argument(
        "--output-dir",
        default=None,
        metavar="DIR",
        help="Write parent BOM and component results here (default: first model dir)",
    )
    ac_cmd.add_argument(
        "--policy",
        dest="policies",
        action="append",
        default=None,
        metavar="NAME",
        help="Policy name(s) to evaluate (repeatable; default: enterprise-strict)",
    )
    ac_cmd.add_argument(
        "--sign",
        action="store_true",
        default=False,
        help="Sign each component BOM with Sigstore after attestation",
    )

    # ── squash push ───────────────────────────────────────────────────────────
    push_cmd = sub.add_parser(
        "push",
        help="Push a CycloneDX SBOM to a supported registry (Dependency-Track, GUAC, Squash)",
        description="squash push MODEL_DIR --registry-url URL  [options]",
    )
    push_cmd.add_argument(
        "model_path",
        help="Model directory containing cyclonedx-mlbom.json",
    )
    push_cmd.add_argument(
        "--registry-url",
        required=True,
        metavar="URL",
        help="Registry endpoint URL",
    )
    push_cmd.add_argument(
        "--api-key",
        default=None,
        metavar="KEY",
        help="API key or token (or set SQUASH_REGISTRY_KEY env var)",
    )
    push_cmd.add_argument(
        "--registry-type",
        choices=["dtrack", "guac", "squash"],
        default="dtrack",
        help="Registry protocol (default: dtrack)",
    )

    # ── Wave 20 — NTIA minimum elements check ─────────────────────────────────
    ntia_cmd = sub.add_parser(
        "ntia-check",
        help="Validate NTIA minimum elements in a CycloneDX BOM",
        description=(
            "Check a CycloneDX BOM for the NTIA Minimum Elements for SBOM "
            "compliance (Nov 2021).\n\n"
            "Example: squash ntia-check model/cyclonedx-mlbom.json"
        ),
    )
    ntia_cmd.add_argument("bom_path", help="Path to the CycloneDX BOM JSON file")
    ntia_cmd.add_argument(
        "--strict",
        action="store_true",
        help="Require non-empty dependsOn fields (stricter NTIA compliance)",
    )
    ntia_cmd.add_argument("--quiet", action="store_true", help="Suppress non-error output")

    # ── Wave 21 — SLSA provenance attestation ─────────────────────────────────
    slsa_cmd = sub.add_parser(
        "slsa-attest",
        help="Generate SLSA provenance statement for a model directory",
        description=(
            "Build a SLSA 1.0 Build Provenance statement for the artefacts in "
            "MODEL_DIR and (optionally) sign it.\n\n"
            "Example: squash slsa-attest ./my-model --level 2"
        ),
    )
    slsa_cmd.add_argument("model_dir", help="Path to the squash model directory")
    slsa_cmd.add_argument(
        "--level",
        type=int,
        choices=[1, 2, 3],
        default=1,
        help="SLSA build track level (default: 1)",
    )
    slsa_cmd.add_argument(
        "--builder-id",
        default="https://squish.local/squash/builder",
        help="URI identifying the build system",
    )
    slsa_cmd.add_argument("--sign", action="store_true", help="Force signing even at L1")
    slsa_cmd.add_argument("--quiet", action="store_true", help="Suppress non-error output")

    # ── Wave 22 — BOM merge ────────────────────────────────────────────────────
    merge_cmd = sub.add_parser(
        "merge",
        help="Merge multiple CycloneDX BOMs into one canonical BOM",
        description=(
            "Deduplicate components by PURL and union vulnerabilities across "
            "multiple CycloneDX BOMs.\n\n"
            "Example: squash merge a/cyclonedx-mlbom.json b/cyclonedx-mlbom.json "
            "--output merged/cyclonedx-mlbom.json"
        ),
    )
    merge_cmd.add_argument(
        "bom_paths", nargs="+", help="Two or more CycloneDX BOM JSON files to merge"
    )
    merge_cmd.add_argument(
        "--output", required=True, help="Destination path for the merged BOM"
    )
    merge_cmd.add_argument("--quiet", action="store_true", help="Suppress non-error output")

    # ── Wave 23 — AI risk assessment ──────────────────────────────────────────
    risk_cmd = sub.add_parser(
        "risk-assess",
        help="Assess AI risk per EU AI Act and/or NIST AI RMF",
        description=(
            "Evaluate the BOM in MODEL_DIR against the EU AI Act (2024/1689) "
            "risk tiers and the NIST AI Risk Management Framework.\n\n"
            "Example: squash risk-assess ./my-model --framework eu-ai-act"
        ),
    )
    risk_cmd.add_argument("model_dir", help="Path to the squash model directory")
    risk_cmd.add_argument(
        "--framework",
        choices=["eu-ai-act", "nist-rmf", "both"],
        default="both",
        help="Risk framework to run (default: both)",
    )
    risk_cmd.add_argument("--quiet", action="store_true", help="Suppress non-error output")

    # ── Wave 24 — Drift monitoring ────────────────────────────────────────────
    monitor_cmd = sub.add_parser(
        "monitor",
        help="Detect drift in a squash model directory",
        description=(
            "Snapshot the attestation state of MODEL_DIR and compare against a "
            "previous snapshot to detect BOM changes, new CVEs, or policy "
            "regressions.\n\n"
            "Example: squash monitor ./my-model --once"
        ),
    )
    monitor_cmd.add_argument("model_dir", help="Path to the squash model directory")
    monitor_cmd.add_argument(
        "--baseline",
        default=None,
        help="SHA-256 baseline snapshot string to compare against (omit to just snapshot)",
    )
    monitor_cmd.add_argument(
        "--interval",
        type=float,
        default=3600.0,
        help="Poll interval in seconds for continuous monitoring (default: 3600)",
    )
    monitor_cmd.add_argument(
        "--once",
        action="store_true",
        help="Snapshot once (or compare against --baseline) then exit",
    )
    monitor_cmd.add_argument("--quiet", action="store_true", help="Suppress non-error output")

    # ── Wave 25 — CI/CD integration ───────────────────────────────────────────
    ci_cmd = sub.add_parser(
        "ci-run",
        help="Run the full squash check pipeline in CI",
        description=(
            "Execute NTIA validation, AI risk assessment, and drift detection "
            "for MODEL_DIR, then emit native CI annotations.\n\n"
            "Example: squash ci-run ./my-model --report-format github"
        ),
    )
    ci_cmd.add_argument("model_dir", help="Path to the squash model directory")
    ci_cmd.add_argument(
        "--report-format",
        choices=["github", "jenkins", "gitlab", "text"],
        default="text",
        help="CI annotation format (default: text; auto-detected if not set)",
    )
    ci_cmd.add_argument("--quiet", action="store_true", help="Suppress non-error output")

    return parser


def _cmd_policies(args: argparse.Namespace, quiet: bool) -> int:
    try:
        from squish.squash.policy import AVAILABLE_POLICIES, PolicyRegistry
    except ImportError as e:
        print(f"squash is not installed: {e}", file=sys.stderr)
        return 2

    validate_path: str | None = getattr(args, "validate", None)

    if validate_path is not None:
        rules_path = Path(validate_path)
        if not rules_path.exists():
            print(f"error: path does not exist: {rules_path}", file=sys.stderr)
            return 1
        try:
            rules = PolicyRegistry.load_rules_from_yaml(rules_path)
        except ImportError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
        except (OSError, ValueError) as e:
            print(f"error loading rules: {e}", file=sys.stderr)
            return 1

        raw_errors = PolicyRegistry.validate_rules(rules)
        if raw_errors:
            if not quiet:
                print(f"✗ {len(raw_errors)} validation error(s):", file=sys.stderr)
                for err in raw_errors:
                    print(f"  {err}", file=sys.stderr)
            return 2

        if not quiet:
            print(f"✓ {len(rules)} rule(s) valid: {rules_path}")
        return 0

    if not quiet:
        print("Available policy templates:")
    for name in sorted(AVAILABLE_POLICIES):
        print(f"  {name}")
    return 0


def _cmd_scan(args: argparse.Namespace, quiet: bool) -> int:
    try:
        from squish.squash.scanner import ModelScanner
    except ImportError as e:
        print(f"squash is not installed: {e}", file=sys.stderr)
        return 2

    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"error: path does not exist: {model_path}", file=sys.stderr)
        return 1

    scan_dir = model_path if model_path.is_dir() else model_path.parent
    result = ModelScanner.scan_directory(scan_dir)

    if not quiet:
        icon = "✓" if result.is_safe else "✗"
        print(f"{icon} Scan {result.status}: {scan_dir}")
        for f in result.findings:
            print(f"  [{f.severity.upper()}] {f.title} — {f.detail}")

    if args.json_result:
        data = {
            "status": result.status,
            "is_safe": result.is_safe,
            "critical": result.critical_count,
            "high": result.high_count,
            "findings": [
                {"severity": f.severity, "title": f.title, "file": f.file_path}
                for f in result.findings
            ],
        }
        Path(args.json_result).write_text(json.dumps(data, indent=2))

    if args.sarif:
        try:
            from squish.squash.sarif import SarifBuilder
        except ImportError as e:  # pragma: no cover
            print(f"sarif export unavailable: {e}", file=sys.stderr)
            return 2
        SarifBuilder.write(result, Path(args.sarif))
        if not quiet:
            print(f"SARIF written to {args.sarif}")

    if getattr(args, "exit_2_on_unsafe", False):
        if result.critical_count > 0 or result.high_count > 0:
            return 2
        if not result.is_safe:
            return 1
        return 0

    return 0 if result.is_safe else 2


def _cmd_diff(args: argparse.Namespace, quiet: bool) -> int:
    try:
        from squish.squash.sbom_builder import SbomDiff
    except ImportError as e:
        print(f"squash is not installed: {e}", file=sys.stderr)
        return 2

    path_a = Path(args.sbom_a)
    path_b = Path(args.sbom_b)
    for p in (path_a, path_b):
        if not p.exists():
            print(f"error: path does not exist: {p}", file=sys.stderr)
            return 1

    try:
        bom_a = json.loads(path_a.read_text(encoding="utf-8"))
        bom_b = json.loads(path_b.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        print(f"error reading SBOM: {e}", file=sys.stderr)
        return 1

    diff = SbomDiff.compare(bom_a, bom_b)

    if not quiet:
        print(f"hash changed:          {diff.hash_changed}")
        print(f"score delta:           {diff.score_delta}")
        print(f"policy status changed: {diff.policy_status_changed}")
        if diff.new_findings:
            print(f"new findings ({len(diff.new_findings)}):")
            for fid in diff.new_findings:
                print(f"  + {fid}")
        if diff.resolved_findings:
            print(f"resolved findings ({len(diff.resolved_findings)}):")
            for fid in diff.resolved_findings:
                print(f"  - {fid}")
        if diff.metadata_changes:
            print("metadata changes:")
            for key, (old, new) in diff.metadata_changes.items():
                print(f"  {key}: {old!r} → {new!r}")

    if getattr(args, "exit_1_on_regression", False) and diff.has_regressions:
        return 1
    return 0


def _cmd_verify(args: argparse.Namespace, quiet: bool) -> int:
    try:
        from squish.squash.oms_signer import OmsVerifier
    except ImportError as e:
        print(f"squash is not installed: {e}", file=sys.stderr)
        return 2

    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"error: path does not exist: {model_path}", file=sys.stderr)
        return 1

    bom_path = model_path / "cyclonedx-mlbom.json" if model_path.is_dir() else model_path
    if not bom_path.exists():
        print(f"error: CycloneDX BOM not found: {bom_path}", file=sys.stderr)
        return 1

    bundle_path = Path(args.bundle) if args.bundle else None
    result = OmsVerifier.verify(bom_path, bundle_path)

    if result is None:
        if args.strict:
            if not quiet:
                print("✗ no bundle found (strict mode)", file=sys.stderr)
            return 2
        if not quiet:
            print("— verification skipped (no bundle)")
        return 0

    if result:
        if not quiet:
            print(f"✓ verified: {bom_path}")
        return 0

    print(f"✗ verification FAILED: {bom_path}", file=sys.stderr)
    return 2


def _cmd_attest(args: argparse.Namespace, quiet: bool) -> int:
    try:
        from squish.squash.attest import (
            AttestConfig,
            AttestPipeline,
            AttestationViolationError,
        )
    except ImportError as e:
        print(f"squash is not installed: {e}", file=sys.stderr)
        return 2

    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"error: path does not exist: {model_path}", file=sys.stderr)
        return 1

    policies = args.policies if args.policies else ["enterprise-strict"]

    config = AttestConfig(
        model_path=model_path,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        model_id=args.model_id,
        hf_repo=args.hf_repo,
        quant_format=args.quant_format,
        policies=policies,
        sign=args.sign,
        fail_on_violation=False,  # handle ourselves below for clean exit codes
        skip_scan=args.skip_scan,
    )

    try:
        result = AttestPipeline.run(config)
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"runtime error: {e}", file=sys.stderr)
        return 2

    if not quiet:
        icon = "✓" if result.passed else "✗"
        print(f"{icon} {result.summary()}")
        if result.cyclonedx_path:
            print(f"   CycloneDX : {result.cyclonedx_path}")
        if result.spdx_json_path:
            print(f"   SPDX JSON : {result.spdx_json_path}")
        if result.master_record_path:
            print(f"   Master    : {result.master_record_path}")
        if result.signature_path:
            print(f"   Signature : {result.signature_path}")

    if args.json_result and result.master_record_path and result.master_record_path.exists():
        import shutil
        shutil.copy2(result.master_record_path, args.json_result)

    if args.fail_on_violation and not result.passed:
        if not quiet:
            print("error: attestation failed (fail-on-violation set)", file=sys.stderr)
        return 2

    return 0 if result.passed else 2


# ────────────────────────────────────────────────────────────────────────────
# Wave 15  — HTML / JSON compliance report
# ────────────────────────────────────────────────────────────────────────────

def _cmd_report(args: argparse.Namespace, quiet: bool) -> int:
    try:
        from squish.squash.report import ComplianceReporter
    except ImportError as e:
        print(f"squash is not installed: {e}", file=sys.stderr)
        return 2

    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"error: path does not exist: {model_path}", file=sys.stderr)
        return 1

    output = Path(args.output) if args.output else None
    fmt: str = getattr(args, "format", "html")

    if fmt == "json":
        # Emit a raw JSON summary of all artifacts (no HTML rendering)
        import json as _json
        from squish.squash.report import _load_artifacts  # type: ignore[attr-defined]
        ctx = _load_artifacts(model_path)
        payload = {
            "model_dir": str(ctx["model_dir"]),
            "has_attest": ctx.get("attest") is not None,
            "has_cdx": ctx.get("cdx") is not None,
            "has_scan": ctx.get("scan") is not None,
            "has_vex": ctx.get("vex") is not None,
            "policy_count": len(ctx.get("policies", {})),
            "bundle_present": ctx.get("bundle_present", False),
        }
        dest = output or (model_path / "squash-report.json")
        dest.write_text(_json.dumps(payload, indent=2), encoding="utf-8")
        if not quiet:
            print(f"Report written to {dest}")
        return 0

    try:
        dest = ComplianceReporter.write(model_path, output)
    except Exception as e:
        print(f"error generating report: {e}", file=sys.stderr)
        return 2

    if not quiet:
        print(f"Report written to {dest}")
    return 0


# ────────────────────────────────────────────────────────────────────────────
# Wave 16  — VEX feed cache management
# ────────────────────────────────────────────────────────────────────────────

def _cmd_vex(args: argparse.Namespace, quiet: bool) -> int:
    try:
        from squish.squash.vex import VexCache
    except ImportError as e:
        print(f"squash is not installed: {e}", file=sys.stderr)
        return 2

    vex_cmd = getattr(args, "vex_command", None)
    if vex_cmd == "update":
        import os
        url = args.url or os.environ.get("SQUASH_VEX_URL", VexCache.DEFAULT_URL)
        timeout = float(args.timeout)
        try:
            cache = VexCache()
            feed = cache.load_or_fetch(url, timeout=timeout, force=True)
            if not quiet:
                print(f"VEX cache updated: {len(feed.statements)} statements from {url}")
        except Exception as e:
            print(f"error updating VEX cache: {e}", file=sys.stderr)
            return 2
        return 0

    if vex_cmd == "status":
        cache = VexCache()
        manifest = cache.manifest()
        if not manifest:
            if not quiet:
                print("VEX cache: empty (run 'squash vex update' to populate)")
            return 0
        if not quiet:
            print(f"URL         : {manifest.get('url', 'unknown')}")
            print(f"Fetched at  : {manifest.get('last_fetched', 'unknown')}")
            print(f"Statements  : {manifest.get('statement_count', 'unknown')}")
            stale = cache.is_stale()
            print(f"Stale       : {'yes' if stale else 'no'}")
        return 0

    # No sub-command — print help
    print("usage: squash vex {update,status} [options]", file=sys.stderr)
    return 1


# ────────────────────────────────────────────────────────────────────────────
# Wave 18  — Composite multi-model attestation
# ────────────────────────────────────────────────────────────────────────────

def _cmd_attest_composed(args: argparse.Namespace, quiet: bool) -> int:
    try:
        from squish.squash.attest import CompositeAttestConfig, CompositeAttestPipeline
    except ImportError as e:
        print(f"squash is not installed: {e}", file=sys.stderr)
        return 2

    model_paths = [Path(p) for p in args.model_paths]
    for mp in model_paths:
        if not mp.exists():
            print(f"error: path does not exist: {mp}", file=sys.stderr)
            return 1

    if len(model_paths) < 2:
        print("error: attest-composed requires at least two model paths", file=sys.stderr)
        return 1

    config = CompositeAttestConfig(
        model_paths=model_paths,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        policies=args.policies or ["enterprise-strict"],
        sign=args.sign,
    )

    try:
        result = CompositeAttestPipeline.run(config)
    except Exception as e:
        print(f"runtime error: {e}", file=sys.stderr)
        return 2

    if not quiet:
        icon = "✓" if result.passed else "✗"
        print(f"{icon} composite attestation {'passed' if result.passed else 'FAILED'}")
        for cr in result.component_results:
            sub_icon = "✓" if cr.passed else "✗"
            print(f"  {sub_icon} {cr.model_path}")
        if result.parent_bom_path:
            print(f"  parent BOM: {result.parent_bom_path}")

    return 0 if result.passed else 2


# ────────────────────────────────────────────────────────────────────────────
# Wave 19  — SBOM registry push
# ────────────────────────────────────────────────────────────────────────────

def _cmd_push(args: argparse.Namespace, quiet: bool) -> int:
    import os

    try:
        from squish.squash.sbom_builder import SbomRegistry
    except ImportError as e:
        print(f"squash is not installed: {e}", file=sys.stderr)
        return 2

    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"error: path does not exist: {model_path}", file=sys.stderr)
        return 1

    bom_path = model_path / "cyclonedx-mlbom.json"
    if not bom_path.exists():
        print(f"error: CycloneDX BOM not found: {bom_path}", file=sys.stderr)
        return 1

    api_key = args.api_key or os.environ.get("SQUASH_REGISTRY_KEY", "")
    registry_url: str = args.registry_url
    registry_type: str = getattr(args, "registry_type", "dtrack")

    try:
        if registry_type == "dtrack":
            pushed_url = SbomRegistry.push_dtrack(bom_path, registry_url, api_key)
        elif registry_type == "guac":
            pushed_url = SbomRegistry.push_guac(bom_path, registry_url)
        else:
            pushed_url = SbomRegistry.push_squash(bom_path, registry_url, api_key)
    except Exception as e:
        print(f"error pushing SBOM: {e}", file=sys.stderr)
        return 2

    if not quiet:
        print(f"✓ SBOM pushed to {pushed_url}")
    return 0


# ── Wave 20 — NTIA check handler ───────────────────────────────────────────────

def _cmd_ntia_check(args: argparse.Namespace, quiet: bool) -> int:
    from squish.squash.policy import NtiaValidator

    bom_path = Path(args.bom_path)
    if not bom_path.exists():
        print(f"error: BOM file not found: {bom_path}", file=sys.stderr)
        return 1
    try:
        result = NtiaValidator.check(bom_path, strict=getattr(args, "strict", False))
    except Exception as e:
        print(f"error: NTIA check failed: {e}", file=sys.stderr)
        return 2
    if not quiet:
        status = "PASS" if result.passed else "FAIL"
        print(f"NTIA minimum elements: {status}")
        print(f"  completeness: {result.completeness_score:.1%}  "
              f"({len(result.present_fields)}/{len(result.present_fields) + len(result.missing_fields)} fields)")
        if result.missing_fields:
            print(f"  missing: {', '.join(result.missing_fields)}")
    return 0 if result.passed else 1


# ── Wave 21 — SLSA attest handler ─────────────────────────────────────────────

def _cmd_slsa_attest(args: argparse.Namespace, quiet: bool) -> int:
    from squish.squash.slsa import SlsaLevel, SlsaProvenanceBuilder

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        print(f"error: model directory not found: {model_dir}", file=sys.stderr)
        return 1
    level_int = getattr(args, "level", 1)
    level = SlsaLevel(level_int)
    builder_id = getattr(args, "builder_id", "https://squish.local/squash/builder")
    try:
        attest = SlsaProvenanceBuilder.build(
            model_dir,
            level=level,
            builder_id=builder_id,
        )
    except Exception as e:
        print(f"error: SLSA attestation failed: {e}", file=sys.stderr)
        return 2
    if not quiet:
        print(f"✓ SLSA L{level.value} provenance written to {attest.output_path}")
        print(f"  subject: {attest.subject_name}")
        print(f"  digest:  sha256:{attest.subject_sha256}")
    return 0


# ── Wave 22 — BOM merge handler ───────────────────────────────────────────────

def _cmd_merge(args: argparse.Namespace, quiet: bool) -> int:
    from squish.squash.sbom_builder import BomMerger

    bom_paths = [Path(p) for p in args.bom_paths]
    output_path = Path(args.output)
    for p in bom_paths:
        if not p.exists():
            print(f"error: BOM file not found: {p}", file=sys.stderr)
            return 1
    try:
        merged = BomMerger.merge(bom_paths, output_path)
    except Exception as e:
        print(f"error: BOM merge failed: {e}", file=sys.stderr)
        return 2
    if not quiet:
        n_comp = len(merged.get("components", []))
        print(f"✓ Merged {len(bom_paths)} BOMs → {output_path}  ({n_comp} components)")
    return 0


# ── Wave 23 — Risk assess handler ─────────────────────────────────────────────

def _cmd_risk_assess(args: argparse.Namespace, quiet: bool) -> int:
    from squish.squash.risk import AiRiskAssessor

    model_dir = Path(args.model_dir)
    bom_path = model_dir / "cyclonedx-mlbom.json"
    if not bom_path.exists():
        print(f"error: CycloneDX BOM not found: {bom_path}", file=sys.stderr)
        return 1
    framework = getattr(args, "framework", "both")
    overall_passed = True
    try:
        if framework in ("eu-ai-act", "both"):
            eu = AiRiskAssessor.assess_eu_ai_act(bom_path)
            if not quiet:
                print(f"EU AI Act: {eu.category.value.upper()}  "
                      f"({'PASS' if eu.passed else 'FAIL'})")
                for r in eu.rationale:
                    print(f"  • {r}")
            if not eu.passed:
                overall_passed = False
        if framework in ("nist-rmf", "both"):
            rmf = AiRiskAssessor.assess_nist_rmf(bom_path)
            if not quiet:
                print(f"NIST RMF:  {rmf.category.value.upper()}  "
                      f"({'PASS' if rmf.passed else 'FAIL'})")
                for r in rmf.rationale:
                    print(f"  • {r}")
            if not rmf.passed:
                overall_passed = False
    except Exception as e:
        print(f"error: risk assessment failed: {e}", file=sys.stderr)
        return 2
    return 0 if overall_passed else 1


# ── Wave 24 — Drift monitor handler ───────────────────────────────────────────

def _cmd_monitor(args: argparse.Namespace, quiet: bool) -> int:
    from squish.squash.governor import DriftMonitor

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        print(f"error: model directory not found: {model_dir}", file=sys.stderr)
        return 1
    baseline = getattr(args, "baseline", None)
    once = getattr(args, "once", False)

    try:
        if baseline is None:
            snap = DriftMonitor.snapshot(model_dir)
            if not quiet:
                print(f"✓ Snapshot: {snap}")
            return 0
        events = DriftMonitor.compare(model_dir, baseline)
    except Exception as e:
        print(f"error: drift monitor failed: {e}", file=sys.stderr)
        return 2

    if not events:
        if not quiet:
            print("✓ No drift detected")
        return 0

    for evt in events:
        print(f"[{evt.event_type}] {evt.component}: {evt.old_value!r} → {evt.new_value!r}")
    return 1


# ── Wave 25 — CI run handler ───────────────────────────────────────────────────

def _cmd_ci_run(args: argparse.Namespace, quiet: bool) -> int:
    from squish.squash.cicd import CicdAdapter

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        print(f"error: model directory not found: {model_dir}", file=sys.stderr)
        return 1
    report_format = getattr(args, "report_format", "text")
    try:
        report = CicdAdapter.run_pipeline(model_dir, report_format=report_format)
    except Exception as e:
        print(f"error: CI pipeline failed: {e}", file=sys.stderr)
        return 2
    if not quiet and report_format in ("github", "text"):
        print(CicdAdapter.job_summary(report))
    return 0 if report.passed else 1


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    quiet: bool = getattr(args, "quiet", False)

    if not quiet:
        logging.basicConfig(
            level=logging.WARNING,
            format="%(levelname)s %(name)s: %(message)s",
        )

    if args.command == "policies":
        sys.exit(_cmd_policies(args, quiet))
    elif args.command == "scan":
        sys.exit(_cmd_scan(args, quiet))
    elif args.command == "diff":
        sys.exit(_cmd_diff(args, quiet))
    elif args.command == "verify":
        sys.exit(_cmd_verify(args, quiet))
    elif args.command == "report":
        sys.exit(_cmd_report(args, quiet))
    elif args.command == "vex":
        sys.exit(_cmd_vex(args, quiet))
    elif args.command == "attest-composed":
        sys.exit(_cmd_attest_composed(args, quiet))
    elif args.command == "push":
        sys.exit(_cmd_push(args, quiet))
    elif args.command == "attest":
        sys.exit(_cmd_attest(args, quiet))
    elif args.command == "ntia-check":
        sys.exit(_cmd_ntia_check(args, quiet))
    elif args.command == "slsa-attest":
        sys.exit(_cmd_slsa_attest(args, quiet))
    elif args.command == "merge":
        sys.exit(_cmd_merge(args, quiet))
    elif args.command == "risk-assess":
        sys.exit(_cmd_risk_assess(args, quiet))
    elif args.command == "monitor":
        sys.exit(_cmd_monitor(args, quiet))
    elif args.command == "ci-run":
        sys.exit(_cmd_ci_run(args, quiet))
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
