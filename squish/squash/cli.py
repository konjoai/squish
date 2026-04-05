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

    return 0 if result.is_safe else 2


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
    elif args.command == "attest":
        sys.exit(_cmd_attest(args, quiet))
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
