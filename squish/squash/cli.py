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
        help="Policy name to evaluate (repeatable). Default: enterprise-strict. "
             "Also: eu-cra, fedramp, cmmc, eu-ai-act, nist-ai-rmf, owasp-llm-top10, iso-42001 "
             "(run 'squash policies' to list all)",
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
    # ── SPDX AI Profile enrichment ────────────────────────────────────────────
    attest.add_argument(
        "--spdx-type",
        default=None,
        metavar="TYPE",
        dest="spdx_type",
        help="SPDX AI Profile: type_of_model (e.g. text-generation, text-classification, "
             "translation, summarization, question-answering). Default: text-generation",
    )
    attest.add_argument(
        "--spdx-safety-risk",
        default=None,
        choices=["high", "medium", "low", "unspecified"],
        dest="spdx_safety_risk",
        help="SPDX AI Profile: safetyRiskAssessment tier. Default: unspecified",
    )
    attest.add_argument(
        "--spdx-dataset",
        action="append",
        default=[],
        dest="spdx_datasets",
        metavar="DATASET_ID",
        help="Training dataset HF ID or URI (repeatable; e.g. --spdx-dataset wikipedia "
             "--spdx-dataset c4). Embedded in the SPDX AI Profile",
    )
    attest.add_argument(
        "--spdx-training-info",
        default=None,
        dest="spdx_training_info",
        metavar="TEXT",
        help="SPDX AI Profile: informationAboutTraining free-text. "
             "Default: see-model-card",
    )
    attest.add_argument(
        "--spdx-sensitive-data",
        default=None,
        choices=["absent", "present", "unknown"],
        dest="spdx_sensitive_data",
        help="SPDX AI Profile: sensitivePIIInTrainingData. Default: absent",
    )
    # ── W49: offline / air-gapped mode ────────────────────────────────────────
    attest.add_argument(
        "--offline",
        action="store_true",
        default=False,
        help="Air-gapped mode: disable all OIDC/network calls (also set by SQUASH_OFFLINE=1)",
    )
    attest.add_argument(
        "--offline-key",
        metavar="PATH",
        default=None,
        dest="offline_key",
        help="Path to Ed25519 .priv.pem for offline signing (requires --sign --offline)",
    )

    # ── squash keygen ─────────────────────────────────────────────────────────
    keygen_cmd = sub.add_parser(
        "keygen",
        help="Generate a local Ed25519 keypair for offline signing",
        description=(
            "Generate an Ed25519 keypair for offline BOM signing.\n\n"
            "Example: squash keygen mykey\n"
            "Example: squash keygen ci-key --key-dir ~/.squash/keys"
        ),
    )
    keygen_cmd.add_argument("name", help="Base filename for the keypair (no extension)")
    keygen_cmd.add_argument(
        "--key-dir",
        metavar="DIR",
        default=".",
        dest="key_dir",
        help="Directory to write <name>.priv.pem and <name>.pub.pem (default: current dir)",
    )
    keygen_cmd.add_argument("--quiet", action="store_true", help="Suppress non-error output")

    # ── squash verify-local ───────────────────────────────────────────────────
    verify_local_cmd = sub.add_parser(
        "verify-local",
        help="Verify a BOM's Ed25519 offline signature against a local public key",
        description=(
            "Verify the local Ed25519 signature for a CycloneDX BOM.\n\n"
            "Example: squash verify-local ./model/cyclonedx-mlbom.json "
            "--key mykey.pub.pem\n"
            "Example: squash verify-local bom.json --key ci-key.pub.pem --sig bom.sig"
        ),
    )
    verify_local_cmd.add_argument("bom_path", help="Path to the CycloneDX BOM file to verify")
    verify_local_cmd.add_argument(
        "--key",
        required=True,
        metavar="PATH",
        dest="pub_key",
        help="Path to the Ed25519 .pub.pem public key",
    )
    verify_local_cmd.add_argument(
        "--sig",
        default=None,
        metavar="PATH",
        dest="sig_file",
        help="Explicit .sig file path (default: <bom_path>.sig)",
    )
    verify_local_cmd.add_argument("--quiet", action="store_true", help="Suppress non-error output")

    # ── squash pack-offline ───────────────────────────────────────────────────
    pack_offline_cmd = sub.add_parser(
        "pack-offline",
        help="Bundle a model directory and squash artefacts into a .squash-bundle.tar.gz",
        description=(
            "Archive a model directory (weights + BOM + signatures + chain) into a "
            "portable, self-contained tarball for air-gapped deployment.\n\n"
            "Example: squash pack-offline ./llama-3.1-8b-q4\n"
            "Example: squash pack-offline ./model --output /tmp/bundle.squash-bundle.tar.gz"
        ),
    )
    pack_offline_cmd.add_argument("model_dir", help="Path to the model directory to bundle")
    pack_offline_cmd.add_argument(
        "--output",
        metavar="PATH",
        default=None,
        dest="output_path",
        help="Output .squash-bundle.tar.gz path (default: <model_dir>-<timestamp>.squash-bundle.tar.gz)",
    )
    pack_offline_cmd.add_argument("--quiet", action="store_true", help="Suppress non-error output")

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

    # Wave 52 — subscribe / unsubscribe / list-subscriptions
    vex_subscribe = vex_sub.add_parser(
        "subscribe",
        help="Register a remote VEX feed URL for periodic polling",
        description=(
            "squash vex subscribe URL [--alias NAME] [--api-key-env VAR] [--polling-hours N]\n\n"
            "Example: squash vex subscribe https://vex.example.com/feed.json --alias corp-feed\n"
            "Example: squash vex subscribe https://api.example.com/vex --api-key-env CORP_VEX_KEY"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    vex_subscribe.add_argument(
        "url",
        metavar="URL",
        help="HTTPS endpoint returning an OpenVEX JSON feed",
    )
    vex_subscribe.add_argument(
        "--alias",
        default="",
        metavar="NAME",
        help="Short human-readable name for this subscription (optional)",
    )
    vex_subscribe.add_argument(
        "--api-key-env",
        default="SQUASH_VEX_API_KEY",
        metavar="VAR",
        help="Environment variable name that holds the API key (default: SQUASH_VEX_API_KEY)",
    )
    vex_subscribe.add_argument(
        "--polling-hours",
        type=int,
        default=24,
        metavar="N",
        help="Refresh interval in hours used by 'squash vex update --all' (default: 24)",
    )
    vex_subscribe.add_argument("--quiet", action="store_true", help="Suppress non-error output")

    vex_unsubscribe = vex_sub.add_parser(
        "unsubscribe",
        help="Remove a registered VEX feed subscription",
        description=(
            "squash vex unsubscribe URL_OR_ALIAS\n\n"
            "Example: squash vex unsubscribe corp-feed\n"
            "Example: squash vex unsubscribe https://vex.example.com/feed.json"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    vex_unsubscribe.add_argument(
        "url_or_alias",
        metavar="URL_OR_ALIAS",
        help="URL or alias of the subscription to remove",
    )
    vex_unsubscribe.add_argument("--quiet", action="store_true", help="Suppress non-error output")

    vex_sub.add_parser("list-subscriptions", help="List all registered VEX feed subscriptions")

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
        help="Policy name(s) to evaluate (repeatable; default: enterprise-strict; "
             "also: eu-cra, fedramp, cmmc — run 'squash policies' to list all)",
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

    # ── Wave 27 — Kubernetes admission webhook ─────────────────────────────────
    webhook_cmd = sub.add_parser(
        "webhook",
        help="Start the Kubernetes admission webhook server",
        description=(
            "Run an HTTPS validating admission webhook that enforces Squash BOM "
            "attestation policy.  Pods annotated with "
            "squash.ai/attestation-required=true must carry a valid "
            "squash.ai/bom-digest annotation whose digest is present in the "
            "configured policy store.\n\n"
            "Example: squash webhook --port 8443 --tls-cert /tls/tls.crt "
            "--tls-key /tls/tls.key --policy-store /var/squash/policy-store.json"
        ),
    )
    webhook_cmd.add_argument(
        "--port",
        type=int,
        default=8443,
        help="TCP port for the webhook server (default: 8443)",
    )
    webhook_cmd.add_argument(
        "--tls-cert",
        metavar="PATH",
        default=None,
        help="Path to PEM-encoded TLS certificate (omit for dev HTTP mode)",
    )
    webhook_cmd.add_argument(
        "--tls-key",
        metavar="PATH",
        default=None,
        help="Path to PEM-encoded TLS private key",
    )
    webhook_cmd.add_argument(
        "--policy-store",
        metavar="PATH",
        default=None,
        help="Path to JSON policy store file: {digest: bool}",
    )
    webhook_cmd.add_argument(
        "--default-deny",
        action="store_true",
        default=False,
        help="Deny pods that lack the attestation-required annotation (default: allow)",
    )
    webhook_cmd.add_argument("--quiet", action="store_true", help="Suppress non-error output")

    # ── Wave 50 — Shadow AI detection ─────────────────────────────────────────
    shadow_ai_cmd = sub.add_parser(
        "shadow-ai",
        help="Detect shadow AI model files running inside Kubernetes pods",
        description=(
            "Scan a Kubernetes pod list for shadow AI model file references "
            "(*.gguf, *.safetensors, *.bin, *.pt, etc.) in volume mounts, "
            "environment variables, and container arguments.\n\n"
            "Example: squash shadow-ai scan pods.json\n"
            "Example: kubectl get pods -o json | squash shadow-ai scan -\n"
            "Example: squash shadow-ai scan pods.json --fail-on-hits"
        ),
    )
    shadow_ai_sub = shadow_ai_cmd.add_subparsers(dest="shadow_ai_cmd", metavar="SUBCOMMAND")
    shadow_ai_sub.required = True
    shadow_ai_scan_cmd = shadow_ai_sub.add_parser(
        "scan",
        help="Scan a pod list JSON for shadow AI model file references",
        description=(
            "Read a Kubernetes pod list (kubectl get pods -o json) from a file or stdin "
            "and report any container that references shadow AI model files.\n\n"
            "Exit codes: 0 = clean, 1 = error, 2 = shadow AI hits found (with --fail-on-hits)"
        ),
    )
    shadow_ai_scan_cmd.add_argument(
        "pod_list",
        metavar="POD_LIST_JSON",
        help="Path to pod list JSON file, or '-' to read from stdin",
    )
    shadow_ai_scan_cmd.add_argument(
        "--namespace",
        metavar="NS",
        action="append",
        dest="namespaces",
        default=[],
        help="Only scan pods in this namespace (repeatable; default: all namespaces)",
    )
    shadow_ai_scan_cmd.add_argument(
        "--extensions",
        nargs="+",
        metavar="EXT",
        default=None,
        help="Override the set of file extensions to flag (e.g. --extensions .gguf .pt)",
    )
    shadow_ai_scan_cmd.add_argument(
        "--output-json",
        metavar="PATH",
        default=None,
        help="Write the full scan result as JSON to this path",
    )
    shadow_ai_scan_cmd.add_argument(
        "--fail-on-hits",
        action="store_true",
        default=False,
        help="Exit with code 2 if any shadow AI model files are detected",
    )
    shadow_ai_scan_cmd.add_argument(
        "--quiet", action="store_true", help="Suppress non-error output"
    )

    # ── Wave 51 — SBOM drift detection ────────────────────────────────────────
    drift_check_cmd = sub.add_parser(
        "drift-check",
        help="Verify a model directory against its CycloneDX BOM (SHA-256 digest check)",
        description=(
            "Compare the SHA-256 digests of every weight file in MODEL_DIR against "
            "the digests recorded in the squish CycloneDX BOM sidecar.  Reports "
            "missing or tampered files and optionally exits non-zero on drift.\n\n"
            "Example: squash drift-check ./my-model --bom ./my-model/cyclonedx-mlbom.json\n"
            "Example: squash drift-check ./my-model --bom bom.json --fail-on-drift\n"
            "Exit codes: 0 = clean, 1 = error, 2 = drift found (with --fail-on-drift)"
        ),
    )
    drift_check_cmd.add_argument(
        "model_dir",
        metavar="MODEL_DIR",
        help="Path to the squish compressed model directory",
    )
    drift_check_cmd.add_argument(
        "--bom",
        metavar="BOM_PATH",
        required=True,
        help="Path to the CycloneDX BOM JSON file (cyclonedx-mlbom.json)",
    )
    drift_check_cmd.add_argument(
        "--fail-on-drift",
        action="store_true",
        default=False,
        help="Exit with code 2 when drift is detected (default: exit 0 and report only)",
    )
    drift_check_cmd.add_argument(
        "--output-json",
        metavar="PATH",
        default=None,
        help="Write the full drift result as JSON to this path",
    )
    drift_check_cmd.add_argument(
        "--quiet", action="store_true", help="Suppress non-error output"
    )

    # ── Wave 29 — VEX publish + integration CLI shims ─────────────────────────
    vex_pub_cmd = sub.add_parser(
        "vex-publish",
        help="Generate and write a static OpenVEX 0.2.0 feed JSON file",
        description=(
            "Build an OpenVEX 0.2.0 document from a list of statement entries and "
            "write it to a configurable output path.  Entries are read from a JSON "
            "file, stdin ('-'), or an inline JSON string.\n\n"
            "Example: squash vex-publish --output feed.json --entries entries.json\n"
            "Example: squash vex-publish --output feed.json --entries '[]'"
        ),
    )
    vex_pub_cmd.add_argument(
        "--output",
        metavar="PATH",
        required=True,
        help="Destination path to write the OpenVEX JSON file",
    )
    vex_pub_cmd.add_argument(
        "--entries",
        metavar="PATH_OR_JSON",
        default="[]",
        help=(
            "Statement entries as a JSON file path, '-' for stdin, or inline JSON "
            "array string (default: '[]')"
        ),
    )
    vex_pub_cmd.add_argument(
        "--author",
        default="squash",
        help="Author field in the VEX document (default: squash)",
    )
    vex_pub_cmd.add_argument(
        "--doc-id",
        metavar="URL",
        default=None,
        help="Optional @id URI for the document; auto-generated if omitted",
    )
    vex_pub_cmd.add_argument("--quiet", action="store_true", help="Suppress non-error output")

    attest_mlflow_cmd = sub.add_parser(
        "attest-mlflow",
        help="Run attestation pipeline and emit result as JSON (MLflow-compatible)",
        description=(
            "Execute the full Squash attestation pipeline on MODEL_PATH and write "
            "the result JSON to stdout (or --output-dir).  Designed for piping into "
            "MLflow artifact upload scripts or CI steps that wrap mlflow.log_artifact.\n\n"
            "Example: squash attest-mlflow ./my-model --policies enterprise-strict"
        ),
    )
    attest_mlflow_cmd.add_argument("model_path", help="Path to the model directory or file")
    attest_mlflow_cmd.add_argument(
        "--output-dir",
        metavar="PATH",
        default=None,
        help="Directory to write attestation artifacts (default: <model_path>/../squash)",
    )
    attest_mlflow_cmd.add_argument(
        "--policies",
        nargs="*",
        metavar="POLICY",
        default=None,
        help="Policy templates to evaluate (default: enterprise-strict; "
             "also: eu-cra, fedramp, cmmc — run 'squash policies' for all)",
    )
    attest_mlflow_cmd.add_argument(
        "--sign", action="store_true", help="Sign BOM via Sigstore keyless"
    )
    attest_mlflow_cmd.add_argument(
        "--fail-on-violation",
        action="store_true",
        help="Exit 1 if any policy violation is found",
    )
    attest_mlflow_cmd.add_argument("--quiet", action="store_true", help="Suppress non-error output")

    attest_wandb_cmd = sub.add_parser(
        "attest-wandb",
        help="Run attestation pipeline and emit result as JSON (W&B-compatible)",
        description=(
            "Execute the full Squash attestation pipeline on MODEL_PATH and write "
            "the result JSON to stdout (or --output-dir).  Designed for piping into "
            "W&B artifact upload scripts or run-summary steps.\n\n"
            "Example: squash attest-wandb ./my-model --policies enterprise-strict"
        ),
    )
    attest_wandb_cmd.add_argument("model_path", help="Path to the model directory or file")
    attest_wandb_cmd.add_argument(
        "--output-dir",
        metavar="PATH",
        default=None,
        help="Directory to write attestation artifacts (default: <model_path>/../squash)",
    )
    attest_wandb_cmd.add_argument(
        "--policies",
        nargs="*",
        metavar="POLICY",
        default=None,
        help="Policy templates to evaluate (default: enterprise-strict; "
             "also: eu-cra, fedramp, cmmc — run 'squash policies' for all)",
    )
    attest_wandb_cmd.add_argument(
        "--sign", action="store_true", help="Sign BOM via Sigstore keyless"
    )
    attest_wandb_cmd.add_argument(
        "--fail-on-violation",
        action="store_true",
        help="Exit 1 if any policy violation is found",
    )
    attest_wandb_cmd.add_argument("--quiet", action="store_true", help="Suppress non-error output")

    attest_hf_cmd = sub.add_parser(
        "attest-huggingface",
        help="Attest a model and push artifacts to a HuggingFace Hub repository",
        description=(
            "Run the Squash attestation pipeline on MODEL_PATH and upload the "
            "resulting artifacts to --repo-id on the HuggingFace Hub.\n\n"
            "Example: squash attest-huggingface ./my-model --repo-id myorg/llama-3-8b"
        ),
    )
    attest_hf_cmd.add_argument("model_path", help="Path to the local model directory")
    attest_hf_cmd.add_argument(
        "--repo-id",
        metavar="ORG/REPO",
        default=None,
        help="HuggingFace Hub repo ID to push artifacts to (skip push if omitted)",
    )
    attest_hf_cmd.add_argument(
        "--hf-token",
        metavar="TOKEN",
        default=None,
        help="HuggingFace API token; falls back to HF_TOKEN env var",
    )
    attest_hf_cmd.add_argument(
        "--output-dir",
        metavar="PATH",
        default=None,
        help="Local artifact output directory (default: <model_path>/../squash)",
    )
    attest_hf_cmd.add_argument(
        "--policies",
        nargs="*",
        metavar="POLICY",
        default=None,
        help="Policy templates to evaluate (default: enterprise-strict; "
             "also: eu-cra, fedramp, cmmc — run 'squash policies' for all)",
    )
    attest_hf_cmd.add_argument(
        "--sign", action="store_true", help="Sign BOM via Sigstore keyless"
    )
    attest_hf_cmd.add_argument(
        "--fail-on-violation",
        action="store_true",
        help="Exit 1 if any policy violation is found",
    )
    attest_hf_cmd.add_argument("--quiet", action="store_true", help="Suppress non-error output")

    attest_lc_cmd = sub.add_parser(
        "attest-langchain",
        help="Run a one-shot attestation pass on a model (LangChain-compatible)",
        description=(
            "Run the Squash attestation pipeline on MODEL_PATH and write the "
            "result JSON to stdout.  Mirrors the behaviour of SquashCallback on "
            "first LLM invocation, allowing offline pre-validation before deploying "
            "a LangChain agent.\n\n"
            "Example: squash attest-langchain ./my-model --policies enterprise-strict"
        ),
    )
    attest_lc_cmd.add_argument("model_path", help="Path to the model directory or file")
    attest_lc_cmd.add_argument(
        "--output-dir",
        metavar="PATH",
        default=None,
        help="Directory for attestation artifacts (default: <model_path>/../squash)",
    )
    attest_lc_cmd.add_argument(
        "--policies",
        nargs="*",
        metavar="POLICY",
        default=None,
        help="Policy templates to evaluate (default: enterprise-strict; "
             "also: eu-cra, fedramp, cmmc — run 'squash policies' for all)",
    )
    attest_lc_cmd.add_argument(
        "--sign", action="store_true", help="Sign BOM via Sigstore keyless"
    )
    attest_lc_cmd.add_argument(
        "--fail-on-violation",
        action="store_true",
        help="Exit 1 if any policy violation is found",
    )
    attest_lc_cmd.add_argument("--quiet", action="store_true", help="Suppress non-error output")

    attest_mcp_cmd = sub.add_parser(
        "attest-mcp",
        help="Scan an MCP tool manifest catalog for supply-chain threats",
        description=(
            "Scan a Model Context Protocol (MCP) tools/list JSON catalog for six "
            "threat classes: prompt injection, SSRF vectors, tool shadowing, "
            "integrity gaps, data exfiltration patterns, and permission over-reach.\n\n"
            "Addresses EU AI Act Art. 9(2)(d): adversarial input resilience for "
            "agentic AI systems that invoke MCP tools at runtime.\n\n"
            "Example: squash attest-mcp ./mcp_catalog.json --policy mcp-strict"
        ),
    )
    attest_mcp_cmd.add_argument("catalog_path", help="Path to the MCP tool catalog JSON file")
    attest_mcp_cmd.add_argument(
        "--policy",
        metavar="POLICY",
        default="mcp-strict",
        help="Policy template to apply (default: mcp-strict)",
    )
    attest_mcp_cmd.add_argument(
        "--sign",
        action="store_true",
        help="Sign the catalog with Sigstore keyless signing after attestation",
    )
    attest_mcp_cmd.add_argument(
        "--fail-on-violation",
        action="store_true",
        help="Exit 1 if any error-severity finding is present",
    )
    attest_mcp_cmd.add_argument(
        "--json-result",
        metavar="PATH",
        default=None,
        help="Write scan result JSON to this file (default: stdout only)",
    )
    attest_mcp_cmd.add_argument(
        "--output-dir",
        metavar="PATH",
        default=None,
        help="Directory for attestation artifacts (default: catalog directory)",
    )
    attest_mcp_cmd.add_argument("--quiet", action="store_true", help="Suppress non-error output")

    # ── Wave 46 — Agent audit trail ───────────────────────────────────────────
    audit_cmd = sub.add_parser(
        "audit",
        help="Agent audit trail management (show / verify)",
        description=(
            "Manage the squash agent audit trail (append-only JSONL with hash chain).\n\n"
            "Examples:\n"
            "  squash audit show --n 20\n"
            "  squash audit verify --log /var/log/squash/audit.jsonl"
        ),
    )
    audit_sub = audit_cmd.add_subparsers(dest="audit_command")

    audit_show = audit_sub.add_parser(
        "show",
        help="Print the last N entries from the audit log",
    )
    audit_show.add_argument(
        "--n",
        type=int,
        default=20,
        help="Number of entries to show (default: 20)",
    )
    audit_show.add_argument(
        "--log",
        metavar="PATH",
        default=None,
        help="Audit log file path (default: $SQUASH_AUDIT_LOG or ~/.squash/audit.jsonl)",
    )
    audit_show.add_argument(
        "--json",
        dest="json_output",
        action="store_true",
        help="Output entries as a JSON array instead of pretty-printed lines",
    )
    audit_show.add_argument("--quiet", action="store_true", help="Suppress non-error output")

    audit_verify = audit_sub.add_parser(
        "verify",
        help="Verify the hash chain integrity of the audit log (exit 0=intact, 2=tampered)",
    )
    audit_verify.add_argument(
        "--log",
        metavar="PATH",
        default=None,
        help="Audit log file path (default: $SQUASH_AUDIT_LOG or ~/.squash/audit.jsonl)",
    )
    audit_verify.add_argument("--quiet", action="store_true", help="Suppress non-error output")

    # ── Wave 47 — RAG knowledge base integrity ────────────────────────────────
    scan_rag_cmd = sub.add_parser(
        "scan-rag",
        help="RAG knowledge base integrity scanner — index a corpus and detect drift",
    )
    scan_rag_sub = scan_rag_cmd.add_subparsers(dest="scan_rag_command")

    scan_rag_index = scan_rag_sub.add_parser(
        "index", help="Hash every document in a corpus and write a signed manifest"
    )
    scan_rag_index.add_argument("corpus_dir", help="Corpus directory to index")
    scan_rag_index.add_argument(
        "--glob",
        default="**/*",
        metavar="PATTERN",
        help='File glob (default "**/*")',
    )
    scan_rag_index.add_argument("--quiet", action="store_true", help="Suppress non-error output")

    scan_rag_verify = scan_rag_sub.add_parser(
        "verify",
        help="Verify live corpus against manifest (exit 0=intact, 2=drift detected)",
    )
    scan_rag_verify.add_argument("corpus_dir", help="Corpus directory to verify")
    scan_rag_verify.add_argument(
        "--json",
        dest="json_output",
        action="store_true",
        help="Print drift report as JSON",
    )
    scan_rag_verify.add_argument("--quiet", action="store_true", help="Suppress non-error output")

    # ── Wave 48 — Model transformation lineage ────────────────────────────────────
    lineage_cmd = sub.add_parser(
        "lineage",
        help="Model transformation lineage chain — record, show, and verify (EU AI Act Annex IV)",
        description=(
            "Manage the Merkle-chained transformation ledger for a model artefact.\n\n"
            "Addresses EU AI Act Annex IV technical documentation requirements (Art. 11)\n"
            "and NIST AI RMF GOVERN 1.7 (supply-chain provenance).  The chain file\n"
            "(.lineage_chain.json) travels with the model so provenance is available\n"
            "after transfer or M&A.\n\n"
            "Examples:\n"
            "  squash lineage record ./my-model --operation compress --params format=INT4 awq=true\n"
            "  squash lineage show   ./my-model\n"
            "  squash lineage verify ./my-model"
        ),
    )
    lineage_sub = lineage_cmd.add_subparsers(dest="lineage_command")

    lineage_record = lineage_sub.add_parser(
        "record", help="Append a transformation event to the lineage chain"
    )
    lineage_record.add_argument("model_dir", help="Model artefact directory")
    lineage_record.add_argument(
        "--operation",
        required=True,
        metavar="OP",
        help="Operation label (e.g. compress, quantize, sign, verify, export)",
    )
    lineage_record.add_argument(
        "--model-id",
        default="",
        dest="model_id",
        help="Model identifier (default: directory name)",
    )
    lineage_record.add_argument(
        "--input-dir",
        default="",
        dest="input_dir",
        help="Source model directory (default: model_dir)",
    )
    lineage_record.add_argument(
        "--params",
        nargs="*",
        metavar="KEY=VALUE",
        default=[],
        help="Arbitrary key=value operation parameters (repeatable)",
    )
    lineage_record.add_argument("--quiet", action="store_true", help="Suppress non-error output")

    lineage_show = lineage_sub.add_parser(
        "show", help="Print the transformation lineage chain for a model directory"
    )
    lineage_show.add_argument("model_dir", help="Model artefact directory")
    lineage_show.add_argument(
        "--json",
        dest="json_output",
        action="store_true",
        help="Output events as a JSON array",
    )
    lineage_show.add_argument("--quiet", action="store_true", help="Suppress non-error output")

    lineage_verify = lineage_sub.add_parser(
        "verify",
        help="Verify the Merkle chain integrity (exit 0=intact, 2=tampered/missing)",
    )
    lineage_verify.add_argument("model_dir", help="Model artefact directory")
    lineage_verify.add_argument("--quiet", action="store_true", help="Suppress non-error output")

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


# ────────────────────────────────────────────────────────────────────────────
# Wave 49 — air-gapped / offline signing helpers
# ────────────────────────────────────────────────────────────────────────────

def _cmd_keygen(args: argparse.Namespace, quiet: bool) -> int:
    """Generate an Ed25519 keypair for offline BOM signing."""
    try:
        from squish.squash.oms_signer import OmsSigner
    except ImportError as e:
        print(f"squash is not installed: {e}", file=sys.stderr)
        return 2

    try:
        priv_path, pub_path = OmsSigner.keygen(args.name, args.key_dir)
    except ImportError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except Exception as e:
        print(f"runtime error: {e}", file=sys.stderr)
        return 2

    if not quiet:
        print(f"✓ keypair generated:")
        print(f"  Private : {priv_path}")
        print(f"  Public  : {pub_path}")
        print()
        print("Keep the private key secret.  Share the public key for verification.")
        print(f"  Sign  : squash attest <model> --sign --offline --offline-key {priv_path}")
        print(f"  Verify: squash verify-local <bom> --key {pub_path}")
    return 0


def _cmd_verify_local(args: argparse.Namespace, quiet: bool) -> int:
    """Verify a BOM's Ed25519 offline signature against a local public key."""
    try:
        from squish.squash.oms_signer import OmsVerifier
    except ImportError as e:
        print(f"squash is not installed: {e}", file=sys.stderr)
        return 2

    bom_path = Path(args.bom_path)
    if not bom_path.exists():
        print(f"error: BOM not found: {bom_path}", file=sys.stderr)
        return 1

    pub_key_path = Path(args.pub_key)
    if not pub_key_path.exists():
        print(f"error: public key not found: {pub_key_path}", file=sys.stderr)
        return 1

    sig_path = Path(args.sig_file) if args.sig_file else None

    try:
        ok = OmsVerifier.verify_local(bom_path, pub_key_path, sig_path)
    except ImportError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except Exception as e:
        print(f"runtime error: {e}", file=sys.stderr)
        return 2

    if ok:
        if not quiet:
            print(f"✓ verified (offline): {bom_path}")
        return 0

    print(f"✗ verification FAILED (offline): {bom_path}", file=sys.stderr)
    return 2


def _cmd_pack_offline(args: argparse.Namespace, quiet: bool) -> int:
    """Bundle a model directory into a portable .squash-bundle.tar.gz archive."""
    try:
        from squish.squash.oms_signer import OmsSigner
    except ImportError as e:
        print(f"squash is not installed: {e}", file=sys.stderr)
        return 2

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        print(f"error: model_dir not found: {model_dir}", file=sys.stderr)
        return 1

    output_path = Path(args.output_path) if args.output_path else None

    try:
        bundle_path = OmsSigner.pack_offline(model_dir, output_path)
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"runtime error: {e}", file=sys.stderr)
        return 2

    size_mb = bundle_path.stat().st_size / (1024 * 1024)
    if not quiet:
        print(f"✓ bundle created: {bundle_path} ({size_mb:.1f} MB)")
    return 0


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

    # Build SpdxOptions only when the user supplied at least one SPDX flag.
    spdx_options = None
    if any([
        args.spdx_type,
        args.spdx_safety_risk,
        args.spdx_datasets,
        args.spdx_training_info,
        args.spdx_sensitive_data,
    ]):
        from squish.squash.spdx_builder import SpdxOptions
        spdx_options = SpdxOptions(
            type_of_model=args.spdx_type or "text-generation",
            safety_risk_assessment=args.spdx_safety_risk or "unspecified",
            dataset_ids=list(args.spdx_datasets),
            information_about_training=args.spdx_training_info or "see-model-card",
            sensitive_personal_information=args.spdx_sensitive_data or "absent",
        )

    config = AttestConfig(
        model_path=model_path,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        model_id=args.model_id,
        hf_repo=args.hf_repo,
        quant_format=args.quant_format,
        policies=policies,
        sign=args.sign,
        offline=args.offline,
        local_signing_key=Path(args.offline_key) if args.offline_key else None,
        fail_on_violation=False,  # handle ourselves below for clean exit codes
        skip_scan=args.skip_scan,
        spdx_options=spdx_options,
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
        api_key = os.environ.get("SQUASH_VEX_API_KEY") or None
        try:
            cache = VexCache()
            feed = cache.load_or_fetch(url, timeout=timeout, force=True, api_key=api_key)
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

    # ── Wave 52: subscribe  ───────────────────────────────────────────────────
    if vex_cmd == "subscribe":
        from squish.squash.vex import VexSubscription, VexSubscriptionStore
        import os
        url = args.url
        if not url.startswith("http"):
            print(f"error: URL must begin with http(s)://: {url!r}", file=sys.stderr)
            return 1
        sub = VexSubscription(
            url=url,
            alias=args.alias or "",
            api_key_env_var=args.api_key_env,
            polling_hours=args.polling_hours,
        )
        store = VexSubscriptionStore()
        store.add(sub)
        _q = getattr(args, "quiet", False) or quiet
        if not _q:
            label = f" (alias: {sub.alias})" if sub.alias else ""
            print(f"Subscribed to {url}{label}")
            print(f"  API key env : {sub.api_key_env_var}")
            print(f"  Polling     : every {sub.polling_hours}h")
        return 0

    if vex_cmd == "unsubscribe":
        from squish.squash.vex import VexSubscriptionStore
        store = VexSubscriptionStore()
        removed = store.remove(args.url_or_alias)
        _q = getattr(args, "quiet", False) or quiet
        if not removed:
            print(f"error: no subscription found for {args.url_or_alias!r}", file=sys.stderr)
            return 1
        if not _q:
            print(f"Unsubscribed from {args.url_or_alias}")
        return 0

    if vex_cmd == "list-subscriptions":
        from squish.squash.vex import VexSubscriptionStore
        store = VexSubscriptionStore()
        subs = store.list()
        if not subs:
            if not quiet:
                print("No VEX feed subscriptions registered.")
            return 0
        if not quiet:
            for sub in subs:
                alias_part = f" [{sub.alias}]" if sub.alias else ""
                polled = sub.last_polled or "never"
                print(f"  {sub.url}{alias_part}")
                print(f"    api-key-env={sub.api_key_env_var}  polling={sub.polling_hours}h  last-polled={polled}")
        return 0

    # No sub-command — print help
    print("usage: squash vex {update,status,subscribe,unsubscribe,list-subscriptions} [options]", file=sys.stderr)
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


# ── Wave 27 — Kubernetes admission webhook handler ─────────────────────────────

def _cmd_webhook(args: argparse.Namespace, quiet: bool) -> int:
    from squish.squash.integrations.kubernetes import (
        KubernetesWebhookHandler,
        WebhookConfig,
        serve_webhook,
    )

    policy_store_path = Path(args.policy_store) if getattr(args, "policy_store", None) else None
    config = WebhookConfig(
        policy_store_path=policy_store_path,
        default_allow=not getattr(args, "default_deny", False),
    )
    handler = KubernetesWebhookHandler(config)
    port: int = getattr(args, "port", 8443)
    tls_cert: str | None = getattr(args, "tls_cert", None)
    tls_key: str | None = getattr(args, "tls_key", None)

    if not quiet:
        mode = "HTTPS" if tls_cert else "HTTP (dev)"
        print(f"squash webhook: starting {mode} server on port {port}")
        if policy_store_path:
            print(f"squash webhook: policy store → {policy_store_path}")

    try:
        serve_webhook(handler, port=port, tls_cert=tls_cert, tls_key=tls_key)
    except Exception as e:
        print(f"error: webhook server failed: {e}", file=sys.stderr)
        return 2
    return 0


# ── Wave 50 — Shadow AI detection ─────────────────────────────────────────────

def _cmd_shadow_ai(args: argparse.Namespace, quiet: bool) -> int:  # noqa: C901
    """Run the shadow-ai scan subcommand."""
    import json as _json

    from squish.squash.integrations.kubernetes import (
        ShadowAiConfig,
        ShadowAiScanner,
        SHADOW_AI_MODEL_EXTENSIONS,
    )

    subcommand = getattr(args, "shadow_ai_cmd", None)
    if subcommand != "scan":
        print("error: unknown shadow-ai subcommand", file=sys.stderr)
        return 1

    # ─ Load pod list JSON ──────────────────────────────────────────────────────
    pod_list_path: str = args.pod_list
    try:
        if pod_list_path == "-":
            raw = sys.stdin.read()
        else:
            raw = Path(pod_list_path).read_text(encoding="utf-8")
        pod_list = _json.loads(raw)
    except (OSError, _json.JSONDecodeError) as exc:
        print(f"error: could not read pod list: {exc}", file=sys.stderr)
        return 1

    # ─ Build config ───────────────────────────────────────────────────────────
    extensions: frozenset[str] | None = None
    raw_exts: list[str] | None = getattr(args, "extensions", None)
    if raw_exts:
        extensions = frozenset(e if e.startswith(".") else f".{e}" for e in raw_exts)

    cfg = ShadowAiConfig(
        scan_extensions=extensions if extensions is not None else SHADOW_AI_MODEL_EXTENSIONS,
        namespaces_include=list(getattr(args, "namespaces", None) or []),
    )

    # ─ Scan ───────────────────────────────────────────────────────────────────
    scanner = ShadowAiScanner(cfg)
    result = scanner.scan_pod_list(pod_list)

    # ─ Output ─────────────────────────────────────────────────────────────────
    if not quiet:
        print(result.summary)
        for hit in result.hits:
            print(
                f"  [{hit.location_type}] {hit.namespace}/{hit.pod_name}"
                f" container={hit.container_name!r}"
                f" value={hit.matched_value!r} ({hit.extension})"
            )

    output_json_path: str | None = getattr(args, "output_json", None)
    if output_json_path:
        import dataclasses
        try:
            Path(output_json_path).write_text(
                _json.dumps(
                    {
                        "ok": result.ok,
                        "pods_scanned": result.pods_scanned,
                        "summary": result.summary,
                        "hits": [dataclasses.asdict(h) for h in result.hits],
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        except OSError as exc:
            print(f"error: could not write output JSON: {exc}", file=sys.stderr)
            return 1

    fail_on_hits: bool = getattr(args, "fail_on_hits", False)
    if not result.ok and fail_on_hits:
        return 2
    return 0


# ── Wave 51 — SBOM drift detection handler ───────────────────────────────────

def _cmd_drift_check(args: argparse.Namespace, quiet: bool) -> int:
    """Run the drift-check subcommand (W51)."""
    import json as _json
    import dataclasses
    from squish.squash.drift import DriftConfig, check_drift

    bom_path = Path(getattr(args, "bom", "") or "")
    model_dir = Path(args.model_dir)

    if not bom_path or not bom_path.exists():
        print(f"error: BOM file not found: {bom_path}", file=sys.stderr)
        return 1

    if not model_dir.exists():
        print(f"error: model directory not found: {model_dir}", file=sys.stderr)
        return 1

    try:
        config = DriftConfig(bom_path=bom_path, model_dir=model_dir)
        result = check_drift(config)
    except (OSError, _json.JSONDecodeError, ValueError) as exc:
        print(f"error: drift check failed: {exc}", file=sys.stderr)
        return 1

    if not quiet:
        print(result.summary)
        for hit in result.hits:
            if hit.missing:
                print(f"  [MISSING]  {hit.path}")
            else:
                print(f"  [TAMPERED] {hit.path}")
                print(f"             expected: {hit.expected_digest}")
                print(f"             actual:   {hit.actual_digest}")

    output_json_path: str | None = getattr(args, "output_json", None)
    if output_json_path:
        try:
            Path(output_json_path).write_text(
                _json.dumps(
                    {
                        "ok": result.ok,
                        "files_checked": result.files_checked,
                        "summary": result.summary,
                        "hits": [dataclasses.asdict(h) for h in result.hits],
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        except OSError as exc:
            print(f"error: could not write output JSON: {exc}", file=sys.stderr)
            return 1

    fail_on_drift: bool = getattr(args, "fail_on_drift", False)
    if not result.ok and fail_on_drift:
        return 2
    return 0


# ── Wave 29 — VEX publish + integration CLI shims ─────────────────────────────

def _cmd_vex_publish(args: argparse.Namespace, quiet: bool) -> int:
    """Generate an OpenVEX 0.2.0 feed JSON file from statement entries."""
    import json as _json
    import sys as _sys

    from squish.squash.vex import VexFeedManifest

    # Resolve entries: inline JSON string, '-' for stdin, or file path
    entries_raw: str = args.entries
    if entries_raw == "-":
        entries_raw = _sys.stdin.read()

    try:
        p = Path(entries_raw)
        if p.exists():
            entries_raw = p.read_text()
    except (OSError, ValueError):
        pass  # not a valid path — treat as inline JSON

    try:
        entries: list[dict] = _json.loads(entries_raw)
    except _json.JSONDecodeError as e:
        print(f"error: could not parse entries JSON: {e}", file=sys.stderr)
        return 1

    if not isinstance(entries, list):
        print("error: --entries must be a JSON array", file=sys.stderr)
        return 1

    doc = VexFeedManifest.generate(
        entries,
        author=args.author,
        doc_id=getattr(args, "doc_id", None),
    )

    errors = VexFeedManifest.validate(doc)
    if errors:
        for err in errors:
            print(f"validation error: {err}", file=sys.stderr)
        return 1

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(_json.dumps(doc, indent=2))

    if not quiet:
        print(
            f"✓ VEX feed written to {output_path} "
            f"({len(entries)} statement(s), spec {VexFeedManifest.SPEC_VERSION})"
        )
    return 0


def _cmd_attest_mlflow(args: argparse.Namespace, quiet: bool) -> int:
    """Run the attestation pipeline and emit result JSON (MLflow-compatible offline shim)."""
    import json as _json

    from squish.squash.attest import AttestConfig, AttestPipeline

    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"error: model path not found: {model_path}", file=sys.stderr)
        return 1

    out_dir = Path(args.output_dir) if getattr(args, "output_dir", None) else None
    config = AttestConfig(
        model_path=model_path,
        output_dir=out_dir or (model_path.parent / "squash"),
        policies=args.policies or ["enterprise-strict"],
        sign=getattr(args, "sign", False),
        fail_on_violation=getattr(args, "fail_on_violation", False),
    )

    try:
        result = AttestPipeline.run(config)
    except Exception as e:
        print(f"error: attestation failed: {e}", file=sys.stderr)
        return 2

    if not quiet:
        icon = "✓" if result.passed else "✗"
        print(f"{icon} mlflow attestation {'passed' if result.passed else 'FAILED'}: {model_path}")
        print(f"  artifacts  : {result.output_dir}")
        print(f"  bom_path   : {result.bom_path}")

    # Emit JSON to stdout for pipe-friendly consumption
    print(_json.dumps(result.to_dict() if hasattr(result, "to_dict") else {
        "passed": result.passed,
        "bom_path": str(result.bom_path) if result.bom_path else None,
        "output_dir": str(result.output_dir) if result.output_dir else None,
    }))
    return 0 if result.passed else 1


def _cmd_attest_wandb(args: argparse.Namespace, quiet: bool) -> int:
    """Run the attestation pipeline and emit result JSON (W&B-compatible offline shim)."""
    import json as _json

    from squish.squash.attest import AttestConfig, AttestPipeline

    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"error: model path not found: {model_path}", file=sys.stderr)
        return 1

    out_dir = Path(args.output_dir) if getattr(args, "output_dir", None) else None
    config = AttestConfig(
        model_path=model_path,
        output_dir=out_dir or (model_path.parent / "squash"),
        policies=args.policies or ["enterprise-strict"],
        sign=getattr(args, "sign", False),
        fail_on_violation=getattr(args, "fail_on_violation", False),
    )

    try:
        result = AttestPipeline.run(config)
    except Exception as e:
        print(f"error: attestation failed: {e}", file=sys.stderr)
        return 2

    if not quiet:
        icon = "✓" if result.passed else "✗"
        print(f"{icon} wandb attestation {'passed' if result.passed else 'FAILED'}: {model_path}")
        print(f"  artifacts  : {result.output_dir}")
        print(f"  bom_path   : {result.bom_path}")

    print(_json.dumps(result.to_dict() if hasattr(result, "to_dict") else {
        "passed": result.passed,
        "bom_path": str(result.bom_path) if result.bom_path else None,
        "output_dir": str(result.output_dir) if result.output_dir else None,
    }))
    return 0 if result.passed else 1


def _cmd_attest_huggingface(args: argparse.Namespace, quiet: bool) -> int:
    """Attest a model and optionally push artifacts to HuggingFace Hub."""
    import os

    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"error: model path not found: {model_path}", file=sys.stderr)
        return 1

    repo_id: str | None = getattr(args, "repo_id", None)
    hf_token: str | None = getattr(args, "hf_token", None) or os.environ.get("HF_TOKEN")
    policies = getattr(args, "policies", None) or ["enterprise-strict"]
    sign = getattr(args, "sign", False)
    fail_on_violation = getattr(args, "fail_on_violation", False)
    out_dir = Path(args.output_dir) if getattr(args, "output_dir", None) else None

    if repo_id:
        # Full push via HFSquash
        try:
            from squish.squash.integrations.huggingface import HFSquash
        except ImportError as e:
            print(f"error: HFSquash not available: {e}", file=sys.stderr)
            return 2
        try:
            result = HFSquash.attest_and_push(
                repo_id,
                model_path,
                hf_token=hf_token or "",
                policies=policies,
                sign=sign,
                fail_on_violation=fail_on_violation,
            )
        except Exception as e:
            print(f"error: HuggingFace attestation failed: {e}", file=sys.stderr)
            return 2
    else:
        # Offline attestation only (no push)
        from squish.squash.attest import AttestConfig, AttestPipeline

        config = AttestConfig(
            model_path=model_path,
            output_dir=out_dir or (model_path.parent / "squash"),
            policies=policies,
            sign=sign,
            fail_on_violation=fail_on_violation,
        )
        try:
            result = AttestPipeline.run(config)
        except Exception as e:
            print(f"error: attestation failed: {e}", file=sys.stderr)
            return 2

    if not quiet:
        icon = "✓" if result.passed else "✗"
        label = f"→ {repo_id}" if repo_id else "(local only)"
        print(f"{icon} huggingface attestation {'passed' if result.passed else 'FAILED'} {label}")
        print(f"  bom_path   : {result.bom_path}")

    return 0 if result.passed else 1


def _cmd_attest_langchain(args: argparse.Namespace, quiet: bool) -> int:
    """Run a one-shot attestation pass on a model (matches SquashCallback first-run behaviour)."""
    import json as _json

    from squish.squash.attest import AttestConfig, AttestPipeline

    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"error: model path not found: {model_path}", file=sys.stderr)
        return 1

    out_dir = Path(args.output_dir) if getattr(args, "output_dir", None) else None
    config = AttestConfig(
        model_path=model_path,
        output_dir=out_dir or (model_path.parent / "squash"),
        policies=getattr(args, "policies", None) or ["enterprise-strict"],
        sign=getattr(args, "sign", False),
        fail_on_violation=getattr(args, "fail_on_violation", False),
    )

    try:
        result = AttestPipeline.run(config)
    except Exception as e:
        print(f"error: attestation failed: {e}", file=sys.stderr)
        return 2

    if not quiet:
        icon = "✓" if result.passed else "✗"
        print(f"{icon} langchain attestation {'passed' if result.passed else 'FAILED'}: {model_path}")
        print(f"  artifacts  : {result.output_dir}")
        print(f"  bom_path   : {result.bom_path}")

    print(_json.dumps(result.to_dict() if hasattr(result, "to_dict") else {
        "passed": result.passed,
        "bom_path": str(result.bom_path) if result.bom_path else None,
        "output_dir": str(result.output_dir) if result.output_dir else None,
    }))
    return 0 if result.passed else 1


def _cmd_attest_mcp(args: argparse.Namespace, quiet: bool) -> int:
    """Scan an MCP tool manifest catalog for supply-chain threats."""
    import json as _json

    from squish.squash.mcp import McpScanner, McpSigner

    catalog_path = Path(args.catalog_path)
    if not catalog_path.exists():
        print(f"error: catalog not found: {catalog_path}", file=sys.stderr)
        return 1

    result = McpScanner.scan_file(catalog_path, getattr(args, "policy", "mcp-strict"))

    if not quiet:
        icon = "✓" if result.status == "safe" else ("⚠" if result.status == "warn" else "✗")
        label = {"safe": "SAFE", "warn": "WARNINGS", "unsafe": "UNSAFE"}.get(result.status, result.status.upper())
        print(f"{icon} MCP attestation {label}: {catalog_path}")
        print(f"  tools      : {result.tool_count}")
        print(f"  catalog_sha: {result.catalog_hash[:16]}…")
        errors = sum(1 for f in result.findings if f.severity == "error")
        warnings = sum(1 for f in result.findings if f.severity == "warning")
        if errors or warnings:
            print(f"  findings   : {errors} error(s), {warnings} warning(s)")
            for finding in result.findings:
                prefix = "  ✗" if finding.severity == "error" else "  ⚠"
                print(f"{prefix} [{finding.rule_id}] {finding.tool_name}: {finding.detail}")

    result_dict = result.to_dict()

    json_result_path = getattr(args, "json_result", None)
    if json_result_path:
        try:
            out_path = Path(json_result_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(_json.dumps(result_dict, indent=2), encoding="utf-8")
            if not quiet:
                print(f"  result     : {out_path}")
        except Exception as exc:
            print(f"error: could not write result file: {exc}", file=sys.stderr)
            return 2

    sign = getattr(args, "sign", False)
    if sign:
        sig_path = McpSigner.sign(catalog_path)
        if sig_path and not quiet:
            print(f"  signed     : {sig_path}")
        elif not sig_path and not quiet:
            print("  signing    : unavailable (sigstore not installed)", file=sys.stderr)

    fail_on_violation = getattr(args, "fail_on_violation", False)
    if fail_on_violation and result.status == "unsafe":
        return 1
    return 0


def _cmd_audit(args: argparse.Namespace, quiet: bool) -> int:
    """Handler for ``squash audit show`` and ``squash audit verify``."""
    audit_command = getattr(args, "audit_command", None)
    if not audit_command:
        print("usage: squash audit <show|verify>", file=sys.stderr)
        return 1

    try:
        from squish.squash.governor import AgentAuditLogger
    except ImportError as e:
        print(f"squash is not installed: {e}", file=sys.stderr)
        return 2

    log_path = getattr(args, "log", None)
    logger = AgentAuditLogger(log_path=log_path)

    if audit_command == "show":
        n = getattr(args, "n", 20)
        entries = logger.read_tail(n)
        if not entries:
            if not quiet:
                print("(audit log is empty or does not exist)")
            return 0
        json_output = getattr(args, "json_output", False)
        if json_output:
            print(json.dumps(entries, indent=2))
        else:
            for e in entries:
                ts = e.get("ts", "?")
                seq = e.get("seq", "?")
                etype = e.get("event_type", "?")
                model = e.get("model_id", "")
                session = e.get("session_id", "")
                latency = e.get("latency_ms", -1)
                lat_str = f" {latency:.1f}ms" if latency >= 0 else ""
                sid_str = f" [{session}]" if session else ""
                mod_str = f" model={model}" if model else ""
                print(f"#{seq} {ts} {etype}{sid_str}{mod_str}{lat_str}")
        return 0

    if audit_command == "verify":
        ok, msg = logger.verify_chain()
        if ok:
            if not quiet:
                path_str = str(logger.path)
                print(f"✓ audit chain intact: {path_str}")
            return 0
        print(f"✗ audit chain TAMPERED: {msg}", file=sys.stderr)
        return 2

    print(f"unknown audit subcommand: {audit_command}", file=sys.stderr)
    return 1


def _cmd_lineage(args: argparse.Namespace, quiet: bool) -> int:
    """Handler for ``squash lineage record``, ``show``, and ``verify``."""
    lineage_command = getattr(args, "lineage_command", None)
    if not lineage_command:
        print("usage: squash lineage {record,show,verify} -- use --help for details", file=sys.stderr)
        return 1

    try:
        from squish.squash.lineage import LineageChain  # lazy — keeps cli.py import-fast
    except ImportError as e:
        print(f"squash is not installed: {e}", file=sys.stderr)
        return 2

    model_dir = Path(args.model_dir)

    if lineage_command == "record":
        operation = args.operation
        model_id = getattr(args, "model_id", "") or model_dir.name
        input_dir = getattr(args, "input_dir", "") or str(model_dir)
        raw_params = getattr(args, "params", []) or []
        params: dict = {}
        for kv in raw_params:
            if "=" in kv:
                k, _, v = kv.partition("=")
                params[k.strip()] = v.strip()
        try:
            model_dir.mkdir(parents=True, exist_ok=True)
            evt = LineageChain.create_event(
                operation=operation,
                model_id=model_id,
                input_dir=input_dir,
                output_dir=str(model_dir),
                params=params,
            )
            event_hash = LineageChain.record(model_dir, evt)
        except Exception as exc:
            print(f"error recording lineage event: {exc}", file=sys.stderr)
            return 2
        if not quiet:
            print(
                f"✓ lineage event recorded\n"
                f"  model_dir  : {model_dir}\n"
                f"  operation  : {operation}\n"
                f"  event_hash : {event_hash}"
            )
        return 0

    if lineage_command == "show":
        if not model_dir.exists():
            print(f"error: directory not found: {model_dir}", file=sys.stderr)
            return 1
        try:
            events = LineageChain.load(model_dir)
        except Exception as exc:
            print(f"error loading lineage: {exc}", file=sys.stderr)
            return 2
        json_output = getattr(args, "json_output", False)
        if json_output:
            print(json.dumps([e.to_dict() for e in events], indent=2))
        else:
            if not events:
                if not quiet:
                    print("(no lineage events recorded)")
                return 0
            for i, e in enumerate(events):
                prev = e.prev_hash[:32] + "\u2026" if e.prev_hash else "(genesis)"
                print(f"#{i + 1} {e.timestamp}  [{e.operation}]  {e.model_id}")
                print(f"     operator  : {e.operator}")
                print(f"     input_dir : {e.input_dir}")
                print(f"     output_dir: {e.output_dir}")
                if e.params:
                    pstr = "  ".join(f"{k}={v}" for k, v in e.params.items())
                    print(f"     params    : {pstr}")
                print(f"     event_hash: {e.event_hash[:32]}\u2026")
                print(f"     prev_hash : {prev}")
        return 0

    if lineage_command == "verify":
        if not model_dir.exists():
            print(f"error: directory not found: {model_dir}", file=sys.stderr)
            return 1
        try:
            result = LineageChain.verify(model_dir)
        except Exception as exc:
            print(f"error verifying chain: {exc}", file=sys.stderr)
            return 2
        if not quiet:
            icon = "\u2713" if result.ok else "\u2717"
            print(f"{icon} lineage chain: {result.message}")
            print(f"   model_dir  : {result.model_dir}")
            print(f"   event_count: {result.event_count}")
            if result.broken_at is not None:
                print(f"   broken_at  : event index {result.broken_at}", file=sys.stderr)
        return 0 if result.ok else 2

    print(f"unknown lineage subcommand: {lineage_command}", file=sys.stderr)
    return 1


def _cmd_scan_rag(args: argparse.Namespace, quiet: bool) -> int:
    """Handler for ``squash scan-rag index`` and ``squash scan-rag verify``."""
    from squish.squash.rag import RagScanner  # lazy — keeps cli.py import-fast

    sub = getattr(args, "scan_rag_command", None)
    if sub is None:
        print("usage: squash scan-rag {index,verify} -- use --help for details", file=sys.stderr)
        return 1

    if sub == "index":
        corpus_dir = args.corpus_dir
        try:
            manifest = RagScanner.index(corpus_dir, glob=args.glob)
        except NotADirectoryError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        except Exception as exc:  # noqa: BLE001
            print(f"error indexing corpus: {exc}", file=sys.stderr)
            return 2
        if not quiet:
            print(
                f"✓ indexed {manifest.file_count} files\n"
                f"  corpus:        {manifest.corpus_dir}\n"
                f"  manifest_hash: {manifest.manifest_hash}\n"
                f"  indexed_at:    {manifest.indexed_at}"
            )
        return 0

    if sub == "verify":
        corpus_dir = args.corpus_dir
        try:
            result = RagScanner.verify(corpus_dir)
        except Exception as exc:  # noqa: BLE001
            print(f"error verifying corpus: {exc}", file=sys.stderr)
            return 2
        if getattr(args, "json_output", False):
            import json as _json
            print(_json.dumps(result.to_dict(), indent=2))
        elif not quiet:
            if result.ok:
                print(f"✓ corpus intact — {result.total_files} files, no drift")
            else:
                print(
                    f"✗ drift detected — {result.drift_count} change(s) in {result.corpus_dir}",
                    file=sys.stderr,
                )
                for item in result.drift:
                    print(f"  [{item.status:8s}] {item.path}", file=sys.stderr)
        return 0 if result.ok else 2

    print(f"unknown scan-rag subcommand: {sub}", file=sys.stderr)
    return 1


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
    elif args.command == "keygen":
        sys.exit(_cmd_keygen(args, quiet))
    elif args.command == "verify-local":
        sys.exit(_cmd_verify_local(args, quiet))
    elif args.command == "pack-offline":
        sys.exit(_cmd_pack_offline(args, quiet))
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
    elif args.command == "webhook":
        sys.exit(_cmd_webhook(args, quiet))
    elif args.command == "shadow-ai":
        sys.exit(_cmd_shadow_ai(args, quiet))
    elif args.command == "vex-publish":
        sys.exit(_cmd_vex_publish(args, quiet))
    elif args.command == "attest-mlflow":
        sys.exit(_cmd_attest_mlflow(args, quiet))
    elif args.command == "attest-wandb":
        sys.exit(_cmd_attest_wandb(args, quiet))
    elif args.command == "attest-huggingface":
        sys.exit(_cmd_attest_huggingface(args, quiet))
    elif args.command == "attest-langchain":
        sys.exit(_cmd_attest_langchain(args, quiet))
    elif args.command == "attest-mcp":
        sys.exit(_cmd_attest_mcp(args, quiet))
    elif args.command == "audit":
        sys.exit(_cmd_audit(args, quiet))
    elif args.command == "scan-rag":
        sys.exit(_cmd_scan_rag(args, quiet))
    elif args.command == "lineage":
        sys.exit(_cmd_lineage(args, quiet))
    elif args.command == "drift-check":
        sys.exit(_cmd_drift_check(args, quiet))
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
