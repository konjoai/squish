"""squish/squash/sarif.py — SARIF 2.1.0 export for squash scan results.

Converts :class:`~squish.squash.scanner.ScanResult` objects (and their API
payload equivalents) to the SARIF 2.1.0 schema so that CI/CD tools such as
GitHub Advanced Security, VS Code, and Reviewdog can display model-security
findings inline.

Usage
-----
CLI::

    squash scan ./model --sarif report.sarif.json

API::

    GET /scan/{job_id}/sarif
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from squish.squash.scanner import ScanResult


class SarifBuilder:
    """Build SARIF 2.1.0 documents from squash scan results."""

    SARIF_VERSION = "2.1.0"
    SARIF_SCHEMA = (
        "https://schemastore.azurewebsites.net/schemas/json/sarif-2.1.0-rtm.5.json"
    )
    TOOL_NAME = "squash"
    TOOL_URI = "https://github.com/squishai/squish"

    # Maps squash severity → SARIF level
    _SEVERITY_MAP: dict[str, str] = {
        "critical": "error",
        "high": "error",
        "medium": "warning",
        "low": "note",
        "info": "note",
    }

    # ---------------------------------------------------------------------------
    # Public entry points
    # ---------------------------------------------------------------------------

    @classmethod
    def from_scan_result(cls, result: "ScanResult") -> dict[str, Any]:
        """Convert a :class:`ScanResult` dataclass to a SARIF 2.1 dict."""
        findings = [
            {
                "severity": f.severity,
                "id": f.finding_id,
                "title": f.title,
                "detail": f.detail,
                "file": f.file_path,
                "cve": f.cve,
            }
            for f in result.findings
        ]
        return cls._build(
            findings,
            scanned_path=result.scanned_path,
            scanner_version=result.scanner_version,
        )

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> dict[str, Any]:
        """Convert an API scan-job payload dict to a SARIF 2.1 dict.

        *payload* is the ``"result"`` value stored in the async scan job,
        as returned by ``GET /scan/{job_id}``.
        """
        return cls._build(
            payload.get("findings", []),
            scanned_path=payload.get("scanned_path", payload.get("path", "unknown")),
            scanner_version=payload.get("scanner_version", "squash/built-in"),
        )

    @classmethod
    def write(cls, result: "ScanResult", path: Path) -> None:
        """Write *result* as a SARIF 2.1 JSON file to *path*."""
        sarif = cls.from_scan_result(result)
        path.write_text(json.dumps(sarif, indent=2), encoding="utf-8")

    # ---------------------------------------------------------------------------
    # Internal
    # ---------------------------------------------------------------------------

    @classmethod
    def _build(
        cls,
        findings: list[dict[str, Any]],
        *,
        scanned_path: str,
        scanner_version: str,
    ) -> dict[str, Any]:
        rules: list[dict[str, Any]] = []
        results: list[dict[str, Any]] = []
        seen_rules: set[str] = set()

        for f in findings:
            rule_id = f.get("id", "unknown")
            level = cls._SEVERITY_MAP.get(f.get("severity", "info"), "note")
            title = f.get("title", rule_id)

            if rule_id not in seen_rules:
                seen_rules.add(rule_id)
                rules.append(
                    {
                        "id": rule_id,
                        "name": title,
                        "shortDescription": {"text": title},
                        "defaultConfiguration": {"level": level},
                    }
                )

            result_entry: dict[str, Any] = {
                "ruleId": rule_id,
                "level": level,
                "message": {"text": f.get("detail", title)},
                "locations": [
                    {
                        "physicalLocation": {
                            "artifactLocation": {
                                "uri": f.get("file", scanned_path),
                            }
                        }
                    }
                ],
            }
            cve = f.get("cve", "")
            if cve:
                result_entry["taxa"] = [
                    {"id": cve, "toolComponent": {"name": "CVE"}}
                ]
            results.append(result_entry)

        return {
            "$schema": cls.SARIF_SCHEMA,
            "version": cls.SARIF_VERSION,
            "runs": [
                {
                    "tool": {
                        "driver": {
                            "name": cls.TOOL_NAME,
                            "version": scanner_version,
                            "informationUri": cls.TOOL_URI,
                            "rules": rules,
                        }
                    },
                    "artifacts": [{"location": {"uri": scanned_path}}],
                    "results": results,
                }
            ],
        }
