"""squish/squash/mcp.py — MCP server tool manifest attestation.

McpScanner scans MCP tool manifest catalogs (the ``tools/list`` payload) for
six threat classes:

  1. **Prompt injection** — tool descriptions, parameter descriptions, or names
     that embed system-prompt hijack phrases.
  2. **SSRF / path-traversal vectors** — ``file://``, ``localhost``, RFC1918,
     cloud-metadata IPs, or gopher/ftp schemes in parameter text.
  3. **Tool shadowing** — names that collide with privileged OS/runtime
     commands (``exec``, ``shell``, ``eval``, ``sudo``, etc.).
  4. **Integrity gaps** — missing required MCP fields (``name``,
     ``description``, ``inputSchema``).
  5. **Data exfiltration patterns** — HTTP endpoints that look like OOB
     beacons (path indicators ``/upload``, ``/collect``, ``/beacon``,
     ``/track``; known collaborator/oastify/requestbin domains).
  6. **Permission over-reach** — tool names or descriptions claiming
     ``admin``, ``root``, ``sudo``, or ``system`` access.

McpSigner signs the catalog JSON with Sigstore keyless signing, producing a
``<catalog>.sig.json`` bundle alongside the catalog file.

EU AI Act Art. 9(2)(d): agentic systems that invoke MCP tools at runtime are
high-risk AI components subject to requirements of technical robustness and
adversarial input resilience.  This attestation fulfils that requirement.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# ── Threat patterns ───────────────────────────────────────────────────────────

# 1. Prompt-injection phrases (match anywhere in the text field, case-insensitive)
_INJECTION_PHRASES: list[str] = [
    r"ignore\s+(?:all\s+)?previous",
    r"disregard\s+(?:all\s+)?(?:previous|above)",
    r"forget\s+(?:all|everything)",
    r"you\s+are\s+now",
    r"act\s+as\s+(?:if|a|an)",
    r"pretend\s+(?:you\s+are|to\s+be)",
    r"jailbreak",
    r"DAN\s+mode",
    r"developer\s+mode",
    r"bypass\s+(?:your|the|all)",
    r"override\s+your",
    r"new\s+(?:role|instructions|task)",
    r"system\s+prompt",
    r"<!--",           # HTML comment injection
    r"<\|(?:im_start|im_end|endoftext)\|>",  # token-boundary injection
]
_INJECTION_RE: re.Pattern[str] = re.compile(
    "|".join(_INJECTION_PHRASES), re.IGNORECASE | re.DOTALL
)

# 2. SSRF / path-traversal indicators
_SSRF_PATTERNS: list[str] = [
    r"file://",
    r"gopher://",
    r"ftp://",
    r"dict://",
    r"(?:^|[^a-z])localhost(?:\b|:)",
    r"\b127\.0\.0\.1\b",
    r"\b0\.0\.0\.0\b",
    r"\b::1\b",
    r"\b169\.254\.169\.254\b",           # AWS/GCP metadata
    r"metadata\.google\.internal",
    r"\b10\.\d{1,3}\.\d{1,3}\.\d{1,3}\b",         # RFC1918
    r"\b192\.168\.\d{1,3}\.\d{1,3}\b",
    r"\b172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3}\b",
]
_SSRF_RE: re.Pattern[str] = re.compile(
    "|".join(_SSRF_PATTERNS), re.IGNORECASE
)

# 3. Tool shadowing — exact-match (lowercase) against dangerous command names
_SHADOW_NAMES: frozenset[str] = frozenset(
    {
        "shell", "bash", "sh", "zsh", "fish", "cmd", "powershell", "pwsh",
        "execute", "exec", "eval", "run", "spawn", "subprocess",
        "sudo", "su", "doas",
        "rm", "del", "delete", "rmdir",
        "chmod", "chown", "chroot",
        "kill", "killall", "shutdown", "reboot", "halt",
        "wget", "curl",  # outbound-fetch shadowing risk
        "nc", "netcat", "socat",
    }
)

# 5. Data exfiltration — URL path indicators and known OOB domains
_EXFIL_PATH_RE: re.Pattern[str] = re.compile(
    r"https?://[^\s]*/(?:upload|collect|log|beacon|track|report|exfil|dump)(?:\b|/)",
    re.IGNORECASE,
)
_EXFIL_DOMAIN_RE: re.Pattern[str] = re.compile(
    r"(?:burpcollaborator|oastify|requestbin|interactsh|canarytokens|ngrok\.io)",
    re.IGNORECASE,
)

# 6. Permission over-reach — phrases that claim elevated OS-level privileges
_PERMISSION_RE: re.Pattern[str] = re.compile(
    r"\b(?:admin(?:istrat(?:or|ive))?|root\s+access|sudo\s+access|"
    r"system\s+(?:admin|access|level)|god\s+mode|privileged\s+(?:mode|access))\b",
    re.IGNORECASE,
)

# ── Dataclasses ───────────────────────────────────────────────────────────────


@dataclass
class McpScanFinding:
    """A single threat finding from scanning one MCP tool entry."""

    rule_id: str      # e.g. "MCP-001"
    severity: str     # "error" | "warning"
    tool_name: str    # tool["name"] or "<unknown>" for integrity-gap findings
    field: str        # which field triggered: "name"|"description"|"inputSchema"|...
    detail: str       # human-readable explanation

    def to_dict(self) -> dict[str, str]:
        return {
            "rule_id": self.rule_id,
            "severity": self.severity,
            "tool_name": self.tool_name,
            "field": self.field,
            "detail": self.detail,
        }


@dataclass
class McpScanResult:
    """Aggregate result of scanning one MCP tool catalog."""

    status: str       # "safe" | "warn" | "unsafe"
    findings: list[McpScanFinding] = field(default_factory=list)
    tool_count: int = 0
    catalog_hash: str = ""  # SHA-256 of the serialized catalog bytes

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "tool_count": self.tool_count,
            "catalog_hash": self.catalog_hash,
            "findings": [f.to_dict() for f in self.findings],
            "summary": {
                "errors": sum(1 for f in self.findings if f.severity == "error"),
                "warnings": sum(1 for f in self.findings if f.severity == "warning"),
            },
        }


# ── McpScanner ────────────────────────────────────────────────────────────────


class McpScanner:
    """Pure-Python scanner for MCP tool manifest catalogs.

    Input format (MCP ``tools/list`` response)::

        {
            "tools": [
                {
                    "name": "get_weather",
                    "description": "Gets the weather for a location.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string", "description": "City name"}
                        },
                        "required": ["location"]
                    }
                }
            ]
        }

    Accepts the full response dict  **or** a plain ``list`` of tool objects.
    Never raises — all exceptions are caught and recorded as findings.
    """

    @staticmethod
    def scan(catalog: dict | list) -> McpScanResult:
        """Scan a parsed MCP tool catalog and return a :class:`McpScanResult`.

        Parameters
        ----------
        catalog:
            A dict with a ``"tools"`` key, or a plain list of tool entries.

        Returns
        -------
        McpScanResult
            status is ``"unsafe"`` if any error-severity finding is present,
            ``"warn"`` if warnings only, ``"safe"`` otherwise.
        """
        findings: list[McpScanFinding] = []
        catalog_hash = ""

        try:
            raw_bytes = json.dumps(catalog, sort_keys=True, ensure_ascii=False).encode()
            catalog_hash = hashlib.sha256(raw_bytes).hexdigest()
        except Exception as exc:  # noqa: BLE001
            log.warning("mcp: catalog hash failed: %s", exc)

        try:
            tools: list[dict] = (
                catalog
                if isinstance(catalog, list)
                else (catalog.get("tools") or [])
            )
        except Exception as exc:  # noqa: BLE001
            return McpScanResult(
                status="unsafe",
                findings=[
                    McpScanFinding(
                        rule_id="MCP-000",
                        severity="error",
                        tool_name="<catalog>",
                        field="<root>",
                        detail=f"Catalog is not a valid dict or list: {exc}",
                    )
                ],
                catalog_hash=catalog_hash,
            )

        for idx, tool in enumerate(tools):
            if not isinstance(tool, dict):
                findings.append(
                    McpScanFinding(
                        rule_id="MCP-004",
                        severity="error",
                        tool_name=f"<index {idx}>",
                        field="<root>",
                        detail=f"Tool entry at index {idx} is not a dict.",
                    )
                )
                continue

            t_name: str = tool.get("name") or f"<index {idx}>"
            findings.extend(McpScanner._check_integrity(tool, t_name))
            findings.extend(McpScanner._check_injection(tool, t_name))
            findings.extend(McpScanner._check_ssrf(tool, t_name))
            findings.extend(McpScanner._check_shadowing(tool, t_name))
            findings.extend(McpScanner._check_exfil(tool, t_name))
            findings.extend(McpScanner._check_permission(tool, t_name))

        has_error = any(f.severity == "error" for f in findings)
        has_warn = any(f.severity == "warning" for f in findings)
        status = "unsafe" if has_error else ("warn" if has_warn else "safe")

        return McpScanResult(
            status=status,
            findings=findings,
            tool_count=len(tools),
            catalog_hash=catalog_hash,
        )

    @staticmethod
    def scan_file(path: Path, policy: str = "mcp-strict") -> McpScanResult:
        """Load a JSON catalog from *path* and scan it.

        Parameters
        ----------
        path:
            Path to a JSON file containing the MCP tool catalog.
        policy:
            Reserved for future per-policy threshold tuning.  Ignored in
            the current implementation — ``mcp-strict`` is always applied.

        Returns
        -------
        McpScanResult
            Never raises; I/O errors are surfaced as error-severity findings.
        """
        try:
            catalog = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            return McpScanResult(
                status="unsafe",
                findings=[
                    McpScanFinding(
                        rule_id="MCP-000",
                        severity="error",
                        tool_name="<catalog>",
                        field="<file>",
                        detail=f"Failed to load catalog from {path}: {exc}",
                    )
                ],
            )
        return McpScanner.scan(catalog)

    # ── Internal checkers ─────────────────────────────────────────────────────

    @staticmethod
    def _collect_text_fields(tool: dict) -> list[tuple[str, str]]:
        """Yield (field_path, text) pairs from all string-valued fields."""
        pairs: list[tuple[str, str]] = []
        name = tool.get("name", "")
        desc = tool.get("description", "")
        if isinstance(name, str):
            pairs.append(("name", name))
        if isinstance(desc, str):
            pairs.append(("description", desc))

        schema = tool.get("inputSchema", {})
        if isinstance(schema, dict):
            props = schema.get("properties", {})
            if isinstance(props, dict):
                for prop_name, prop_def in props.items():
                    if isinstance(prop_def, dict):
                        prop_desc = prop_def.get("description", "")
                        if isinstance(prop_desc, str) and prop_desc:
                            pairs.append(
                                (f"inputSchema.properties.{prop_name}.description", prop_desc)
                            )
                        prop_default = prop_def.get("default", "")
                        if isinstance(prop_default, str) and prop_default:
                            pairs.append(
                                (f"inputSchema.properties.{prop_name}.default", prop_default)
                            )
        return pairs

    @staticmethod
    def _check_integrity(tool: dict, t_name: str) -> list[McpScanFinding]:
        """MCP-004: Required fields must be present and non-empty."""
        findings: list[McpScanFinding] = []
        for req_field in ("name", "description", "inputSchema"):
            val = tool.get(req_field)
            if val is None or (isinstance(val, str) and not val.strip()):
                findings.append(
                    McpScanFinding(
                        rule_id="MCP-004",
                        severity="error",
                        tool_name=t_name,
                        field=req_field,
                        detail=(
                            f"Tool '{t_name}' is missing required field '{req_field}'. "
                            "All MCP tool entries must declare name, description, and "
                            "inputSchema for safe attestation."
                        ),
                    )
                )
        return findings

    @staticmethod
    def _check_injection(tool: dict, t_name: str) -> list[McpScanFinding]:
        """MCP-001: Prompt injection phrases detected in any text field."""
        findings: list[McpScanFinding] = []
        for fpath, text in McpScanner._collect_text_fields(tool):
            m = _INJECTION_RE.search(text)
            if m:
                findings.append(
                    McpScanFinding(
                        rule_id="MCP-001",
                        severity="error",
                        tool_name=t_name,
                        field=fpath,
                        detail=(
                            f"Prompt injection pattern '{m.group()}' detected in "
                            f"field '{fpath}' of tool '{t_name}'. "
                            "EU AI Act Art. 9(2)(d): adversarial input resilience required."
                        ),
                    )
                )
        return findings

    @staticmethod
    def _check_ssrf(tool: dict, t_name: str) -> list[McpScanFinding]:
        """MCP-002: SSRF / path-traversal indicators in any text field."""
        findings: list[McpScanFinding] = []
        for fpath, text in McpScanner._collect_text_fields(tool):
            m = _SSRF_RE.search(text)
            if m:
                findings.append(
                    McpScanFinding(
                        rule_id="MCP-002",
                        severity="error",
                        tool_name=t_name,
                        field=fpath,
                        detail=(
                            f"SSRF/path-traversal indicator '{m.group().strip()}' "
                            f"found in field '{fpath}' of tool '{t_name}'."
                        ),
                    )
                )
        return findings

    @staticmethod
    def _check_shadowing(tool: dict, t_name: str) -> list[McpScanFinding]:
        """MCP-003: Tool name collides with a privileged OS/runtime command."""
        findings: list[McpScanFinding] = []
        name = tool.get("name", "")
        if isinstance(name, str) and name.lower() in _SHADOW_NAMES:
            findings.append(
                McpScanFinding(
                    rule_id="MCP-003",
                    severity="error",
                    tool_name=t_name,
                    field="name",
                    detail=(
                        f"Tool name '{name}' shadows a privileged OS/runtime command. "
                        "Agentic LLMs may confuse this with system-level capability."
                    ),
                )
            )
        return findings

    @staticmethod
    def _check_exfil(tool: dict, t_name: str) -> list[McpScanFinding]:
        """MCP-005: Data exfiltration URL patterns in any text field."""
        findings: list[McpScanFinding] = []
        for fpath, text in McpScanner._collect_text_fields(tool):
            if _EXFIL_PATH_RE.search(text) or _EXFIL_DOMAIN_RE.search(text):
                findings.append(
                    McpScanFinding(
                        rule_id="MCP-005",
                        severity="warning",
                        tool_name=t_name,
                        field=fpath,
                        detail=(
                            f"Potential data exfiltration endpoint detected in "
                            f"field '{fpath}' of tool '{t_name}'."
                        ),
                    )
                )
        return findings

    @staticmethod
    def _check_permission(tool: dict, t_name: str) -> list[McpScanFinding]:
        """MCP-006: Tool claims admin/root/sudo privileges."""
        findings: list[McpScanFinding] = []
        for fpath, text in McpScanner._collect_text_fields(tool):
            m = _PERMISSION_RE.search(text)
            if m:
                findings.append(
                    McpScanFinding(
                        rule_id="MCP-006",
                        severity="warning",
                        tool_name=t_name,
                        field=fpath,
                        detail=(
                            f"Permission over-reach: '{m.group()}' found in "
                            f"field '{fpath}' of tool '{t_name}'. "
                            "Review whether elevated privileges are necessary."
                        ),
                    )
                )
        return findings


# ── McpSigner ─────────────────────────────────────────────────────────────────


class McpSigner:
    """Sign an MCP tool catalog file with Sigstore keyless signing.

    Follows the same lazy-import / never-raises contract as
    :class:`squish.squash.oms_signer.OmsSigner`.
    """

    @staticmethod
    def sign(catalog_path: Path) -> Path | None:
        """Sign *catalog_path* and write a ``<catalog>.sig.json`` bundle.

        Returns the path to the signature bundle, or ``None`` if signing is
        unavailable (``sigstore`` not installed) or fails.  Never raises.
        """
        try:
            from sigstore.sign import Signer  # type: ignore[import]
            from sigstore.models import Bundle  # type: ignore[import]  # noqa: F401
        except ImportError:
            log.debug("mcp: sigstore not installed — skipping signing")
            return None

        try:
            catalog_path = Path(catalog_path)
            payload = catalog_path.read_bytes()
            signer = Signer.production()
            with signer.sign(io_obj=__import__("io").BytesIO(payload)) as result:
                bundle_json = result.to_bundle().to_json()
            sig_path = catalog_path.with_suffix(".sig.json")
            sig_path.write_text(bundle_json, encoding="utf-8")
            log.info("mcp: catalog signed → %s", sig_path)
            return sig_path
        except Exception as exc:  # noqa: BLE001
            log.warning("mcp: signing failed: %s", exc)
            return None
