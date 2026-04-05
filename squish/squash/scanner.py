"""squish/squash/scanner.py — AI model security scanner.

Detects three classes of threats in AI model artifacts:

1. **Pickle/unsafe deserialization** — PyTorch ``.bin`` / ``.pt`` / ``.pth``
   files may contain arbitrary Python code executed at load time via
   ``pickle.load()``.  A reverse shell can be embedded in a model weight file.

2. **GGUF metadata arbitrary code execution** — GGUF files support a rich
   key-value metadata section.  Certain keys (``general.architecture``,
   ``tokenizer.ggml.pre``, ``.model_path``) have been weaponised in PoC attacks
   to trigger shell execution via malicious tokenizer configs loaded post-GGUF.

3. **ProtectAI ModelScan integration** — when ``modelscan`` is installed,
   delegate to it as a subprocess and parse its JSON output into
   :class:`ScanResult`.  This avoids rebuilding what ProtectAI already ships.

All scan methods return a :class:`ScanResult` and never raise on scan errors —
a failed scan is reported as ``status="error"`` with the traceback in
``findings``.  Hard block only fires when ``status="unsafe"``.

Usage::

    result = ModelScanner.scan(Path("./model.bin"))
    if result.status == "unsafe":
        sys.exit(2)
"""

from __future__ import annotations

import hashlib
import json
import logging
import struct
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# Pickle opcodes that indicate code execution risk.
# REDUCE (0x52), GLOBAL (0x63), BUILD (0x62), INST (0x69), NEWOBJ (0x81),
# STACK_GLOBAL (\x93), EXT1 (0x82), EXT2 (0x83), EXT4 (0x84)
_DANGEROUS_OPCODES: frozenset[bytes] = frozenset(
    [b"\x52", b"\x63", b"\x62", b"\x69", b"\x81", b"\x93", b"\x82", b"\x83", b"\x84"]
)

# GGUF magic bytes
_GGUF_MAGIC = b"GGUF"

# GGUF metadata keys associated with ACE vectors
_GGUF_ACE_KEYS: frozenset[bytes] = frozenset(
    [
        b"tokenizer.ggml.model_path",
        b"tokenizer.ggml.pre",
        b"tokenizer.chat_template",
        b"general.file_type",
    ]
)

# Patterns indicating shell command injection in tokenizer templates
_SHELL_INJECTION_PATTERNS: list[bytes] = [
    b"os.system",
    b"subprocess",
    b"__import__",
    b"exec(",
    b"eval(",
    b"open(",
    b"__builtins__",
    b"/bin/sh",
    b"/bin/bash",
    b"cmd.exe",
    b"powershell",
]


@dataclass
class ScanFinding:
    """A single security finding from a model scan."""

    severity: str  # "critical" | "high" | "medium" | "low" | "info"
    finding_id: str
    title: str
    detail: str
    file_path: str
    cve: str = ""


@dataclass
class ScanResult:
    """Aggregate result of scanning a model artifact."""

    scanned_path: str
    status: str  # "clean" | "unsafe" | "warning" | "error" | "skipped"
    findings: list[ScanFinding] = field(default_factory=list)
    scanner_version: str = "squash/built-in"

    @property
    def is_safe(self) -> bool:
        return self.status in ("clean", "warning", "skipped")

    @property
    def critical_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == "critical")

    @property
    def high_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == "high")

    def summary(self) -> str:
        return (
            f"[{self.status.upper()}] {self.scanned_path}: "
            f"{len(self.findings)} findings "
            f"({self.critical_count} critical, {self.high_count} high)"
        )

    def to_cdx_vulnerabilities(self) -> list[dict[str, Any]]:
        """Convert findings to CycloneDX 1.7 vulnerabilities array."""
        vulns = []
        for f in self.findings:
            v: dict[str, Any] = {
                "id": f.finding_id,
                "source": {"name": "squash-scanner"},
                "ratings": [
                    {
                        "severity": f.severity,
                        "method": "other",
                    }
                ],
                "description": f.title,
                "detail": f.detail,
                "affects": [
                    {
                        "ref": f.file_path,
                    }
                ],
            }
            if f.cve:
                v["id"] = f.cve
            vulns.append(v)
        return vulns


class ModelScanner:
    """Scan AI model artifacts for security threats.

    Call :meth:`scan_directory` to scan all weight files in a model directory,
    or :meth:`scan` for a single file.
    """

    @staticmethod
    def scan_directory(model_dir: Path) -> ScanResult:
        """Scan all weight files in *model_dir* and return an aggregate result.

        Scans PyTorch files with the pickle scanner, GGUF files with the GGUF
        scanner, and delegates to ProtectAI ModelScan if available.
        """
        all_findings: list[ScanFinding] = []
        scanned_files = 0
        status = "clean"

        for ext in ("*.bin", "*.pt", "*.pth", "*.pkl"):
            for fp in model_dir.rglob(ext):
                r = ModelScanner._scan_pickle(fp)
                all_findings.extend(r.findings)
                scanned_files += 1
                if r.status == "unsafe":
                    status = "unsafe"
                elif r.status == "warning" and status == "clean":
                    status = "warning"

        for fp in model_dir.rglob("*.gguf"):
            r = ModelScanner._scan_gguf(fp)
            all_findings.extend(r.findings)
            scanned_files += 1
            if r.status == "unsafe":
                status = "unsafe"
            elif r.status == "warning" and status == "clean":
                status = "warning"

        # Opportunistically delegate to ProtectAI ModelScan
        modelscan_result = ModelScanner._run_modelscan(model_dir)
        if modelscan_result is not None:
            all_findings.extend(modelscan_result.findings)
            if modelscan_result.status == "unsafe":
                status = "unsafe"

        if scanned_files == 0 and not all_findings:
            # Nothing to scan (e.g. pure safetensors npy-dir) — not a failure
            status = "skipped"

        return ScanResult(
            scanned_path=str(model_dir),
            status=status,
            findings=all_findings,
        )

    @staticmethod
    def scan(file_path: Path) -> ScanResult:
        """Scan a single model file."""
        suffix = file_path.suffix.lower()
        if suffix in (".bin", ".pt", ".pth", ".pkl"):
            return ModelScanner._scan_pickle(file_path)
        if suffix == ".gguf":
            return ModelScanner._scan_gguf(file_path)
        return ScanResult(
            scanned_path=str(file_path),
            status="skipped",
            findings=[],
        )

    # ──────────────────────────────────────────────────────────────────────
    # Pickle scanner
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _scan_pickle(file_path: Path) -> ScanResult:
        """Scan a PyTorch/pickle file for dangerous opcodes."""
        findings: list[ScanFinding] = []
        status = "clean"

        try:
            data = file_path.read_bytes()
        except OSError as e:
            return ScanResult(
                scanned_path=str(file_path),
                status="error",
                findings=[
                    ScanFinding(
                        severity="info",
                        finding_id="SCAN-IO-001",
                        title="File read error",
                        detail=str(e),
                        file_path=str(file_path),
                    )
                ],
            )

        # Check for PyTorch ZIP container (PyTorch saves as ZIP with a pickle inside)
        if data[:2] == b"PK":
            # ZIP container — would need full extraction; flag as warning for manual review
            findings.append(
                ScanFinding(
                    severity="medium",
                    finding_id="SCAN-PKL-002",
                    title="PyTorch ZIP container detected — manual review recommended",
                    detail=(
                        "This file uses the PyTorch ZIP+pickle format. "
                        "Automated opcode scanning is limited. "
                        "Use ProtectAI ModelScan for full analysis: "
                        "pip install modelscan && modelscan -p " + str(file_path)
                    ),
                    file_path=str(file_path),
                )
            )
            status = "warning"
            return ScanResult(
                scanned_path=str(file_path),
                status=status,
                findings=findings,
            )

        # Raw pickle — scan opcodes byte-by-byte
        dangerous_found: list[str] = []
        for opcode in _DANGEROUS_OPCODES:
            if opcode in data:
                opcode_hex = opcode.hex()
                dangerous_found.append(opcode_hex)

        if dangerous_found:
            findings.append(
                ScanFinding(
                    severity="critical",
                    finding_id="SCAN-PKL-001",
                    title="Dangerous pickle opcodes detected — potential code execution",
                    detail=(
                        f"Opcodes detected: {', '.join(dangerous_found)}. "
                        "These opcodes can execute arbitrary code when the file is loaded. "
                        "Do NOT load this model with torch.load() or safetensors.load_file(). "
                        "Reject this model artifact."
                    ),
                    file_path=str(file_path),
                )
            )
            status = "unsafe"

        # Scan for shell injection patterns regardless of opcode presence
        for pattern in _SHELL_INJECTION_PATTERNS:
            if pattern in data:
                findings.append(
                    ScanFinding(
                        severity="high",
                        finding_id="SCAN-PKL-003",
                        title=f"Shell injection pattern found: {pattern.decode(errors='replace')}",
                        detail=(
                            f"Pattern '{pattern.decode(errors='replace')}' found in binary data. "
                            "This may indicate a malicious payload embedded in the model file."
                        ),
                        file_path=str(file_path),
                    )
                )
                if status == "clean":
                    status = "unsafe"

        return ScanResult(
            scanned_path=str(file_path),
            status=status,
            findings=findings,
        )

    # ──────────────────────────────────────────────────────────────────────
    # GGUF scanner
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _scan_gguf(file_path: Path) -> ScanResult:
        """Scan a GGUF file for ACE vectors in metadata."""
        findings: list[ScanFinding] = []
        status = "clean"

        try:
            data = file_path.read_bytes()
        except OSError as e:
            return ScanResult(
                scanned_path=str(file_path),
                status="error",
                findings=[
                    ScanFinding(
                        severity="info",
                        finding_id="SCAN-IO-002",
                        title="GGUF file read error",
                        detail=str(e),
                        file_path=str(file_path),
                    )
                ],
            )

        if not data.startswith(_GGUF_MAGIC):
            return ScanResult(
                scanned_path=str(file_path),
                status="skipped",
                findings=[],
            )

        # Scan the raw bytes of the GGUF metadata section for suspicious patterns.
        # GGUF metadata comes after the header (magic + version + tensor_count + kv_count).
        # For a fast surface scan, search the first 2MB for dangerous patterns.
        header_window = data[:2 * 1024 * 1024]

        for pattern in _SHELL_INJECTION_PATTERNS:
            if pattern in header_window:
                findings.append(
                    ScanFinding(
                        severity="critical",
                        finding_id="SCAN-GGUF-001",
                        title=f"Potential ACE payload in GGUF metadata: {pattern.decode(errors='replace')}",
                        detail=(
                            f"Pattern '{pattern.decode(errors='replace')}' found in GGUF metadata section. "
                            "GGUF metadata can be read by llama.cpp-based loaders and trigger "
                            "code execution via malicious tokenizer configs. "
                            "Do NOT load this GGUF file. Reject and quarantine."
                        ),
                        file_path=str(file_path),
                    )
                )
                status = "unsafe"

        # Check for suspiciously long metadata strings (overflow vectors)
        # GGUF string metadata length fields are uint64 LE starting at offset 24
        # (after magic=4, version=4, tensor_count=8, kv_count=8)
        try:
            offset = 24
            while offset < min(len(data) - 8, 1 * 1024 * 1024):
                # Each KV entry: string key (uint64 length + bytes) + value_type (uint32) + value
                key_len = struct.unpack_from("<Q", data, offset)[0]
                if key_len > 256:
                    findings.append(
                        ScanFinding(
                            severity="medium",
                            finding_id="SCAN-GGUF-002",
                            title=f"Abnormally long GGUF metadata key ({key_len} bytes)",
                            detail=(
                                f"GGUF metadata key at offset {offset} is {key_len} bytes. "
                                "Legitimate GGUF keys are typically <64 bytes. "
                                "This may indicate buffer overflow padding or obfuscated payload."
                            ),
                            file_path=str(file_path),
                        )
                    )
                    if status == "clean":
                        status = "warning"
                    break
                offset += 8 + key_len + 4  # skip key + value type
                if offset >= len(data):
                    break
        except struct.error:
            pass  # truncated GGUF — not a scan error

        return ScanResult(
            scanned_path=str(file_path),
            status=status,
            findings=findings,
        )

    # ──────────────────────────────────────────────────────────────────────
    # ProtectAI ModelScan delegation
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _run_modelscan(model_dir: Path) -> ScanResult | None:
        """Run ProtectAI ModelScan as a subprocess if installed.

        Returns ``None`` when modelscan is not installed (non-fatal).
        Returns a :class:`ScanResult` with findings when it runs.
        """
        modelscan_bin = _find_modelscan()
        if modelscan_bin is None:
            return None

        try:
            proc = subprocess.run(
                [modelscan_bin, "-p", str(model_dir), "-r", "json"],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if not proc.stdout.strip():
                return ScanResult(
                    scanned_path=str(model_dir),
                    status="clean",
                    scanner_version="modelscan",
                )

            raw: dict = json.loads(proc.stdout)
            # modelscan JSON: {"summary": {"total_issues": N, ...}, "issues": [{...}]}
            issues = raw.get("issues", [])
            findings: list[ScanFinding] = []
            for issue in issues:
                severity = issue.get("severity", "high").lower()
                findings.append(
                    ScanFinding(
                        severity=severity,
                        finding_id=f"MODELSCAN-{issue.get('code', '000')}",
                        title=issue.get("description", "ModelScan finding"),
                        detail=issue.get("details", ""),
                        file_path=issue.get("location", str(model_dir)),
                        cve=issue.get("cve", ""),
                    )
                )

            status = "unsafe" if findings else "clean"
            return ScanResult(
                scanned_path=str(model_dir),
                status=status,
                findings=findings,
                scanner_version="modelscan",
            )
        except (json.JSONDecodeError, subprocess.TimeoutExpired, OSError) as e:
            log.warning("modelscan run failed: %s", e)
            return None


def _find_modelscan() -> str | None:
    """Find the modelscan binary, returning None if absent."""
    import shutil
    return shutil.which("modelscan")
