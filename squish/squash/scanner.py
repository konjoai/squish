"""squish/squash/scanner.py — AI model security scanner.

Detects four classes of threats in AI model artifacts:

1. **Pickle/unsafe deserialization** — PyTorch ``.bin`` / ``.pt`` / ``.pth``
   files may contain arbitrary Python code executed at load time via
   ``pickle.load()``.  A reverse shell can be embedded in a model weight file.

2. **GGUF metadata arbitrary code execution** — GGUF files support a rich
   key-value metadata section.  Certain keys (``general.architecture``,
   ``tokenizer.ggml.pre``, ``.model_path``) have been weaponised in PoC attacks
   to trigger shell execution via malicious tokenizer configs loaded post-GGUF.

3. **ONNX external data references** — ONNX graphs can declare ``External``
   data refs that point outside the model directory, triggering path traversal
   or SSRF on load.  Scanned via ``onnx`` library or raw protobuf parse.

4. **safetensors header tampering** — The safetensors format encodes tensor
   sizes in a JSON header.  A malicious file can declare sizes that exceed actual
   data, causing out-of-bounds reads when loaded.

5. **ProtectAI ModelScan integration** — when ``modelscan`` is installed,
   run it **first** as the primary backend and merge its findings into the
   aggregate result.

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

        Scan order:
        1. ProtectAI ModelScan (if installed) — first-class backend
        2. PyTorch pickle scanner (``.bin/.pt/.pth/.pkl``)
        3. GGUF metadata scanner
        4. ONNX external-data reference scanner
        5. safetensors header size validator
        6. Zip archive scanner (detects compressed pickle payloads)

        ModelScan findings take precedence; built-in scanners fill coverage gaps.
        """
        all_findings: list[ScanFinding] = []
        scanned_files = 0
        status = "clean"

        def _merge(r: ScanResult) -> None:
            nonlocal status, scanned_files
            all_findings.extend(r.findings)
            scanned_files += 1
            if r.status == "unsafe":
                status = "unsafe"
            elif r.status in ("warning", "error") and status == "clean":
                status = r.status

        # 1. ProtectAI ModelScan — run first; skip built-in pickle scan if it ran
        modelscan_result = ModelScanner._run_modelscan(model_dir)
        modelscan_ran = modelscan_result is not None
        if modelscan_ran:
            all_findings.extend(modelscan_result.findings)  # type: ignore[union-attr]
            if modelscan_result.status == "unsafe":  # type: ignore[union-attr]
                status = "unsafe"

        # 2. Pickle scanner (only when ModelScan is absent — avoid double-counting)
        if not modelscan_ran:
            for ext in ("*.bin", "*.pt", "*.pth", "*.pkl"):
                for fp in model_dir.rglob(ext):
                    _merge(ModelScanner._scan_pickle(fp))

        # 3. GGUF scanner
        for fp in model_dir.rglob("*.gguf"):
            _merge(ModelScanner._scan_gguf(fp))

        # 4. ONNX scanner
        for fp in model_dir.rglob("*.onnx"):
            _merge(ModelScanner._scan_onnx(fp))

        # 5. safetensors header validator
        for fp in model_dir.rglob("*.safetensors"):
            _merge(ModelScanner._scan_safetensors(fp))

        # 6. Zip archive scanner
        for fp in model_dir.rglob("*.zip"):
            _merge(ModelScanner._scan_zip(fp))

        if scanned_files == 0 and not all_findings:
            # Nothing to scan — not a failure
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
        if suffix == ".onnx":
            return ModelScanner._scan_onnx(file_path)
        if suffix == ".safetensors":
            return ModelScanner._scan_safetensors(file_path)
        if suffix == ".zip":
            return ModelScanner._scan_zip(file_path)
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
    # ONNX scanner
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _scan_onnx(file_path: Path) -> ScanResult:
        """Scan an ONNX model for external data references pointing outside the model dir.

        ONNX ``External`` data refs with a ``location`` field beginning with ``..``
        or an absolute path are a path-traversal / SSRF vector when the model is
        loaded on a server.

        Uses the ``onnx`` library for accurate parsing when available; falls back
        to a raw byte-scan for the string ``external_data`` as a lightweight heuristic.
        """
        findings: list[ScanFinding] = []
        status = "clean"
        model_dir = file_path.parent

        try:
            data = file_path.read_bytes()
        except OSError as e:
            return ScanResult(
                scanned_path=str(file_path),
                status="error",
                findings=[
                    ScanFinding(
                        severity="info",
                        finding_id="SCAN-IO-003",
                        title="ONNX file read error",
                        detail=str(e),
                        file_path=str(file_path),
                    )
                ],
            )

        # Fast heuristic: does the protobuf even mention external data?
        if b"external_data" not in data and b"location" not in data[:4096]:
            return ScanResult(
                scanned_path=str(file_path),
                status="clean",
                findings=[],
            )

        # Try accurate parsing via onnx library
        try:
            import onnx  # type: ignore[import-untyped]

            model = onnx.load(str(file_path))
            for initializer in model.graph.initializer:
                if initializer.HasField("data_location") and initializer.data_location == 1:
                    # data_location == EXTERNAL
                    for kv in initializer.external_data:
                        if kv.key == "location":
                            loc = kv.value
                            # Flag if path escapes model directory or is absolute
                            if loc.startswith("..") or loc.startswith("/") or (
                                len(loc) > 2 and loc[1] == ":"
                            ):
                                findings.append(
                                    ScanFinding(
                                        severity="high",
                                        finding_id="SCAN-ONNX-001",
                                        title=f"ONNX external data ref escapes model dir: {loc!r}",
                                        detail=(
                                            f"Initializer '{initializer.name}' references external "
                                            f"data at '{loc}', which points outside the model directory. "
                                            "This is a path-traversal vector when the model is loaded "
                                            "on a server. Reject this ONNX file."
                                        ),
                                        file_path=str(file_path),
                                    )
                                )
                                status = "unsafe"
            return ScanResult(
                scanned_path=str(file_path),
                status=status,
                findings=findings,
            )
        except ImportError:
            pass  # onnx not installed — fall through to heuristic
        except Exception as e:
            log.warning("onnx parse failed for %s: %s", file_path, e)

        # Heuristic fallback: scan raw bytes for path-traversal strings
        for pattern in (b"../", b"..\\", b"\x00/etc", b"\x00/proc"):
            if pattern in data:
                findings.append(
                    ScanFinding(
                        severity="medium",
                        finding_id="SCAN-ONNX-002",
                        title=f"Suspicious path pattern in ONNX file: {pattern!r}",
                        detail=(
                            f"Pattern {pattern!r} found in ONNX file bytes. "
                            "Install 'onnx' (pip install onnx) for precise external-ref scanning. "
                            "Treat this file as untrusted until confirmed safe."
                        ),
                        file_path=str(file_path),
                    )
                )
                if status == "clean":
                    status = "warning"

        return ScanResult(
            scanned_path=str(file_path),
            status=status,
            findings=findings,
        )

    # ──────────────────────────────────────────────────────────────────────
    # safetensors header validator
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _scan_safetensors(file_path: Path) -> ScanResult:
        """Validate a safetensors file header.

        safetensors layout:
        - Bytes 0–7: little-endian uint64 ``header_length``
        - Bytes 8–(8+header_length): UTF-8 JSON header
        - Remaining bytes: tensor data

        A malicious file can:
        a) Declare a header length that exceeds the actual file size (truncation attack).
        b) Declare tensor data ranges that exceed ``file_size - 8 - header_length``
           (OOB-read / decompression bomb).
        """
        findings: list[ScanFinding] = []
        status = "clean"

        try:
            file_size = file_path.stat().st_size
            with file_path.open("rb") as fh:
                raw_len = fh.read(8)
                if len(raw_len) < 8:
                    return ScanResult(
                        scanned_path=str(file_path),
                        status="warning",
                        findings=[
                            ScanFinding(
                                severity="medium",
                                finding_id="SCAN-ST-003",
                                title="safetensors file too small to contain a valid header",
                                detail=f"File is {file_size} bytes — cannot contain the 8-byte length prefix.",
                                file_path=str(file_path),
                            )
                        ],
                    )

                header_len = struct.unpack("<Q", raw_len)[0]

                # Check 1: header length must fit within file
                if header_len > file_size - 8:
                    return ScanResult(
                        scanned_path=str(file_path),
                        status="unsafe",
                        findings=[
                            ScanFinding(
                                severity="critical",
                                finding_id="SCAN-ST-001",
                                title="safetensors header length exceeds file size — potential overflow attack",
                                detail=(
                                    f"Header declares {header_len} bytes but file is only "
                                    f"{file_size} bytes (8-byte prefix + {file_size - 8} remaining). "
                                    "This is a Buffer overflow / decompression-bomb attack. "
                                    "Reject this file immediately."
                                ),
                                file_path=str(file_path),
                            )
                        ],
                    )

                # Check 2: parse JSON header and validate tensor data offsets
                raw_header = fh.read(header_len)
                try:
                    header: dict = json.loads(raw_header)
                except (json.JSONDecodeError, UnicodeDecodeError) as e:
                    return ScanResult(
                        scanned_path=str(file_path),
                        status="warning",
                        findings=[
                            ScanFinding(
                                severity="medium",
                                finding_id="SCAN-ST-004",
                                title="safetensors header JSON is malformed",
                                detail=f"JSON parse error: {e}. File may be corrupted or tampered.",
                                file_path=str(file_path),
                            )
                        ],
                    )

                data_region_size = file_size - 8 - header_len
                for tensor_name, meta in header.items():
                    if tensor_name == "__metadata__" or not isinstance(meta, dict):
                        continue
                    offsets = meta.get("data_offsets")
                    if not isinstance(offsets, list) or len(offsets) != 2:
                        continue
                    start, end = offsets[0], offsets[1]
                    if end > data_region_size:
                        findings.append(
                            ScanFinding(
                                severity="critical",
                                finding_id="SCAN-ST-002",
                                title=(
                                    f"safetensors tensor '{tensor_name}' data_offsets "
                                    f"[{start}, {end}] exceed data region ({data_region_size} bytes)"
                                ),
                                detail=(
                                    f"Tensor '{tensor_name}' claims data at bytes [{start}, {end}] "
                                    f"but the data region is only {data_region_size} bytes. "
                                    "This is an out-of-bounds read attack. Reject this file."
                                ),
                                file_path=str(file_path),
                            )
                        )
                        status = "unsafe"

        except OSError as e:
            return ScanResult(
                scanned_path=str(file_path),
                status="error",
                findings=[
                    ScanFinding(
                        severity="info",
                        finding_id="SCAN-IO-004",
                        title="safetensors file read error",
                        detail=str(e),
                        file_path=str(file_path),
                    )
                ],
            )

        return ScanResult(
            scanned_path=str(file_path),
            status=status,
            findings=findings,
        )

    # ──────────────────────────────────────────────────────────────────────
    # Zip archive scanner
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _scan_zip(file_path: Path) -> ScanResult:
        """Scan a zip archive for embedded model files that should be scanned.

        Zip archives containing ``.bin``, ``.pt``, ``.pkl``, or ``.gguf`` files
        cannot be safely loaded without extracting — flag for manual review.
        Also detect zipslip (path traversal in ZIP entry names).
        """
        import zipfile

        findings: list[ScanFinding] = []
        status = "clean"

        try:
            # Explicitly stat the file so OSError propagates for missing paths,
            # because zipfile.is_zipfile() silently swallows FileNotFoundError.
            file_path.stat()
            if not zipfile.is_zipfile(file_path):
                return ScanResult(
                    scanned_path=str(file_path),
                    status="skipped",
                    findings=[],
                )

            with zipfile.ZipFile(file_path, "r") as zf:
                names = zf.namelist()

            dangerous_exts = {".bin", ".pt", ".pth", ".pkl", ".gguf"}
            embedded = [n for n in names if Path(n).suffix.lower() in dangerous_exts]
            if embedded:
                findings.append(
                    ScanFinding(
                        severity="critical",
                        finding_id="SCAN-ZIP-001",
                        title=f"ZIP archive contains {len(embedded)} unscanned model file(s)",
                        detail=(
                            f"Embedded files: {', '.join(embedded[:5])}"
                            + (f" (+{len(embedded)-5} more)" if len(embedded) > 5 else "")
                            + ". These cannot be safely loaded without extraction and scanning. "
                            "Extract and run squash scan --model-path <extracted_dir> before use."
                        ),
                        file_path=str(file_path),
                    )
                )
                status = "unsafe"

            # Zipslip detection: entries with path traversal
            traversal = [n for n in names if n.startswith("..") or "/../" in n or n.startswith("/")]
            if traversal:
                findings.append(
                    ScanFinding(
                        severity="critical",
                        finding_id="SCAN-ZIP-002",
                        title=f"Zipslip path traversal detected ({len(traversal)} entries)",
                        detail=(
                            f"Malicious entries: {', '.join(traversal[:3])}. "
                            "Extracting this archive can overwrite arbitrary files outside the "
                            "target directory (CVE-class: Zipslip). Reject immediately."
                        ),
                        file_path=str(file_path),
                    )
                )
                status = "unsafe"

        except (OSError, zipfile.BadZipFile) as e:
            return ScanResult(
                scanned_path=str(file_path),
                status="error",
                findings=[
                    ScanFinding(
                        severity="info",
                        finding_id="SCAN-IO-005",
                        title="ZIP file read error",
                        detail=str(e),
                        file_path=str(file_path),
                    )
                ],
            )

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
