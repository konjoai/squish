"""squish/serving/local_model_scanner.py — Scan for locally-installed models.

Detects models installed via Squish, Ollama, and LM Studio/Hugging Face and
presents them in a unified ``LocalModel`` format.

Also provides ``scan_before_load()`` — a lightweight pre-import safety scan
that inspects downloaded model files for dangerous pickle opcodes, invalid
GGUF magic, and malformed safetensors headers without executing any model code.

Also provides ``scan_hf_repo_metadata()`` — a **pre-download** safety scan that
queries the HuggingFace API for the file listing of a repo and flags known-
dangerous file types *before* any bytes are transferred.  No external scanner
is required; the analysis is performed natively on the file manifest.

Public API
──────────
LocalModel              — dataclass for a single discovered model
LocalModelScanner       — scans all sources and merges results
PreDownloadScanResult   — result of scan_before_load()
scan_before_load()      — scan a download dir before any model import
HFRepoScanResult        — result of scan_hf_repo_metadata()
scan_hf_repo_metadata() — pre-download HF metadata scan (no download needed)
"""
from __future__ import annotations

__all__ = [
    "LocalModel",
    "LocalModelScanner",
    "PreDownloadScanResult",
    "scan_before_load",
    "HFRepoScanResult",
    "scan_hf_repo_metadata",
]

import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# HuggingFace pre-download metadata scan (W100)
# ---------------------------------------------------------------------------

# File extensions that are always dangerous — arbitrary Python pickles.
_ALWAYS_DANGEROUS_EXTENSIONS: frozenset = frozenset([
    ".pkl", ".pickle",
])

# Extensions that *may* contain dangerous pickle payloads (PyTorch legacy).
# These are flagged as warnings unless the repo also contains safetensors
# equivalents, in which case the pickle files are redundant but still risky.
_POTENTIALLY_DANGEROUS_EXTENSIONS: frozenset = frozenset([
    ".bin", ".pt", ".pth",
])

# Safe model weight formats (no executable code).
_SAFE_WEIGHT_EXTENSIONS: frozenset = frozenset([
    ".safetensors", ".gguf", ".npz", ".ggml",
])

# HF API endpoint for model metadata.
_HF_API_BASE: str = "https://huggingface.co/api/models"

# Maximum number of files listed in the scan report.
_REPORT_MAX_FILES: int = 50

# Request timeout for HF API calls (seconds).
_HF_API_TIMEOUT: int = 15


@dataclass
class HFFileSummary:
    """Metadata for a single file in a HuggingFace repository."""
    filename: str
    size_bytes: int = 0
    flagged: bool = False
    flag_reason: str = ""


@dataclass
class HFRepoScanResult:
    """Result returned by ``scan_hf_repo_metadata()``.

    Attributes
    ----------
    status:       ``"safe"`` | ``"unsafe"`` | ``"warning"`` | ``"error"``
    repo_id:      HuggingFace repository id that was scanned.
    findings:     Human-readable list of security findings (empty when safe).
    file_summary: Per-file metadata list (up to ``_REPORT_MAX_FILES`` entries).
    total_files:  Total number of files reported by the API.
    total_size_bytes: Sum of all file sizes reported by the API.
    safe_weight_count:    Number of safe weight files (.safetensors / .gguf).
    dangerous_count:      Number of always-dangerous files (.pkl / .pickle).
    potentially_unsafe_count: Number of potentially-dangerous files (.bin / .pt).
    """
    status:                   str
    repo_id:                  str
    findings:                 List[str]            = field(default_factory=list)
    file_summary:             List[HFFileSummary]  = field(default_factory=list)
    total_files:              int                  = 0
    total_size_bytes:         int                  = 0
    safe_weight_count:        int                  = 0
    dangerous_count:          int                  = 0
    potentially_unsafe_count: int                  = 0


def scan_hf_repo_metadata(
    repo_id: str,
    token: Optional[str] = None,
) -> HFRepoScanResult:
    """Pre-download safety scan using the HuggingFace API file listing.

    Queries ``https://huggingface.co/api/models/<repo_id>?blobs=true`` to
    obtain the complete file manifest for the repository.  No model bytes are
    downloaded.  Classifies each file and returns a structured result with:

    - A ``"safe"`` status when only safetensors / GGUF / ONNX weight files are
      present.
    - A ``"warning"`` status when potentially-dangerous ``.bin`` / ``.pt`` files
      exist alongside safe equivalents (common for legitimate HF repos that ship
      both formats).
    - An ``"unsafe"`` status when ``.pkl`` / ``.pickle`` files are present, or
      when ``.bin`` files exist with *no* safe-format counterpart.
    - An ``"error"`` status when the API call fails or returns unexpected data.

    Parameters
    ----------
    repo_id:
        HuggingFace repository id, e.g. ``"mlx-community/Qwen3-8B-4bit"``.
    token:
        Optional HuggingFace bearer token for private repositories.

    Returns
    -------
    HFRepoScanResult
    """
    url = f"{_HF_API_BASE}/{repo_id}?blobs=true"
    headers: Dict[str, str] = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=_HF_API_TIMEOUT) as resp:
            raw = resp.read()
        data = json.loads(raw)
    except urllib.error.HTTPError as exc:
        status_code = exc.code
        if status_code == 401:
            msg = f"HF API 401 Unauthorized — provide --token for private repo {repo_id!r}"
        elif status_code == 404:
            msg = f"HF API 404 — repository {repo_id!r} not found"
        else:
            msg = f"HF API HTTP {status_code} for {repo_id!r}: {exc.reason}"
        return HFRepoScanResult(
            status="error",
            repo_id=repo_id,
            findings=[msg],
        )
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
        return HFRepoScanResult(
            status="error",
            repo_id=repo_id,
            findings=[f"HF API request failed for {repo_id!r}: {exc}"],
        )

    siblings: List[Dict] = data.get("siblings", [])
    if not isinstance(siblings, list):
        return HFRepoScanResult(
            status="error",
            repo_id=repo_id,
            findings=[f"Unexpected HF API response structure for {repo_id!r}"],
        )

    return _classify_hf_siblings(repo_id, siblings)


def _classify_hf_siblings(
    repo_id: str,
    siblings: List[Dict],
) -> HFRepoScanResult:
    """Classify a file list returned by the HF API and build a scan result."""
    findings: List[str] = []
    file_summary: List[HFFileSummary] = []
    total_size = 0
    safe_weight_count = 0
    dangerous_count = 0
    potentially_unsafe_count = 0

    for entry in siblings[:_REPORT_MAX_FILES]:
        filename: str = entry.get("rfilename", "")
        size: int = int(entry.get("size", 0) or 0)
        total_size += size
        suffix = Path(filename).suffix.lower()
        flagged = False
        flag_reason = ""

        if suffix in _ALWAYS_DANGEROUS_EXTENSIONS:
            dangerous_count += 1
            flagged = True
            flag_reason = f"dangerous file type {suffix!r} — arbitrary pickle execution"
            findings.append(
                f"[UNSAFE] {filename}: {flag_reason}"
            )
        elif suffix in _POTENTIALLY_DANGEROUS_EXTENSIONS:
            potentially_unsafe_count += 1
            flagged = True
            flag_reason = f"potentially unsafe file type {suffix!r} — may contain pickle payload"
        elif suffix in _SAFE_WEIGHT_EXTENSIONS:
            safe_weight_count += 1

        file_summary.append(HFFileSummary(
            filename=filename,
            size_bytes=size,
            flagged=flagged,
            flag_reason=flag_reason,
        ))

    # Account for any files beyond _REPORT_MAX_FILES
    for entry in siblings[_REPORT_MAX_FILES:]:
        size = int(entry.get("size", 0) or 0)
        total_size += size
        suffix = Path(entry.get("rfilename", "")).suffix.lower()
        if suffix in _ALWAYS_DANGEROUS_EXTENSIONS:
            dangerous_count += 1
            findings.append(
                f"[UNSAFE] {entry.get('rfilename', '?')}: dangerous file type {suffix!r}"
            )
        elif suffix in _POTENTIALLY_DANGEROUS_EXTENSIONS:
            potentially_unsafe_count += 1
        elif suffix in _SAFE_WEIGHT_EXTENSIONS:
            safe_weight_count += 1

    # Determine overall status
    if dangerous_count > 0:
        status = "unsafe"
    elif potentially_unsafe_count > 0 and safe_weight_count == 0:
        # .bin/.pt files with no safetensors/gguf counterpart — high risk
        findings.append(
            f"[UNSAFE] {repo_id}: {potentially_unsafe_count} pickle-format weight "
            f"file(s) (.bin/.pt) with no safe-format (.safetensors/.gguf) counterpart"
        )
        status = "unsafe"
    elif potentially_unsafe_count > 0:
        # .bin/.pt present but safe formats also exist — warn, don't block
        findings.append(
            f"[WARNING] {repo_id}: {potentially_unsafe_count} legacy pickle-format "
            f"weight file(s) (.bin/.pt) present alongside {safe_weight_count} "
            f"safe-format file(s) — prefer the safetensors variant"
        )
        status = "warning"
    else:
        status = "safe"

    return HFRepoScanResult(
        status=status,
        repo_id=repo_id,
        findings=findings,
        file_summary=file_summary,
        total_files=len(siblings),
        total_size_bytes=total_size,
        safe_weight_count=safe_weight_count,
        dangerous_count=dangerous_count,
        potentially_unsafe_count=potentially_unsafe_count,
    )


# ---------------------------------------------------------------------------
# Pre-download safety scan (post-download, pre-import)
# ---------------------------------------------------------------------------

# Pickle opcodes that allow arbitrary code execution.
# Source: Python pickle module protocol reference + ProtectAI threat analysis.
_DANGEROUS_OPCODES: frozenset[int] = frozenset([
    ord("R"),    # REDUCE      — calls func(*args), arbitrary execution
    ord("c"),    # GLOBAL      — imports module + attribute by name
    ord("i"),    # INST        — imports + instantiates class
    ord("b"),    # BUILD       — calls __setstate__, can trigger arbitrary code
    0x81,        # NEWOBJ      — calls cls.__new__(cls, *args)
    0x92,        # NEWOBJ_EX   — calls cls.__new__(cls, *args, **kwargs)
    0x93,        # STACK_GLOBAL — protocol 4 global import
])

# Extensions scanned for pickle payloads.
_PICKLE_EXTENSIONS: frozenset[str] = frozenset([
    ".bin", ".pt", ".pth", ".pkl", ".pickle",
])

# GGUF magic bytes (4-byte prefix of every valid GGUF file).
_GGUF_MAGIC: bytes = b"GGUF"

# safetensors header is a little-endian uint64 length followed by JSON.
_SAFETENSORS_MIN_HEADER: int = 8


@dataclass
class PreDownloadScanResult:
    """Result returned by ``scan_before_load()``.

    Attributes
    ----------
    status:   ``"clean"`` | ``"unsafe"`` | ``"error"``
    findings: Human-readable list of issues found (empty when clean).
    scanned:  Number of files inspected.
    """
    status:   str
    findings: list[str] = field(default_factory=list)
    scanned:  int = 0


def scan_before_load(download_dir: Path) -> PreDownloadScanResult:
    """Scan a downloaded model directory for unsafe content before any import.

    Inspects every file matching known model extensions without executing any
    model code.  Checks performed:

    - Pickle files (``.bin``, ``.pt``, ``.pkl``, …): scan raw bytes for
      dangerous opcodes (REDUCE, GLOBAL, INST, BUILD, NEWOBJ, STACK_GLOBAL).
    - GGUF files: validate 4-byte magic ``GGUF`` at offset 0.
    - safetensors files: validate uint64 header length is plausible.

    Parameters
    ----------
    download_dir:
        Directory that was just downloaded from HuggingFace or another source.

    Returns
    -------
    PreDownloadScanResult
        ``.status == "unsafe"`` if dangerous content is detected.
        ``.status == "clean"``  if no issues found.
        ``.status == "error"``  if the directory is unreadable.
    """
    if not download_dir.is_dir():
        return PreDownloadScanResult(
            status="error",
            findings=[f"Directory not found: {download_dir}"],
        )

    findings: list[str] = []
    scanned = 0

    for path in sorted(download_dir.rglob("*")):
        if not path.is_file():
            continue

        suffix = path.suffix.lower()

        if suffix in _PICKLE_EXTENSIONS:
            scanned += 1
            result = _scan_pickle_file(path)
            if result:
                findings.extend(result)

        elif suffix == ".gguf":
            scanned += 1
            result = _scan_gguf_file(path)
            if result:
                findings.extend(result)

        elif suffix == ".safetensors":
            scanned += 1
            result = _scan_safetensors_file(path)
            if result:
                findings.extend(result)

    return PreDownloadScanResult(
        status="unsafe" if findings else "clean",
        findings=findings,
        scanned=scanned,
    )


def _scan_pickle_file(path: Path) -> list[str]:
    """Return findings for a single pickle file; empty list = clean."""
    try:
        data = path.read_bytes()
    except OSError as exc:
        return [f"[READ ERROR] {path.name}: {exc}"]

    found: list[str] = []
    for i, byte in enumerate(data):
        if byte in _DANGEROUS_OPCODES:
            opcode_name = {
                ord("R"): "REDUCE",
                ord("c"): "GLOBAL",
                ord("i"): "INST",
                ord("b"): "BUILD",
                0x81:     "NEWOBJ",
                0x92:     "NEWOBJ_EX",
                0x93:     "STACK_GLOBAL",
            }.get(byte, hex(byte))
            found.append(
                f"[UNSAFE] {path.name}: dangerous pickle opcode {opcode_name} at byte {i}"
            )
            break  # one finding per file is enough; don't flood output
    return found


def _scan_gguf_file(path: Path) -> list[str]:
    """Return findings for a GGUF file; empty list = valid magic."""
    try:
        header = path.read_bytes()[:4]
    except OSError as exc:
        return [f"[READ ERROR] {path.name}: {exc}"]
    if header != _GGUF_MAGIC:
        return [f"[UNSAFE] {path.name}: invalid GGUF magic {header!r} (expected b'GGUF')"]
    return []


def _scan_safetensors_file(path: Path) -> list[str]:
    """Return findings for a safetensors file; empty list = plausible header."""
    import struct
    try:
        raw = path.read_bytes()[:_SAFETENSORS_MIN_HEADER]
    except OSError as exc:
        return [f"[READ ERROR] {path.name}: {exc}"]
    if len(raw) < _SAFETENSORS_MIN_HEADER:
        return [f"[UNSAFE] {path.name}: truncated safetensors header ({len(raw)} bytes)"]
    (header_len,) = struct.unpack_from("<Q", raw, 0)
    file_size = path.stat().st_size
    if header_len == 0 or header_len > file_size:
        return [
            f"[UNSAFE] {path.name}: safetensors header length {header_len} "
            f"exceeds file size {file_size}"
        ]
    return []


# ---------------------------------------------------------------------------
# LocalModel dataclass
# ---------------------------------------------------------------------------

@dataclass
class LocalModel:
    """A locally-installed model discovered by the scanner.

    Attributes
    ----------
    name:       Canonical model name (e.g. ``qwen3:8b`` or ``llama3.1:8b``).
    path:       Absolute path to the model directory or file.
    source:     Origin scanner: ``"squish"``, ``"ollama"``, ``"lm_studio"``,
                or ``"gguf"``.
    size_bytes: Approximate on-disk size in bytes (0 if unknown).
    family:     Model family guess (e.g. ``"qwen"``, ``"llama"``).
    params:     Parameter count string (e.g. ``"7B"`` or ``""`` if unknown).
    """
    name:       str
    path:       Path
    source:     str
    size_bytes: int = 0
    family:     str = ""
    params:     str = ""

    def __post_init__(self) -> None:
        self.path = Path(self.path)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dir_size(path: Path) -> int:
    """Return total on-disk size of a directory tree in bytes."""
    try:
        return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    except Exception:
        return 0


def _guess_family(name: str) -> str:
    name_l = name.lower()
    for family in ("qwen", "llama", "gemma", "phi", "mistral", "deepseek",
                   "mixtral", "falcon", "vicuna", "alpaca"):
        if family in name_l:
            return family
    return ""


def _guess_params(name: str) -> str:
    import re
    m = re.search(r"(\d+\.?\d*)b", name.lower())
    return f"{m.group(1)}B" if m else ""


# ---------------------------------------------------------------------------
# LocalModelScanner
# ---------------------------------------------------------------------------

class LocalModelScanner:
    """Scan multiple sources for locally-installed models.

    Parameters
    ----------
    squish_models_dir:
        Directory containing Squish-compressed models.
        Defaults to ``~/models``.
    ollama_manifests_dir:
        Root of the Ollama manifests tree.
        Defaults to ``~/.ollama/models/manifests/registry.ollama.ai/library``.
    lm_studio_dir:
        Root of the LM Studio / Hugging Face cache.
        Defaults to ``~/.cache/lm-studio/models``.
    """

    def __init__(
        self,
        squish_models_dir: Optional[Path] = None,
        ollama_manifests_dir: Optional[Path] = None,
        lm_studio_dir: Optional[Path] = None,
    ) -> None:
        home = Path.home()
        self._squish_dir  = squish_models_dir   or (home / "models")
        self._ollama_dir  = ollama_manifests_dir or (
            home / ".ollama" / "models" / "manifests" / "registry.ollama.ai" / "library"
        )
        self._lm_studio_dir = lm_studio_dir or (home / ".cache" / "lm-studio" / "models")

    # ------------------------------------------------------------------
    # Individual scanners
    # ------------------------------------------------------------------

    def scan_squish(self) -> list[LocalModel]:
        """Return models installed by Squish (``~/models/`` subdirs)."""
        if not self._squish_dir.is_dir():
            return []
        models: list[LocalModel] = []
        for d in sorted(self._squish_dir.iterdir()):
            if not d.is_dir() or d.name.startswith("."):
                continue
            # Convert dir name to canonical id (e.g. "Qwen3-8B-bf16" → "qwen3:8b")
            name = _dir_to_canonical(d.name)
            models.append(LocalModel(
                name=name,
                path=d,
                source="squish",
                size_bytes=_dir_size(d),
                family=_guess_family(d.name),
                params=_guess_params(d.name),
            ))
        return models

    def scan_ollama(self) -> list[LocalModel]:
        """Return models found in the Ollama manifest directory."""
        if not self._ollama_dir.is_dir():
            return []
        models: list[LocalModel] = []
        for model_dir in sorted(self._ollama_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            model_name = model_dir.name
            for tag_file in model_dir.iterdir():
                if not tag_file.is_file():
                    continue
                tag  = tag_file.name
                name = f"{model_name}:{tag}"
                size = 0
                try:
                    manifest = json.loads(tag_file.read_text(encoding="utf-8"))
                    for layer in manifest.get("layers", []):
                        size += layer.get("size", 0)
                except Exception:
                    pass
                models.append(LocalModel(
                    name=name,
                    path=tag_file,
                    source="ollama",
                    size_bytes=size,
                    family=_guess_family(model_name),
                    params=_guess_params(model_name),
                ))
        return models

    def scan_lm_studio(self) -> list[LocalModel]:
        """Return GGUF and safetensors models found in the LM Studio cache.

        LM Studio stores models at ``<lm_studio_dir>/<publisher>/<repo>/``.
        Model names are derived as ``publisher/repo`` to mirror LM Studio's
        own naming convention.  GGUF files are single-file models; safetensors
        repos are identified by the presence of a ``model.safetensors`` or
        ``model.safetensors.index.json`` file inside the repo directory.

        The ``LMSTUDIO_MODELS_DIR`` environment variable overrides the default
        scan root, allowing users with non-standard install paths to be detected.
        """
        import os

        # Honour env override (set by user or by test fixtures)
        env_override = os.environ.get("LMSTUDIO_MODELS_DIR", "")
        root = Path(env_override) if env_override else self._lm_studio_dir
        if not root.is_dir():
            return []

        models: list[LocalModel] = []
        seen_paths: set[Path] = set()

        # ── GGUF single-file models ──────────────────────────────────────────
        for gguf in sorted(root.rglob("*.gguf")):
            if gguf in seen_paths:
                continue
            seen_paths.add(gguf)
            # Derive a human-readable name from the directory path relative
            # to the scan root.  LM Studio layout: root/publisher/repo/file.gguf
            # → name = "publisher/repo"
            try:
                rel_parts = gguf.relative_to(root).parts
                if len(rel_parts) >= 3:
                    # publisher/repo/model.gguf  or deeper
                    lm_name = f"{rel_parts[0]}/{rel_parts[1]}"
                elif len(rel_parts) == 2:
                    lm_name = rel_parts[0]
                else:
                    lm_name = gguf.stem.lower()
            except ValueError:
                lm_name = gguf.stem.lower()

            models.append(LocalModel(
                name=lm_name,
                path=gguf,
                source="lm_studio",
                size_bytes=gguf.stat().st_size,
                family=_guess_family(gguf.stem),
                params=_guess_params(gguf.stem),
            ))

        # ── Safetensors repo models ──────────────────────────────────────────
        # A safetensors model is a directory containing model.safetensors or
        # model.safetensors.index.json (sharded).
        _ST_MARKERS = ("model.safetensors", "model.safetensors.index.json")
        for candidate in sorted(root.rglob("model.safetensors*")):
            if not candidate.is_file():
                continue
            repo_dir = candidate.parent
            if repo_dir in seen_paths:
                continue
            seen_paths.add(repo_dir)
            try:
                rel_parts = repo_dir.relative_to(root).parts
                if len(rel_parts) >= 2:
                    lm_name = f"{rel_parts[0]}/{rel_parts[1]}"
                elif len(rel_parts) == 1:
                    lm_name = rel_parts[0]
                else:
                    lm_name = repo_dir.name.lower()
            except ValueError:
                lm_name = repo_dir.name.lower()

            models.append(LocalModel(
                name=lm_name,
                path=repo_dir,
                source="lm_studio",
                size_bytes=_dir_size(repo_dir),
                family=_guess_family(repo_dir.name),
                params=_guess_params(repo_dir.name),
            ))

        return models

    def find_all(self) -> list[LocalModel]:
        """Return merged, deduplicated results from all scanners.

        Deduplication is by canonical ``name`` — the first occurrence wins
        (squish > ollama > lm_studio priority order).
        """
        seen: set[str] = set()
        merged: list[LocalModel] = []
        for model in (
            self.scan_squish() + self.scan_ollama() + self.scan_lm_studio()
        ):
            if model.name not in seen:
                seen.add(model.name)
                merged.append(model)
        return merged


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _dir_to_canonical(dir_name: str) -> str:
    """Convert a Squish model directory name to a canonical ``family:size`` id.

    Examples::

        "Qwen3-8B-bf16"            → "qwen3:8b"
        "Llama-3.1-8B-Instruct-bf16" → "llama3.1:8b"
    """
    import re
    name = dir_name.lower()
    # Strip common suffixes
    for suffix in ("-bf16", "-int4", "-int2", "-int8", "-instruct", "-chat",
                   "-squished", "-squish", "-mlx", "-gguf"):
        name = name.replace(suffix, "")
    # Replace last hyphen before a size pattern with colon
    name = re.sub(r"-(\d+\.?\d*b)(\b|$)", r":\1", name)
    # Clean up remaining hyphens between version/name parts → keep single hyphen
    name = re.sub(r"-+", "-", name).strip("-")
    return name
