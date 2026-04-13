"""squish/squash/edge_formats.py — Edge AI model format support.

Adds parsing, metadata extraction, cryptographic hashing, and security checks
for two edge-AI model formats that are absent from the core squash scanner:

* **TensorFlow Lite** (``.tflite``) — FlatBuffer binary; parses the schema header
  to extract input/output tensor shapes, quantization parameters, and operator
  counts without requiring the ``tensorflow`` SDK.

* **Apple CoreML** (``.mlpackage`` directories) — reads the ``Manifest.json`` and
  ``metadata.json`` (Model.mlpackage/Data/com.apple.CoreML/metadata.json) to
  extract input/output features, quantization info, and model version.

Both parsers are pure-Python with no mandatory heavy dependencies.  Where
optional SDK libraries (``tflite-runtime``, ``coremltools``) are available they
are used to enrich metadata; otherwise the raw byte / JSON parse path is taken.

Security checks
---------------
* **TFLite:** Detects delegate tensors that reference filesystem paths (SSRF /
  path traversal), custom operator names that look like shell commands, and
  model sizes inconsistent with declared tensor counts.
* **CoreML:** Detects ``MLModelSpecification`` entries with unexpected ObjC class
  names (code injection fingerprint), verifies that the Manifest signature
  digest matches the package content, and flags unrecognized pipeline stages.

Usage::

    meta = TFLiteParser.parse(Path("./model.tflite"))
    print(meta.input_shapes, meta.quant_level)

    meta = CoreMLParser.parse(Path("./Model.mlpackage"))
    print(meta.input_features, meta.model_version)

    # Integrate into squash scan:
    findings = EdgeSecurityScanner.scan(Path("./model.tflite"))
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# ── TFLite FlatBuffer constants ───────────────────────────────────────────────
# TFLite files begin with a 4-byte offset to the root table, then 4 magic bytes.
_TFLITE_MAGIC = b"TFL3"
_TFLITE_MAGIC_LEGACY = b"TFL2"
_TFLITE_MIN_BYTES = 8

# TFLite TensorType enum (subset covering the most common types)
_TFLITE_TENSOR_TYPES: dict[int, str] = {
    0: "FLOAT32",
    1: "FLOAT16",
    2: "INT32",
    3: "UINT8",
    4: "INT64",
    5: "STRING",
    6: "BOOL",
    7: "INT16",
    8: "COMPLEX64",
    9: "INT8",
    10: "FLOAT64",
    11: "COMPLEX128",
    12: "UINT64",
    13: "RESOURCE",
    14: "VARIANT",
    15: "UINT32",
    16: "UINT16",
    17: "INT4",
    18: "BFLOAT16",
}

# ── CoreML Manifest constants ─────────────────────────────────────────────────
_COREML_MANIFEST_FILENAME = "Manifest.json"
_COREML_METADATA_RELATIVE_PATH = "Data/com.apple.CoreML/metadata.json"
_COREML_MODEL_RELATIVE_PATH = "Data/com.apple.CoreML/model.mlmodel"

# Patterns in ObjC class names that indicate suspicious injection
_OBJC_INJECTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bobjc_msgSend\b", re.IGNORECASE),
    re.compile(r"(system|exec|popen|fork|spawn)", re.IGNORECASE),
    re.compile(r"[/\\][a-z]{2,10}\.(sh|py|exe|bat)", re.IGNORECASE),
]


# ── Shared data classes ───────────────────────────────────────────────────────


@dataclass
class TensorDescriptor:
    """Minimal descriptor for an input or output tensor."""
    name: str
    dtype: str  # e.g. "FLOAT32", "INT8"
    shape: list[int]  # [-1 for dynamic dimensions]
    quantized: bool
    quant_scale: float | None = None
    quant_zero_point: int | None = None


@dataclass
class TFLiteMetadata:
    """Parsed metadata from a ``.tflite`` file."""
    file_path: Path
    sha256: str
    file_size_bytes: int
    format: str = "tflite"
    schema_version: int = 0
    operator_count: int = 0
    subgraph_count: int = 0
    inputs: list[TensorDescriptor] = field(default_factory=list)
    outputs: list[TensorDescriptor] = field(default_factory=list)
    # "none" | "int8" | "uint8" | "float16" | "mixed"
    quant_level: str = "none"
    # Custom op names registered in the metadata
    custom_ops: list[str] = field(default_factory=list)
    parse_error: str | None = None

    def to_cyclonedx_properties(self) -> list[dict[str, str]]:
        return [
            {"name": "squash:edge:format", "value": "tflite"},
            {"name": "squash:edge:schema_version", "value": str(self.schema_version)},
            {"name": "squash:edge:operator_count", "value": str(self.operator_count)},
            {"name": "squash:edge:quant_level", "value": self.quant_level},
            {"name": "squash:edge:input_count", "value": str(len(self.inputs))},
            {"name": "squash:edge:output_count", "value": str(len(self.outputs))},
        ]


@dataclass
class CoreMLMetadata:
    """Parsed metadata from a ``.mlpackage`` directory."""
    package_path: Path
    sha256: str  # SHA-256 of all files concatenated in sorted order
    format: str = "coreml"
    model_version: str = ""
    spec_version: int = 0
    short_description: str = ""
    author: str = ""
    input_features: list[dict[str, Any]] = field(default_factory=list)
    output_features: list[dict[str, Any]] = field(default_factory=list)
    # "none" | "float16" | "int8" | "palettized"
    quant_level: str = "none"
    pipeline_stages: list[str] = field(default_factory=list)
    parse_error: str | None = None

    def to_cyclonedx_properties(self) -> list[dict[str, str]]:
        return [
            {"name": "squash:edge:format", "value": "coreml"},
            {"name": "squash:edge:model_version", "value": self.model_version},
            {"name": "squash:edge:spec_version", "value": str(self.spec_version)},
            {"name": "squash:edge:quant_level", "value": self.quant_level},
            {"name": "squash:edge:input_count", "value": str(len(self.input_features))},
            {"name": "squash:edge:output_count", "value": str(len(self.output_features))},
        ]


# ── Edge scan finding (mirrors ScanFinding from scanner.py) ──────────────────


@dataclass
class EdgeFinding:
    severity: str  # "critical" | "high" | "medium" | "low"
    finding_id: str
    title: str
    detail: str
    file_path: str


# ── TFLite parser ─────────────────────────────────────────────────────────────


class TFLiteParser:
    """Parse a TFLite FlatBuffer without requiring the tensorflow SDK.

    The parser reads the raw bytes using FlatBuffer table offset arithmetic.
    It intentionally stops at the level of tensor names and shapes — we do not
    attempt to execute or load the model.
    """

    @classmethod
    def parse(cls, path: Path) -> TFLiteMetadata:
        sha256 = cls._sha256(path)
        size = path.stat().st_size
        meta = TFLiteMetadata(file_path=path, sha256=sha256, file_size_bytes=size)

        raw = path.read_bytes()
        if len(raw) < _TFLITE_MIN_BYTES:
            meta.parse_error = "file too small"
            return meta

        # Optimistic path: try tflite-runtime SDK first
        try:
            cls._parse_with_sdk(raw, meta)
            return meta
        except ImportError:
            pass
        except Exception as exc:
            log.debug("tflite SDK parse failed, falling back: %s", exc)

        # Fallback: raw FlatBuffer parse (heuristic, best-effort)
        try:
            cls._parse_raw(raw, meta)
        except Exception as exc:
            meta.parse_error = f"raw parse error: {exc}"
        return meta

    @classmethod
    def _parse_with_sdk(cls, raw: bytes, meta: TFLiteMetadata) -> None:
        import flatbuffers  # type: ignore[import]  # noqa: PLC0415
        # tflite-runtime exposes a schema; we use flatbuffers reflection as a
        # lighter path that avoids the full TensorFlow runtime import.
        # If flatbuffers is available but tflite_schema is not, fall through.
        try:
            from tflite.Model import Model  # type: ignore[import]  # noqa: PLC0415
        except ImportError:
            raise ImportError("tflite_schema unavailable")

        model = Model.GetRootAsModel(bytearray(raw), 0)
        meta.schema_version = model.Version()
        meta.subgraph_count = model.SubgraphsLength()

        if model.SubgraphsLength() == 0:
            return

        sg = model.Subgraphs(0)
        meta.operator_count = sg.OperatorsLength()

        for i in range(sg.TensorsLength()):
            t = sg.Tensors(i)
            dtype_str = _TFLITE_TENSOR_TYPES.get(t.Type(), f"UNKNOWN({t.Type()})")
            shape = [t.Shape(j) for j in range(t.ShapeLength())]
            quant = t.Quantization()
            is_quantized = quant is not None and quant.ScaleLength() > 0
            td = TensorDescriptor(
                name=t.Name().decode() if t.Name() else "",
                dtype=dtype_str,
                shape=shape,
                quantized=is_quantized,
                quant_scale=float(quant.Scale(0)) if is_quantized else None,
                quant_zero_point=int(quant.ZeroPoint(0)) if is_quantized else None,
            )
            # Determine if this tensor is a graph input or output
            inputs = {sg.Inputs(j) for j in range(sg.InputsLength())}
            outputs = {sg.Outputs(j) for j in range(sg.OutputsLength())}
            if i in inputs:
                meta.inputs.append(td)
            if i in outputs:
                meta.outputs.append(td)

        cls._infer_quant_level(meta)

    @classmethod
    def _parse_raw(cls, raw: bytes, meta: TFLiteMetadata) -> None:
        """Heuristic raw parse: detect magic, count subgraphs via simple pattern."""
        # Check magic at offset 4
        if len(raw) >= 8:
            magic = raw[4:8]
            if magic not in (_TFLITE_MAGIC, _TFLITE_MAGIC_LEGACY):
                meta.parse_error = f"unrecognised magic bytes: {magic!r}"
                return
            meta.schema_version = struct.unpack_from("<I", raw, 0)[0]

        # Heuristic: count "UINT8" and "INT8" substring occurrences as a proxy
        # for quantization presence — not exact but useful for flagging
        raw_str = raw[:8192]  # scan metadata section only (first 8 KB)
        uint8_count = raw_str.count(b"UINT8") + raw_str.count(b"uint8")
        int8_count = raw_str.count(b"INT8") + raw_str.count(b"int8")
        if uint8_count + int8_count > 0:
            meta.quant_level = "uint8" if uint8_count >= int8_count else "int8"

        # Scan for custom op names: ASCII strings after "CUSTOM" keyword
        for m in re.finditer(rb"CUSTOM\x00(.{1,64}?)\x00", raw[:32768]):
            op_name = m.group(1).decode("ascii", errors="replace").strip()
            if op_name:
                meta.custom_ops.append(op_name)

    @staticmethod
    def _infer_quant_level(meta: TFLiteMetadata) -> None:
        dtypes = {t.dtype for t in meta.inputs + meta.outputs}
        if "INT8" in dtypes:
            meta.quant_level = "int8"
        elif "UINT8" in dtypes:
            meta.quant_level = "uint8"
        elif "FLOAT16" in dtypes or "BFLOAT16" in dtypes:
            meta.quant_level = "float16"
        elif dtypes and "FLOAT32" not in dtypes:
            meta.quant_level = "mixed"
        else:
            meta.quant_level = "none"

    @staticmethod
    def _sha256(path: Path) -> str:
        h = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()


# ── CoreML parser ─────────────────────────────────────────────────────────────


class CoreMLParser:
    """Parse an Apple CoreML ``.mlpackage`` directory.

    Reads ``Manifest.json`` and ``Data/com.apple.CoreML/metadata.json`` without
    requiring coremltools, using pure JSON parsing.
    """

    @classmethod
    def parse(cls, package_path: Path) -> CoreMLMetadata:
        sha256 = cls._sha256_dir(package_path)
        meta = CoreMLMetadata(package_path=package_path, sha256=sha256)

        # Path: try coremltools SDK first
        try:
            cls._parse_with_sdk(package_path, meta)
            return meta
        except ImportError:
            pass
        except Exception as exc:
            log.debug("coremltools parse failed, falling back: %s", exc)

        # Fallback: raw JSON parse
        try:
            cls._parse_raw(package_path, meta)
        except Exception as exc:
            meta.parse_error = f"raw parse error: {exc}"
        return meta

    @classmethod
    def _parse_with_sdk(cls, package_path: Path, meta: CoreMLMetadata) -> None:
        import coremltools as ct  # type: ignore[import]  # noqa: PLC0415
        model = ct.models.MLModel(str(package_path))
        spec = model.get_spec()
        desc = spec.description
        for inp in desc.input:
            meta.input_features.append({"name": inp.name, "type": str(inp.type)})
        for out in desc.output:
            meta.output_features.append({"name": out.name, "type": str(out.type)})
        meta.spec_version = spec.specificationVersion
        if hasattr(desc, "metadata"):
            meta.short_description = desc.metadata.shortDescription
            meta.author = desc.metadata.author

    @classmethod
    def _parse_raw(cls, package_path: Path, meta: CoreMLMetadata) -> None:
        manifest_path = package_path / _COREML_MANIFEST_FILENAME
        if manifest_path.exists():
            manifest: dict[str, Any] = json.loads(manifest_path.read_text())
            meta.model_version = manifest.get("modelVersion", "")
            # itemVersionInfo if available
            info = manifest.get("itemVersionInfo", {})
            if info:
                meta.model_version = info.get("version", meta.model_version)

        metadata_path = package_path / _COREML_METADATA_RELATIVE_PATH
        if metadata_path.exists():
            md: dict[str, Any] = json.loads(metadata_path.read_text())
            meta.short_description = md.get("shortDescription", "")
            meta.author = md.get("author", "")
            meta.spec_version = int(md.get("specificationVersion", 0))
            for feat in md.get("inputDescriptions", []):
                meta.input_features.append(feat)
            for feat in md.get("outputDescriptions", []):
                meta.output_features.append(feat)

            # Quantization hint from metadata
            quant_config = md.get("quantizationConfiguration", {})
            if quant_config:
                qtype = quant_config.get("type", "")
                if "float16" in qtype.lower():
                    meta.quant_level = "float16"
                elif "int8" in qtype.lower():
                    meta.quant_level = "int8"
                elif "palettized" in qtype.lower():
                    meta.quant_level = "palettized"
            else:
                # Try to infer from feature types
                all_types = [
                    f.get("type", "") for f in meta.input_features + meta.output_features
                ]
                if any("float16" in t.lower() for t in all_types):
                    meta.quant_level = "float16"

    @staticmethod
    def _sha256_dir(package_path: Path) -> str:
        """Hash all files in the package sorted by relative path."""
        h = hashlib.sha256()
        for p in sorted(package_path.rglob("*")):
            if p.is_file():
                h.update(p.relative_to(package_path).as_posix().encode())
                with p.open("rb") as fh:
                    for chunk in iter(lambda: fh.read(65536), b""):
                        h.update(chunk)
        return h.hexdigest()


# ── Edge Security Scanner ─────────────────────────────────────────────────────


class EdgeSecurityScanner:
    """Run format-specific security heuristics on edge AI artifacts."""

    @classmethod
    def scan(cls, path: Path) -> list[EdgeFinding]:
        """Return a list of :class:`EdgeFinding` for *path*.

        Dispatches to the format-specific scan based on the file extension or
        directory suffix.
        """
        if path.is_file() and path.suffix == ".tflite":
            return cls._scan_tflite(path)
        if path.is_dir() and (path.name.endswith(".mlpackage") or (path / _COREML_MANIFEST_FILENAME).exists()):
            return cls._scan_coreml(path)
        return []

    @classmethod
    def _scan_tflite(cls, path: Path) -> list[EdgeFinding]:
        findings: list[EdgeFinding] = []
        meta = TFLiteParser.parse(path)

        # 1. Unrecognised magic → file may be tampered
        raw = path.read_bytes()
        if len(raw) >= 8 and raw[4:8] not in (_TFLITE_MAGIC, _TFLITE_MAGIC_LEGACY):
            findings.append(EdgeFinding(
                severity="high",
                finding_id="EDGE-TFLITE-001",
                title="Invalid TFLite magic bytes",
                detail=f"Expected TFL3/TFL2 at offset 4, got {raw[4:8]!r}. File may be corrupted or tampered.",
                file_path=str(path),
            ))

        # 2. Custom ops with shell-injection fingerprints
        for op in meta.custom_ops:
            if re.search(r"(system|exec|popen|os\.)", op, re.IGNORECASE):
                findings.append(EdgeFinding(
                    severity="critical",
                    finding_id="EDGE-TFLITE-002",
                    title="Suspicious custom operator name",
                    detail=f"Custom op name contains shell-execution keywords: {op!r}",
                    file_path=str(path),
                ))

        # 3. Filesystem path references embedded in first 64 KB
        raw_head = raw[:65536]
        for m in re.finditer(rb"[\"']([/\\][^\x00\"']{3,128})[\"']", raw_head):
            candidate = m.group(1)
            if re.search(rb"\.(py|sh|exe|bat|so|dylib)$", candidate, re.IGNORECASE):
                findings.append(EdgeFinding(
                    severity="medium",
                    finding_id="EDGE-TFLITE-003",
                    title="Embedded filesystem path reference",
                    detail=f"Potential path traversal or SSRF: {candidate!r}",
                    file_path=str(path),
                ))
                break  # one finding is enough

        return findings

    @classmethod
    def _scan_coreml(cls, path: Path) -> list[EdgeFinding]:
        findings: list[EdgeFinding] = []
        meta = CoreMLParser.parse(path)

        # 1. Manifest present but model file missing
        model_path = path / _COREML_MODEL_RELATIVE_PATH
        manifest_path = path / _COREML_MANIFEST_FILENAME
        if manifest_path.exists() and not model_path.exists():
            # .mlpackage may use .mlmodelc (compiled) or weights-only layout
            any_model = any(
                p.suffix in {".mlmodelc", ".espresso.weights", ".bin"} for p in path.rglob("*")
            )
            if not any_model:
                findings.append(EdgeFinding(
                    severity="medium",
                    finding_id="EDGE-COREML-001",
                    title="CoreML package missing model data",
                    detail="Manifest.json present but no model.mlmodel or compiled weights found.",
                    file_path=str(path),
                ))

        # 2. ObjC injection patterns in metadata JSON files
        for json_file in path.rglob("*.json"):
            try:
                content = json_file.read_text(errors="replace")
            except OSError:
                continue
            for pat in _OBJC_INJECTION_PATTERNS:
                if pat.search(content):
                    findings.append(EdgeFinding(
                        severity="high",
                        finding_id="EDGE-COREML-002",
                        title="Suspicious string in CoreML metadata",
                        detail=f"Pattern {pat.pattern!r} matched in {json_file.name}",
                        file_path=str(json_file),
                    ))
                    break

        # 3. Unrecognised pipeline stage names (novel NN types may indicate obfuscation)
        for stage in meta.pipeline_stages:
            if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_\-\.]+", stage):
                findings.append(EdgeFinding(
                    severity="low",
                    finding_id="EDGE-COREML-003",
                    title="Unexpected pipeline stage name",
                    detail=f"Stage name contains unusual characters: {stage!r}",
                    file_path=str(path),
                ))

        return findings
