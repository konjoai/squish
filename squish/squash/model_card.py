"""squish/squash/model_card.py — AI regulation–compliant model card generator.

Generates structured model documentation from existing squash attestation
artifacts (ML-BOM, scan results, policy results, VEX reports).  Three output
formats are supported:

    card = ModelCardGenerator(model_dir=Path("./my-model"))
    card.generate("hf")          # → squash-model-card-hf.md
    card.generate("eu-ai-act")   # → squash-model-card-euaiact.md
    card.generate("iso-42001")   # → squash-model-card-iso42001.md
    card.generate("all")         # → all three formats

Supported formats
-----------------
``hf``
    HuggingFace model card: YAML frontmatter + standard README.md sections.
    Compatible with ``huggingface_hub`` model card schema.

``eu-ai-act``
    EU AI Act Article 13 technical documentation template, covering:
    Art. 13(1) AI system identification, Art. 13(2) capabilities/limitations,
    Art. 13(3) human oversight measures, Art. 13(4) technical robustness.

``iso-42001``
    ISO/IEC 42001:2023 AI Management System record template, covering:
    Clause 6.1 risk assessment, Clause 8.3 documentation, Clause 8.4
    lifecycle, Clause 9.1 performance monitoring, Clause 10 improvement.

All formats are **stdlib-only** — no external dependencies required.
Artifacts (ML-BOM, scan result, policy reports, VEX report, squish.json)
are loaded if present in ``model_dir``; missing artifacts are handled
gracefully with placeholder text.
"""

from __future__ import annotations

import datetime
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

KNOWN_FORMATS: frozenset[str] = frozenset({"hf", "eu-ai-act", "iso-42001", "all"})


# ── Configuration ────────────────────────────────────────────────────────────


@dataclass
class ModelCardConfig:
    """Configuration for the model card generator.

    Attributes:
        model_dir:     Directory containing squash attestation artifacts.
        model_id:      Override the model identifier (defaults to squish.json
                       ``model_id`` or the directory name).
        model_name:    Human-readable model name used in card titles.
        model_version: Version string (e.g. ``"1.0.0"``).
        language:      BCP-47 language codes for the ``language`` frontmatter field.
        license:       SPDX licence identifier for the ``license`` frontmatter field.
        tags:          Extra tags appended to the HF card ``tags`` list.
        output_dir:    Directory to write card files to (defaults to ``model_dir``).
    """

    model_dir: Path
    model_id: str = ""
    model_name: str = ""
    model_version: str = ""
    language: list[str] = field(default_factory=lambda: ["en"])
    license: str = "apache-2.0"
    tags: list[str] = field(default_factory=list)
    output_dir: Path | None = None


# ── Data classes ─────────────────────────────────────────────────────────────


@dataclass
class ModelCardSection:
    """A single section of a model card.

    Attributes:
        title:   Section heading text (without ``#`` characters).
        content: Markdown body for the section.
        level:   Markdown heading level (2 = ``##``, 3 = ``###``, …).
    """

    title: str
    content: str
    level: int = 2


@dataclass
class ModelCard:
    """A renderable, writable model card.

    Attributes:
        config:           Original ``ModelCardConfig``.
        fmt:              Output format identifier (``"hf"``, ``"eu-ai-act"``, etc.).
        yaml_frontmatter: Key-value pairs written as the YAML front matter block.
        sections:         Ordered list of ``ModelCardSection`` objects.
        generated_at:     ISO-8601 UTC timestamp of generation.
    """

    config: ModelCardConfig
    fmt: str
    yaml_frontmatter: dict[str, Any]
    sections: list[ModelCardSection]
    generated_at: str = ""

    # ── Rendering ────────────────────────────────────────────────────────────

    def render(self) -> str:
        """Return the full model card as a markdown string."""
        lines: list[str] = []

        # YAML front matter
        lines.append("---")
        for key, val in self.yaml_frontmatter.items():
            if isinstance(val, list):
                lines.append(f"{key}:")
                for item in val:
                    if isinstance(item, dict):
                        # first item gets "- key: value", rest are indented
                        pairs = list(item.items())
                        if pairs:
                            k0, v0 = pairs[0]
                            lines.append(f"  - {k0}: {v0}")
                            for ki, vi in pairs[1:]:
                                lines.append(f"    {ki}: {vi}")
                    else:
                        lines.append(f"  - {item}")
            elif isinstance(val, dict):
                lines.append(f"{key}:")
                for sk, sv in val.items():
                    lines.append(f"  {sk}: {sv}")
            elif isinstance(val, bool):
                lines.append(f"{key}: {'true' if val else 'false'}")
            elif isinstance(val, str) and (" " in val or ":" in val or not val):
                lines.append(f'{key}: "{val}"')
            else:
                lines.append(f"{key}: {val}")
        lines.append("---")
        lines.append("")

        # Sections
        for section in self.sections:
            prefix = "#" * section.level
            lines.append(f"{prefix} {section.title}")
            lines.append("")
            lines.append(section.content.rstrip())
            lines.append("")

        return "\n".join(lines)

    def write(self, output_dir: Path | None = None) -> Path:
        """Write the rendered card to disk.

        Args:
            output_dir: Directory to write to.  Falls back to
                        ``self.config.output_dir`` then ``self.config.model_dir``.

        Returns:
            The absolute path of the written file.
        """
        dest_dir = output_dir or self.config.output_dir or self.config.model_dir
        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        suffix_map = {"hf": "hf", "eu-ai-act": "euaiact", "iso-42001": "iso42001"}
        suffix = suffix_map.get(self.fmt, self.fmt.replace(" ", "-"))
        dest = dest_dir / f"squash-model-card-{suffix}.md"
        dest.write_text(self.render(), encoding="utf-8")
        return dest


# ── Generator ────────────────────────────────────────────────────────────────


class ModelCardGenerator:
    """Generate regulation-compliant model cards from squash attestation artifacts.

    Usage::

        gen = ModelCardGenerator(Path("./my-model"))
        paths = gen.generate("all")   # writes three .md files
        paths = gen.generate("hf")    # writes squash-model-card-hf.md only

    If squash attestation artifacts are not present in ``model_dir`` the
    generator degrades gracefully, using placeholder text.
    """

    _ARTIFACT_MAP: dict[str, str] = {
        "squish_json": "squish.json",
        "bom": "cyclonedx-mlbom.json",
        "scan": "squash-scan.json",
        "attest": "squash-attest.json",
        "vex": "squash-vex-report.json",
    }

    def __init__(
        self,
        model_dir: Path | str,
        config: ModelCardConfig | None = None,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.config = config or ModelCardConfig(model_dir=self.model_dir)
        self._artifacts: dict[str, Any] = {}
        self._load_artifacts()

    # ── Artifact loading ──────────────────────────────────────────────────────

    def _load_artifacts(self) -> None:
        """Load squash JSON artifacts from model_dir where present."""
        for key, filename in self._ARTIFACT_MAP.items():
            path = self.model_dir / filename
            if path.exists():
                try:
                    self._artifacts[key] = json.loads(path.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError) as exc:
                    log.debug("Could not load %s: %s", filename, exc)

        # Policy reports: can be multiple (squash-policy-<name>.json)
        policy_files = sorted(self.model_dir.glob("squash-policy-*.json"))
        if policy_files:
            self._artifacts["policies"] = []
            for pf in policy_files:
                try:
                    self._artifacts["policies"].append(
                        json.loads(pf.read_text(encoding="utf-8"))
                    )
                except (json.JSONDecodeError, OSError) as exc:
                    log.debug("Could not load %s: %s", pf.name, exc)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _model_id(self) -> str:
        if self.config.model_id:
            return self.config.model_id
        squish_json = self._artifacts.get("squish_json", {})
        return squish_json.get("model_id", self.model_dir.name)

    def _model_name(self) -> str:
        if self.config.model_name:
            return self.config.model_name
        squish_json = self._artifacts.get("squish_json", {})
        return squish_json.get("model_id", self.model_dir.name)

    def _quant_format(self) -> str:
        squish_json = self._artifacts.get("squish_json", {})
        return squish_json.get("quant_format", "unknown")

    def _bom_component(self) -> dict[str, Any]:
        bom = self._artifacts.get("bom", {})
        comps = bom.get("components", [])
        return comps[0] if comps else {}

    def _scan_summary(self) -> str:
        scan = self._artifacts.get("scan", {})
        findings = scan.get("findings", [])
        if not findings:
            return "No security findings detected."
        errors = [f for f in findings if f.get("severity") == "error"]
        warnings = [f for f in findings if f.get("severity") == "warning"]
        parts = []
        if errors:
            parts.append(f"{len(errors)} error(s)")
        if warnings:
            parts.append(f"{len(warnings)} warning(s)")
        summary = ", ".join(parts) if parts else str(len(findings))
        return f"{len(findings)} finding(s): {summary}"

    def _policy_summary(self) -> str:
        policies = self._artifacts.get("policies", [])
        if not policies:
            return "No policy evaluations on record."
        passing = sum(1 for p in policies if p.get("passed", False))
        return f"{passing}/{len(policies)} policy evaluations passing."

    def _vex_summary(self) -> str:
        vex = self._artifacts.get("vex", {})
        stmts = vex.get("statements", [])
        if not stmts:
            return "No CVEs declared in scope."
        not_affected = sum(1 for s in stmts if s.get("status") == "not_affected")
        return f"{len(stmts)} CVE(s) evaluated; {not_affected} not affected."

    @staticmethod
    def _iso_date() -> str:
        return datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d")

    @staticmethod
    def _utc_now_iso() -> str:
        return datetime.datetime.now(datetime.UTC).isoformat()

    # ── Format builders ───────────────────────────────────────────────────────

    def _build_hf_card(self) -> ModelCard:
        """Build a HuggingFace-compatible model card."""
        model_name = self._model_name()
        quant = self._quant_format()
        comp = self._bom_component()

        # Collect performance metrics from BOM modelCard if available
        perf_metrics: list[dict[str, Any]] = []
        if comp:
            mc = comp.get("modelCard", {})
            qa = mc.get("quantitativeAnalysis", {})
            for m in qa.get("performanceMetrics", []):
                if m.get("type") and m.get("value") is not None:
                    perf_metrics.append({"type": m["type"], "value": m["value"]})

        frontmatter: dict[str, Any] = {
            "model_id": self._model_id(),
            "language": self.config.language,
            "license": self.config.license,
            "tags": [*self.config.tags, "squash-attested", quant.lower()],
            "pipeline_tag": "text-generation",
        }
        if perf_metrics:
            frontmatter["model-index"] = [
                {
                    "name": model_name,
                    "results": [
                        {"metrics": [{"type": m["type"], "value": m["value"]} for m in perf_metrics]}
                    ],
                }
            ]

        sections = [
            ModelCardSection(
                f"Model Card — {model_name}",
                f"This model card was generated by [Squish Squash](https://squish.ai/squash) "
                f"on {self._iso_date()}.",
                level=1,
            ),
            ModelCardSection(
                "Model Details",
                f"| Field | Value |\n|---|---|\n"
                f"| Model ID | `{self._model_id()}` |\n"
                f"| Format | {quant} |\n"
                f"| Compression tool | Squish |\n"
                f"| Attestation | Squash AI-SBOM |",
            ),
            ModelCardSection(
                "Security Assessment",
                f"| Dimension | Result |\n|---|---|\n"
                f"| Scan result | {self._scan_summary()} |\n"
                f"| Policy compliance | {self._policy_summary()} |\n"
                f"| VEX report | {self._vex_summary()} |",
            ),
            ModelCardSection(
                "Intended Use",
                "This model is intended for text generation tasks. "
                "Review the security scan and policy results before deploying in production.",
            ),
            ModelCardSection(
                "Limitations",
                "Quantised models may exhibit reduced accuracy compared to the BF16 reference. "
                "Always validate on your target task before deployment.",
            ),
            ModelCardSection(
                "How to Use",
                f"```bash\n# Serve with squish\nsquish run {self._model_id()}\n```",
            ),
        ]

        return ModelCard(
            config=self.config,
            fmt="hf",
            yaml_frontmatter=frontmatter,
            sections=sections,
            generated_at=self._utc_now_iso(),
        )

    def _build_euaiact_card(self) -> ModelCard:
        """Build an EU AI Act Article 13 technical documentation card."""
        model_name = self._model_name()
        comp = self._bom_component()

        provider = (
            comp.get("supplier", {}).get("name", "Not specified") if comp else "Not specified"
        )
        hashes = comp.get("hashes", []) if comp else []
        hash_str = (
            "; ".join(f"{h['alg']}: {h['content'][:16]}…" for h in hashes[:2])
            if hashes
            else "Not available"
        )

        frontmatter: dict[str, Any] = {
            "regulation": "EU AI Act",
            "article": "Art. 13 — Transparency and Provision of Information to Deployers",
            "model_id": self._model_id(),
            "generated_date": self._iso_date(),
            "squash_attested": True,
        }

        sections = [
            ModelCardSection(
                f"EU AI Act Technical Documentation — {model_name}",
                f"Technical documentation generated by Squish Squash on {self._iso_date()} "
                f"per **EU AI Act Article 13** (Transparency and Provision of Information).",
                level=1,
            ),
            ModelCardSection(
                "Art. 13(1) — AI System Identification",
                f"| Field | Value |\n|---|---|\n"
                f"| Model ID | `{self._model_id()}` |\n"
                f"| Format | {self._quant_format()} |\n"
                f"| Provider | {provider} |\n"
                f"| Integrity Hash | {hash_str} |",
            ),
            ModelCardSection(
                "Art. 13(2) — Capabilities and Limitations",
                f"- **Task type**: Text generation\n"
                f"- **Quantization format**: {self._quant_format()}\n"
                f"- **Accuracy note**: Quantised models may exhibit reduced task accuracy "
                f"vs the BF16 reference. Always validate on your target task before deployment.\n"
                f"- **Scan result**: {self._scan_summary()}\n"
                f"- **Policy compliance**: {self._policy_summary()}",
            ),
            ModelCardSection(
                "Art. 13(3) — Human Oversight Measures",
                "- Model outputs must be reviewed by qualified personnel before acting on them.\n"
                "- Automated decision pipelines must implement a human-in-the-loop checkpoint "
                "for high-risk decisions.\n"
                "- Deployers must log all decisions made using this AI system output.",
            ),
            ModelCardSection(
                "Art. 13(4) — Technical Robustness and Security",
                f"- **VEX vulnerability status**: {self._vex_summary()}\n"
                "- Model weights are attested using Squash AI-SBOM.\n"
                "- All weight files are integrity-checked via SHA-256 at load time.",
            ),
            ModelCardSection(
                "Art. 17 — Quality Management System",
                "- **Attestation artifacts**: `squash-attest.json`, `cyclonedx-mlbom.json`, "
                "`squash-scan.json`\n"
                "- Compliance policies are evaluated at attestation time.\n"
                "- Change log maintained via Squash lineage tracking (`squash-lineage.json`).",
            ),
        ]

        return ModelCard(
            config=self.config,
            fmt="eu-ai-act",
            yaml_frontmatter=frontmatter,
            sections=sections,
            generated_at=self._utc_now_iso(),
        )

    def _build_iso42001_card(self) -> ModelCard:
        """Build an ISO/IEC 42001:2023 AI Management System documentation card."""
        model_name = self._model_name()

        frontmatter: dict[str, Any] = {
            "standard": "ISO/IEC 42001:2023",
            "document_type": "AI System Technical Record",
            "model_id": self._model_id(),
            "review_date": self._iso_date(),
            "classification": "INTERNAL",
        }

        sections = [
            ModelCardSection(
                f"ISO 42001 AI System Record — {model_name}",
                f"AI management system documentation generated by Squish Squash on "
                f"{self._iso_date()} per **ISO/IEC 42001:2023** (AI Management Systems).",
                level=1,
            ),
            ModelCardSection(
                "Clause 6.1 — AI Risk Assessment",
                f"| Risk Dimension | Status |\n|---|---|\n"
                f"| Security scan | {self._scan_summary()} |\n"
                f"| Policy compliance | {self._policy_summary()} |\n"
                f"| Vulnerability exposure | {self._vex_summary()} |\n"
                f"| Bias / fairness | Not assessed automatically — manual review required |",
            ),
            ModelCardSection(
                "Clause 8.3 — AI System Documentation",
                f"- **Model ID**: `{self._model_id()}`\n"
                f"- **Format**: {self._quant_format()}\n"
                "- **Attestation artifacts**: `squash-attest.json`\n"
                "- **Provenance record**: `squash-provenance.json` (if generated)\n"
                "- **Lineage chain**: `squash-lineage.json` (if generated)",
            ),
            ModelCardSection(
                "Clause 8.4 — AI System Lifecycle",
                "| Phase | Artefact | Status |\n|---|---|---|\n"
                "| Compress | `squish.json` | ✅ Generated at compress time |\n"
                "| Attest | `squash-attest.json` | See attestation record |\n"
                "| Scan | `squash-scan.json` | See scan result |\n"
                "| Policy | `squash-policy-*.json` | See policy reports |\n"
                "| Deploy | CI/CD hook | Operator responsibility |",
            ),
            ModelCardSection(
                "Clause 9.1 — Performance Monitoring",
                "- Drift detection: use `squash drift-check` to compare SBOM versions.\n"
                "- Runtime monitoring: configure `squash monitor` for inference-time alerts.\n"
                "- Periodic re-attestation: recommended every 90 days or on weight updates.",
            ),
            ModelCardSection(
                "Clause 10 — Improvement",
                "- Remediation: use `squash remediate` to convert unsafe weight formats.\n"
                "- Policy updates: use `squash policies --validate` to preview policy changes.\n"
                "- Incident log: all policy violations are recorded in `squash-policy-*.json`.",
            ),
        ]

        return ModelCard(
            config=self.config,
            fmt="iso-42001",
            yaml_frontmatter=frontmatter,
            sections=sections,
            generated_at=self._utc_now_iso(),
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def generate(
        self,
        fmt: str = "hf",
        output_dir: Path | str | None = None,
    ) -> list[Path]:
        """Generate model card(s) and write them to disk.

        Args:
            fmt:        Output format.  One of ``"hf"``, ``"eu-ai-act"``,
                        ``"iso-42001"``, or ``"all"``.
            output_dir: Directory for output files.  Falls back to
                        ``ModelCardConfig.output_dir`` then ``model_dir``.

        Returns:
            List of absolute ``Path`` objects for all written files.

        Raises:
            ValueError: If ``fmt`` is not a known format identifier.
        """
        if fmt not in KNOWN_FORMATS:
            raise ValueError(
                f"Unknown format {fmt!r}. Must be one of: {sorted(KNOWN_FORMATS)}"
            )

        fmts = ["hf", "eu-ai-act", "iso-42001"] if fmt == "all" else [fmt]
        written: list[Path] = []
        out = Path(output_dir) if output_dir else None

        for f in fmts:
            card = self._build_card(f)
            path = card.write(output_dir=out)
            written.append(path)
            log.info("model-card written: %s", path)

        return written

    def _build_card(self, fmt: str) -> ModelCard:
        builders = {
            "hf": self._build_hf_card,
            "eu-ai-act": self._build_euaiact_card,
            "iso-42001": self._build_iso42001_card,
        }
        builder = builders.get(fmt)
        if builder is None:
            raise ValueError(f"No builder for format {fmt!r}")
        return builder()
