# EU AI Act Compliance-as-Code Platform — Implementation Plan
**Based on:** Squish/Squash repository analysis + Strategic research document  
**Date:** April 26, 2026  
**Author:** Wesley Scholl / Konjo AI  
**Status:** Strategic Blueprint

---

## Executive Summary

**Context:** The EU AI Act enforcement deadline (August 2, 2026) is 98 days away. The strategic research document identifies an **automated EU AI Act compliance platform** as the #1 blue-ocean opportunity. Your existing Squash compliance layer already implements 70% of the required foundation.

**Current State:** Squash has:
- ✅ CycloneDX 1.7 ML-BOM generation (ECMA-424 compliant)
- ✅ SPDX 2.3 SBOM generation
- ✅ EU AI Act policy templates (Articles 9, 11, 12, 13, 17)
- ✅ ModelScan integration for security scanning
- ✅ Sigstore signing + SLSA provenance
- ✅ Multi-tenant cloud API with compliance scoring
- ✅ VEX feed integration
- ✅ Risk tier classification (UNACCEPTABLE/HIGH/LIMITED/MINIMAL)

**The Gap:** Squash is a compliance *enforcement* layer. The strategic document calls for a compliance *automation* layer — CI/CD middleware that autonomously generates Annex IV technical documentation by extracting regulatory artifacts from the development lifecycle in real-time.

**Strategic Positioning:** Transform Squash from "post-deployment attestation tool" to "Compliance-as-Code CI/CD platform" — the GitGuardian/Snyk for AI governance.

---

## Current Architecture Analysis

### What Squash Already Does Well

| Component | Status | Coverage |
|-----------|--------|----------|
| **SBOM Generation** | ✅ Production | CycloneDX 1.7 ML-BOM with weight hashes, model metadata, quantization parameters |
| **Policy Enforcement** | ✅ Production | 10+ policy templates (EU AI Act, NIST AI RMF, OWASP LLM Top 10, ISO 42001, FedRAMP) |
| **Security Scanning** | ✅ Production | ModelScan integration; scans pickle, GGUF, ONNX, safetensors without execution |
| **Provenance** | ✅ Production | SLSA L3 provenance, Sigstore signing, lineage tracking |
| **Multi-Tenant API** | ✅ Production | 40+ REST endpoints for compliance scoring, VEX evaluation, attestation history |
| **Model Card Generation** | ✅ Production | HF, EU AI Act, ISO 42001 formats |
| **Risk Classification** | ✅ Production | EU AI Act Article 6 risk tiers with conformance verdicts |

### Critical Gaps for EU AI Act Article 11 / Annex IV

The research document states:

> "Article 11 mandates the continuous maintenance of exhaustive technical documentation outlined in Annex IV. This is not standard legal paperwork; it is a granular engineering audit requiring detailed explanations of algorithmic logic, dataset provenance, key design assumptions, hyperparameter configurations, and human oversight mechanisms."

**Current Squash coverage vs. Annex IV requirements:**

| Annex IV Requirement | Current Squash | Gap |
|---------------------|----------------|-----|
| **§1(a) General description** | ✅ `model_card.py` | None |
| **§1(b) Intended purpose** | ✅ `model_card.py` | None |
| **§1(c) Development information** | ⚠️ Partial | No training hyperparameters, no optimizer config, no data pipeline metadata |
| **§2(a) Data governance** | ❌ Missing | No training dataset provenance, no demographic distributions, no bias testing results |
| **§2(b) Data preprocessing** | ❌ Missing | No tokenization metadata, no normalization steps, no augmentation pipeline |
| **§3(a) Model architecture** | ✅ `sbom_builder.py` | None (architecture family detected) |
| **§3(b) Training methodology** | ❌ Missing | No loss curves, no validation metrics, no checkpoint strategy |
| **§4 Risk management** | ✅ `risk.py` | None |
| **§5 Human oversight** | ⚠️ Partial | `governor.py` enforces HITL but doesn't document override procedures |
| **§6(a) Performance metrics** | ⚠️ Partial | `eval_binder.py` appends lm_eval scores but not domain-specific benchmarks |
| **§6(b) Robustness testing** | ❌ Missing | No adversarial robustness, no OOD detection, no failure mode analysis |
| **§7 Lifecycle management** | ✅ `lineage.py` | None |

**Strategic conclusion:** Squash implements the **post-deployment enforcement layer** (Articles 9, 12, 14, 17) but lacks the **training-time artifact extraction layer** (Annex IV §1–3, §6b). This is the high-value opportunity.

---

## Strategic Blueprint: The Three-Layer Compliance Stack

### Layer 1: Artifact Extraction Engine (NEW — HIGH PRIORITY)

**Purpose:** Autonomously scrape technical documentation artifacts from the ML development lifecycle.

**Architecture:**
```
Training Code → Artifact Extractor → Structured Metadata → Annex IV Generator
     ↓                  ↓                    ↓                      ↓
  config.yaml    parse_hyperparams()    training_config.json   annex_iv_§1c.md
  data.py        extract_datasets()     data_provenance.json   annex_iv_§2a.md
  train.py       capture_metrics()      validation_results.json annex_iv_§3b.md
```

**Implementation components:**

#### 1.1 Training Config Parser (`squash/artifact_extractor.py`)
```python
class ArtifactExtractor:
    """Extract Annex IV artifacts from training codebases and experiment trackers."""
    
    @staticmethod
    def from_training_config(config_path: Path) -> TrainingArtifacts:
        """Parse training YAML/JSON → structured metadata.
        
        Extracts:
        - Optimizer (type, learning rate, momentum, weight decay)
        - Loss function
        - Batch size, gradient accumulation
        - Mixed precision config
        - Scheduler (type, warmup steps, decay rate)
        - Regularization (dropout, weight decay, gradient clipping)
        """
        
    @staticmethod
    def from_tensorboard_logs(log_dir: Path) -> TrainingMetrics:
        """Parse TensorBoard event files → loss curves + validation metrics."""
        
    @staticmethod
    def from_wandb_run(run_id: str, api_key: str) -> TrainingArtifacts:
        """Pull config + metrics from W&B API."""
        
    @staticmethod
    def from_mlflow_run(run_id: str, tracking_uri: str) -> TrainingArtifacts:
        """Pull params + metrics from MLflow Tracking."""
```

**Why this matters:** Annex IV §1(c) requires "the development process, including... the hyperparameters used, the training approaches and the validation and testing approaches." Current Squash has **zero** visibility into training runs. This component bridges that gap.

#### 1.2 Dataset Provenance Tracker (`squash/dataset_tracker.py`)
```python
class DatasetProvenanceBuilder:
    """Build Annex IV §2 data governance documentation."""
    
    @staticmethod
    def from_huggingface_dataset(dataset_name: str) -> DatasetProvenance:
        """Extract metadata from HF Datasets library.
        
        Returns:
        - Source URLs
        - License
        - Size (rows, tokens)
        - Language distribution
        - Split ratios (train/val/test)
        - Content warnings (if present in dataset card)
        """
        
    @staticmethod
    def from_local_dataset(data_dir: Path, schema_path: Path) -> DatasetProvenance:
        """Analyze local data directory → demographic distributions."""
        
    @staticmethod
    def bias_scan(dataset: Any, protected_attributes: list[str]) -> BiasReport:
        """Run statistical bias detection (using Aequitas or similar).
        
        Annex IV §2(a): "measures taken to examine the presence of possible 
        biases" — this provides the evidence.
        """
```

**Why this matters:** Annex IV §2(a) requires "the datasheets describing the training methodologies and techniques and the training data sets used, including a general description of these data sets, information about their provenance, scope and main characteristics." Current Squash has **zero** dataset tracking.

#### 1.3 Code Dependency Scanner (`squash/code_scanner.py`)
```python
class CodeDependencyScanner:
    """Scan training codebase for Annex IV §3 technical details."""
    
    @staticmethod
    def scan_training_script(script_path: Path) -> CodeArtifacts:
        """AST parse Python training script → extract:
        
        - Model class definition (architecture)
        - Loss function calls
        - Optimizer instantiation
        - Data loader config
        - Preprocessing transforms
        - Checkpoint save strategy
        """
        
    @staticmethod
    def scan_requirements(requirements_txt: Path) -> DependencyGraph:
        """Build software BOM for training environment.
        
        Addresses Annex IV §1(c): "the software used to design, test and 
        validate the AI model."
        """
```

**Why this matters:** Annex IV §1(c) requires documentation of "the software used to design, test and validate the AI model." Current Squash only tracks **inference-time** dependencies (via SBOM). Training-time dependencies are invisible.

### Layer 2: Annex IV Document Generator (NEW — HIGH PRIORITY)

**Purpose:** Transform extracted artifacts → auditor-ready Annex IV technical file.

**Architecture:**
```
Structured Metadata → Template Engine → Annex IV Markdown → PDF Export
        ↓                     ↓                  ↓               ↓
  artifacts.json      jinja2 render      annex_iv.md      annex_iv.pdf
```

**Implementation:**

#### 2.1 Template-Based Generator (`squash/annex_iv_generator.py`)
```python
class AnnexIVGenerator:
    """Generate EU AI Act Annex IV technical documentation."""
    
    REQUIRED_SECTIONS = [
        "1a_general_description",
        "1b_intended_purpose", 
        "1c_development_process",
        "2a_data_governance",
        "2b_data_preprocessing",
        "3a_model_architecture",
        "3b_training_methodology",
        "4_risk_management",
        "5_human_oversight",
        "6a_performance_metrics",
        "6b_robustness_testing",
        "7_lifecycle_management"
    ]
    
    def generate(self, artifacts: dict[str, Any]) -> AnnexIVDocument:
        """Generate complete Annex IV doc from structured artifacts.
        
        Returns:
        - Markdown document
        - Completeness score (% of required fields filled)
        - Missing field warnings
        """
        
    def validate_completeness(self, doc: AnnexIVDocument) -> ValidationReport:
        """Check if doc satisfies Annex IV minimum requirements.
        
        Hard fails:
        - Missing §1(a) general description
        - Missing §2(a) data provenance
        - Missing §3(a) model architecture
        
        Warnings:
        - Missing §6(b) robustness testing (common gap)
        """
```

**Why this matters:** The research document states:

> "For a typical mid-sized technology company, fulfilling the Annex IV documentation requirements manually necessitates months of dedicated engineering time and substantial expenditures on specialized legal consultants."

This component **eliminates that manual work** by auto-generating the technical file from machine-readable artifacts.

### Layer 3: CI/CD Integration Layer (EXTEND EXISTING)

**Purpose:** Embed compliance checks into every merge, deployment, and model registry push.

**Current Squash has:**
- ✅ GitHub Actions workflow templates (`integrations/github-actions/`)
- ✅ Jenkins integration (`integrations/jenkins/`)
- ✅ GitLab CI templates (`integrations/gitlab/`)
- ✅ Argo Workflows (`integrations/argo/`)

**Extend with:**

#### 3.1 Pre-Commit Hooks
```bash
# .git/hooks/pre-commit
#!/bin/bash
squash artifact-extract --training-config config.yaml
squash annex-iv-validate --artifacts ./artifacts.json

if [ $? -ne 0 ]; then
  echo "❌ Annex IV validation failed — commit blocked"
  exit 1
fi
```

#### 3.2 GitHub Actions Composite (Wave 84 in PLAN.md — already roadmapped!)
```yaml
# .github/actions/squash-compliance/action.yml
name: 'EU AI Act Compliance Check'
inputs:
  model-path:
    required: true
  training-config:
    required: true
  policy:
    default: 'eu-ai-act'
    
runs:
  using: "composite"
  steps:
    - name: Extract training artifacts
      run: squash artifact-extract --config ${{ inputs.training-config }}
      
    - name: Generate Annex IV document
      run: squash annex-iv-generate --artifacts ./artifacts.json --output ./annex_iv.md
      
    - name: Validate completeness
      run: squash annex-iv-validate --doc ./annex_iv.md
      
    - name: Policy check
      run: squash attest --model ${{ inputs.model-path }} --policy ${{ inputs.policy }}
```

#### 3.3 MLflow Integration (Wave 85 in PLAN.md — already roadmapped!)
**Current gap:** `attest-mlflow` only emits JSON. No real MLflow SDK integration.

**Extend to:**
```python
# squash/mlflow_bridge.py (Wave 85)
class MlflowBridge:
    def log_annex_iv_artifacts(self, run_id: str, artifacts: dict) -> None:
        """Log Annex IV structured metadata as MLflow run artifacts."""
        mlflow.log_dict(artifacts["training_config"], "annex_iv/1c_development.json")
        mlflow.log_dict(artifacts["data_provenance"], "annex_iv/2a_data_governance.json")
        mlflow.log_text(artifacts["annex_iv_markdown"], "annex_iv/technical_file.md")
        
    def set_compliance_tags(self, model_name: str, version: int) -> None:
        """Tag model version with compliance metadata."""
        mlflow.set_model_version_tag(
            name=model_name,
            version=version,
            key="eu_ai_act_compliant",
            value="true"
        )
        mlflow.set_model_version_tag(
            name=model_name,
            version=version, 
            key="annex_iv_completeness",
            value="92%"  # from validation report
        )
```

**Why this matters:** Enterprises use MLflow for model registry. Compliance metadata must live **inside** the registry, not in separate systems.

---

## Implementation Roadmap: 6-Week Sprint Plan

### Week 1-2: Artifact Extraction Engine
**Deliverables:**
1. `squash/artifact_extractor.py` — Training config parser
2. `squash/dataset_tracker.py` — HuggingFace Datasets integration
3. `squash/code_scanner.py` — AST-based dependency scanner
4. `tests/test_artifact_extraction.py` — ≥30 tests

**Acceptance criteria:**
- Extract hyperparameters from PyTorch Lightning, HF Trainer, Keras config files
- Pull training metrics from TensorBoard event files
- Generate dataset provenance JSON from HF Datasets metadata
- Scan Python training scripts → extract model class, loss function, optimizer

**Dependencies:**
- `tensorboard` (for event file parsing)
- `datasets` (for HF integration)
- `ast` (stdlib — for code scanning)

### Week 3-4: Annex IV Document Generator
**Deliverables:**
1. `squash/annex_iv_generator.py` — Template-based doc generator
2. `squash/data/annex_iv_template.jinja2` — Jinja2 template
3. `squash annex-iv-generate` CLI command
4. `squash annex-iv-validate` CLI command
5. `tests/test_annex_iv_generation.py` — ≥25 tests

**Acceptance criteria:**
- Generate Markdown document covering all 12 Annex IV sections
- Completeness score calculation (% of required fields present)
- Missing field warnings with remediation hints
- PDF export via `weasyprint` or `pandoc`

**Dependencies:**
- `jinja2` (for templating)
- `weasyprint` or `pandoc` (for PDF export — optional)

### Week 5: CI/CD Integration
**Deliverables:**
1. `.github/actions/squash-compliance/action.yml` — Composite action
2. `squash/mlflow_bridge.py` — Real MLflow SDK integration (Wave 85)
3. Updated GitHub Actions workflow examples
4. `tests/test_cicd_integration.py` — ≥15 tests

**Acceptance criteria:**
- Single `uses: ./.github/actions/squash-compliance` step runs full pipeline
- MLflow model versions tagged with `eu_ai_act_compliant: true`
- Annex IV artifacts logged to MLflow runs

**Dependencies:**
- `mlflow` (optional — only imported when used)

### Week 6: Multi-Tenant Cloud API Extensions
**Deliverables:**
1. `GET /cloud/tenants/{id}/annex-iv-completeness` — Per-tenant completeness dashboard
2. `GET /cloud/annex-iv-gaps` — Platform-wide Annex IV gap analysis
3. `POST /attest/auto-extract` — One-shot: extract artifacts + generate + attest
4. Updated API docs + Swagger UI
5. `tests/test_cloud_api_annex_iv.py` — ≥20 tests

**Acceptance criteria:**
- Multi-tenant API tracks Annex IV completeness per model
- Dashboard shows % complete for each of 12 Annex IV sections
- Remediation plan generator (already in W81) includes Annex IV gaps

**Dependencies:** None (FastAPI already integrated)

---

## Technical Deep Dive: System 2 Reasoning for Artifact Extraction

The research document emphasizes:

> "This proposed platform focuses on deep control flow semantic analysis. Instead of merely scanning for syntax errors, the AI reasons through the unique cyclomatic complexity... to ensure that equipment is forced into predefined safe states during failure events."

**Application to Compliance:** Use System 2 reasoning models (Claude Sonnet 4.5, GPT-4o, DeepSeek R1) to **automatically infer missing Annex IV sections** from existing artifacts.

### Example: Automated Data Governance Report

**Input:** HuggingFace dataset name (`"allenai/c4"`)

**System 2 prompt:**
```
You are an EU AI Act compliance auditor. Given the dataset "allenai/c4", generate the Annex IV §2(a) data governance section.

Required elements:
1. Dataset provenance (origin, license, collection methodology)
2. Demographic distributions (if applicable)
3. Bias mitigation measures
4. Data quality controls
5. Privacy protections

Use the HuggingFace dataset card as your source. Output structured JSON.
```

**Output:**
```json
{
  "dataset_name": "allenai/c4",
  "provenance": {
    "source": "Common Crawl",
    "license": "ODC-BY",
    "collection_date": "2019-04",
    "collection_method": "Web scraping with heuristic filters"
  },
  "demographic_distribution": "Not applicable (general web text)",
  "bias_mitigation": [
    "Blocked adult content via badwords list",
    "Removed duplicate n-grams",
    "Language detection (en only)"
  ],
  "quality_controls": [
    "Minimum word count (5)",
    "Terminal punctuation required",
    "Language ID confidence ≥ 0.99"
  ],
  "privacy_protections": "No PII scraping (Common Crawl robots.txt respected)"
}
```

**Why this works:** System 2 models can **synthesize** compliance documentation from existing public metadata (HF dataset cards, model cards, papers) — eliminating 80% of manual drafting work.

**Implementation:**
```python
# squash/reasoning_agent.py (NEW)
class ComplianceReasoningAgent:
    """Use System 2 LLM to auto-generate missing Annex IV sections."""
    
    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        self.model = model
        
    def generate_data_governance_section(self, dataset_name: str) -> dict:
        """Auto-generate Annex IV §2(a) from HF dataset metadata."""
        prompt = self._build_prompt("data_governance", dataset_name)
        response = self._call_llm(prompt)
        return json.loads(response)
        
    def generate_robustness_testing_section(self, model_dir: Path) -> dict:
        """Auto-generate Annex IV §6(b) from scan results."""
        scan_result = ModelScanner.scan(model_dir)
        prompt = self._build_prompt("robustness_testing", scan_result)
        response = self._call_llm(prompt)
        return json.loads(response)
```

**Strategic advantage:** This is the **"System 2 reasoning for compliance"** differentiator. Competitors (Vanta, Drata, OneTrust) are rule-based checklist tools. You're building an **AI-powered compliance agent** that reasons through regulatory requirements.

---

## Competitive Moat Analysis

The research document states:

> "The competitive moat for this product is inherently strong. It transforms a crippling regulatory bottleneck into a seamless engineering workflow. Because the software operates at the intersection of legal interpretation and deep machine learning operations, it inherently repels lightweight competitors."

**Why Squash has a 12-18 month head start:**

1. **Data moat:** You already have CycloneDX 1.7 ML-BOM generation, SLSA provenance, and Sigstore signing — competitors are 6-12 months behind on this infrastructure.

2. **Regulatory moat:** EU AI Act enforcement is **98 days away**. Enterprises need solutions **now**. First-mover advantage is decisive.

3. **Technical moat:** The "Artifact Extraction Engine" requires deep ML ops expertise (parsing TensorBoard logs, HF Trainer configs, MLflow runs). This is **not** a compliance tool — it's an **ML infrastructure tool** with compliance as the output.

4. **Integration moat:** You already have GitHub Actions, Jenkins, MLflow, W&B integrations. Competitors start from zero.

5. **Licensing moat:** BUSL-1.1 license (already in place) prevents cloud providers from cloning Squash without contributing back. This is the **Elastic/Confluent strategy** — source-available with commercial restrictions.

**Strategic positioning:**

| Competitor | Strength | Weakness vs. Squash |
|-----------|----------|-------------------|
| **Vanta** | SOC 2, ISO 27001 automation | No AI-specific capabilities; treats models as black boxes |
| **Drata** | Cloud infrastructure compliance | No ML-BOM generation; no model scanning |
| **OneTrust** | Privacy regulations (GDPR) | Enterprise sales cycles (6-12 months); no CI/CD native |
| **Protecto** | Data privacy for AI | No EU AI Act coverage; no SBOM |
| **Credo AI** | AI governance platform | No code integration; manual documentation |

**Your unique position:** The only **CI/CD-native, developer-first, open-core** EU AI Act compliance platform. Positioned as "GitGuardian for AI governance" — not "OneTrust for AI."

---

## Go-to-Market Strategy

### Target Customer Segments

**Primary (Tier 1 — highest urgency):**
1. **EU-based ML/AI teams** (10-100 engineers) deploying high-risk AI systems (HR tech, credit scoring, medical devices)
   - Pain: August 2, 2026 deadline
   - Budget: €50k-€200k/year compliance spend
   - Champion: VP Engineering or Head of ML Platform

2. **US tech companies with EU operations** (selling AI SaaS into Europe)
   - Examples: Databricks, Scale AI, Hugging Face (EU customers)
   - Pain: Legal risk of non-compliance (€35M fines)
   - Budget: $100k-$500k/year
   - Champion: CISO or Chief Privacy Officer

**Secondary (Tier 2):**
3. **AI consulting firms** (McKinsey Digital, Deloitte AI) delivering models to regulated clients
   - Pain: Manual compliance work non-billable
   - Budget: Per-project (€10k-€50k)
   - Champion: Engagement Manager

4. **Model hubs** (Hugging Face, Replicate) needing scalable compliance for hosted models
   - Pain: Liability for non-compliant models on platform
   - Budget: Revenue share or SaaS
   - Champion: Head of Trust & Safety

### Pricing Model

**Community (Free)**
- Single-user SBOM generation
- Basic policy checks (EU AI Act, NIST AI RMF)
- CLI + GitHub Actions
- Limit: 10 models/month

**Professional (€199/user/month)**
- Multi-user teams (up to 50 users)
- Full Annex IV auto-generation
- CI/CD integrations (Jenkins, GitLab, Argo)
- MLflow + W&B artifact logging
- Email support
- Limit: 100 models/month

**Enterprise (Custom — starts €5k/month)**
- Multi-tenant cloud API
- VEX feed subscription
- Real-time drift detection
- Compliance dashboard
- Slack/MS Teams integration
- SLA + dedicated support
- Custom policy templates
- SSO / SCIM provisioning
- Unlimited models

**Revenue model:**
- **Land:** Free tier (GitHub Actions composite) → organic adoption
- **Expand:** Professional tier (department-wide rollout)
- **Enterprise:** Platform-wide deployment with multi-tenant API

**Retention mechanism:** VEX feed subscription (recurring vulnerability alerts) creates **ongoing value** beyond one-time compliance documentation.

---

## Next Immediate Actions (This Week)

### Day 1-2: Repository Setup
1. Create feature branch: `git checkout -b feature/annex-iv-automation`
2. Scaffold new modules:
   ```bash
   touch squish/squash/artifact_extractor.py
   touch squish/squash/dataset_tracker.py  
   touch squish/squash/code_scanner.py
   touch squish/squash/annex_iv_generator.py
   touch squish/squash/reasoning_agent.py
   ```
3. Create test files:
   ```bash
   touch tests/test_artifact_extraction.py
   touch tests/test_annex_iv_generation.py
   ```

### Day 3-4: Prototype Artifact Extractor
**Goal:** Parse PyTorch Lightning config → extract hyperparameters

**Test case:**
```python
# tests/test_artifact_extraction.py
def test_extract_from_lightning_config():
    config = """
    model:
      learning_rate: 1e-4
      optimizer: AdamW
      weight_decay: 0.01
      
    trainer:
      max_epochs: 10
      gradient_clip_val: 1.0
    """
    
    extractor = ArtifactExtractor()
    artifacts = extractor.from_yaml(config)
    
    assert artifacts["optimizer"]["type"] == "AdamW"
    assert artifacts["optimizer"]["learning_rate"] == 1e-4
    assert artifacts["training"]["max_epochs"] == 10
```

**Implementation:**
```python
# squish/squash/artifact_extractor.py
import yaml
from pathlib import Path
from typing import Any

class ArtifactExtractor:
    """Extract Annex IV artifacts from training configs."""
    
    @staticmethod
    def from_yaml(config_str: str) -> dict[str, Any]:
        """Parse YAML config → structured artifacts."""
        config = yaml.safe_load(config_str)
        
        artifacts = {
            "optimizer": {},
            "training": {},
            "model": {}
        }
        
        # Extract optimizer config
        if "model" in config:
            if "optimizer" in config["model"]:
                artifacts["optimizer"]["type"] = config["model"]["optimizer"]
            if "learning_rate" in config["model"]:
                artifacts["optimizer"]["learning_rate"] = config["model"]["learning_rate"]
            if "weight_decay" in config["model"]:
                artifacts["optimizer"]["weight_decay"] = config["model"]["weight_decay"]
                
        # Extract trainer config
        if "trainer" in config:
            if "max_epochs" in config["trainer"]:
                artifacts["training"]["max_epochs"] = config["trainer"]["max_epochs"]
            if "gradient_clip_val" in config["trainer"]:
                artifacts["training"]["gradient_clip"] = config["trainer"]["gradient_clip_val"]
                
        return artifacts
```

### Day 5-7: Dataset Provenance Prototype
**Goal:** Pull HuggingFace dataset metadata → generate Annex IV §2(a) JSON

**Test case:**
```python
# tests/test_dataset_tracker.py
def test_extract_hf_dataset_provenance():
    tracker = DatasetProvenanceBuilder()
    provenance = tracker.from_huggingface_dataset("allenai/c4")
    
    assert provenance["license"] == "ODC-BY"
    assert "Common Crawl" in provenance["source"]
    assert len(provenance["quality_controls"]) > 0
```

**Implementation:**
```python
# squish/squash/dataset_tracker.py
from datasets import load_dataset_builder

class DatasetProvenanceBuilder:
    """Build Annex IV §2 data governance documentation."""
    
    @staticmethod
    def from_huggingface_dataset(dataset_name: str) -> dict:
        """Extract metadata from HF Datasets."""
        try:
            builder = load_dataset_builder(dataset_name)
            info = builder.info
            
            return {
                "dataset_name": dataset_name,
                "license": info.license or "Unknown",
                "source": info.homepage or "Unknown",
                "description": info.description[:500],  # truncate
                "features": list(info.features.keys()),
                "splits": list(info.splits.keys()),
                "size_bytes": sum(split.num_bytes for split in info.splits.values())
            }
        except Exception as e:
            return {
                "error": str(e),
                "dataset_name": dataset_name
            }
```

---

## Long-Term Vision: The Compliance Graph

**Year 1 (2026):** EU AI Act compliance platform (Annex IV automation)

**Year 2 (2027):** Multi-framework compliance graph
- Nodes: Models, datasets, training runs, deployments
- Edges: Provenance, dependencies, transformations
- Queries: "Which models are affected by CVE-2027-12345?"
- Alerts: "Dataset X deprecated → 47 downstream models at risk"

**Year 3 (2028):** Autonomous compliance agent
- Monitor: Continuous drift detection
- Reason: System 2 LLM identifies non-compliance
- Act: Auto-generate remediation pull requests
- Verify: Run policy checks in CI/CD

**Strategic endgame:** Squash becomes the **compliance operating system** for AI — the same way Kubernetes became the container orchestration standard. Every AI system deployed globally runs on Squash infrastructure.

---

## Critical Success Factors

### Must-Have (Non-Negotiable)
1. ✅ **August 2, 2026 deadline** — Ship v1.0 by July 15, 2026 (77 days)
2. ✅ **CI/CD native** — GitHub Actions composite action (no manual workflows)
3. ✅ **System 2 reasoning** — Auto-generate missing sections (differentiator)
4. ✅ **Open-core licensing** — BUSL-1.1 (prevent cloud provider cloning)
5. ✅ **Multi-tenant API** — Already built (W52-W81)

### Nice-to-Have (Competitive Advantage)
1. 🎯 **Real-time drift detection** — Already built (`drift.py`)
2. 🎯 **VEX feed integration** — Already built (`vex.py`)
3. 🎯 **NIST AI RMF crosswalk** — Wave 83 in roadmap
4. 🎯 **ISO 42001 compliance** — Already in `model_card.py`
5. 🎯 **Vertex AI + Azure DevOps integration** — Already built (W66-W67)

### Avoid (Anti-Patterns)
1. ❌ **Building a UI** — CLI + API first; UI is distraction
2. ❌ **Manual documentation workflows** — Defeats "Compliance-as-Code" positioning
3. ❌ **SOC 2 / ISO 27001 scope creep** — Stay AI-specific; Vanta/Drata own infra compliance
4. ❌ **Training AI models** — Extract artifacts, don't replace ML engineers

---

## Conclusion: Why This Wins

**Market timing:** EU AI Act enforcement in 98 days. Enterprises are desperate.

**Technical moat:** Intersection of ML ops + legal compliance. High barrier to entry.

**Strategic positioning:** Only **CI/CD-native, open-core** solution. Positioned against legacy GRC vendors (OneTrust, ServiceNow) who can't ship fast enough.

**Product-market fit:**
- **Pain:** Manual Annex IV documentation = months of engineering time
- **Solution:** Auto-generate from existing artifacts in CI/CD
- **Value prop:** Ship compliant models in days, not months
- **Pricing:** €199/user/month (vs. €50k consulting engagements)

**Execution plan:**
- Week 1-2: Artifact extraction
- Week 3-4: Annex IV generation  
- Week 5: CI/CD integration
- Week 6: Multi-tenant API extensions
- Week 7-8: Beta testing with 3-5 design partners
- Week 9-10: Public launch + EU AI Act deadline marketing blitz

**Next step:** Start coding `artifact_extractor.py` today. The compliance market waits for no one.

---

**End of Strategic Blueprint**  
**Implementation begins: Monday, April 28, 2026**