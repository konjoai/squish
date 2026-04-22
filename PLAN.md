# squish / squash — Master Strategic Plan
> Updated: 2026-04-16 (post-research sprint, W81+W95 era)
> Status: **Active**. Commit `ec2bdf3` (squash W81) × W99 (squish server). 4327+ tests passing.

---

## Patch Notes (Post-W99)

- 2026-04-22: `squish pull` / `squish import` Hugging Face URL normalization hotfix completed.
  Added canonical `owner/repo` normalization for `hf:` and `huggingface.co` URL inputs,
  including `/tree/<branch>` links.
- 2026-04-22: `cmd_doctor` version resolution hardened to prevent crashes when dependency
  modules do not expose `__version__`; fallback now uses distribution metadata.

---

## Current State Inventory

### Squash Enterprise Compliance Layer (W57–W81)
| Wave | Feature | Status |
|------|---------|--------|
| W57–W64 | CloudDB, drift, policy, compliance score/history/overview | ✅ Complete |
| W65–W70 | VEX feed, GCP Vertex AI, Azure DevOps attestation ingest | ✅ Complete |
| W71–W72 | EU AI Act conformance per-tenant + platform | ✅ Complete |
| W73–W76 | SARIF, audit export bundle, plan gating | ✅ Complete |
| W77–W79 | Cloud CLI commands, enforcement action, cloud-attest/vex | ✅ Complete |
| W80 | EU AI Act risk-tier classification (UNACCEPTABLE/HIGH/LIMITED/MINIMAL) | ✅ Complete |
| W81 | Remediation plan generator — priority-ordered steps | ✅ Complete |

### Squish Server Performance Layer (W85–W99)
| Wave | Feature | Status |
|------|---------|--------|
| W85 | CLI color dedup / `_term.py` consolidation | ✅ Complete |
| W86 | Observability profiler + `squish trace` | ✅ Complete |
| W87 | Agent tool execution fix + `tool_name_map.py` | ✅ Complete |
| W88 | Ollama/LocalAI drop-in compat | ✅ Complete |
| W89 | Local model scanner + `ollama:`/`hf:` URI schemes | ✅ Complete |
| W90 | Lean startup profiler + `FeatureState` refactor | ✅ Complete |
| W91 | Sub-3s TTFT blazing default + 70B loader | ✅ Complete |
| W92 | Pre-compress pipeline + HF batch upload workflow | ✅ Complete |
| W93 | macOS SquishBar (Swift: model picker, progress, hotkey) | ✅ Complete |
| W94 | Cross-platform support review | ✅ Complete |
| W95 | README final audit + public release (v68.0.0) | ✅ Complete |
| W96–W99 | LM Studio compat, inference fixes, lean server, speed restore | ✅ Complete |

### Existing Enterprise Capabilities (already built — do NOT re-implement)
- **CycloneDX 1.7 ML-BOM** — `squash/sbom_builder.py`: full ECMA-424 compliant JSON sidecar generated on `squish compress`. Component type `machine-learning-model`. SHA-256 weight hash. Populated by `eval_binder.py`.
- **SPDX 2.3 SBOM** — `squash/spdx_builder.py`: SPDX-2.3 TVText sidecar alongside CycloneDX.
- **ModelScan v0.8.8** — `squash/scanner.py`: ProtectAI `modelscan` as primary scan backend when installed; falls back to native pickle-opcode / GGUF / ONNX / safetensors header scanner. Never executes model code.
- **MLflow offline shim** — `squash/cli.py::_cmd_attest_mlflow` + `POST /attest/mlflow`: emits MLflow-compatible attestation JSON for `mlflow.log_artifact`. **Not** real MLflow SDK tracking.
- **HQQ quantizer** — `squish/quant/hqq.py`: half-quadratic quantization at {2, 3, 4, 8} bits. **GAP: no 1-bit; no arbitrary float nbits.**
- **AQLM loader** — `tests/test_quant_aqlm.py`: test coverage for AQLM format loading.
- **GitHub Actions CI/CD** — `squash/cicd.py`: env detection + annotation for GitHub/Jenkins/GitLab/CircleCI. Wave 92: `.github/workflows/model_upload.yml`. **GAP: no reusable composite action for third-party pipelines.**
- **EU AI Act risk classification** — Articles 9/10/13/17 mapped to risk tiers. Conformance per-tenant + platform-wide. **GAP: no NIST AI RMF GOVERN/MAP/MEASURE/MANAGE mapping.**

---

## Research Findings Summary (Enterprise Sprint)

### Domain 1: Sub-3-bit Quantization
| Method | Bits | Speed vs FP16 | License | Squash relevance |
|--------|------|--------------|---------|-----------------|
| **HQQ** | 1–8 (any) | 3× @ 4-bit | Apache 2.0 | Extend existing `hqq.py` to support 1-bit + arbitrary nbits |
| **AQLM** | <3 (multi-codebook) | 1.5× @ 2-bit | Apache 2.0 | ICML 2024; loading support exists, inference improvements needed |
| **QuIP#** | 2–3 | 3.3× | Apache 2.0 | E8 lattice codebooks; Hadamard incoherence preprocessing |
| **QTIP** | 2–4 | **3.4×** | GPL-3.0 | Load pre-quantized HF Hub models only — never bundle GPL kernel |
| **YAQA** | 2–4 | TBD | TBD | Cornell successor to QTIP; research-track |
| **BitNet b1.58** | 1.58 | N/A (training) | — | Certification only — detect ternary weight distribution |

**Key insight:** HQQ supports *any* float nbits 1–8. The `_VALID_BITS = (2, 3, 4, 8)` restriction in `hqq.py` is an implementation limit, not a theoretical one.

### Domain 2: Security
- **safetensors**: Trail of Bits audit → zero ACE vulns. 100× faster than pickle. Already enforced in squash scanner.
- **ModelScan 0.8.8**: Scans without executing. Already integrated. Severity: CRITICAL/HIGH/MEDIUM/LOW. Exit codes 0/1/2/3/4.
- **Threat vector gap**: No pre-download scan on `squish pull hf:` URI downloads.

### Domain 3: Rust Performance  
- **candle** (HF, Apache-2.0+MIT, 20k⭐): `candle-pyo3` → call Rust from Python. `squish_quant_rs/` already exists at repo root — Rust extension scaffold is there.
- **burn** (Apache-2.0+MIT, v0.20.1): Full Rust inference with swappable backends (Metal, CUDA, ROCm, Vulkan, WASM).

### Domain 4: Enterprise Workflow
- **MLflow REST API**: Model Registry with stage transitions (None→Staging→Production), aliases (champion/challenger), webhooks on `MODEL_VERSION CREATED`, tags for compliance metadata, budget policies. Current squash shim only emits JSON — gap is real MLflow SDK tracking.
- **GitHub Actions Composite**: `action.yml` with `runs.using: "composite"`. Lives at `.github/actions/squash-*`. Called via `uses: ./.github/actions/squash-compress`. Inputs/outputs supported. Third-party reusable.
- **CycloneDX 1.7 ML-BOM**: Already fully implemented. ECMA-424 standard (Ecma International).

### Domain 5: Regulatory
- **EU AI Act** (EU 2024/1689): Already implemented W71–W80. Articles 9/10/13/17 mapped.
- **NIST AI RMF 1.0** (NIST.AI.100-1): Published. v1.1 in revision. Four functions: **GOVERN, MAP, MEASURE, MANAGE**. Crosswalks to ISO/IEC 42001, EU AI Act, NIST CSF. Not yet in squash.
- **NIST Generative AI Profile** (NIST.AI.600-1): Available. Maps RMF to GenAI-specific risks.

---

## Wave Roadmap (W82–W90 Squash Enterprise)

### W82 — HQQ Arbitrary-Bit Quantization (1–8 bits, any float nbits)
**Why:** Current `hqq.py` restricts `bits` to `{2, 3, 4, 8}`. HQQ paper supports any float nbits 1–8, including 1-bit, 1.5-bit, 2.5-bit, 3.5-bit. This is the "INT 2.5" the user asked about. No calibration data. Apache 2.0.

**Changes:**
- `squish/quant/hqq.py`: relax `_VALID_BITS` constraint — accept any `float` in `[1.0, 8.0]`. Update `_grid_levels()` to compute `round(2**nbits)`. Add `nbits` field (alongside `bits: int`) to `HQQConfig` for float precision. Update `HQQConfig.__post_init__` validation.
- `tests/tests/quant/test_hqq_unit.py` (new file in `tests/quant/`): shape/dtype contracts for 1-bit, 2.5-bit, 3.5-bit. Cosine similarity regression for 1-bit (≥ 0.50 floor — binary is 1-bit reference). Encode/decode roundtrip.
- `CHANGELOG.md` entry.

**Gate:** Cosine similarity ≥ 0.9999 (4-bit), ≥ 0.995 (2-bit), ≥ 0.50 (1-bit). Arbitrary nbits accepted without exception.

---

### W83 — NIST AI RMF 1.0 Compliance Report
**Why:** EU AI Act is in squash. NIST AI RMF (NIST.AI.100-1) is the US enterprise gold standard. Enterprises need both. Four functions: **GOVERN** (policies/accountability), **MAP** (context/categorize), **MEASURE** (analyze/evaluate), **MANAGE** (prioritize/respond). RMF v1.1 in revision — implement v1.0 now.

**Changes:**
- `squish/squash/nist_rmf.py` (new): `NistRmfReport` dataclass. `generate_nist_rmf_report(tenant_record, policy_stats, vex_alerts, attestation_score)` → structured dict with `govern`, `map`, `measure`, `manage` sub-sections. Crosswalk to EU AI Act risk tier.
- `squish/squash/api.py`: `GET /cloud/tenants/{id}/nist-rmf-report` endpoint.
- `squish/squash/cli.py`: `squash cloud-nist <tenant_id>` subcommand.
- `tests/test_squash_w83.py`: ≥20 tests. Gate: all pass.

---

### W84 — GitHub Actions Reusable Composite Action
**Why:** Wave 92 added `model_upload.yml` (internal workflow). There is no third-party reusable composite action at `.github/actions/squash-*`. Enterprise teams need a drop-in action: scan → compress → attest → emit BOM, all in one `uses:` step.

**Changes:**
- `.github/actions/squash-scan/action.yml`: composite action. Inputs: `model-path`, `strict` (safetensors-only flag). Outputs: `scan-result` (clean/issues/error), `report-path`. Steps: `pip install squish`, `squash scan "${{ inputs.model-path }}"`.
- `.github/actions/squash-compress/action.yml`: composite action. Inputs: `model-path`, `method` (hqq/aqlm/awq, default=hqq), `nbits` (default=4). Outputs: `compression-ratio`, `bom-path`.
- `.github/actions/squash-attest/action.yml`: composite action. Inputs: `model-path`, `policies`. Outputs: `passed` (true/false), `attestation-path`.
- `docs/github-actions.md`: usage example.
- `tests/test_squash_w84.py`: validate action.yml YAML schema (required fields, steps structure, outputs format). ≥15 tests.

---

### W85 — MLflow Real SDK Integration
**Why:** Current `attest-mlflow` command emits JSON suitable for `mlflow.log_artifact`. It does NOT register models in MLflow Model Registry, set stage transitions, or write tags. Enterprise MLflow users need real tracking: run logging, model registration, stage → Production, alias `champion`, compliance tags.

**Changes:**
- `squish/squash/mlflow_bridge.py` (new): `MlflowBridge` class. Methods: `log_compress_run(meta)` → logs params/metrics; `register_model(run_id, model_name)` → creates ModelVersion; `transition_to_production(model_name, version)` → stages to Production with `archive_existing_versions=True`; `set_compliance_tags(model_name, version, tags_dict)`. Optional import (`mlflow` not a hard dependency).
- `squish/squash/cli.py`: `squash mlflow-register <model_dir> --model-name <name> [--stage production] [--alias champion]`.
- `squish/squash/api.py`: `POST /attest/mlflow/register` — body: `{model_path, model_name, mlflow_tracking_uri}`.
- `tests/test_squash_w85.py`: mock `mlflow.tracking.MlflowClient`. ≥20 tests.

---

### W86 — Pre-Download ModelScan for `squish pull hf:`
**Why:** Current scanner runs post-load. `squish pull hf:<repo>` downloads model weights before scanning. An adversarial HF model triggers ACE at load time before scan runs. Need pre-load scan.

**Changes:**
- `squish/serving/local_model_scanner.py`: add `scan_before_load(download_dir)` hook.
- `squish/cli.py` pull path: run `ModelScanner.scan(download_dir)` before any import; abort on `status="unsafe"`.
- `tests/test_squash_w86.py`: inject a synthetic pickle-with-REDUCE into a mock download; assert pull aborts with rc=2.

---

### W87 — QTIP/YAQA Pre-Quantized Model Loading (HF Hub, load-only)
**Why:** QTIP (NeurIPS 2024 Spotlight, 3.4× FP16 at 2-bit) and YAQA (Cornell successor) are GPL-3.0. Strategy: support loading pre-quantized QTIP/YAQA models from HF Hub via `transformers` library — never bundle GPL quantization kernels. Enterprise users can use 2-bit inference on pre-compressed checkpoints.

**Changes:**
- `squish/squash/qtip_loader.py` (new): `load_qtip_checkpoint(hf_repo_id)` → downloads model via `huggingface_hub`, invokes `from transformers import AutoModelForCausalLM` with appropriate config, returns model handle. License gate: logs `WARNING: QTIP model loaded via GPL-compatible runtime path — squish does not distribute GPL code`.
- `squish/squash/cli.py`: `squash load-qtip <hf_repo> [--local-dir PATH]`.
- `tests/test_squash_w87.py`: mock `huggingface_hub.snapshot_download`. ≥10 tests.

---

## Wave Roadmap (W100–W105 Squish Server)

### W100 — Rust Inference Bridge via candle-pyo3
**Why:** `squish_quant_rs/` scaffold exists. candle (HF, Apache-2.0, 20k⭐) provides `candle-pyo3` for calling Rust kernels from Python. Eliminates GIL on quantized matmul. 4× speed potential on CPU inference.

**Changes:**
- `squish_quant_rs/src/lib.rs`: implement INT4 matmul via candle `QTensor`. Expose as `quantized_matmul(w_codes, scales, zeros, x)` via PyO3.
- `squish/quant/quantizer.py`: detect `squish_quant_rs` availability; fall back to NumPy/MLX path if absent.
- CI: `cargo build --release -p squish_quant_rs` step.

---

## Module Accounting
| Wave | New Files | Deleted/Absorbed | Net |
|------|-----------|-----------------|-----|
| W82 | 0 (extends hqq.py in-place) | 0 | 0 |
| W83 | +1 (`squash/nist_rmf.py`) | 0 | +1 |
| W84 | 0 (action.yml YAML files only) | 0 | 0 |
| W85 | +1 (`squash/mlflow_bridge.py`) | 0 | +1 |
| W86 | 0 (extends local_model_scanner.py) | 0 | 0 |
| W87 | +1 (`squash/qtip_loader.py`) | 0 | +1 |
| **Total after W87** | **115 Python files** (112 + 3) | | — |
| **Ceiling** | **125** | | **10 slots remaining** |

---

## License Constraints (HARD RULES)
| Library | License | Rule |
|---------|---------|------|
| HQQ | Apache 2.0 | ✅ Integrate directly |
| AQLM | Apache 2.0 | ✅ Integrate directly |
| QuIP# | Apache 2.0 | ✅ Integrate directly |
| candle | Apache-2.0 + MIT | ✅ Integrate directly |
| burn | Apache-2.0 + MIT | ✅ Integrate directly |
| ModelScan | Apache 2.0 | ✅ Already integrated |
| CycloneDX | Apache 2.0 (ECMA-424) | ✅ Already integrated |
| MLflow | Apache 2.0 | ✅ Integrate directly |
| QTIP | GPL-3.0 | ⚠️ Load pre-quantized models ONLY — never bundle GPL kernels |
| YAQA | TBD | ⚠️ Treat as GPL until confirmed otherwise |

---

## Quantization Decision Tree (for `squish compress --format auto`)
```
user request:
  max_compression + best_quality → HQQ 4-bit (default, no calibration needed)
  max_compression + lower_memory → HQQ 3-bit (–3.4pp arc_easy, safe for Qwen2.5/Llama)
  max_compression + extreme      → HQQ 2-bit or AQLM 2-bit (coherent, Apache 2.0)
  sub-2-bit                      → HQQ 1-bit (binary; cosine ≥ 0.50 floor)
  pre-compressed 2-bit on HF     → QTIP load-only (GPL runtime path, logged)
  training-time 1.58-bit         → BitNet detection + certification only
```
⚠️ INT3 for gemma ≤4B: **UNSAFE (–15.2pp arc_easy)**. Auto-selector must block.
⚠️ INT3 for Qwen3-4B: **UNSAFE (–14.8pp arc_easy)**. Auto-selector must block.

---

## Accuracy Gates (DO NOT SHIP WITHOUT)
| Format | Model | Gate | Last validated |
|--------|-------|------|----------------|
| INT4 AWQ g=32 | Qwen2.5-1.5B | ≥ 70.6% arc_easy | 2026-03-28 |
| INT3 g=32 | Qwen2.5-1.5B | ≥ 67.2% arc_easy | 2026-03-28 |
| INT3 | gemma-3-*b ≤4B | **BLOCKED** | 2026-03-28 |
| INT3 | Qwen3-4B | **BLOCKED** | 2026-03-28 |
| INT2 naive | any | **NEVER SHIP** | confirmed incoherent |
| HQQ 1-bit | — | ≥ 0.50 cosine sim | lm_eval-waiver required |
| HQQ 2.5-bit | — | ≥ 0.990 cosine sim | lm_eval-waiver required |

---

## Memory Contracts (M3 16GB — HARD STOP)
| Model | Format | Peak RSS |
|-------|--------|---------|
| Qwen2.5-1.5B | INT4 | < 1.5 GB |
| Qwen2.5-1.5B | INT3 | < 1.0 GB |
| Qwen3:8B | INT4 | **DO NOT RUN** (14 GB squish dir crashes) |
| Qwen3:8B | INT3 | < 4.0 GB |
| gemma-3-4b | INT4 | < 8.7 GB (safe) |

---

## Next Immediate Action
**Start W82** — HQQ arbitrary nbits. Zero new files (extends `hqq.py` in-place). Zero module-count impact. Highest research-to-implementation ratio of any wave.

---

*Owner: wesleyscholl / Konjo AI Research*
*Update after each completed wave. Never let this drift from the actual implementation.*
