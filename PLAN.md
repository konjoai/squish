# Squish — Master Strategic Plan
> Updated: 2026-04-28 (post-squash separation)
> Status: **Active**. Current version: v9.14.0 (squish-only). Squash compliance layer extracted to `konjoai/squash` (Apache 2.0, `pip install squash-ai`).

---

## Current State

### Squish Server & Inference Performance (W85–W99 — COMPLETE)
| Wave | Feature | Status |
|------|---------|--------|
| W85 | CLI color dedup / `_term.py` consolidation | ✅ |
| W86 | Observability profiler + `squish trace` | ✅ |
| W87 | Agent tool execution fix + `tool_name_map.py` | ✅ |
| W88 | Ollama/LocalAI drop-in compat | ✅ |
| W89 | Local model scanner + `ollama:`/`hf:` URI schemes | ✅ |
| W90 | Lean startup profiler + `FeatureState` refactor | ✅ |
| W91 | Sub-3s TTFT blazing default + 70B loader | ✅ |
| W92 | Pre-compress pipeline + HF batch upload workflow | ✅ |
| W93 | macOS SquishBar (Swift: model picker, progress, hotkey) | ✅ |
| W94 | Cross-platform support review | ✅ |
| W95 | README final audit + public release (v68.0.0) | ✅ |
| W96–W99 | LM Studio compat, inference fixes, lean server, speed restore | ✅ |

### Squash Separation (2026-04-28)
- `squish/squash/` extracted to standalone repo `konjoai/squash`
- 80 `tests/test_squash_*.py` removed from squish test suite
- `squish/server.py` and `squish/cli.py` updated to import from standalone `squash` package (optional dependency)
- `pyproject.toml`: removed `squash`/`squash-api` extras and `squash` CLI entry point

### Quantization Accuracy Constraints (HARD STOP)
| Format | Model | Gate |
|--------|-------|------|
| INT4 AWQ g=32 | Qwen2.5-1.5B | ≥ 70.6% arc_easy |
| INT3 g=32 | Qwen2.5-1.5B | ≥ 67.2% arc_easy |
| INT3 | gemma-3-*b ≤4B | **BLOCKED** (−15pp) |
| INT3 | Qwen3-4B | **BLOCKED** (−14.8pp) |
| INT2 naive | any | **NEVER SHIP** |

---

## Wave Roadmap

### W100 — Pre-Download ModelScan for `squish pull hf:` ✅ COMPLETE
**Why:** `squish pull hf:<repo>` downloads model weights before scanning. An adversarial HF model can trigger ACE at load time before the post-load scan runs. This closes the pre-load attack surface.

**Changes (2026-04-28):**
- `squish/serving/local_model_scanner.py`: added `HFFileSummary`, `HFRepoScanResult`,
  `scan_hf_repo_metadata(repo_id, token) → HFRepoScanResult`, and
  `_classify_hf_siblings()`. Native pickle-header classification — no `modelscan` dep.
- `squish/cli.py`: `_pull_from_hf` calls `scan_hf_repo_metadata` **before**
  `snapshot_download`; prints compact scan report; aborts with `sys.exit(2)` on
  `status="unsafe"`. API errors allow download with warning (firewall / private-repo
  safe). Post-download `scan_before_load()` byte scan retained as second layer.
- `tests/test_predownload_scan.py`: 30 new tests (total: 48). All HF API calls mocked.
  `_classify_hf_siblings` tested at unit level; `scan_hf_repo_metadata` tested with
  mocked HTTP including 401/404/URLError/unexpected structure paths.

**Gate:** 48/48 tests pass. `squish pull hf:` aborts on unsafe model before any bytes transferred. Zero new mandatory dependencies.

---

### W101 — Rust Inference Bridge via candle-pyo3
**Why:** `squish_quant_rs/` scaffold exists. candle (HF, Apache-2.0, 20k★) provides `candle-pyo3` for calling Rust kernels from Python. Eliminates GIL on quantized matmul. 4× CPU inference speed potential.

**Changes:**
- `squish_quant_rs/src/lib.rs`: implement INT4 matmul via candle `QTensor`. Expose as `quantized_matmul(w_codes, scales, zeros, x)` via PyO3.
- `squish/quant/quantizer.py`: detect `squish_quant_rs` availability; fall back to NumPy/MLX path if absent.
- CI: `cargo build --release -p squish_quant_rs` step.

**Gate:** Maturin build succeeds. Python fallback still works when `squish_quant_rs` absent.

---

## Accuracy Gates (DO NOT SHIP WITHOUT)
| Format | Model | Gate | Last validated |
|--------|-------|------|----------------|
| INT4 AWQ g=32 | Qwen2.5-1.5B | ≥ 70.6% arc_easy | 2026-03-28 |
| INT3 g=32 | Qwen2.5-1.5B | ≥ 67.2% arc_easy | 2026-03-28 |
| INT3 | gemma-3-*b ≤4B | **BLOCKED** | confirmed |
| INT3 | Qwen3-4B | **BLOCKED** | confirmed |

---

## Memory Constraints (M3 16GB — HARD STOP)
| Model | Format | Peak RSS |
|-------|--------|---------|
| Qwen2.5-1.5B | INT4 | < 1.5 GB |
| Qwen2.5-1.5B | INT3 | < 1.0 GB |
| Qwen3:8B | INT4 | **DO NOT RUN** (14 GB crash) |
| Qwen3:8B | INT3 | < 4.0 GB |
| gemma-3-4b | INT4 | < 8.7 GB |

---

## Build & Test Commands

```bash
# Full Python test suite
python3 -m pytest tests/ -v --timeout=120

# Python-only mode
python3 -m pytest tests/ -v -k "not mojo"

# Rust workspace
cargo test --workspace --locked

# Install dev dependencies
pip install -e ".[dev,eval,linux]"

# JavaScript bindings
cd js && npm install && npm run build
```

---

## Next Immediate Action
**Start W101** — Rust Inference Bridge via candle-pyo3. `squish_quant_rs/` scaffold exists. Expose INT4 matmul via PyO3; Python fallback must still work when the Rust crate is absent.

---

*Owner: wesleyscholl / Konjo AI Research*
*Update after each completed wave. Never let this drift from actual implementation.*
