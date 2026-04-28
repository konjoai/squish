# SESSION.md — Squish Running Context

> Update this file at the start of every session. Reading it takes 30 seconds
> and prevents re-discovering context that is already known.

---

## Current date
2026-04-28

## Last commits
- **`f109942`** — docs(squash): update compliance section to point to standalone konjoai/squash repo
- **`75935cb`** — feat(squash): W82 HQQ float bits + W83 NIST AI RMF 1.0 controls
- **`ec2bdf3`** — feat(squash): W81 remediation plan generator

## This session (2026-04-28)
- Removed `squish/squash/` module from squish — now standalone at `konjoai/squash` (`pip install squash-ai`)
- Deleted 80 `tests/test_squash_*.py` test files
- Updated `squish/server.py` and `squish/cli.py` to import from standalone `squash` package (optional; try/except guarded)
- Updated `pyproject.toml`: removed `squash`/`squash-api` extras and `squash` CLI entrypoint
- Updated `tests/test_cli_eval.py`, `test_cli_sbom.py`, `test_sbom_builder.py`, `test_eval_binder.py`, `test_oms_signer.py`, `test_governor_middleware.py` to use `squash.*` imports with `pytest.importorskip` guards

---

## Module count
- Python files in `squish/` reduced from 112 to **~68** after squash extraction.

---

## Open accuracy-validation items
- **W56 AQLM**: lm_eval on Qwen2.5-1.5B never run. Not blocking current work.
- **mixed_attn lm_eval**: Code-complete (W41), still unvalidated. Not blocking.

---

## Next wave: W100 — Pre-Download ModelScan for `squish pull hf:`

**Goal:** Scan model files before loading — close the pre-load ACE attack surface on `squish pull hf:` downloads.

**Changes:**
- `squish/serving/local_model_scanner.py`: add `scan_before_load(download_dir)`
- `squish/cli.py` pull path: call `scan_before_load()`, abort rc=2 on unsafe
- `tests/test_predownload_scan.py`: ≥12 tests

**W101+:** Rust Inference Bridge via candle-pyo3 (`squish_quant_rs/`).
