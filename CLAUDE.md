# squish

Local LLM inference server — MLX-accelerated on Apple Silicon, with speculative decoding, quantization (INT4/INT3/SQINT2), agent tool execution, Ollama/OpenAI-compatible API, and the macOS SquishBar.

**v9.34.2**

## Stack
Python 3.10+ · MLX + mlx-lm (Apple Silicon) · FastAPI · transformers · HuggingFace Hub · Swift (macOS SquishBar)

## Commands
```bash
python -m pytest tests/ -x                   # full test suite
python -m pytest tests/ -x -k "test_name"    # run a single test
python -m squish serve                        # start inference server
squish pull hf:<repo>                         # download + pre-scan HF model
squish trace                                  # observability report
squish compat                                 # backend compatibility check
```

## Critical Constraints
- No `unwrap()`/`expect()` in Python — raise with a clear message or log + re-raise
- No silent failures — `logging.warning` if a fallback swallows an error
- MLX imports must be gated behind platform check — never imported on Linux paths
- `squish.squash` is now an **optional** import — never hard-depend on `squash-ai`
- Quantization accuracy gates are hard stops: INT4 AWQ g=32 ≥ 70.6% arc_easy (Qwen2.5-1.5B); INT2 naive is **NEVER SHIP**
- Pre-scan HF models **before** loading weights — `HFFileSummary` scan runs at `squish pull hf:` time
- Prompt injection: system prompt content must never be controllable by request payload
- Never log raw user prompt content at INFO level or above — log a hash or truncated prefix
- Version bumps touch `pyproject.toml` + `squish/__init__.py`

## Module Map
| Module | Role |
|--------|------|
| `squish/server.py` | FastAPI app entry point, startup profiler, backend routing |
| `squish/cli.py` | `squish` CLI — serve, pull, trace, compat, agent |
| `squish/catalog.py` | Model registry: URI parsing (`ollama:` / `hf:`) + HF batch upload |
| `squish/serving/` | Backend router, Ollama/LocalAI compat, blazing TTFT, tool calling |
| `squish/hardware/` | Platform detector, production profiler, Apple Silicon routing |
| `squish/api/` | OpenAI-compatible v1 router |
| `squish/agent/` | Agent loop, tool name map, tool execution |
| `squish/quant/` | AWQ/INT3/INT4/SQINT2 quantization pipeline |
| `squish/kv/` | KV cache management |
| `squish/context/` | Context window management |
| `squish/platform/` | Cross-platform router and detector |
| `apps/macos/SquishBar/` | Swift macOS menu bar app (model picker, progress, hotkey) |

## Planning Docs
- `MODULES.md` — per-wave module reference (Waves 1–99+)
- `CHANGELOG.md` — all notable changes

## Konjo Quality Framework

Three walls against AI slop — all enforced by CI.

**Wall 1 — Pre-commit** (`bash .konjo/scripts/install-hooks.sh`):
ruff lint, ruff format, bare-except scan, DRY check, TODO scan. Blocks the commit.

**Wall 2 — CI gate** (`.github/workflows/konjo-gate.yml`):
Coverage ≥ 80% · mutation survival ≤ 10% · complexity ≤ 15 · file ≤ 500L · zero DRY violations. Blocks the merge.

**Wall 3 — Adversarial review** (local only — disabled in CI):
`git diff HEAD~1 | python3 .konjo/scripts/konjo_review.py`

See `KONJO_QUALITY_FRAMEWORK.md` for the full specification.

## Skills
See `.claude/skills/` — auto-loaded when relevant.
Run `/konjo` to boot a full session (Brief + Discovery + Plan).
