---
name: konjo-boot
description: Boot a Konjo session for squish. Produces a Session Brief, runs Discovery, identifies the next sprint.
user-invocable: true
---
# Konjo Session Boot — squish

## Step 1 — Read
Read: CLAUDE.md, README.md, CHANGELOG.md, PLAN.md, MODULES.md, docs/.

## Step 2 — Session Brief
```
REPO         squish — local LLM inference server (MLX/PyTorch, speculative decoding, quantization, Ollama-compat API)
LAST SHIPPED [most recent change from CHANGELOG.md]
OPEN WORK    [current wave state from PLAN.md]
BLOCKERS     [failing tests, broken modules, open issues]
HEALTH       [Green / Yellow / Red]
```

## Step 3 — Discovery
Search: arXiv (quantization, speculative decoding, local inference), GitHub (MLX updates, llama.cpp, ollama), HuggingFace (model releases).

## Step 4 — Identify Work
Load PLAN.md (wave state) + MODULES.md. Validate against codebase. Flag drift.

## Invocation Keywords: `konjo` / `konjo squish` / `squish konjo` / `read KONJO_PROMPT.md and begin`
