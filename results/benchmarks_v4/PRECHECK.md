# Squish v4 benchmark pre-flight — what's actually wired up

**Host:** Apple M3 MacBook Pro, 16 GB unified memory, macOS 25.5.0
**Date:** 2026-06-01
**Squish:** 9.14.0  (worktree contains v4 commit `8a8ef47`) · **Ollama:** 0.18.2 · **Python:** 3.14.3

This file records what the v4 spec asked for and what the v4 implementation
actually delivers. Every feature below was probed live on this M3 before the
main benchmark started.

## Feature probe results

### ✓ Models on disk

| Path                                              | Size      | Used for      |
|---------------------------------------------------|-----------|---------------|
| `/Users/wscholl/models/Qwen2.5-7B-Instruct-int4`  | 4.00 GB   | Target model  |
| `/Users/wscholl/models/Qwen2.5-1.5B-Instruct-int4`| 1.00 GB   | Draft (sub)   |
| Ollama `qwen2.5:7b`                               | 4.36 GB   | Ollama target |

The v4 methodology lists `Qwen2.5-0.5B-Instruct-INT4` as the draft model.
That model is not on disk; the closest same-family / same-tokenizer draft is
`Qwen2.5-1.5B-int4`. This is academic — see spec decode result below.

### ✗ `squishd` UDS daemon — partially functional

`squishd start <model_dir>` from `squish/daemon/squishd.py` does bind
`/tmp/squish.sock` and responds to control pings (`_cmd=ping/status`). **But
it cannot load any of the MLX-native quantized models on disk.** The daemon's
`_load_model()` hard-codes a call to `squish.quant.compressed_loader.load_compressed_model`
with `npz_path = <model_dir>-compressed`, which expects the squish npy-dir
format (`manifest.json`, `__q4.npy`). Our Qwen2.5-7B-int4 is in mlx-native
quant format (`model.safetensors` + `config.json` only — no manifest), so
the preload crashes:

```
FileNotFoundError: '/Users/wscholl/models/Qwen2.5-7B-Instruct-int4-compressed'
  at squish/quant/compressed_loader.py:264 in _safe_key_to_original
  via squish/daemon/squishd.py:445 in _load_model
```

The socket stays up and ping/status work, but every inference request returns
`{"error": "model not available: …"}`. Compressing Qwen2.5-7B to squish
npy-dir format would unblock this, but the article's headline target weights
are the mlx-native int4 (which is the version on `huggingface.co/mlx-community`),
so doing that defeats the purpose. **For Phase 1 we fall back to the older
HTTP-based `squish daemon start` (which uses `python -m squish.server` and
correctly loads mlx-native quant). Documented in the v4 follow-up list
below.**

### ✗ Speculative decoding — broken on this branch

Starting the server with `--draft-model` crashes during init:

```
File "squish/server.py", line 1276, in load_draft_model
    from squish.speculative import load_draft_model as _load_draft
ImportError: cannot import name 'load_draft_model' from 'squish.speculative'
```

The function exists at `squish/speculative/speculative.py:580` but the
`squish/speculative/__init__.py` package init (one-line module docstring) does
not re-export it. A one-line patch would unblock the path, but per scope
guards we do **not** fix v4 implementation bugs in this benchmarking session.

**Phase 3 (spec decode throughput) is skipped.** The row in the final table
reports `not-implemented` rather than fabricating a number.

### ✗ v4 `PromptKVStore` (new disk-backed KV cache) — implemented but not wired up

`squish/kv/prompt_kv_cache.py` ships the new `PromptKVStore` class with
SHA-256-keyed per-layer cache, LRU eviction, and matching tests in
`tests/test_prompt_kv_cache.py`. The class works in isolation — but `server.py`
never calls `store.get()` or `store.put()`, and there is no `--prompt-kv-cache`
or similar flag in either `squish run` or `python -m squish.server`. Grep
confirms:

```
$ grep -rn 'PromptKVStore\|prompt_kv_cache' squish/
squish/__init__.py:393:    "PromptKVStore": "squish.kv.prompt_kv_cache",   # ← lazy re-export only
squish/kv/prompt_kv_cache.py:1: …  (the module itself)
```

There is **no call site** in the inference path. The class is dead code
from the server's perspective.

### ✓ Older `--disk-prompt-cache` flag — works (with a soft dependency)

The pre-existing `--disk-prompt-cache <DIR>` flag on `python -m squish.server`
is wired through `squish.kv.kv_cache.DiskKVCache` and **does** integrate
with the inference path. It has one undocumented prerequisite: it silently
no-ops unless `--kv-cache-mode int8` (or `snap`) is also passed, because the
store path passes the global `_kv_cache` object which is only set when
`--kv-cache-mode != fp16`. Confirmed on this host:

```
REQ a1bb3ab2  disk-prompt-cache MISS  orig_tokens=41
REQ 38937a3c  disk-prompt-cache HIT   orig_tokens=41 → skipped prefill
```
A 50-token wall request on the same prompt dropped 3.98 s → 2.76 s on the
cached re-run, and a `.npz` landed in the cache dir.

**Phase 2 measures the OLD flag** (which is what v4 inherited and what works)
and notes the gap between the new `PromptKVStore` class and what is actually
reachable through the CLI/server.

### ✓ `squish daemon start` (HTTP) — works as in v3

This is the pre-v4 daemon: spawns `python -m squish.server` in the background
and waits for `127.0.0.1:<port>` to bind. We use this for the Phase 1 daemon
comparison because it loads mlx-native quant correctly.

## Naming/architecture deltas from the original v4 spec

The original prompt described `/tmp/squish.sock`, `squish run --daemon
--prompt …`, and `~/.cache/squish/kv_cache/` as functioning paths. They map
to v4 code as follows:

| Spec described                          | What v4 ships                                                          | Status  |
|-----------------------------------------|------------------------------------------------------------------------|---------|
| `squishd start --foreground`            | `python -m squish.daemon.squishd start --foreground`                   | ✓       |
| `/tmp/squish.sock` UDS                  | `SOCK_PATH` constant in `squish/daemon/squishd.py`                     | ✓ binds |
| Daemon loads any model                  | Daemon expects squish npy-dir; mlx-native quant fails                  | ✗ bug   |
| `squish run --daemon --prompt …`        | Flag present in argparse but `cmd_run` never reads it                  | ✗ dead  |
| `~/.cache/squish/kv_cache/` auto-path   | Default of `PromptKVStore.cache_dir` — class never invoked             | ✗ dead  |
| `--no-spec` flag                        | Flag present in argparse but server ignores it                         | ✗ dead  |
| Spec decode load path                   | `load_draft_model` not re-exported from `squish.speculative.__init__`  | ✗ import error |

## What this means for the v4 benchmark

| Phase                              | Runs? | Measured against                                              |
|------------------------------------|-------|---------------------------------------------------------------|
| 0  Pre-flight                      | ✅    | This file                                                     |
| 1  Daemon "cold wall" (warm-warm)  | ✅    | `squish daemon start` HTTP daemon vs Ollama warm              |
| 2  KV cache TTFT (cold vs warm)    | ✅    | Old `--disk-prompt-cache` + `--kv-cache-mode int8`            |
| 3  Spec-decode throughput          | ❌    | Skipped — server crashes on `--draft-model` (import error)    |
| 4  Revisit RAM / disk / warm tps   | ✅    | Reuses v3 protocol with `squish_daemon` added                 |
| 5  Update RESULTS.md / methodology | ✅    | Measured numbers only — every `projected_` removed            |
| 6  Final honest table              | ✅    | Phase 3 row marked `not-implemented`                          |

## Follow-up items (separate PR — do NOT fix in this session)

1. Re-export `load_draft_model` from `squish/speculative/__init__.py` so
   `--draft-model` doesn't crash at server start.
2. Either make `--disk-prompt-cache` imply `--kv-cache-mode int8` (or fail
   loudly when the prereq is missing), or change the store path so it works
   with the default `fp16` mode.
3. Wire `PromptKVStore` into `server.py`'s inference path, or remove the
   class and revert the methodology.json claims about it.
4. Teach `squishd._load_model` to detect mlx-native quant (the same
   `_model_is_already_quantized` check `server.py` already uses) and call
   `mlx_lm.load()` directly instead of `load_compressed_model`.
5. Wire `args.daemon` and `args.no_spec` in `cmd_run` — both are declared
   in argparse but never read.

These are tracked here so the PR description has a clean follow-up list.
