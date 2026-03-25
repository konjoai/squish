#!/usr/bin/env python3
"""
squish/cli.py

Entry point for the Squish local-inference CLI.

Sub-commands
───────────
  squish pull     MODEL             Download + compress a model
  squish catalog                    Browse available models
  squish run      [MODEL] [OPTIONS] Start the inference server
  squish serve    [MODEL] [OPTIONS] Alias for `squish run`
  squish chat     [MODEL] [OPTIONS] Interactive terminal chat (no browser needed)
  squish compress [MODEL] [OPTIONS] Compress a model to npy-dir format  (legacy: it)
  squish quantize [OPTIONS]         Mixed-precision quantize a model     (legacy: convert-model)
  squish train    MODEL [OPTIONS]   Train a LoRA adapter                (legacy: train-adapter)
  squish merge    MODEL [OPTIONS]   Merge LoRA adapters                 (legacy: merge-model)
  squish config   [KEY] [VAL]       Read/write user configuration
  squish setup                      Interactive setup wizard (detect hw, pull, start)
  squish models                     List local models (auto-discovers ~/.squish/models/)
  squish info                       System info: Metal, RAM, disk
  squish bench    [MODEL] [OPTIONS] Quick throughput/latency benchmark
  squish doctor   [--report]        Check all dependencies
  squish daemon   start|stop|status Manage background server
  squish rotate   MODEL             SpinQuant Cayley-SGD rotation calibration
  squish predict  [MODEL]           LIFE analytical performance prediction

MODEL shorthand resolves via the Squish catalog:
  qwen3:8b, gemma3:4b, deepseek-r1:7b, llama3.2:3b, phi4:14b …
  Legacy aliases still work: 7b, 14b, 1.5b, 32b, 72b
  Any path starting with ~ or / → used as-is

Usage:
    python3 -m squish.cli pull qwen3:8b
    python3 -m squish.cli run 7b
    python3 -m squish.cli chat 7b
    python3 -m squish.cli catalog

After `pip install -e .`:
    squish pull qwen3:8b
    squish run qwen3:8b
    squish chat qwen3:8b
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import NoReturn

# Suppress macOS "can't turn off malloc stack logging" noise in child processes.
# Must run before any subprocesses or uvicorn workers are spawned.  On macOS,
# uvicorn uses 'spawn' (not 'fork') for workers, so children re-inherit the
# raw OS environment — the cleanup in server.py runs too late for those.
for _k in (
    "MallocStackLogging", "MallocStackLoggingNoCompact",
    "MallocScribble", "MallocPreScribble", "MallocGuardEdges",
    "MallocCheckHeapStart", "MallocCheckHeapEach",
):
    os.environ.pop(_k, None)
del _k

# When running as `python3 squish/cli.py` (not via `-m`), the repo root is NOT
# on sys.path, which breaks `from squish.X import ...` inside subcommands like
# compress (AWQ path) and convert.  Inject the repo root so the package is
# always importable regardless of invocation style.
_cli_dir = os.path.dirname(os.path.abspath(__file__))   # …/squish/squish
_repo_root = os.path.dirname(_cli_dir)                   # …/squish
if _repo_root not in sys.path:  # pragma: no cover
    sys.path.insert(0, _repo_root)
# Remove the package directory itself from sys.path.  Python inserts the
# script's own directory as sys.path[0] when running a .py file directly,
# which causes squish sub-packages (squish/token/, squish/grammar/, …)
# to shadow stdlib modules of the same name (token, grammar, …).
while _cli_dir in sys.path:  # pragma: no cover
    sys.path.remove(_cli_dir)
del _cli_dir, _repo_root

# squish.catalog is imported lazily on first use to keep CLI startup fast
# (eager import pulled in urllib/ssl/http chains, adding ~43 ms to every invocation).
_CATALOG_AVAILABLE = True  # catalog is always part of the squish package


def list_catalog(*args, **kwargs):
    from squish.catalog import list_catalog as _lc
    return _lc(*args, **kwargs)


def _catalog_pull(*args, **kwargs):
    from squish.catalog import pull as _pull
    return _pull(*args, **kwargs)


def _catalog_resolve(*args, **kwargs):
    from squish.catalog import resolve as _resolve
    return _resolve(*args, **kwargs)


def _catalog_suggest(query: str, max_results: int = 3):
    """Return up to *max_results* fuzzy catalog matches for *query*."""
    from squish.catalog import suggest as _suggest
    return _suggest(query, max_results=max_results)



# ── Terminal colours ─────────────────────────────────────────────────────────
# All palette selection (dark/light 24-bit vs terminal-native ANSI vs no-color)
# is handled centrally in squish._term.  Importing C as _C here keeps all
# downstream uses of _C.P, _C.V, _C.T etc. unchanged while eliminating the
# duplicate detection logic that previously lived in this file.
# Respects NO_COLOR, SQUISH_DARK_BG, COLORFGBG, and FORCE_COLOR env vars.
from squish._term import C as _C  # noqa: E402


# ── Model registry ───────────────────────────────────────────────────────────

# Resolve models directory: SQUISH_MODELS_DIR env var → ~/.squish/models → <repo>/models → ~/models (legacy)
def _resolve_models_dir() -> Path:
    env_override = os.environ.get("SQUISH_MODELS_DIR", "").strip()
    if env_override:
        return Path(env_override).expanduser()
    # Check ~/.squish/models (canonical install location)
    primary = Path.home() / ".squish" / "models"  # pragma: no cover
    if primary.exists():  # pragma: no cover
        return primary
    # Check <squish repo root>/models/ — works when running directly from the repo
    repo_models = Path(__file__).resolve().parent.parent / "models"  # pragma: no cover
    if repo_models.exists():  # pragma: no cover
        return repo_models
    # Check ~/models (legacy location)
    legacy = Path.home() / "models"  # pragma: no cover
    if legacy.exists():  # pragma: no cover
        return legacy
    return primary  # pragma: no cover  # default even if absent — gives a consistent error path

_MODELS_DIR = _resolve_models_dir()

# Legacy shorthand → directory name (kept for backward compatibility).
# New models should be added to squish/catalog.py instead.
_MODEL_SHORTHAND = {
    # Qwen 2.5
    "1.5b":  "Qwen2.5-1.5B-Instruct-bf16",
    "7b":    "Qwen2.5-7B-Instruct-bf16",
    "14b":   "Qwen2.5-14B-Instruct-bf16",
    "32b":   "Qwen2.5-32B-Instruct-bf16",
    "72b":   "Qwen2.5-72B-Instruct-bf16",
    # Qwen 3
    "qwen3:0.6b":   "Qwen3-0.6B-bf16",
    "qwen3:1.7b":   "Qwen3-1.7B-bf16",
    "qwen3:4b":     "Qwen3-4B-bf16",
    "qwen3:8b":     "Qwen3-8B-bf16",
    "qwen3:14b":    "Qwen3-14B-bf16",
    "qwen3:30b-a3b":"Qwen3-30B-A3B-bf16",
    "qwen3:32b":    "Qwen3-32B-bf16",
    # Llama 3.x
    "llama3.2:1b":  "Llama-3.2-1B-Instruct-bf16",
    "llama3.2:3b":  "Llama-3.2-3B-Instruct-bf16",
    "llama3.1:8b":  "Meta-Llama-3.1-8B-Instruct-bf16",
    # Gemma 3
    "gemma3:1b":    "gemma-3-1b-it-bf16",
    "gemma3:4b":    "gemma-3-4b-it-bf16",
    "gemma3:12b":   "gemma-3-12b-it-bf16",
    "gemma3:27b":   "gemma-3-27b-it-bf16",
    # DeepSeek-R1
    "deepseek-r1:7b":  "DeepSeek-R1-Distill-Qwen-7B-bf16",
    "deepseek-r1:14b": "DeepSeek-R1-Distill-Qwen-14B-bf16",
    "deepseek-r1:32b": "DeepSeek-R1-Distill-Qwen-32B-bf16",
    "r1:7b":           "DeepSeek-R1-Distill-Qwen-7B-bf16",
    "r1:14b":          "DeepSeek-R1-Distill-Qwen-14B-bf16",
    # Phi-4
    "phi4:14b":     "phi-4-bf16",
    # Mistral
    "mistral:7b":   "Mistral-7B-Instruct-v0.3-bf16",
    # SmolLM2
    "smollm2:135m": "SmolLM2-135M-Instruct-bf16",
    "smollm2:360m": "SmolLM2-360M-Instruct-bf16",
    "smollm2:1.7b": "SmolLM2-1.7B-Instruct-bf16",
}

# Compressed dir naming convention
_COMPRESSED_SUFFIX = "-compressed"

# Default server port
_DEFAULT_PORT = 11435


def _detect_ram_gb() -> float:
    """Return total UMA memory in GB (macOS sysctl hw.memsize)."""
    try:
        import subprocess as _sp
        out = _sp.check_output(["sysctl", "-n", "hw.memsize"], stderr=_sp.DEVNULL)
        return int(out.strip()) / 1e9
    except Exception:
        return 0.0


def _recommend_model(ram_gb: float) -> str:
    """Return catalog model ID best suited for available UMA RAM."""
    if ram_gb >= 64:
        return "qwen3:32b"
    if ram_gb >= 32:
        return "qwen3:14b"
    if ram_gb >= 24:
        return "llama3.3:70b"
    if ram_gb >= 16:
        return "qwen3:8b"
    return "qwen3:1.7b"


def _detect_local_ai_services() -> list[dict]:
    """
    Probe well-known local AI service ports and return a list of detected services.

    Each entry has keys: name, base_url, models (list[str]), model_count (int).
    Never raises — all probe errors are silently swallowed.
    """
    import urllib.error
    import urllib.request

    _SERVICES = [
        ("Ollama",    "http://127.0.0.1:11434", "/api/tags"),
        ("LM Studio", "http://127.0.0.1:1234",  "/v1/models"),
        ("Jan",       "http://127.0.0.1:1337",  "/v1/models"),
        ("LocalAI",   "http://127.0.0.1:8080",  "/v1/models"),
    ]

    detected: list[dict] = []
    for name, base_url, path in _SERVICES:
        try:
            req = urllib.request.Request(
                base_url + path,
                headers={"Accept": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=0.5) as resp:
                raw = json.loads(resp.read())
            # Ollama returns {"models": [...]}; OpenAI-compat returns {"data": [...]}
            if isinstance(raw, dict):
                model_list = raw.get("models") or raw.get("data") or []
                models = [
                    (m.get("name") or m.get("id") or "")
                    for m in model_list
                    if isinstance(m, dict)
                ]
            else:
                models = []
            detected.append({
                "name":        name,
                "base_url":    base_url,
                "models":      [m for m in models if m],
                "model_count": len([m for m in models if m]),
            })
        except Exception:
            continue
    return detected


def _open_browser_when_ready(url: str, port: int, timeout_s: int = 30) -> None:
    """
    Spawn a detached subprocess that polls ``http://127.0.0.1:<port>/health``
    until it returns HTTP 200, then opens *url* in the default browser.

    Uses ``subprocess.Popen`` with ``start_new_session=True`` instead of a
    daemon thread so the watcher process survives ``os.execv()`` (which is
    called immediately after this function returns to replace the current
    process with the uvicorn server).
    """
    import subprocess

    poll_script = (
        "import time, urllib.request, webbrowser; "
        f"url={url!r}; health='http://127.0.0.1:{port}/health'; deadline=time.time()+{timeout_s}\n"
        "while time.time() < deadline:\n"
        "    try:\n"
        "        r = urllib.request.urlopen(health, timeout=0.5)\n"
        "        if r.status == 200:\n"
        "            webbrowser.open(url)\n"
        "            break\n"
        "    except Exception:\n"
        "        pass\n"
        "    time.sleep(0.5)\n"
    )
    subprocess.Popen(
        [sys.executable, "-c", poll_script],
        start_new_session=True,
    )


def _resolve_model(name: str | None, quant_mode: str = "int4") -> tuple[Path, Path]:  # pragma: no cover
    """
    Resolve MODEL shorthand / path to (model_dir, compressed_dir).

    quant_mode selects which compressed variant to load: "int4" (default),
    "int3", "int2", or "int8".  The compressed dir is expected at
    ``<models_dir>/<ModelBase>-<quant_mode>``, e.g. ``Qwen3-8B-int4``.

    Raises SystemExit if the path doesn't exist.
    """
    if name is None:
        # Auto-pick: prefer 7B if available, else first available
        for shorthand in ("7b", "14b", "1.5b"):
            candidate = _MODELS_DIR / _MODEL_SHORTHAND[shorthand]
            if candidate.exists():
                name = shorthand
                break
        if name is None:
            _die("No model specified and no default found in ~/models/\n"
                 "Usage: squish run 7b  or  squish run ~/models/my-model")

    if name in _MODEL_SHORTHAND:
        model_dir = _MODELS_DIR / _MODEL_SHORTHAND[name]
    elif _CATALOG_AVAILABLE:
        # Try the dynamic catalog (handles qwen3:8b, gemma3:4b, etc.)
        entry = _catalog_resolve(name)
        if entry is not None:
            model_dir = _MODELS_DIR / entry.dir_name
        else:
            model_dir = Path(name).expanduser()
    else:
        model_dir = Path(name).expanduser()

    if not model_dir.exists():
        hint = name if (name and "/" not in str(name)) else "qwen3:8b"
        _die(
            f"Model directory not found: {model_dir}\n"
            f"  Run:  squish pull {hint}  to download it.\n"
            f"  Browse available models: squish catalog"
        )

    # ── Squished-only dir detection ───────────────────────────────────────────
    # Squished dirs have manifest.json but no config.json. They contain weights
    # only; the base model dir provides the architecture config. Auto-detect it.
    if (model_dir / "manifest.json").exists() and not (model_dir / "config.json").exists():
        import re as _re
        _stem = _re.sub(r'(-squished-.+|-compressed)$', '', model_dir.name)
        _base_dir: Path | None = None
        for _sfx in ("-bf16", "-fp16", ""):
            _cand = model_dir.parent / (_stem + _sfx)
            if _cand.exists() and (_cand / "config.json").exists():
                _base_dir = _cand
                break
        if _base_dir is not None:
            print(f"  \u2139  Squished weights dir — using base model for config: {_base_dir.name}")
            return _base_dir, model_dir
        _die(
            f"Squished model dir has no config.json and no matching base model was found:\n"
            f"  Weights : {model_dir}\n"
            f"  Expected: {model_dir.parent / (_stem + '-bf16')}\n"
            f"  Tip: ensure the base BF16 model directory exists alongside the squished dir."
        )

    # Build compressed dir using <ModelBase>-<quant> convention, e.g. Qwen3-8B-int4
    import re as _re
    _base = _re.sub(r'-(bf16|fp16|[0-9]+bit)(-mlx)?$', '', model_dir.name)
    compressed_dir = model_dir.parent / f"{_base}-{quant_mode}"

    if not compressed_dir.exists():
        # Backward compat: try old <model>-compressed dirs for existing installations
        _old_compressed = Path(str(model_dir) + _COMPRESSED_SUFFIX)
        if _old_compressed.exists():
            compressed_dir = _old_compressed
        else:
            # Also try mlx_lm native -4bit dir
            _squish4bit = model_dir.parent / (model_dir.name.replace("-bf16", "") + "-4bit")
            if _squish4bit.exists():
                compressed_dir = _squish4bit
            else:
                print(f"  ⚠  No {quant_mode.upper()} compressed dir found at {compressed_dir}")
                print(f"     To download: squish pull {name or ''} --{quant_mode}")
                print("     Starting with uncompressed model (slower load)…")
                compressed_dir = model_dir

    return model_dir, compressed_dir


def _die(msg: str) -> NoReturn:
    print(f"\n  {_C.PK}✗{_C.R}  {_C.W}{msg}{_C.R}\n", file=sys.stderr)
    sys.exit(1)


def _box(lines: list[str]) -> None:
    """Print a styled box around lines using squish brand colours."""
    V = _C.V; W = _C.W; R = _C.R
    width = max(len(ln) for ln in lines) + 4
    print(f"{V}┌{'─' * width}┐{R}")
    for ln in lines:
        print(f"{V}│{R}  {W}{ln:<{width-2}}{R}{V}│{R}")
    print(f"{V}└{'─' * width}┘{R}")


# ── squish models ─────────────────────────────────────────────────────────────

def cmd_models(args):
    """List available local models with rich table formatting."""
    from squish.ui import console, make_table, hint, success, warn, _RICH_AVAILABLE
    import rich.box as _rbox  # noqa: F401 (unused if rich absent)

    if not _MODELS_DIR.exists():
        warn(f"Models directory not found: {_MODELS_DIR}")
        hint("Run: squish pull qwen3:8b")
        return

    rows = []
    # Build catalog lookup so we can annotate MoE models in the listing
    _catalog_by_dir: dict = {}
    try:
        from squish.catalog import list_catalog
        for _ce in list_catalog():
            _catalog_by_dir[_ce.dir_name] = _ce
    except Exception:  # noqa: BLE001
        pass

    for d in sorted(_MODELS_DIR.iterdir()):
        if not d.is_dir():
            continue
        if d.name.startswith("."):
            continue
        compressed = Path(str(d) + _COMPRESSED_SUFFIX)
        comp_str = "✓ ready" if compressed.exists() else "raw only"
        # estimate disk size
        try:
            total = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
            size_str = f"{total / 1e9:.1f} GB"
        except Exception:
            size_str = "?"
        # MoE badge from catalog metadata
        _entry = _catalog_by_dir.get(d.name)
        if _entry is not None and getattr(_entry, "moe", False):
            _active = getattr(_entry, "active_params_b", None)
            badge = (
                f"MoE {_entry.params}/{_active:.1f}B active"
                if _active is not None
                else "MoE"
            )
        else:
            badge = ""
        rows.append((d.name, size_str, comp_str, badge))

    if not rows:
        warn("No model directories found.")
        hint("Download a model with: squish pull qwen3:8b")
        hint("Browse all models    : squish catalog")
        return

    if _RICH_AVAILABLE:
        from rich.console import Console as _Con
        from rich.table import Table as _Tbl
        from rich import box as _box
        tbl = _Tbl(
            title="Local Models",
            box=_box.SIMPLE,
            header_style="rgb(139,92,246) bold",
            border_style="rgb(100,116,139)",
            show_lines=False,
            title_style="rgb(248,250,252) bold",
        )
        tbl.add_column("Model", style="rgb(248,250,252)")
        tbl.add_column("Disk", style="rgb(167,139,250)", justify="right")
        tbl.add_column("Status", style="rgb(52,211,153)")
        tbl.add_column("Notes", style="rgb(100,116,139)")
        for name, size, comp, badge in rows:
            tbl.add_row(name, size, comp, badge)
        console.print()
        console.print(tbl)
        console.print()
        console.print("  [rgb(100,116,139)]Aliases   :[/rgb(100,116,139)] [rgb(248,250,252)]1.5b, 7b, 14b, 32b, 72b[/rgb(248,250,252)]")
        console.print("  [rgb(100,116,139)]Catalog IDs:[/rgb(100,116,139)] [rgb(248,250,252)]qwen3:8b, gemma3:4b, deepseek-r1:7b, llama3.2:3b …[/rgb(248,250,252)]")
        console.print("  [rgb(100,116,139)]Commands  :[/rgb(100,116,139)] [rgb(248,250,252)]squish catalog[/rgb(248,250,252)]  [rgb(100,116,139)]·[/rgb(100,116,139)]  [rgb(248,250,252)]squish pull qwen3:8b[/rgb(248,250,252)]")
        console.print()
    else:
        print()
        print(f"  Local models in {_MODELS_DIR}:")
        print()
        w0 = max(len(r[0]) for r in rows) + 2
        print(f"  {'Model':<{w0}} {'Disk':>8}  {'Status'}")
        print(f"  {'─'*w0} {'─'*8}  {'─'*14}")
        for name, size, comp, badge in rows:
            note = f"  [{badge}]" if badge else ""
            print(f"  {name:<{w0}} {size:>8}  {comp}{note}")
        print()
        print("  Legacy aliases : 1.5b, 7b, 14b, 32b, 72b")
        print("  Catalog IDs    : qwen3:8b, gemma3:4b, deepseek-r1:7b, llama3.2:3b …")
        print("  Browse catalog : squish catalog")
        print("  Download model : squish pull qwen3:8b")
        print()

    # ── External models (Ollama / LM Studio) ─────────────────────────────────
    try:
        from squish.serving.local_model_scanner import LocalModelScanner as _Scn
        _ext = _Scn()
        _ext_models = _ext.scan_ollama() + _ext.scan_lm_studio()
        if _ext_models:
            print("  External models detected:")
            print()
            _w_src  = max(len(m.source) for m in _ext_models)
            _w_name = max(len(m.name)   for m in _ext_models)
            print(f"  {'Source':<{_w_src+2}}  {'Model':<{_w_name+2}}  Import hint")
            print(f"  {'─'*(_w_src+2)}  {'─'*(_w_name+2)}  {'─'*42}")
            for _m in _ext_models:
                _hint = f"squish import {_m.source}:{_m.name}"
                print(f"  {_m.source:<{_w_src+2}}  {_m.name:<{_w_name+2}}  {_hint}")
            print()
    except Exception:
        pass


# ── squish rm ────────────────────────────────────────────────────────────────

def cmd_compat(args):
    """
    Print client configuration snippets for popular AI tools.

    No server required — prints env-var and config snippets for:
    OpenAI SDK, Ollama CLI, Open WebUI, Continue.dev, LocalAI, aider, and more.
    """
    port = getattr(args, "port", 11435)
    host = getattr(args, "host", "localhost")
    base = f"http://{host}:{port}"

    rows = [
        ("OpenAI SDK",    f"OPENAI_BASE_URL={base}/v1  OPENAI_API_KEY=squish"),
        ("Ollama CLI",    f"OLLAMA_HOST={base}"),
        ("Open WebUI",    f"Set Ollama API → {base}"),
        ("Continue.dev",  f'"apiBase": "{base}/v1"'),
        ("LocalAI client",f"LOCALAI_API_BASE={base}"),
        ("aider",         f"--openai-api-base {base}/v1 --openai-api-key squish"),
        ("Cursor",        f"Custom model → {base}/v1"),
        ("LM Studio",     f"BaseURL → {base}/v1"),
        ("LangChain",     f'openai_api_base="{base}/v1", openai_api_key="squish"'),
        ("Anything LLM",  f"OpenAI-compatible API → {base}/v1"),
    ]

    w_client = max(len(r[0]) for r in rows)
    print(f"\n  {_C.P}Squish Drop-in Compatibility Snippets{_C.R}  {_C.DIM}(port {port}){_C.R}")
    print(f"  {_C.DIM}{'─' * (w_client + 4 + 60)}{_C.R}")
    print(f"  {_C.T}{'Client':<{w_client+2}}{_C.R}  {_C.T}Configuration{_C.R}")
    print(f"  {'─' * (w_client + 2)}  {'─' * 58}")
    for client, snippet in rows:
        print(f"  {client:<{w_client+2}}  {_C.DIM}{snippet}{_C.R}")
    print()
    print(f"  {_C.DIM}squish run <model> starts the server on port {port}{_C.R}")
    print()


def cmd_rm(args):  # pragma: no cover
    """Remove a local model (raw weights and/or compressed dir)."""
    import shutil

    name = args.model

    # Resolve directories without requiring them to exist
    model_dir: Path | None = None
    compressed_dir: Path | None = None

    # Try catalog first (so short names like qwen3:8b work)
    try:
        from squish.catalog import list_catalog
        entries = {e.id: e for e in list_catalog()}
        if name in entries:
            model_dir       = _MODELS_DIR / entries[name].dir_name
            compressed_dir  = Path(str(model_dir) + _COMPRESSED_SUFFIX)
        elif name in _MODEL_SHORTHAND:
            model_dir       = _MODELS_DIR / _MODEL_SHORTHAND[name]
            compressed_dir  = Path(str(model_dir) + _COMPRESSED_SUFFIX)
        else:
            p = Path(name).expanduser()
            model_dir       = p if p.is_absolute() else _MODELS_DIR / name
            compressed_dir  = Path(str(model_dir) + _COMPRESSED_SUFFIX)
    except Exception:
        p = Path(name).expanduser()
        model_dir       = p if p.is_absolute() else _MODELS_DIR / name
        compressed_dir  = Path(str(model_dir) + _COMPRESSED_SUFFIX)

    has_raw  = model_dir.exists()
    has_comp = compressed_dir.exists() and compressed_dir != model_dir

    if not has_raw and not has_comp:
        _die(f"No local files found for model '{name}'.\n"
             f"  Expected raw dir : {model_dir}\n"
             f"  Expected comp dir: {compressed_dir}")

    # Build list of what will be removed
    targets: list[tuple[str, Path]] = []
    if has_raw and (args.compressed_only is False or not args.compressed_only):
        targets.append(("raw weights", model_dir))
    if has_comp and not args.raw_only:
        targets.append(("compressed weights", compressed_dir))

    if not targets:
        print("Nothing to remove (flags excluded all targets).")
        return

    print()
    print(f"  Will remove the following directories for '{name}':")
    total_bytes = 0
    for label, path in targets:
        try:
            sz = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
        except Exception:
            sz = 0
        total_bytes += sz
        print(f"    [{label}]  {path}  ({sz / 1e9:.2f} GB)")
    print(f"  Total: {total_bytes / 1e9:.2f} GB will be freed")
    print()

    if args.dry_run:
        print("  --dry-run: no files removed.")
        return

    # Confirm unless --yes
    if not args.yes:
        ans = input("  Type 'yes' to confirm deletion: ").strip().lower()
        if ans != "yes":
            print("  Aborted.")
            return

    for label, path in targets:
        print(f"  Removing {label}: {path} …", end=" ", flush=True)
        try:
            shutil.rmtree(path)
            print("done.")
        except Exception as exc:
            print(f"ERROR: {exc}")

    print()
    print("  Done. Run 'squish models' to verify.")
    print()


# ── squish search ─────────────────────────────────────────────────────────────

def cmd_search(args):
    """Search the catalog for models matching a query string."""
    from squish.catalog import search

    hits = search(args.query)

    if not hits:
        print(f"  No catalog entries match '{args.query}'.")
        return

    print()
    print(f"  Catalog search results for '{args.query}':")
    print()
    w_id   = max(len(e.id) for e in hits) + 2
    w_para = max(len(str(getattr(e, 'params', ''))) for e in hits) + 2
    print(f"  {'ID':<{w_id}} {'Params':>{w_para}}  Tags")
    print(f"  {'─'*w_id} {'─'*max(w_para,6)}  {'─'*24}")
    for e in hits:
        tags_str = ", ".join(getattr(e, "tags", [])) or "—"
        params   = str(getattr(e, "params", "—"))
        print(f"  {e.id:<{w_id}} {params:>{w_para}}  {tags_str}")
    print()
    print("  Pull a model: squish pull <id>")
    print()


# ── squish info ───────────────────────────────────────────────────────────────

def cmd_info(args):  # pragma: no cover
    """Print system info relevant to local inference."""
    import platform
    import subprocess as sp

    print()
    _box(["Squish — System Info"])
    print()

    # macOS / chip info
    print(f"  OS            : {platform.system()} {platform.release()}")
    try:
        chip = sp.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            text=True, stderr=sp.DEVNULL).strip()
        print(f"  Chip          : {chip}")
    except Exception:
        pass

    # Unified memory
    try:
        mem_bytes = int(sp.check_output(
            ["sysctl", "-n", "hw.memsize"], text=True).strip())
        mem_gb = mem_bytes / 1e9
        print(f"  Unified RAM   : {mem_gb:.0f} GB")
        print(f"  Metal budget  : {mem_gb * 0.90:.1f} GB  (90% of RAM — Phase 0.1)")
    except Exception:
        pass

    # Disk space
    m = _MODELS_DIR
    if m.exists():
        stat = shutil.disk_usage(m)
        print(f"  Models dir    : {m}")
        print(f"  Disk free     : {stat.free / 1e9:.1f} GB")
        print(f"  Disk used     : {stat.used / 1e9:.1f} GB  (total: {stat.total / 1e9:.0f} GB)")

    # Python / MLX
    try:
        import mlx.core as mx
        print(f"  MLX           : v{mx.__version__}  (Metal backend active)")
    except Exception:
        print("  MLX           : not installed")
    print(f"  Python        : {sys.version.split()[0]}")

    # Models available
    if m.exists():
        n_models = sum(1 for d in m.iterdir() if d.is_dir() and not d.name.startswith("."))
        n_comp   = sum(1 for d in m.iterdir() if d.is_dir() and (Path(str(d)+_COMPRESSED_SUFFIX).exists()))
        print(f"  Local models  : {n_models} model(s),  {n_comp} compressed")

    # Server status
    import socket
    s = socket.socket()
    s.settimeout(0.5)
    try:
        s.connect(("127.0.0.1", _DEFAULT_PORT))
        print(f"  Server        : ✓ running on :{_DEFAULT_PORT}")
    except Exception:
        print("  Server        : not running  (start with: squish run 7b)")
    finally:
        s.close()
    print()


# ── squish setup ─────────────────────────────────────────────────────────────

def cmd_setup(args):  # pragma: no cover
    """Interactive setup wizard: detect hardware, recommend + pull model, start server."""
    import platform as _platform

    print()
    _box(["  squish setup — Interactive Setup Wizard"])
    print()

    # 1. Hardware detection
    is_apple_silicon = _platform.system() == "Darwin" and _platform.machine() == "arm64"
    ram_gb = _detect_ram_gb()
    chip = "Apple Silicon" if is_apple_silicon else _platform.machine()
    ok_sym  = f"{_C.G}✓{_C.R}"
    err_sym = f"{_C.PK}✗{_C.R}"

    print(f"  {ok_sym if is_apple_silicon else err_sym}  Platform    : {_platform.system()} {chip}")
    print(f"  {ok_sym}  RAM         : {ram_gb:.0f} GB")

    if not is_apple_silicon:
        print(f"\n  {_C.PK}squish requires Apple Silicon (M1–M5). Aborting.{_C.R}\n")
        sys.exit(1)

    # 2. Recommend a model
    recommended = _recommend_model(ram_gb)
    try:
        import shutil as _shutil
        disk_free_gb = _shutil.disk_usage(Path.home()).free / 1e9
    except Exception:
        disk_free_gb = 999.0
    print(f"  {ok_sym}  Disk free   : {disk_free_gb:.1f} GB")
    print(f"  {ok_sym}  Recommended : {recommended}")
    print()

    # 3. Check / pull the recommended model
    if _CATALOG_AVAILABLE:
        entry = _catalog_resolve(recommended)
        if entry is not None:
            local_dir = _MODELS_DIR / entry.dir_name
            local_comp = Path(str(local_dir) + _COMPRESSED_SUFFIX)
            already_local = local_comp.exists() or local_dir.exists()
            if already_local:
                print(f"  {ok_sym}  {recommended} already downloaded.\n")
            else:
                try:
                    answer = input(
                        f"  Pull {recommended} now?  [{_C.G}Y{_C.R}/n] "
                    ).strip().lower()
                except (EOFError, KeyboardInterrupt):
                    answer = "n"
                if answer in ("", "y", "yes"):
                    import argparse as _ap2
                    _pull_args = _ap2.Namespace(
                        model=recommended, int4=False, int8=False,
                        int3=False, int2=False, token=None,
                        models_dir=None, refresh_catalog=False, verbose=True,
                    )
                    cmd_pull(_pull_args)
                else:
                    print(f"\n  Skipped. Run `squish pull {recommended}` whenever you're ready.\n")
                    return
    else:
        print(f"  Catalog unavailable. Run `squish pull {recommended}` manually.\n")
        return

    # 4. Offer to start the server
    try:
        answer = input(f"\n  Start server now?  [{_C.G}Y{_C.R}/n] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        answer = "n"
    if answer in ("", "y", "yes"):
        import argparse as _ap2  # noqa: F811
        run_args = _ap2.Namespace(
            model=recommended,
            port=_DEFAULT_PORT, host="127.0.0.1", api_key="squish",
            draft_model="", batch_scheduler=False, batch_size=8,
            kv_cache_mode="fp16", log_level="warning",
            all_optimizations=False,  # cmd_run will set True (default) unless --stock
            agent=False,
            stock=False,
            no_agent=False,
            moe_lookahead=False,
            whatsapp=False, whatsapp_verify_token="",
            whatsapp_app_secret="", whatsapp_access_token="",
            whatsapp_phone_number_id="",
            system_prompt="",
            signal=False, signal_account="", signal_socket="127.0.0.1:7583",
        )
        cmd_run(run_args)
    else:
        print(f"\n  Run when ready:  squish run {recommended}\n")


# ── squish run ────────────────────────────────────────────────────────────────

def cmd_run(args):  # pragma: no cover
    """Start the Squish inference server."""

    # ── Detect other local AI services ───────────────────────────────────────
    _local_services = _detect_local_ai_services()
    if _local_services:
        _svc_names = ", ".join(s["name"] for s in _local_services)
        print(f"\n  ℹ  Detected local AI: {_svc_names}")
        print("     Squish can run larger models with faster TTFT — proceeding with squish.\n")

    # ── Smart defaults ────────────────────────────────────────────────────────
    # No model specified + no local models → auto-pull the RAM-appropriate default
    if not args.model and _CATALOG_AVAILABLE:
        has_local = (
            _MODELS_DIR.exists()
            and any(
                d.is_dir() and not d.name.startswith(".")
                for d in _MODELS_DIR.iterdir()
            )
        )
        if not has_local:
            ram_gb  = _detect_ram_gb()
            default = _recommend_model(ram_gb)
            print(f"\n  No local models found — auto-pulling {default} "
                  f"for your {ram_gb:.0f} GB machine…")
            import argparse as _ap2
            _pull_args = _ap2.Namespace(
                model=default, int4=False, int8=False,
                int3=False, int2=False, token=None,
                models_dir=None, refresh_catalog=False, verbose=True,
            )
            cmd_pull(_pull_args)
            args.model = default

    # Default: run the full squish optimization stack.
    # Pass --stock to get a plain mlx_lm/Ollama-equivalent baseline with no
    # squish-specific optimizations (useful for benchmarking against stock tools).
    # Pass --agent to additionally enable the agent-mode KV/grammar preset.
    if not getattr(args, "stock", False):
        args.all_optimizations = True

    # Auto-pull named model if specified but not yet downloaded locally
    if args.model and _CATALOG_AVAILABLE:
        if args.model in _MODEL_SHORTHAND:
            _expected_dir: Path = _MODELS_DIR / _MODEL_SHORTHAND[args.model]
        else:
            _cat_entry = _catalog_resolve(args.model)
            _expected_dir = (
                _MODELS_DIR / _cat_entry.dir_name
                if _cat_entry is not None
                else Path(args.model).expanduser()
            )
        if not _expected_dir.exists():
            print(f"\n  Model '{args.model}' not found locally — pulling now…")
            import argparse as _ap2
            _pull_args = _ap2.Namespace(
                model=args.model, int4=False, int8=False,
                int3=False, int2=False, token=None,
                models_dir=None, refresh_catalog=False, verbose=True,
            )
            cmd_pull(_pull_args)

    # ── RAM-aware quant auto-selection for >8B models ─────────────────────────
    # Only applies when no explicit quant flag was given by the user.
    if _CATALOG_AVAILABLE and args.model and not any(
        getattr(args, q, False) for q in ("int2", "int3", "int4", "int8")
    ):
        _ram_gb = _detect_ram_gb()
        _auto_entry = _catalog_resolve(args.model)
        if _auto_entry is not None:
            _sq_gb = getattr(_auto_entry, "squished_size_gb", 0.0) or 0.0
            if _sq_gb > _ram_gb * 0.75:
                args.int2 = True
                _est_gb = _sq_gb * 0.55
                print(f"  ℹ  Auto-selecting INT2 (~{_est_gb:.1f} GB est.) for {_ram_gb:.0f} GB RAM")
            elif _sq_gb > _ram_gb * 0.55:
                args.int3 = True
                print(f"  ℹ  Auto-selecting INT3 for {_ram_gb:.0f} GB RAM")

    _quant_mode = (
        "int3" if getattr(args, "int3", False) else
        "int2" if getattr(args, "int2", False) else
        "int8" if getattr(args, "int8", False) else
        "int4"
    )
    model_dir, compressed_dir = _resolve_model(args.model, quant_mode=_quant_mode)

    # Explicit --compressed-dir overrides the auto-detected compressed path.
    if getattr(args, "compressed_dir", None):
        _explicit_comp = Path(args.compressed_dir).expanduser()
        if not _explicit_comp.exists():
            _die(f"--compressed-dir not found: {_explicit_comp}")
        compressed_dir = _explicit_comp

    # Phase 16A: verify model hash integrity before serving
    if _CATALOG_AVAILABLE and args.model:
        _entry = _catalog_resolve(args.model)
        if _entry is not None:
            from squish.catalog import verify_hash as _verify_hash
            _ok, _msg = _verify_hash(_entry, compressed_dir)
            if _msg:
                _sym = "⚠" if not _ok else "ℹ"
                print(f"\n  {_sym}  {_msg}")
            if not _ok:
                print("  Continuing anyway — run `squish pull` to re-download.")

    server_script = Path(__file__).resolve().parent / "server.py"
    if not server_script.exists():
        _die(f"server.py not found at {server_script}")

    # Ensure the repo root (parent of this package) is in PYTHONPATH so that
    # squish.server (and any subprocess it spawns) can always import squish.*
    # without the squish/squish/ directory landing on sys.path and shadowing
    # stdlib modules like 'token' (squish/token/__init__.py) or 'grammar'.
    _repo_root_str = str(Path(__file__).resolve().parent.parent)
    _env_pythonpath = os.environ.get("PYTHONPATH", "")
    if _repo_root_str not in _env_pythonpath.split(os.pathsep):
        os.environ["PYTHONPATH"] = _repo_root_str + (
            os.pathsep + _env_pythonpath if _env_pythonpath else ""
        )

    port     = args.port or _DEFAULT_PORT
    host     = args.host or "127.0.0.1"
    api_key  = args.api_key or "squish"

    # Warn when binding to a non-loopback address — server will be reachable on LAN
    if host not in ("127.0.0.1", "localhost", "::1"):
        import warnings
        warnings.warn(
            f"\n  ⚠  CORS is wide-open and the server will be reachable from your LAN "
            f"at http://{host}:{port}/v1.\n"
            f"  Set a strong SQUISH_API_KEY env var or bind to 127.0.0.1 if unintended.",
            stacklevel=0,
        )

    _mode_label = "stock (no optimizations)" if getattr(args, "stock", False) else (
        "agent + all optimizations" if getattr(args, "agent", False)
        else "squish (all optimizations)"
    )
    print()
    _box([
        "  Squish — Local Inference Server  ",
        f"  Model     : {model_dir.name}",
        f"  Mode      : {_mode_label}",
        f"  Endpoint  : http://{host}:{port}/v1",
        f"  Web UI    : http://{host}:{port}/chat",
        f"  API key   : {api_key}",
        "",
        "  OpenAI drop-in:",
        f"    OPENAI_BASE_URL=http://{host}:{port}/v1",
        f"    OPENAI_API_KEY={api_key}",
        "",
        "  Ollama drop-in:",
        f"    OLLAMA_HOST=http://{host}:{port}",
        "",
        "  Press Ctrl+C to stop",
    ])
    print()

    cmd = [
        sys.executable, "-m", "squish.server",
        "--model-dir",      str(model_dir),
        "--compressed-dir", str(compressed_dir),
        "--port",           str(port),
        "--host",           host,
        # API key is passed via env var to avoid exposure in `ps aux`
    ]
    # Inject into the environ that execv will inherit
    os.environ.setdefault("SQUISH_API_KEY", api_key)
    if args.draft_model:
        cmd += ["--draft-model", args.draft_model]
    if args.batch_scheduler:
        cmd += ["--batch-scheduler", "--batch-size", str(args.batch_size)]
    if args.kv_cache_mode and args.kv_cache_mode != "fp16":
        cmd += ["--kv-cache-mode", args.kv_cache_mode]
    if getattr(args, "log_level", "warning") != "warning":
        cmd += ["--log-level", args.log_level]
    if getattr(args, "trace_output", ""):
        cmd += ["--trace", "--trace-output", args.trace_output]
    if getattr(args, "all_optimizations", False):
        cmd += ["--all-optimizations"]
    if getattr(args, "agent", False):
        cmd += ["--agent"]
    if getattr(args, "whatsapp", False):
        cmd += ["--whatsapp"]
    if getattr(args, "whatsapp_verify_token", ""):
        cmd += ["--whatsapp-verify-token", args.whatsapp_verify_token]
    if getattr(args, "whatsapp_app_secret", ""):
        cmd += ["--whatsapp-app-secret", args.whatsapp_app_secret]
    if getattr(args, "whatsapp_access_token", ""):
        cmd += ["--whatsapp-access-token", args.whatsapp_access_token]
    if getattr(args, "whatsapp_phone_number_id", ""):
        cmd += ["--whatsapp-phone-number-id", args.whatsapp_phone_number_id]
    if getattr(args, "system_prompt", ""):
        cmd += ["--system-prompt", args.system_prompt]
    if getattr(args, "thinking_budget", -1) >= 0:
        cmd += ["--thinking-budget", str(args.thinking_budget)]
    if getattr(args, "signal", False):
        cmd += ["--signal"]
    if getattr(args, "signal_account", ""):
        cmd += ["--signal-account", args.signal_account]
    if getattr(args, "signal_socket", "") and args.signal_socket != "127.0.0.1:7583":
        cmd += ["--signal-socket", args.signal_socket]

    try:
        # Strip macOS malloc-debugging env vars so child processes don't inherit them
        # and produce spurious "MallocStackLogging: can't turn off … not enabled" noise.
        for _msl_key in ("MallocStackLogging", "MallocStackLoggingNoCompact",
                         "MallocScribble", "MallocGuardEdges", "MallocPreScribble"):
            os.environ.pop(_msl_key, None)
        # ── Auto-open browser unless --no-browser flag is set ─────────────────
        if not getattr(args, "no_browser", False):
            _chat_url = f"http://{host}:{port}/chat"
            _open_browser_when_ready(_chat_url, port)
        os.execv(sys.executable, cmd)  # replace this process — clean signals
    except Exception as e:
        _die(f"Failed to start server: {e}")


# ── squish chat ───────────────────────────────────────────────────────────────

def cmd_chat(args):  # pragma: no cover
    """
    Interactive terminal chat against a running (or auto-started) server.

    If no server is running, starts one in a subprocess first.
    Uses Server-Sent Events streaming for real-time token display.
    """
    import socket
    import urllib.error
    import urllib.request

    port    = args.port or _DEFAULT_PORT
    host    = args.host or "127.0.0.1"
    api_key = args.api_key or "squish"
    base_url = f"http://{host}:{port}/v1"

    # ── Auto-start server if not running ─────────────────────────────────────
    _server_proc = None

    def _server_up() -> bool:
        s = socket.socket()
        s.settimeout(1.0)
        try:
            s.connect((host, port))
            s.close()
            return True
        except Exception:
            return False

    if not _server_up():
        model_dir, compressed_dir = _resolve_model(args.model)
        print(f"  Starting server for {model_dir.name} …")
        _repo_root_str = str(Path(__file__).resolve().parent.parent)
        _cur_pp = os.environ.get("PYTHONPATH", "")
        _pythonpath = _repo_root_str + (os.pathsep + _cur_pp if _cur_pp else "")
        _server_proc = subprocess.Popen([
            sys.executable, "-m", "squish.server",
            "--model-dir",      str(model_dir),
            "--compressed-dir", str(compressed_dir),
            "--port",           str(port),
            "--host",           host,
            # API key via env var — keeps it out of `ps aux`
        ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
           cwd=_repo_root_str,
           env={k: v for k, v in {
               **os.environ,
               "SQUISH_API_KEY": api_key,
               "PYTHONPATH": _pythonpath,
           }.items() if not k.startswith("Malloc")})

        # Wait (up to 30s) for server to come up
        for _ in range(60):
            time.sleep(0.5)
            if _server_up():
                break
        else:
            _server_proc.terminate()
            _die("Server did not start within 30s. Check logs above.")

        print(f"  ✓ Server ready on {base_url}")

    # ── Chat loop ─────────────────────────────────────────────────────────────
    messages = []
    model    = args.chat_model or "squish"

    SYSTEM = (args.system or
              "You are a knowledgeable, concise assistant running entirely locally on "
              "Apple Silicon. You have full privacy — nothing leaves this machine.")
    if SYSTEM:
        messages.append({"role": "system", "content": SYSTEM})

    print()
    print("  Squish Chat  (type /quit to exit, /clear to reset, /system to change system prompt)")
    print("  ─────────────────────────────────────────────────────────────────")
    print()

    def _stream_chat(msgs: list) -> str:
        """Send messages, stream tokens to stdout, return full response."""
        payload = json.dumps({
            "model":       model,
            "messages":    msgs,
            "max_tokens":  args.max_tokens,
            "temperature": args.temperature,
            "stream":      True,
        }).encode()
        req = urllib.request.Request(
            f"{base_url}/chat/completions",
            data    = payload,
            headers = {
                "Content-Type":  "application/json",
                "Authorization": f"Bearer {api_key}",
            },
        )
        full = ""
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                for raw_line in resp:
                    line = raw_line.decode("utf-8").strip()
                    if not line.startswith("data: "):
                        continue
                    payload_str = line[6:]
                    if payload_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(payload_str)
                        delta = chunk["choices"][0]["delta"].get("content", "")
                        if delta:
                            print(delta, end="", flush=True)
                            full += delta
                    except Exception:
                        pass
        except urllib.error.URLError as e:
            print(f"\n  ✗ Request failed: {e}")
        print()
        return full

    try:
        while True:
            try:
                user_input = input("  You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n  Goodbye.")
                break

            if not user_input:
                continue
            if user_input.lower() in ("/quit", "/exit", "quit", "exit"):
                print("  Goodbye.")
                break
            if user_input.lower() == "/clear":
                messages = [m for m in messages if m["role"] == "system"]
                print("  ✓ Conversation cleared.")
                continue
            if user_input.lower().startswith("/system "):
                new_sys = user_input[8:].strip()
                messages = [m for m in messages if m["role"] != "system"]
                messages.insert(0, {"role": "system", "content": new_sys})
                print("  ✓ System prompt updated.")
                continue
            if user_input.lower() == "/help":
                print("  Commands: /quit  /clear  /system <text>  /help")
                continue

            messages.append({"role": "user", "content": user_input})
            print("\n  Assistant: ", end="", flush=True)
            reply = _stream_chat(messages)
            if reply:
                messages.append({"role": "assistant", "content": reply})
            # Rolling context window — keeps non-system messages within limit
            max_hist = getattr(args, "max_history", 40)
            non_sys = [m for m in messages if m["role"] != "system"]
            if len(non_sys) > max_hist:
                system_msgs = [m for m in messages if m["role"] == "system"]
                messages = system_msgs + non_sys[-max_hist:]
            print()

    finally:
        if _server_proc is not None:
            _server_proc.terminate()


# ── squish bench ──────────────────────────────────────────────────────────────

def _cmd_bench_track(args):  # pragma: no cover
    """Dispatch to the new squish.benchmarks track runners."""
    from squish.benchmarks.base import parse_engines
    from squish.benchmarks.compare import ResultComparator
    from squish.benchmarks.report import ReportGenerator

    _TRACK_MAP = {
        "quality": "squish.benchmarks.quality_bench.QualityBenchRunner",
        "code":    "squish.benchmarks.code_bench.CodeBenchRunner",
        "tool":    "squish.benchmarks.tool_bench.ToolBenchRunner",
        "agent":   "squish.benchmarks.agent_bench.AgentBenchRunner",
        "perf":    "squish.benchmarks.perf_bench.PerfBenchRunner",
    }

    track = args.track
    model = getattr(args, "model", "squish")
    engines_spec = getattr(args, "engines", "squish")
    limit = getattr(args, "limit", None)
    out_dir = getattr(args, "out", "eval_output")

    try:
        engines = parse_engines(engines_spec)
    except ValueError as exc:
        _die(str(exc))
        return

    # Import the runner class
    module_path, class_name = _TRACK_MAP[track].rsplit(".", 1)
    try:
        import importlib
        mod = importlib.import_module(module_path)
        runner_cls = getattr(mod, class_name)
    except ImportError as exc:
        _die(f"Could not import {module_path}: {exc}")
        return

    runner = runner_cls()
    print(f"\n  squish bench --track {track}  model={model}  engines={engines_spec}")
    if limit:
        print(f"  limit={limit}")
    print()

    results = []
    for engine in engines:
        print(f"  [{engine.name}] running {track} track …")
        try:
            record = runner.run(engine, model, limit=limit)
        except Exception as exc:  # noqa: BLE001
            print(f"  [{engine.name}] ERROR: {exc}")
            continue
        out_path = runner.output_path(engine.name, model, base_dir=out_dir)
        record.save(out_path)
        print(f"  [{engine.name}] saved → {out_path}")
        results.append(record)

        # Print metrics summary
        for k, v in record.metrics.items():
            print(f"    {k}: {v}")
        print()

    if not results:
        print("  No results collected.")
        return

    # Optionally compare / report
    if getattr(args, "compare", False):
        print("  Generating comparison table …")
        try:
            comp = ResultComparator()
            output = comp.generate(write_files=True)
            for path in output.values():
                print(f"  comparison saved → {path}")
        except Exception as exc:  # noqa: BLE001
            print(f"  compare failed: {exc}")

    if getattr(args, "report", False):
        print("  Generating benchmark report …")
        try:
            rg = ReportGenerator()
            report_path = rg.generate(results=results, write_file=True)
            print(f"  report saved → {report_path}")
        except Exception as exc:  # noqa: BLE001
            print(f"  report failed: {exc}")


def cmd_bench(args):  # pragma: no cover
    """Quick throughput benchmark against a running server."""
    import socket
    import urllib.request

    # ── New benchmark suite tracks ──────────────────────────────────────────
    if getattr(args, "track", None):
        _cmd_bench_track(args)
        return

    port    = args.port or _DEFAULT_PORT
    host    = args.host or "127.0.0.1"
    api_key = args.api_key or "squish"
    base_url = f"http://{host}:{port}/v1"

    # Check server up
    s = socket.socket()
    s.settimeout(1.0)
    try:
        s.connect((host, port))
    except Exception:
        _die(f"No server running on {host}:{port}. Start with: squish run 7b")
    finally:
        s.close()

    # ── Optional cold-start timing ─────────────────────────────────────────
    if getattr(args, "cold_start", False):
        print("  \u23f1  Cold-start: timing model load via /v1/models (may take up to 2 min) \u2026")
        sys.stdout.flush()
        t_cs = time.perf_counter()
        try:
            req_cs = urllib.request.Request(
                f"{base_url}/models",
                headers={"Authorization": f"Bearer {api_key}"},
            )
            with urllib.request.urlopen(req_cs, timeout=120) as _r:
                _r.read()
            cold_ms = (time.perf_counter() - t_cs) * 1000
            print(f"  Cold-start load time : {cold_ms:,.0f} ms  ({cold_ms/1000:.2f}s)")
            print("  Tip: restart the server before running --cold-start for a true cold measurement.")
        except Exception as _e:
            print(f"  Cold-start check failed: {_e}")
        print()

    prompts = [
        "Explain quantum entanglement in two sentences.",
        "What is the time complexity of quicksort?",
        "Write a Python function that reverses a string.",
        "What causes the Northern Lights?",
    ]

    print(f"\n  Squish bench — {len(prompts)} prompts, {args.max_tokens} max tokens")
    print(f"  Server: {base_url}")
    print()

    results = []
    prompted = []
    for i, prompt in enumerate(prompts):
        payload = json.dumps({
            "model": "squish",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": args.max_tokens,
            "temperature": 0.7,
            "stream": True,
        }).encode()
        req = urllib.request.Request(
            f"{base_url}/chat/completions", data=payload,
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {api_key}"},
        )
        t0 = time.perf_counter()
        ttft = None
        n_toks = 0
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                for raw_line in resp:
                    line = raw_line.decode().strip()
                    if not line.startswith("data: "):
                        continue
                    payload_str = line[6:]
                    if payload_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(payload_str)
                        delta = chunk["choices"][0]["delta"].get("content", "")
                        if delta:
                            if ttft is None:
                                ttft = time.perf_counter() - t0
                            n_toks += len(delta.split())  # approximate
                    except Exception:
                        pass
        except Exception as e:
            print(f"  [{i+1}] ERROR: {e}")
            continue

        total = time.perf_counter() - t0
        tps   = n_toks / total if total > 0 else 0
        print(f"  [{i+1}] {prompt[:50]:<50}  TTFT={ttft*1000:.0f}ms  "
              f"{n_toks:>4} tok  {tps:.1f} tok/s")
        results.append({"ttft": ttft, "tps": tps, "n_toks": n_toks})
        prompted.append(prompt)

    if results:
        print()
        avg_ttft = sum(r["ttft"] for r in results if r["ttft"]) / len(results)
        avg_tps  = sum(r["tps"] for r in results) / len(results)
        print(f"  Average TTFT: {avg_ttft*1000:.0f} ms")
        print(f"  Average throughput: {avg_tps:.1f} tok/s (≈word/s)")

        if getattr(args, "markdown", False) or getattr(args, "save", None):
            import platform as _plat
            try:
                chip = subprocess.check_output(
                    ["sysctl", "-n", "machdep.cpu.brand_string"],
                    text=True, stderr=subprocess.DEVNULL,
                ).strip()
            except Exception:
                chip = _plat.machine()
            try:
                mem_bytes = int(subprocess.check_output(
                    ["sysctl", "-n", "hw.memsize"], text=True,
                ).strip())
                mem_gb = f"{mem_bytes/1e9:.0f} GB"
            except Exception:
                mem_gb = "?"
            lines = [
                f"## Squish Benchmark — {time.strftime('%Y-%m-%d')}",
                "",
                f"> Hardware: {chip} · {mem_gb} unified memory",
                f"> Server: {base_url} · {args.max_tokens} max tokens",
                "",
                "| Prompt | TTFT (ms) | Tokens | Tok/s |",
                "|--------|----------:|-------:|------:|",
            ]
            for prompt, r in zip(prompted, results, strict=False):
                lines.append(
                    f"| {prompt[:55]:<55} | "
                    f"{r['ttft']*1000:.0f} | {r['n_toks']} | {r['tps']:.1f} |"
                )
            lines += [
                f"| **Average** | **{avg_ttft*1000:.0f}** | — | **{avg_tps:.1f}** |",
                "",
                "_Reproduced with: `squish bench --markdown`_",
            ]
            md = "\n".join(lines)
            save_path = Path(getattr(args, "save", None) or "squish_bench.md")
            save_path.write_text(md)
            print(f"\n  Markdown table saved to {save_path}")
    print()


# ── CLI entry point ───────────────────────────────────────────────────────────

def cmd_doctor(args):
    """Check that all squish components are installed correctly."""
    import concurrent.futures
    import importlib
    import platform as _platform
    import socket

    print()
    _box(["squish doctor — dependency check"])
    print()

    ok = True
    _results: list[dict] = []

    def _check(label: str, passed: bool, fix: str = "") -> None:
        nonlocal ok
        sym = f"{_C.G}✓{_C.R}" if passed else f"{_C.PK}✗{_C.R}"
        print(f"  {sym}  {label}")
        if not passed:
            ok = False
            if fix:
                print(f"       {_C.DIM}Fix:{_C.R} {fix}")
        _results.append({"label": label, "passed": passed, "fix": fix})

    # OS
    _check("macOS / Apple Silicon",
           _platform.system() == "Darwin" and _platform.machine() == "arm64",
           "squish requires macOS on Apple Silicon (M-series)")

    def _ver_ok(found: str, required: str) -> bool:
        """Return True if found >= required using tuple comparison."""
        try:
            def _to_tuple(v: str):
                return tuple(int(x) for x in v.split("+")[0].split(".") if x.isdigit())
            return _to_tuple(found) >= _to_tuple(required)
        except Exception:  # pragma: no cover
            return True  # unknown format → assume ok

    # Pre-import all slow packages in parallel to reduce wall-clock time.
    _slow_pkgs = ["mlx.core", "mlx_lm", "numpy", "transformers", "zstandard", "squish_quant"]

    def _try_import(pkg: str) -> tuple:
        try:
            return (pkg, importlib.import_module(pkg), None)
        except ImportError as exc:
            return (pkg, None, exc)

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(_slow_pkgs)) as _pool:
        _pkg_cache: dict[str, tuple] = {
            pkg: (mod, err)
            for pkg, mod, err in _pool.map(_try_import, _slow_pkgs)
        }

    # MLX
    _mx_mod, _ = _pkg_cache["mlx.core"]
    if _mx_mod is not None:
        _check(f"mlx ≥ 0.18  (found {_mx_mod.__version__})",
               _ver_ok(_mx_mod.__version__, "0.18"),
               "pip install --upgrade mlx")
    else:  # pragma: no cover
        _check("mlx", False, "pip install mlx")

    # mlx-lm
    _mlxlm_mod, _ = _pkg_cache["mlx_lm"]
    if _mlxlm_mod is not None:
        _mlxlm_version = getattr(_mlxlm_mod, "__version__", "0")
        _check(f"mlx-lm ≥ 0.19  (found {_mlxlm_version})",
               _ver_ok(_mlxlm_version, "0.19"),
               "pip install --upgrade mlx-lm")
    else:  # pragma: no cover
        _check("mlx-lm", False, "pip install mlx-lm")

    # numpy
    _np_mod, _ = _pkg_cache["numpy"]
    if _np_mod is not None:
        _check(f"numpy ≥ 1.26  (found {_np_mod.__version__})",
               _ver_ok(_np_mod.__version__, "1.26"),
               "pip install --upgrade numpy")
    else:  # pragma: no cover
        _check("numpy", False, "pip install numpy")

    # transformers
    _tf_mod, _ = _pkg_cache["transformers"]
    if _tf_mod is not None:
        _check(f"transformers ≥ 4.40  (found {_tf_mod.__version__})",
               _ver_ok(_tf_mod.__version__, "4.40"),
               "pip install --upgrade transformers")
    else:  # pragma: no cover
        _check("transformers", False, "pip install transformers")

    # zstandard
    _zstd_mod, _ = _pkg_cache["zstandard"]
    if _zstd_mod is not None:
        _check(f"zstandard ≥ 0.22  (found {_zstd_mod.__version__})",
               _ver_ok(_zstd_mod.__version__, "0.22"),
               "pip install --upgrade zstandard")
    else:  # pragma: no cover
        _check("zstandard (optional zstd entropy layer)", False, "pip install zstandard")

    # squish_quant Rust extension
    _squant_mod, _ = _pkg_cache["squish_quant"]
    if _squant_mod is not None:
        _check("squish_quant Rust extension (6 GB/s quantizer)", True)
    else:  # pragma: no cover
        _check("squish_quant Rust extension (optional — 4× faster quantization)", False,
               "cd squish_quant_rs && python3 -m maturin build --release && pip install .")

    # squish.quant.quantizer self-test
    try:
        import numpy as np

        from squish.quant.quantizer import (
            mean_cosine_similarity,
            quantize_embeddings,
            reconstruct_embeddings,
        )
        rng = np.random.default_rng(0)
        emb = rng.standard_normal((32, 128)).astype(np.float32)
        r   = quantize_embeddings(emb, group_size=64)
        rec = reconstruct_embeddings(r)
        sim = mean_cosine_similarity(emb, rec)
        _check(f"squish.quant.quantizer round-trip  (cosine={sim:.5f})", sim > 0.999,
               "Run: python3 -m squish.quant.quantizer")
    except Exception as e:  # pragma: no cover
        _check(f"squish.quant.quantizer self-test: {e}", False)

    # Models directory
    models_dir = _MODELS_DIR
    if models_dir.exists():
        n = sum(1 for d in models_dir.iterdir() if d.is_dir() and not d.name.startswith("."))
        _check(f"models dir {models_dir}  ({n} model(s))", True)
    else:  # pragma: no cover
        _check(f"models dir {models_dir}", False,
               f"mkdir -p {models_dir}")

    # Disk space in models dir
    try:
        import shutil as _shutil
        _disk_path = _MODELS_DIR if _MODELS_DIR.exists() else Path.home()
        _stat = _shutil.disk_usage(_disk_path)
        _free_gb = _stat.free / 1e9
        _check(
            f"Disk free: {_free_gb:.1f} GB  (\u2265 5 GB recommended for small models)",
            _free_gb >= 5.0,
            "Free at least 5 GB of disk space before pulling a model",
        )
    except Exception:  # pragma: no cover
        pass  # non-fatal

    # Server status
    s = socket.socket()
    s.settimeout(0.5)
    try:
        s.connect(("127.0.0.1", _DEFAULT_PORT))
        _check(f"server running on :{_DEFAULT_PORT}", True)  # pragma: no cover
    except Exception:  # pragma: no cover
        _check("server not running (optional)", True)  # not an error
    finally:
        s.close()

    print()
    if ok:
        print("  All checks passed. squish is ready.\n")
    else:
        print("  Some checks failed. See fixes above.\n")

    # ── --report: write shareable JSON snapshot ───────────────────────────────
    if getattr(args, "report", False):
        import datetime as _dt
        import platform as _plat
        _report_dir = Path.home() / ".squish"
        _report_dir.mkdir(parents=True, exist_ok=True)
        _ts = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        _report_path = _report_dir / f"doctor-report-{_ts}.json"
        _report = {
            "squish_version": "9.0.0",
            "timestamp": _ts,
            "platform": _plat.platform(),
            "python": _plat.python_version(),
            "overall": "pass" if ok else "fail",
            "checks": _results,
        }
        _report_path.write_text(json.dumps(_report, indent=2))
        print(f"  Report saved to: {_report_path}\n")


def cmd_update(args):
    """Upgrade squish and its core dependencies to the latest published versions."""
    try:
        import importlib.metadata as _im
        current_version = _im.version("squish")
    except Exception:
        current_version = "unknown"

    print()
    _box(["squish update"])
    print(f"  Current version: {current_version}")
    print()

    packages = ["squish", "mlx", "mlx-lm", "huggingface_hub"]
    if getattr(args, "all", False):
        # Include heavy optional deps when --all is passed
        packages += ["mlx-vlm", "transformers", "sentencepiece", "tiktoken"]

    for pkg in packages:
        print(f"  Upgrading {pkg}…")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--upgrade", pkg],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            # Extract new version from pip output if available
            new_lines = [
                ln for ln in result.stdout.splitlines()
                if "Successfully installed" in ln or "already up-to-date" in ln.lower()
                or "already satisfied" in ln.lower()
            ]
            summary = new_lines[-1].strip() if new_lines else "done"
            print(f"    ✓ {summary}")
        else:
            print(f"    ✗ pip install --upgrade {pkg} failed:")
            for ln in (result.stderr or result.stdout).splitlines()[-5:]:
                print(f"      {ln}")

    try:
        import importlib.metadata as _im2
        new_version = _im2.version("squish")
    except Exception:
        new_version = "unknown"

    print()
    if new_version != current_version and new_version != "unknown":
        print(f"  Squish upgraded: {current_version} → {new_version}")
    else:
        print(f"  Squish is up to date ({new_version})")
    print()


def cmd_daemon(args):  # pragma: no cover
    """Start, stop, or check the Squish daemon (persistent background server)."""
    import signal

    pid_file = Path.home() / ".squish" / "daemon.pid"
    log_file = Path.home() / ".squish" / "daemon.log"
    pid_file.parent.mkdir(parents=True, exist_ok=True)

    def _read_pid() -> int | None:
        try:
            return int(pid_file.read_text().strip())
        except Exception:
            return None

    def _is_running(pid: int) -> bool:
        try:
            os.kill(pid, 0)
            return True
        except (ProcessLookupError, PermissionError):
            return False

    action = args.daemon_action or "status"

    if action == "status":
        pid = _read_pid()
        if pid and _is_running(pid):
            print(f"\n  ✓  Squish daemon running  (pid {pid})")
            print(f"     Endpoint : http://{args.host}:{args.port}/v1")
            print(f"     Log      : {log_file}\n")
        else:
            print("\n  ✗  Squish daemon not running  (start with: squish daemon start)\n")
        return

    if action == "stop":
        pid = _read_pid()
        if pid and _is_running(pid):
            try:
                os.kill(pid, signal.SIGTERM)
                time.sleep(1)
                if _is_running(pid):
                    os.kill(pid, signal.SIGKILL)
                pid_file.unlink(missing_ok=True)
                print(f"\n  ✓  Daemon stopped  (pid {pid})\n")
            except Exception as e:
                _die(f"Could not stop daemon: {e}")
        else:
            print("\n  Daemon is not running.\n")
            pid_file.unlink(missing_ok=True)
        return

    if action == "start":
        pid = _read_pid()
        if pid and _is_running(pid):
            print(f"\n  Daemon already running  (pid {pid}).")
            print("  Stop first with: squish daemon stop\n")
            return

        model_dir, compressed_dir = _resolve_model(args.model)
        port    = args.port
        host    = args.host
        api_key = args.api_key

        print(f"\n  Starting Squish daemon for {model_dir.name} …")
        print(f"  Endpoint : http://{host}:{port}/v1")
        print(f"  Log      : {log_file}\n")

        _repo_root_str = str(Path(__file__).resolve().parent.parent)
        _cur_pp = os.environ.get("PYTHONPATH", "")
        _pythonpath = _repo_root_str + (os.pathsep + _cur_pp if _cur_pp else "")
        with open(log_file, "a") as log:
            proc = subprocess.Popen(
                [
                    sys.executable, "-m", "squish.server",
                    "--model-dir",      str(model_dir),
                    "--compressed-dir", str(compressed_dir),
                    "--port",           str(port),
                    "--host",           host,
                    # API key via env var — keeps it out of `ps aux`
                ],
                stdout=log,
                stderr=log,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
                cwd=_repo_root_str,
                env={k: v for k, v in {
                    **os.environ,
                    "SQUISH_API_KEY": api_key,
                    "PYTHONPATH": _pythonpath,
                }.items() if not k.startswith("Malloc")},
            )

        pid_file.write_text(str(proc.pid))

        # Wait up to 30s for server to respond
        import socket as _sock
        for _ in range(60):
            time.sleep(0.5)
            s = _sock.socket()
            s.settimeout(0.5)
            try:
                s.connect((host, port))
                s.close()
                print(f"  ✓  Daemon ready  (pid {proc.pid})\n")
                return
            except Exception:
                pass
        print(f"  ⚠  Daemon started (pid {proc.pid}) but hasn't responded yet.")
        print(f"     Check logs: tail -f {log_file}\n")


def cmd_compress(args):  # pragma: no cover
    """Compress a model directory to Squish npy-dir or .squizd format."""
    # ── Resolve explicit --format to bool flags for backward compat ──────────
    _compress_format = getattr(args, "compress_format", None)
    if _compress_format == "int8":
        # Explicit int8: override any --int4 flag
        args.int4 = False
    elif _compress_format == "int4":
        args.int4 = True
    elif _compress_format in ("astc", "hybrid"):
        # ASTC / hybrid: check hardware capability; fall back to INT4 if unsupported
        try:
            from squish.loaders.astc_loader import ASTCLoader
            _loader = ASTCLoader()
            if not _loader.supports_astc_6x6_hdr():
                print(
                    f"\n  Warning: --format {_compress_format} requires Apple Silicon with ASTC "
                    "texture support.\n"
                    "  Falling back to INT4 compression on this hardware.\n"
                )
                _compress_format = "int4"
                args.int4 = True
        except ImportError:
            print(
                f"\n  Warning: ASTC loader unavailable; falling back to INT4.\n"
            )
            _compress_format = "int4"
            args.int4 = True

    # Resolve model path (accept shorthand or full path)
    if args.model in _MODEL_SHORTHAND:
        model_dir = _MODELS_DIR / _MODEL_SHORTHAND[args.model]
    elif _CATALOG_AVAILABLE:
        entry = _catalog_resolve(args.model)
        if entry is not None:
            model_dir = _MODELS_DIR / entry.dir_name
        else:
            model_dir = Path(args.model).expanduser()
    else:
        model_dir = Path(args.model).expanduser()

    if not model_dir.exists():
        _die(f"Model directory not found: {model_dir}")

    output_dir = Path(args.output).expanduser() if args.output else Path(str(model_dir) + _COMPRESSED_SUFFIX)

    _use_int4  = getattr(args, "int4", False)
    _no_awq    = getattr(args, "no_awq", False)
    # AWQ runs automatically with --int4 unless --no-awq is passed.
    # Can also be forced on for INT8 with explicit --awq.
    _run_awq   = (not _no_awq and _use_int4) or getattr(args, "awq", False)

    # ── Lock file — prevent two concurrent compressions to the same output ───
    _lock_path = output_dir.parent / f".squish_compress_{output_dir.name}.lock"
    if _lock_path.exists():
        try:
            _lock_pid = int(_lock_path.read_text().strip())
            # Check if that PID is still alive
            import signal as _signal
            os.kill(_lock_pid, 0)  # raises if dead
            _die(
                f"Compression already running (PID {_lock_pid}) for {output_dir.name}.\n"
                f"  If that process is gone, delete: {_lock_path}"
            )
        except (ValueError, ProcessLookupError, PermissionError):
            _lock_path.unlink(missing_ok=True)  # stale lock — clean it up

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    _lock_path.write_text(str(os.getpid()))

    try:
        _cmd_compress_inner(args, model_dir, output_dir, _use_int4, _no_awq, _run_awq)
    finally:
        _lock_path.unlink(missing_ok=True)


def _ram_available_gb() -> tuple[float, float]:
    """Return (total_ram_gb, free_ram_gb) without requiring psutil.

    On macOS uses sysctl hw.memsize for total and vm_stat for free.
    On Linux falls back to /proc/meminfo.
    Returns (0.0, 0.0) if detection fails — callers must handle gracefully.
    """
    import ctypes
    import ctypes.util
    import subprocess as _sp
    total = 0.0
    free  = 0.0
    try:
        if sys.platform == "darwin":
            _libc = ctypes.CDLL(ctypes.util.find_library("c"))
            _buf  = ctypes.c_uint64(0)
            _sz   = ctypes.c_size_t(8)
            _libc.sysctlbyname(b"hw.memsize", ctypes.byref(_buf), ctypes.byref(_sz), None, 0)
            total = _buf.value / 1e9
            _vm = _sp.check_output(["vm_stat"], text=True)
            _page = 16384  # Apple Silicon / Intel page size
            _pages: dict[str, int] = {}
            for _line in _vm.splitlines()[1:]:
                if ":" in _line:
                    _k, _v = _line.split(":", 1)
                    try:
                        _pages[_k.strip().rstrip(".")] = int(_v.strip().rstrip("."))
                    except ValueError:
                        pass
            free = (_pages.get("Pages free", 0)
                    + _pages.get("Pages inactive", 0)
                    + _pages.get("Pages speculative", 0)) * _page / 1e9
        elif sys.platform.startswith("linux"):
            with open("/proc/meminfo") as _f:
                _info = {l.split(":")[0]: int(l.split()[1]) for l in _f if ":" in l}
            total = _info.get("MemTotal", 0) / 1e6
            free  = (_info.get("MemFree", 0) + _info.get("MemAvailable", 0)) / 2e6
    except Exception:
        pass
    return total, free


def _bf16_native_available() -> bool:
    """Return True if the Rust extension's BF16-native quantization functions are available."""
    try:
        import squish_quant  # noqa: PLC0415
        return hasattr(squish_quant, "quantize_int8_bf16")
    except ImportError:
        return False


def _max_tensor_gb_from_shards(shard_files: list) -> float:
    """
    Scan safetensors shard metadata (header only — no data loaded) to find
    the largest single tensor size in GB.  Falls back to shard-level size
    if the safetensors API is unavailable.
    """
    try:
        from safetensors import safe_open  # noqa: PLC0415
    except ImportError:
        # Fallback: assume largest tensor is ~25% of the largest shard
        return max(f.stat().st_size for f in shard_files) / 1e9 * 0.25

    max_bytes = 0
    _dtype_bytes = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8, "I8": 1, "I32": 4, "I64": 8}
    for shard in shard_files:
        try:
            with safe_open(str(shard), framework="numpy") as f:
                for key in f.keys():
                    sl = f.get_slice(key)
                    shape = sl.get_shape()
                    nbytes = 1
                    for dim in shape:
                        nbytes *= dim
                    dtype_str = str(sl.get_dtype()).upper()
                    nbytes *= _dtype_bytes.get(dtype_str, 2)
                    if nbytes > max_bytes:
                        max_bytes = nbytes
        except Exception:
            pass
    if max_bytes == 0:
        return max(f.stat().st_size for f in shard_files) / 1e9 * 0.25
    return max_bytes / 1e9


def _peak_ram_estimate_gb(model_dir: "Path", run_awq: bool) -> "tuple[float, float, float]":
    """
    Compute (model_size_gb, peak_gb, max_tensor_or_shard_gb) for RAM pre-flight checks.

    AWQ path   — mlx_lm.load loads the entire model as BF16 (+activations)
                  → peak = model_size × 2.0

    No-AWQ path — iter_shard_tensors() (Phase 2) streams one tensor at a time via
                  safetensors safe_open (OS demand-paged mmap).  The peak is
                  determined by the largest single tensor, not the full shard:

                  With BF16-native Rust quantization (Phase 3, no f32 copy needed):
                    → peak ≈ max_tensor_size × 1.5  (BF16 + INT4/INT8 output)

                  Without BF16-native (legacy f32 cast path):
                    → peak ≈ max_tensor_size × 3.5  (BF16 + f32 copy + output)

                  Headroom: 1.0 GB (model config files, tokenizer, Python overhead)
    """
    shard_files = sorted(model_dir.glob("*.safetensors"))
    model_size_gb = sum(
        f.stat().st_size for f in model_dir.rglob("*") if f.is_file()
    ) / 1e9
    if shard_files and not run_awq:
        max_tensor_gb = _max_tensor_gb_from_shards(shard_files)
        # BF16-native path (Phase 3): no f32 intermediate, 1× BF16 + 0.5× output
        # Legacy f32 path: 1× BF16 + 2× f32 + 0.5× output = 3.5× BF16 size
        multiplier = 1.5 if _bf16_native_available() else 3.5
        peak_gb = max_tensor_gb * multiplier + 1.0
        max_shard_or_tensor_gb = max_tensor_gb
    else:
        # AWQ loads the full model via mlx_lm.load (BF16) + activation buffers
        peak_gb = model_size_gb * 2.0 + 2.0
        max_shard_or_tensor_gb = model_size_gb
    return model_size_gb, peak_gb, max_shard_or_tensor_gb


def _cmd_compress_inner(args, model_dir, output_dir, _use_int4, _no_awq, _run_awq):
    """Inner body of compression — called after the lock is acquired."""
    # ── Memory safety pre-flight ─────────────────────────────────────────────
    HEADROOM_GB = 2.0
    model_size_gb, peak_gb, max_shard_gb = _peak_ram_estimate_gb(model_dir, _run_awq)
    total_ram_gb, free_ram_gb = _ram_available_gb()

    if total_ram_gb > 0:
        if peak_gb > total_ram_gb:
            # Hard block — will definitely OOM the machine
            if _run_awq:
                detail = "AWQ calibration loads full BF16 model + activation buffers"
            else:
                _mult = "1.5" if _bf16_native_available() else "3.5"
                detail = f"streaming compress: largest tensor ({max_shard_gb:.2f} GB) × {_mult} + headroom"
            msg = (
                f"\n  ✗  Not enough RAM to compress this model safely.\n"
                f"\n"
                f"     Model size  : {model_size_gb:.1f} GB\n"
                f"     Peak needed : ~{peak_gb:.1f} GB  ({detail})\n"
                f"     Total RAM   : {total_ram_gb:.1f} GB\n"
            )
            if _run_awq:
                # Compute no-AWQ peak (shard-based) to see if that fits
                _, no_awq_peak, _ = _peak_ram_estimate_gb(model_dir, run_awq=False)
                if no_awq_peak <= total_ram_gb:
                    msg += (
                        f"\n  Try:  squish it {args.model} --int4 --no-awq"
                        f"\n        (skips AWQ calibration; peak ~{no_awq_peak:.1f} GB, slight accuracy cost)\n"
                    )
                else:
                    msg += (
                        f"\n  Options:\n"
                        f"     1. Use INT8 (larger group-64 quant, ~same accuracy):\n"
                        f"        squish it {args.model} --no-awq\n"
                        f"     2. Use a smaller model:\n"
                        f"        squish it qwen3:4b --int4\n"
                        f"        squish it qwen3:1.7b --int4\n"
                    )
            else:
                msg += (
                    f"\n  Options:\n"
                    f"     1. Use a smaller model whose shards fit in available RAM:\n"
                    f"        squish it qwen3:4b --int4   (largest shard ~2 GB, peak ~{2*3+HEADROOM_GB:.0f} GB)\n"
                    f"        squish it qwen3:1.7b --int4  (largest shard ~0.9 GB, peak ~{0.9*3+HEADROOM_GB:.0f} GB)\n"
                    f"     2. Run on a machine with more RAM (Kaggle CPU: 30 GB free)\n"
                )
            _die(msg)
        elif peak_gb > free_ram_gb + 1.0:
            # Soft warning — might work but will cause heavy swapping
            print(
                f"\n  ⚠  Low free RAM warning:\n"
                f"     Peak needed : ~{peak_gb:.1f} GB\n"
                f"     Free RAM    : {free_ram_gb:.1f} GB  (of {total_ram_gb:.1f} GB total)\n"
                f"     Compression will proceed but may be slow due to memory pressure.\n"
                f"     Close other apps to free up RAM before continuing.\n"
            )
            import time as _time
            _time.sleep(3)  # Give user a moment to read and Ctrl+C if desired

    awq_suffix = " + AWQ" if _run_awq else ""
    quant_label = (
        f"INT4{awq_suffix} (~44% disk savings, ≤2% accuracy delta)"
        if _use_int4 else f"INT8{awq_suffix} group-64"
    )
    print(f"\n  Compressing: {model_dir}")
    print(f"  Quantization: {quant_label}")
    print(f"  Output:      {output_dir}\n")

    # ── AWQ calibration pass (auto for --int4, optional for --int8) ──────────
    awq_scales_dir = None
    if _run_awq:
        n_samples = getattr(args, "awq_samples", 20)
        print(f"  Running AWQ calibration ({n_samples} samples)...")
        print("  Note: loads full model in memory — may take 2–5 min for large models.")
        try:
            import tempfile

            import mlx_lm
            model_awq, tokenizer_awq = mlx_lm.load(str(model_dir))
            from squish.quant.awq import collect_activation_scales, save_awq_scales
            scales = collect_activation_scales(
                model_awq, tokenizer_awq,
                n_samples=n_samples, verbose=True,
            )
            awq_scales_dir = tempfile.mkdtemp(prefix="squish_awq_")
            save_awq_scales(scales, awq_scales_dir, verbose=False)
            print(f"  ✓  AWQ scales collected → {awq_scales_dir}")
            del model_awq
            try:
                import mlx.core as mx
                mx.clear_cache()
            except Exception:
                pass
        except ImportError as _e:
            print(f"  Warning: AWQ skipped — {_e}. Install mlx-lm to enable AWQ.")
        except Exception as _e:
            print(f"  Warning: AWQ calibration failed — {_e}. Continuing without AWQ.")

    cmd = [
        sys.executable, "-m", "squish.convert",
        "--model-dir", str(model_dir),
        "--output",    str(output_dir),
        "--format",    "npy-dir",
    ]
    if args.passthrough:
        cmd += ["--passthrough"] + args.passthrough
    if args.outlier_threshold != 20.0:
        cmd += ["--outlier-threshold", str(args.outlier_threshold)]
    if getattr(args, "int4", False):
        cmd.append("--int4")
    if getattr(args, "int4_group_size", None):
        cmd += ["--int4-group-size", str(args.int4_group_size)]
    if getattr(args, "aqlm", False):
        cmd.append("--aqlm")
        cmd += ["--aqlm-codebooks", str(getattr(args, "aqlm_codebooks", 2))]
        cmd += ["--aqlm-cbsize",    str(getattr(args, "aqlm_cbsize",    16))]
    if awq_scales_dir:
        cmd += ["--awq-scales", awq_scales_dir]
    if args.verbose:
        cmd.append("--verbose")

    print("  Compressing weights — this may take 3–8 min for large models …")
    sys.stdout.flush()
    import threading as _threading

    _compress_done = False

    def _heartbeat():
        elapsed = 0
        while not _compress_done:
            time.sleep(15)
            elapsed += 15
            if not _compress_done:
                print(f"  … still compressing ({elapsed}s elapsed) — please wait", flush=True)

    _hb_thread = _threading.Thread(target=_heartbeat, daemon=True)
    _hb_thread.start()
    # Inherit the environment and inject the repo root into PYTHONPATH so that
    # the subprocess can always find squish.convert regardless of whether squish
    # is pip-installed in sys.executable's site-packages.
    _repo_root_str = str(Path(__file__).parent.parent)
    _env = os.environ.copy()
    _env["PYTHONPATH"] = _repo_root_str + os.pathsep + _env.get("PYTHONPATH", "")
    try:
        result = subprocess.run(cmd, cwd=_repo_root_str, env=_env)
    finally:
        _compress_done = True

    if result.returncode != 0:
        _die("Compression failed — see output above.")
    print(f"\n  ✓  Compressed model saved to {output_dir}")

    # ── Optional zstd entropy pass ──────────────────────────────────────────
    zstd_level = getattr(args, "zstd_level", 0)
    if zstd_level and zstd_level > 0:
        tensors_dir = output_dir / "tensors"
        if tensors_dir.exists():
            print(f"  Applying zstd entropy compression at level {zstd_level} …")
            try:
                from squish.io.entropy import compress_npy_dir
                compress_npy_dir(tensors_dir, level=zstd_level, verbose=True)
                print("  ✓  Entropy compression complete.")
            except ImportError:
                print("  Warning: zstandard not installed — skipping entropy pass.")
                print("  Install with: pip install zstandard")
        else:
            print(f"  Warning: tensors/ not found at {tensors_dir} — skipping entropy pass.")

    print(f"     Run with: squish run {model_dir}\n")


# ── squish pull URI-scheme helpers ─────────────────────────────────────────────

def _pull_from_ollama(ollama_name: str, models_dir: Path, token: str | None) -> None:  # pragma: no cover
    """Pull a model by probing the local Ollama instance then falling back to HF.

    Parameters
    ----------
    ollama_name:
        Ollama model identifier, e.g. ``"qwen3:8b"`` or ``"llama3.1:8b"``.
    models_dir:
        Destination directory for Squish models.
    token:
        HuggingFace token for private repos.
    """
    import urllib.error
    import urllib.request

    # Try to find the model in the local Ollama instance
    ollama_base = "http://127.0.0.1:11434"
    try:
        url = f"{ollama_base}/api/show"
        req = urllib.request.Request(
            url,
            data=json.dumps({"name": ollama_name}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=3) as r:
            _show = json.loads(r.read())
        print(f"  Found {ollama_name!r} in local Ollama — attempting to map to Squish catalog…")
    except (urllib.error.URLError, OSError):
        print(f"\n  Ollama is not running or model {ollama_name!r} not found locally.")
        print("  Start Ollama first:  ollama serve")
        print(f"  Or use direct HF pull:  squish pull hf:mlx-community/{ollama_name.replace(':', '-')}")
        print()
        return

    # Check catalog for this model
    entry = _catalog_resolve(ollama_name)
    if entry is not None and entry.squished_repo:
        print(f"  Catalog match found — pulling pre-compressed: {entry.squished_repo}")
        _pull_from_hf(entry.squished_repo, models_dir, token)
    else:
        # Best-effort: map to mlx-community HF repo
        hf_guess = f"mlx-community/{ollama_name.replace(':', '-')}-bf16"
        print(f"  No catalog match. Attempting HF pull: {hf_guess}")
        _pull_from_hf(hf_guess, models_dir, token)


def _pull_from_hf(hf_repo: str, models_dir: Path, token: str | None) -> None:  # pragma: no cover
    """Download and compress a model from HuggingFace.

    If the repo matches a Squish catalog entry with a ``squish_repo``, the
    pre-compressed weights are downloaded directly.  Otherwise the bf16 source
    is fetched and compressed locally.

    Parameters
    ----------
    hf_repo:
        HuggingFace repository id, e.g. ``"mlx-community/Qwen3-8B-bf16"``.
    models_dir:
        Destination directory for Squish models.
    token:
        HuggingFace token for private repos.
    """
    try:
        from huggingface_hub import snapshot_download  # type: ignore[import]
    except ImportError:
        print("\n  huggingface_hub is required for HF pulls.")
        print("  Install with:  pip install huggingface-hub")
        return

    # Check if any catalog entry matches this HF repo
    if _CATALOG_AVAILABLE:
        from squish.catalog import list_catalog as _list_catalog
        for entry in _list_catalog():
            if getattr(entry, "hf_mlx_repo", "") == hf_repo:
                if entry.squished_repo:
                    print(f"  Catalog match ({entry.id}) — downloading pre-squished from {entry.squished_repo}")
                    # Delegate to standard pull
                    import types
                    fake_args = types.SimpleNamespace(
                        model=entry.id, models_dir=str(models_dir),
                        token=token, int2=False, int3=False, int8=False,
                        verbose=False, force=False,
                    )
                    cmd_pull(fake_args)
                    return
                print(f"  Catalog match ({entry.id}) — no pre-squished available; compressing locally.")
                break

    # Fresh download
    print(f"\n  Downloading from HuggingFace: {hf_repo}")
    print("  (Model is outside catalog — compressing locally after download)")
    dest = snapshot_download(
        repo_id=hf_repo,
        local_dir=str(models_dir / hf_repo.split("/")[-1]),
        token=token,
    )
    print(f"\n  Downloaded to: {dest}")
    print("  To compress: squish compress <model-dir>")
    print()


# ── squish pull ───────────────────────────────────────────────────────────────

def cmd_pull(args):  # pragma: no cover
    """
    Download and compress a model from the Squish catalog.

    If pre-compressed Squish weights exist on HuggingFace they are downloaded
    directly (no on-device compression needed).  Otherwise the bf16 MLX weights
    are fetched and compressed locally.

    Examples
    --------
      squish pull qwen3:8b
      squish pull gemma3:4b --int8
      squish pull deepseek-r1:7b --token hf_…
      squish pull llama3.1:8b --models-dir /Volumes/SSD/models
    """
    if not _CATALOG_AVAILABLE:
        _die("squish.catalog is not available. Ensure the package is properly installed.")

    name = args.model
    models_dir = Path(args.models_dir).expanduser() if args.models_dir else _MODELS_DIR
    token = args.token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    # ── URI scheme dispatch ────────────────────────────────────────────────────
    if name.startswith("ollama:"):
        _pull_from_ollama(name[len("ollama:"):], models_dir, token)
        return
    if name.startswith("hf:") or "huggingface.co" in name:
        hf_ref = name.removeprefix("hf:")
        _pull_from_hf(hf_ref, models_dir, token)
        return

    # Resolve first so we can print a clear error before any download starts
    entry = _catalog_resolve(name)
    if entry is None:
        _die(
            f"Unknown model: {name!r}\n"
            f"Run `squish catalog` to browse available models."
        )

    quant_mode = (
        "int3" if getattr(args, "int3", False) else
        "int2" if getattr(args, "int2", False) else
        "int8" if args.int8 else
        "int4"
    )
    quant_label = quant_mode.upper()
    print()
    _box([
        "  squish pull",
        f"  Model      : {entry.id}  —  {entry.name}",
        f"  Parameters : {entry.params}",
        f"  Raw size   : ~{entry.size_gb:.1f} GB",
        f"  Compressed : ~{entry.squished_size_gb:.1f} GB  ({quant_label})",
        f"  Context    : {entry.context:,} tokens",
        f"  Dest       : {models_dir}",
    ])
    print()

    try:
        compressed_dir = _catalog_pull(
            name=name,
            models_dir=models_dir,
            int4=quant_mode == "int4",
            token=token,
            refresh_catalog=args.refresh_catalog,
            verbose=args.verbose,
            quant_mode=quant_mode,
        )
    except ImportError as exc:
        _die(str(exc))
    except ValueError as exc:
        _die(str(exc))
    except RuntimeError as exc:
        # _SSLError (subclass of RuntimeError) and other runtime errors get a
        # clean, actionable message rather than a full traceback.
        _die(str(exc))

    print()
    _box([
        f"  ✓  {entry.id} ready!",
        f"  Run  : squish run {entry.id}",
        f"  Chat : squish chat {entry.id}",
        f"  Path : {compressed_dir}",
    ])
    print()

    # ── optional: pull EAGLE-3 draft head ────────────────────────────────────
    if getattr(args, "with_draft", False):
        print("  Pulling EAGLE-3 draft head …")
        # Resolve model alias → HF repo using the same catalog as pull-head
        hf_repo = _EAGLE_HEAD_CATALOG.get(name.lower(), None)
        if hf_repo is None:
            print(
                f"  ⚠  No pre-distilled EAGLE head found for {name!r}.\n"
                f"     Run `squish pull-head <full-hf-repo>` to supply a custom head."
            )
        else:
            slug = hf_repo.split("/")[-1].lower()
            head_dir = Path.home() / ".squish" / "eagle-heads" / slug
            if head_dir.exists() and any(head_dir.iterdir()):
                print(f"  Draft head already present at {head_dir} — skipping download.")
            else:
                # Delegate to the same logic used by cmd_pull_head
                class _HeadArgs:
                    model = name
                    output = ""
                    token = args.token

                cmd_pull_head(_HeadArgs())


# ── squish import ─────────────────────────────────────────────────────────────

def cmd_import(args):  # pragma: no cover
    """Import a model from Ollama, a GGUF file, or HuggingFace.

    Usage examples
    --------------
      squish import ollama:qwen3:8b
      squish import /path/to/model.gguf
      squish import hf:mlx-community/Qwen3-8B-bf16
    """
    name = args.import_source
    models_dir = Path(getattr(args, "models_dir", None) or _MODELS_DIR).expanduser()
    token = getattr(args, "token", None) or os.environ.get("HF_TOKEN")

    if name.startswith("ollama:"):
        _pull_from_ollama(name[len("ollama:"):], models_dir, token)
    elif name.startswith("hf:") or "huggingface.co" in name:
        _pull_from_hf(name.removeprefix("hf:"), models_dir, token)
    elif Path(name).suffix.lower() in (".gguf", ".bin", ".safetensors"):
        gguf_path = Path(name).expanduser()
        if not gguf_path.exists():
            print(f"\n  File not found: {gguf_path}")
            return
        print(f"\n  Importing GGUF: {gguf_path}")
        print("  squish does not directly convert GGUF — use mlx_lm.convert first:")
        print(f"    python3 -m mlx_lm.convert --hf-path <HF_ID> -q --q-bits 4 --mlx-path {models_dir}")
        print()
    else:
        # Bare model name — try catalog then HF
        entry = _catalog_resolve(name) if _CATALOG_AVAILABLE else None
        if entry is not None:
            import types
            fake_args = types.SimpleNamespace(
                model=name, models_dir=str(models_dir),
                token=token, int2=False, int3=False, int8=False,
                verbose=False, force=False,
            )
            cmd_pull(fake_args)
        else:
            # Last resort: try as HF repo
            _pull_from_hf(name, models_dir, token)


# ── squish catalog ────────────────────────────────────────────────────────────

def cmd_catalog(args):
    """Browse the Squish model catalog."""
    if not _CATALOG_AVAILABLE:  # pragma: no cover
        _die("squish.catalog is not available. Ensure the package is properly installed.")

    entries = list_catalog(
        tag=args.tag or None,
        refresh=args.refresh,
    )

    if not entries:
        tag_msg = f" (tag: {args.tag})" if args.tag else ""
        print(f"\n  No models found{tag_msg}.")
        return

    print()
    _box([
        "  Squish Model Catalog",
        f"  {len(entries)} model(s) available",
        "  Run `squish pull <id>` to download",
    ])
    print()

    # Header
    col_id   = max(len(e.id) for e in entries) + 2
    print(f"  {'ID':<{col_id}} {'Params':>7}  {'Raw':>7}  {'Squished':>9}  {'Prebuilt':>9}  Notes")
    print(f"  {'─'*col_id} {'─'*7}  {'─'*7}  {'─'*9}  {'─'*9}  {'─'*24}")

    for e in entries:
        prebuilt = "⚡ yes" if e.has_prebuilt else "compress"
        notes = e.notes if e.notes else ", ".join(e.tags)
        # Phase 14B: add MoE badge when moe=True
        if getattr(e, "moe", False):
            active = getattr(e, "active_params_b", None)
            if active is not None:
                moe_badge = f"[MoE: {e.params} total / {active:.1f}B active]"
            else:
                moe_badge = "[MoE]"
            notes = f"{moe_badge}  {notes}" if notes else moe_badge
        print(
            f"  {e.id:<{col_id}} {e.params:>7}  "
            f"{e.size_gb:>6.1f}G  {e.squished_size_gb:>8.1f}G  "
            f"{prebuilt:>9}  {notes}"
        )

    print()
    print("  Prebuilt ⚡ = pre-compressed weights on HuggingFace (instant download)")
    print()

    if args.tag:
        print(f"  Showing tag: {args.tag!r}")
        print("  Other tags: small, fast, balanced, large, reasoning, moe, edge")
    else:
        print("  Filter by tag: squish catalog --tag reasoning")
        print("  Refresh list : squish catalog --refresh")
    print()


# ── EAGLE-3 head download (Phase 1B) ─────────────────────────────────────────

_EAGLE_HEAD_CATALOG: dict[str, str] = {
    # model-alias → HuggingFace repo for the EAGLE-3 head
    "qwen3:8b":       "yuhuili/EAGLE3-Qwen3-Instruct-8B",
    "qwen3:4b":       "yuhuili/EAGLE3-Qwen3-Instruct-4B",
    "qwen3:14b":      "yuhuili/EAGLE3-Qwen3-Instruct-14B",
    "qwen3:30b-a3b":  "yuhuili/EAGLE3-Qwen3-Instruct-30B-A3B",
    "qwen2.5:7b":     "yuhuili/EAGLE3-Qwen2.5-Instruct-7B",
    "qwen2.5:14b":    "yuhuili/EAGLE3-Qwen2.5-Instruct-14B",
    "llama3.1:8b":    "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B",
    "llama3.2:3b":    "yuhuili/EAGLE3-Llama-3.2-Instruct-3B",
}


def cmd_pull_head(args):  # pragma: no cover
    """
    Download an EAGLE-3 draft head from HuggingFace and convert it to MLX format.

    Examples
    --------
      squish pull-head qwen3:8b
      squish pull-head yuhuili/EAGLE3-Qwen3-Instruct-8B --output ~/.squish/heads/qwen3-8b
      squish pull-head qwen3:8b --token hf_…
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        _die(
            "huggingface_hub is required for pull-head.\n"
            "Install it with: pip install huggingface-hub"
        )

    model_arg = args.model
    token = args.token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    # Resolve alias → HF repo
    hf_repo = _EAGLE_HEAD_CATALOG.get(model_arg.lower(), model_arg)
    if "/" not in hf_repo:
        _die(
            f"Unknown model alias {model_arg!r}.\n"
            f"Pass a full HuggingFace repo (e.g. yuhuili/EAGLE3-Qwen3-Instruct-8B) "
            f"or one of: {', '.join(_EAGLE_HEAD_CATALOG)}"
        )

    # Determine output directory
    if args.output:
        out_dir = Path(args.output).expanduser()
    else:
        slug = hf_repo.split("/")[-1].lower()
        out_dir = Path.home() / ".squish" / "eagle-heads" / slug

    out_dir.mkdir(parents=True, exist_ok=True)

    print()
    _box([
        "  squish pull-head",
        f"  HF repo    : {hf_repo}",
        f"  Output dir : {out_dir}",
    ])
    print()

    # Download from HuggingFace Hub
    print(f"  Downloading {hf_repo} …")
    raw_dir = snapshot_download(
        repo_id=hf_repo,
        local_dir=str(out_dir / "_raw"),
        token=token or None,
        ignore_patterns=["*.bin"],  # prefer safetensors / MLX
    )

    # If weights are already in MLX format (config.json + *.safetensors),
    # just symlink / copy; otherwise convert via mlx_lm.
    import shutil
    raw_path = Path(raw_dir)
    has_mlx = (raw_path / "config.json").exists() and any(raw_path.glob("*.safetensors"))

    if has_mlx:
        mlx_dir = out_dir
        for f in raw_path.iterdir():
            dst = mlx_dir / f.name
            if not dst.exists():
                shutil.copy2(f, dst)
        print("  Weights already in MLX safetensors format — no conversion needed.")
    else:
        # Convert PyTorch / BF16 safetensors to MLX
        print("  Converting to MLX format …")
        try:
            from mlx_lm import convert as _mlx_convert
            _mlx_convert(
                hf_path=raw_dir,
                mlx_path=str(out_dir),
                quantize=False,  # EAGLE heads are already compact; keep fp16
            )
        except ImportError:
            _die("mlx_lm is required for conversion. Install with: pip install mlx-lm")
        except Exception as exc:
            _die(f"Conversion failed: {exc}")

    # Clean up raw download directory
    shutil.rmtree(out_dir / "_raw", ignore_errors=True)

    print()
    print(f"  EAGLE-3 head saved to: {out_dir}")
    print()
    print("  Start with EAGLE-3 speculative decoding:")
    print(f"    squish run --model <your-model> --eagle-head-dir {out_dir}")
    print()


def _preoptimize_weights_with_hqq(
    source_path: "Path",
    ffn_bits: int,
    group_size: int,
    max_iter: int = 10,
) -> "Path":
    """Apply HQQ pre-optimization to FFN weights before mlx_lm.convert.

    Loads each safetensors shard from *source_path*, applies Half-Quadratic
    Quantization (encode → decode) to all MLP projection weights, and writes
    the float-optimized shards to a temporary directory.  mlx_lm.convert is
    then called on the temp directory instead of the original — resulting in
    dramatically better INT2/INT3 quality because the naive affine quantizer in
    mlx_lm operates on weights that are already aligned to the HQQ-optimal grid.

    Only gate_proj, up_proj, and down_proj weights are pre-optimized (the
    layers where low-bit naive quantization degrades quality the most).
    Attention projections and embeddings are left untouched.

    Args:
        source_path: Path to the BF16 source model directory.
        ffn_bits: Target FFN quantization bits (2 or 3).
        group_size: Quantization group size used for HQQ.
        max_iter: HQQ solver iterations (10 is sufficient for most cases).

    Returns:
        Path to the temporary directory containing pre-optimized weights.
        Caller is responsible for cleanup (``shutil.rmtree``).
    """
    import shutil as _shutil
    import tempfile as _tempfile

    try:
        import mlx.core as _mx
    except ImportError:
        _die("mlx is required for HQQ pre-optimisation. Install with: pip install mlx")

    import numpy as _np

    from squish.quant.hqq_quant import HQQConfig, HQQQuantizer

    _FFN_PATTERNS = ("gate_proj", "up_proj", "down_proj")

    if not source_path.exists():
        _die(f"HQQ pre-optimisation: source path not found: {source_path}")

    cfg = HQQConfig(bits=ffn_bits, group_size=group_size, max_iter=max_iter)
    quantizer = HQQQuantizer(cfg)

    tmp_dir = Path(_tempfile.mkdtemp(prefix="squish_hqq_"))
    try:
        # Copy everything that is NOT a safetensors shard (config, tokenizer, …)
        for item in sorted(source_path.iterdir()):
            if item.suffix == ".safetensors":
                continue
            dest = tmp_dir / item.name
            if item.is_dir():
                _shutil.copytree(item, dest)
            else:
                _shutil.copy2(item, dest)

        shard_files = sorted(source_path.glob("*.safetensors"))
        n_shards = len(shard_files)
        if n_shards == 0:
            _die(
                f"No .safetensors files found in {source_path}. "
                "HQQ pre-optimisation requires a BF16 safetensors model."
            )

        total_optimized = 0
        for shard_idx, shard_path in enumerate(shard_files, 1):
            print(f"  [hqq] shard {shard_idx}/{n_shards}: {shard_path.name}")
            weights: dict = _mx.load(str(shard_path))
            modified_weights: dict[str, "_mx.array"] = {}
            for key, tensor in weights.items():
                # Only optimize FFN projection weight tensors (2-D linear weights)
                is_ffn = any(pat in key for pat in _FFN_PATTERNS)
                is_weight = key.endswith(".weight")
                if is_ffn and is_weight and tensor.ndim == 2:
                    orig_dtype = tensor.dtype
                    W_f32 = _np.array(tensor.astype(_mx.float32))
                    hqq_tensor = quantizer.encode(W_f32)
                    W_adapted = quantizer.decode(hqq_tensor).astype(_np.float32)
                    modified_weights[key] = _mx.array(W_adapted).astype(orig_dtype)
                    total_optimized += 1
                else:
                    modified_weights[key] = tensor

            dest_shard = tmp_dir / shard_path.name
            _mx.save_safetensors(str(dest_shard), modified_weights)

        print(f"  [hqq] pre-optimised {total_optimized} FFN weight tensors → {tmp_dir.name}")
    except Exception:
        _shutil.rmtree(tmp_dir, ignore_errors=True)
        raise

    return tmp_dir


def _apply_blazing_m3_preset(args: "Any") -> "Any":
    """Apply Wave-81 ``--blazing-m3`` preset values to *args* (mutated in-place).

    Preset targets sub-3 s TTFT on M3 16 GB Apple Silicon with Qwen2.5-7B
    or Llama-3.1-8B:

    * FFN weights  → INT2  (``--ffn-bits 2``)
    * Attention    → INT4  (``--attn-bits 4``)
    * Embeddings   → INT8  (``--embed-bits 8``)
    * Group size   → 32    (halves reconstruction error vs 64 at 2% size cost)
    * HQQ          → True  (calibration-free proximal solver improves INT2 quality)

    Only sets a field when the user has not already overridden it via an
    explicit CLI flag (detected via the ``_user_set_*`` sentinel attributes).

    Parameters
    ----------
    args : argparse Namespace (or any object supporting getattr/setattr)

    Returns
    -------
    The same *args* object with fields mutated.
    """
    if not getattr(args, "blazing_m3", False):
        return args

    default_gs = int(getattr(args, "_default_group_size", 64))

    if int(getattr(args, "ffn_bits", 4)) == 4:          # default untouched
        args.ffn_bits = 2
    if getattr(args, "attn_bits", None) is None:         # not set by user
        args.attn_bits = 4
    if int(getattr(args, "embed_bits", 6)) == 6:         # default untouched
        args.embed_bits = 8
    if int(getattr(args, "group_size", 64)) == default_gs:  # default untouched
        args.group_size = 32
    if not getattr(args, "hqq", False):                  # not explicitly set
        args.hqq = True

    return args


def cmd_convert_model(args):
    """
    Convert and optionally quantize a model with mixed-precision quantization.

    Runs two mlx_lm.convert passes for different precision per layer group:
      - FFN layers (all linear except lm_head + embed_tokens): --ffn-bits
      - Embedding/output layers (lm_head, embed_tokens): --embed-bits

    Usage:
      squish convert-model --source-path path/to/model \\
        --output-path path/to/output \\
        --ffn-bits 4 --embed-bits 6
    """
    # ── Wave 81: apply --blazing-m3 preset before any other logic ───────────
    _apply_blazing_m3_preset(args)

    source_path = Path(args.source_path).expanduser().resolve()
    output_path = Path(args.output_path).expanduser().resolve()

    # Allow HF model IDs as source (e.g. "Qwen/Qwen3-14B").  A path that
    # does not exist but contains a forward-slash and no leading / is treated
    # as an HF repository ID and passed through to mlx_lm.convert as-is.
    _is_hf_id = (
        not source_path.exists()
        and "/" in args.source_path
        and not args.source_path.startswith("/")
    )
    if not source_path.exists() and not _is_hf_id:
        _die(f"Source path not found: {source_path}")
    if _is_hf_id:
        # Override resolved path with the raw HF ID string for mlx_lm.convert
        source_path = args.source_path  # type: ignore[assignment]

    attn_bits: int | None = getattr(args, "attn_bits", None)
    group_size: int = getattr(args, "group_size", 64)
    mixed_recipe: str | None = getattr(args, "mixed_recipe", None)

    if args.dry_run:
        print(f"  [dry-run] source      : {source_path}")
        print(f"  [dry-run] output      : {output_path}")
        print(f"  [dry-run] ffn-bits    : {args.ffn_bits}")
        if attn_bits is not None and attn_bits != args.ffn_bits:
            print(f"  [dry-run] attn-bits   : {attn_bits}")
        print(f"  [dry-run] embed-bits  : {args.embed_bits}")
        print(f"  [dry-run] group-size  : {group_size}")
        if mixed_recipe:
            print(f"  [dry-run] mixed-recipe: {mixed_recipe}")
        return

    # Do NOT pre-create the output directory — mlx_lm.convert refuses to write
    # to an already-existing path, so we let it create the directory itself.

    try:
        import mlx_lm as _mlx_lm
    except ImportError:
        _die("mlx_lm is required for convert-model. Install with: pip install mlx-lm>=0.19")

    if getattr(args, "cpu", False):
        # Force all MLX ops onto CPU to avoid Metal GPU command-buffer timeout
        # that occurs when quantizing large models (>14B) on limited RAM devices.
        try:
            import mlx.core as mx
            mx.set_default_device(mx.cpu)
            print("  [cpu mode] MLX device set to CPU (avoids Metal GPU timeout)")
        except Exception:
            pass  # non-fatal — falls back to GPU if MLX is not importable here

    ffn_bits: int = args.ffn_bits
    embed_bits: int = args.embed_bits
    # attn_bits / group_size already extracted above (before dry-run check)
    effective_attn_bits: int = attn_bits if attn_bits is not None else ffn_bits

    # ── Wave 78: auto-tighten group_size for INT2 ────────────────────────────
    # INT2 has only 4 quantisation levels per group; smaller groups mean more
    # scale/zero parameters so reconstruction error drops sharply.  Default
    # group_size=64 is already fine for INT4; for INT2 we default to 32 unless
    # the user explicitly chose a different value.
    _user_set_gs = (group_size != getattr(args, "_default_group_size", 64))
    if ffn_bits == 2 and not _user_set_gs and group_size > 32:
        print(
            f"  [auto] INT2 detected — tightening group_size {group_size} → 32 "
            "(2× more scale/zero params, ~2% larger model, significantly better quality).\n"
            "  Pass --group-size to override."
        )
        group_size = 32

    # ── Wave 78: small-model quality warning for INT2 ────────────────────────
    if ffn_bits == 2:
        _param_count: int | None = None
        if not _is_hf_id:
            _cfg_path2 = Path(str(source_path)) / "config.json"
            if _cfg_path2.exists():
                try:
                    import json as _json2
                    _cfg2 = _json2.loads(_cfg_path2.read_text())
                    _h_dim = _cfg2.get("hidden_size", 0)
                    _n_l   = _cfg2.get("num_hidden_layers", 0)
                    _i_dim = _cfg2.get("intermediate_size", 0) or _h_dim * 4
                    # Rough parameter count: 2 * n_layers * (3 * h * i) + vocab * h
                    _param_count = (
                        2 * _n_l * (3 * _h_dim * _i_dim)
                        + _cfg2.get("vocab_size", 32000) * _h_dim
                    ) if _h_dim > 0 and _n_l > 0 else None
                except Exception:
                    pass
        if _param_count is not None and _param_count < 1_000_000_000:
            _pc_b = _param_count / 1e9
            print(
                f"\n  WARNING: Model appears to be ≈{_pc_b:.1f}B parameters.\n"
                "  INT2 quality degrades heavily below 1B params — expect ~35% MMLU\n"
                "  (random-chance level).  Consider INT3/INT4 for small models."
            )
        elif ffn_bits == 2:
            print(
                "\n  NOTE: INT2 requires --attn-bits 4 (or higher) for coherent output.\n"
                "  --mixed-recipe mixed_2_6 is strongly recommended for best INT2 quality."
            )

    if mixed_recipe:
        # 4-tier mixed-precision predicate:
        #   embed/lm_head                           → embed_bits (8)
        #   all self_attn/* layers                  → effective_attn_bits (4)
        #   down_proj / v_proj in "critical" layers → high_bits  (6 for mixed_2_6)
        #   everything else (gate/up proj, etc.)    → ffn_bits   (2)
        #
        # "Critical" layers mirror mlx_lm's mixed_2_6 definition:
        #   first 12.5%, last 12.5%, and every 3rd layer in the remainder.
        _high = 6 if mixed_recipe == "mixed_2_6" else 4
        _low  = ffn_bits
        _ab   = effective_attn_bits
        _eb   = embed_bits
        _gs   = group_size

        # Read num_hidden_layers from config.json without loading the model.
        _num_layers = 32  # sensible default
        if not _is_hf_id:
            _cfg_path = Path(str(source_path)) / "config.json"
            if _cfg_path.exists():
                try:
                    import json as _json
                    _num_layers = _json.loads(_cfg_path.read_text()).get(
                        "num_hidden_layers", 32
                    )
                except Exception:
                    pass
        _nl = _num_layers

        def quant_predicate(path: str, _module) -> dict:
            # Extract the first digit segment in the path as the layer index.
            layer_idx = 0
            for part in path.split("."):
                if part.isdigit():
                    layer_idx = int(part)
                    break
            use_high_bits = (
                layer_idx < _nl // 8
                or layer_idx >= 7 * _nl // 8
                or (layer_idx - _nl // 8) % 3 == 2
            )
            is_embed     = "lm_head" in path or "embed_tokens" in path
            is_attn      = "self_attn" in path or "cross_attn" in path
            is_v_proj    = "v_proj" in path or "v_a_proj" in path or "v_b_proj" in path
            is_down_proj = "down_proj" in path

            if is_embed:
                bits = _eb
            elif is_attn:
                bits = _ab
            elif (is_v_proj or is_down_proj) and use_high_bits:
                bits = _high
            else:
                bits = _low
            return {"bits": bits, "group_size": _gs}

        q_bits = _low
        print(f"  Quantizing with {mixed_recipe}: low={_low}-bit · high={_high}-bit · "
              f"attn={_ab}-bit · embed={_eb}-bit · gs={_gs} · num_layers={_nl} …")
    else:
        needs_predicate = (
            ffn_bits != embed_bits
            or effective_attn_bits != ffn_bits
        )

        if not needs_predicate:
            # Uniform quantization — simple path, no per-layer predicate needed.
            quant_predicate = None
            q_bits = ffn_bits
            print(f"  Quantizing all layers to {ffn_bits}-bit (group_size={group_size}) …")
        else:
            # Three-tier mixed-precision:
            #   embed/lm_head  →  embed_bits
            #   self_attn/*    →  effective_attn_bits  (defaults to ffn_bits)
            #   everything else (MLP gate/up/down) →  ffn_bits
            q_bits = ffn_bits
            _gs = group_size  # captured in closure
            _fb = ffn_bits
            _ab = effective_attn_bits
            _eb = embed_bits

            def quant_predicate(path: str, _module) -> bool | dict:  # noqa: E501
                is_embed = "lm_head" in path or "embed_tokens" in path
                is_attn  = "self_attn" in path or "cross_attn" in path
                if is_embed:
                    bits = _eb
                elif is_attn:
                    bits = _ab
                else:
                    bits = _fb
                return {"bits": bits, "group_size": _gs}

            parts = [f"FFN={ffn_bits}-bit"]
            if effective_attn_bits != ffn_bits:
                parts.append(f"attn={effective_attn_bits}-bit")
            if embed_bits != ffn_bits:
                parts.append(f"embed={embed_bits}-bit")
            print(f"  Quantizing: {', '.join(parts)}, group_size={group_size} …")

    try:
        # ── Wave 78: HQQ pre-optimisation pass ──────────────────────────────
        # When --hqq is requested and we're targeting low-bit FFN quantisation,
        # pre-process the BF16 weights through HQQ before mlx_lm.convert.
        # HQQ finds near-optimal scale/zero per group; mlx_lm's naive rounding
        # applied to HQQ-decoded weights produces significantly better results
        # than naive rounding applied to the original BF16 weights.
        use_hqq: bool = getattr(args, "hqq", False) and ffn_bits <= 3 and not _is_hf_id
        _hqq_tmp_dir = None
        _hqq_source = source_path
        if use_hqq:
            print(f"  [hqq] Pre-optimising FFN weights at {ffn_bits}-bit with HQQ …")
            _hqq_tmp_dir = _preoptimize_weights_with_hqq(
                source_path=Path(str(source_path)),
                ffn_bits=ffn_bits,
                group_size=group_size,
            )
            _hqq_source = _hqq_tmp_dir

        _mlx_lm.convert(
            hf_path=str(_hqq_source),
            mlx_path=str(output_path),
            quantize=True,
            q_bits=q_bits,
            q_group_size=group_size,
            quant_predicate=quant_predicate,
        )
    except Exception as exc:
        _die(f"Quantization failed: {exc}")
    finally:
        if _hqq_tmp_dir is not None:
            import shutil as _shutil_hqq
            _shutil_hqq.rmtree(_hqq_tmp_dir, ignore_errors=True)

    print(f"\n  Mixed-precision model saved to: {output_path}")
    print(f"  Load with: squish run --mlx-model-dir {output_path}")


# ── squish check ──────────────────────────────────────────────────────────────

def cmd_check_model(args):
    """Inspect a quantized model and report quantisation quality metrics.

    Loads the model's config.json and safetensors metadata (without loading
    weights into RAM) to report:
      - Detected quantisation bits and group_size per layer type
      - Estimated theoretical reconstruction quality (SNR) via HQQ analysis on
        a random weight sample for each unique (bits, group_size) combination
      - Warnings for known problematic configurations (INT2 with large groups,
        missing attn-bits override, tiny models at extreme low bit-width)

    Usage:
        squish check --model ./path/to/quantized-model
    """
    import json as _json
    import math as _math
    import numpy as _np

    model_path = Path(args.model).expanduser().resolve()
    if not model_path.exists():
        _die(f"Model path not found: {model_path}")

    # ── Read config.json ──────────────────────────────────────────────────────
    cfg_path = model_path / "config.json"
    if not cfg_path.exists():
        _die(f"No config.json found in {model_path}. Is this an mlx_lm model directory?")

    with cfg_path.open() as fh:
        cfg = _json.load(fh)

    model_type    = cfg.get("model_type", "unknown")
    hidden_size   = cfg.get("hidden_size", 0)
    n_layers      = cfg.get("num_hidden_layers", 0)
    vocab_size    = cfg.get("vocab_size", 0)
    inter_size    = cfg.get("intermediate_size", 0) or hidden_size * 4
    quant_cfg     = cfg.get("quantization_config") or cfg.get("quantization") or {}
    global_bits   = quant_cfg.get("bits", None)
    global_gs     = quant_cfg.get("group_size", None)

    print(f"\n  Model:       {model_path.name}")
    print(f"  Type:        {model_type}")
    print(f"  Layers:      {n_layers}")
    print(f"  Hidden dim:  {hidden_size}")
    print(f"  Inter dim:   {inter_size}")

    # Rough parameter count
    if hidden_size > 0 and n_layers > 0:
        n_params = (
            2 * n_layers * (3 * hidden_size * inter_size)
            + vocab_size * hidden_size
        )
        print(f"  Params:      ~{n_params / 1e9:.2f}B")
    else:
        n_params = 0

    print()

    # ── Analyse quantisation config ───────────────────────────────────────────
    if not quant_cfg:
        print("  No quantisation config found — this may be a BF16 (unquantized) model.")
    else:
        if isinstance(quant_cfg, dict) and not any(isinstance(v, dict) for v in quant_cfg.values()):
            # Flat config: applies uniformly to all layers
            print(f"  Quantisation: uniform  bits={global_bits}  group_size={global_gs}")
            _check_layer_config("all layers", global_bits, global_gs, n_params, hidden_size)
        else:
            # Per-path config (mlx_lm mixed-precision)
            bits_seen: dict[str, int] = {}
            gs_seen:   dict[str, int] = {}
            for path, params in quant_cfg.items():
                if not isinstance(params, dict):
                    continue
                b  = params.get("bits", global_bits)
                gs = params.get("group_size", global_gs)
                if "self_attn" in path or "cross_attn" in path:
                    label = "attention"
                elif any(k in path for k in ("gate_proj", "up_proj", "down_proj")):
                    label = "ffn"
                elif any(k in path for k in ("lm_head", "embed_tokens")):
                    label = "embed"
                else:
                    label = "other"
                if b is not None:
                    bits_seen[label] = b
                if gs is not None:
                    gs_seen[label] = gs

            print("  Quantisation config (per layer type):")
            for label in ("ffn", "attention", "embed", "other"):
                b  = bits_seen.get(label)
                gs = gs_seen.get(label)
                if b is None and gs is None:
                    continue
                print(f"    {label:<12}: bits={b}  group_size={gs}")
                _check_layer_config(label, b, gs, n_params, hidden_size)

    # ── HQQ quality simulation on synthetic weights ───────────────────────────
    print()
    print("  Theoretical quality (HQQ simulation on synthetic weights):")
    print("  ─" * 30)

    configs_to_test = []
    if global_bits is not None:
        configs_to_test.append((f"uniform {global_bits}-bit", global_bits, global_gs or 64))
    else:
        for label, b in bits_seen.items() if quant_cfg and isinstance(quant_cfg, dict) else []:
            gs = gs_seen.get(label, 64)
            if b is not None:
                configs_to_test.append((label, b, gs))

    if not configs_to_test:
        configs_to_test = [("(assumed 4-bit)", 4, 64)]

    try:
        from squish.quant.hqq_quant import HQQConfig, HQQQuantizer
        _rng = _np.random.default_rng(42)
        for label, bits, gs in configs_to_test:
            if bits is None:
                continue
            _W = _rng.standard_normal((256, gs * 4)).astype(_np.float32) * 0.02
            _cfg_hqq = HQQConfig(bits=bits, group_size=min(gs, _W.shape[1]), max_iter=10)
            _q = HQQQuantizer(_cfg_hqq)
            _t = _q.encode(_W)
            _r = _q.decode(_t)
            _snr  = _q.quantisation_error_db(_W, _r)
            _lerr = _q.relative_error(_W, _r)
            _flag = ""
            if bits <= 2 and _snr < 10:
                _flag = "  ← WARNING: very low SNR — expect degraded output"
            elif bits <= 3 and _snr < 15:
                _flag = "  ← caution: low SNR, use --hqq and --attn-bits 4"
            print(f"    {label:<20}: SNR={_snr:+.1f} dB  rel_err={_lerr:.4f}{_flag}")
    except ImportError:
        print("    (HQQ simulation unavailable — squish.quant.hqq_quant not found)")

    print()
    print("  Tip: use `squish quantize --hqq --ffn-bits 2 --attn-bits 4` for best INT2 quality.")
    print()


def _check_layer_config(
    label: str,
    bits: "int | None",
    group_size: "int | None",
    n_params: int,
    hidden_size: int,
) -> None:
    """Print warnings for known bad quantisation configurations."""
    if bits is None:
        return
    if bits == 2 and (group_size is None or group_size > 32):
        gs_str = str(group_size) if group_size is not None else "default"
        print(
            f"    WARNING [{label}]: INT2 with group_size={gs_str} is risky. "
            "Use group_size≤32 for coherent output."
        )
    if bits <= 2 and n_params > 0 and n_params < 1_000_000_000:
        print(
            f"    WARNING [{label}]: Model is small (~{n_params/1e9:.1f}B params) — "
            "INT2 quality on models <1B is typically poor (random-level)."
        )
    if bits == 2 and label == "attention":
        print(
            f"    WARNING [{label}]: INT2 attention projections without HQQ "
            "will produce incoherent / looping output. Use --attn-bits 4."
        )


# ── squish train-adapter ───────────────────────────────────────────────────────

def _apply_dare_sparsification(
    adapter_dir: Path,
    sparsity_ratio: float = 0.9,
) -> None:
    """DARE sparsification: zero out *sparsity_ratio* fraction of each delta weight.

    Surviving weights are rescaled by ``1/(1 - sparsity_ratio)`` to preserve
    expected magnitude.  Operates in-place on all ``adapter_model*.safetensors``
    files in *adapter_dir*.

    Does nothing (with a warning) when ``safetensors`` is not installed.
    """
    import numpy as np

    try:
        from safetensors.numpy import load_file, save_file  # noqa: PLC0415
    except ImportError:
        print("  [warn] safetensors not available — skipping DARE sparsification")
        return

    rescale = 1.0 / (1.0 - sparsity_ratio)
    rng = np.random.default_rng(42)

    for st_file in sorted(adapter_dir.glob("adapter_model*.safetensors")):
        orig_size = st_file.stat().st_size
        weights = load_file(str(st_file))
        sparsified = {}
        for key, arr in weights.items():
            mask = (rng.random(arr.shape) > sparsity_ratio).astype(arr.dtype)
            sparsified[key] = (arr * mask * rescale).astype(arr.dtype)
        save_file(sparsified, str(st_file))
        new_size = st_file.stat().st_size
        print(
            f"  DARE: {st_file.name}: {orig_size // 1024} KB → {new_size // 1024} KB"
        )


def cmd_train_adapter(args):
    """Train a LoRA adapter using mlx_lm's built-in LoRA training pipeline.

    After training, applies DARE sparsification to the saved adapter weights
    (sets a random 90% of delta-weight values to zero and rescales the rest).

    Usage::

        squish train-adapter qwen3:8b \\
          --dataset ./data/train.jsonl \\
          --domain legal \\
          --rank 8 --epochs 3 \\
          --output-dir ~/.squish/adapters/legal
    """
    dataset_path = Path(args.dataset).expanduser().resolve()
    if not dataset_path.exists():
        _die(f"Dataset not found: {dataset_path}")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import mlx_lm as _mlx_lm  # noqa: F401,PLC0415
    except ImportError:
        _die(
            "mlx_lm is required for train-adapter. "
            "Install with: pip install mlx-lm>=0.19"
        )

    print(f"  Training LoRA adapter on {args.model!r} …")
    print(f"  Dataset : {dataset_path}")
    print(f"  Domain  : {args.domain}")
    print(f"  Rank    : {args.rank}   Epochs: {args.epochs}")

    try:
        _mlx_lm.lora.train(
            model=args.model,
            dataset=str(dataset_path),
            output_dir=str(output_dir),
            rank=args.rank,
            num_epochs=args.epochs,
            gradient_checkpointing=True,
        )
    except Exception as exc:
        _die(f"LoRA training failed: {exc}")

    # Apply DARE sparsification to the produced adapter weights
    print("  Applying DARE sparsification (0.9 drop rate) …")
    _apply_dare_sparsification(output_dir, sparsity_ratio=0.9)

    print(f"\n  Adapter saved to: {output_dir}")
    print(f"  Run with: squish run {args.model} (load adapter via server API)")


# ── squish merge-model ────────────────────────────────────────────────────────

def _find_adapter_safetensors(path: Path) -> Path:
    """Return the first ``*.safetensors`` file in *path*, or *path* itself if it is one.

    Raises
    ------
    FileNotFoundError
        If no ``.safetensors`` file is found.
    """
    if path.is_file() and path.suffix == ".safetensors":
        return path
    files = sorted(path.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"No .safetensors files found in: {path}")
    return files[0]


def cmd_merge_model(args):
    """Offline DARE+TIES multi-adapter merge.

    Merges one or more LoRA adapters into a single flat safetensors file via
    one of three methods:

    - ``dare``       — DARE sparsification (99% drop) then simple average
    - ``ties``       — TIES sign-conflict resolution then average (no DARE)
    - ``dare-ties``  — DARE sparsification then TIES sign resolution (default)

    Produces ``adapter_model.safetensors`` in ``--output-path``.

    Usage::

        squish merge-model qwen3:8b \\
          --adapters legal:~/.squish/adapters/legal code:~/.squish/adapters/code \\
          --method dare-ties \\
          --output-path ~/.squish/models/qwen3-8b-merged
    """
    import numpy as np

    output_path = Path(args.output_path).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    # Parse "domain:path" adapter specs
    adapter_specs: list[tuple[str, Path]] = []
    for spec in args.adapters:
        if ":" not in spec:
            _die(
                f"Invalid adapter spec {spec!r}. "
                "Expected format: 'domain:path/to/adapter'"
            )
        domain, path_str = spec.split(":", 1)
        adapter_path = Path(path_str).expanduser().resolve()
        if not adapter_path.exists():
            _die(f"Adapter path not found: {adapter_path}")
        adapter_specs.append((domain, adapter_path))

    try:
        from safetensors.numpy import load_file, save_file  # noqa: PLC0415
    except ImportError:
        _die(
            "safetensors is required for merge-model. "
            "Install with: pip install safetensors"
        )

    print(
        f"  Merging {len(adapter_specs)} adapter(s) "
        f"using method={args.method!r} …"
    )

    # Load all adapter weights (flat dict per adapter)
    all_weights: list[dict] = []
    for domain, ap in adapter_specs:
        try:
            st_path = _find_adapter_safetensors(ap)
            w = load_file(str(st_path))
        except Exception as exc:
            _die(f"Failed to load adapter {domain!r}: {exc}")
        all_weights.append(w)

    rng = np.random.default_rng(0)
    merged: dict = {}
    all_keys: set = set()
    for w in all_weights:
        all_keys |= set(w.keys())

    sign_conflicts = 0
    total_keys = len(all_keys)

    for key in sorted(all_keys):
        deltas = [w[key] for w in all_weights if key in w]

        # ── DARE: sparsify each delta at 99% drop rate ────────────────────────
        if args.method in ("dare", "dare-ties"):
            sparsified = []
            rescale = 1.0 / (1.0 - 0.99)
            for d in deltas:
                mask = (rng.random(d.shape) > 0.99).astype(d.dtype)
                sparsified.append(d * mask * rescale)
            deltas = sparsified

        # ── TIES: majority-sign vote, keep only aligned contributions ─────────
        if args.method in ("ties", "dare-ties") and len(deltas) > 1:
            stacked = np.stack(deltas, axis=0)
            majority_sign = np.sign(np.sign(stacked).sum(axis=0))
            aligned = [d * (np.sign(d) == majority_sign) for d in deltas]
            conflicts = int(
                (np.sign(stacked) != majority_sign[None]).any(axis=0).sum()
            )
            sign_conflicts += conflicts
            merged_delta = np.stack(aligned, axis=0).mean(axis=0)
        else:
            merged_delta = np.stack(deltas, axis=0).mean(axis=0)

        merged[key] = merged_delta.astype(deltas[0].dtype)

    if total_keys:
        conflict_rate = sign_conflicts / total_keys
        print(
            f"  Sign conflict rate: {conflict_rate:.3f} "
            f"({sign_conflicts}/{total_keys})"
        )

    out_file = output_path / "adapter_model.safetensors"
    save_file(merged, str(out_file))
    print(f"\n  Merged adapter saved to: {out_file}")
    print(f"  Load with: squish run {args.base_model}")


# ── squish rotate ──────────────────────────────────────────────────────────────

def cmd_rotate(args):  # pragma: no cover
    """
    Run SpinQuant Cayley-SGD rotation calibration on a model.

    Applies an orthogonal rotation to model weights that improves INT8
    quantization quality (reduces per-channel variance → better quant grid).
    After rotation the model is functionally identical but quantizes ~1.5–3×
    more accurately than the unrotated version.

    Under the hood this calls :mod:`squish.quant.spin_quant.run_rotation`.
    """
    try:
        from squish.quant.spin_quant import run_rotation
    except ImportError as exc:
        print(f"\n  Error: could not import squish.quant.spin_quant — {exc}")
        print("  Make sure the squish package is installed.")
        sys.exit(1)

    model_dir = _resolve_model(args.model)
    output_dir = args.output_dir or str(Path(model_dir).parent / (Path(model_dir).name + "-rotated"))

    print("\n  SpinQuant rotation calibration")
    print(f"  Model      : {model_dir}")
    print(f"  Output     : {output_dir}")
    print(f"  Steps      : {args.steps}")
    print(f"  LR         : {args.lr}")
    print(f"  Seed       : {args.seed}")
    print()

    run_rotation(
        model_dir  = model_dir,
        output_dir = output_dir,
        steps      = args.steps,
        lr         = args.lr,
        seed       = args.seed,
    )
    print(f"\n  Rotated model saved to: {output_dir}")
    print(f"  Load with: squish run {output_dir}")


# ── squish predict ─────────────────────────────────────────────────────────────

def cmd_predict(args):  # pragma: no cover
    """
    Run the LIFE analytical performance predictor.

    Prints predicted TTFT, TPOT, and throughput for the given model and
    hardware configuration, derived from the LIFE analytical model
    (memory-bandwidth, compute, and overhead terms).

    Under the hood this calls :mod:`squish.life_model.predict`.
    """
    try:
        from squish.life_model import predict as _life_predict
    except ImportError as exc:
        print(f"\n  Error: could not import squish.life_model — {exc}")
        print("  Make sure the squish package is installed.")
        sys.exit(1)

    model_arg  = args.model or ""
    model_dir  = _resolve_model(model_arg) if model_arg else None

    result = _life_predict(
        model_dir  = model_dir,
        batch_size = args.batch_size,
        seq_len    = args.seq_len,
        output_len = args.output_len,
    )

    if args.json_output:
        print(json.dumps(result, indent=2))
        return

    w = 28
    print()
    print(f"  {'LIFE Performance Prediction':^{w}}")
    print(f"  {'─' * w}")
    if model_dir:
        print(f"  {'Model':<16}: {Path(model_dir).name}")
    print(f"  {'Batch size':<16}: {args.batch_size}")
    print(f"  {'Seq len (input)':<16}: {args.seq_len}")
    print(f"  {'Output len':<16}: {args.output_len}")
    print(f"  {'─' * w}")
    print(f"  {'TTFT (prefill)':<16}: {result.get('ttft_ms', 0):.1f} ms")
    print(f"  {'TPOT (per tok)':<16}: {result.get('tpot_ms', 0):.2f} ms")
    print(f"  {'Throughput':<16}: {result.get('tokens_per_sec', 0):.1f} tok/s")
    print(f"  {'Memory (KV)':<16}: {result.get('kv_memory_gb', 0):.2f} GB")
    print()
    if result.get("bottleneck"):
        print(f"  Bottleneck: {result['bottleneck']}")
        print()


def cmd_trace(args):
    """
    View span traces and the slow-module bottleneck report.

    Actions:
      view  (default) — print top-20 slowest spans + remediation report
      reset            — clear all accumulated spans (DELETE /v1/trace)
      obs              — print APM /v1/obs-report (p99 bottlenecks + hints)

    Flags:
      --chrome PATH   Save a Chrome-DevTools trace JSON to PATH (use with view)
    """
    import urllib.error
    import urllib.request

    host = getattr(args, "host", "127.0.0.1")
    port = getattr(args, "port", 11435)
    base = f"http://{host}:{port}"
    action = getattr(args, "trace_action", "view") or "view"

    def _get(path: str) -> dict:
        url = base + path
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            return json.loads(resp.read())

    def _delete(path: str) -> dict:
        url = base + path
        req = urllib.request.Request(url, method="DELETE")
        with urllib.request.urlopen(req, timeout=5) as resp:
            return json.loads(resp.read())

    def _server_err(e):
        print(f"\n  Could not connect to squish server at {base}: {e}")
        print("  Start the server first:  squish run <model>")
        print()

    if action == "reset":
        try:
            result = _delete("/v1/trace")
            print(f"  Trace cleared: {result}")
        except (urllib.error.URLError, OSError) as e:
            _server_err(e)
        return

    chrome_path = getattr(args, "chrome", None)
    if action == "view" and chrome_path:
        try:
            import urllib.request as _ur
            url = f"{base}/v1/trace?format=chrome"
            with _ur.urlopen(url, timeout=5) as resp:
                raw = resp.read()
            Path(chrome_path).write_bytes(raw)
            print(f"  Chrome trace written to {chrome_path}")
            print(f"  Open at https://speedscope.app  or  chrome://tracing")
        except (urllib.error.URLError, OSError) as e:
            _server_err(e)
        return

    if action == "obs":
        try:
            data = _get("/v1/obs-report")
        except (urllib.error.URLError, OSError) as e:
            _server_err(e)
            return
        status = data.get("status", "unknown")
        symbol = "✓" if status == "ok" else "⚑"
        print(f"\n  {_C.P}Squish Observability Report{_C.R}  status={_C.G if status == 'ok' else _C.MG}{symbol} {status}{_C.R}")
        print()
        bottlenecks = data.get("bottlenecks", [])
        if not bottlenecks:
            print(f"  {_C.G}No bottlenecks detected (all p99 < threshold){_C.R}")
        else:
            print(f"  {_C.MG}Bottlenecks (p99 above threshold):{_C.R}")
            for b in bottlenecks:
                print(f"    {_C.T}{b['op']:<35}{_C.R} p99={_C.MG}{b['p99_ms']:.1f} ms{_C.R}  n={b['n_samples']}")
                if b.get("hint"):
                    print(f"      {_C.DIM}→ {b['hint']}{_C.R}")
        profile = data.get("profile", {})
        if profile:
            print()
            print(f"  {_C.P}APM Profile:{_C.R}")
            print(f"  {'Operation':<35}  {'n':>6}  {'mean':>8}  {'p50':>8}  {'p99':>8}")
            print(f"  {'─' * 35}  {'─' * 6}  {'─' * 8}  {'─' * 8}  {'─' * 8}")
            for op, s in sorted(profile.items()):
                print(f"  {op:<35}  {s['n_samples']:>6}  {s['mean_ms']:>7.1f}ms"
                      f"  {s['p50_ms']:>7.1f}ms  {_C.MG if s['p99_ms'] >= 200 else ''}{s['p99_ms']:>7.1f}ms{_C.R}")
        print()
        return

    # Default: view traces
    try:
        data = _get("/v1/trace")
    except (urllib.error.URLError, OSError) as e:
        _server_err(e)
        return

    enabled = data.get("tracing_enabled", False)
    total   = data.get("total_spans", 0)
    spans   = data.get("slowest_spans", [])

    trace_status = "ON" if enabled else f"{_C.MG}OFF{_C.R} — start server with --trace to collect spans"
    print(f"\n  {_C.P}Squish Span Trace{_C.R}  tracing={trace_status}  total={total}")
    if not enabled:
        print(f"  {_C.DIM}hint: {data.get('hint', '')}{_C.R}")
    print()

    if spans:
        print(f"  {'Span Name':<40}  {'Duration':>10}  {'Status':<10}")
        print(f"  {'─' * 40}  {'─' * 10}  {'─' * 10}")
        for s in spans:
            dur_ms = s.get("duration_ms", 0.0)
            name   = s.get("name", "")[:40]
            status = s.get("status", "ok")
            if dur_ms < 50:
                col = _C.G
            elif dur_ms < 500:
                col = _C.MG  # yellow-ish (magenta in ANSI mode)
            else:
                col = _C.PK  # red/pink
            print(f"  {name:<40}  {col}{dur_ms:>8.1f}ms{_C.R}  {status:<10}")
    else:
        print(f"  No spans collected yet.")

    # Remediation report from obs-report
    try:
        obs = _get("/v1/obs-report")
        bottlenecks = obs.get("bottlenecks", [])
        if bottlenecks:
            print(f"\n  {_C.MG}Remediation Report:{_C.R}")
            for b in bottlenecks:
                if b.get("hint"):
                    print(f"    {_C.T}{b['op']}{_C.R}: {b['hint']}")
    except Exception:
        pass  # obs-report is best-effort

    print()


def cmd_config(args):
    """Read or write Squish user configuration."""
    from squish import config as _cfg
    from squish.ui import console, success, warn, error as _err, _RICH_AVAILABLE

    action = getattr(args, "config_action", "show") or "show"
    key = getattr(args, "config_key", None) or ""
    value = getattr(args, "config_value", None)

    if action == "show":
        cfg = _cfg.load()
        import json as _json
        cfg_json = _json.dumps(cfg, indent=2)
        if _RICH_AVAILABLE:
            console.print()
            console.print("  [rgb(167,139,250) bold]Squish Configuration[/]  "
                          f"[rgb(100,116,139)]({_cfg.config_path()})[/]")
            console.print()
            from rich.syntax import Syntax
            console.print(Syntax(cfg_json, "json", theme="nord", background_color="default"))
            console.print()
        else:
            print()
            print(f"  Squish Configuration  ({_cfg.config_path()})")
            print()
            print(cfg_json)
            print()

    elif action == "get":
        if not key:
            _err("Usage: squish config get KEY")
            sys.exit(1)
        val = _cfg.get(key)
        if val is None:
            warn(f"Key not set: {key!r}")
            sys.exit(1)
        print(val)

    elif action == "set":
        if not key or value is None:
            _err("Usage: squish config set KEY VALUE")
            sys.exit(1)
        # coerce value to bool/int if possible
        coerced: object
        if value.lower() in ("true", "yes", "1"):
            coerced = True
        elif value.lower() in ("false", "no", "0"):
            coerced = False
        else:
            try:
                coerced = int(value)
            except ValueError:
                try:
                    coerced = float(value)
                except ValueError:
                    coerced = value
        _cfg.set(key, coerced)
        success(f"Set {key!r} = {coerced!r}")

    else:  # shouldn't reach here due to argparse choices, but be safe
        _err(f"Unknown config action: {action!r}")
        sys.exit(1)


def cmd_welcome():
    """Called when `squish` is invoked with no subcommand."""
    from squish.ui import banner, console, hint, _RICH_AVAILABLE
    from squish import config as _cfg

    banner()

    # How much unified memory?
    ram_gb = 0
    try:
        import subprocess as _sp
        _proc = _sp.run(
            ["system_profiler", "SPHardwareDataType"],
            capture_output=True, text=True, timeout=3,
        )
        for _line in _proc.stdout.splitlines():
            if "Memory:" in _line:
                _parts = _line.split()
                for _i, _p in enumerate(_parts):
                    if _p.isdigit():
                        _gb = int(_p)
                        _unit = _parts[_i + 1].upper() if _i + 1 < len(_parts) else "GB"
                        ram_gb = _gb if "GB" in _unit else _gb // 1024
                        break
                break
    except Exception:  # noqa: BLE001
        pass

    # Choose a sensible default recommendation
    if ram_gb >= 64:
        suggestion = "qwen3:32b"
    elif ram_gb >= 32:
        suggestion = "qwen3:14b"
    elif ram_gb >= 16:
        suggestion = "qwen3:8b"
    else:
        suggestion = "qwen3:4b"

    saved_default = _cfg.get("default_model")
    model = saved_default or suggestion

    # Check if there are any local models already
    local_count = 0
    if _MODELS_DIR.exists():
        local_count = sum(
            1 for d in _MODELS_DIR.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        )

    if _RICH_AVAILABLE:
        console.print()
        if local_count > 0:
            console.print(f"  [rgb(100,116,139)]You have[/] "
                          f"[rgb(248,250,252)]{local_count}[/] "
                          f"[rgb(100,116,139)]model(s) downloaded.[/]  "
                          f"[rgb(100,116,139)]Start with:[/]")
            console.print()
            console.print(f"    [rgb(167,139,250) bold]squish run {model}[/]")
        else:
            console.print(f"  [rgb(100,116,139)]Detected[/] "
                          f"[rgb(248,250,252)]{ram_gb or '?'} GB[/] "
                          f"[rgb(100,116,139)]memory.  Get started with:[/]")
            console.print()
            console.print(f"    [rgb(167,139,246) bold]squish pull {model}[/]")
            console.print(f"    [rgb(167,139,246) bold]squish run  {model}[/]")
        console.print()
        console.print("  [rgb(100,116,139)]Other commands :[/]  "
                      "[rgb(248,250,252)]squish catalog[/]  "
                      "[rgb(100,116,139)]·[/]  "
                      "[rgb(248,250,252)]squish models[/]  "
                      "[rgb(100,116,139)]·[/]  "
                      "[rgb(248,250,252)]squish --help[/]")
        console.print()
    else:
        print()
        if local_count > 0:
            print(f"  You have {local_count} model(s). Start with:")
            print(f"    squish run {model}")
        else:
            print(f"  Detected {ram_gb or '?'} GB memory. Get started:")
            print(f"    squish pull {model}")
            print(f"    squish run  {model}")
        print()
        print("  Other commands: squish catalog  ·  squish models  ·  squish --help")
        print()


def build_parser() -> "argparse.ArgumentParser":
    """Build and return the squish argument parser.

    Separated from :func:`main` so tests can introspect subcommands without
    calling ``sys.exit`` or executing any side-effects.
    """
    ap = argparse.ArgumentParser(
        prog="squish",
        description="Squish — private local inference for Apple Silicon",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  squish catalog                     Browse available models
  squish catalog --tag reasoning     Filter by tag (reasoning, small, large, moe…)
  squish pull qwen3:8b               Download + compress Qwen3-8B (INT4 by default)
  squish pull gemma3:4b --int8       Download with INT8 compression (larger, marginally higher accuracy)
  squish run qwen3:8b                Start server on :11435
  squish run 7b --batch-scheduler    Legacy shorthand, with continuous batching
  squish chat qwen3:8b               Interactive terminal chat
  squish chat                        Chat against already-running server
  squish models                      List local models
  squish info                        System info + server status
  squish doctor                      Check all dependencies
  squish daemon start qwen3:8b       Start persistent background server
  squish daemon status               Check daemon status
  squish daemon stop                 Stop daemon
  squish bench                       Quick throughput benchmark
  squish it qwen3:8b                 Compress a local model to INT8 npy-dir format

Model IDs (sample):
  qwen3:8b   gemma3:4b   deepseek-r1:7b   llama3.2:3b   phi4:14b
  Legacy: 7b  14b  1.5b  32b  72b

OpenAI drop-in (after squish run):
  export OPENAI_BASE_URL=http://localhost:11435/v1
  export OPENAI_API_KEY=squish
  # Any openai-compatible client/agent framework now works locally

Ollama drop-in:
  export OLLAMA_HOST=http://localhost:11435
  ollama list    # or use any Ollama-compatible tool
""",
    )
    sub = ap.add_subparsers(dest="command")

    ap.add_argument(
        "--version", action="version",
        version="squish 9.0.0",
        help="Show squish version and exit",
    )

    # ── run ──
    # Docker env-var defaults: SQUISH_MODEL, SQUISH_HOST, SQUISH_PORT let
    # `docker run -e SQUISH_MODEL=/models/...` work without extra CMD args.
    _env_model    = os.environ.get("SQUISH_MODEL") or None
    _env_host     = os.environ.get("SQUISH_HOST") or None
    _env_port_raw = os.environ.get("SQUISH_PORT", "")
    _env_port: int | None = int(_env_port_raw) if _env_port_raw.strip().isdigit() else None

    p_run = sub.add_parser("run", help="Start the inference server")
    p_run.add_argument("model", nargs="?", default=_env_model,
                       help="Model: 7b, 14b, 1.5b, or path (or SQUISH_MODEL env var)")
    p_run.add_argument("--port",    type=int, default=_env_port or _DEFAULT_PORT)
    p_run.add_argument("--host",    default=_env_host or "127.0.0.1",
                       help="0.0.0.0 to expose on LAN (or SQUISH_HOST env var)")
    p_run.add_argument("--api-key", default="squish")
    p_run.add_argument("--draft-model",      default="",
                       help="Path to draft model for speculative decoding")
    p_run.add_argument("--batch-scheduler",  action="store_true",
                       help="Enable continuous batching (improves concurrent throughput)")
    p_run.add_argument("--batch-size",       type=int, default=8)
    p_run.add_argument("--kv-cache-mode",    choices=["fp16", "int8", "snap"], default="fp16")
    p_run.add_argument("--log-level",
                       choices=["critical", "error", "warning", "info", "debug", "trace"],
                       default="warning",
                       help="Server log verbosity (default: warning)")
    p_run.add_argument("--trace-output", default="",
                       metavar="FILE",
                       help="Save a Chrome DevTools flame-graph JSON to FILE on server exit. "
                            "Open at https://speedscope.app or chrome://tracing to see every "
                            "module with start/end timing. Implies --trace.")
    p_run.add_argument("--all-optimizations", action="store_true", default=False,
                       help="Enable ALL built-in optimization modules at once "
                            "(enabled by default; this flag is a no-op unless --stock was also set)")
    p_run.add_argument("--stock", action="store_true", default=False,
                       help="Stock mode: disable ALL squish optimizations and agent preset. "
                            "Runs a plain mlx_lm baseline comparable to Ollama or unoptimized "
                            "inference. Use this to benchmark against stock tools.")
    p_run.add_argument("--no-agent", action="store_true", default=False,
                       help="Disable the agent preset (kept for backwards compatibility; "
                            "agent is now opt-in via --agent, so this flag is a no-op).")
    p_run.add_argument("--compressed-dir", default="", metavar="DIR",
                       help="Explicit path to squished/compressed weights dir. "
                            "Overrides auto-detected compressed dir alongside MODEL.")
    p_run.add_argument("--int4", action="store_true", default=False,
                       help="Use INT4 compressed weights (default)")
    p_run.add_argument("--int3", action="store_true", default=False,
                       help="Use INT3 compressed weights")
    p_run.add_argument("--int2", action="store_true", default=False,
                       help="Use INT2 compressed weights")
    p_run.add_argument("--int8", action="store_true", default=False,
                       help="Use INT8 compressed weights")
    # ── Phase 13D: Agent preset ──
    p_run.add_argument("--agent", action="store_true", default=False,
                       help="Agent-mode preset: enables --agent-kv, --grammar, "
                            "--chunked-prefill, --radix-cache, sets batch-size=1, "
                            "and auto-sizes context from available UMA memory.\n"
                            "Designed for 16 GB M-series running 7–14 B models in "
                            "long agent loops.")
    # ── Phase 14: MoE Expert Lookahead ──
    p_run.add_argument("--moe-lookahead", action="store_true", default=False,
                       help="[Experimental] Enable MoE expert lookahead router. "
                            "Automatically set when --agent is active and the model "
                            "is a MoE catalog entry (e.g. DeepSeek-Coder-V2-Lite).")
    # ── WhatsApp / Meta Cloud API ─────────────────────────────────────────────
    p_run.add_argument("--whatsapp", action="store_true", default=False,
                       help="Enable WhatsApp webhook at /webhook/whatsapp (Meta Cloud API).")
    p_run.add_argument("--whatsapp-verify-token", default="",
                       help="Webhook verify token set in Meta Developer Dashboard "
                            "(or WHATSAPP_VERIFY_TOKEN env var).")
    p_run.add_argument("--whatsapp-app-secret", default="",
                       help="Meta App Secret for signature validation "
                            "(or WHATSAPP_APP_SECRET env var).")
    p_run.add_argument("--whatsapp-access-token", default="",
                       help="Meta access token for sending replies "
                            "(or WHATSAPP_ACCESS_TOKEN env var).")
    p_run.add_argument("--whatsapp-phone-number-id", default="",
                       help="Meta Phone Number ID "
                            "(or WHATSAPP_PHONE_NUMBER_ID env var).")
    p_run.add_argument("--system-prompt", default="",
                       help="Custom system prompt for WhatsApp/Signal bot sessions.")
    # ── Signal / signal-cli ───────────────────────────────────────────────────
    p_run.add_argument("--signal", action="store_true", default=False,
                       help="Enable Signal bot (requires signal-cli daemon).")
    p_run.add_argument("--signal-account", default="",
                       help="E.164 number registered in signal-cli (or SIGNAL_ACCOUNT env var).")
    p_run.add_argument("--signal-socket", default="127.0.0.1:7583",
                       help="signal-cli daemon address: host:port or UNIX socket path.")
    p_run.add_argument("--thinking-budget", type=int, default=-1, metavar="N",
                       help="Qwen3 chain-of-thought budget: -1=unlimited (default), "
                            "0=disable thinking (/no_think mode, fastest), "
                            ">0=cap reasoning at N tokens.")
    p_run.add_argument("--no-browser", action="store_true", default=False,
                       help="Do not auto-open the Squish Agent chat UI in a browser after startup.")
    p_run.set_defaults(func=cmd_run)

    # ── serve (alias for run) ──
    p_serve = sub.add_parser("serve", help="Start the inference server (alias for 'run')")
    p_serve.add_argument("model", nargs="?", default=_env_model,
                         help="Model: qwen3:8b, 7b, 14b, or path (or SQUISH_MODEL env var)")
    p_serve.add_argument("--port",    type=int, default=_env_port or _DEFAULT_PORT)
    p_serve.add_argument("--host",    default=_env_host or "127.0.0.1",
                         help="0.0.0.0 to expose on LAN (or SQUISH_HOST env var)")
    p_serve.add_argument("--api-key", default="squish")
    p_serve.add_argument("--draft-model",      default="")
    p_serve.add_argument("--batch-scheduler",  action="store_true")
    p_serve.add_argument("--batch-size",       type=int, default=8)
    p_serve.add_argument("--kv-cache-mode",    choices=["fp16", "int8", "snap"], default="fp16")
    p_serve.add_argument("--log-level",
                         choices=["critical", "error", "warning", "info", "debug", "trace"],
                         default="warning",
                         help="Server log verbosity (default: warning)")
    p_serve.add_argument("--all-optimizations", action="store_true", default=False,
                         help="Enable ALL built-in optimization modules at once "
                              "(enabled by default; this flag is a no-op unless --stock was also set)")
    p_serve.add_argument("--stock", action="store_true", default=False,
                         help="Stock mode: disable ALL squish optimizations and agent preset. "
                              "Runs a plain mlx_lm baseline comparable to Ollama or unoptimized "
                              "inference. Use this to benchmark against stock tools.")
    p_serve.add_argument("--no-agent", action="store_true", default=False,
                         help="Disable the agent preset (kept for backwards compatibility; "
                              "agent is now opt-in via --agent, so this flag is a no-op).")
    p_serve.add_argument("--compressed-dir", default="", metavar="DIR",
                         help="Explicit path to squished/compressed weights dir.")
    p_serve.add_argument("--int4", action="store_true", default=False,
                         help="Use INT4 compressed weights (default)")
    p_serve.add_argument("--int3", action="store_true", default=False,
                         help="Use INT3 compressed weights")
    p_serve.add_argument("--int2", action="store_true", default=False,
                         help="Use INT2 compressed weights")
    p_serve.add_argument("--int8", action="store_true", default=False,
                         help="Use INT8 compressed weights")
    # ── Phase 13D: Agent preset ──
    p_serve.add_argument("--agent", action="store_true", default=False,
                         help="Agent-mode preset: enables --agent-kv, --grammar, "
                              "--chunked-prefill, --radix-cache, sets batch-size=1, "
                              "and auto-sizes context from available UMA memory.\n"
                              "Designed for 16 GB M-series running 7–14 B models in "
                              "long agent loops.")
    # ── Phase 14: MoE Expert Lookahead ──
    p_serve.add_argument("--moe-lookahead", action="store_true", default=False,
                         help="[Experimental] Enable MoE expert lookahead router. "
                              "Automatically set when --agent is active and the model "
                              "is a MoE catalog entry.")
    # ── WhatsApp / Meta Cloud API ─────────────────────────────────────────────
    p_serve.add_argument("--whatsapp", action="store_true", default=False,
                         help="Enable WhatsApp webhook at /webhook/whatsapp (Meta Cloud API).")
    p_serve.add_argument("--whatsapp-verify-token", default="",
                         help="Webhook verify token (or WHATSAPP_VERIFY_TOKEN env var).")
    p_serve.add_argument("--whatsapp-app-secret", default="",
                         help="Meta App Secret for signature validation "
                              "(or WHATSAPP_APP_SECRET env var).")
    p_serve.add_argument("--whatsapp-access-token", default="",
                         help="Meta access token for sending replies "
                              "(or WHATSAPP_ACCESS_TOKEN env var).")
    p_serve.add_argument("--whatsapp-phone-number-id", default="",
                         help="Meta Phone Number ID (or WHATSAPP_PHONE_NUMBER_ID env var).")
    p_serve.add_argument("--system-prompt", default="",
                         help="Custom system prompt for WhatsApp/Signal bot sessions.")
    # ── Signal / signal-cli ───────────────────────────────────────────────────
    p_serve.add_argument("--signal", action="store_true", default=False,
                         help="Enable Signal bot (requires signal-cli daemon).")
    p_serve.add_argument("--signal-account", default="",
                         help="E.164 number registered in signal-cli (or SIGNAL_ACCOUNT env var).")
    p_serve.add_argument("--signal-socket", default="127.0.0.1:7583",
                         help="signal-cli daemon address: host:port or UNIX socket path.")
    p_serve.add_argument("--thinking-budget", type=int, default=-1, metavar="N",
                         help="Qwen3 chain-of-thought budget: -1=unlimited (default), "
                              "0=disable thinking (/no_think mode, fastest), "
                              ">0=cap reasoning at N tokens.")
    p_serve.add_argument("--no-browser", action="store_true", default=False,
                         help="Do not auto-open the Squish Agent chat UI in a browser after startup.")
    p_serve.set_defaults(func=cmd_run)

    # ── chat ──
    p_chat = sub.add_parser("chat", help="Interactive terminal chat")
    p_chat.add_argument("model", nargs="?", help="Model shorthand or path (auto-starts server if needed)")
    p_chat.add_argument("--port",        type=int, default=_DEFAULT_PORT)
    p_chat.add_argument("--host",        default="127.0.0.1")
    p_chat.add_argument("--api-key",     default="squish")
    p_chat.add_argument("--chat-model",  default="squish",
                        help="Model ID to send in requests (default: squish)")
    p_chat.add_argument("--system",      default="",
                        help="System prompt (default: private local assistant)")
    p_chat.add_argument("--max-tokens",  type=int, default=1024)
    p_chat.add_argument("--temperature", type=float, default=0.7)
    p_chat.add_argument("--max-history", type=int, default=40,
                        help="Keep at most this many non-system messages in context "
                             "(prevents unbounded token growth; default 40)")
    p_chat.set_defaults(func=cmd_chat)

    # ── models ──
    p_models = sub.add_parser("models", help="List local models")
    p_models.set_defaults(func=cmd_models)

    # ── squish compat ──────────────────────────────────────────────────────────
    p_compat = sub.add_parser(
        "compat",
        help="Print client configuration snippets for popular AI tools",
    )
    p_compat.add_argument("--host", default="localhost",
                          help="Server host (default: localhost)")
    p_compat.add_argument("--port", type=int, default=11435,
                          help="Server port (default: 11435)")
    p_compat.set_defaults(func=cmd_compat)

    # ── setup ──
    p_setup = sub.add_parser(
        "setup",
        help="Interactive setup wizard: detect hardware, recommend + pull model, start server",
    )
    p_setup.set_defaults(func=cmd_setup)

    # ── info ──
    p_info = sub.add_parser("info", help="System info")
    p_info.set_defaults(func=cmd_info)

    # ── bench ──
    p_bench = sub.add_parser("bench", help="Quick throughput benchmark")
    p_bench.add_argument("--port",       type=int, default=_DEFAULT_PORT)
    p_bench.add_argument("--host",       default="127.0.0.1")
    p_bench.add_argument("--api-key",    default="squish")
    p_bench.add_argument("--max-tokens", type=int, default=128)
    p_bench.add_argument("--markdown",   action="store_true",
                         help="Print a markdown table after benchmarking")
    p_bench.add_argument("--save",       default="", metavar="FILE",
                         help="Save markdown table to FILE (implies --markdown; "
                              "default: squish_bench.md)")
    p_bench.add_argument("--cold-start",  action="store_true",
                         help="Time model load (first-request latency) before the benchmark prompts")
    p_bench.add_argument("--track",
                         choices=["quality", "code", "tool", "agent", "perf"],
                         metavar="TRACK",
                         help="Run a benchmark track: quality|code|tool|agent|perf")
    p_bench.add_argument("--engines",    default="squish",
                         help="Comma-separated engine specs for bench tracks (default: squish)")
    p_bench.add_argument("--model",      default="squish",
                         help="Model name to benchmark (default: squish)")
    p_bench.add_argument("--compare",    action="store_true",
                         help="After running, generate a comparison table from eval_output/")
    p_bench.add_argument("--report",     action="store_true",
                         help="After running, generate a full benchmark report to docs/")
    p_bench.add_argument("--limit",      type=int, default=None,
                         help="Limit items/tokens per track (passed to runner.run)")
    p_bench.add_argument("--out",        default="eval_output",
                         help="Directory to save ResultRecord JSON files (default: eval_output)")
    p_bench.set_defaults(func=cmd_bench)

    # ── doctor ──
    p_doctor = sub.add_parser("doctor", help="Check all dependencies and system requirements")
    p_doctor.add_argument(
        "--report", action="store_true", default=False,
        help="Write a shareable JSON diagnostic report to ~/.squish/doctor-report-<ts>.json",
    )
    p_doctor.set_defaults(func=cmd_doctor)

    # ── update ──
    p_update = sub.add_parser("update", help="Upgrade squish and core dependencies to latest")
    p_update.add_argument(
        "--all", action="store_true", default=False,
        help="Include optional/heavy dependencies (mlx-vlm, transformers, etc.)",
    )
    p_update.set_defaults(func=cmd_update)

    # ── daemon ──
    p_daemon = sub.add_parser("daemon", help="Manage the persistent background server daemon")
    p_daemon.add_argument("daemon_action", nargs="?",
                          choices=["start", "stop", "status"],
                          default="status",
                          help="start | stop | status (default: status)")
    p_daemon.add_argument("model", nargs="?", help="Model shorthand or path (for start)")
    p_daemon.add_argument("--port",    type=int, default=_DEFAULT_PORT)
    p_daemon.add_argument("--host",    default="127.0.0.1")
    p_daemon.add_argument("--api-key", default="squish")
    p_daemon.set_defaults(func=cmd_daemon)

    # ── compress (primary name) + it (hidden legacy alias) ──
    p_compress = sub.add_parser(
        "compress",
        aliases=["it"],
        help="Compress a model to npy-dir format (INT8 or INT4)",
    )
    p_compress.add_argument("model", help="Model path (e.g. ~/.squish/models/llama3.1-8b-4bit) or shorthand (7b, 14b)")
    p_compress.add_argument("--output",            default=None,
                            help="Output directory (default: <model>-compressed)")
    p_compress.add_argument("--passthrough",       nargs="*", default=[], metavar="PATTERN",
                            help="Tensor substrings to keep as float32 (e.g. embed lm_head)")
    p_compress.add_argument("--outlier-threshold", type=float, default=20.0)
    p_compress.add_argument("--int4",    action="store_true",
                            help="INT4 nibble-packed (~44%% disk savings vs INT8). "
                                 "Requires squish_quant Rust ext. "
                                 "⚠ Not recommended for models < 3B — use INT8 for best quality.")
    p_compress.add_argument("--aqlm",    action="store_true", default=False,
                            help="AQLM additive codebook quantization (ICML 2024). "
                                 "~2-bit effective precision via additive codebook lookup. "
                                 "Pure numpy, no GPU required.")
    p_compress.add_argument("--aqlm-codebooks", type=int, default=2, metavar="M",
                            help="Number of additive codebooks for AQLM (default: 2).")
    p_compress.add_argument("--aqlm-cbsize", type=int, default=16, metavar="K",
                            help="Number of codewords per AQLM codebook (default: 16).")
    p_compress.add_argument("--zstd-level", type=int, default=0, metavar="N",
                            help="Apply zstd entropy compression at level N (1-22) after "
                                 "quantization.  Level 3 is a good default; 0 = skip (default). "
                                 "Requires: pip install zstandard")
    p_compress.add_argument("--no-awq", action="store_true", default=False,
                            help="Disable AWQ calibration (auto-enabled when --int4 is used). "
                                 "Use this to skip the ~2–5 min calibration step.")
    p_compress.add_argument("--awq", action="store_true", default=False,
                            help="Force AWQ calibration even without --int4 (INT8). "
                                 "When --int4 is used AWQ runs automatically unless --no-awq is passed.")
    p_compress.add_argument("--awq-samples", type=int, default=20, metavar="N",
                            help="Number of calibration samples for AWQ (default: 20)")
    p_compress.add_argument("--verbose",           action="store_true")
    p_compress.add_argument(
        "--int4-group-size",
        type=int,
        default=None,
        dest="int4_group_size",
        metavar="N",
        help="Override per-group size for INT4 quantization (power of two ≤ 32 "
             "that divides the weight matrix column count). Default: auto-select 32. "
             "Use 16 for finer-grained scales at ~2× scale storage overhead.",
    )
    p_compress.add_argument(
        "--format",
        choices=["int8", "int4", "astc", "hybrid"],
        default=None,
        dest="compress_format",
        metavar="FORMAT",
        help=(
            "Output compression format. Choices: int4 (default, 4-bit group quantisation), "
            "int8 (8-bit group quantisation), "
            "astc (ASTC 6×6 HDR texture ~3.56 BPW, Apple Silicon only — "
            "writes .squizd format), "
            "hybrid (ASTC for FFN layers + INT4 for attention, Apple Silicon only — "
            "writes .squizd format). "
            "ASTC formats fall back to INT4 on non-Apple or non-ASTC hardware."
        ),
    )
    p_compress.set_defaults(func=cmd_compress)

    # ── pull ──
    p_pull = sub.add_parser(
        "pull",
        help="Download + compress a model from the Squish catalog",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Download a model and compress it with Squish.\n\n"
            "Pre-compressed weights are downloaded directly when available.\n"
            "Otherwise the bf16 MLX variant is fetched and compressed locally.\n\n"
            "Examples:\n"
            "  squish pull qwen3:8b\n"
            "  squish pull deepseek-r1:14b --token hf_…"
        ),
    )
    p_pull.add_argument("model", help="Model ID (e.g. qwen3:8b, gemma3:4b, 7b)")
    p_pull.add_argument("--int8", action="store_true",
                        help="Use INT8 group-64 compression instead of INT4 (default is INT4).")
    p_pull.add_argument("--int4", action="store_true",
                        help="Use INT4 nibble-packed compression (default; flag kept for backward compatibility).")
    p_pull.add_argument("--int3", action="store_true",
                        help="Use INT3-MiLo (3-bit + low-rank compensator, ~4.4 bpw). "
                             "Best quality-vs-size for M3/M4 16GB Macs.")
    p_pull.add_argument("--int2", action="store_true",
                        help="Use INT2-WOQ (2-bit pack-4 asymmetric, ~3 bpw). "
                             "Smallest footprint; fits 14B on 8GB RAM.")
    p_pull.add_argument("--token", default="",
                        help="HuggingFace access token (or set $HF_TOKEN)")
    p_pull.add_argument("--models-dir", default="",
                        help=f"Override models directory (default: {_MODELS_DIR})")
    p_pull.add_argument("--refresh-catalog", action="store_true",
                        help="Force-refresh the online catalog before resolving")
    p_pull.add_argument(
        "--with-draft",
        action="store_true",
        default=False,
        help=(
            "Also download a pre-distilled EAGLE-3 draft head from "
            "squish-community/eagle-heads on HuggingFace after pulling the "
            "model weights.  Skipped if the head file already exists locally. "
            "Equivalent to running `squish pull-head <model>` separately."
        ),
    )
    p_pull.add_argument("--verbose", action="store_true")
    p_pull.set_defaults(func=cmd_pull)

    # ── import ──
    p_import = sub.add_parser(
        "import",
        help="Import a model from Ollama, a GGUF file, or HuggingFace",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Import a model from an external source into the Squish model store.\n\n"
            "Examples:\n"
            "  squish import ollama:qwen3:8b\n"
            "  squish import /path/to/model.gguf\n"
            "  squish import hf:mlx-community/Qwen3-8B-bf16"
        ),
    )
    p_import.add_argument("import_source",
                          help="Source: ollama:<name>, /path/to/model.gguf, or hf:<repo>")
    p_import.add_argument("--models-dir", default="",
                          help="Override models directory (default: ~/.squish/models)")
    p_import.add_argument("--token", default="",
                          help="HuggingFace API token (or set HF_TOKEN env var)")
    p_import.set_defaults(func=cmd_import)

    # ── pull-head (EAGLE-3) ──
    p_head = sub.add_parser(
        "pull-head",
        help="Download an EAGLE-3 draft head for speculative decoding",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Download an EAGLE-3 draft head and convert it to MLX format.\n\n"
            "An EAGLE head pairs with a specific target model and typically\n"
            "achieves 75-85%% draft acceptance versus 55-65%% for a separate\n"
            "draft model, at a fraction of the memory cost.\n\n"
            "Examples:\n"
            "  squish pull-head qwen3:8b\n"
            "  squish pull-head yuhuili/EAGLE3-Qwen3-Instruct-8B --output ./my-head\n"
            "  squish pull-head qwen3:8b --token hf_…"
        ),
    )
    p_head.add_argument("model",
                        help="Model alias (e.g. qwen3:8b) or full HF repo "
                             "(e.g. yuhuili/EAGLE3-Qwen3-Instruct-8B)")
    p_head.add_argument("--output", default="",
                        help="Output directory (default: ~/.squish/eagle-heads/<slug>)")
    p_head.add_argument("--token", default="",
                        help="HuggingFace API token (or set HF_TOKEN env var)")
    p_head.set_defaults(func=cmd_pull_head)

    # ── catalog ──
    p_catalog = sub.add_parser("catalog", help="Browse available models in the Squish catalog")
    p_catalog.add_argument("--tag", default="",
                           help="Filter by tag: small, fast, balanced, large, reasoning, moe, edge")
    p_catalog.add_argument("--refresh", action="store_true",
                           help="Force-refresh the catalog from HuggingFace")
    p_catalog.set_defaults(func=cmd_catalog)

    p_rm = sub.add_parser("rm", help="Remove a local model (frees disk space)")
    p_rm.add_argument("model", help="Model ID, alias, or path (e.g. qwen3:8b, 7b, ~/models/Llama-3)")
    p_rm.add_argument("--compressed-only", action="store_true",
                      help="Remove only the compressed (-compressed) directory")
    p_rm.add_argument("--raw-only", action="store_true",
                      help="Remove only the raw weights directory")
    p_rm.add_argument("--dry-run", action="store_true",
                      help="Show what would be removed without deleting anything")
    p_rm.add_argument("-y", "--yes", action="store_true",
                      help="Skip confirmation prompt")
    p_rm.set_defaults(func=cmd_rm, compressed_only=False, raw_only=False)  # pragma: no cover

    p_search = sub.add_parser("search", help="Search the model catalog")
    p_search.add_argument("query", help="Search query (matched against ID, tags, params, description)")
    p_search.set_defaults(func=cmd_search)

    # ── check ──
    p_check = sub.add_parser(
        "check",
        help="Inspect a quantized model and report quality metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Inspect a quantized model directory and report:\n\n"
            "  • Detected quantisation bits and group_size per layer type\n"
            "  • Theoretical reconstruction quality (SNR dB via HQQ simulation)\n"
            "  • Warnings for known problematic configurations\n\n"
            "Example:\n"
            "  squish check --model ~/.squish/models/qwen3-2b-int2\n"
        ),
    )
    p_check.add_argument(
        "--model", required=True, metavar="PATH",
        help="Path to quantized model directory (mlx_lm format with config.json)"
    )
    p_check.set_defaults(func=cmd_check_model)

    # ── quantize (primary name) + convert-model (hidden legacy alias) ──
    p_convert = sub.add_parser(
        "quantize",
        aliases=["convert-model"],
        help="Mixed-precision quantize a model (different bits per layer group)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Quantize an existing model with three-tier mixed precision.\n\n"
            "  --ffn-bits   : MLP gate/up/down projections (most of the weights)\n"
            "  --attn-bits  : Q/K/V/O attention projections (keeps coherence at low bits)\n"
            "  --embed-bits : lm_head + embed_tokens\n\n"
            "For coherent INT2/INT3 generation keep attention at 4-bit:\n"
            "  squish quantize --source-path ./model-bf16 --output-path ./model-int2 \\\n"
            "    --ffn-bits 2 --attn-bits 4 --embed-bits 8 --group-size 32\n\n"
            "  squish quantize --source-path ./model-bf16 --output-path ./model-int3 \\\n"
            "    --ffn-bits 3 --attn-bits 4 --embed-bits 8 --group-size 32\n\n"
            "Legacy uniform INT4 (no per-tier override needed):\n"
            "  squish quantize --source-path ./model-bf16 --output-path ./model-int4 \\\n"
            "    --ffn-bits 4 --embed-bits 8\n"
        ),
    )
    p_convert.add_argument("--source-path", required=True, metavar="PATH",
                           help="Source model directory (HF format or mlx_lm format)")
    p_convert.add_argument("--output-path", required=True, metavar="PATH",
                           help="Output directory for mixed-precision model")
    p_convert.add_argument("--ffn-bits", type=int, default=4, metavar="N",
                           help="Quantization bits for MLP FFN layers (default: 4)")
    p_convert.add_argument("--attn-bits", type=int, default=None, metavar="N",
                           dest="attn_bits",
                           help=(
                               "Quantization bits for attention Q/K/V/O projections. "
                               "Defaults to --ffn-bits when omitted. "
                               "Use 4 when --ffn-bits is 2 or 3 to fix garbage/looping output."
                           ))
    p_convert.add_argument("--embed-bits", type=int, default=6, metavar="N",
                           help="Quantization bits for lm_head + embed_tokens (default: 6)")
    p_convert.add_argument("--group-size", type=int, default=64, metavar="N",
                           dest="group_size",
                           help=(
                               "Quantization group size for weight matrices (default: 64). "
                               "32 gives better accuracy for INT2 at ~2%% overhead."
                           ))
    p_convert.add_argument("--dry-run", action="store_true", default=False,
                           help="Print what would be done without converting")
    p_convert.add_argument("--cpu", action="store_true", default=False,
                           help="Force MLX to run on CPU (avoids Metal GPU timeout for large models)")
    p_convert.add_argument(
        "--mixed-recipe",
        default=None,
        choices=["mixed_2_6", "mixed_3_4"],
        dest="mixed_recipe",
        help=(
            "Apply a layer-aware mixed-precision recipe on top of --ffn-bits. "
            "'mixed_2_6': 2-bit base with 6-bit protection for critical down_proj/v_proj "
            "layers (~50%% of layers get 6-bit). Strongly recommended when --ffn-bits 2 "
            "to recover near-INT3 quality. "
            "'mixed_3_4': 3-bit base with 4-bit for critical layers."
        ),
    )
    p_convert.add_argument(
        "--hqq",
        action="store_true",
        default=False,
        dest="hqq",
        help=(
            "Enable Half-Quadratic Quantization (HQQ) pre-optimisation for FFN weights.\n"
            "Applies a calibration-free proximal-point solver to find near-optimal\n"
            "quantisation scales/zeros before mlx_lm.convert runs.\n"
            "Strongly recommended when --ffn-bits 2 or 3 — dramatically reduces\n"
            "the random/incoherent output seen with naive INT2 quantisation.\n"
            "Requires a local BF16 source directory (not a HuggingFace model ID).\n"
            "Adds 1-3 minutes to quantisation time depending on model size."
        ),
    )
    p_convert.set_defaults(func=cmd_convert_model, _default_group_size=64)

    # ── train (primary name) + train-adapter (hidden legacy alias) ──
    p_train = sub.add_parser(
        "train",
        aliases=["train-adapter"],
        help="Train a LoRA adapter using mlx_lm",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Fine-tune a LoRA adapter on a JSONL chat dataset using mlx_lm.\n\n"
            "After training, DARE sparsification is applied:\n"
            "  90%% of delta weights are zeroed and the remainder rescaled.\n\n"
            "Example:\n"
            "  squish train-adapter qwen3:8b \\\n"
            "    --dataset ./data/legal.jsonl \\\n"
            "    --domain legal \\\n"
            "    --rank 8 --epochs 3 \\\n"
            "    --output-dir ~/.squish/adapters/legal\n"
        ),
    )
    p_train.add_argument("model", help="Base model ID (e.g. qwen3:8b)")
    p_train.add_argument("--dataset", required=True, metavar="PATH",
                         help="JSONL dataset path with {\"messages\":[...]} records")
    p_train.add_argument("--domain", required=True, metavar="NAME",
                         help="Domain identifier for the adapter (e.g. 'legal')")
    p_train.add_argument("--rank", type=int, default=8, metavar="N",
                         help="LoRA rank (default: 8)")
    p_train.add_argument("--epochs", type=int, default=3, metavar="N",
                         help="Training epochs (default: 3)")
    p_train.add_argument("--output-dir", default="~/.squish/adapters", metavar="PATH",
                         help="Output directory for the adapter weights")
    p_train.set_defaults(func=cmd_train_adapter)

    # ── merge (primary name) + merge-model (hidden legacy alias) ──
    p_merge = sub.add_parser(
        "merge",
        aliases=["merge-model"],
        help="Merge LoRA adapters via DARE+TIES",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Merge one or more LoRA adapters into a single flat model via\n"
            "DARE sparsification and/or TIES sign-conflict resolution.\n\n"
            "Methods:\n"
            "  dare       — DARE sparsification then simple average\n"
            "  ties       — TIES sign resolution (no DARE)\n"
            "  dare-ties  — DARE then TIES (default)\n\n"
            "Example:\n"
            "  squish merge-model qwen3:8b \\\n"
            "    --adapters legal:~/.squish/adapters/legal \\\n"
            "               code:~/.squish/adapters/code \\\n"
            "    --method dare-ties \\\n"
            "    --output-path ~/.squish/models/qwen3-merged\n"
        ),
    )
    p_merge.add_argument("base_model", help="Base model ID (e.g. qwen3:8b)")
    p_merge.add_argument("--adapters", required=True, nargs="+", metavar="DOMAIN:PATH",
                         help="One or more 'domain:path' adapter specs")
    p_merge.add_argument("--method",
                         choices=["dare-ties", "dare", "ties"],
                         default="dare-ties",
                         help="Merge method (default: dare-ties)")
    p_merge.add_argument("--output-path", required=True, metavar="PATH",
                         help="Output directory for the merged adapter")
    p_merge.set_defaults(func=cmd_merge_model)

    # ── squish rotate ──────────────────────────────────────────────────────────
    p_rotate = sub.add_parser(
        "rotate",
        help="Run SpinQuant Cayley-SGD rotation calibration on a model",
    )
    p_rotate.add_argument("model", metavar="MODEL",
                          help="Model directory or shorthand (e.g. qwen3:8b)")
    p_rotate.add_argument("--output-dir", default="", metavar="DIR",
                          help="Destination directory for rotated weights.\n"
                               "Default: <model_dir>-rotated/")
    p_rotate.add_argument("--steps", type=int, default=100, metavar="N",
                          help="Number of Cayley-SGD optimisation steps (default 100).")
    p_rotate.add_argument("--lr", type=float, default=1e-4, metavar="LR",
                          help="Learning rate for the Cayley-SGD optimizer (default 1e-4).")
    p_rotate.add_argument("--seed", type=int, default=42,
                          help="Random seed for calibration (default 42).")
    p_rotate.set_defaults(func=cmd_rotate)

    # ── squish predict ─────────────────────────────────────────────────────────
    p_predict = sub.add_parser(
        "predict",
        help="Run the LIFE analytical performance predictor on a model / hardware combo",
    )
    p_predict.add_argument("model", nargs="?", default="",
                           metavar="MODEL",
                           help="Model directory or shorthand (optional; uses running "
                                "server config when omitted).")
    p_predict.add_argument("--batch-size", type=int, default=1, metavar="N",
                           help="Concurrent request count to model (default 1).")
    p_predict.add_argument("--seq-len", type=int, default=512, metavar="N",
                           help="Input sequence length for TTFT estimate (default 512).")
    p_predict.add_argument("--output-len", type=int, default=128, metavar="N",
                           help="Output token count for TPOT estimate (default 128).")
    p_predict.add_argument("--json", action="store_true", dest="json_output",
                           help="Print results as JSON instead of a human-readable table.")
    p_predict.set_defaults(func=cmd_predict)

    # ── squish trace ───────────────────────────────────────────────────────────
    p_trace = sub.add_parser(
        "trace",
        help="View span traces and slow-module bottleneck report",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "View collected span traces and APM bottleneck report from the running server.\n\n"
            "Actions:\n"
            "  view  (default) — print top-20 slowest spans + remediation hints\n"
            "  reset           — clear all accumulated spans\n"
            "  obs             — print APM /v1/obs-report (p99 bottlenecks + hints)\n\n"
            "Examples:\n"
            "  squish trace\n"
            "  squish trace obs\n"
            "  squish trace reset\n"
            "  squish trace view --chrome trace.json\n"
        ),
    )
    p_trace.add_argument(
        "trace_action",
        nargs="?",
        default="view",
        choices=["view", "reset", "obs"],
        help="Action to perform (default: view)",
    )
    p_trace.add_argument("--chrome", metavar="PATH",
                         help="Save Chrome DevTools trace JSON to PATH (use with view)")
    p_trace.add_argument("--host", default="127.0.0.1",
                         help="Server host (default: 127.0.0.1)")
    p_trace.add_argument("--port", type=int, default=11435,
                         help="Server port (default: 11435)")
    p_trace.set_defaults(func=cmd_trace)

    # ── config ──
    p_config = sub.add_parser(
        "config",
        help="Read or write user configuration (~/.squish/config.json)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Read or write Squish user configuration.\n\n"
            "Examples:\n"
            "  squish config show\n"
            "  squish config set default_model qwen3:8b\n"
            "  squish config get port\n"
            "  squish config set whatsapp.access_token EAABxxx\n"
        ),
    )
    p_config.add_argument(
        "config_action",
        nargs="?",
        choices=["show", "get", "set"],
        default="show",
        help="show | get KEY | set KEY VALUE (default: show)",
    )
    p_config.add_argument("config_key",   nargs="?", default=None,
                          help="Config key (dot-notation supported, e.g. whatsapp.access_token)")
    p_config.add_argument("config_value", nargs="?", default=None,
                          help="Value to set")
    p_config.set_defaults(func=cmd_config)

    return ap


def main():
    ap = build_parser()
    args = ap.parse_args()

    # Configure structured logging based on --log-level (or env var default)
    try:
        from squish.logging_config import configure_logging as _configure_logging
        _configure_logging(level=getattr(args, "log_level", "warning"))
    except Exception:
        pass  # never block CLI startup on logging config failure

    if not args.command:
        # No subcommand — show interactive welcome instead of raw argparse help
        cmd_welcome()
        sys.exit(0)

    args.func(args)  # pragma: no cover


if __name__ == "__main__":
    main()
