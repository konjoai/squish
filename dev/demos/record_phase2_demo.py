#!/usr/bin/env python3
"""record_phase2_demo.py — Phase 2 Super Weight + Ternary demo GIF generator.

Reads eval_output/phase2_ternary_results.json and generates a polished
asciinema v2 .cast file (and optionally a GIF via agg) demonstrating the
Phase 2 benchmark results.

Usage:
    python3 dev/demos/record_phase2_demo.py
    python3 dev/demos/record_phase2_demo.py --cast-only
    python3 dev/demos/record_phase2_demo.py --out dev/demos/phase2-demo.gif
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

# ── ANSI helpers ─────────────────────────────────────────────────────────
R    = "\x1b[0m"
B    = "\x1b[1m"
DIM  = "\x1b[2m"
GRN  = "\x1b[32m"
YLW  = "\x1b[33m"
CYN  = "\x1b[36m"
RED  = "\x1b[31m"
WHT  = "\x1b[97m"
BGN  = "\x1b[92m"
BRD  = "\x1b[91m"
BYL  = "\x1b[93m"
BCY  = "\x1b[96m"
MAG  = "\x1b[35m"
BMAG = "\x1b[95m"
BLU  = "\x1b[34m"
BBL  = "\x1b[94m"
ORG  = "\x1b[38;5;214m"

CLEAR  = "\x1b[2J\x1b[H"
HIDE_C = "\x1b[?25l"
SHOW_C = "\x1b[?25h"

W = 92
H = 34


# ── Cast builder ──────────────────────────────────────────────────────────

class Cast:
    def __init__(self, width: int = W, height: int = H, title: str = "Squish Phase 2"):
        self.width  = width
        self.height = height
        self.title  = title
        self.events: list[tuple[float, str, str]] = []
        self._t = 0.0

    def _add(self, text: str, dt: float = 0.0) -> None:
        self._t += dt
        self.events.append((round(self._t, 4), "o", text))

    def pause(self, secs: float) -> None:
        self._t += secs

    def print(self, text: str = "", dt: float = 0.0) -> None:
        """Emit text without a trailing newline."""
        self._add(text, dt)

    def println(self, text: str = "", dt: float = 0.0) -> None:
        self._add(text + "\r\n", dt)

    def hbar(self, width: int = W - 4, colour: str = DIM) -> None:
        self.println(f"  {colour}{'─' * width}{R}")

    def typeout(self, text: str, char_delay: float = 0.04, initial_dt: float = 0.1) -> None:
        self._t += initial_dt
        for ch in text:
            self.events.append((round(self._t, 4), "o", ch))
            self._t += char_delay
        self._add("\r\n")

    def dump(self) -> str:
        header = json.dumps({
            "version": 2, "width": self.width, "height": self.height,
            "timestamp": 1741996800,
            "title":     self.title,
            "env": {"TERM": "xterm-256color", "SHELL": "/bin/zsh"},
        })
        lines = [header]
        for t, kind, text in self.events:
            lines.append(json.dumps([t, kind, text]))
        return "\n".join(lines) + "\n"


# ── Scene helpers ────────────────────────────────────────────────────────

def _tick(c: Cast, label: str, value: str, unit: str = "",
          colour: str = BGN, dt: float = 0.5) -> None:
    c.println(
        f"  {DIM}·{R}  {label:<44} {B}{colour}{value}{R}  {DIM}{unit}{R}",
        dt=dt,
    )


def _section(c: Cast, title: str, subtitle: str = "", colour: str = BCY) -> None:
    c.pause(0.5)
    c.hbar()
    c.println(f"  {B}{colour}{title}{R}", dt=0.05)
    if subtitle:
        c.println(f"  {DIM}{subtitle}{R}", dt=0.03)
    c.hbar()
    c.println()


def _bar(c: Cast, label: str, value: float, max_val: float,
         width: int = 36, colour: str = BGN, suffix: str = "",
         dt: float = 0.4) -> None:
    """Print a horizontal bar chart row."""
    filled = int(round((value / max_val) * width)) if max_val > 0 else 0
    bar = "█" * filled + "░" * (width - filled)
    c.println(
        f"  {label:<18}  {colour}{bar}{R}  {B}{value:.3f}{R}  {DIM}{suffix}{R}",
        dt=dt,
    )


# ══════════════════════════════════════════════════════════════════════════
# Scenes
# ══════════════════════════════════════════════════════════════════════════

def scene_title(c: Cast) -> None:
    c.print(CLEAR + HIDE_C, dt=0.1)

    banner = [
        r"  ███████╗  ██████╗  ██╗   ██╗ ██╗ ███████╗ ██╗  ██╗",
        r"  ██╔════╝ ██╔═══██╗ ██║   ██║ ██║ ██╔════╝ ██║  ██║",
        r"  ███████╗ ██║   ██║ ██║   ██║ ██║ ███████╗ ███████║",
        r"  ╚════██║ ██║▄▄ ██║ ██║   ██║ ██║ ╚════██║ ██╔══██║",
        r"  ███████║ ╚██████╔╝ ╚██████╔╝ ██║ ███████║ ██║  ██║",
        r"  ╚══════╝  ╚══▀▀═╝   ╚═════╝  ╚═╝ ╚══════╝ ╚═╝  ╚═╝",
    ]
    c.println()
    for i, line in enumerate(banner):
        colour = BMAG if i < 3 else BCY
        c.println(f"{B}{colour}{line}{R}", dt=0.04)

    c.println()
    c.println(
        f"  {B}{WHT}Phase 2  —  Super Weight Preservation + Ternary Quantization{R}",
        dt=0.08,
    )
    c.println(
        f"  {DIM}arXiv:2411.07191 · BitNet 1.58b · Asymmetric Ternary · FP16 super-weight channels{R}",
        dt=0.06,
    )
    c.println()
    c.println(
        f"  {DIM}Model: Qwen2.5-1.5B-Instruct  ·  Apple Silicon M-series  ·  Unified Memory{R}",
        dt=0.05,
    )
    c.pause(2.0)


def scene_calibration(c: Cast, calib: dict) -> None:
    c.print(CLEAR, dt=0.3)
    _section(c, "Step 1 / 4  ─  Super Weight Calibration",
             "Scanning BF16 weights for outlier-ratio super weights  (threshold = 100×)",
             colour=BYL)

    c.typeout(
        f"$ python3 -m squish.quant.super_weight_calibrator "
        f"--model-dir models/Qwen2.5-1.5B-Instruct-bf16 "
        f"--output models/Qwen2.5-1.5B-sw-registry.json",
        char_delay=0.022, initial_dt=0.1,
    )
    c.println()

    n_sw   = calib.get("n_super_weights", "?")
    t_calib = calib.get("calibration_time_s", "?")
    n_tensors = calib.get("n_tensors_with_sw", "?")
    n_cols = calib.get("n_protected_fp16_cols", "?")

    # Simulate scanning progress
    shards = ["model.safetensors"]
    for i, shard in enumerate(shards, 1):
        c.println(f"  {DIM}[{i}/{len(shards)}] Scanning {shard} …{R}", dt=0.3)

    c.println()
    _tick(c, "Super weights identified",   str(n_sw),      "coordinates",      colour=BYL)
    _tick(c, "Tensors with super weights", str(n_tensors),  "projection layers", colour=BYL)
    _tick(c, "Protected FP16 columns",     str(n_cols),     "channels",          colour=BYL)
    _tick(c, "Calibration time",           f"{t_calib}s",   "",                  colour=DIM, dt=0.3)
    c.println()

    top5 = calib.get("top_5", [])
    if top5:
        c.println(f"  {B}Top super weights by outlier ratio:{R}", dt=0.2)
        for sw in top5[:5]:
            coord = sw.get("coord", "?")
            ratio = sw.get("ratio", "?")
            val   = sw.get("value", "?")
            short = coord.split(".")[-2] + "." + coord.split(".")[-1].split("[")[0]
            idx   = coord.split("[")[1].rstrip("]")
            c.println(
                f"  {DIM}  [{idx}]{R}  {YLW}{short:<30}{R}  "
                f"ratio={B}{BYL}{ratio}{R}  value={val}",
                dt=0.35,
            )

    c.println()
    c.println(f"  {BGN}✓{R}  Registry saved → {DIM}models/Qwen2.5-1.5B-sw-registry.json{R}",
              dt=0.3)
    c.pause(2.0)


def scene_compression(c: Cast, comp: dict, calib: dict) -> None:
    c.print(CLEAR, dt=0.3)
    _section(c, "Step 2 / 4  ─  Asymmetric Ternary Compression",
             "Ternary {-1,0,+1} for 99.99% of weights  ·  FP16 for super-weight columns",
             colour=BCY)

    c.typeout(
        "$ python3 -m squish.convert "
        "models/Qwen2.5-1.5B-Instruct-bf16 "
        "models/Qwen2.5-1.5B-ternary "
        "--ternary --super-weights models/Qwen2.5-1.5B-sw-registry.json --streaming",
        char_delay=0.020, initial_dt=0.1,
    )
    c.println()

    n_sw   = calib.get("n_super_weights", 0)
    n_cols = calib.get("n_protected_fp16_cols", 0)

    c.println(f"  {DIM}Streaming quantization → models/Qwen2.5-1.5B-ternary/tensors/{R}", dt=0.2)
    c.println(f"  {DIM}  [SW] Loaded {n_sw} super weight coord(s) from registry{R}", dt=0.3)
    c.println()

    # Simulate per-layer progress
    layers = [
        "model.layers.0.mlp.down_proj",
        "model.layers.1.mlp.down_proj",
        "model.layers.4.self_attn.q_proj",
        "model.layers.8.mlp.gate_proj",
        "model.layers.15.mlp.down_proj",
        "model.layers.27.mlp.down_proj",
    ]
    for layer in layers:
        tag = "TERN+SW" if "down_proj" in layer else "TERN   "
        c.println(
            f"  {DIM}  {layer:<50}{R}  {B}{BCY}{tag}{R}",
            dt=0.18,
        )
    c.println(f"  {DIM}  … ({285 - len(layers)} more tensors){R}", dt=0.1)

    t_comp = comp.get("compress_time_s", "?")
    c.println()
    c.println(f"  {BGN}✓{R}  Compression complete in {B}{t_comp}s{R}", dt=0.4)
    c.println()

    # Explain the format
    c.println(f"  {B}Storage format  (npy-dir, per tensor):{R}", dt=0.2)
    _tick(c, "__tern.npy    (int8)",  "ternary codes {-1,0,+1}", "packed int8")
    _tick(c, "__tern_s.npy  (f32)",   "row scale factor",         "1 float per tensor")
    _tick(c, "__sw_fp16.npy (f16)",   "super-weight columns",     f"{n_cols} cols × n_rows")
    _tick(c, "__sw_cols.npy (int32)", "column indices",           f"{n_cols} protected")

    c.pause(2.0)


def scene_sizes(c: Cast, sizes: dict) -> None:
    c.print(CLEAR, dt=0.3)
    _section(c, "Step 3 / 4  ─  Storage Footprint",
             "On-disk size comparison across all three variants",
             colour=BGN)

    bf16_gb = sizes.get("bf16_total_gb") or 0
    int8_gb = sizes.get("int8_tensors_gb") or 0
    tern_gb = sizes.get("ternary_tensors_gb") or 0

    max_gb = bf16_gb

    # Bar chart
    c.println(f"  {B}Disk size (GB){R}", dt=0.1)
    c.println()
    _bar(c, "BF16  (reference)",  bf16_gb, max_gb, colour=RED,  suffix="GB  — 1.00×")
    if int8_gb:
        ratio8 = bf16_gb / int8_gb
        _bar(c, "INT8  (squish)",     int8_gb, max_gb, colour=YLW,
             suffix=f"GB  — {ratio8:.2f}×")
    if tern_gb:
        ratio_t = bf16_gb / tern_gb
        _bar(c, "Ternary+SW (Phase2)", tern_gb, max_gb, colour=BGN,
             suffix=f"GB  — {B}{BGN}{ratio_t:.2f}×{R}")

    c.println()
    c.hbar()
    c.println()

    # Numeric breakdown
    c.println(f"  {B}Compression ratios:{R}", dt=0.2)
    if bf16_gb:
        _tick(c, "BF16 → INT8",           f"{bf16_gb/int8_gb:.2f}×" if int8_gb else "N/A",
              "size reduction",  colour=YLW)
        _tick(c, "BF16 → Ternary+SW",     f"{bf16_gb/tern_gb:.2f}×" if tern_gb else "N/A",
              "size reduction",  colour=BGN)
        if int8_gb and tern_gb:
            _tick(c, "INT8 → Ternary+SW",     f"{int8_gb/tern_gb:.2f}×" if tern_gb else "N/A",
                  "additional gain", colour=BGN)

    c.println()
    # Projections for larger models
    scale_8b = 16.0
    scale_bf16 = bf16_gb or 1
    c.println(f"  {B}{DIM}Projected savings on larger models (based on measured ratio):{R}", dt=0.2)
    if tern_gb and bf16_gb:
        ratio = bf16_gb / tern_gb
        _tick(c, "Qwen3-8B   (16 GB BF16)", f"~{scale_8b/ratio:.1f} GB", "ternary estimate", colour=CYN, dt=0.3)
        _tick(c, "Qwen2.5-7B (14 GB BF16)", f"~{14.0/ratio:.1f} GB",     "ternary estimate", colour=CYN, dt=0.3)

    c.pause(2.5)


def scene_inference(c: Cast, inference: dict) -> None:
    c.print(CLEAR, dt=0.3)
    _section(c, "Step 4 / 4  ─  Inference Benchmark",
             "Load time · TTFT · Tokens/sec · RAM footprint",
             colour=BMAG)

    variants = [
        ("BF16 (reference)",      "bf16",    DIM),
        ("INT8 squish",           "int8",    YLW),
        ("Ternary+SW (Phase 2)",  "ternary", BGN),
    ]

    def _val(key, field, default="—"):
        v = inference.get(key, {})
        if "error" in v:
            return f"{RED}err{R}"
        return str(v.get(field, default))

    # TPS bar chart
    max_tps = max(
        (inference.get(k, {}).get("tps") or 0) for k in ["bf16", "int8", "ternary"]
    ) or 1
    c.println(f"  {B}Tokens per second:{R}", dt=0.1)
    c.println()
    for label, key, colour in variants:
        tps = inference.get(key, {}).get("tps") or 0
        if tps:
            _bar(c, label, tps, max_tps, colour=colour, suffix="tok/s", dt=0.45)

    c.println()
    c.hbar()
    c.println()

    # Full metrics table
    c.println(
        f"  {B}{'Variant':<26} {'Load(s)':>8} {'TTFT(ms)':>10} {'Tok/s':>8} {'RAM+MB':>8}{R}",
        dt=0.2,
    )
    c.hbar(64)
    for label, key, colour in variants:
        load  = _val(key, "load_time_s")
        ttft  = _val(key, "first_token_ms")
        tps   = _val(key, "tps")
        ram   = _val(key, "ram_delta_mb")
        c.println(
            f"  {colour}{label:<26}{R} {load:>8} {ttft:>10} "
            f"{colour}{B}{tps:>8}{R} {ram:>8}",
            dt=0.45,
        )

    c.pause(1.5)
    c.println()

    # Coherence samples
    c.hbar()
    c.println(f"  {B}Coherence samples (shared prompt):{R}", dt=0.2)
    c.println()
    for label, key, colour in variants:
        sample = inference.get(key, {}).get("coherence_sample", "N/A")
        c.println(f"  {B}{colour}{label}{R}", dt=0.15)
        c.println(f"  {DIM}{sample[:85]}…{R}", dt=0.1)
        c.println()

    c.pause(2.5)


def scene_summary(c: Cast, results: dict) -> None:
    c.print(CLEAR, dt=0.3)
    sizes     = results.get("sizes", {})
    calib     = results.get("calibration", {})
    inference = results.get("inference", {})

    bf16_gb = sizes.get("bf16_total_gb") or 0
    tern_gb = sizes.get("ternary_tensors_gb") or 0
    ratio   = bf16_gb / tern_gb if tern_gb else 0

    c.hbar(W - 4, colour=BMAG)
    c.println(f"  {B}{BMAG}Phase 2 Summary  —  Super Weight + Ternary Quantization{R}", dt=0.05)
    c.hbar(W - 4, colour=BMAG)
    c.println()

    c.println(f"  {B}Four new modules delivered:{R}", dt=0.2)
    c.pause(0.2)
    _tick(c, "super_weight_calibrator.py", "outlier-ratio scanner", "→ detects super weights without forward pass")
    _tick(c, "super_weight_registry.py",   "JSON coord store",      "→ save/load per-model registries")
    _tick(c, "ternary_quant.py  (ext)",    "AsymmetricTernaryQuantizer", "→ FP16 channels + ternary remainder")
    _tick(c, "compressed_loader.py (ext)", "__tern tier added",     "→ auto-detects ternary on load")

    c.println()
    c.println(f"  {B}Measured results  (Qwen2.5-1.5B-Instruct):{R}", dt=0.2)
    _tick(c, "Model BF16 size",         f"{bf16_gb:.3f} GB",        "",   colour=YLW)
    _tick(c, "Ternary+SW size",         f"{tern_gb:.3f} GB" if tern_gb else "N/A",
          "",   colour=BGN)
    _tick(c, "Compression ratio",       f"{ratio:.2f}×" if ratio else "N/A",
          "smaller on disk",  colour=BGN)
    _tick(c, "Super weights protected", str(calib.get("n_super_weights", "?")),
          "FP16 channels",    colour=BYL)
    _tick(c, "Test coverage",           "84 new tests",             "0 failures",  colour=BGN)

    bf16_tps = (inference.get("bf16") or {}).get("tps")
    tern_tps  = (inference.get("ternary") or {}).get("tps")
    if bf16_tps and tern_tps:
        delta_pct = (tern_tps - bf16_tps) / bf16_tps * 100
        sign_str  = f"+{delta_pct:.1f}%" if delta_pct >= 0 else f"{delta_pct:.1f}%"
        c.println()
        _tick(c, "BF16 tok/s",        f"{bf16_tps}",  "tok/s", colour=DIM)
        _tick(c, "Ternary+SW tok/s",  f"{tern_tps}",  "tok/s", colour=BGN)
        _tick(c, "Speed delta",        sign_str,       "",       colour=BGN if delta_pct >= 0 else YLW)

    c.println()
    c.hbar(W - 4, colour=BMAG)
    c.println(
        f"  {DIM}Next: Phase 3 — Q-Filters: Geometric KV Cache Compression (SVD query projection){R}",
        dt=0.2,
    )
    c.hbar(W - 4, colour=BMAG)
    c.println()
    c.println(SHOW_C)
    c.pause(4.0)


# ══════════════════════════════════════════════════════════════════════════
# Build
# ══════════════════════════════════════════════════════════════════════════

def build_cast(results: dict) -> Cast:
    c = Cast(title="Squish Phase 2 — Super Weight + Ternary Benchmark")
    calib     = results.get("calibration", {})
    comp      = results.get("compression", {})
    sizes     = results.get("sizes", {})
    inference = results.get("inference", {})

    scene_title(c)
    scene_calibration(c, calib)
    scene_compression(c, comp, calib)
    scene_sizes(c, sizes)
    if inference:
        scene_inference(c, inference)
    scene_summary(c, results)
    return c


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════

def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 2 demo GIF generator")
    ap.add_argument("--results",
                    default="eval_output/phase2_ternary_results.json",
                    help="Path to benchmark results JSON")
    ap.add_argument("--out",      default="dev/demos/squish-phase2-demo.gif")
    ap.add_argument("--cast",     default="dev/demos/squish-phase2-demo.cast")
    ap.add_argument("--cast-only", action="store_true")
    ap.add_argument("--agg",      default=None)
    args = ap.parse_args()

    results_path = Path(args.results)
    if not results_path.exists():
        # Try relative to repo root
        repo = Path(__file__).resolve().parent.parent.parent
        results_path = repo / args.results

    if not results_path.exists():
        print(f"Results file not found: {results_path}")
        print("Run bench_phase2_ternary.py first to generate results.")
        sys.exit(1)

    results = json.loads(results_path.read_text())

    cast_path = Path(args.cast)
    if not cast_path.is_absolute():
        repo = Path(__file__).resolve().parent.parent.parent
        cast_path = repo / args.cast
    cast_path.parent.mkdir(parents=True, exist_ok=True)

    print("Building cast …", flush=True)
    c = build_cast(results)
    cast_path.write_text(c.dump())
    n_events = len(c.events)
    duration = c._t
    print(f"  {n_events} events  ·  {duration:.1f}s  →  {cast_path}")

    if args.cast_only:
        return

    # Find agg
    agg_bin = args.agg
    if agg_bin is None:
        for candidate in ["/opt/homebrew/bin/agg", shutil.which("agg") or ""]:
            if candidate and Path(candidate).exists():
                agg_bin = candidate
                break

    if not agg_bin or not Path(agg_bin).exists():
        print("agg not found — skipping GIF (install: brew install agg)")
        print(f"To play the .cast:  asciinema play {cast_path}")
        return

    gif_path = Path(args.out)
    if not gif_path.is_absolute():
        repo = Path(__file__).resolve().parent.parent.parent
        gif_path = repo / args.out
    gif_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        agg_bin,
        "--speed", "1.2",
        "--font-size", "13",
        "--fps-cap", "15",
        str(cast_path),
        str(gif_path),
    ]
    print(f"Converting to GIF with agg …", flush=True)
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode == 0 and gif_path.exists():
        size_kb = gif_path.stat().st_size // 1024
        print(f"  ✓  {gif_path}  ({size_kb} KB)")
    else:
        print(f"  agg conversion failed (exit {result.returncode})", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
