#!/usr/bin/env python3
"""Squish — Demo Day Walkthrough

Runs end-to-end against the actual `squish.kv.kv_cache` module and prints a
six-section briefing covering:

  1. INT8 / INT4 / INT2 KV cache construction with synthetic tokens.
  2. Memory footprint comparison (bytes per token, per mode).
  3. Reconstruction SNR for INT4 and INT2 vs float16 ground truth.
  4. `recommended_kv_mode` / `recommended_kv_mode_3tier` recommendations
     across realistic context lengths.
  5. `HadamardKVCache` construction and INT2-with-rotation comparison.
  6. Putting it all together: end-to-end roundtrip through `update()`.

Runs CPU-only — no MLX, no GPU, no real model weights.  Uses `rich` for
beautiful tables and panels when available; degrades gracefully to plain
print otherwise.

Usage
-----
    python3 demo/demo.py                # run all sections
    pip install rich && python3 demo/demo.py    # prettier output
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np

# Make `squish` importable when running from a fresh checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from squish.kv.kv_cache import (  # noqa: E402
    HadamardKVCache,
    KV_INT2_AUTO_THRESHOLD,
    KV_INT4_DEFAULT_THRESHOLD,
    QuantizedKVCache,
    _dequantize_int2_per_channel,
    _dequantize_int4_per_channel,
    _dequantize_int8_per_channel,
    _quantize_int2_per_channel,
    _quantize_int4_per_channel,
    _quantize_int8_per_channel,
    recommended_kv_mode,
    recommended_kv_mode_3tier,
)

# ── Rich setup (optional) ────────────────────────────────────────────────────
try:
    from rich.box import HEAVY, ROUNDED
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    HAVE_RICH = True
except ImportError:                           # pragma: no cover
    HAVE_RICH = False
    Console = Panel = Table = Text = None     # type: ignore
    HEAVY = ROUNDED = None                    # type: ignore


_console = Console() if HAVE_RICH else None


def _heading(title: str, subtitle: str = "") -> None:
    """Render a section heading."""
    if HAVE_RICH:
        body = Text(title, style="bold cyan")
        if subtitle:
            body.append(f"\n{subtitle}", style="dim")
        _console.print(Panel(body, box=HEAVY, border_style="cyan", expand=False))
    else:
        bar = "═" * max(len(title), len(subtitle)) if subtitle else "═" * len(title)
        print(f"\n{bar}\n{title}")
        if subtitle:
            print(subtitle)
        print(bar)


def _say(msg: str, style: str = "") -> None:
    if HAVE_RICH:
        _console.print(msg, style=style)
    else:
        print(msg)


def _table(title: str, columns: list[tuple[str, str]], rows: list[list[str]]) -> None:
    """Render a table.  ``columns`` is ``[(header, justify), …]``."""
    if HAVE_RICH:
        t = Table(title=title, box=ROUNDED, header_style="bold magenta",
                  title_style="bold")
        for header, justify in columns:
            t.add_column(header, justify=justify)
        for row in rows:
            t.add_row(*row)
        _console.print(t)
    else:
        print(f"\n{title}")
        widths = [max(len(c[0]), max((len(r[i]) for r in rows), default=0))
                  for i, c in enumerate(columns)]
        sep = "  ".join("─" * w for w in widths)
        print("  ".join(c[0].ljust(w) for c, w in zip(columns, widths)))
        print(sep)
        for row in rows:
            print("  ".join(cell.ljust(w) for cell, w in zip(row, widths)))


# ── Common helpers ───────────────────────────────────────────────────────────


def _snr_db(signal: np.ndarray, recon: np.ndarray) -> float:
    s = signal.astype(np.float64)
    r = recon.astype(np.float64)
    sig = float(np.mean(s * s))
    err = float(np.mean((s - r) ** 2))
    return 10.0 * math.log10(sig / err) if err > 0 else float("inf")


def _make_synthetic_kv(
    n_tokens: int,
    n_heads: int = 8,
    head_dim: int = 128,
    rotate: bool = True,
    seed: int = 7,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Create realistic-ish synthetic K and V token streams.

    Heavy-tailed activations (~5 % outlier columns at 10× variance) optionally
    rotated through a random Hadamard matrix — the post-rotation distribution
    is what `HadamardKVCache` actually sees.
    """
    rng = np.random.default_rng(seed)
    keys, vals = [], []
    for _ in range(n_tokens):
        k = rng.standard_normal((n_heads, head_dim)).astype(np.float32) * 0.3
        v = rng.standard_normal((n_heads, head_dim)).astype(np.float32) * 0.3
        # Heavy outlier channels — the failure mode that motivates Hadamard
        # rotation in the first place.
        k[:, ::20] *= 10.0
        v[:, ::20] *= 10.0
        if rotate:
            H = HadamardKVCache._build_hadamard(
                head_dim, np.random.default_rng(seed)
            ).astype(np.float32)
            k = k @ H
            v = v @ H
        keys.append(k.astype(np.float16))
        vals.append(v.astype(np.float16))
    return keys, vals


# ── Section 1: cache construction ───────────────────────────────────────────


def section_cache_construction() -> None:
    _heading(
        "1. KV Cache Construction",
        "INT8 (KIVI default) · INT4 (W105) · INT2 (W104) · all from one API",
    )

    rows = []
    n_layers, window, n_tokens = 4, 8, 32
    for mode in ("int8", "int4", "int2"):
        cache = QuantizedKVCache(n_layers=n_layers, window=window, mode=mode)
        keys, vals = _make_synthetic_kv(n_tokens, n_heads=4, head_dim=64, rotate=False)
        for k, v in zip(keys, vals):
            cache.update(0, k, v)
        layer = cache._layers[0]
        old_q = layer.keys_old_q
        old_shape = "—" if old_q is None else f"{old_q.shape}"
        old_dtype = "—" if old_q is None else str(old_q.dtype)
        rows.append([
            mode,
            str(cache.n_layers),
            f"{n_tokens}",
            old_shape,
            old_dtype,
            f"{layer.memory_bytes:,} B",
        ])

    _table(
        "QuantizedKVCache(n_layers=4, window=8) — 32 tokens injected",
        [
            ("mode",    "left"),
            ("layers",  "right"),
            ("tokens",  "right"),
            ("old_q.shape", "left"),
            ("old_q.dtype", "left"),
            ("layer 0 RAM", "right"),
        ],
        rows,
    )
    _say(
        "Each row shows the actual buffer shape and dtype after eviction. "
        "INT2 stores head_dim/4 packed bytes per token; INT4 stores head_dim/2; "
        "INT8 stores the full head_dim.",
        style="dim",
    )


# ── Section 2: memory comparison ────────────────────────────────────────────


def section_memory_table() -> None:
    _heading(
        "2. Memory Footprint (bytes per token)",
        "Asymptotic per-token cost across head dimensions",
    )

    head_dims = [64, 96, 128, 160, 192, 256]
    SCALE_BYTES = 4   # one fp32 scale per token

    rows = []
    for d in head_dims:
        b_int8 = d + SCALE_BYTES
        b_int4 = d // 2 + SCALE_BYTES
        b_int2 = d // 4 + SCALE_BYTES
        rows.append([
            f"{d}",
            f"{b_int8} B",
            f"{b_int4} B",
            f"{b_int2} B",
            f"{b_int8 / b_int4:.2f}×",
            f"{b_int8 / b_int2:.2f}×",
        ])

    _table(
        "Per-token storage (codes + scale)",
        [
            ("head_dim", "right"),
            ("INT8",     "right"),
            ("INT4",     "right"),
            ("INT2",     "right"),
            ("INT4 vs INT8", "right"),
            ("INT2 vs INT8", "right"),
        ],
        rows,
    )

    # Realistic 7 B-class context budget.
    _heading(
        "Memory at scale — Qwen2.5-7B-class (28 layers, 8 KV heads, head_dim=128)",
        "Per-token cost × n_layers × n_kv_heads, multiplied by 2 for K and V",
    )
    n_layers, n_kv_heads, head_dim = 28, 8, 128
    factor = n_layers * n_kv_heads * 2          # K + V buffers
    rows = []
    for ctx_k in (4, 8, 16, 32, 64):
        ctx = ctx_k * 1024
        b8 = (head_dim + SCALE_BYTES) * factor * ctx
        b4 = (head_dim // 2 + SCALE_BYTES) * factor * ctx
        b2 = (head_dim // 4 + SCALE_BYTES) * factor * ctx
        rows.append([
            f"{ctx_k} K",
            _fmt_bytes(b8),
            _fmt_bytes(b4),
            _fmt_bytes(b2),
        ])
    _table(
        "Total KV cache RAM",
        [
            ("context",     "right"),
            ("INT8",        "right"),
            ("INT4 (W105)", "right"),
            ("INT2 (W104)", "right"),
        ],
        rows,
    )


def _fmt_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:,.1f} {unit}" if unit != "B" else f"{n:,} {unit}"
        n /= 1024
    return f"{n:,} B"


# ── Section 3: reconstruction SNR ───────────────────────────────────────────


def section_snr() -> None:
    _heading(
        "3. Reconstruction SNR vs float16 ground truth",
        "Higher is better.  Hadamard-rotated activations.",
    )

    rng = np.random.default_rng(2026)
    H = HadamardKVCache._build_hadamard(
        128, np.random.default_rng(7)
    ).astype(np.float32)
    base = rng.standard_normal((512, 128)).astype(np.float32) * 0.3
    base[:, ::20] *= 10.0                       # heavy-tailed outliers
    rotated = (base @ H).astype(np.float16)
    raw = base.astype(np.float16)

    rows = []
    for label, signal in (("raw heavy-tailed", raw),
                          ("Hadamard-rotated", rotated)):
        snr8 = _snr_db(signal, _dequantize_int8_per_channel(
            *_quantize_int8_per_channel(signal)))
        snr4 = _snr_db(signal, _dequantize_int4_per_channel(
            *_quantize_int4_per_channel(signal), 128))
        snr2 = _snr_db(signal, _dequantize_int2_per_channel(
            *_quantize_int2_per_channel(signal), 128))
        rows.append([
            label,
            f"{snr8:5.2f} dB",
            f"{snr4:5.2f} dB",
            f"{snr2:5.2f} dB",
        ])

    _table(
        "SNR — n=512 tokens, head_dim=128, σ=0.3 + 5 % outlier columns @ 10×",
        [
            ("input", "left"),
            ("INT8",  "right"),
            ("INT4",  "right"),
            ("INT2",  "right"),
        ],
        rows,
    )
    _say(
        "Rotation typically lifts INT2 by ≥ 1 dB by spreading outlier energy. "
        "Each extra bit Shannon-bounds at ~+6 dB, which the table reflects.",
        style="dim",
    )


# ── Section 4: mode recommendations ─────────────────────────────────────────


def section_recommendations() -> None:
    _heading(
        "4. recommended_kv_mode — pick storage by planned context length",
        "2-tier (W104 default) vs 3-tier (W105)",
    )

    rows = []
    contexts = [1024, 4096, 8192, 12_288, 16_384, 24_000, 32_768, 64_000]
    for ctx in contexts:
        m2 = recommended_kv_mode(ctx)
        m3 = recommended_kv_mode_3tier(ctx)
        rows.append([
            f"{ctx:,}",
            m2,
            m3,
            "↓"  if m2 != m3 else "",
        ])

    _table(
        f"Thresholds: INT4@{KV_INT2_AUTO_THRESHOLD:,}  ·  INT2@{KV_INT4_DEFAULT_THRESHOLD:,}",
        [
            ("context tokens", "right"),
            ("2-tier (W104)",   "left"),
            ("3-tier (W105)",   "left"),
            ("changed?",         "left"),
        ],
        rows,
    )
    _say(
        "The 3-tier mapping introduces INT4 in the 8 K–16 K band where INT2 is "
        "overkill but INT8 is too dear.  Production callers typically just "
        "drive their cache mode with `recommended_kv_mode_3tier(planned_ctx)`.",
        style="dim",
    )


# ── Section 5: Hadamard rotation ────────────────────────────────────────────


def section_hadamard() -> None:
    _heading(
        "5. HadamardKVCache — QuaRot-style rotation before quantization",
        "Spreads outlier energy uniformly across dimensions so INT2 does not collapse",
    )

    n_layers, window, n_tokens = 2, 4, 16
    keys, vals = _make_synthetic_kv(
        n_tokens, n_heads=4, head_dim=64, rotate=False, seed=314
    )

    quantized = QuantizedKVCache(n_layers=n_layers, window=window, mode="int2")
    hadamard  = HadamardKVCache(n_layers=n_layers, window=window, mode="int2", seed=42)
    for k, v in zip(keys, vals):
        quantized.update(0, k, v)
        hadamard.update(0, k, v)

    full_q_k, _ = quantized._layers[0].get_full_kv()
    full_h_k, _ = hadamard._layers[0].get_full_kv()
    # Compare against the original (un-rotated) input.  HadamardKVCache stores
    # H·K so its raw `get_full_kv()` is in the rotated frame; for a fair
    # "user-visible quality" comparison we need the un-rotated path that
    # `get_kv_mlx()` exposes.  Since this demo is CPU-only, we hand-rotate
    # the cached output back with H.T to mirror what the MLX path does.
    H = hadamard._get_H_k(64).astype(np.float32)
    full_h_k_unrot = (full_h_k.astype(np.float32) @ H.T).astype(np.float16)

    # Reference: stack of original keys (n_heads, n_tokens, head_dim)
    ref_k = np.stack(keys, axis=1)              # (n_heads, n_tokens, head_dim)

    snr_q = _snr_db(ref_k, full_q_k)
    snr_h = _snr_db(ref_k, full_h_k_unrot)

    _table(
        "Cache-roundtrip SNR — synthetic outlier-heavy K stream, INT2 storage",
        [
            ("variant",                    "left"),
            ("storage class",              "left"),
            ("SNR vs original",            "right"),
        ],
        [
            ["QuantizedKVCache(mode=int2)",  "raw INT2",            f"{snr_q:.2f} dB"],
            ["HadamardKVCache(mode=int2)",   "QuaRot rotated INT2", f"{snr_h:.2f} dB"],
        ],
    )

    delta = snr_h - snr_q
    arrow = "↑" if delta > 0 else "↓" if delta < 0 else "→"
    _say(
        f"Hadamard lift on outlier-heavy input: {arrow} {abs(delta):.2f} dB. "
        "On real LLM activations the gap is consistently larger because real "
        "K/V channels are more imbalanced than this synthetic seed.",
        style="dim",
    )


# ── Section 6: end-to-end roundtrip ─────────────────────────────────────────


def section_roundtrip() -> None:
    _heading(
        "6. End-to-end roundtrip — `update()` + `get_full_kv()`",
        "Same code path the MLX integration uses — just with synthetic numpy",
    )

    n_heads, head_dim, n_tokens = 8, 128, 64
    keys, vals = _make_synthetic_kv(
        n_tokens, n_heads=n_heads, head_dim=head_dim, rotate=False, seed=2025
    )

    rows = []
    for mode in ("int8", "int4", "int2"):
        cache = HadamardKVCache(
            n_layers=1, window=16, mode=mode, seed=42
        )
        t0 = time.perf_counter()
        for k, v in zip(keys, vals):
            cache.update(0, k, v)
        elapsed = (time.perf_counter() - t0) * 1000
        full_k, full_v = cache._layers[0].get_full_kv()
        H = cache._get_H_k(head_dim).astype(np.float32)
        recon = (full_k.astype(np.float32) @ H.T).astype(np.float16)
        ref   = np.stack(keys, axis=1)
        rows.append([
            mode,
            f"{full_k.shape}",
            f"{cache._layers[0].memory_bytes:,} B",
            f"{elapsed:5.1f} ms",
            f"{_snr_db(ref, recon):5.2f} dB",
        ])

    _table(
        f"HadamardKVCache — {n_tokens} tokens, n_heads={n_heads}, head_dim={head_dim}",
        [
            ("mode",         "left"),
            ("full_k.shape", "left"),
            ("layer RAM",    "right"),
            ("update() wall","right"),
            ("end-to-end SNR", "right"),
        ],
        rows,
    )


# ── Entry point ──────────────────────────────────────────────────────────────


def main() -> int:
    if HAVE_RICH:
        _console.print(Panel.fit(
            Text.from_markup(
                "[bold cyan]squish[/]  ·  [white]demo day[/]\n"
                "[dim]Per-token symmetric KV-cache quantization · INT8 / INT4 / INT2[/]"
            ),
            border_style="bright_magenta",
            box=HEAVY,
        ))
    else:
        print("=" * 72)
        print("squish — demo day  (per-token KV quantization, INT8 / INT4 / INT2)")
        print("=" * 72)
        print("Tip: `pip install rich` for prettier output.")

    section_cache_construction()
    section_memory_table()
    section_snr()
    section_recommendations()
    section_hadamard()
    section_roundtrip()

    _heading(
        "Done.",
        "Try the interactive HTML at  demo/index.html  for the visual version.",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
