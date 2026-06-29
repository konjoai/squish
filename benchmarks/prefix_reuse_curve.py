"""Prefix-reuse speedup curve — git-diff summarization prompts, 5–95 % overlap.

Measures the in-memory prompt-prefix KV reuse path end-to-end: for each target
overlap it sends the *same* measured prompt twice, varying only what sits in the
reuse slot beforehand —

  * cold  : an unrelated prompt is sent first (0 % shared prefix) → full prefill
  * reuse : a sibling prompt sharing the leading tokens is sent first → the shared
            prefix's KV is restored and only the differing suffix is prefilled

Both runs are greedy (temperature 0, fixed seed); their outputs MUST be
byte-identical if reuse is lossless. Prompts are real ``git diff`` summarization
tasks pulled from this repo's history. Self-contained: launches its own squish
server with the standard fp16 block+prompt-kv config, runs the sweep, tears down.

Usage:
    .venv/bin/python -m benchmarks.prefix_reuse_curve \
        --model ~/models/Qwen2.5-7B-Instruct-int4
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import signal
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
_LOG = logging.getLogger("prefix_reuse_curve")

OVERLAPS = list(range(5, 100, 5))  # 5,10,...,95
HOST, PORT = "127.0.0.1", 8098


# ── prompt material: real git diffs from this repo ────────────────────────────


def _diff_corpus(repo: Path, min_tokens: int, tok) -> list[int]:
    """Concatenated token ids of several real commit diffs (deterministic order)."""
    hashes = subprocess.run(
        ["git", "-C", str(repo), "log", "--format=%H", "-60"],
        capture_output=True, text=True, check=True,
    ).stdout.split()
    text_parts: list[str] = []
    ids: list[int] = []
    for h in hashes:
        diff = subprocess.run(
            ["git", "-C", str(repo), "show", "--no-color", h],
            capture_output=True, text=True, check=True,
        ).stdout
        if not diff.strip():
            continue
        text_parts.append(diff)
        ids = tok("\n".join(text_parts), add_special_tokens=False).input_ids
        if len(ids) >= min_tokens:
            break
    if len(ids) < min_tokens:
        raise RuntimeError(f"diff corpus too small: {len(ids)} < {min_tokens} tokens")
    return ids


_INSTR = (
    "You are a senior software engineer reviewing a pull request. Read the git diff "
    "below and write a concise summary of what changed, the main risks, and where a "
    "reviewer should focus.\n\nGIT DIFF:\n"
)


def _build_prompts(tok, corpus: list[int], context: int):
    """For each overlap, a (seed, measured) pair sharing ~overlap% leading tokens,
    plus one unrelated cold-control prompt. Returns (control, {overlap: (seed, meas)})."""
    instr_ids = tok(_INSTR, add_special_tokens=False).input_ids
    body = context - len(instr_ids)
    if body <= 0 or len(corpus) < 3 * context:
        raise RuntimeError("context too large for instruction + corpus")
    # Two disjoint regions of the corpus give divergent suffixes A and B.
    region_a = corpus[:context]
    region_b = corpus[context : 2 * context]
    control_ids = instr_ids + corpus[2 * context : 3 * context - len(instr_ids)]

    def decode(ids: list[int]) -> str:
        return tok.decode(ids)

    pairs = {}
    for ov in OVERLAPS:
        shared = round((ov / 100.0) * body)
        common = instr_ids + region_a[:shared]
        seed = common + region_a[shared:body]            # seed = common + A-suffix
        meas = common + region_b[shared:body]            # measured = common + B-suffix
        pairs[ov] = (decode(seed), decode(meas))
    return decode(control_ids), pairs


# ── server lifecycle ──────────────────────────────────────────────────────────


def _wait_ready(timeout: int = 180) -> bool:
    for _ in range(timeout):
        try:
            urllib.request.urlopen(f"http://{HOST}:{PORT}/v1/models", timeout=5)
            return True
        except (urllib.error.URLError, OSError):
            time.sleep(1)
    return False


def _start_server(py: str, model: str) -> subprocess.Popen:
    for d in ("/tmp/prc_block", "/tmp/prc_pkv"):
        shutil.rmtree(d, ignore_errors=True)
    cmd = [
        py, "-m", "squish.server", "--mlx-model-dir", model,
        "--port", str(PORT), "--host", HOST, "--log-level", "warning",
        "--block-kv-cache", "/tmp/prc_block", "--block-kv-size", "64",
        "--prompt-kv-cache", "/tmp/prc_pkv",
    ]
    log = open("/tmp/prc_server.log", "w")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    if not _wait_ready():
        proc.send_signal(signal.SIGINT)
        raise RuntimeError("squish server did not become ready (see /tmp/prc_server.log)")
    return proc


# ── one timed request ─────────────────────────────────────────────────────────


def _chat(prompt: str, max_tokens: int, seed: int = 7) -> tuple[float, str]:
    body = json.dumps({
        "model": "squish",
        "messages": [{"role": "user", "content": prompt}],
        "stream": False, "max_tokens": max_tokens, "temperature": 0.0, "seed": seed,
    }).encode()
    req = urllib.request.Request(
        f"http://{HOST}:{PORT}/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=600) as resp:
        data = json.load(resp)
    dt = time.perf_counter() - t0
    return dt, data["choices"][0]["message"]["content"]


def _measure(prompt: str, seed_prompt: str, reps: int, gen: int) -> tuple[float, str]:
    """Median wall time for ``prompt`` after priming the slot with ``seed_prompt``."""
    times, out = [], ""
    for _ in range(reps):
        _chat(seed_prompt, 2)            # set the reuse slot
        dt, out = _chat(prompt, gen)     # the timed, measured request
        times.append(dt)
    return statistics.median(times), out


# ── sweep ─────────────────────────────────────────────────────────────────────


def _realized_overlap(tok, seed: str, meas: str) -> float:
    a = tok(seed, add_special_tokens=False).input_ids
    b = tok(meas, add_special_tokens=False).input_ids
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return 100.0 * n / max(1, len(b))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.expanduser("~/models/Qwen2.5-7B-Instruct-int4"))
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--contexts", default="512,1024,2048",
                    help="comma-separated approx prompt sizes in tokens")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--gen", type=int, default=8, help="tokens generated per request")
    ap.add_argument("--out", default="results/prefix_reuse_curve")
    ap.add_argument("--resume", default="", metavar="DIR",
                    help="Resume into an existing run dir: each measured (size, overlap) "
                    "point is checkpointed to rows.jsonl, so done points are skipped.")
    args = ap.parse_args()

    contexts = [int(c) for c in args.contexts.split(",") if c.strip()]

    # Output dir + checkpoint live from the start, so a kill loses at most the
    # in-flight point. Resume loads the checkpoint and skips done (size, overlap).
    if args.resume:
        out_dir = Path(args.resume)
        if not out_dir.is_dir():
            _LOG.error("--resume dir does not exist: %s", out_dir)
            return 2
    else:
        ts = subprocess.run(["date", "+%Y%m%dT%H%M%S"],
                            capture_output=True, text=True).stdout.strip()
        out_dir = Path(args.out) / ts
        out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / "rows.jsonl"
    rows: list[dict] = []
    done: set[tuple[int, int]] = set()
    if ckpt.exists():
        for ln in ckpt.read_text().splitlines():
            if ln.strip():
                r = json.loads(ln)
                rows.append(r)
                done.add((r["context"], r["target_overlap_pct"]))
        _LOG.info("[resume] loaded %d checkpointed points from %s", len(rows), ckpt)
    _LOG.info("[out] %s", out_dir)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)

    _LOG.info("[corpus] building git-diff prompt material…")
    corpus = _diff_corpus(Path.cwd(), min_tokens=3 * max(contexts) + 256, tok=tok)

    _LOG.info("[server] starting squish (fp16 block+pkv)…")
    proc = _start_server(args.python, args.model)
    try:
        _chat("warmup", 2)
        for ctx in contexts:
            control, pairs = _build_prompts(tok, corpus, ctx)
            for ov in OVERLAPS:
                if (ctx, ov) in done:
                    _LOG.info("  ctx %5d  overlap %2d%%  SKIP (checkpointed)", ctx, ov)
                    continue
                seed, meas = pairs[ov]
                realized = _realized_overlap(tok, seed, meas)
                realized_tokens = len(tok(meas, add_special_tokens=False).input_ids)
                t_reuse, out_reuse = _measure(meas, seed, args.reps, args.gen)
                t_cold, out_cold = _measure(meas, control, args.reps, args.gen)
                lossless = out_cold == out_reuse
                speedup = t_cold / t_reuse if t_reuse else float("nan")
                row = {
                    "context": ctx, "realized_tokens": realized_tokens,
                    "target_overlap_pct": ov, "realized_overlap_pct": round(realized, 1),
                    "cold_s": round(t_cold, 3), "reuse_s": round(t_reuse, 3),
                    "speedup": round(speedup, 2), "lossless": lossless,
                }
                rows.append(row)
                with ckpt.open("a") as fh:  # crash-safe per-point checkpoint
                    fh.write(json.dumps(row) + "\n")
                _LOG.info(
                    "  ctx %5d  overlap %2d%% (real %.0f%%)  cold %.2fs  reuse %.2fs  "
                    "%.2fx  lossless=%s",
                    ctx, ov, realized, t_cold, t_reuse, speedup, lossless,
                )
    finally:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()

    rows.sort(key=lambda r: (r["context"], r["target_overlap_pct"]))
    contexts = sorted({r["context"] for r in rows})
    meta = {"model": args.model, "contexts": contexts, "reps": args.reps, "gen": args.gen}
    (out_dir / "curve.json").write_text(json.dumps({"meta": meta, "rows": rows}, indent=2))

    lines = [
        "# Prefix-reuse speedup curve (git-diff summarization)\n",
        f"model={Path(args.model).name}  contexts={contexts} tok  "
        f"reps={args.reps}  gen={args.gen}\n",
    ]
    for ctx in contexts:
        lines += [
            f"\n## context ≈ {ctx} tokens\n",
            "| overlap% (target/real) | cold s | reuse s | speedup | lossless |",
            "|---|---|---|---|---|",
        ]
        for r in (x for x in rows if x["context"] == ctx):
            lines.append(
                f"| {r['target_overlap_pct']} / {r['realized_overlap_pct']} | "
                f"{r['cold_s']} | {r['reuse_s']} | {r['speedup']}× | "
                f"{'✓' if r['lossless'] else '✗ FAIL'} |"
            )
    all_lossless = all(r["lossless"] for r in rows)
    lines.append(f"\nAll points lossless: **{all_lossless}**\n")
    (out_dir / "curve.md").write_text("\n".join(lines))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
        for ctx in contexts:
            sub = [r for r in rows if r["context"] == ctx]
            xs = [r["realized_overlap_pct"] for r in sub]
            ax1.plot(xs, [r["reuse_s"] for r in sub], "o-", label=f"{ctx} reuse")
            ax1.plot(xs, [r["cold_s"] for r in sub], "o--", alpha=0.4,
                     label=f"{ctx} cold")
            ax2.plot(xs, [r["speedup"] for r in sub], "o-", label=f"{ctx} tok")
        ax1.set_xlabel("prefix overlap %"); ax1.set_ylabel("end-to-end seconds")
        ax1.set_title("Cold vs reuse, by prompt size")
        ax1.legend(fontsize=7); ax1.grid(alpha=0.3)
        ax2.set_xlabel("prefix overlap %"); ax2.set_ylabel("speedup ×")
        ax2.set_title("Reuse speedup, by prompt size")
        ax2.legend(fontsize=8); ax2.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(out_dir / "curve.png", dpi=120)
    except ImportError:
        _LOG.info("[plot] matplotlib unavailable — skipping PNG")

    _LOG.info("\n[done] wrote %s", out_dir)
    _LOG.info("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
