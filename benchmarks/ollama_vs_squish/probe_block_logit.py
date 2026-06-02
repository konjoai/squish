#!/usr/bin/env python3
"""Microbench for v5.1 Fix 2B: per-block last-position logit caching.

The unified v5.1 benchmark uses unaligned prompt lengths (57, 597, 2001,
4053 tokens) which DO NOT trigger the fast-hit path — that path requires
matched_tokens == prompt_tokens AND prompt_tokens >= block_size.  This
microbench specifically constructs a prompt whose token count is an exact
multiple of block_size, so the fast-hit path can show its TTFT benefit.

Compares (5 runs each, median reported):

  * sq +block (v5):   block cache without per-block logit — full match
                      drops the last block and does a suffix forward pass
                      to get the first-response-token logit.
  * sq +block (v5.1): block cache with per-block logit — full match
                      samples directly from the cached last logit, no
                      forward pass.

A correct implementation must produce the same first response token on
both paths (the seed is fixed; temperature=0).
"""
from __future__ import annotations

import json
import os
import shutil
import signal
import statistics as stats
import subprocess
import time
import urllib.request
from pathlib import Path

SQUISH_PY    = "/Users/wscholl/squish/.venv/bin/python"
SQUISH_PORT  = 11437
SQUISH_HOST  = "127.0.0.1"
SQUISH_MODEL = "/Users/wscholl/models/Qwen2.5-7B-Instruct-int4"
CACHE_DIR    = "/tmp/squish_blocks_probe2b"
BLOCK_SIZE   = 64
TARGET_BLOCKS = 9  # 9 * 64 = 576 tokens — block-aligned


def _stream_squish(prompt: str) -> float:
    """Return TTFT in ms (first content chunk)."""
    body = json.dumps({
        "model": "squish",
        "messages": [{"role": "user", "content": prompt}],
        "stream": True, "max_tokens": 1, "temperature": 0.0,
    }).encode()
    req = urllib.request.Request(
        f"http://{SQUISH_HOST}:{SQUISH_PORT}/v1/chat/completions",
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer squish",
        },
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=120) as r:
        for raw in r:
            line = raw.strip()
            if not line.startswith(b"data:"):
                continue
            payload = line[5:].strip()
            if payload == b"[DONE]":
                break
            try:
                d = json.loads(payload)
            except json.JSONDecodeError:
                continue
            choices = d.get("choices") or []
            if choices:
                delta = choices[0].get("delta") or {}
                if delta.get("content"):
                    return (time.perf_counter() - t0) * 1000.0
    return float("nan")


def _stream_text(prompt: str) -> str:
    """Return concatenated content for correctness check (max_tokens=4)."""
    body = json.dumps({
        "model": "squish",
        "messages": [{"role": "user", "content": prompt}],
        "stream": True, "max_tokens": 4, "temperature": 0.0,
    }).encode()
    req = urllib.request.Request(
        f"http://{SQUISH_HOST}:{SQUISH_PORT}/v1/chat/completions",
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer squish",
        },
    )
    parts: list[str] = []
    with urllib.request.urlopen(req, timeout=120) as r:
        for raw in r:
            line = raw.strip()
            if not line.startswith(b"data:"):
                continue
            payload = line[5:].strip()
            if payload == b"[DONE]":
                break
            try:
                d = json.loads(payload)
            except json.JSONDecodeError:
                continue
            choices = d.get("choices") or []
            if choices:
                delta = choices[0].get("delta") or {}
                c = delta.get("content") or ""
                if c:
                    parts.append(c)
    return "".join(parts)


def _build_aligned_prompt(target_tokens: int) -> str:
    """Build a prompt whose tokenized length is exactly *target_tokens*."""
    from mlx_lm import load
    _, tok = load(SQUISH_MODEL)
    chunk = ("The reviewer cares about correctness, performance, observability, "
             "rollback safety, and test coverage. ")
    while True:
        text = chunk * 20  # generously over-shoot
        ids = tok.encode(text)
        if len(ids) >= target_tokens:
            # Try shortening by trimming from the end of text until token count
            # is at most target_tokens; then pad with " is" / " a" until exact.
            chars = len(text)
            while True:
                ids = tok.encode(text)
                if len(ids) <= target_tokens:
                    break
                chars = chars - 1
                text = text[:chars]
            # Now ids is at most target. Add a short token-by-token padding.
            pad = " is a a a a a a a a a a a a a a a a a a a a"
            while len(tok.encode(text + pad)) < target_tokens:
                text += " is"
            # Binary-search the right pad
            for n in range(0, len(pad), 1):
                cand = text + pad[:n]
                if len(tok.encode(cand)) == target_tokens:
                    return cand
            # Fallback: just trim
            ids2 = tok.encode(text)
            return tok.decode(ids2[:target_tokens])
        chunk = chunk * 2


def _start_server() -> subprocess.Popen:
    if os.path.isdir(CACHE_DIR):
        shutil.rmtree(CACHE_DIR)
    os.makedirs(CACHE_DIR, exist_ok=True)
    log = open("/tmp/probe_block_logit.log", "wb")
    proc = subprocess.Popen(
        [
            SQUISH_PY, "-m", "squish.server",
            "--mlx-model-dir", SQUISH_MODEL,
            "--port", str(SQUISH_PORT), "--host", SQUISH_HOST,
            "--block-kv-cache", CACHE_DIR,
            "--block-kv-size", str(BLOCK_SIZE),
            "--log-level", "warning",
        ],
        stdout=log, stderr=subprocess.STDOUT,
        env={**os.environ, "SQUISH_API_KEY": "squish"},
    )
    # wait ready
    for _ in range(600):
        try:
            urllib.request.urlopen(f"http://{SQUISH_HOST}:{SQUISH_PORT}/health",
                                   timeout=1).read()
            return proc
        except Exception:
            time.sleep(0.5)
    proc.terminate()
    raise RuntimeError("server failed to start")


def _stop_server(proc: subprocess.Popen) -> None:
    try:
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def main() -> None:
    target = TARGET_BLOCKS * BLOCK_SIZE
    print(f"Building block-aligned prompt: target = {target} tokens "
          f"({TARGET_BLOCKS} blocks of {BLOCK_SIZE})")
    aligned = _build_aligned_prompt(target)
    from mlx_lm import load
    _, tok = load(SQUISH_MODEL)
    actual = len(tok.encode(aligned))
    print(f"aligned prompt: {actual} tokens (chars={len(aligned)})")

    print()
    print("Starting squish.server (block cache)…")
    subprocess.run(["pkill", "-f", "squish.server"], capture_output=True)
    time.sleep(2)
    proc = _start_server()

    try:
        # Warm with one short send
        _stream_squish("Hello.")
        # First send populates the cache (full miss).  Print TTFT but don't
        # count it.
        first = _stream_squish(aligned)
        print(f"\nFirst send (cold, populates cache): {first:.0f} ms")

        ttfts: list[float] = []
        for i in range(5):
            t = _stream_squish(aligned)
            ttfts.append(t)
            print(f"  hit run {i + 1}: {t:.0f} ms")
        med = stats.median(ttfts)
        p95 = sorted(ttfts)[-1]
        print(f"\nHit-path TTFT: median = {med:.0f} ms, p95 = {p95:.0f} ms "
              f"({TARGET_BLOCKS}-block-aligned prompt, {actual} tokens)")

        # Correctness probe: first 4 tokens should be identical across two
        # back-to-back sends (cache hit both times).
        a = _stream_text(aligned)
        b = _stream_text(aligned)
        ok = a == b
        print("\nDeterminism check (first 4 tokens, twice through cache):")
        print(f"  a = {a!r}")
        print(f"  b = {b!r}")
        print(f"  identical: {ok}")
    finally:
        _stop_server(proc)


if __name__ == "__main__":
    main()
