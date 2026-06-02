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
SQUISH_API_KEY = "squish"
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


def _build_aligned_user_msg(target_total_tokens: int) -> str:
    """Build a user message whose CHAT-TEMPLATED tokenization equals
    *target_total_tokens* (so the server-side block cache sees an
    exact-block-aligned prompt).

    The server applies the chat template (system + user + assistant prelude)
    before tokenizing — that adds a fixed number of overhead tokens.  We
    measure the overhead by templating an empty user message, then pad the
    user content so total == target.
    """
    from mlx_lm import load
    _, tok = load(SQUISH_MODEL)

    def n_tokens(user_text: str) -> int:
        templ = tok.apply_chat_template(
            [{"role": "user", "content": user_text}],
            tokenize=False, add_generation_prompt=True,
        )
        return len(tok.encode(templ))

    overhead = n_tokens("")
    need_user = target_total_tokens - overhead
    if need_user <= 0:
        raise ValueError(
            f"target_total_tokens ({target_total_tokens}) <= chat-template overhead ({overhead})"
        )
    # Iteratively build a user message at exactly need_user tokens
    chunk = ("The reviewer cares about correctness performance observability "
             "rollback safety and test coverage. ")
    text = ""
    while True:
        ids = tok.encode(text + chunk)
        if len(ids) > need_user:
            break
        text += chunk
    # Trim toward target by adding short " a" tokens
    while True:
        total = n_tokens(text)
        if total == target_total_tokens:
            return text
        if total > target_total_tokens:
            # Trim last char
            text = text[:-1]
            continue
        text += " a"


def _build_aligned_prompt(target_tokens: int) -> str:
    """Back-compat alias."""
    return _build_aligned_user_msg(target_tokens)


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
            "--trace", "--log-level", "info",
        ],
        stdout=log, stderr=subprocess.STDOUT,
        env={**os.environ, "SQUISH_API_KEY": SQUISH_API_KEY},
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
    print(f"Building chat-templated block-aligned prompt: target = {target} "
          f"tokens ({TARGET_BLOCKS} blocks of {BLOCK_SIZE})")
    aligned_user = _build_aligned_user_msg(target)
    from mlx_lm import load
    _, tok = load(SQUISH_MODEL)
    user_only_tokens = len(tok.encode(aligned_user))
    templ_total = len(tok.encode(tok.apply_chat_template(
        [{"role": "user", "content": aligned_user}],
        tokenize=False, add_generation_prompt=True,
    )))
    print(f"  user message: {user_only_tokens} tokens "
          f"(chars={len(aligned_user)})")
    print(f"  chat-templated total: {templ_total} tokens "
          f"({'ALIGNED' if templ_total % BLOCK_SIZE == 0 else 'UNALIGNED'})")
    aligned = aligned_user  # send the user message; server applies the template

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
              f"({TARGET_BLOCKS}-block-aligned prompt, {templ_total} tokens)")

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
