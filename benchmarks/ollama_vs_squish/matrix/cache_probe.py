"""Measure — never assume — prefix-cache reuse for BOTH systems, per run.

Fairness rule #2 of the sprint: Ollama 0.30.7 / llama.cpp *can* reuse a cached
prefix within a session, so we must determine and log whether each system
actually reused prefill for every run, and FAIL any cell whose measured hit rate
contradicts the intended reuse level rather than reporting a number we cannot
explain.

How each system is measured (all values the engines themselves report):

* **Ollama** — the terminal ``done`` chunk of ``/api/generate`` reports
  ``prompt_eval_count``: the number of prompt tokens actually evaluated. A
  reused prefix is *not* re-evaluated, so the measured hit fraction is
  ``1 - prompt_eval_count / total_prompt_tokens``. Direct, from the engine.

* **Squish** — preferred: the OpenAI-style ``usage.prompt_tokens_details.
  cached_tokens`` field if the server emits it (read directly, like Ollama).
  Fallback when that field is absent: diff the ``/metrics`` counters
  (``squish_radix_prefix_hits_total`` / ``squish_prefix_cache_hits_total``) to
  confirm a reuse *event* occurred, and estimate the reused fraction from the
  collapse in prefill time versus the cold baseline at the same context length.
  The method used is recorded on every verdict so nothing is silently inferred.

This module is pure parsing/arithmetic and is unit-tested without a server.
"""

from __future__ import annotations

from dataclasses import dataclass


# ── prometheus parsing ────────────────────────────────────────────────────────


def parse_prometheus(text: str) -> dict[str, float]:
    """Parse a Prometheus text-exposition body into {metric_name: value}.

    Ignores HELP/TYPE comment lines and labels (Squish's metrics are unlabelled).
    """
    out: dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        name = parts[0].split("{", 1)[0]
        try:
            out[name] = float(parts[-1])
        except ValueError:
            continue
    return out


# ── verdict type ──────────────────────────────────────────────────────────────


@dataclass
class CacheVerdict:
    system: str
    intended: float
    measured: float | None
    method: str
    status: str  # "ok" | "mismatch" | "unknown"
    note: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "ok"


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


# ── per-system measurement ────────────────────────────────────────────────────


def ollama_hit_fraction(done_chunk: dict, total_prompt_tokens: int) -> tuple[float, str]:
    """Measured reuse fraction for an Ollama run from its terminal chunk.

    ``prompt_eval_count`` is the count of prompt tokens actually prefilled; a
    fully cached prefix omits or zeroes it. Returns (fraction, method).
    """
    if total_prompt_tokens <= 0:
        return 0.0, "ollama:no_prompt_tokens"
    evaluated = done_chunk.get("prompt_eval_count")
    if evaluated is None:
        # llama.cpp omits the field when the entire prompt was served from cache.
        return 1.0, "ollama:prompt_eval_count_absent(full_cache)"
    frac = 1.0 - (float(evaluated) / float(total_prompt_tokens))
    return _clamp01(frac), "ollama:prompt_eval_count"


def squish_hit_fraction(
    usage: dict | None,
    metrics_before: dict[str, float],
    metrics_after: dict[str, float],
    total_prompt_tokens: int,
    *,
    prefill_cold_s: float | None = None,
    prefill_warm_s: float | None = None,
) -> tuple[float, str]:
    """Measured reuse fraction for a Squish run.

    Preference order:
      1. ``usage.prompt_tokens_details.cached_tokens`` (direct, if emitted).
      2. ``/metrics`` counter delta confirms a reuse event AND a prefill-time
         collapse estimates the reused fraction (1 - warm/cold prefill).
      3. counter delta only -> a reuse event happened but fraction unknown.
    """
    if usage:
        details = usage.get("prompt_tokens_details") or {}
        cached = details.get("cached_tokens")
        if cached is not None and total_prompt_tokens > 0:
            return _clamp01(float(cached) / total_prompt_tokens), "squish:usage.cached_tokens"

    def _delta(name: str) -> float:
        return metrics_after.get(name, 0.0) - metrics_before.get(name, 0.0)

    radix_hit = _delta("squish_radix_prefix_hits_total") > 0
    exact_hit = _delta("squish_prefix_cache_hits_total") > 0

    if exact_hit:
        return 1.0, "squish:exact_prefix_cache_hit"
    if radix_hit and prefill_cold_s and prefill_warm_s and prefill_cold_s > 0:
        frac = 1.0 - (prefill_warm_s / prefill_cold_s)
        return _clamp01(frac), "squish:radix_hit+prefill_ratio"
    if radix_hit:
        return float("nan"), "squish:radix_hit_event_only"
    return 0.0, "squish:no_reuse_event"


# ── classification against intent ─────────────────────────────────────────────


def classify(system: str, intended: float, measured: float, method: str) -> CacheVerdict:
    """Decide whether a measured hit fraction matches the intended reuse level.

    Tolerances: 0% allows <=0.05; 100% allows >=0.95; partial allows the larger
    of an absolute 0.10 band or a 15%-of-intended relative band (tokenizer
    boundary slack makes an exact match impossible).
    """
    if measured != measured:  # NaN — a reuse event with no quantifiable fraction
        return CacheVerdict(
            system,
            intended,
            None,
            method,
            "unknown",
            "reuse event observed but fraction not quantifiable",
        )
    if intended <= 0.0:
        ok = measured <= 0.05
        note = "" if ok else f"expected ~0 hits, measured {measured:.1%}"
    elif intended >= 1.0:
        ok = measured >= 0.95
        note = "" if ok else f"expected full hit, measured {measured:.1%}"
    else:
        band = max(0.10, 0.15 * intended)
        ok = abs(measured - intended) <= band
        note = "" if ok else (f"expected ~{intended:.0%}±{band:.0%}, measured {measured:.1%}")
    return CacheVerdict(system, intended, measured, method, "ok" if ok else "mismatch", note)


def verify_run(
    system: str,
    intended: float,
    measured: float,
    method: str,
) -> CacheVerdict:
    """Convenience wrapper: classify a single run's measured fraction."""
    return classify(system, intended, measured, method)
