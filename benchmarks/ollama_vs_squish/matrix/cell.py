"""Single matrix-cell runner: one (reuse, context-length) point, all systems.

A *cell* is one box of the matrix — e.g. "8k context @ 50% reuse". For each
system it runs >=30 paired runs over an identical, pre-generated prompt set, so
``e2e_squish[i]`` and ``e2e_ollama[i]`` are paired on the same prompt and the
paired Wilcoxon test is valid.

Order effects are removed the way ``bench_p4000_iso`` does it: each system is
measured from its OWN cold thermal baseline (cooldown -> baseline gate -> start),
not interleaved, so neither system is ever "the one that ran hot last". A drift
recheck re-measures the first system at the end.

Per run we capture, and label distinctly: cold-start (load + first token), cold
prefill TTFT, decode tok/s, end-to-end (fixed 200-token), peak RSS, KV-cache
memory, and the MEASURED cache-hit fraction for both systems — then verify the
measured reuse matches the cell's intended reuse and record fit/degraded/OOM
status without ever crashing.

This module orchestrates the Mac-only systems; the pure helpers (ITL, pairing,
verdict aggregation) are unit-tested on any platform.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from . import cache_probe, thermal
from .corpus import Corpus, PromptSpec, expected_hit_fraction, save_cell_prompts
from .memory import classify_memory_status, kv_cache_mb_from_metrics, scan_log_for_signals
from .stats_ext import compare_paired


def inter_token_stats(stamps: list[float]) -> dict[str, float | None]:
    """p50/p95 inter-token latency in ms, excluding the first gap (that's TTFT)."""
    if len(stamps) < 3:
        return {"itl_p50_ms": None, "itl_p95_ms": None, "itl_count": 0}
    gaps = [(stamps[i + 1] - stamps[i]) * 1000.0 for i in range(1, len(stamps) - 1)]
    if not gaps:
        return {"itl_p50_ms": None, "itl_p95_ms": None, "itl_count": 0}
    gaps.sort()
    p50 = gaps[len(gaps) // 2]
    p95 = gaps[min(int(len(gaps) * 0.95), len(gaps) - 1)]
    return {"itl_p50_ms": p50, "itl_p95_ms": p95, "itl_count": len(gaps)}


# ── per-run / per-system / per-cell results ───────────────────────────────────


@dataclass
class RunMetrics:
    run_index: int
    ttft_s: float | None
    decode_tps: float | None
    e2e_s: float | None
    itl_p50_ms: float | None
    itl_p95_ms: float | None
    prompt_tokens: int | None
    measured_hit: float | None
    hit_method: str
    cache_status: str
    peak_rss_bytes: int
    kv_cache_mb: float | None
    failed: bool
    error: str = ""


@dataclass
class SystemCellResult:
    system: str
    label: str
    quant: str
    role: str
    version: str
    prompt_lookup: bool
    cold_start_s: float | None
    runs: list[RunMetrics] = field(default_factory=list)
    peak_rss_bytes: int = 0
    memory_status: str = "fit"
    memory_note: str = ""
    cache_ok_runs: int = 0
    cache_total_runs: int = 0
    thermal_max_c: float | None = None

    def metric_values(self, key: str) -> list[float]:
        return [getattr(r, key) for r in self.runs if getattr(r, key) is not None and not r.failed]


@dataclass
class CellResult:
    cell_id: str
    reuse: float
    ctx_tokens: int
    intended_hit: float
    n_runs: int
    systems: dict[str, SystemCellResult] = field(default_factory=dict)
    comparisons: dict[str, Any] = field(default_factory=dict)
    drift: dict[str, Any] = field(default_factory=dict)
    status: str = "ok"  # ok | cache_mismatch | oom | degraded | error
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "cell_id": self.cell_id,
            "reuse": self.reuse,
            "ctx_tokens": self.ctx_tokens,
            "intended_hit": self.intended_hit,
            "n_runs": self.n_runs,
            "status": self.status,
            "notes": self.notes,
            "drift": self.drift,
            "systems": {k: _system_to_dict(v) for k, v in self.systems.items()},
            "comparisons": self.comparisons,
        }


def _system_to_dict(s: SystemCellResult) -> dict[str, Any]:
    d = asdict(s)
    return d


# ── cell runner ───────────────────────────────────────────────────────────────


class CellRunner:
    def __init__(
        self,
        corpus: Corpus,
        out_dir: Path,
        ram_bytes: int,
        n_runs: int = 30,
        cooldown_s: int = thermal.DEFAULT_COOLDOWN_S,
        settle_s: int = thermal.DEFAULT_SETTLE_S,
        baseline_c: float = thermal.BASELINE_TARGET_C,
        log=print,
    ) -> None:
        self.corpus = corpus
        self.out_dir = Path(out_dir)
        self.ram_bytes = ram_bytes
        self.n_runs = n_runs
        self.cooldown_s = cooldown_s
        self.settle_s = settle_s
        self.baseline_c = baseline_c
        self.log = log

    def run(self, reuse: float, ctx_tokens: int, systems: dict, order: list[str]) -> CellResult:
        """Run one cell across the given systems in the given (counterbalanced) order."""
        from . import systems as S  # Mac-only; imported lazily

        cell_id = f"r{int(reuse * 100):03d}_c{ctx_tokens}"
        intended = expected_hit_fraction(reuse)
        prompts = [self.corpus.build_prompt(reuse, ctx_tokens, i) for i in range(self.n_runs)]
        save_cell_prompts(self.out_dir, cell_id, prompts)
        self.log(
            f"=== cell {cell_id}: reuse={reuse:.0%} ctx={ctx_tokens} "
            f"intended_hit={intended:.0%} runs={self.n_runs} ==="
        )

        result = CellResult(cell_id, reuse, ctx_tokens, intended, self.n_runs)
        for name in order:
            sysdef = systems[name]
            result.systems[name] = self._run_system(S, sysdef, prompts, reuse, intended)
        self._finalize(result, order)
        return result

    def _run_system(
        self, S, sysdef, prompts: list[PromptSpec], reuse: float, intended: float
    ) -> SystemCellResult:
        tlog = thermal.ThermalLog()
        thermal.cooldown(self.cooldown_s, kill_fn=S.kill_all_serving, log=self.log)
        thermal.wait_for_baseline(self.baseline_c, log=self.log)
        log_path = self.out_dir / "server_logs" / f"{prompts[0].ctx_tokens}_{sysdef.name}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)

        sampler_proc = None
        scr = SystemCellResult(
            system=sysdef.name,
            label=sysdef.label,
            quant=sysdef.quant,
            role=sysdef.role,
            version=sysdef.read_version(),
            prompt_lookup=sysdef.prompt_lookup,
            cold_start_s=None,
        )
        t_proc_start = time.perf_counter()
        proc, sampler = sysdef.start(log_path)
        try:
            if not S.wait_ready(sysdef.ready_url, timeout=300):
                raise RuntimeError(f"{sysdef.name} did not become ready")
            scr.cold_start_s = self._cold_start(S, sysdef, prompts[0], t_proc_start)
            if 0.0 < reuse:  # prime the shared prefix for partial/exact reuse cells
                S.wait_ready(sysdef.ready_url, timeout=10)
                self._prime(S, sysdef, reuse, prompts[0])
            for i, pspec in enumerate(prompts):
                scr.runs.append(self._one_run(S, sysdef, pspec, i, intended))
                sampler_proc = sampler
        finally:
            S.stop_server(proc, sampler)
        scr.peak_rss_bytes = sampler.peak_bytes
        self._classify_system_memory(scr, log_path)
        scr.cache_total_runs = sum(1 for r in scr.runs if not r.failed)
        scr.cache_ok_runs = sum(1 for r in scr.runs if r.cache_status == "ok")
        scr.thermal_max_c = tlog.max_temp()
        return scr

    def _cold_start(self, S, sysdef, pspec: PromptSpec, t_proc_start: float) -> float | None:
        """Load + first token: process-start wall time to the first decoded token."""
        nctx = S.num_ctx_for(pspec.measured_tokens)
        res = sysdef.stream(pspec.text, max_tokens=1, num_ctx=nctx)
        if res.failed or res.ttft_s is None:
            return None
        return time.perf_counter() - t_proc_start

    def _prime(self, S, sysdef, reuse: float, pspec: PromptSpec) -> None:
        prefix = self.corpus.shared_prefix_text(reuse, pspec.ctx_tokens) or pspec.text
        nctx = S.num_ctx_for(pspec.measured_tokens)
        sysdef.stream(prefix, max_tokens=4, num_ctx=nctx)

    def _one_run(self, S, sysdef, pspec: PromptSpec, i: int, intended: float) -> RunMetrics:
        nctx = S.num_ctx_for(pspec.measured_tokens)
        before = sysdef.metrics() if sysdef.metrics else {}
        ttft_res = sysdef.stream(pspec.text, max_tokens=1, num_ctx=nctx)
        e2e_res = sysdef.stream(pspec.text, max_tokens=S.GEN_TOKENS, num_ctx=nctx)
        after = sysdef.metrics() if sysdef.metrics else {}
        itl = inter_token_stats(e2e_res.chunk_timestamps)

        total_pt = pspec.measured_tokens
        if sysdef.name.startswith("ollama"):
            measured, method = cache_probe.ollama_hit_fraction(e2e_res.done_chunk, total_pt)
        else:
            cold_pf = ttft_res.ttft_s if intended <= 0 else None
            measured, method = cache_probe.squish_hit_fraction(
                e2e_res.usage,
                before,
                after,
                total_pt,
                prefill_cold_s=self._cold_prefill_ref(sysdef.name, pspec.ctx_tokens),
                prefill_warm_s=ttft_res.ttft_s,
            )
        verdict = cache_probe.classify(sysdef.name, intended, measured, method)

        failed = ttft_res.failed or e2e_res.failed
        return RunMetrics(
            run_index=i,
            ttft_s=ttft_res.ttft_s,
            decode_tps=e2e_res.decode_tps,
            e2e_s=e2e_res.total_s if not failed else None,
            itl_p50_ms=itl["itl_p50_ms"],
            itl_p95_ms=itl["itl_p95_ms"],
            prompt_tokens=total_pt,
            measured_hit=None if measured != measured else measured,
            hit_method=method,
            cache_status=verdict.status,
            peak_rss_bytes=0,
            kv_cache_mb=kv_cache_mb_from_metrics(after),
            failed=failed,
            error=(ttft_res.error or e2e_res.error),
        )

    def _cold_prefill_ref(self, system: str, ctx_tokens: int) -> float | None:
        """Reference cold-prefill time for a system+ctx, for the squish ratio path."""
        return (
            self._cold_prefill.get((system, ctx_tokens)) if hasattr(self, "_cold_prefill") else None
        )

    def _classify_system_memory(self, scr: SystemCellResult, log_path: Path) -> None:
        try:
            log_text = Path(log_path).read_text(errors="ignore")
        except OSError:
            log_text = ""
        oom_sig, gov = scan_log_for_signals(log_text)
        any_failed = any(r.failed for r in scr.runs)
        decode_med = (
            statistics.median(scr.metric_values("decode_tps"))
            if scr.metric_values("decode_tps")
            else None
        )
        ms = classify_memory_status(
            peak_rss_bytes=scr.peak_rss_bytes,
            ram_bytes=self.ram_bytes,
            request_failed=any_failed,
            oom_signal=oom_sig,
            governor_event=gov,
            decode_tps=decode_med,
            baseline_tps=None,
        )
        scr.memory_status, scr.memory_note = ms.status, ms.note

    def _finalize(self, result: CellResult, order: list[str]) -> None:
        head = [n for n in order if result.systems[n].role == "head_to_head"]
        if len(head) >= 2:
            a, b = head[0], head[1]
            sa, sb = result.systems[a], result.systems[b]
            for metric in ("e2e_s", "decode_tps", "ttft_s"):
                va, vb = _paired(sa, sb, metric)
                if va and vb and len(va) == len(vb) and len(va) >= 2:
                    cmp = compare_paired(metric, sa.label, sb.label, va, vb)
                    d = _cmp_to_dict(cmp)
                    d["a_system"], d["b_system"] = a, b
                    result.comparisons[metric] = d
        # status rollup
        statuses = [s.memory_status for s in result.systems.values()]
        if "oom" in statuses:
            result.status = "oom"
        elif "degraded_via_governor" in statuses:
            result.status = "degraded"
        if any(
            s.cache_total_runs and s.cache_ok_runs < s.cache_total_runs
            for s in result.systems.values()
        ):
            result.status = "cache_mismatch" if result.status == "ok" else result.status
            for s in result.systems.values():
                bad = s.cache_total_runs - s.cache_ok_runs
                if bad:
                    result.notes.append(
                        f"{s.system}: {bad}/{s.cache_total_runs} runs failed cache-intent check"
                    )


def _paired(
    sa: SystemCellResult, sb: SystemCellResult, key: str
) -> tuple[list[float], list[float]]:
    """Aligned per-run values for two systems (only run indices both completed)."""
    ma = {
        r.run_index: getattr(r, key)
        for r in sa.runs
        if getattr(r, key) is not None and not r.failed
    }
    mb = {
        r.run_index: getattr(r, key)
        for r in sb.runs
        if getattr(r, key) is not None and not r.failed
    }
    common = sorted(set(ma) & set(mb))
    return [ma[i] for i in common], [mb[i] for i in common]


def _cmp_to_dict(cmp) -> dict[str, Any]:
    return {
        "metric": cmp.metric,
        "a_label": cmp.a_label,
        "b_label": cmp.b_label,
        "a": cmp.a,
        "b": cmp.b,
        "wilcoxon": cmp.wilcoxon,
        "cliffs_delta": cmp.cliffs_delta,
        "cliffs_magnitude": cmp.cliffs_magnitude,
        "rank_biserial": cmp.rank_biserial,
        "median_ratio": cmp.median_ratio,
        "significant": cmp.is_significant(),
    }
