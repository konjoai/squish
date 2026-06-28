# Adversarial review — criticism → control → evidence

For each likely criticism, the exact control that answers it and the evidence
the harness produces. "Evidence" names the field/file/function a reviewer can
inspect.

| Criticism | Control | Evidence |
|-----------|---------|----------|
| "Same prompt resent with a warm cache." | Reuse levels are explicit axes; 0%/unique is the no-cache head-to-head; the cache-hit % is measured per run. | `corpus.build_prompt` (reuse axis), per-run `measured_hit`/`hit_method` in `*_raw.json`, `cache_probe.classify`. |
| "Ollama also caches; you disabled it." | Ollama caching is **measured** from its own `prompt_eval_count` and both systems share a cache posture per condition. | `cache_probe.ollama_hit_fraction`, METHODOLOGY "Cache posture per condition", `keep_alive=-1` + `num_ctx` in `systems.stream_ollama`. |
| "INT3 vs Q4 is unfair." | Head-to-head is `squish_int4` vs `ollama_q4km` (both ~4-bit, `role="head_to_head"`); INT3 is a separate capability row paired with the accuracy gate. | `systems.build_systems` roles; `report.metric_table` uses only head-to-head systems. |
| "Faster but lower quality." | Report arc_easy / perplexity beside speed, within ~1pp; INT3 carries the accuracy gate (INT4 AWQ g=32 ≥ 70.6% arc_easy). | Accuracy gate in CLAUDE.md / quant pipeline; capability row labelling. (Run `squish` lm-eval and paste beside the speed table.) |
| "Ollama was crippled / reloaded / small ctx." | `keep_alive=-1`, `num_ctx` sized to prompt+gen, latest version, two versions cross-checked, flags logged. | `systems.stream_ollama` body, `num_ctx_for`, `System.read_version`, `matrix_spec.CROSS_CHECK_CELLS`. |
| "Thermal / order favouritism." | 50 °C baseline gate, 120 s cooldown, 25 s settle, drift ≤1.7%, live die-temp log, per-system cold isolation + counterbalanced order. | `thermal.wait_for_baseline`, `thermal.drift_check`, `thermal.TemperatureSampler`, `matrix_spec.counterbalanced_order`. |
| "Synthetic repetitive padding." | Real varied corpus from large pools; no single-paragraph repetition; prompts saved with seeds/hashes. | `corpus._sentence`/`_document`, `save_cell_prompts` → `prompts/<cell>/`, `manifest.json`. |
| "Only 4k / one length / one model." | 4k–32k context, 0–100% reuse, second-model (3B) 0% and 50% rows recommended. | `matrix_spec.CONTEXT_LENGTHS`/`REUSE_LEVELS`/`second_model_rows`. |
| "n=5, no stats." | ≥30 paired runs, paired Wilcoxon, Cliff's delta, full distributions; runner refuses <30. | `stats_ext.compare_paired`, `run_matrix` `--n-runs` guard, per-cell `comparisons`. |
| "e2e hides which phase wins." | Cold prefill TTFT and decode tok/s reported separately, plus e2e. | distinct `ttft_s` / `decode_tps` / `e2e_s` per run; `report.metric_table` per metric. |
| "Speculative decoding inflates it." | Default-on for the matrix plus a `squish_int4_nospec` isolation pass with prompt-lookup OFF. | `systems` `squish_int4_nospec` (`--no-prompt-lookup`), `prompt_lookup` field per run. |
| "KV blows up / OOM hidden at 32k." | KV-cache memory + peak RSS recorded; per-cell fit / degraded / OOM status; harness catches failures. | `memory.classify_memory_status`, `kv_cache_mb`, `peak_rss_bytes`, cell `status`. |
| "Not reproducible." | Raw JSON, saved prompts, seeds, commit hash, one command; stdlib stats (no hidden scipy). | `results/benchmark_matrix/.../*_raw.json`, `prompts/`, `manifest.json`, `stats_ext` (pure). |

## How a "FAIL" surfaces rather than hides

- A cell whose measured cache-hit % contradicts its intended reuse is marked
  `status="cache_mismatch"` with a per-system note (`cell._finalize`), not
  silently reported.
- A system that OOMs or degrades is marked `oom` / `degraded_via_governor`; the
  request failure is caught (`systems.stream_*` return `failed=True`) so the run
  records a non-fit instead of crashing.
- Post-flight (`report.postflight`) prints PASS/FAIL for: cache intent, ≥30
  paired runs, presence of p-value + effect size, and OOM handling.
