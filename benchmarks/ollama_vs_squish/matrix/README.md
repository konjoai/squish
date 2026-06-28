# Reuse × context-length benchmark matrix

An airtight rebuild of the Ollama-vs-Squish head-to-head: every number labelled
for exactly what it measures, across a matrix of prompt-reuse levels (0/25/50/
75/100%) and context lengths (4k/8k/16k/32k), with cache state **measured** per
run for both systems, thermal control preserved, fair like-for-like configs
(Squish INT4 vs Ollama Q4_K_M), and full statistical rigour.

See **`METHODOLOGY.md`** for the controls/configs and **`ADVERSARIAL_REVIEW.md`**
for the criticism→control→evidence table.

## Run

```bash
# kill-test: ONE cell (8k @ 50% reuse), then STOP for human review
python -m benchmarks.ollama_vs_squish.matrix.run_killtest

# full matrix — only after the kill-test is reviewed and approved
python -m benchmarks.ollama_vs_squish.matrix.run_matrix --i-have-approved
```

Override paths/models via env (`BENCH_OLLAMA_BIN`, `BENCH_OLLAMA_MODEL`,
`BENCH_SQUISH_INT4`, `BENCH_SQUISH_INT3`, `BENCH_SQUISH_PY`, ...). Drop real
corpus documents into `corpus_files/*.txt` to use them instead of the synthetic
generator.

## Layout

| Module | Role | Runs on |
|--------|------|---------|
| `corpus.py` | varied corpus + exact-token reuse-prefix construction | any |
| `stats_ext.py` | paired Wilcoxon + Cliff's delta (stdlib only) | any |
| `cache_probe.py` | measured cache-hit % for both systems + intent check | any |
| `memory.py` | peak-RSS sampler + fit/degraded/OOM classifier | any |
| `thermal.py` | die-temp log, baseline gate, cooldown, drift check | any (sensors: mac) |
| `matrix_spec.py` | the matrix axes + counterbalanced ordering | any |
| `report.py` | per-metric tables, one-screen headlines, plots, post-flight | any |
| `host.py` | RAM detection + tokenizer adapter | any (tokenizer: mac) |
| `systems.py` | Squish/Ollama launchers + streaming clients (fairness flags) | bench host |
| `cell.py` | single-cell paired runner | bench host |
| `run_killtest.py` / `run_matrix.py` | entrypoints | bench host |

Outputs go to `results/benchmark_matrix/{killtest,matrix}/<UTC>/`: raw per-run
JSON, saved prompts + manifest, server logs, per-metric summary tables, plots,
and the post-flight verification.

The pure-logic modules are unit-tested in `tests/test_benchmark_matrix.py`; the
bench-host modules are exercised on Apple Silicon.
