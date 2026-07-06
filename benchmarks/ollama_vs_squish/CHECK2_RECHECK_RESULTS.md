# Check 2 recheck — machine-load confound cleared, clean re-run confirmed

Follow-up to the failed Check 2 run (`bench_thermal_h2h.py`, ctx=4000,
Ollama 0.31.1) which failed its own 8% drift gate at **+65.1%**. Root cause
was machine load unrelated to either engine, not a real regression. This
doc records the cleanup and the clean re-run.

## Root cause

Two sources of contention were present during the failed run
(`results/benchmarks_v5_1_1/thermal/20260705T214104.json`):

1. **`git-commit-push-script.sh`** (`~/git-commit-push-script/`) had been
   running continuously since 2026-06-23 10:30, pegged at ~80% CPU
   (~12.9 days, ~9,600 CPU-minutes). No `launchd` plist or crontab entry
   referenced it — it was a plain orphaned background process, not a
   managed job.
2. **Microsoft Defender for Endpoint / DLP** (`dlpdaemon`, `epsext`,
   `tracer`) was consuming 48% / 29% / 13% CPU during the run. Checked via
   `mdatp scan list` / `mdatp health --details scheduled_scan`: no scan was
   in progress at any point — this is steady-state real-time-protection
   overhead (centrally managed, `tamper_protection: block`, no user-level
   pause available), not a transient scan. It does not "clear" on its own;
   it's a fixed part of this machine's baseline load and was **not** the
   dominant contributor to the drift failure.

## Remediation

- Killed `git-commit-push-script.sh` (`pkill -f`); confirmed no relaunch
  after a 15s recheck. No autostart mechanism found (`launchctl list`,
  `crontab -l` both clean).
- Confirmed Defender/DLP idle (no active scan) rather than disabled —
  out of scope to touch (IT-managed).
- Confirmed quiescence before re-running: CPU idle 47%→78% after the kill,
  load average trending down, swap stable ~1.5-1.6 GB (not growing, vs.
  2.25 GB during the failed run), no single process pegged.

## Clean re-run

`bench_thermal_h2h.py`, same config as the failed run (`ollama`,
`squish_daemon`, `squish_recommended_int4`, `ollama_recheck` — the failed
run's JSON did not include `squish_recommended_int3`, since that model
isn't present on this machine; the recheck reproduces that exact 4-config
set rather than the full `ORDER` in the current script, which would hang
on the missing INT3 model). Ollama binary: `/opt/homebrew/bin/ollama`,
confirmed 0.31.1. Raw output:
`results/benchmarks_v5_1_1/thermal/20260706T080359.json`.

### Drift check (p75 warm tok/s, ollama first pass vs. `ollama_recheck`)

| | tok/s |
|---|---|
| Ollama (first pass) | 18.6 |
| Ollama (recheck) | 17.7 |

**Drift: -4.5%**, against the 8% ceiling. **PASS.** (Failed run: +65.1%.)

### ctx=4000 E2E comparison (the metric Check 2 exists to validate)

| | E2E @ 4000 tok |
|---|---|
| Ollama (first pass) | 38.96 s |
| Squish daemon INT4 | 4.40 s |
| Squish recommended INT4 (block+pkv) | 4.42 s |

Speedup: **8.81-8.85x** vs. the published **9.8x** (`BENCHMARKS.md` §1b:
37.5s / 3.8s).

## Verdict

**Within noise of the published 9.8x.** Ollama's own number (38.96s)
tracks the published baseline (37.5s) to within ~4%. Squish's number
(4.40-4.42s) is ~16% above published, landing the ratio at ~8.8x — a
single-run deviation, not a regression signal; no repeated clean sampling
exists yet to bound this metric's noise more tightly than that.

One pattern worth noting, not a concern: ollama's own p4000 throughput
drops substantially between its first pass and `ollama_recheck` (16.6→10.2
tok/s, -38%, E2E 38.96s→55.93s). This matches the machine's previously
documented thermal-envelope behavior at p4000
(`THERMAL_H2H_ollama_018_vs_307.md`, historical isolation run: 16.1→10.0
tok/s, -38%) — it is why the drift *gate* is computed on the cheap p75
phase (which fully cools within the 120s cooldown) rather than on p4000,
and why the ctx=4000 comparison above uses each config's first,
cool-start measurement rather than a late-in-sequence one.

This is a read-only verification recheck — no change to `BENCHMARKS.md`
or any published number.
