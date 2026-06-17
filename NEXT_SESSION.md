# Next Session

## First-run health gate (`squish doctor` folded into `squish run`) — v9.34.2

### Marker contract
- Marker file: `~/.squish/.doctor_ok` (module constant `cli._DOCTOR_MARKER`).
- Contents: the exact `squish.__version__` string that last passed the checks.
- `squish run` re-runs `run_health_checks()` iff the marker is absent or its
  contents `!= __version__`. On a full pass (no REQUIRED check failed) it is
  rewritten atomically (tmp file + `os.replace`) with the current version. On a
  required-check failure it is NOT written and `run` aborts via `_die`.
- Bypass: `--skip-doctor` (on `run`/`serve`) or `SQUISH_SKIP_DOCTOR=1` — neither
  reads nor writes the marker.
- Optional checks (currently only `squash-ai`) never affect pass/proceed/marker.

### Follow-up idea
- `squish doctor` could write the same `~/.squish/.doctor_ok` marker on a full
  pass, so that explicitly running `squish doctor` pre-satisfies the first-run
  gate of the next `squish run` (today only `run` writes it). Keep the write
  helper shared if this is done, so the atomic-write + version-string contract
  stays in one place.
