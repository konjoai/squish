# Release Pipeline Diagnosis

Date: 2026-06-05
Investigator: Claude Code sprint

---

## Current State

| Item | Value |
|------|-------|
| pyproject.toml version (before fix) | 9.33.5 |
| pyproject.toml version (after fix) | 9.33.6 |
| Formula version | squish-ai 9.33.5 |
| publish.yml trigger | `push: tags: v*.*.*` |
| PyPI auth method | OIDC Trusted Publisher (no token needed) |

---

## Root Causes

### 1. orjson upper-bound cap blocks Python 3.14 wheel (CRITICAL)

**Before:** `orjson>=3.9,<3.11` in `pyproject.toml`

`orjson 3.10.18` satisfies `<3.11` (3.10.x < 3.11.0 in PEP 440) so Python 3.13
installs work. However the cap blocks `orjson 3.11.9`, which is the first release
to ship `cp314` wheels for macOS arm64. On Python 3.14 (default on macOS Tahoe),
pip falls through to source compilation, which requires a Rust toolchain and fails
for most users.

**After:** `orjson>=3.11.9` — removes the cap, pins the minimum to the first version
with cp314 wheels.

### 2. Homebrew formula calls pip live during install (CRITICAL)

**Before:** `Formula/squish.rb` `install` block called `system libexec/"bin/pip", "install", "squish-ai==#{version}"` at install time. The macOS Homebrew sandbox blocks outbound network connections during `brew install`, so this silently fails or times out.

**After:** `virtualenv_install_with_resources` with fully populated resource blocks covering all 39 transitive dependencies (pre-downloaded by Homebrew before sandbox is entered). No network access during install.

### 3. requires-python too broad (MINOR)

**Before:** `requires-python = ">=3.10"` — advertises Python 3.10 support but Python 3.10 is EOL (Oct 2026) and mlx dropped it in 0.20.

**After:** `requires-python = ">=3.11,<3.15"` — accurately reflects supported range.

### 4. Missing formula auto-update step in publish.yml (PIPELINE BUG)

**Before:** `publish.yml` published to PyPI and created a GitHub Release, but never updated `Formula/squish.rb` with the new version URL and sha256. After every release, the formula was stale until manually updated.

**After:** A `Update Homebrew formula URL and sha256` step runs after the PyPI publish. It waits for CDN propagation, computes the new sha256, and commits the formula update. Resource blocks still require manual regeneration when dependency versions change (see note in formula).

---

## What Was Working

- PyPI publish via OIDC Trusted Publisher: correctly configured, no token required
- GitHub Release creation: working
- Build step (`python -m build`): working
- `twine check`: working
- Formula sha256 for 9.33.5 tarball: correct
- Homebrew `depends_on "python@3.13"`: already correct

---

## What Was Broken

| Issue | Status |
|-------|--------|
| `orjson>=3.9,<3.11` caps Python 3.14 wheel | Fixed in 9.33.6 |
| Formula calls pip live (sandbox blocks network) | Fixed — resource blocks added |
| Formula uses manual `pip install` instead of `virtualenv_install_with_resources` | Fixed |
| publish.yml missing formula auto-update step | Fixed |
| `requires-python` advertised 3.10 (EOL, mlx dropped it) | Fixed to `>=3.11,<3.15` |
| Python 3.13/3.14 not in classifiers | Fixed |

---

## Secrets / Auth

The publish workflow uses OIDC Trusted Publisher — no `PYPI_API_TOKEN` secret is needed.
The only secret used is `GITHUB_TOKEN` (auto-provisioned by Actions) for the release and formula push.

To verify OIDC is configured on PyPI:
1. Log in to pypi.org → Manage → squish-ai → Publishing
2. Confirm `konjoai/squish` GitHub Actions publisher is listed

---

## Known Limitations

Resource blocks in the Homebrew formula are pinned to specific wheel versions. They are
NOT automatically regenerated when `publish.yml` bumps the formula to a new version.
After any release that changes `mlx`, `mlx-lm`, `numpy`, `pydantic-core`, or other
binary dependencies, regenerate resource blocks manually:

```bash
pip install squish-ai homebrew-pypi-poet
poet squish-ai
```

Then commit the updated `Formula/squish.rb` before tagging the release.
