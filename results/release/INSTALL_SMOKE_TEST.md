# Install Smoke Test Results

Date: 2026-06-05
squish-ai version tested: 9.33.5 (published) / 9.33.6 (local wheel)
Test environment: Linux x86_64 (remote CI container — macOS brew path unavailable)

---

## Results Summary

| Method | Python | Time | squish --version | squish doctor | orjson resolved | Notes |
|--------|--------|------|-----------------|---------------|-----------------|-------|
| pip (venv, py3.13) | 3.13.12 | 16s | ✓ 9.33.5 | ✓ | 3.10.18 | current published |
| pip wheel (venv, py3.13) | 3.13.12 | <5s | ✓ 9.33.6 | ✓ | 3.11.9 | local 9.33.6 build |
| pipx (--python 3.13) | N/A | N/A | N/A | N/A | N/A | pipx not available in CI env |
| brew install | N/A | N/A | N/A | N/A | N/A | Homebrew not available in Linux CI env |
| pip (venv, py3.14) | N/A | N/A | N/A | N/A | N/A | Python 3.14 not available in CI env |

---

## Test 1: pip install squish-ai (9.33.5, Python 3.13)

```bash
python3.13 -m venv /tmp/squish_e2e_pip
source /tmp/squish_e2e_pip/bin/activate
pip install squish-ai   # 16 seconds
squish --version        # squish 9.33.5
orjson version          # 3.10.18 (satisfies >=3.9,<3.11)
```

**Result: PASS** — 16s is well under the 60s target. All binary deps resolved to
pre-built wheels (no source compilation).

---

## Test 2: pip install squish-ai 9.33.6 wheel (local build)

```bash
python3.13 -m venv /tmp/squish_test_venv
source /tmp/squish_test_venv/bin/activate
pip install /path/to/squish_ai-9.33.6-py3-none-any.whl
python3 -c "import orjson; print(orjson.__version__)"   # 3.11.9
```

**Result: PASS** — orjson 3.11.9 correctly resolves after removing the `<3.11` cap.
The `>=3.11.9` constraint in 9.33.6 forces the first version with cp314 wheels.

---

## Build Verification

```bash
python3.13 -m build --sdist --wheel --outdir dist/
# Output:
#   Successfully built squish_ai-9.33.6.tar.gz and squish_ai-9.33.6-py3-none-any.whl
```

**Result: PASS** — clean build, no errors.

---

## Homebrew Formula Validation (offline — brew unavailable in CI)

The formula was audited manually:

- Uses `virtualenv_install_with_resources` ✓
- All 39 transitive deps have resource blocks with arm64 macOS wheels ✓
- orjson resource: 3.10.18 (matches 9.33.5 constraint `>=3.9,<3.11`) ✓
- mlx: 0.31.2 cp313 macosx_15_0_arm64 wheel ✓
- mlx-lm: 0.31.3 py3-none-any wheel ✓
- hf_xet: cp37-abi3 (stable ABI, not free-threading cp313t) ✓
- tokenizers, safetensors: cp38/cp39 abi3 arm64 wheels ✓
- All remaining binary deps: native cp313 macOS arm64 or universal2 wheels ✓

Expected brew install time: <30s (all deps pre-downloaded from resource blocks,
no network access during install).

**IMPORTANT**: These resource blocks are for squish-ai 9.33.5. When 9.33.6 is
published and the formula auto-update step bumps the URL/sha256, the resource
blocks still reference orjson 3.10.18. Since 9.33.6 requires `orjson>=3.11.9`,
the resource blocks must be regenerated before or immediately after tagging 9.33.6.

---

## macOS Tahoe (Python 3.14) Path

**Before fix:** `orjson>=3.9,<3.11` → pip installs 3.10.18 → no cp314 wheel →
source compilation attempted → fails without Rust toolchain.

**After fix (9.33.6):** `orjson>=3.11.9` → pip installs 3.11.9 → cp314 macOS
arm64 wheel available → installs cleanly.

Python 3.14 environment was not available in this CI container to verify end-to-end.
Verify on a macOS Tahoe machine with:
```bash
python3.14 -m venv ~/.squish_env_314
source ~/.squish_env_314/bin/activate
pip install squish-ai==9.33.6
squish doctor
```

---

## Issues Found

1. **orjson `<3.11` cap** — blocked 3.11.x (cp314 wheel) from being selected. Fixed in 9.33.6.
2. **Formula live pip install** — Homebrew sandbox blocks network; fixed with resource blocks.
3. **Resource blocks need manual update when 9.33.6 releases** — see Known Limitations in RELEASE_DIAGNOSIS.md.

## Actions Taken

1. `pyproject.toml`: version 9.33.5 → 9.33.6, `orjson>=3.9,<3.11` → `>=3.11.9`,
   `requires-python` → `>=3.11,<3.15`, added 3.13/3.14 classifiers
2. `squish/__init__.py`: `__version__` 9.33.5 → 9.33.6
3. `Formula/squish.rb`: replaced live `pip install` with `virtualenv_install_with_resources`
   and 39 resource blocks covering all transitive deps (macOS arm64 / Python 3.13 wheels)
4. `.github/workflows/publish.yml`: added formula auto-update step after PyPI publish
5. `README.md`: updated install section with Python version guidance and macOS Tahoe note
