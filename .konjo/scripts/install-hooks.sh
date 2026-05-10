#!/usr/bin/env bash
# Konjo Quality Framework — Hook Installer
#
# Usage (from any repo root):
#   bash .konjo/scripts/install-hooks.sh
#
# What this does:
#   1. Installs .git/hooks/pre-commit pointing to .konjo/hooks/pre-commit
#   2. Detects repo type and checks for required tools
#   3. Prints a INSTALLED/MISSING checklist

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
GRN='\033[0;32m'; YEL='\033[0;33m'; RED='\033[0;31m'; RST='\033[0m'; BOLD='\033[1m'

ok()   { echo -e "${GRN}  ✓${RST} $1"; }
warn() { echo -e "${YEL}  ⚠${RST} $1"; }
err()  { echo -e "${RED}  ✗${RST} $1"; }

echo -e "${BOLD}Konjo Quality Framework — Install${RST}"
echo ""

# ── 1. Install pre-commit hook ────────────────────────────────────────────────
HOOK_SRC="$REPO_ROOT/.konjo/hooks/pre-commit"
HOOK_DST="$REPO_ROOT/.git/hooks/pre-commit"

if [[ ! -f "$HOOK_SRC" ]]; then
    err ".konjo/hooks/pre-commit not found — copy the .konjo/ directory from lopi first"
    exit 1
fi

chmod +x "$HOOK_SRC"

if [[ -L "$HOOK_DST" ]]; then
    rm "$HOOK_DST"
fi

ln -sf "../../.konjo/hooks/pre-commit" "$HOOK_DST"
ok "Installed .git/hooks/pre-commit → .konjo/hooks/pre-commit"

# ── 2. Detect repo type ───────────────────────────────────────────────────────
HAS_RUST=false; HAS_PYTHON=false; HAS_MOJO=false
[[ -f "$REPO_ROOT/Cargo.toml" ]] && HAS_RUST=true
{ [[ -f "$REPO_ROOT/pyproject.toml" ]] || [[ -f "$REPO_ROOT/requirements.txt" ]]; } && HAS_PYTHON=true
[[ -f "$REPO_ROOT/pixi.toml" ]] && HAS_MOJO=true

echo ""
echo -e "${BOLD}Repo type:${RST}"
$HAS_RUST    && ok "Rust" || true
$HAS_PYTHON  && ok "Python" || true
$HAS_MOJO    && ok "Mojo" || true

# ── 3. Check required tools ───────────────────────────────────────────────────
echo ""
echo -e "${BOLD}Tool availability:${RST}"
ALL_PRESENT=true

check_tool() {
    local cmd="$1" install_hint="$2"
    if command -v "$cmd" &>/dev/null; then
        ok "$cmd — $(command -v "$cmd")"
    else
        err "$cmd not found — $install_hint"
        ALL_PRESENT=false
    fi
}

# Universal
check_tool "python3"  "install Python 3.10+"
check_tool "git"      "install git"

# Rust
if $HAS_RUST; then
    check_tool "cargo"         "install Rust: curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
    check_tool "cargo-clippy"  "rustup component add clippy"

    if cargo install --list 2>/dev/null | grep -q "cargo-nextest"; then
        ok "cargo-nextest"
    else
        warn "cargo-nextest not installed — cargo install cargo-nextest (faster test runner)"
    fi
    if cargo install --list 2>/dev/null | grep -q "cargo-llvm-cov"; then
        ok "cargo-llvm-cov"
    else
        warn "cargo-llvm-cov not installed — cargo install cargo-llvm-cov (coverage)"
    fi
    if cargo install --list 2>/dev/null | grep -q "cargo-audit"; then
        ok "cargo-audit"
    else
        warn "cargo-audit not installed — cargo install cargo-audit (security)"
    fi
    if cargo install --list 2>/dev/null | grep -q "cargo-deny"; then
        ok "cargo-deny"
    else
        warn "cargo-deny not installed — cargo install cargo-deny (license/advisory gate)"
    fi
    if cargo install --list 2>/dev/null | grep -q "cargo-mutants"; then
        ok "cargo-mutants"
    else
        warn "cargo-mutants not installed — cargo install cargo-mutants (mutation testing)"
    fi
fi

# Python
if $HAS_PYTHON; then
    check_tool "ruff"     "pip install ruff"
    check_tool "mypy"     "pip install mypy (optional)"
    check_tool "vulture"  "pip install vulture (dead code)"
    check_tool "radon"    "pip install radon (complexity)"
fi

# Wall 3 review
if [[ -n "${ANTHROPIC_API_KEY:-}" ]]; then
    ok "ANTHROPIC_API_KEY — set ✓"
else
    warn "ANTHROPIC_API_KEY not set — Wall 3 (adversarial review) will soft-fail at pre-commit"
    warn "  For CI: add ANTHROPIC_API_KEY to GitHub Actions secrets"
    warn "  For local: export ANTHROPIC_API_KEY=your_key"
fi

if python3 -c "import anthropic" 2>/dev/null; then
    ok "anthropic Python SDK"
else
    warn "anthropic not installed — pip install anthropic (required for Wall 3)"
fi

# ── 4. Summary ────────────────────────────────────────────────────────────────
echo ""
if $ALL_PRESENT; then
    echo -e "${GRN}${BOLD}All required tools present. Framework installed.${RST}"
else
    echo -e "${YEL}${BOLD}Some tools missing — install them before the full gate runs.${RST}"
    echo -e "${YEL}Pre-commit hooks will warn on missing tools but not block.${RST}"
    echo -e "${YEL}CI gates WILL block on missing coverage/mutation tools.${RST}"
fi

echo ""
echo "Next steps:"
echo "  1. Add ANTHROPIC_API_KEY to GitHub Actions secrets"
echo "  2. Add .github/workflows/konjo-gate.yml to enable Wall 2"
echo "  3. Run: git commit --allow-empty -m 'test: verify konjo hooks' (to test hook)"
echo ""
echo "Docs: .konjo/KONJO_QUALITY_FRAMEWORK.md"
