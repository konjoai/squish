#!/bin/zsh
# ============================================================
# squish-release.sh
# Full release pipeline for squish-ai
#
# Usage:
#   ./scripts/squish-release.sh <version>
#   ./scripts/squish-release.sh 9.33.5
#
# What it does:
#   1.  Validates environment and inputs
#   2.  Bumps version in pyproject.toml and squish/__init__.py
#   3.  Commits and pushes version bump
#   4.  Tags and pushes — triggers PyPI publish workflow
#   5.  Waits for PyPI to confirm the new version
#   6.  Fetches new sha256 and tarball URL from PyPI
#   7.  Updates both formula files — URL, sha256, removes old bottle block
#   8.  Ensures post_install block exists in both formulas
#   9.  Builds brew bottle via --build-bottle
#   10. Extracts bottle sha256 and rebuild number from output
#   11. Renames bottle file to single-dash format brew expects
#   12. Uploads bottle to GitHub release
#   13. Adds bottle block to both formulas
#   14. Commits and pushes both formula repos
#   15. Final end-to-end validation — pours bottle, checks version + doctor
# ============================================================

set -e

# ── Colors ────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
GRAY='\033[0;90m'
NC='\033[0m'

step()    { echo -e "\n${CYAN}▸${NC} ${WHITE}$1${NC}"; }
success() { echo -e "${GREEN}✓${NC} $1"; }
warn()    { echo -e "${YELLOW}⚠${NC} $1"; }
error()   { echo -e "${RED}✗${NC} $1"; exit 1; }
info()    { echo -e "${GRAY}  $1${NC}"; }

# ── Config ────────────────────────────────────────────────────
SQUISH_REPO="${SQUISH_REPO:-/Users/wscholl/squish}"
TAP_REPO="/opt/homebrew/Library/Taps/konjoai/homebrew-squish"
GITHUB_RELEASE_REPO="konjoai/squish"

# ── Validate input ────────────────────────────────────────────
VERSION="$1"
if [[ -z "$VERSION" ]]; then
  error "Usage: $0 <version>  e.g.  $0 9.33.5"
fi
VERSION="${VERSION#v}"  # strip leading v if provided

echo ""
echo -e "${WHITE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${WHITE}  squish release pipeline — v${VERSION}${NC}"
echo -e "${WHITE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

# ── Step 1: Validate environment ─────────────────────────────
step "Validating environment"

[[ -d "$SQUISH_REPO" ]]   || error "squish repo not found at $SQUISH_REPO"
[[ -d "$TAP_REPO" ]]      || error "tap repo not found at $TAP_REPO — run: brew tap konjoai/squish"
command -v gh      >/dev/null || error "gh CLI not found — brew install gh"
command -v brew    >/dev/null || error "brew not found"
command -v python3 >/dev/null || error "python3 not found"
command -v curl    >/dev/null || error "curl not found"

# Confirm we're on main and clean
cd "$SQUISH_REPO"
BRANCH=$(git rev-parse --abbrev-ref HEAD)
[[ "$BRANCH" == "main" ]] || error "Not on main branch (on $BRANCH) — switch to main first"
DIRTY=$(git status --porcelain)
[[ -z "$DIRTY" ]] || error "Working directory is dirty — commit or stash changes first"

# Confirm version doesn't already exist
git fetch --tags >/dev/null 2>&1
git tag | grep -q "^v${VERSION}$" && error "Tag v${VERSION} already exists"

success "Environment OK"

# ── Step 2: Bump version ─────────────────────────────────────
step "Bumping version to $VERSION"

CURRENT=$(grep '^version = ' pyproject.toml | sed 's/version = "//;s/"//')
info "Current: $CURRENT → New: $VERSION"

sed -i '' "s/^version = \"${CURRENT}\"/version = \"${VERSION}\"/" pyproject.toml
sed -i '' "s/__version__ = \"${CURRENT}\"/__version__ = \"${VERSION}\"/" squish/__init__.py

grep "^version" pyproject.toml | head -1
grep "__version__" squish/__init__.py | head -1
success "Version bumped"

# ── Step 3: Commit and push ───────────────────────────────────
step "Committing version bump"

git add pyproject.toml squish/__init__.py
git commit -m "chore: bump version to ${VERSION}"
git push
success "Pushed to main"

# ── Step 4: Tag and push ──────────────────────────────────────
step "Tagging v${VERSION} — triggers PyPI publish workflow"

git tag "v${VERSION}"
git push origin "v${VERSION}"
success "Tag pushed — https://github.com/${GITHUB_RELEASE_REPO}/actions"

# ── Step 5: Wait for PyPI ─────────────────────────────────────
step "Waiting for PyPI to confirm squish-ai==${VERSION}"

echo -n "  Polling PyPI"
MAX_WAIT=600
WAITED=0
until curl -s "https://pypi.org/pypi/squish-ai/${VERSION}/json" | \
  python3 -c "import sys,json; json.loads(sys.stdin.read().encode('utf-8','replace').decode('utf-8'))" \
  >/dev/null 2>&1; do
  echo -n "."
  sleep 10
  WAITED=$((WAITED + 10))
  [[ $WAITED -ge $MAX_WAIT ]] && error "PyPI publish timed out after ${MAX_WAIT}s — check GitHub Actions"
done
echo ""
success "squish-ai==${VERSION} is live on PyPI"

# ── Step 6: Get sha256 and URL from PyPI ─────────────────────
step "Fetching sha256 and tarball URL"

PYPI_DATA=$(curl -s "https://pypi.org/pypi/squish-ai/${VERSION}/json")

SHA256=$(echo "$PYPI_DATA" | python3 -c "
import sys, json
data = json.loads(sys.stdin.read().encode('utf-8','replace').decode('utf-8'))
for f in data['urls']:
    if f['packagetype'] == 'sdist':
        print(f['digests']['sha256'])
")

TARBALL_URL=$(echo "$PYPI_DATA" | python3 -c "
import sys, json
data = json.loads(sys.stdin.read().encode('utf-8','replace').decode('utf-8'))
for f in data['urls']:
    if f['packagetype'] == 'sdist':
        print(f['url'])
")

[[ -n "$SHA256" ]]      || error "Could not fetch sha256 from PyPI"
[[ -n "$TARBALL_URL" ]] || error "Could not fetch tarball URL from PyPI"

info "sha256: $SHA256"
info "url:    $TARBALL_URL"
success "Got release artifacts"

# ── Step 7: Update formula files ─────────────────────────────
step "Updating formula files — URL, sha256, remove old bottle block"

update_formula() {
  local formula_path="$1"

  # Update URL
  sed -i '' "s|url \"https://files.pythonhosted.org.*\"|url \"${TARBALL_URL}\"|" "$formula_path"

  # Update source sha256 — use explicit old value replacement
  OLD_SHA=$(grep 'sha256 "[a-f0-9]\{64\}"' "$formula_path" | head -1 | grep -o '"[a-f0-9]*"' | tr -d '"')
  sed -i '' "s/sha256 \"${OLD_SHA}\"/sha256 \"${SHA256}\"/" "$formula_path"

  # Remove old bottle block
  python3 -c "
import re, sys
path = '$formula_path'
content = open(path).read()
content = re.sub(r'\n\s*bottle do\n.*?end\n', '\n', content, flags=re.DOTALL)
open(path, 'w').write(content)
"
  info "Updated $(basename $(dirname $formula_path))/Formula/squish.rb"
}

update_formula "$SQUISH_REPO/Formula/squish.rb"
update_formula "$TAP_REPO/Formula/squish.rb"

grep "url\|sha256" "$SQUISH_REPO/Formula/squish.rb" | head -2
success "Formula files updated"

# ── Step 8: Ensure post_install block exists ─────────────────
step "Ensuring post_install block exists"

ensure_post_install() {
  local formula_path="$1"
  if ! grep -q "post_install" "$formula_path"; then
    python3 -c "
path = '$formula_path'
content = open(path).read()
content = content.replace(
    '  test do',
    '  def post_install\n    system libexec/\"bin/python3\", \"-c\", \"import squish\"\n  end\n\n  test do'
)
open(path, 'w').write(content)
print('  added post_install to', path)
"
  else
    info "post_install already present in $(basename $formula_path)"
  fi
}

ensure_post_install "$SQUISH_REPO/Formula/squish.rb"
ensure_post_install "$TAP_REPO/Formula/squish.rb"
success "post_install confirmed"

# ── Step 9: Build brew bottle ─────────────────────────────────
step "Building Homebrew bottle for v${VERSION}"

brew uninstall squish 2>/dev/null || true

# Build from tap directory so bottle file lands there
cd "$TAP_REPO"
brew install --build-bottle squish
brew bottle squish 2>&1 | tee /tmp/squish_bottle_output.txt

success "Bottle built"

# ── Step 10: Extract bottle metadata ─────────────────────────
step "Extracting bottle metadata"

# Bottle file is created in current directory (TAP_REPO)
BOTTLE_DOUBLE=$(ls "$TAP_REPO"/squish--${VERSION}.arm64_tahoe.bottle*.tar.gz 2>/dev/null | head -1)
[[ -n "$BOTTLE_DOUBLE" ]] || error "Bottle file not found in $TAP_REPO — check brew bottle output"

BOTTLE_SHA=$(grep "arm64_tahoe" /tmp/squish_bottle_output.txt | grep -o '"[a-f0-9]*"' | tr -d '"')
BOTTLE_REBUILD=$(grep "rebuild" /tmp/squish_bottle_output.txt | awk '{print $2}' || echo "")

# rebuild number — default to empty (no rebuild line means rebuild 0 / omitted)
if [[ -z "$BOTTLE_REBUILD" ]]; then
  BOTTLE_SINGLE="${TAP_REPO}/squish-${VERSION}.arm64_tahoe.bottle.tar.gz"
  BOTTLE_REBUILD_BLOCK=""
else
  BOTTLE_SINGLE="${TAP_REPO}/squish-${VERSION}.arm64_tahoe.bottle.${BOTTLE_REBUILD}.tar.gz"
  BOTTLE_REBUILD_BLOCK="\n    rebuild ${BOTTLE_REBUILD}"
fi

info "Bottle file: $(basename $BOTTLE_DOUBLE)"
info "Bottle SHA:  $BOTTLE_SHA"
[[ -n "$BOTTLE_REBUILD" ]] && info "Rebuild:     $BOTTLE_REBUILD"
success "Bottle metadata extracted"

# ── Step 11: Rename and upload bottle ────────────────────────
step "Uploading bottle to GitHub release v${VERSION}"

cp "$BOTTLE_DOUBLE" "$BOTTLE_SINGLE"
gh release upload "v${VERSION}" "$BOTTLE_SINGLE" --repo "$GITHUB_RELEASE_REPO"
success "Bottle uploaded"

# ── Step 12: Add bottle block to both formulas ───────────────
step "Adding bottle block to formulas"

if [[ -n "$BOTTLE_REBUILD" ]]; then
  BOTTLE_BLOCK="
  bottle do
    root_url \"https://github.com/${GITHUB_RELEASE_REPO}/releases/download/v${VERSION}\"
    rebuild ${BOTTLE_REBUILD}
    sha256 cellar: :any, arm64_tahoe: \"${BOTTLE_SHA}\"
  end"
else
  BOTTLE_BLOCK="
  bottle do
    root_url \"https://github.com/${GITHUB_RELEASE_REPO}/releases/download/v${VERSION}\"
    sha256 cellar: :any, arm64_tahoe: \"${BOTTLE_SHA}\"
  end"
fi

add_bottle_block() {
  local formula_path="$1"
  python3 -c "
import re
path = '$formula_path'
block = '''${BOTTLE_BLOCK}'''
content = open(path).read()
content = re.sub(
    r'(  sha256 \"${SHA256}\")',
    r'\1' + block,
    content,
    count=1
)
open(path, 'w').write(content)
print('  bottle block added to', path)
"
}

add_bottle_block "$SQUISH_REPO/Formula/squish.rb"
add_bottle_block "$TAP_REPO/Formula/squish.rb"

grep -A 6 "bottle do" "$SQUISH_REPO/Formula/squish.rb"
success "Bottle blocks added"

# ── Step 13: Commit and push both formula repos ───────────────
step "Committing formula updates"

cd "$SQUISH_REPO"
git add Formula/squish.rb
git commit -m "release: squish-ai ${VERSION} formula + bottle"
git push
success "konjoai/squish formula pushed"

cd "$TAP_REPO"
git add Formula/squish.rb
git commit -m "release: squish-ai ${VERSION} formula + bottle"
git push origin main
success "konjoai/homebrew-squish formula pushed"

# ── Step 14: End-to-end validation ───────────────────────────
step "Running end-to-end validation"

brew uninstall squish 2>/dev/null || true
brew install squish

INSTALLED=$(squish --version 2>/dev/null)
info "Installed: $INSTALLED"
[[ "$INSTALLED" == *"$VERSION"* ]] || error "Version mismatch — got '$INSTALLED', expected $VERSION"

# Check bottle was poured (not built from source)
brew info squish | grep -q "Poured from bottle" && \
  success "Poured from bottle" || \
  warn "May have built from source — check brew info squish"

# Doctor
squish doctor

# Rust extension
squish doctor 2>&1 | grep -q "squish_quant" && \
  success "Rust extension confirmed green" || \
  warn "Rust extension check — see doctor output above"

# ── Done ──────────────────────────────────────────────────────
echo ""
echo -e "${WHITE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}  squish v${VERSION} released successfully ✓${NC}"
echo -e "${WHITE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "  ${GRAY}PyPI    ${CYAN}https://pypi.org/project/squish-ai/${VERSION}/${NC}"
echo -e "  ${GRAY}GitHub  ${CYAN}https://github.com/${GITHUB_RELEASE_REPO}/releases/tag/v${VERSION}${NC}"
echo -e "  ${GRAY}HF      ${CYAN}https://huggingface.co/squishai${NC}"
echo ""
echo -e "  ${WHITE}Install${NC}  ${CYAN}brew install konjoai/squish/squish${NC}"
echo ""
