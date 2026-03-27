"""Wave 121 regression: 19 dead argparse flags purged from server.py.

Each deleted flag is verified absent (no add_argument call).
The 3 intentionally-preserved --no-X aliases are verified still present.
Live consumer flags and the overall flag reduction contract are also asserted.
"""
import pytest

_SERVER = None


def _src():
    global _SERVER
    if _SERVER is None:
        with open("squish/server.py") as f:
            _SERVER = f.read()
    return _SERVER


# ── 19 deleted flags ──────────────────────────────────────────────────────────

_DELETED_FLAGS = [
    "ada-serve-slo",
    "agent-kv-sink",
    "agent-kv-window",
    "chunk-kv-budget",
    "chunk-kv-size",
    "conf-spec-high-gate",
    "conf-spec-low-gate",
    "fast-warmup",
    "kv-share-every",
    "kv-slab-pages",
    "mtp-heads",
    "no-metal-warmup",
    "paris-kv-centroids",
    "quip",
    "retrieval-attention",
    "retrieval-hot-window",
    "retrieval-top-k",
    "seq-packing-budget",
    "streaming-sink-size",
]


@pytest.mark.parametrize("flag", _DELETED_FLAGS)
def test_deleted_flag_absent(flag):
    """Confirm no add_argument declaration remains for each purged flag."""
    assert f'"--{flag}"' not in _src(), (
        f'--{flag} still has an add_argument entry in server.py'
    )


# ── Orphaned section comment verification ─────────────────────────────────────

def test_phase13a_comment_removed():
    """The Phase 13A section comment (only had dead flags) must be gone."""
    assert "Phase 13A: Asymmetric INT2 KV Cache" not in _src()


def test_phase2_retrieval_comment_removed():
    """The inline '# Phase 2 retrieval attention' comment must be gone."""
    assert "# Phase 2 retrieval attention" not in _src()


# ── 3 preserved --no-X aliases (false positives) ─────────────────────────────

_PRESERVED_NEGATION_FLAGS = [
    "no-babbling-suppression",
    "no-fast-gelu",
    "no-semantic-cache",
]


@pytest.mark.parametrize("flag", _PRESERVED_NEGATION_FLAGS)
def test_negation_alias_preserved(flag):
    """These --no-X flags alias to a consumed dest; they must remain."""
    assert f'"--{flag}"' in _src(), (
        f'--{flag} was incorrectly deleted (it aliases a consumed dest)'
    )


# ── Live consumer flags must remain ──────────────────────────────────────────

_LIVE_FLAGS = [
    "prompt-lookup",
    "prompt-lookup-n",
    "prompt-lookup-k",
    "no-fused-sampler",
    "no-cache-warmup",
    "kvtc",
    "kvtc-rank",
    "kvtc-bits",
    "metal-flash-attn",
    "deja-vu",
    "jacobi",
    "jacobi-n",
    "jacobi-variant",
    "layer-overlap",
    "layer-overlap-prefetch",
    "fused-qkv",
    "lora-adapter",
    "all-optimizations",
    "batch-scheduler",
    "babbling-suppression",
    "fast-gelu",
    "semantic-cache",
]


@pytest.mark.parametrize("flag", _LIVE_FLAGS)
def test_live_flag_preserved(flag):
    assert f'"--{flag}"' in _src() or f"'--{flag}'" in _src(), (
        f'Live flag --{flag} was incorrectly deleted'
    )


# ── Quantitative line-count contract ─────────────────────────────────────────

def test_line_count_under_4800():
    """Wave 121 must result in server.py under 4800 lines."""
    count = _src().count("\n") + 1
    assert count < 4800, f"server.py has {count} lines (expected < 4800 after Wave 121)"


def test_line_count_over_4700():
    """Sanity check: server.py should not have been accidentally truncated."""
    count = _src().count("\n") + 1
    assert count > 4700, f"server.py has {count} lines — looks truncated"


# ── Flag count contract ───────────────────────────────────────────────────────

def test_registered_flag_count():
    """After removing 19 dead flags, total add_argument count must be ≤ 90."""
    import re
    count = len(re.findall(r'add_argument\(\s*["\']--[a-z]', _src()))
    assert count <= 90, (
        f"Found {count} add_argument declarations — expected ≤ 90 after Wave 121"
    )


def test_no_new_dead_flags():
    """Sentinel: run the dead_flags_analysis script and assert ≤ 3 dead flags.

    The 3 allowed 'dead' flags are the preserved --no-X aliases that have
    dest= pointing to a consumed positive-flag attribute.
    """
    import subprocess
    result = subprocess.run(
        ["python3", "dev/dead_flags_analysis.py"],
        capture_output=True, text=True, cwd="."
    )
    # Find the count line
    for line in result.stdout.splitlines():
        if "Potentially unconsumed:" in line:
            count = int(line.split(":")[-1].strip())
            assert count <= 3, (
                f"dead_flags_analysis found {count} dead flags — expected ≤ 3 "
                f"(the 3 preserved --no-X aliases).\nOutput:\n{result.stdout}"
            )
            return
    # If the line was not found, the script output changed
    assert False, f"Could not parse dead_flags_analysis output:\n{result.stdout}"
