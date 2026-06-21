"""Battle-test correctness gates for the perf optimizations landed in the last
~4 days (#155 / #154 / #153 / #152 / #59).

This drives the *same* battle-test cases as ``benchmarks/perf/battle_test.py`` in
quick mode and asserts the correctness contracts (no duplicated logic): every
case whose contract is lossless must emit byte-identical output, and every case
must reach a non-FAIL verdict. Heavy perf numbers live in the runner; here we
gate correctness so a regression can't merge silently (see squish CI memo — the
standalone Test matrix isn't a required check, so these guard the contracts).

Model-gated: skips cleanly without a local MLX model. Pure-logic cases
(PyramidKV budgets, fp16 passthrough property) run everywhere MLX imports.
"""
import importlib.util
import os
from pathlib import Path

import pytest

mx = pytest.importorskip("mlx.core")

_MODEL = os.environ.get("PL_TEST_MODEL", "") or os.path.expanduser(
    "~/models/Qwen2.5-1.5B-Instruct-int4")
_needs_model = pytest.mark.skipif(
    not os.path.isdir(_MODEL), reason=f"model not present: {_MODEL}")

_BT_PATH = Path(__file__).resolve().parents[1] / "benchmarks" / "perf" / "battle_test.py"


def _load_battle():
    """Import the benchmark runner module by path (it lives outside the package)."""
    spec = importlib.util.spec_from_file_location("battle_test", _BT_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── Pure-logic gates (no model) ───────────────────────────────────────────────

def test_pyramidkv_budget_contract():
    """#154 — PyramidKV keeps the mean budget, tapers, and respects the floor."""
    bt = _load_battle()
    c = bt.case_pyramidkv(None)
    assert c.verdict == "PASS", c.detail


def test_fp16_passthrough_contract(tmp_path):
    """#152 — npy-dir passthrough returns float16 and is bf16 bit-identical."""
    bt = _load_battle()
    c = bt.case_fp16_passthrough(tmp_path)
    assert c.lossless is True and c.verdict == "PASS", c.detail
    assert c.optimized == 2.0  # half the bytes/elem vs the old fp32 upcast


# ── Model-gated correctness gates (lossless contracts) ────────────────────────

@_needs_model
def test_lossless_contracts_hold():
    """#155 — prompt-lookup, async pipelining, and prefix reuse stay byte-identical
    to their baselines; the adaptive guard does not regress chat p95 >5%."""
    bt = _load_battle()
    from mlx_lm import load
    model, tok = load(_MODEL)
    eos = set(getattr(tok, "eos_token_ids", None) or [tok.eos_token_id])

    lossless_cases = {
        "prompt-lookup speculative": bt.case_prompt_lookup(model, tok, eos, 2),
        "async cooldown pipelining": bt.case_async_pipelining(model, tok, eos, 2),
        "prompt-prefix KV reuse": bt.case_prefix_reuse(model, tok, eos, 2),
    }
    for name, c in lossless_cases.items():
        assert c.lossless is True, f"{name} diverged from baseline: {c.detail}"
        assert c.verdict != "FAIL", f"{name} FAILED: {c.detail}"

    guard = bt.case_adaptive_guard(model, tok, eos, 3)
    assert guard.verdict != "FAIL", f"adaptive guard regressed chat: {guard.detail}"


@_needs_model
def test_batched_decode_runs_without_collapse():
    """#153 — the per-request KV-cached scheduler batches without erroring and
    aggregate throughput doesn't collapse vs single-stream. (The bit-identity
    contract is gated deterministically on CPU by test_scheduler_kv_cache.py;
    batching's throughput WIN is concurrency-dependent, so it's not asserted as a
    hard floor here.)"""
    bt = _load_battle()
    from mlx_lm import load
    model, tok = load(_MODEL)
    c = bt.case_batched_decode(model, tok, 2)
    assert c.verdict != "FAIL", c.detail
    assert c.speedup > 0.85, f"batched throughput collapsed: {c.detail}"
