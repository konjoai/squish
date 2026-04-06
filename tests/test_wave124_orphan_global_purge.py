"""Wave 124 — orphaned global declarations purge in main() (-8 lines).

30 variables were declared `global` inside main() but never assigned anywhere
in the codebase. Their declarations have been removed. _ProductionProfiler is
NOT in this list — it has 4 live read references in server.py and is kept.
"""
import pathlib
import re

SERVER = pathlib.Path(__file__).parent.parent / "squish" / "server.py"
SRC = SERVER.read_text(encoding="utf-8")
LINES = SRC.splitlines()

DEAD_VARS = [
    "_seq_packer",
    "_ada_serve_scheduler",
    "_conf_spec_verifier",
    "_kvsharer_map",
    "_kv_slab_allocator",
    "_paris_kv_codebook",
    "_streaming_sink_cache",
    "_diffkv_policy_mgr",
    "_smallkv_cache",
    "_lookahead_engine",
    "_spec_reason_orch",
    "_sage_attn_kernel",
    "_sage_attn2_kernel",
    "_sparge_engine",
    "_squeeze_cache",
    "_yoco_config",
    "_cla_config",
    "_kvtuner_config",
    "_robust_sched",
    "_gemfilter_config",
    "_svdq_config",
    "_sparse_spec_config",
    "_sparse_verify_config",
    "_trail_config",
    "_specontext_config",
    "_forelen_config",
    "_ipw_config",
    "_layer_skip_config",
    "_long_spec_config",
    "_fr_spec_config",
]


def test_dead_globals_absent():
    """All 30 dead variable names must be completely absent from server.py."""
    for var in DEAD_VARS:
        assert var not in SRC, f"{var} still present in server.py (should be fully removed)"


def test_prompt_lookup_decoder_global_preserved():
    """`global _prompt_lookup_decoder` must still be present — it IS assigned in main()."""
    assert "global _prompt_lookup_decoder" in SRC


def test_production_profiler_global_preserved():
    """`_ProductionProfiler` must still be present — 4 live references."""
    assert "_ProductionProfiler" in SRC


def test_no_orphan_global_block():
    """The 9-line block of orphaned global declarations must be gone."""
    assert "global _seq_packer, _ada_serve_scheduler" not in SRC
    assert "global _conf_spec_verifier, _kvsharer_map, _kv_slab_allocator" not in SRC
    assert "global _paris_kv_codebook, _streaming_sink_cache" not in SRC


def test_line_count():
    """server.py must be ≤ 4743 lines (Wave 124 target was 4713; squash routing added
    ~30 lines after that wave — those additions are exempt from purge targets)."""
    count = len(LINES)
    assert count <= 4743, (
        f"Expected ≤ 4743 lines (squash-routing-adjusted), got {count}"
    )
    assert count > 4600, f"Sanity floor: expected > 4600 lines; got {count}"
