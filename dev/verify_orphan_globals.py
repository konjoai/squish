"""Verify the 31 orphaned `global` declarations from orphan_globals_scan.py.

For each orphaned global decl in main(), check:
1. Does the module-level variable declaration exist (= None)?
2. Is the variable read/written ANYWHERE in the file (outside the `global` stmt)?
3. Is it referenced in any other .py file under squish/?
"""
import pathlib
import re
import subprocess

ORPHANS = [
    "_ProductionProfiler",
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

src_path = pathlib.Path("squish/server.py")
src = src_path.read_text()
lines = src.splitlines()

root = pathlib.Path("squish")
all_py = list(root.rglob("*.py"))
all_srcs = {p: p.read_text() for p in all_py if p != src_path}

print(f"{'Variable':<35} {'server.py refs (excl global decl)':>5}  {'other files':>5}  verdict")
print("-" * 75)

for var in ORPHANS:
    # Count all occurrences in server.py
    total_refs = len(re.findall(re.escape(var), src))
    # Count `global _var` lines in server.py
    global_decl_count = len(re.findall(r'\bglobal\b[^\n]*\b' + re.escape(var) + r'\b', src))
    # Real references = total - global declarations
    real_refs = total_refs - global_decl_count

    # Check other files
    other_refs = sum(
        len(re.findall(re.escape(var), text))
        for text in all_srcs.values()
    )

    if real_refs == 0 and other_refs == 0:
        verdict = "DEAD — safe to remove global decl + module var"
    elif real_refs == 0:
        verdict = f"global decl dead in main(); used in other files ({other_refs} refs)"
    else:
        verdict = f"LIVE in server.py ({real_refs} refs beyond global decl)"

    print(f"{var:<35} {real_refs:>5}  {other_refs:>5}  {verdict}")
