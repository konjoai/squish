#!/usr/bin/env python3
"""
dev/scripts/reorganize_subpackages.py

Reorganize squish/ flat layout into functional subpackages.
Run from repo root: python3 dev/scripts/reorganize_subpackages.py

Steps:
  1. Create subpackage __init__.py files
  2. git mv source files into subpackages
  3. git mv test files into subdirectories
  4. Update all import references across all .py files
  5. Update squish/__init__.py _LAZY_IMPORTS dict
"""
import os, re, subprocess, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent  # .../squish
SQUISH = REPO / "squish"
TESTS = REPO / "tests"

# ── 1. Subpackages to create ──
SUBPKGS = [
    "quant", "kv", "attention", "speculative", "token",
    "streaming", "lora", "moe", "grammar", "serving",
    "hardware", "context", "io",
]

# ── 2. Source file mapping: stem → subpackage ──
SRC_MAP = {
    # quant
    "adaptive_quantize": "quant", "awq": "quant", "awq_v2": "quant",
    "comm_vq": "quant", "compressed_loader": "quant", "dfloat11": "quant",
    "fp8_quant": "quant", "gptq_layer": "quant", "milo_quant": "quant",
    "mx_quant": "quant", "nf4_quant": "quant", "quant_aware": "quant",
    "quant_calib": "quant", "quantizer": "quant", "quip_sharp": "quant",
    "rans_codec": "quant", "spin_quant": "quant", "squeeze_llm": "quant",
    "svdq": "quant", "ternary_quant": "quant", "vptq": "quant",
    "zero_quant_v2": "quant",
    # kv
    "agent_kv": "kv", "cache_warmup": "kv", "cocktail_kv": "kv",
    "codec_kv": "kv", "context_cache": "kv", "diffkv": "kv",
    "hierarchical_kv": "kv", "kv_cache": "kv", "kv_compress": "kv",
    "kv_defrag": "kv", "kv_slab": "kv", "kvsharer": "kv",
    "kvtuner": "kv", "mix_kvq": "kv", "mixed_precision_kv": "kv",
    "paged_attention": "kv", "paged_kv": "kv", "paris_kv": "kv",
    "pm_kvq": "kv", "pq_cache": "kv", "prefix_pool": "kv",
    "radix_cache": "kv", "semantic_cache": "kv",
    "semantic_response_cache": "kv", "shadow_kv": "kv",
    "smallkv": "kv", "spe_cache": "kv", "tool_cache": "kv",
    # attention
    "cla": "attention", "cross_doc_attn": "attention",
    "dedupe_attn": "attention", "dual_chunk_attn": "attention",
    "duo_attention": "attention", "duo_decoding": "attention",
    "dynamic_ntk": "attention", "flash_attention": "attention",
    "flash_decode": "attention", "flash_mla": "attention",
    "flash_prefill": "attention", "gqa": "attention",
    "minference_patch": "attention", "morph_attn": "attention",
    "native_sparse_attn": "attention", "retention_attn": "attention",
    "rope_scaling": "attention", "sage_attention": "attention",
    "sage_attention2": "attention", "sliding_window_attn": "attention",
    "sparge_attn": "attention", "sparse_attn_index": "attention",
    "squeeze_attention": "attention", "yoco": "attention",
    # speculative
    "conf_spec": "speculative", "copy_spec": "speculative",
    "distil_spec": "speculative", "dovetail": "speculative",
    "eagle3": "speculative", "fr_spec": "speculative",
    "head_infer": "speculative", "hydra_spec": "speculative",
    "long_spec": "speculative", "medusa": "speculative",
    "mirror_sd": "speculative", "online_sd": "speculative",
    "prompt_lookup": "speculative", "qspec": "speculative",
    "quant_spec": "speculative", "quant_spec_decode": "speculative",
    "rasd": "speculative", "spec_bench": "speculative",
    "spec_reason": "speculative", "specontext": "speculative",
    "speculative": "speculative", "sparse_spec": "speculative",
    "sparse_verify": "speculative", "sub_spec": "speculative",
    "swift": "speculative", "swiftspec": "speculative",
    "trail": "speculative", "tree_verifier": "speculative",
    # token
    "act_sparsity": "token", "c2t": "token", "forelen": "token",
    "gemfilter": "token", "ipw": "token", "knapspec": "token",
    "layer_fuse": "token", "layer_skip": "token",
    "layerwise_decode": "token", "lookahead_reasoning": "token",
    "token_budget_gate": "token", "token_healer": "token",
    "token_merging": "token", "token_swift": "token",
    # streaming
    "activation_offload": "streaming", "agile_io": "streaming",
    "chunked_prefill": "streaming", "long_context_chunk": "streaming",
    "seq_compact": "streaming", "seq_packing": "streaming",
    "stream_rag": "streaming", "streaming": "streaming",
    "streaming_chunk": "streaming", "streaming_sink": "streaming",
    # lora
    "lora_compose": "lora", "lora_inference": "lora",
    "lora_manager": "lora", "matryoshka_emb": "lora",
    "model_merge": "lora", "weight_sharing": "lora",
    # moe
    "mobile_moe": "moe", "pipo": "moe", "sparse_moe": "moe",
    "sparse_weight": "moe", "structured_prune": "moe",
    # grammar
    "grammar_cache": "grammar", "grammar_engine": "grammar",
    "schema_gen": "grammar", "schema_validator": "grammar",
    # serving
    "ada_serve": "serving", "adaptive_batcher": "serving",
    "adaptive_budget": "serving", "batch_embed": "serving",
    "budget_spec": "serving", "continuous_batching": "serving",
    "fault_tolerance": "serving", "health_check": "serving",
    "memory_governor": "serving", "model_pool": "serving",
    "ollama_compat": "serving", "power_monitor": "serving",
    "ppl_tracker": "serving", "rate_limiter": "serving",
    "remote": "serving", "request_coalesce": "serving",
    "robust_scheduler": "serving", "safety_layer": "serving",
    "scheduler": "serving", "tool_calling": "serving",
    # hardware
    "ane_profiler": "hardware", "fused_kernels": "hardware",
    "fused_rmsnorm": "hardware", "fused_sampler": "hardware",
    "layerwise_loader": "hardware", "metal_fusion": "hardware",
    "neuron_profile": "hardware", "neuron_router": "hardware",
    "parallel_sampler": "hardware", "pipeline_bubble": "hardware",
    "production_profiler": "hardware",
    # context
    "context_summarizer": "context", "contextual_rerank": "context",
    "cot_compress": "context", "delta_compress": "context",
    "lazy_llm": "context", "meta_reasoner": "context",
    "prompt_compressor": "context", "rag_prefetch": "context",
    # io
    "entropy": "io", "loader_utils": "io", "split_loader": "io",
}

# ── 3. Test file mapping ──
TEST_MAP = {
    # integration
    "test_bench_2bit.py": "integration",
    "test_bench_agent.py": "integration",
    "test_bench_cli.py": "integration",
    "test_bench_code.py": "integration",
    "test_bench_perf.py": "integration",
    "test_bench_quality.py": "integration",
    "test_bench_tool.py": "integration",
    "test_git_integration.py": "integration",
    "test_hardware_integration.py": "integration",
    "test_integration_status.py": "integration",
    "test_phase1_awq_kvcache.py": "integration",
    "test_phase2_demo_stack.py": "integration",
    "test_phase2_memory.py": "integration",
    "test_phase_a_cli.py": "integration",
    "test_phase_b_grammar.py": "integration",
    "test_phase_c_power.py": "integration",
    "test_phase_d_scheduler.py": "integration",
    "test_phase_e_cli.py": "integration",
    "test_phase_e_lora.py": "integration",
    "test_phase_f_backends.py": "integration",
    "test_server_wiring.py": "integration",
    "test_squish_lm_eval_unit.py": "integration",
    **{f"test_wave{n}_server_wiring.py": "integration" for n in range(12, 27)},
    # kv
    "test_agent_kv_unit.py": "kv",
    "test_cocktail_kv_unit.py": "kv",
    "test_codec_kv_unit.py": "kv",
    "test_diffkv_unit.py": "kv",
    "test_disk_kvcache.py": "kv",
    "test_h2o_kvcache_unit.py": "kv",
    "test_hadamard_kvcache_unit.py": "kv",
    "test_hierarchical_kv_unit.py": "kv",
    "test_kv_budget_broker.py": "kv",
    "test_kv_slab_unit.py": "kv",
    "test_kvcache_branches.py": "kv",
    "test_kvcache_coverage.py": "kv",
    "test_kvcache_extended.py": "kv",
    "test_kvsharer_unit.py": "kv",
    "test_kvtuner_unit.py": "kv",
    "test_paged_attention_unit.py": "kv",
    "test_paris_kv_unit.py": "kv",
    "test_pm_kvq_unit.py": "kv",
    "test_pq_cache_unit.py": "kv",
    "test_radix_cache_trie.py": "kv",
    "test_semantic_cache_unit.py": "kv",
    "test_shadow_kv_unit.py": "kv",
    "test_smallkv_unit.py": "kv",
    "test_spe_cache_unit.py": "kv",
    "test_tool_cache_unit.py": "kv",
    # attention
    "test_cla_unit.py": "attention",
    "test_duo_attention_unit.py": "attention",
    "test_duo_decoding_unit.py": "attention",
    "test_minference_patch_unit.py": "attention",
    "test_sage_attention2_unit.py": "attention",
    "test_sage_attention_unit.py": "attention",
    "test_sparge_attn_unit.py": "attention",
    "test_squeeze_attention_unit.py": "attention",
    "test_yoco_unit.py": "attention",
    # speculative
    "test_conf_spec_unit.py": "speculative",
    "test_copy_spec_unit.py": "speculative",
    "test_dovetail_unit.py": "speculative",
    "test_eagle3_unit.py": "speculative",
    "test_fr_spec_unit.py": "speculative",
    "test_head_infer_unit.py": "speculative",
    "test_long_spec_unit.py": "speculative",
    "test_medusa_spec_unit.py": "speculative",
    "test_mirror_sd_unit.py": "speculative",
    "test_online_sd_unit.py": "speculative",
    "test_prompt_lookup_unit.py": "speculative",
    "test_qspec_unit.py": "speculative",
    "test_quant_spec_unit.py": "speculative",
    "test_rasd_unit.py": "speculative",
    "test_sparse_spec_unit.py": "speculative",
    "test_sparse_verify_unit.py": "speculative",
    "test_spec_bench_unit.py": "speculative",
    "test_spec_reason_unit.py": "speculative",
    "test_specontext_unit.py": "speculative",
    "test_speculative_helpers.py": "speculative",
    "test_sub_spec_unit.py": "speculative",
    "test_swift_unit.py": "speculative",
    "test_swiftspec_unit.py": "speculative",
    "test_trail_unit.py": "speculative",
    # token
    "test_act_sparsity_emit_profile_unit.py": "token",
    "test_c2t_unit.py": "token",
    "test_forelen_unit.py": "token",
    "test_gemfilter_unit.py": "token",
    "test_ipw_unit.py": "token",
    "test_knapspec_unit.py": "token",
    "test_layer_skip_unit.py": "token",
    "test_lookahead_reasoning_unit.py": "token",
    "test_token_merging_unit.py": "token",
    "test_token_swift_unit.py": "token",
    # streaming
    "test_agile_io_unit.py": "streaming",
    "test_chunked_prefill.py": "streaming",
    "test_seq_packing_unit.py": "streaming",
    "test_streaming_sink_unit.py": "streaming",
    # quant
    "test_awq_extended.py": "quant",
    "test_awq_unit.py": "quant",
    "test_comm_vq_unit.py": "quant",
    "test_compressed_loader_unit.py": "quant",
    "test_compression_pipeline.py": "quant",
    "test_dfloat11_unit.py": "quant",
    "test_int4_loader.py": "quant",
    "test_milo_unit.py": "quant",
    "test_nf4_unit.py": "quant",
    "test_quantizer_extended.py": "quant",
    "test_quantizer_extras.py": "quant",
    "test_quantizer_full.py": "quant",
    "test_quip_unit.py": "quant",
    "test_spin_quant_unit.py": "quant",
    "test_squeeze_llm_unit.py": "quant",
    "test_svdq_unit.py": "quant",
    "test_vptq_unit.py": "quant",
    # lora
    "test_dare_ties_unit.py": "lora",
    "test_lora_compose_unit.py": "lora",
    "test_matryoshka_emb_unit.py": "lora",
    "test_model_merge_unit.py": "lora",
    # moe
    "test_mobile_moe_unit.py": "moe",
    "test_pipo_unit.py": "moe",
    # grammar
    "test_fsm_gamma_unit.py": "grammar",
    "test_grammar_cache_unit.py": "grammar",
    "test_grammar_domino_dccd_unit.py": "grammar",
    "test_grammar_phase15_unit.py": "grammar",
    "test_schema_gen_unit.py": "grammar",
    # serving
    "test_ada_serve_unit.py": "serving",
    "test_adaptive_budget_unit.py": "serving",
    "test_agent_preset_unit.py": "serving",
    "test_continuous_batching_unit.py": "serving",
    "test_fault_tolerance_unit.py": "serving",
    "test_memory_governor_unit.py": "serving",
    "test_ollama_unit.py": "serving",
    "test_orca_scheduler_unit.py": "serving",
    "test_ppl_tracker_unit.py": "serving",
    "test_remote_ext.py": "serving",
    "test_remote_unit.py": "serving",
    "test_robust_scheduler_unit.py": "serving",
    "test_scheduler_bucket_argus_unit.py": "serving",
    "test_scheduler_extended.py": "serving",
    "test_scheduler_helpers.py": "serving",
    "test_server_display.py": "serving",
    "test_server_extended.py": "serving",
    "test_server_unit.py": "serving",
    "test_tool_calling_ext.py": "serving",
    "test_tool_calling_phase15_unit.py": "serving",
    # hardware
    "test_ane_profiler_unit.py": "hardware",
    "test_compiled_ffn_unit.py": "hardware",
    "test_fused_kernels_unit.py": "hardware",
    "test_layerwise_unit.py": "hardware",
    "test_metal_fusion_unit.py": "hardware",
    "test_neuron_profile_unit.py": "hardware",
    "test_neuron_router_unit.py": "hardware",
    # context
    "test_context_summarizer_unit.py": "context",
    "test_lazy_llm_unit.py": "context",
    "test_meta_reasoner_unit.py": "context",
    "test_prompt_compressor_unit.py": "context",
    # io
    "test_entropy_branches.py": "io",
    "test_entropy_brotli.py": "io",
    "test_entropy_extended.py": "io",
    "test_entropy_unit.py": "io",
    "test_loader_utils_extended.py": "io",
    "test_loader_utils_extras.py": "io",
    "test_loader_utils_unit.py": "io",
    "test_rans_codec_unit.py": "io",
    "test_split_loader_unit.py": "io",
}


def main():
    print(f"Repo root : {REPO}")
    print(f"squish/   : {SQUISH}")
    print(f"tests/    : {TESTS}")
    print()

    # ── Step 1: Create subpackage directories and __init__.py files ──
    print("=== Step 1: Creating subpackage __init__.py files ===")
    for pkg in SUBPKGS:
        pkg_dir = SQUISH / pkg
        pkg_dir.mkdir(exist_ok=True)
        init = pkg_dir / "__init__.py"
        if not init.exists():
            init.write_text('"""squish.{} subpackage."""\n'.format(pkg))
            print(f"  created squish/{pkg}/__init__.py")
        else:
            print(f"  exists  squish/{pkg}/__init__.py")
    print()

    # ── Step 2: Create test subdirectory conftest.py shims ──
    print("=== Step 2: Creating test subdirectory conftest.py shims ===")
    for subdir in sorted(set(TEST_MAP.values())):
        d = TESTS / subdir
        d.mkdir(exist_ok=True)
        c = d / "conftest.py"
        if not c.exists():
            c.write_text("")
            print(f"  created tests/{subdir}/conftest.py")
        else:
            print(f"  exists  tests/{subdir}/conftest.py")
    print()

    # ── Step 3: git mv source files ──
    print("=== Step 3: Moving source files (git mv) ===")
    src_moved = 0
    src_skipped = 0
    for stem, pkg in SRC_MAP.items():
        src = SQUISH / f"{stem}.py"
        dst = SQUISH / pkg / f"{stem}.py"
        if src.exists() and not dst.exists():
            subprocess.run(
                ["git", "mv", str(src), str(dst)],
                cwd=REPO,
                check=True,
            )
            print(f"  squish/{stem}.py  →  squish/{pkg}/{stem}.py")
            src_moved += 1
        elif dst.exists():
            print(f"  skip (dst exists): squish/{pkg}/{stem}.py")
            src_skipped += 1
        else:
            print(f"  skip (src missing): squish/{stem}.py")
            src_skipped += 1
    print(f"\n  Moved {src_moved} source files, skipped {src_skipped}")
    print()

    # ── Step 4: git mv test files ──
    print("=== Step 4: Moving test files (git mv) ===")
    test_moved = 0
    test_skipped = 0
    for fname, subdir in TEST_MAP.items():
        src = TESTS / fname
        dst = TESTS / subdir / fname
        if src.exists() and not dst.exists():
            subprocess.run(
                ["git", "mv", str(src), str(dst)],
                cwd=REPO,
                check=True,
            )
            print(f"  tests/{fname}  →  tests/{subdir}/{fname}")
            test_moved += 1
        elif dst.exists():
            print(f"  skip (dst exists): tests/{subdir}/{fname}")
            test_skipped += 1
        else:
            print(f"  skip (src missing): tests/{fname}")
            test_skipped += 1
    print(f"\n  Moved {test_moved} test files, skipped {test_skipped}")
    print()

    # ── Step 5: Build IMPORT_SUBS substitution map ──
    print("=== Step 5: Building import substitution map ===")
    IMPORT_SUBS = {}
    for stem, pkg in SRC_MAP.items():
        old = f"squish.{stem}"
        new = f"squish.{pkg}.{stem}"
        IMPORT_SUBS[old] = new

    # Sort longest first to avoid partial substitution
    # e.g. "squish.swiftspec" must be replaced before "squish.swift"
    # e.g. "squish.sage_attention2" before "squish.sage_attention"
    # e.g. "squish.quant_spec_decode" before "squish.quant_spec"
    sorted_subs = sorted(IMPORT_SUBS.items(), key=lambda x: -len(x[0]))
    print(f"  {len(sorted_subs)} substitution rules built")
    print()

    # ── Step 6: Apply substitutions to all .py files ──
    print("=== Step 6: Updating import references across all .py files ===")
    all_py = list(SQUISH.rglob("*.py")) + list(TESTS.rglob("*.py"))
    changed = 0
    for py_file in all_py:
        try:
            text = py_file.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError) as exc:
            print(f"  WARNING: could not read {py_file.relative_to(REPO)}: {exc}")
            continue
        new_text = text
        for old, new in sorted_subs:
            new_text = new_text.replace(old, new)
        if new_text != text:
            py_file.write_text(new_text, encoding="utf-8")
            changed += 1
            print(f"  updated imports: {py_file.relative_to(REPO)}")
    print(f"\n  Updated imports in {changed} files")
    print()

    # ── Step 7: Smoke-check — verify no stale flat references remain ──
    print("=== Step 7: Smoke-checking for un-updated references ===")
    problematic = []
    for py_file in SQUISH.rglob("*.py"):
        try:
            text = py_file.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for stem, pkg in SRC_MAP.items():
            flat_ref = f"squish.{stem}"
            new_ref = f"squish.{pkg}.{stem}"
            # A stale reference is one that contains the flat form but NOT the
            # new form at the same position (i.e. the flat form is still bare).
            # We look for the flat form that is NOT already part of the new form.
            if flat_ref in text:
                # Check whether every occurrence of flat_ref is already the
                # prefix of the new_ref (i.e. "squish.quant.awq" contains
                # "squish.awq" only if pkg == stem, which never happens here).
                # Simpler: if flat_ref still appears and new_ref does not replace
                # all of them, flag it.
                import re as _re
                # Match flat_ref not immediately followed by ".<pkg>." suffix
                # which would indicate the substitution already happened.
                suffix = f".{stem}"
                # new_ref == f"squish.{pkg}.{stem}", flat_ref == f"squish.{stem}"
                # If new_ref contains flat_ref as a substring we need to be careful.
                # Since pkg != stem always, new_ref never contains flat_ref as
                # a standalone token, so a simple count comparison works.
                count_flat = text.count(flat_ref)
                count_new = text.count(new_ref)
                if count_flat > count_new:
                    problematic.append((py_file, stem))

    if problematic:
        print(f"\n  WARNING: {len(problematic)} possible un-updated references:")
        for f, s in problematic[:20]:
            print(f"    {f.relative_to(REPO)}: squish.{s}")
        if len(problematic) > 20:
            print(f"    ... and {len(problematic) - 20} more")
    else:
        print("\n  All import references updated successfully.")

    print()
    print("=== Done ===")
    print(f"  Source files moved : {src_moved}")
    print(f"  Test files moved   : {test_moved}")
    print(f"  Import files patched: {changed}")


if __name__ == "__main__":
    main()
