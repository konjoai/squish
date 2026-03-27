"""
Wave 120 regression tests — Dead Global Purge.

Verifies that 188 dead module-level globals and their hot-path references
were fully removed from squish.server.
"""
import re
import pytest

SERVER_PATH = "squish/server.py"

# Spot-check a representative sample of the 188 purged globals.
_SAMPLE_DEAD_GLOBALS = [
    "_ada_gptq", "_akvq_cache", "_attention_store", "_auto_round",
    "_big_little_llm", "_bit_distiller", "_cache_blend", "_calm_exit",
    "_chain_of_draft", "_chunk_kv_manager", "_cla_config", "_contrastive_search",
    "_delta_zip_store", "_diffkv_policy_mgr", "_dora_adapter", "_eagle2_spec",
    "_efficient_qat", "_expert_choice", "_flash_attn3", "_flexgen_offload",
    "_fp8_act_quant", "_hadamard_quant", "_head_pruner", "_hgrn2",
    "_ia3_adapter", "_kvquant_cache", "_lade_decoder", "_layer_collapse",
    "_lean_kv_quant", "_llm_lingua2", "_loftq_config", "_lookahead_decode",
    "_magic_pig_v2", "_mamba2_ssm", "_marlin_gemm", "_medusa_heads",
    "_mirostat_sampler", "_mix_of_depths", "_moe_infinity", "_nsa_attn_config",
    "_orca_scheduler", "_owq_quantizer", "_pd_disaggregator", "_pipe_infer",
    "_preble_router", "_prm_beam_search", "_qmoe_compressor", "_razor_attn",
    "_rest_decoder", "_ring_attn_config", "_sarathi_scheduler", "_self_extend",
    "_sparse_gpt", "_spec_reason_orch", "_speculative_streamer", "_spqr_quantizer",
    "_tensor_parallel", "_token_entropy_pruner", "_typical_sampler", "_yarn_rope",
]

# False positives that must NOT be removed (live via globals() dict assignment).
_LIVE_GLOBALS = ["_lazy_expert", "_structured_sparsity"]


@pytest.fixture(scope="module")
def server_text():
    with open(SERVER_PATH) as f:
        return f.read()


@pytest.mark.parametrize("var", _SAMPLE_DEAD_GLOBALS)
def test_dead_global_module_decl_absent(server_text, var):
    """Module-level `_var = None` declaration must be gone."""
    pattern = rf'^{re.escape(var)}\s*=\s*None\b'
    assert not re.search(pattern, server_text, re.MULTILINE), (
        f"Dead global {var!r} still has a module-level None declaration"
    )


@pytest.mark.parametrize("var", _SAMPLE_DEAD_GLOBALS)
def test_dead_global_stmt_absent(server_text, var):
    """Standalone `global _var` declaration in main() must be gone."""
    pattern = rf'^\s+global {re.escape(var)}\s*$'
    assert not re.search(pattern, server_text, re.MULTILINE), (
        f"Dead global {var!r} still has a standalone global declaration"
    )


def test_pd_disaggregator_hot_path_absent(server_text):
    """All _pd_disaggregator hot-path usage must be gone."""
    assert "_pd_disaggregator" not in server_text


def test_speculative_streamer_hot_path_absent(server_text):
    """_speculative_streamer.reset() block must be gone."""
    assert "_speculative_streamer" not in server_text


def test_chunk_kv_manager_hot_path_absent(server_text):
    """_chunk_kv_manager.invalidate_reuse_cache() block must be gone."""
    assert "_chunk_kv_manager" not in server_text


def test_flash_attn3_status_row_absent(server_text):
    """flash-attn3 must not appear in the optimization status table."""
    assert "flash-attn3" not in server_text


@pytest.mark.parametrize("var", _LIVE_GLOBALS)
def test_live_globals_preserved(server_text, var):
    """False-positive globals (set via globals() dict) must still be present."""
    assert var in server_text, f"Live global {var!r} was incorrectly deleted"


def test_server_line_count_under_threshold(server_text):
    """server.py must stay under 5000 lines after Wave 120 purge."""
    line_count = server_text.count('\n') + 1
    assert line_count < 5000, (
        f"server.py has {line_count} lines — expected < 5000 after Wave 120"
    )


def test_no_new_dead_globals(server_text):
    """Sentinel: catch any new always-None module-level globals with no assignment."""
    init_none = set(re.findall(r'^(_\w+)\s*=\s*None\b', server_text, re.MULTILINE))
    assigned_non_none = set()
    for var in init_none:
        if re.search(r'^\s+' + re.escape(var) + r'\s*=[^=\n]', server_text, re.MULTILINE):
            assigned_non_none.add(var)
        if re.search(r'globals\(\)\["' + re.escape(var) + r'"\]\s*=', server_text):
            assigned_non_none.add(var)
    dead = (init_none - assigned_non_none) - {"_lazy_expert", "_structured_sparsity"}
    assert not dead, (
        f"New dead globals detected — add to Wave 120 or document why they're needed:\n"
        + "\n".join(f"  {v}" for v in sorted(dead))
    )
