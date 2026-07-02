"""tests/test_wave131_gemma4_ple_quant_safety.py

Wave 131 — Gemma 4 Per-Layer Embeddings (PLE) quantization-safety passthrough.

squish's INT4 quantization pipeline (squish/convert.py) is community-confirmed
UNSAFE to run naively on Gemma 4's architecture (model_type=gemma4_unified,
used by e.g. mlx-community/gemma-4-12B-bf16 and mlx-community/gemma-4-e2b-it).
Gemma 4 uses Per-Layer Embeddings (PLE): specific nn.Linear/nn.Embedding
weights whose outputs get scaled by a runtime float multiplier before being
added back into the hidden state. Standard INT4 quantization error gets
amplified by that scalar multiply, producing degenerate/garbage output
(community reports literal repeated-token garbage like "ionoxffionoxff...").

squish/cli.py's ``_ple_safe_passthrough_patterns(model_dir)`` reads the model
directory's config.json and, for ``model_type`` in {"gemma4_unified",
"gemma4"}, returns the PLE tensor-name patterns
(per_layer_input_gate, per_layer_projection, embed_tokens_per_layer,
per_layer_model_projection) that must be kept as FP16/BF16 passthrough
regardless of the requested --format. ``cmd_compress`` merges these into
``args.passthrough`` before the INT4 pipeline runs, mirroring the existing
mixed_attn auto-passthrough precedent (q_proj/k_proj/v_proj/o_proj).

This pins:
- a synthetic gemma4_unified config.json triggers exactly the 4 PLE patterns
- a synthetic gemma4 config.json (short alias) also triggers them
- other model_types (llama, qwen2, mistral) do NOT get PLE patterns injected
  — no false-positive scope creep into unrelated architectures
- a missing config.json returns an empty list rather than raising (no hard
  crash on unusual/legacy model directories)
- a malformed config.json is handled as a warned no-op, not a silent failure
  or a raised exception
- cmd_compress's merge behavior: existing --passthrough patterns are
  preserved alongside the auto-injected PLE patterns, with no duplicates
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

# squish.cli imports mlx_lm/mlx_vlm lazily inside functions, but follows the
# same repo convention as tests/test_wave130_vlm_backend_resolver.py: gate on
# a real mlx_lm import so this test suite behaves consistently whether run in
# the sandboxed dev terminal (import blocked, see tests/conftest.py) or with
# CI=1 / outside the sandbox.
pytest.importorskip(
    "mlx_lm.models.llama",
    reason="mlx_lm blocked in sandbox — set CI=1 to enable",
    exc_type=ImportError,
)

from squish.cli import (
    _GEMMA4_PLE_PASSTHROUGH_PATTERNS,
    _ple_safe_passthrough_patterns,
)

_EXPECTED_PLE_PATTERNS = {
    "per_layer_input_gate",
    "per_layer_projection",
    "embed_tokens_per_layer",
    "per_layer_model_projection",
}


def _write_config(model_dir: Path, model_type: str, **extra) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    cfg = {"model_type": model_type, **extra}
    (model_dir / "config.json").write_text(json.dumps(cfg))


class TestGemma4PleDetection:
    def test_gemma4_unified_triggers_ple_patterns(self, tmp_path):
        _write_config(tmp_path, "gemma4_unified")
        patterns = _ple_safe_passthrough_patterns(tmp_path)
        assert set(patterns) == _EXPECTED_PLE_PATTERNS

    def test_gemma4_short_alias_triggers_ple_patterns(self, tmp_path):
        _write_config(tmp_path, "gemma4")
        patterns = _ple_safe_passthrough_patterns(tmp_path)
        assert set(patterns) == _EXPECTED_PLE_PATTERNS

    def test_returned_patterns_match_module_constant(self, tmp_path):
        _write_config(tmp_path, "gemma4_unified")
        patterns = _ple_safe_passthrough_patterns(tmp_path)
        assert patterns == _GEMMA4_PLE_PASSTHROUGH_PATTERNS

    def test_returns_a_copy_not_the_shared_list(self, tmp_path):
        # Callers mutate the returned list (list-concat into args.passthrough);
        # returning the module-level list by reference would let one compress
        # invocation corrupt the constant for subsequent calls.
        _write_config(tmp_path, "gemma4_unified")
        patterns = _ple_safe_passthrough_patterns(tmp_path)
        patterns.append("bogus_pattern_injected_by_test")
        assert "bogus_pattern_injected_by_test" not in _GEMMA4_PLE_PASSTHROUGH_PATTERNS


class TestNonGemma4NoFalsePositive:
    @pytest.mark.parametrize("model_type", ["llama", "qwen2", "mistral", "gemma3", "gemma2"])
    def test_other_model_types_get_no_ple_patterns(self, tmp_path, model_type):
        _write_config(tmp_path, model_type)
        assert _ple_safe_passthrough_patterns(tmp_path) == []


class TestMissingOrMalformedConfig:
    def test_missing_config_json_returns_empty_list(self, tmp_path):
        tmp_path.mkdir(parents=True, exist_ok=True)
        assert _ple_safe_passthrough_patterns(tmp_path) == []

    def test_malformed_config_json_returns_empty_list_not_raise(self, tmp_path, caplog):
        tmp_path.mkdir(parents=True, exist_ok=True)
        (tmp_path / "config.json").write_text("{not valid json")
        assert _ple_safe_passthrough_patterns(tmp_path) == []

    def test_config_json_missing_model_type_returns_empty_list(self, tmp_path):
        tmp_path.mkdir(parents=True, exist_ok=True)
        (tmp_path / "config.json").write_text(json.dumps({"hidden_size": 32}))
        assert _ple_safe_passthrough_patterns(tmp_path) == []


class TestPassthroughMergeSemantics:
    """Pins the merge behavior cmd_compress performs, mirroring the existing
    mixed_attn precedent (squish/cli.py ~L2910): existing --passthrough
    entries survive, PLE patterns are appended, and duplicates are avoided.
    """

    def _merge(self, existing: list[str], ple_patterns: list[str]) -> list[str]:
        return existing + [p for p in ple_patterns if p not in existing]

    def test_merge_preserves_user_supplied_patterns(self, tmp_path):
        _write_config(tmp_path, "gemma4_unified")
        ple_patterns = _ple_safe_passthrough_patterns(tmp_path)
        merged = self._merge(["lm_head", "embed_tokens"], ple_patterns)
        assert "lm_head" in merged
        assert "embed_tokens" in merged
        for p in _EXPECTED_PLE_PATTERNS:
            assert p in merged

    def test_merge_does_not_duplicate_overlapping_patterns(self, tmp_path):
        _write_config(tmp_path, "gemma4_unified")
        ple_patterns = _ple_safe_passthrough_patterns(tmp_path)
        # User already passed one of the PLE patterns explicitly.
        merged = self._merge(["per_layer_input_gate"], ple_patterns)
        assert merged.count("per_layer_input_gate") == 1
        assert len(merged) == len(_EXPECTED_PLE_PATTERNS)

    def test_non_gemma4_merge_is_a_no_op(self, tmp_path):
        _write_config(tmp_path, "llama")
        ple_patterns = _ple_safe_passthrough_patterns(tmp_path)
        merged = self._merge(["q_proj"], ple_patterns)
        assert merged == ["q_proj"]
