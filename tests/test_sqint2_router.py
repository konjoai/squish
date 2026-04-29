"""tests/test_sqint2_router.py — Unit tests for W103.3 MixedPrecisionRouter.

Covers:
  - Constructor validation (n_layers ≥ 1)
  - boundary_layers property: correct set for all edge-case n_layers values
  - format_for:
      * boundary-layer projections → INT4 (all projection types)
      * non-boundary attention projections (q/k/v/o) → INT3
      * non-boundary MLP gate_proj / up_proj → SQINT2
      * non-boundary down_proj → INT4 (conservative output side)
      * non-.weight tensors (biases) → None
      * tensors without layer index (embed_tokens, lm_head, global norm) → None
      * norms inside transformer blocks → INT4 (shape-check left to caller)
      * out-of-range layer_idx → INT4 (defensive)
      * non-standard naming (GPT-2 h.N, bracket notation) → correct routing
  - summary(): counts per format match expected tensor distribution
  - repr()
  - Idempotency: same name always returns same format
  - Integration: MixedPrecisionRouter importable from squish.quant.quantizer
  - Module count: squish/ stays ≤ 125 (W103.3 adds no new squish/ file)
"""

from __future__ import annotations

from pathlib import Path

import pytest

from squish.quant.quantizer import MixedPrecisionRouter


# ── helpers ──────────────────────────────────────────────────────────────────

def _layer(layer_idx: int, comp: str, proj: str) -> str:
    """Build a canonical Llama/Qwen tensor name."""
    return f"model.layers.{layer_idx}.{comp}.{proj}.weight"


def _attn(layer_idx: int, proj: str) -> str:
    return _layer(layer_idx, "self_attn", proj)


def _mlp(layer_idx: int, proj: str) -> str:
    return _layer(layer_idx, "mlp", proj)


# ── 1. Constructor validation ─────────────────────────────────────────────────


class TestConstructor:
    def test_valid_single_layer(self):
        r = MixedPrecisionRouter(1)
        assert r.n_layers == 1

    def test_valid_32_layers(self):
        r = MixedPrecisionRouter(32)
        assert r.n_layers == 32

    def test_zero_layers_raises(self):
        with pytest.raises(ValueError, match="n_layers must be"):
            MixedPrecisionRouter(0)

    def test_negative_layers_raises(self):
        with pytest.raises(ValueError):
            MixedPrecisionRouter(-1)


# ── 2. boundary_layers property ───────────────────────────────────────────────


class TestBoundaryLayers:
    def test_32_layers(self):
        r = MixedPrecisionRouter(32)
        assert r.boundary_layers == frozenset({0, 1, 30, 31})

    def test_4_layers_all_boundary(self):
        # n_layers == 4: {0,1} ∪ {2,3} = all four
        r = MixedPrecisionRouter(4)
        assert r.boundary_layers == frozenset({0, 1, 2, 3})

    def test_3_layers_all_boundary(self):
        r = MixedPrecisionRouter(3)
        assert r.boundary_layers == frozenset({0, 1, 2})

    def test_2_layers_all_boundary(self):
        r = MixedPrecisionRouter(2)
        assert r.boundary_layers == frozenset({0, 1})

    def test_1_layer_all_boundary(self):
        r = MixedPrecisionRouter(1)
        assert r.boundary_layers == frozenset({0})

    def test_5_layers_four_boundary(self):
        # {0,1} ∪ {3,4} — layer 2 is the sole non-boundary
        r = MixedPrecisionRouter(5)
        assert r.boundary_layers == frozenset({0, 1, 3, 4})

    def test_28_layers_qwen2_5_7b(self):
        # Qwen2.5-7B has 28 decoder blocks
        r = MixedPrecisionRouter(28)
        assert r.boundary_layers == frozenset({0, 1, 26, 27})

    def test_boundary_is_frozenset(self):
        r = MixedPrecisionRouter(32)
        assert isinstance(r.boundary_layers, frozenset)

    def test_boundary_immutable(self):
        r = MixedPrecisionRouter(32)
        with pytest.raises((AttributeError, TypeError)):
            r.boundary_layers.add(5)  # frozenset.add raises AttributeError


# ── 3. format_for — boundary layers stay INT4 ─────────────────────────────────


class TestBoundaryLayersInt4:
    """Every tensor in a boundary layer must route to INT4 regardless of type."""

    @pytest.mark.parametrize("layer_idx", [0, 1, 30, 31])
    @pytest.mark.parametrize("proj", ["q_proj", "k_proj", "v_proj", "o_proj",
                                       "gate_proj", "up_proj", "down_proj"])
    def test_all_proj_in_boundary_are_int4(self, layer_idx, proj):
        r = MixedPrecisionRouter(32)
        comp = "self_attn" if proj in ("q_proj", "k_proj", "v_proj", "o_proj") else "mlp"
        name = _layer(layer_idx, comp, proj)
        assert r.format_for(name) == "int4", (
            f"Boundary layer {layer_idx} proj {proj} should be int4, not {r.format_for(name)!r}"
        )

    def test_first_layer_gate_proj_is_int4(self):
        r = MixedPrecisionRouter(32)
        assert r.format_for(_mlp(0, "gate_proj")) == "int4"

    def test_last_layer_up_proj_is_int4(self):
        r = MixedPrecisionRouter(32)
        assert r.format_for(_mlp(31, "up_proj")) == "int4"


# ── 4. format_for — SQINT2 for MLP gate_proj + up_proj ───────────────────────


class TestSQINT2Routing:
    @pytest.mark.parametrize("proj", ["gate_proj", "up_proj"])
    def test_mlp_ffn_projections_are_sqint2(self, proj):
        r = MixedPrecisionRouter(32)
        assert r.format_for(_mlp(5, proj)) == "sqint2"

    def test_sqint2_across_multiple_mid_layers(self):
        r = MixedPrecisionRouter(32)
        for layer_idx in range(2, 30):  # all non-boundary
            assert r.format_for(_mlp(layer_idx, "gate_proj")) == "sqint2"
            assert r.format_for(_mlp(layer_idx, "up_proj")) == "sqint2"

    def test_sqint2_with_different_model_prefix(self):
        # Some models use different prefix (language_model.model.layers.N)
        name = "language_model.model.layers.5.mlp.gate_proj.weight"
        r = MixedPrecisionRouter(32)
        assert r.format_for(name) == "sqint2"


# ── 5. format_for — INT3 for attention Q/K/V/O ───────────────────────────────


class TestINT3Routing:
    @pytest.mark.parametrize("proj", ["q_proj", "k_proj", "v_proj", "o_proj"])
    def test_attention_projections_are_int3(self, proj):
        r = MixedPrecisionRouter(32)
        assert r.format_for(_attn(5, proj)) == "int3"

    def test_int3_across_multiple_mid_layers(self):
        r = MixedPrecisionRouter(32)
        for layer_idx in range(2, 30):
            for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
                assert r.format_for(_attn(layer_idx, proj)) == "int3"


# ── 6. format_for — INT4 for down_proj and other fallbacks ───────────────────


class TestINT4Routing:
    def test_down_proj_is_int4(self):
        r = MixedPrecisionRouter(32)
        assert r.format_for(_mlp(5, "down_proj")) == "int4"

    def test_input_layernorm_inside_block_is_int4(self):
        r = MixedPrecisionRouter(32)
        assert r.format_for("model.layers.5.input_layernorm.weight") == "int4"

    def test_post_attention_layernorm_is_int4(self):
        r = MixedPrecisionRouter(32)
        assert r.format_for("model.layers.5.post_attention_layernorm.weight") == "int4"

    def test_q_norm_is_int4(self):
        # Qwen2/3 per-head RMS norm — not an attention projection matrix
        r = MixedPrecisionRouter(32)
        assert r.format_for("model.layers.5.self_attn.q_norm.weight") == "int4"

    def test_k_norm_is_int4(self):
        r = MixedPrecisionRouter(32)
        assert r.format_for("model.layers.5.self_attn.k_norm.weight") == "int4"

    def test_out_of_range_layer_idx_is_int4(self):
        r = MixedPrecisionRouter(32)  # valid indices 0-31
        assert r.format_for("model.layers.100.mlp.gate_proj.weight") == "int4"


# ── 7. format_for — None for non-quantisable tensors ─────────────────────────


class TestSkipRouting:
    def test_embed_tokens_is_none(self):
        r = MixedPrecisionRouter(32)
        assert r.format_for("model.embed_tokens.weight") is None

    def test_lm_head_is_none(self):
        r = MixedPrecisionRouter(32)
        assert r.format_for("lm_head.weight") is None

    def test_global_norm_is_none(self):
        r = MixedPrecisionRouter(32)
        assert r.format_for("model.norm.weight") is None

    def test_bias_is_none(self):
        r = MixedPrecisionRouter(32)
        assert r.format_for("model.layers.5.self_attn.q_proj.bias") is None

    def test_non_weight_tensor_is_none(self):
        r = MixedPrecisionRouter(32)
        assert r.format_for("model.layers.5.mlp.gate_proj.scale") is None

    def test_activation_is_none(self):
        r = MixedPrecisionRouter(32)
        assert r.format_for("model.layers.5.mlp.gate_proj.running_mean") is None

    def test_empty_string_is_none(self):
        r = MixedPrecisionRouter(32)
        assert r.format_for("") is None


# ── 8. format_for — alternate model naming conventions ───────────────────────


class TestNamingConventions:
    """Routing must work across HF / GPT-2 / bracket naming variants."""

    def test_gpt2_h_dot_n_mlp(self):
        # transformer.h.5.mlp.gate_proj.weight — GPT-2 style
        # c_fc / c_proj are GPT-2 specific names not in SQINT2_PROJS → int4 fallback
        r = MixedPrecisionRouter(12)
        assert r.format_for("transformer.h.5.mlp.c_fc.weight") == "int4"

    def test_gpt2_h_dot_n_attn(self):
        # GPT-2 uses "c_attn" and "c_proj" not q_proj/v_proj → int4
        r = MixedPrecisionRouter(12)
        assert r.format_for("transformer.h.5.attn.c_attn.weight") == "int4"

    def test_bracket_layer_notation(self):
        # Some exporters use layers[5] notation
        r = MixedPrecisionRouter(32)
        name = "model.layers[5].mlp.gate_proj.weight"
        result = r.format_for(name)
        # Layer index 5 is non-boundary, gate_proj → sqint2 OR int4 depending on regex
        # Not required to handle brackets — document result to prevent silent regressions.
        assert result in ("sqint2", "int4")

    def test_nested_prefix_model(self):
        # language_model.model.layers.5.mlp.gate_proj.weight
        r = MixedPrecisionRouter(32)
        assert r.format_for("language_model.model.layers.5.mlp.gate_proj.weight") == "sqint2"

    def test_layer_zero_non_boundary_tiny_model_n5(self):
        # n_layers=5: boundary={0,1,3,4}; layer 2 is the only non-boundary
        r = MixedPrecisionRouter(5)
        assert r.format_for(_mlp(2, "gate_proj")) == "sqint2"
        assert r.format_for(_attn(2, "q_proj")) == "int3"
        # All others should be int4 (boundary)
        assert r.format_for(_mlp(0, "gate_proj")) == "int4"
        assert r.format_for(_mlp(4, "up_proj")) == "int4"


# ── 9. small model: n_layers < 4 — all layers are boundary ───────────────────


class TestSmallModelAllBoundary:
    @pytest.mark.parametrize("n_layers", [1, 2, 3, 4])
    def test_gate_proj_is_int4_when_all_boundary(self, n_layers):
        r = MixedPrecisionRouter(n_layers)
        for layer_idx in range(n_layers):
            result = r.format_for(_mlp(layer_idx, "gate_proj"))
            assert result == "int4", (
                f"n_layers={n_layers}, layer {layer_idx}: expected int4 (all boundary), got {result!r}"
            )

    @pytest.mark.parametrize("n_layers", [1, 2, 3, 4])
    def test_attn_is_int4_when_all_boundary(self, n_layers):
        r = MixedPrecisionRouter(n_layers)
        for layer_idx in range(n_layers):
            assert r.format_for(_attn(layer_idx, "q_proj")) == "int4"


# ── 10. summary() ─────────────────────────────────────────────────────────────


class TestSummary:
    def test_summary_counts_all_formats(self):
        r = MixedPrecisionRouter(32)
        names = [
            _mlp(5, "gate_proj"),   # sqint2
            _mlp(5, "up_proj"),     # sqint2
            _attn(5, "q_proj"),     # int3
            _attn(5, "v_proj"),     # int3
            _mlp(5, "down_proj"),   # int4
            _mlp(0, "gate_proj"),   # int4 (boundary)
            "model.embed_tokens.weight",  # skip
        ]
        s = r.summary(names)
        assert s["sqint2"] == 2
        assert s["int3"] == 2
        assert s["int4"] == 2
        assert s["skip"] == 1

    def test_summary_returns_all_keys(self):
        r = MixedPrecisionRouter(32)
        s = r.summary([])
        assert set(s.keys()) == {"sqint2", "int3", "int4", "skip"}

    def test_summary_all_zeros_on_empty(self):
        r = MixedPrecisionRouter(32)
        s = r.summary([])
        assert all(v == 0 for v in s.values())

    def test_summary_realistic_qwen2_5_7b(self):
        """Qwen2.5-7B has 28 layers. Verify format distribution matches spec."""
        r = MixedPrecisionRouter(28)
        # Build realistic tensor name list: 28 layers × {q,k,v,o,gate,up,down} + embed/norm
        names = []
        for i in range(28):
            names += [
                _attn(i, "q_proj"), _attn(i, "k_proj"),
                _attn(i, "v_proj"), _attn(i, "o_proj"),
                _mlp(i, "gate_proj"), _mlp(i, "up_proj"), _mlp(i, "down_proj"),
            ]
        names += ["model.embed_tokens.weight", "lm_head.weight", "model.norm.weight"]
        s = r.summary(names)

        # Non-boundary layers: 28 - 4 = 24 layers have sqint2/int3 projections
        # 24 × gate_proj + 24 × up_proj = 48 sqint2
        assert s["sqint2"] == 48, f"Expected 48 sqint2, got {s['sqint2']}"
        # 24 × (q+k+v+o) = 96 int3
        assert s["int3"] == 96, f"Expected 96 int3, got {s['int3']}"
        # Boundary: 4 layers × 7 projections = 28 int4
        # Non-boundary down_proj: 24 int4
        # Total int4: 28 + 24 = 52
        assert s["int4"] == 52, f"Expected 52 int4, got {s['int4']}"
        # 3 skips (embed, lm_head, norm)
        assert s["skip"] == 3, f"Expected 3 skip, got {s['skip']}"


# ── 11. repr() ────────────────────────────────────────────────────────────────


class TestRepr:
    def test_repr_contains_n_layers(self):
        r = MixedPrecisionRouter(32)
        assert "32" in repr(r)

    def test_repr_contains_boundary(self):
        r = MixedPrecisionRouter(32)
        assert "boundary" in repr(r)

    def test_repr_is_string(self):
        r = MixedPrecisionRouter(8)
        assert isinstance(repr(r), str)


# ── 12. idempotency ───────────────────────────────────────────────────────────


class TestIdempotency:
    def test_same_name_always_same_result(self):
        r = MixedPrecisionRouter(32)
        name = _mlp(10, "gate_proj")
        results = {r.format_for(name) for _ in range(100)}
        assert len(results) == 1  # deterministic

    def test_different_routers_same_n_layers_agree(self):
        r1 = MixedPrecisionRouter(32)
        r2 = MixedPrecisionRouter(32)
        names = [
            _mlp(5, "gate_proj"), _attn(5, "q_proj"),
            _mlp(0, "up_proj"), "model.embed_tokens.weight",
        ]
        for name in names:
            assert r1.format_for(name) == r2.format_for(name)


# ── 13. integration: importable from public API ───────────────────────────────


class TestIntegration:
    def test_importable_from_quantizer(self):
        from squish.quant.quantizer import MixedPrecisionRouter as R
        assert R is MixedPrecisionRouter

    def test_module_count_unchanged(self):
        """W103.3 adds routing to quantizer.py (no new squish/ file). Count stays 84."""
        import squish
        root = Path(squish.__file__).parent
        py_files = [
            f for f in root.rglob("*.py")
            if "experimental" not in f.parts
            and "__pycache__" not in f.parts
        ]
        count = len(py_files)
        assert count == 84, (
            f"Module count = {count}, expected 84 after W103.3. "
            "W103.3 adds routing in-place to quantizer.py — no new squish/ files."
        )
        assert count <= 125

    def test_format_for_returns_valid_or_none(self):
        """format_for must only return one of the four documented values."""
        valid = {"sqint2", "int3", "int4", None}
        r = MixedPrecisionRouter(32)
        sample_names = [
            _mlp(5, "gate_proj"), _mlp(5, "up_proj"), _mlp(5, "down_proj"),
            _attn(5, "q_proj"), _attn(5, "k_proj"), _attn(5, "v_proj"), _attn(5, "o_proj"),
            _mlp(0, "gate_proj"), _attn(31, "q_proj"),
            "model.embed_tokens.weight", "lm_head.weight", "model.norm.weight",
            "model.layers.5.input_layernorm.weight",
            "model.layers.5.self_attn.q_norm.weight",
        ]
        for name in sample_names:
            result = r.format_for(name)
            assert result in valid, f"{name!r} → unexpected {result!r}"
