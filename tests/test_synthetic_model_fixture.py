"""
tests/test_synthetic_model_fixture.py

Tests that validate the synthetic model fixture at
tests/fixtures/synthetic_model/ and the generation script.

These tests do NOT require a running server, MLX Metal GPU, or real model
weights.  They verify:
  - The fixture directory exists and contains the required files
  - config.json is valid and has the expected Qwen2 fields
  - model.safetensors is a valid safetensors v1 file with correct metadata
  - The fixture can be used as a model directory by squish helpers that
    only inspect the filesystem structure
"""
from __future__ import annotations

import json
import pathlib
import struct

import pytest

_FIXTURE_DIR = pathlib.Path(__file__).parent / "fixtures" / "synthetic_model"
_GENERATE_SCRIPT = (
    pathlib.Path(__file__).parent.parent / "dev" / "scripts" / "make_synthetic_model.py"
)


# ── Fixture presence ───────────────────────────────────────────────────────────

class TestFixtureFilesExist:
    def test_fixture_dir_exists(self):
        assert _FIXTURE_DIR.is_dir(), f"Fixture dir missing: {_FIXTURE_DIR}\nRun: python dev/scripts/make_synthetic_model.py"

    def test_config_json_exists(self):
        assert (_FIXTURE_DIR / "config.json").is_file()

    def test_safetensors_exists(self):
        assert (_FIXTURE_DIR / "model.safetensors").is_file()

    def test_tokenizer_config_exists(self):
        assert (_FIXTURE_DIR / "tokenizer_config.json").is_file()

    def test_generation_script_exists(self):
        assert _GENERATE_SCRIPT.is_file()


# ── config.json validity ───────────────────────────────────────────────────────

class TestConfigJson:
    @pytest.fixture(autouse=True)
    def _load(self):
        if not (_FIXTURE_DIR / "config.json").is_file():
            pytest.skip("fixture not generated")
        self.config = json.loads((_FIXTURE_DIR / "config.json").read_text())

    def test_model_type_is_qwen2(self):
        assert self.config["model_type"] == "qwen2"

    def test_has_hidden_size(self):
        assert isinstance(self.config["hidden_size"], int)
        assert self.config["hidden_size"] > 0

    def test_has_two_layers(self):
        assert self.config["num_hidden_layers"] == 2

    def test_has_vocab_size(self):
        assert self.config["vocab_size"] == 256

    def test_num_kv_heads_divides_attention_heads(self):
        assert self.config["num_attention_heads"] % self.config["num_key_value_heads"] == 0

    def test_tie_word_embeddings_true(self):
        # Fixture uses tied embeddings → no separate lm_head.weight in file
        assert self.config["tie_word_embeddings"] is True

    def test_intermediate_size_positive(self):
        assert self.config["intermediate_size"] > 0


# ── safetensors format ─────────────────────────────────────────────────────────

class TestSafetensors:
    @pytest.fixture(autouse=True)
    def _load(self):
        path = _FIXTURE_DIR / "model.safetensors"
        if not path.is_file():
            pytest.skip("fixture not generated")
        self.path = path
        raw = path.read_bytes()
        header_len = struct.unpack_from("<Q", raw, 0)[0]
        header_bytes = raw[8 : 8 + header_len]
        self.header = json.loads(header_bytes.decode("utf-8").strip())
        self.tensor_names = [k for k in self.header if k != "__metadata__"]

    def test_header_is_valid_json(self):
        assert isinstance(self.header, dict)

    def test_has_embed_tokens(self):
        assert "model.embed_tokens.weight" in self.tensor_names

    def test_has_layer_zero_q_proj(self):
        assert "model.layers.0.self_attn.q_proj.weight" in self.tensor_names

    def test_has_layer_one_mlp_down(self):
        assert "model.layers.1.mlp.down_proj.weight" in self.tensor_names

    def test_has_final_norm(self):
        assert "model.norm.weight" in self.tensor_names

    def test_all_tensors_are_float32(self):
        for name in self.tensor_names:
            assert self.header[name]["dtype"] == "F32", f"{name} is not F32"

    def test_tensor_shapes_are_positive(self):
        for name in self.tensor_names:
            shape = self.header[name]["shape"]
            assert all(d > 0 for d in shape), f"{name} has non-positive dimension"

    def test_embed_tokens_shape(self):
        shape = self.header["model.embed_tokens.weight"]["shape"]
        assert shape == [256, 64]  # [vocab_size, hidden_size]

    def test_q_proj_shape_consistent_with_config(self):
        config = json.loads((_FIXTURE_DIR / "config.json").read_text())
        expected = [config["hidden_size"], config["hidden_size"]]
        actual = self.header["model.layers.0.self_attn.q_proj.weight"]["shape"]
        assert actual == expected

    def test_data_offsets_are_monotone(self):
        prev_end = 0
        for name in self.tensor_names:
            start, end = self.header[name]["data_offsets"]
            assert start >= prev_end, f"{name}: data_offsets overlap"
            assert end > start
            prev_end = end

    def test_file_size_matches_last_offset(self):
        header_len_bytes = struct.unpack_from("<Q", self.path.read_bytes(), 0)[0]
        last_end = max(
            self.header[n]["data_offsets"][1] for n in self.tensor_names
        )
        expected_size = 8 + header_len_bytes + last_end
        assert self.path.stat().st_size == expected_size

    def test_tensor_count_matches_expected(self):
        # 2 layers × 12 tensors each + embed_tokens + model.norm = 26
        assert len(self.tensor_names) == 26


# ── tokenizer_config.json ──────────────────────────────────────────────────────

class TestTokenizerConfig:
    @pytest.fixture(autouse=True)
    def _load(self):
        path = _FIXTURE_DIR / "tokenizer_config.json"
        if not path.is_file():
            pytest.skip("fixture not generated")
        self.tok_config = json.loads(path.read_text())

    def test_has_model_type(self):
        assert "model_type" in self.tok_config

    def test_has_eos_token(self):
        assert "eos_token" in self.tok_config

    def test_model_type_is_qwen2(self):
        assert self.tok_config["model_type"] == "qwen2"
