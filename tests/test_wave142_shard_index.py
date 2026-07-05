"""tests/test_wave142_shard_index.py

Wave 142 — shard_index.py: parse a sharded model's weight_map into a
layer-aware tensor -> shard lookup.

Sequential AWQ calibration and per-shard streaming pull both need to know
which raw shard(s) a given decoder layer's weights live in, so a shard can
be evicted the moment no later layer needs it. This pins the parsing and
lookup logic against a realistic weight_map shape (two shards, three
layers split across the shard boundary, plus non-layer tensors).
"""

from __future__ import annotations

import json

import pytest

from squish.quant.shard_index import ShardIndex, load_shard_index

# Mirrors a real sharded model: embed_tokens + layers 0-1 in shard 1,
# layer 2 spans the shard boundary (layernorm in shard 1, everything else
# in shard 2) to exercise the "one layer needs two shards" case, plus
# lm_head/norm in shard 2.
SAMPLE_WEIGHT_MAP = {
    "model.embed_tokens.weight": "model-00001-of-00002.safetensors",
    "model.layers.0.input_layernorm.weight": "model-00001-of-00002.safetensors",
    "model.layers.0.self_attn.q_proj.weight": "model-00001-of-00002.safetensors",
    "model.layers.0.mlp.down_proj.weight": "model-00001-of-00002.safetensors",
    "model.layers.1.input_layernorm.weight": "model-00001-of-00002.safetensors",
    "model.layers.1.self_attn.q_proj.weight": "model-00001-of-00002.safetensors",
    "model.layers.2.input_layernorm.weight": "model-00001-of-00002.safetensors",
    "model.layers.2.self_attn.q_proj.weight": "model-00002-of-00002.safetensors",
    "model.layers.2.mlp.down_proj.weight": "model-00002-of-00002.safetensors",
    "model.norm.weight": "model-00002-of-00002.safetensors",
    "lm_head.weight": "model-00002-of-00002.safetensors",
}


@pytest.fixture
def index() -> ShardIndex:
    return ShardIndex(weight_map=dict(SAMPLE_WEIGHT_MAP), num_layers=3)


class TestLoadShardIndex:
    def test_returns_none_when_index_missing(self, tmp_path):
        assert load_shard_index(tmp_path) is None

    def test_parses_real_shaped_index(self, tmp_path):
        (tmp_path / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": SAMPLE_WEIGHT_MAP})
        )
        idx = load_shard_index(tmp_path)
        assert idx is not None
        assert idx.num_layers == 3
        assert idx.weight_map == SAMPLE_WEIGHT_MAP

    def test_num_layers_from_max_layer_index_plus_one(self, tmp_path):
        (tmp_path / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"model.layers.5.input_layernorm.weight": "s.safetensors"}})
        )
        idx = load_shard_index(tmp_path)
        assert idx.num_layers == 6

    def test_zero_layers_when_no_layer_tensors(self, tmp_path):
        (tmp_path / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"model.embed_tokens.weight": "s.safetensors"}})
        )
        idx = load_shard_index(tmp_path)
        assert idx.num_layers == 0

    def test_malformed_json_raises_clear_error(self, tmp_path):
        (tmp_path / "model.safetensors.index.json").write_text("{not valid")
        with pytest.raises(ValueError, match="Could not parse"):
            load_shard_index(tmp_path)

    def test_missing_weight_map_key_raises(self, tmp_path):
        (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}}))
        with pytest.raises(ValueError, match="no weight_map"):
            load_shard_index(tmp_path)


class TestTensorsForLayer:
    def test_returns_only_that_layers_tensors(self, index):
        names = index.tensors_for_layer(0)
        assert set(names) == {
            "model.layers.0.input_layernorm.weight",
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.mlp.down_proj.weight",
        }

    def test_empty_for_out_of_range_layer(self, index):
        assert index.tensors_for_layer(99) == []


class TestShardsForLayer:
    def test_single_shard_layer(self, index):
        assert index.shards_for_layer(0) == {"model-00001-of-00002.safetensors"}

    def test_layer_spanning_shard_boundary(self, index):
        # Layer 2's input_layernorm is in shard 1, the rest in shard 2 —
        # both shards must be reported, or an eviction-timing bug could
        # release shard 1 while layer 2 still needs its layernorm weight.
        assert index.shards_for_layer(2) == {
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
        }


class TestNonLayerTensors:
    def test_excludes_all_layer_tensors(self, index):
        names = set(index.non_layer_tensors())
        assert names == {"model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"}
        assert not any("layers" in n for n in names)


class TestShardToLayers:
    def test_maps_each_shard_to_its_layer_set(self, index):
        mapping = index.shard_to_layers()
        assert mapping["model-00001-of-00002.safetensors"] == {0, 1, 2}
        assert mapping["model-00002-of-00002.safetensors"] == {2}

    def test_non_layer_tensors_excluded_from_mapping(self, index):
        mapping = index.shard_to_layers()
        all_layers = set().union(*mapping.values())
        assert all_layers == {0, 1, 2}
