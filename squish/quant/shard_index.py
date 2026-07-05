"""squish/quant/shard_index.py — parse a sharded safetensors model's
``model.safetensors.index.json`` into a tensor -> shard lookup, grouped by
decoder-layer index.

Sequential (layer-at-a-time) AWQ calibration and per-shard streaming pull
both need to know, for a given layer, which raw shard file(s) hold its
weights — so a shard can be fetched/kept resident only while a layer that
needs it is being processed, and released once no later layer needs it.

``squish/catalog.py``'s ``_is_raw_model_dir_complete`` already parses
``weight_map`` to verify a download is complete, but doesn't expose a
layer-indexed lookup — this module is that lookup, built as its own small
module rather than folded into the scanner, which has a different job
(download-completeness checking, not layer/shard bookkeeping).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

_LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.")


@dataclass
class ShardIndex:
    """Parsed view of a model's weight_map, layer-aware."""

    weight_map: dict[str, str]  # tensor name -> shard filename
    num_layers: int

    def tensors_for_layer(self, layer_idx: int) -> list[str]:
        """Tensor names belonging to decoder layer *layer_idx*."""
        prefix = f"model.layers.{layer_idx}."
        return [name for name in self.weight_map if name.startswith(prefix)]

    def shards_for_layer(self, layer_idx: int) -> set[str]:
        """Shard filename(s) that contain any tensor for layer *layer_idx*.

        Usually one shard, but a shard may be needed by more than one layer
        if shard boundaries don't align with layer boundaries — callers
        must not evict a shard until every layer needing it is done.
        """
        return {self.weight_map[name] for name in self.tensors_for_layer(layer_idx)}

    def non_layer_tensors(self) -> list[str]:
        """Tensors outside the per-layer loop: embed_tokens, lm_head, final norm.

        Not needed for AWQ calibration (only decoder layers' nn.Linear
        activations matter), but are needed in the final compressed output
        since they're part of the served model.
        """
        return [name for name in self.weight_map if not _LAYER_RE.match(name)]

    def shard_to_layers(self) -> dict[str, set[int]]:
        """Reverse mapping: which layer indices does each shard file cover?

        Drives eviction timing — a shard is safe to release only after
        every layer index in its set has been processed.
        """
        result: dict[str, set[int]] = {}
        for name, shard in self.weight_map.items():
            m = _LAYER_RE.match(name)
            if m:
                result.setdefault(shard, set()).add(int(m.group(1)))
        return result


def load_shard_index(model_dir: str | Path) -> ShardIndex | None:
    """Load and parse ``model.safetensors.index.json`` from *model_dir*.

    Returns ``None`` if the index file doesn't exist — a single-shard model
    has no sharding to track, so callers should fall back to treating the
    whole model as one unit rather than treating this as an error.
    """
    index_path = Path(model_dir) / "model.safetensors.index.json"
    if not index_path.exists():
        return None
    try:
        data = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not parse {index_path}: {exc}") from exc

    weight_map = data.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError(f"{index_path} has no weight_map dict")

    layer_indices = {int(m.group(1)) for name in weight_map if (m := _LAYER_RE.match(name))}
    num_layers = max(layer_indices) + 1 if layer_indices else 0
    return ShardIndex(weight_map=dict(weight_map), num_layers=num_layers)
