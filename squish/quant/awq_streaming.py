"""squish/quant/awq_streaming.py — sequential (layer-at-a-time) AWQ
calibration, never instantiating the full model.

``squish/quant/awq.py``'s ``collect_activation_scales`` calls
``mlx_lm.load(model_dir)`` — loading the entire bf16 model into unified
memory to run real forward passes and capture per-channel activation
statistics. That's the actual RAM wall for models bigger than local
memory (see ``docs/ARCHITECTURE.md`` section 11). This module replaces
"load the whole model, run full forward passes" with the GPTQ/AutoAWQ
pattern: process one decoder layer at a time, carrying the hidden state
forward between layers.

Reuses, unchanged:
- Wave 142's proof that a standalone ``mlx_lm`` ``TransformerBlock(args)``
  can be constructed and run in isolation with just its own weights
  (``squish/quant/block_adapters.py``'s ``resolve_dense_architecture`` /
  ``is_standard_dense_block``).
- Wave 142's ``ShardIndex`` (``squish/quant/shard_index.py``) to know
  which raw shard(s) a given layer's weights live in.
- ``awq.py``'s activation-hook mechanism (``_ActivationHook``,
  ``_attach_activation_hooks``/``_detach_activation_hooks``,
  ``_scales_from_hooks``) — the math and instrumentation technique don't
  change, only how the activations that feed them are produced (one
  layer at a time instead of one whole-model forward pass at a time).

Scope: this wave covers the "standard dense block" adapter only (Llama,
Qwen2, Qwen3, Mistral, Phi3, Starcoder2, and any architecture
``mlx_lm.utils.MODEL_REMAPPING``/import resolution reaches that shares
the same attention+MLP shape). Any architecture that resolves to a
different block kind (MoE, SSM/hybrid) falls back automatically —
``collect_activation_scales_streaming`` returns ``None`` rather than
guessing, and the caller is expected to fall back to plain (non-AWQ)
streaming quantization (``process_weights_streaming`` with
``awq_scales=None``), per the "no adapter yet is a quality trade-off, not
a hard failure" design.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from squish.quant.awq import (
    _attach_activation_hooks,
    _detach_activation_hooks,
    _DEFAULT_CALIBRATION_TEXTS,
    _MODEL_FAMILY_DEFAULTS,
    _scales_from_hooks,
)
from squish.quant.block_adapters import is_standard_dense_block, resolve_dense_architecture
from squish.quant.shard_index import ShardIndex

_LOG = logging.getLogger("squish.quant.awq_streaming")


def _construct_block(resolved, args):
    """Construct a standalone decoder block, tolerating per-family
    constructor signature differences (e.g. llama.TransformerBlock takes
    a ``use_sliding`` kwarg that qwen3.TransformerBlock does not)."""
    try:
        return resolved.transformer_block_cls(args=args, use_sliding=False)
    except TypeError:
        return resolved.transformer_block_cls(args=args)


def _find_embed_tokens_tensor(shard_index: ShardIndex) -> str | None:
    """Find the embed_tokens weight's full tensor name in the weight_map."""
    for name in shard_index.non_layer_tensors():
        if name.endswith("embed_tokens.weight"):
            return name
    return None


def collect_activation_scales_streaming(
    model_dir: str | Path,
    tokenizer,
    shard_index: ShardIndex,
    texts: list | None = None,
    n_samples: int = 64,
    alpha: float = 0.5,
    seq_len: int = 512,
    verbose: bool = True,
    min_scale: float = 0.0,
    model_family: str | None = None,
) -> dict | None:
    """Sequential (layer-at-a-time) counterpart to
    :func:`squish.quant.awq.collect_activation_scales`.

    Never holds more than one decoder layer's weights, plus the
    carried-forward hidden states for *n_samples* short calibration
    sequences, in memory at once — unlike the full-load path, this works
    regardless of whether the model fits in unified memory.

    Returns the same ``layer_name -> np.ndarray`` scale dict contract as
    the full-load function, or ``None`` if the architecture doesn't
    resolve to the standard dense block adapter (the caller must fall
    back to plain non-AWQ quantization in that case, not treat this as a
    hard error).
    """
    import mlx.core as mx
    import mlx.nn as nn
    from safetensors.numpy import load_file as st_load_numpy

    model_dir = Path(model_dir)
    config = json.loads((model_dir / "config.json").read_text())
    model_type = config.get("model_type", "")

    resolved = resolve_dense_architecture(model_type)
    if resolved is None:
        if verbose:
            print(
                f"  [streaming-AWQ] model_type={model_type!r} has no dense-block "
                f"adapter yet — falling back to plain (non-AWQ) quantization."
            )
        return None

    args = resolved.model_args_cls.from_dict(config)

    if texts is None:
        family_cfg = _MODEL_FAMILY_DEFAULTS.get(model_family or model_type or "")
        texts = family_cfg["texts"] if family_cfg else _DEFAULT_CALIBRATION_TEXTS
    sample_texts = [texts[i % len(texts)] for i in range(n_samples)]

    sample_ids = []
    for text in sample_texts:
        ids = tokenizer.encode(text, add_special_tokens=True)[:seq_len]
        if ids:
            sample_ids.append(ids)
    if not sample_ids:
        if verbose:
            print("  [streaming-AWQ] no usable calibration samples after tokenization.")
        return None

    # ── Embeddings: load once, look up calibration ids, discard the table ──
    embed_name = _find_embed_tokens_tensor(shard_index)
    if embed_name is None:
        if verbose:
            print("  [streaming-AWQ] no embed_tokens tensor in weight_map — aborting.")
        return None
    embed_shard = shard_index.weight_map[embed_name]
    embed_table = st_load_numpy(str(model_dir / embed_shard))[embed_name].astype(np.float32)

    hidden_states = [mx.array(embed_table[ids][None, :, :]) for ids in sample_ids]
    del embed_table

    if verbose:
        print(
            f"  [streaming-AWQ] {resolved.module_name} · {shard_index.num_layers} layers · "
            f"{len(hidden_states)} calibration samples"
        )

    scales: dict[str, np.ndarray] = {}

    for layer_idx in range(shard_index.num_layers):
        prefix = f"model.layers.{layer_idx}."
        tensor_names = shard_index.tensors_for_layer(layer_idx)
        needed_shards = shard_index.shards_for_layer(layer_idx)

        layer_raw: dict[str, np.ndarray] = {}
        for shard_name in needed_shards:
            shard_data = st_load_numpy(str(model_dir / shard_name))
            for name in tensor_names:
                if name in shard_data:
                    layer_raw[name[len(prefix) :]] = shard_data[name]
            del shard_data

        block = _construct_block(resolved, args)
        if not is_standard_dense_block(block):
            if verbose:
                print(
                    f"  [streaming-AWQ] layer {layer_idx} failed the structural dense "
                    f"check — aborting streaming calibration, falling back."
                )
            return None

        weight_pairs = [(name, mx.array(arr.astype(np.float32))) for name, arr in layer_raw.items()]
        block.load_weights(weight_pairs, strict=False)

        linear_layers = {
            name: sub for name, sub in block.named_modules() if isinstance(sub, nn.Linear)
        }
        hooks, orig_classes = _attach_activation_hooks(linear_layers)

        next_hidden_states = []
        for h in hidden_states:
            try:
                out = block(h, mask=None, cache=None)
                mx.eval(out)
            except (RuntimeError, ValueError, TypeError, AttributeError) as exc:
                _LOG.debug(
                    "Streaming calibration forward pass failed at layer %d; "
                    "carrying the input through unchanged for this sample: %s",
                    layer_idx,
                    exc,
                )
                out = h
            next_hidden_states.append(out)

        _detach_activation_hooks(linear_layers, orig_classes)

        for local_name, scale in _scales_from_hooks(hooks, alpha, min_scale).items():
            scales[f"{prefix}{local_name}"] = scale

        hidden_states = next_hidden_states
        del layer_raw, block, hooks, orig_classes

        if verbose and (layer_idx + 1) % 8 == 0:
            print(f"    [{layer_idx + 1}/{shard_index.num_layers}] layers calibrated …")

    if verbose:
        print(f"  [streaming-AWQ] computed AWQ scales for {len(scales)} layers")

    return scales
