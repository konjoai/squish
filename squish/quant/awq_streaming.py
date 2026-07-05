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
    delete_source: bool = False,
) -> dict | None:
    """Sequential (layer-at-a-time) counterpart to
    :func:`squish.quant.awq.collect_activation_scales`.

    Never holds more than one decoder layer's weights, plus the
    carried-forward hidden states for *n_samples* short calibration
    sequences, in memory at once — unlike the full-load path, this works
    regardless of whether the model fits in unified memory.

    If ``delete_source`` is set, each raw shard file is deleted once every
    layer that needs it has finished calibrating — mirroring
    ``process_weights_streaming``'s ``--delete-source`` safety rule
    (Wave 139/141): a shard is only removed after its last consumer has
    successfully completed, using :meth:`ShardIndex.shard_to_layers` to
    know when that point is reached. A shard shared between embed_tokens
    and layer 0 (shard boundaries don't always align with layer
    boundaries) is handled correctly: embed_tokens' one-time read happens
    before the layer loop, so by the time layer 0 finishes, the shard has
    no remaining consumers. Deletion failures are logged as warnings, not
    raised — losing cleanup for one shard isn't a reason to fail an
    otherwise-successful calibration run.

    Returns the same ``layer_name -> np.ndarray`` scale dict contract as
    the full-load function, or ``None`` if the architecture doesn't
    resolve to the standard dense block adapter (the caller must fall
    back to plain non-AWQ quantization in that case, not treat this as a
    hard error).

    Note for callers wiring this into a CLI: with ``delete_source=True``,
    an early ``None`` return (unsupported block kind found partway through)
    can still leave some already-processed layers' shards deleted — those
    layers' scales are simply discarded along with everything else, but
    the raw shards backing them are genuinely gone. This mirrors
    ``process_weights_streaming``'s own documented tradeoff: a caller that
    falls back to plain quantization after a ``None`` here must re-pull
    the model first if any raw shards were already reclaimed.
    """
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm.models.base import create_attention_mask

    from squish.convert import load_mlx_weights_shard

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
    embed_table = load_mlx_weights_shard(model_dir / embed_shard)[embed_name]

    hidden_states = [mx.array(embed_table[ids][None, :, :]) for ids in sample_ids]
    del embed_table

    # ── Delete-as-you-go bookkeeping: remaining layer-consumers per shard ──
    # embed_tokens' read above already happened, so a shard with no layer
    # consumers left (never appears in shard_to_layers) is safe to delete
    # right now; a shard shared with layer 0+ is deferred to the per-layer
    # loop below, which correctly sees it as having no remaining consumers
    # once its last layer finishes.
    shards_deleted = 0
    bytes_reclaimed = 0
    shard_remaining_layers = shard_index.shard_to_layers()
    if delete_source and embed_shard not in shard_remaining_layers:
        try:
            shard_path = model_dir / embed_shard
            shard_bytes = shard_path.stat().st_size
            shard_path.unlink()
            shards_deleted += 1
            bytes_reclaimed += shard_bytes
            if verbose:
                print(f"    [delete-source] removed {embed_shard} (embed_tokens-only shard)")
        except OSError as exc:
            _LOG.warning(
                "Could not delete raw shard %s after embed_tokens read: %s "
                "(calibration is unaffected; disk space was not reclaimed)",
                embed_shard,
                exc,
            )

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
            shard_data = load_mlx_weights_shard(model_dir / shard_name)
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
                # create_attention_mask(h) matches exactly what the full-load
                # model uses per-layer: "causal" for seq_len > 1, None for a
                # single token. Passing mask=None unconditionally here was a
                # real bug (caught by Wave 145's accuracy validation, not
                # assumed) — every token could attend to every other token,
                # including future ones, which is not what real causal
                # attention does and measurably skewed activation statistics.
                mask = create_attention_mask(h)
                out = block(h, mask=mask, cache=None)
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

        # ── Delete-as-you-go: drop each needed shard once this layer (its
        # last remaining consumer, or one of several) has finished. Runs
        # only after scales for this layer are already merged into `scales`
        # above — a shard is never removed until its calibration
        # contribution is confirmed captured.
        if delete_source:
            for shard_name in needed_shards:
                consumers = shard_remaining_layers.get(shard_name)
                if consumers is None:
                    continue  # not a layer-owned shard (shouldn't happen here)
                consumers.discard(layer_idx)
                if consumers:
                    continue  # another later layer still needs this shard
                try:
                    shard_path = model_dir / shard_name
                    shard_bytes = shard_path.stat().st_size
                    shard_path.unlink()
                    shards_deleted += 1
                    bytes_reclaimed += shard_bytes
                    if verbose:
                        print(
                            f"    [delete-source] removed {shard_name} "
                            f"(last needed by layer {layer_idx})"
                        )
                except OSError as exc:
                    _LOG.warning(
                        "Could not delete raw shard %s after layer %d: %s "
                        "(calibration is unaffected; disk space was not reclaimed)",
                        shard_name,
                        layer_idx,
                        exc,
                    )

        if verbose and (layer_idx + 1) % 8 == 0:
            print(f"    [{layer_idx + 1}/{shard_index.num_layers}] layers calibrated …")

    if verbose:
        print(f"  [streaming-AWQ] computed AWQ scales for {len(scales)} layers")
        if delete_source:
            print(
                f"  [streaming-AWQ] delete-source: {shards_deleted} raw shard(s) removed, "
                f"{bytes_reclaimed / 1e9:.2f} GB reclaimed"
            )

    return scales
