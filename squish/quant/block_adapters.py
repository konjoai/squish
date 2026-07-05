"""squish/quant/block_adapters.py — resolve a model architecture to the
right sequential-AWQ-calibration adapter, keyed by decoder-block *kind*
(structural shape), not by architecture family name.

Wave 142's spike proved a standalone ``mlx_lm`` ``TransformerBlock(args)``
can be reconstructed and run with just its own weights. Most named
architectures in the catalog (Llama, Qwen2, Qwen3, ...) share the exact
same block shape — a residual stream through self-attention (plain
``nn.Linear`` q/k/v/o projections) and an MLP (plain ``nn.Linear``
gate/up/down projections), no MoE routing, no recurrent/SSM state. One
"standard dense block" adapter covers all of them; a new family name that
reuses this same shape needs zero new code here at all, since resolution
is driven by ``mlx_lm.utils.MODEL_REMAPPING`` (see
:func:`resolve_dense_architecture`), not a hand-maintained list.

Why detection isn't simply ``type(block).__name__ == "TransformerBlock"``:
verified directly against the installed mlx_lm 0.31.3 model files that
class-name matching alone is unsafe in both directions —

- **False positive**: ``mlx_lm/models/olmoe.py`` names its block class
  ``TransformerBlock`` too, but it's genuinely MoE (contains an
  ``OlmoeSparseMoeBlock`` submodule) — same name, different kind.
- **False negative**: ``mlx_lm/models/phi.py``'s dense block is named
  ``PhiDecoderLayer``, not ``TransformerBlock`` — different name, same
  kind.

So :func:`is_standard_dense_block` inspects the *actual instantiated
submodule structure* (via ``mlx.nn.Module.named_modules()``), checking
submodule *class* names for MoE/SSM markers — never attribute names,
since a dense SwiGLU MLP's own ``gate_proj`` attribute would otherwise
false-positive against a naive "gate" substring check.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass

# Class-name substrings (case-sensitive, matched against type(submodule).__name__)
# that mark a block as MoE — observed directly in mlx_lm's own MoE model files
# (mixtral, qwen3_moe, deepseek_v2/v3, olmoe, granitemoe, jamba).
_MOE_MARKERS = ("Moe", "MoE", "Sparse", "Expert")

# Class-name substrings marking SSM/recurrent state (mamba, mamba2,
# recurrent_gemma/Griffin, jamba's Mamba mixer).
_SSM_MARKERS = ("Mamba", "SSM", "RGLRU", "Recurrent")


@dataclass
class ResolvedArchitecture:
    """A dense-family architecture resolved to its mlx_lm building blocks."""

    model_args_cls: type
    transformer_block_cls: type
    module_name: str


def resolve_dense_architecture(model_type: str) -> ResolvedArchitecture | None:
    """Resolve *model_type* to its ``ModelArgs``/``TransformerBlock`` classes.

    Reuses ``mlx_lm.utils.MODEL_REMAPPING`` — the same table and resolution
    logic (``MODEL_REMAPPING.get(model_type, model_type)`` then
    ``importlib.import_module(f"mlx_lm.models.{model_type}")``) that
    ``mlx_lm.utils._get_classes`` itself uses to load a model — rather than
    a hand-maintained duplicate list, so this automatically covers any
    architecture mlx_lm supports (e.g. "mistral" remaps to "llama", which
    has no dedicated mistral.py; verified against mlx_lm 0.31.3).

    A resolved module having a ``TransformerBlock`` class name is NOT
    itself proof of "dense" — some MoE architectures (e.g. olmoe.py) reuse
    that exact class name. This function only resolves *which classes to
    construct*; :func:`is_standard_dense_block` on the constructed
    instance is the actual safety check. Returns ``None`` if the model_type
    can't be imported, or the module lacks ``ModelArgs``/``TransformerBlock``
    entirely (e.g. genuinely different block-shape families like Mamba) —
    the caller must fall back to plain (non-AWQ) quantization rather than
    guess.
    """
    from mlx_lm.utils import MODEL_REMAPPING

    resolved_type = MODEL_REMAPPING.get(model_type, model_type)
    try:
        mod = importlib.import_module(f"mlx_lm.models.{resolved_type}")
    except ImportError:
        return None
    model_args_cls = getattr(mod, "ModelArgs", None)
    block_cls = getattr(mod, "TransformerBlock", None)
    if model_args_cls is None or block_cls is None:
        return None
    return ResolvedArchitecture(
        model_args_cls=model_args_cls,
        transformer_block_cls=block_cls,
        module_name=resolved_type,
    )


def is_standard_dense_block(block) -> bool:
    """Structurally verify *block* is a plain attention+MLP decoder layer.

    Checks every submodule's *class* name for MoE/SSM markers — never
    attribute names, since e.g. a dense SwiGLU MLP's own ``gate_proj``
    linear layer must not be mistaken for MoE gating.
    """
    if not hasattr(block, "named_modules"):
        return False
    for _name, submodule in block.named_modules():
        cls_name = type(submodule).__name__
        if any(marker in cls_name for marker in _MOE_MARKERS):
            return False
        if any(marker in cls_name for marker in _SSM_MARKERS):
            return False
    return True
