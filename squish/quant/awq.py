#!/usr/bin/env python3
"""
squish/awq.py

Activation-Aware Weight Quantization (AWQ) calibration for Squish.

AWQ improves INT8 quantization accuracy by protecting the ~1% of weight
channels that are most sensitive to quantization error.  Those channels are
identified by large activation magnitudes — high |x[c]| means any quantization
error in W[:,c] gets amplified at the output.

Algorithm (Lin et al., 2023  https://arxiv.org/abs/2306.00978):
  1. Run a calibration set through the model.
  2. For each linear layer, collect per-input-channel activation magnitudes.
  3. Compute per-channel scale:  s[c] = mean_act[c] ** alpha    (alpha ≈ 0.5)
  4. Before quantization: W_awq[:, c] /= s[c]
     The input rescaling (X_awq[c] = X[c] * s[c]) is absorbed into the
     previous LayerNorm's gamma:  gamma_awq[c] = gamma[c] * s[c]

Net effect: salient channels are moved into a tighter range that INT8 can
represent accurately — typically +0.5-2% accuracy on MMLU / HellaSwag.

Usage
-----
Calibration (needs the FP16 model loaded):

    python3 -m squish.quant.awq \\
        --model-dir ~/models/Qwen2.5-7B-Instruct-bf16 \\
        --output    ~/models/Qwen2.5-7B-Instruct-bf16/awq_scales \\
        --n-samples 128 \\
        --alpha 0.5

Then pass scales to squish.convert:

    python3 -m squish.convert \\
        --model-dir ~/models/Qwen2.5-7B-Instruct-bf16 \\
        --output    ~/models/squish_7b \\
        --awq-scales ~/models/Qwen2.5-7B-Instruct-bf16/awq_scales

Or apply programmatically before quantizing a weight dict:

    from squish.quant.awq import load_awq_scales, apply_awq_to_weights
    scales = load_awq_scales(awq_dir)
    weights = apply_awq_to_weights(weights, scales)
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Default calibration texts — mix of factual, reasoning, and conversational
# so activation statistics cover the full distribution of real usage.
# ---------------------------------------------------------------------------
_DEFAULT_CALIBRATION_TEXTS = [
    # Factual knowledge (arc_easy style)
    "The capital of France is Paris, which is also the largest city in the country.",
    "Machine learning models learn patterns from data by adjusting internal parameters.",
    "In 1969, NASA's Apollo 11 mission successfully landed astronauts on the Moon.",
    "Python is a high-level programming language known for its readable syntax.",
    "The human brain contains approximately 86 billion neurons connected by synapses.",
    "Climate change is driven by greenhouse gas emissions from fossil fuels.",
    "The speed of light in a vacuum is approximately 299,792 kilometers per second.",
    "Shakespeare wrote 37 plays and 154 sonnets during his lifetime.",
    "Quantum mechanics describes the behavior of particles at the atomic scale.",
    "The Amazon rainforest produces about 20% of the world's oxygen.",
    "Mathematics is the language of the universe, as Galileo famously stated.",
    "Artificial intelligence systems are increasingly used in medical diagnosis.",
    "The Great Wall of China stretches over 13,000 miles across northern China.",
    "DNA carries the genetic instructions for the development of all living organisms.",
    "The Internet was developed from ARPANET, a US Department of Defense project.",
    "Water molecules consist of two hydrogen atoms bonded to one oxygen atom.",
    "The Roman Empire at its height controlled most of Europe and the Middle East.",
    "Photosynthesis converts carbon dioxide and water into glucose using sunlight.",
    "Neural networks are inspired by the structure of the human brain.",
    "The Theory of Relativity was developed by Albert Einstein in the early 1900s.",
    # Commonsense reasoning (hellaswag/winogrande style)
    "She put on her coat before going outside because it was cold and snowing heavily.",
    "The dog wagged its tail when its owner came home after a long day at work.",
    "After mixing flour, eggs, and butter, she put the dough in the oven to bake.",
    "He drove carefully on the icy road because he did not want to skid off the path.",
    "The children laughed loudly when the clown pretended to fall off his tiny bicycle.",
    "She looked both ways before crossing the street to make sure no cars were coming.",
    "The library was quiet because everyone was trying to concentrate on their books.",
    "He turned off the lights and locked the door before leaving for his vacation.",
    "The cat knocked the glass off the table and watched it shatter on the floor.",
    "After the storm, the streets were flooded and the power lines had fallen down.",
    # Physical / procedural reasoning (piqa style)
    "To sharpen a pencil, you insert the blunt end into the sharpener and rotate it gently.",
    "To open a jar with a tight lid, run warm water over the lid and try turning it again.",
    "When hammering a nail, hold it steady with your fingers then strike the head firmly.",
    "To remove a splinter, sterilize a needle with alcohol and gently lift the skin edge.",
    "To fold a paper airplane, crease the paper down the middle and fold the wings evenly.",
    "She filled the kettle with cold water before placing it on the stove to boil.",
    "He sanded the wooden surface smooth before applying a coat of paint to it.",
    "To unstick a zip, rub a candle or bar of soap along the teeth of the fastener.",
    "Rolling out dough requires sprinkling flour on the surface so it does not stick.",
    "You can test if an egg is fresh by placing it in a glass of cold water to see if it floats.",
]


# ---------------------------------------------------------------------------
# Activation collection
# ---------------------------------------------------------------------------

class _ActivationHook:
    """
    Forward hook that accumulates per-channel mean absolute activation values.

    For a linear layer with weight W  (out_features × in_features):
    We want to measure the magnitude of each INPUT channel [0..in_features-1]
    so we can decide which channels are salient.

    This hook captures the INPUT tensor to the linear layer.  Shape will be
    (batch, seq_len, in_features) for typical transformer calls.
    """
    def __init__(self):
        self.channel_sum   = None   # float64 accumulator, shape (in_features,)
        self.channel_count = 0

    def __call__(self, module, inp, output):
        # inp is a tuple; inp[0] is the activation tensor
        x = inp[0]
        try:
            import mlx.core as mx
            # Convert MLX array to numpy for statistics
            arr = np.array(x.astype(mx.float32))
        except Exception:
            arr = np.asarray(x, dtype=np.float32)

        # arr shape: (..., in_features) — flatten all but last dim
        flat = arr.reshape(-1, arr.shape[-1])           # (N, in_features)
        abs_mean = np.abs(flat).mean(axis=0)            # (in_features,)

        if self.channel_sum is None:
            self.channel_sum = abs_mean.astype(np.float64)
        else:
            self.channel_sum += abs_mean.astype(np.float64)
        self.channel_count += 1

    def mean_activation(self) -> np.ndarray:
        """Return mean per-channel activation magnitude (float32)."""
        if self.channel_sum is None or self.channel_count == 0:
            return np.array([], dtype=np.float32)
        return (self.channel_sum / self.channel_count).astype(np.float32)


def collect_activation_scales(  # pragma: no cover
    model,
    tokenizer,
    texts: list | None = None,
    n_samples: int = 64,
    alpha: float = 0.5,
    seq_len: int = 512,
    verbose: bool = True,
    min_scale: float = 0.0,
) -> dict:
    """
    Run calibration data through ``model`` and compute per-layer AWQ scales.

    Parameters
    ----------
    model       : mlx_lm model object (already loaded, on Metal)
    tokenizer   : HuggingFace tokenizer matching the model
    texts       : list of calibration strings (defaults to built-in set)
    n_samples   : how many total forward passes to run (more = better stats)
    alpha       : scale exponent  0 = no AWQ, 1 = full activation scaling
                  0.5 is the default recommended in the AWQ paper
    seq_len     : max token length per sample (truncated / padded)
    verbose     : print progress
    min_scale   : floor for computed scales (default 0.0 = no floor).
                  Set to 1.0 to only amplify salient channels and never
                  attenuate less-active ones, which avoids reducing weight
                  values for channels whose mean activation < 1.0.

    Returns
    -------
    dict mapping ``layer_name → np.ndarray(shape=(in_features,), dtype=float32)``
    of AWQ scales.  These are the ``s`` vectors to be applied as::

        W_awq[:, c] = W[:, c] / s[c]   (applied before quantization)
        gamma_awq[c] = gamma[c] * s[c]  (absorbed into preceding LayerNorm)
    """
    import mlx.core as mx

    if texts is None:
        texts = _DEFAULT_CALIBRATION_TEXTS

    # Cycle the text list to reach n_samples
    sample_texts = [texts[i % len(texts)] for i in range(n_samples)]

    # Collect all nn.Linear modules and attach hooks
    hooks   = {}      # layer_name → _ActivationHook

    # MLX module interception via __class__ swizzling.
    #
    # Python's special-method dispatch for obj(x) always resolves __call__
    # through the TYPE (type(obj).__call__), not the instance dict, so simple
    # instance-level monkey-patching of __call__ is silently ignored.
    #
    # The correct approach is to dynamically create a per-instance subclass
    # that overrides __call__ at the type level, then swap the instance's
    # __class__ to that subclass ("class swizzling").  This is safe because
    # both the original and the new class share the same memory layout (same
    # __slots__, same underlying C struct) — Python 3 allows the assignment
    # when layout-compatible.
    import mlx.nn as nn

    linear_layers = {}
    for name, module in (model.named_modules()
                         if hasattr(model, "named_modules") else []):
        if isinstance(module, nn.Linear):
            linear_layers[name] = module

    # Fallback: recursive children() traversal
    if not linear_layers:
        def _collect(mod, prefix=""):
            children = mod.children() if hasattr(mod, "children") else {}
            if isinstance(children, dict):
                items = children.items()
            else:
                items = []
            for child_name, child in items:
                full = f"{prefix}.{child_name}" if prefix else child_name
                if isinstance(child, nn.Linear):
                    linear_layers[full] = child
                _collect(child, full)
        _collect(model)

    if verbose:
        print(f"  Found {len(linear_layers)} linear layers to calibrate")

    # Instrument each module via class swizzling
    orig_classes: dict[str, type] = {}
    for name, module in linear_layers.items():
        hook = _ActivationHook()
        hooks[name] = hook
        orig_cls = type(module)
        orig_classes[name] = orig_cls

        # Create a unique subclass that captures input activations
        def _make_hooked_cls(base_cls, h: "_ActivationHook") -> type:
            class _Hooked(base_cls):  # type: ignore[valid-type]
                _awq_hook = h
                def __call__(self, x, *args, **kwargs):
                    self._awq_hook(None, (x,), None)
                    return super().__call__(x, *args, **kwargs)
            _Hooked.__name__ = f"_Hooked_{base_cls.__name__}"
            _Hooked.__qualname__ = _Hooked.__name__
            return _Hooked

        try:
            module.__class__ = _make_hooked_cls(orig_cls, hook)
        except TypeError:
            # Fallback for C-backed types that disallow __class__ swizzling:
            # fall through without instrumentation for this layer.
            pass

    if verbose:
        print(f"  Running {n_samples} calibration forward passes ...")
    t0 = time.perf_counter()

    for i, text in enumerate(sample_texts):
        ids = tokenizer.encode(text, add_special_tokens=True)[:seq_len]
        if not ids:
            continue
        x = mx.array([ids], dtype=mx.int32)
        try:
            out = model(x)
            mx.eval(out)        # materialise lazy graph so hooks complete
        except Exception:
            pass                # some models need kv_cache — skip on error

        if verbose and (i + 1) % 16 == 0:
            print(f"    [{i+1}/{n_samples}] calibrated …")

    elapsed = time.perf_counter() - t0
    if verbose:
        print(f"  Calibration done in {elapsed:.1f}s")

    # Restore original classes
    for name, module in linear_layers.items():
        if name in orig_classes:
            try:
                module.__class__ = orig_classes[name]
            except TypeError:
                pass

    # Compute AWQ scales from collected statistics
    scales = {}
    for name, hook in hooks.items():
        mean_act = hook.mean_activation()
        if mean_act.size == 0:
            continue
        # s[c] = mean_act[c]^alpha  (clipped to ≥ 1e-4 to avoid div-by-zero)
        s = np.clip(mean_act, 1e-4, None) ** alpha
        if min_scale > 0.0:
            s = np.maximum(s, min_scale)
        scales[name] = s.astype(np.float32)

    if verbose:
        print(f"  Computed AWQ scales for {len(scales)} layers")

    return scales


# ---------------------------------------------------------------------------
# Scale persistence
# ---------------------------------------------------------------------------

def save_awq_scales(scales: dict, output_dir: str | Path, verbose: bool = True) -> None:
    """
    Persist AWQ scale vectors to ``output_dir`` as ``{layer_name}.awq.npy`` files.

    Layer names with ``/`` or ``.`` are converted to path-safe names using ``_``
    and ``__`` respectively — matching the safe_key convention in convert.py.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    n = 0
    for layer_name, scale in scales.items():
        safe = layer_name.replace("/", "_").replace(".", "__")
        np.save(str(out / f"{safe}.awq.npy"), scale)
        n += 1

    # Write index file so loaders know which layers have scales
    index = {k: k.replace("/", "_").replace(".", "__") + ".awq.npy"
             for k in scales}
    import json
    with open(out / "awq_index.json", "w") as f:
        json.dump(index, f, indent=2)

    (out / ".awq_ready").touch()

    if verbose:
        print(f"  Saved AWQ scales for {n} layers → {out}")


def load_awq_scales(awq_dir: str | Path) -> dict:
    """
    Load AWQ scale vectors from a directory written by :func:`save_awq_scales`.

    Returns a dict mapping ``layer_name → np.ndarray(float32)``.
    Returns an empty dict if the directory does not exist or has no AWQ files.
    """
    import json

    d = Path(awq_dir)
    if not d.exists():
        return {}

    index_path = d / "awq_index.json"
    if index_path.exists():
        with open(index_path) as f:
            index = json.load(f)
        scales = {}
        for layer_name, filename in index.items():
            p = d / filename
            if p.exists():
                scales[layer_name] = np.load(str(p))
        return scales

    # Fallback: enumerate all .awq.npy files and reverse-map the safe key
    scales = {}
    for p in sorted(d.glob("*.awq.npy")):
        layer_name = p.stem.replace("__", ".").replace("_", "/")
        scales[layer_name] = np.load(str(p))
    return scales


# ---------------------------------------------------------------------------
# Scale application
# ---------------------------------------------------------------------------

def apply_awq_to_weights(
    weights: dict,
    awq_scales: dict,
    alpha: float = 0.5,
    verbose: bool = False,
) -> dict:
    """
    Apply pre-computed AWQ scales to a flat weight dict before quantization.

    ``weights`` maps tensor names (e.g. ``model.layers.0.self_attn.q_proj.weight``)
    to float32 numpy arrays.

    For each matched linear layer weight W (shape out × in), the scale is
    applied column-wise per the AWQ paper (Lin et al., 2023 §3.2)::

        W_awq[:, c] = W[:, c] * s[c]         (multiply  — weight channel amplified)
        gamma_awq[c] = gamma[c] / s[c]       (divide    — LN gamma attenuated)

    This preserves the mathematical identity  ``(X / s) @ (W * s).T = X @ W.T``
    while improving INT4 precision: salient input channels (large ``s[c]``) get
    larger weight values → occupy more of the INT4 quantization range → smaller
    relative quantization error where it matters most.

    **Grouping rule**: only layers whose input is DIRECTLY a LayerNorm output
    are processed (q/k/v projections and gate/up MLP projections).  Layers
    whose input comes from something else (o_proj receives attention output;
    down_proj receives gate*up) are skipped because their scale cannot be
    absorbed into a preceding norm without distorting the output.

    For each LayerNorm that feeds multiple projections (e.g. q, k, v all share
    ``input_layernorm``), a SINGLE group scale is computed as the channel-wise
    mean across all members.  The LayerNorm is updated exactly once with this
    group scale, guaranteeing the identity
    ``(X / s) @ (W * s).T = X @ W.T`` for every member.

    Parameters
    ----------
    weights     : flat dict of {tensor_name: np.ndarray(float32)}
    awq_scales  : output of load_awq_scales() or collect_activation_scales()
    alpha       : for documentation purposes only — scales already account for alpha
    verbose     : print which tensors are AWQ-adjusted

    Returns
    -------
    Modified weight dict (modifies in-place and returns same dict).
    """
    # Only layers whose input is directly a LayerNorm output.
    # o_proj input = concatenated attention heads (NOT a LayerNorm).
    # down_proj input = gate * up after SiLU  (NOT a LayerNorm).
    # Including those would corrupt the model because their scale can't be
    # absorbed anywhere.
    _LN_INPUT_SUFFIXES = (
        "q_proj.weight", "k_proj.weight", "v_proj.weight",
        "gate_proj.weight", "up_proj.weight",
        "fc1.weight",
        "dense_h_to_4h.weight",
    )

    def _find_scale(layer_path: str) -> "np.ndarray | None":
        if layer_path in awq_scales:
            return awq_scales[layer_path]
        best, best_len = "", 0
        for k in awq_scales:
            if layer_path.endswith(k) and len(k) > best_len:
                best, best_len = k, len(k)
        return awq_scales[best] if best else None

    # ------------------------------------------------------------------
    # Step 1: group tensors by their preceding LayerNorm.
    # ------------------------------------------------------------------
    # ln_groups:     norm_name → [tensor_name, ...]
    # tensor_scales: tensor_name → individual per-channel scale
    ln_groups: dict[str, list[str]] = {}
    tensor_scales: dict[str, np.ndarray] = {}

    for tensor_name, arr in weights.items():
        if arr.ndim < 2:
            continue
        if not any(tensor_name.endswith(sfx) for sfx in _LN_INPUT_SUFFIXES):
            continue

        layer_path = tensor_name[: tensor_name.rfind(".")]
        scale = _find_scale(layer_path)
        if scale is None:
            continue

        W = arr.reshape(-1, arr.shape[-1])
        if scale.shape[0] != W.shape[1]:
            continue  # shape mismatch — skip

        norm_name = _preceding_norm_name(tensor_name, weights)
        if not norm_name or norm_name not in weights:
            continue  # can't absorb scale anywhere — skip

        tensor_scales[tensor_name] = scale
        ln_groups.setdefault(norm_name, []).append(tensor_name)

    if not ln_groups:
        if awq_scales:
            print("  [AWQ] Warning: no scales matched any weight tensors. "
                  "Check that layer names in awq_scales match model weight names.")
        return weights

    # ------------------------------------------------------------------
    # Step 2: for each LN group, compute ONE group scale (channel-wise
    # arithmetic mean across all member layers).  Using a single scale for
    # both the LN update and all member weight updates guarantees exact
    # mathematical equivalence: (X · s) @ (W / s).T == X @ W.T
    # ------------------------------------------------------------------
    ln_group_scales: dict[str, np.ndarray] = {
        norm_name: np.stack([tensor_scales[t] for t in tensor_names]).mean(axis=0)
        for norm_name, tensor_names in ln_groups.items()
    }

    # ------------------------------------------------------------------
    # Step 3: apply — update each LayerNorm EXACTLY ONCE, then scale weights.
    # ------------------------------------------------------------------
    n_applied = 0
    for norm_name, tensor_names in ln_groups.items():
        group_s = ln_group_scales[norm_name]

        # Attenuate LN gamma: gamma_awq[c] = gamma[c] / s[c]
        # (the input X is effectively divided by s, balancing the weight amplification)
        weights[norm_name] = weights[norm_name] / group_s

        for tensor_name in tensor_names:
            arr = weights[tensor_name]
            W = arr.reshape(-1, arr.shape[-1])
            # Amplify salient weight columns: W_awq[:, c] = W[:, c] * s[c]
            weights[tensor_name] = (W * group_s[np.newaxis, :]).reshape(arr.shape)
            if verbose:
                print(f"  [AWQ] {tensor_name}  s̄={group_s.mean():.4f}  "
                      f"s_max={group_s.max():.4f}")
            n_applied += 1

    print(f"  [AWQ] Applied group scales to {n_applied} weight tensors "
          f"across {len(ln_groups)} LayerNorm groups")

    return weights


def prepare_awq_application(awq_scales: dict) -> tuple[dict, dict]:
    """
    Pre-compute per-tensor AWQ operations from calibration scales.

    This is the **streaming-safe** entry point for AWQ application.  It
    groups per-layer activation scales into attention and MLP blocks,
    computes one group-average scale per block, and returns two lookup
    tables that can be applied one tensor at a time:

    proj_apply : layer_path → group_scale  (np.ndarray)
        For each linear projection weight (q/k/v, gate/up):
        ``W_awq[:, c] = W[:, c] * group_scale[c]``   ← MULTIPLY (amplify salient channels)

    ln_apply   : full_tensor_name → group_scale  (np.ndarray)
        For the LayerNorm that feeds each group:
        ``gamma_awq[c] = gamma[c] / group_scale[c]``  ← DIVIDE  (attenuate LN output)
        Each LayerNorm name appears at most once.

    The two operations together preserve the identity
    ``(X / s) @ (W * s).T = X @ W.T`` for every member, so the model
    output is mathematically equivalent in BF16.  Under INT4 quantization,
    salient channels (large ``s``) now have large weight values that occupy
    more of the quantization range  → smaller relative quantization error
    for the channels that matter most.

    Note: o_proj and down_proj are intentionally excluded — their inputs
    are not LayerNorm outputs, so no inverse compensation is possible.

    Parameters
    ----------
    awq_scales  : output of :func:`collect_activation_scales` / :func:`load_awq_scales`
                  Keys are layer paths, e.g. ``"model.layers.0.self_attn.q_proj"``.

    Returns
    -------
    (proj_apply, ln_apply) — both dicts map string keys to float32 np.ndarray.
    """
    # Layers that directly follow a LayerNorm and can be grouped.
    _ATTN_LEAVES = frozenset({"q_proj", "k_proj", "v_proj"})
    _MLP_LEAVES  = frozenset({"gate_proj", "up_proj", "fc1", "dense_h_to_4h"})

    # Group per-layer scales by their block root (everything above the component).
    # Example: "model.layers.3.self_attn.q_proj"
    #    component parent = "model.layers.3.self_attn"  → leaf = "q_proj"
    #    block root        = "model.layers.3"
    attn_groups: dict[str, list[tuple[str, np.ndarray]]] = {}
    mlp_groups:  dict[str, list[tuple[str, np.ndarray]]] = {}

    for layer_key, scale in awq_scales.items():
        dot = layer_key.rfind(".")
        if dot == -1:
            continue
        leaf   = layer_key[dot + 1:]
        parent = layer_key[:dot]
        dot2   = parent.rfind(".")
        root   = parent[:dot2] if dot2 != -1 else parent

        if leaf in _ATTN_LEAVES:
            attn_groups.setdefault(root, []).append((layer_key, scale))
        elif leaf in _MLP_LEAVES:
            mlp_groups.setdefault(root, []).append((layer_key, scale))

    proj_apply: dict[str, np.ndarray] = {}
    ln_apply:   dict[str, np.ndarray] = {}

    # Standard LN name candidates (in order of preference) per architecture family.
    # The first name is used for ln_apply; callers should check whether the tensor
    # actually exists and skip silently if not.
    _ATTN_LN_NAMES = (
        "{root}.input_layernorm.weight",
        "{root}.ln_1.weight",
        "{root}.layer_norm1.weight",
    )
    _MLP_LN_NAMES = (
        "{root}.post_attention_layernorm.weight",
        "{root}.ln_2.weight",
        "{root}.layer_norm2.weight",
        "{root}.ffn_norm.weight",
    )

    for root, members in attn_groups.items():
        group_s = np.stack([s for _, s in members]).mean(axis=0)
        for lk, _ in members:
            proj_apply[lk] = group_s
        # Use first LN name candidate — compress.py verifies existence at apply time.
        ln_apply[_ATTN_LN_NAMES[0].format(root=root)] = group_s

    for root, members in mlp_groups.items():
        group_s = np.stack([s for _, s in members]).mean(axis=0)
        for lk, _ in members:
            proj_apply[lk] = group_s
        ln_apply[_MLP_LN_NAMES[0].format(root=root)] = group_s

    return proj_apply, ln_apply


def _preceding_norm_name(weight_name: str, weights: dict) -> str | None:
    """
    Guess the name of the LayerNorm whose output feeds this linear weight.

    For ``model.layers.{i}.self_attn.q_proj.weight``:
      → try ``model.layers.{i}.input_layernorm.weight``
         and ``model.layers.{i}.self_attn.q_norm.weight``

    For ``model.layers.{i}.mlp.gate_proj.weight``:
      → try ``model.layers.{i}.post_attention_layernorm.weight``
    """
    parts = weight_name.split(".")
    # Strip ".weight"
    if parts and parts[-1] == "weight":
        parts = parts[:-1]

    # Detect component (self_attn / mlp)
    if "self_attn" in parts or "attention" in parts:
        # Walk back to block root: everything before self_attn/attention
        root = ".".join(parts[:parts.index("self_attn")
                              if "self_attn" in parts
                              else parts.index("attention")])
        candidates = [
            f"{root}.input_layernorm.weight",
            f"{root}.self_attn.q_norm.weight",
            f"{root}.ln_1.weight",
        ]
    elif "mlp" in parts:
        root = ".".join(parts[:parts.index("mlp")])
        candidates = [
            f"{root}.post_attention_layernorm.weight",
            f"{root}.ln_2.weight",
            f"{root}.ffn_norm.weight",
        ]
    else:
        return None

    for c in candidates:
        if c in weights:
            return c
    return None


# ---------------------------------------------------------------------------
# Standalone CLI
# ---------------------------------------------------------------------------

def main():  # pragma: no cover
    ap = argparse.ArgumentParser(
        description="AWQ calibration — compute per-channel activation scales for a model"
    )
    ap.add_argument("--model-dir",  required=True,
                    help="Path to HuggingFace model (BF16 safetensors)")
    ap.add_argument("--output",     required=True,
                    help="Directory to write .awq.npy scale files")
    ap.add_argument("--n-samples",  type=int,   default=64,
                    help="Number of calibration forward passes (default 64)")
    ap.add_argument("--alpha",      type=float, default=0.5,
                    help="Scale exponent α: 0=no AWQ, 0.5=default, 1=full")
    ap.add_argument("--seq-len",    type=int,   default=512,
                    help="Max token length per sample (default 512)")
    ap.add_argument("--calibration-file",
                    help="Optional file with one calibration sentence per line")
    ap.add_argument("--verbose",    action="store_true")
    args = ap.parse_args()

    print("\nSquish AWQ Calibration")
    print(f"  Model:     {args.model_dir}")
    print(f"  Output:    {args.output}")
    print(f"  n_samples: {args.n_samples}")
    print(f"  alpha:     {args.alpha}\n")

    # Load calibration texts
    texts = _DEFAULT_CALIBRATION_TEXTS
    if args.calibration_file:
        with open(args.calibration_file) as f:
            texts = [ln.strip() for ln in f if ln.strip()]
        print(f"  Loaded {len(texts)} calibration texts from {args.calibration_file}")

    # Load the model
    print("Loading model (BF16) ...")
    try:
        from mlx_lm import load as mlx_load
        model, tokenizer = mlx_load(args.model_dir)
    except Exception as e:
        sys.exit(f"Error loading model: {e}")

    # Calibrate
    scales = collect_activation_scales(
        model,
        tokenizer,
        texts=texts,
        n_samples=args.n_samples,
        alpha=args.alpha,
        seq_len=args.seq_len,
        verbose=True,
    )

    # Save
    save_awq_scales(scales, args.output, verbose=True)

    print(f"\nDone.  Run squish.convert with --awq-scales {args.output} to apply.")


if __name__ == "__main__":
    main()
