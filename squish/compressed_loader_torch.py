"""squish/compressed_loader_torch.py — Load a squish npy-dir into a HuggingFace model.

This module provides the Linux/CUDA counterpart to the MLX-specific
``squish.quant.compressed_loader``.  It loads a squish-compressed npy-dir
(produced by ``squish convert`` / ``squish it``) by:

  1. Instantiating the HuggingFace model in the requested dtype on *device*.
  2. Iterating the model's state dict; for each key look up the matching
     tensor in the npy-dir via ``loader_utils._dequantize_npy``.
  3. Patching the state dict in-place and loading it back with
     ``model.load_state_dict(..., strict=False)``.
  4. Returning ``(model, tokenizer)`` with the same contract as
     ``transformers.AutoModelForCausalLM.from_pretrained``.

Public API
----------
    load_compressed_model_torch(npy_dir, model_dir,
                                device, dtype, verbose) → (model, tokenizer)

Notes
-----
- *npy_dir* is the ``tensors/`` sub-directory of a squish npy-dir.  If the
  path has a ``tensors/`` child that directory is used automatically.
- Tensors not found in the npy-dir fall back to the original weights from
  *model_dir* (safetensors BF16 / FP16).  This preserves embeddings and
  other non-compressed tensors transparently.
- The function is pure Python + numpy + torch; it does not require mlx.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import torch


def _import_torch():
    try:
        import torch
        return torch
    except ImportError as exc:
        raise RuntimeError(
            "squish.compressed_loader_torch requires PyTorch. "
            "Install with: pip install torch transformers"
        ) from exc


def _collect_npy_keys(tensor_dir: Path) -> dict[str, Path]:
    """Return a mapping of weight base-key → npy path prefix.

    For example a key ``model.layers.0.mlp.gate_proj.weight`` would have
    *one* of these files in tensor_dir:
        ``model_layers_0_mlp_gate_proj_weight__q4a.npy``
        ``model_layers_0_mlp_gate_proj_weight__pt.npy``
        ... etc.

    Returns a dict: HF weight key → tensor_dir (so consumers can call
    ``_dequantize_npy(tensor_dir, hf_key_as_stem)``).
    """
    # Build the set of stems present in tensor_dir.
    # ``_dequantize_npy`` receives the key with dots replaced by underscores.
    stems: set[str] = set()
    for f in tensor_dir.iterdir():
        if f.name.endswith(".npy") or f.name.endswith(".npy.zst"):
            stem = f.name
            for suffix in ("__q4a.npy", "__s4a.npy", "__z4a.npy",
                           "__q4.npy",  "__s4.npy",  "__pt.npy",
                           "__nf4.npy", "__s_nf4.npy", "__q.npy", "__s.npy",
                           "__vq_idx.npy", "__pt_df11.npy", "__s4_df11.npy",
                           "__q4a.npy.zst", "__pt.npy.zst"):
                if stem.endswith(suffix):
                    stems.add(stem[: -len(suffix)])
                    break
    return {k: tensor_dir for k in stems}


def load_compressed_model_torch(
    npy_dir: str | Path,
    model_dir: str | Path,
    device: str = "cpu",
    dtype: str = "float16",
    verbose: bool = True,
) -> tuple:
    """Load a squish npy-dir into a HuggingFace CausalLM model.

    Parameters
    ----------
    npy_dir:
        Path to the squish npy-dir root (contains ``tensors/`` and
        ``manifest.json``) **or** directly to the ``tensors/`` sub-directory.
    model_dir:
        Path to the original BF16/FP16 HuggingFace model directory.  Used to
        instantiate the model architecture and to fall back to original weights
        for tensors not present in the npy-dir.
    device:
        Target device, e.g. ``"cuda"``, ``"cuda:0"``, ``"cpu"``.
    dtype:
        Weight dtype for the HF model: ``"float16"`` (default) or ``"bfloat16"``.
    verbose:
        Print progress information.

    Returns
    -------
    (model, tokenizer)
        Same contract as ``transformers.AutoModelForCausalLM.from_pretrained``.
    """
    torch = _import_torch()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from squish.io.loader_utils import _dequantize_npy

    npy_dir  = Path(npy_dir)
    model_dir = Path(model_dir)

    # Resolve tensor_dir: accept both the npy-dir root and tensors/ directly.
    tensor_dir = npy_dir / "tensors" if (npy_dir / "tensors").is_dir() else npy_dir
    if not tensor_dir.is_dir():
        raise FileNotFoundError(
            f"tensors directory not found at {npy_dir} or {npy_dir / 'tensors'}"
        )

    _dtype_map = {
        "float32":  torch.float32,
        "float16":  torch.float16,
        "bfloat16": torch.bfloat16,
    }
    torch_dtype = _dtype_map.get(dtype, torch.float16)

    t0 = time.perf_counter()

    if verbose:
        print(f"  [compressed_loader_torch] npy_dir={tensor_dir.parent}")
        print(f"  [compressed_loader_torch] device={device}  dtype={dtype}")

    # ── Step 1: collect which keys are in the npy-dir ──────────────────────
    npy_keys = _collect_npy_keys(tensor_dir)
    if verbose:
        print(f"  [compressed_loader_torch] found {len(npy_keys)} tensors in npy-dir")

    # ── Step 2: load the HF model skeleton (meta device → no weights allocated) ──
    if verbose:
        print(f"  [compressed_loader_torch] loading model skeleton from {model_dir}")
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        torch_dtype=torch_dtype,
        device_map=device,
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model.eval()

    # ── Step 3: patch compressed weights from npy-dir ─────────────────────
    # HF state dict keys use dots; npy stems use underscores.
    patched = 0
    with torch.no_grad():
        sd = model.state_dict()
        patched_sd: dict[str, "torch.Tensor"] = {}

        for hf_key in sd.keys():
            stem = hf_key.replace(".", "_")
            if stem in npy_keys:
                try:
                    arr_f32: np.ndarray = _dequantize_npy(tensor_dir, stem)
                    patched_sd[hf_key] = (
                        torch.from_numpy(np.ascontiguousarray(arr_f32))
                        .to(device=device, dtype=torch_dtype)
                    )
                    patched += 1
                except Exception as exc:
                    if verbose:
                        print(f"  [compressed_loader_torch] WARN: {hf_key}: {exc}")

        if patched_sd:
            result = model.load_state_dict(patched_sd, strict=False)
            if verbose and result.missing_keys:
                print(
                    f"  [compressed_loader_torch] {len(result.missing_keys)} "
                    "keys left as original (embeddings, norms, etc.)"
                )

    elapsed = time.perf_counter() - t0
    if verbose:
        print(
            f"  [compressed_loader_torch] patched {patched}/{len(sd)} tensors "
            f"in {elapsed:.2f}s"
        )

    return model, tokenizer
