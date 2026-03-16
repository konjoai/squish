"""
dev/scripts/make_synthetic_model.py
===================================
Generates a minimal 2-layer Qwen2-compatible model directory for use as a
test fixture.  The model is tiny (64-dim hidden, 256-token vocab) so the
resulting safetensors file is < 30 KB.

Usage
-----
    python dev/scripts/make_synthetic_model.py

Output directory (deleted and recreated on each run)::

    tests/fixtures/synthetic_model/
        config.json
        tokenizer_config.json
        model.safetensors

The fixture is committed to the repo and consumed by tests that need a
real (but tiny) Qwen2 model directory on disk — e.g., tests for
`squish info`, the compressed loader's Tier-0 native-format detection,
and convert.py's weight inspection utilities.
"""
from __future__ import annotations

import json
import pathlib
import shutil
import struct

import numpy as np

# ── Parameters for the tiny model ─────────────────────────────────────────────
_HIDDEN   = 64
_HEADS    = 4          # attention heads
_KV_HEADS = 2          # key-value heads
_FFN      = 128        # intermediate_size
_VOCAB    = 256
_LAYERS   = 2
_HEAD_DIM = _HIDDEN // _HEADS   # 16
_KV_DIM   = _KV_HEADS * _HEAD_DIM   # 32

_OUT_DIR = pathlib.Path(__file__).parent.parent.parent / "tests" / "fixtures" / "synthetic_model"

# ── Weight tensor definitions ──────────────────────────────────────────────────

def _model_tensors() -> dict[str, np.ndarray]:
    """Return OrderedDict of name → float32 ndarray for a 2-layer Qwen2 model."""
    rng = np.random.default_rng(42)

    def _randn(*shape: int) -> np.ndarray:
        return rng.standard_normal(shape).astype(np.float32) * 0.02

    def _ones(*shape: int) -> np.ndarray:
        return np.ones(shape, dtype=np.float32)

    tensors: dict[str, np.ndarray] = {}

    # Embedding
    tensors["model.embed_tokens.weight"] = _randn(_VOCAB, _HIDDEN)

    for i in range(_LAYERS):
        p = f"model.layers.{i}"
        # Self-attention projections
        tensors[f"{p}.self_attn.q_proj.weight"] = _randn(_HIDDEN, _HIDDEN)
        tensors[f"{p}.self_attn.k_proj.weight"] = _randn(_KV_DIM, _HIDDEN)
        tensors[f"{p}.self_attn.v_proj.weight"] = _randn(_KV_DIM, _HIDDEN)
        tensors[f"{p}.self_attn.o_proj.weight"] = _randn(_HIDDEN, _HIDDEN)
        # Bias vectors (q/k/v have biases in Qwen2)
        tensors[f"{p}.self_attn.q_proj.bias"] = _randn(_HIDDEN)
        tensors[f"{p}.self_attn.k_proj.bias"] = _randn(_KV_DIM)
        tensors[f"{p}.self_attn.v_proj.bias"] = _randn(_KV_DIM)
        # MLP
        tensors[f"{p}.mlp.gate_proj.weight"] = _randn(_FFN, _HIDDEN)
        tensors[f"{p}.mlp.up_proj.weight"]   = _randn(_FFN, _HIDDEN)
        tensors[f"{p}.mlp.down_proj.weight"] = _randn(_HIDDEN, _FFN)
        # Layer norms
        tensors[f"{p}.input_layernorm.weight"]           = _ones(_HIDDEN)
        tensors[f"{p}.post_attention_layernorm.weight"]  = _ones(_HIDDEN)

    # Final norm (no lm_head — tie_word_embeddings=true)
    tensors["model.norm.weight"] = _ones(_HIDDEN)

    return tensors


# ── Minimal safetensors writer ─────────────────────────────────────────────────
# The safetensors format: 8-byte LE uint64 header length + JSON header + data.

def _write_safetensors(path: pathlib.Path, tensors: dict[str, np.ndarray]) -> None:
    """Write a safetensors v1 file.  Only supports float32 tensors."""
    DTYPE_MAP = {np.float32: "F32"}

    # 1. Lay out the data region and build the header JSON
    metadata: dict = {}
    data_parts: list[bytes] = []
    offset = 0
    for name, arr in tensors.items():
        flat = arr.flatten()
        raw  = flat.tobytes()
        dtype_str = DTYPE_MAP.get(arr.dtype.type, "F32")
        metadata[name] = {
            "dtype": dtype_str,
            "shape": list(arr.shape),
            "data_offsets": [offset, offset + len(raw)],
        }
        data_parts.append(raw)
        offset += len(raw)

    header_json = json.dumps({"__metadata__": {}, **metadata}, separators=(",", ":")).encode("utf-8")
    # Pad header to 8-byte alignment
    pad = (8 - len(header_json) % 8) % 8
    header_json += b" " * pad

    with open(path, "wb") as f:
        # 8-byte LE header length
        f.write(struct.pack("<Q", len(header_json)))
        f.write(header_json)
        for part in data_parts:
            f.write(part)


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    if _OUT_DIR.exists():
        shutil.rmtree(_OUT_DIR)
    _OUT_DIR.mkdir(parents=True)

    # config.json
    config = {
        "model_type": "qwen2",
        "hidden_size": _HIDDEN,
        "num_hidden_layers": _LAYERS,
        "num_attention_heads": _HEADS,
        "num_key_value_heads": _KV_HEADS,
        "intermediate_size": _FFN,
        "vocab_size": _VOCAB,
        "max_position_embeddings": 512,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1000000.0,
        "rope_traditional": False,
        "tie_word_embeddings": True,
        "torch_dtype": "float32",
    }
    (_OUT_DIR / "config.json").write_text(json.dumps(config, indent=2))

    # tokenizer_config.json (minimal — needed by mlx_lm.load)
    tokenizer_config = {
        "model_type": "qwen2",
        "tokenizer_class": "Qwen2Tokenizer",
        "bos_token": "<|endoftext|>",
        "eos_token": "<|endoftext|>",
        "pad_token": "<|endoftext|>",
        "unk_token": "<|endoftext|>",
    }
    (_OUT_DIR / "tokenizer_config.json").write_text(json.dumps(tokenizer_config, indent=2))

    # model.safetensors
    tensors = _model_tensors()
    _write_safetensors(_OUT_DIR / "model.safetensors", tensors)

    total_params = sum(a.size for a in tensors.values())
    file_bytes = (_OUT_DIR / "model.safetensors").stat().st_size

    print(f"Written to: {_OUT_DIR}")
    print(f"  Tensors : {len(tensors)}")
    print(f"  Params  : {total_params:,}")
    print(f"  File    : {file_bytes / 1024:.1f} KB")


if __name__ == "__main__":
    main()
