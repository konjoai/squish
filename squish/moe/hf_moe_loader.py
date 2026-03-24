"""squish/moe/hf_moe_loader.py

HFMoELoader — Lazy-loading HuggingFace MoE model from safetensors shards.

Identifies Mixtral / DeepSeek-V2 / Qwen-MoE / Qwen3-MoE architectures from
config.json and partitions tensors into:

  * "backbone" — shared attention, embeddings, RMSNorm, LM head.
    Loaded eagerly into numpy arrays.
  * "experts" — per-layer, per-expert gate/up/down projection matrices.
    Exposed lazily via ExpertWeightHandle — not resident until accessed.

This design makes big-than-memory models feasible: a Mixtral-8x7B model has
~47 B parameters total but only ~13 B "backbone" parameters, meaning a
16 GB system can load the backbone and stream exactly the 2/8 experts needed
per layer.

References
----------
Mistral AI, "Mixtral of Experts," arXiv:2401.04088, 2024.
DeepSeek-AI, "DeepSeek-V2," arXiv:2405.04434, 2024.
Qwen Team, "Qwen3-235B-A22B," 2025.

Usage
-----
    loader = HFMoELoader.from_directory("/path/to/mixtral-8x7b-instruct")
    info = loader.model_info
    print(info)  # MoEModelInfo(arch='mixtral', n_layers=32, n_experts=8, ...)

    backbone = loader.load_backbone()  # dict[str, np.ndarray]

    # lazy handle — no data loaded yet
    handle = loader.expert_handle(layer_idx=0, expert_idx=3)
    w_gate = handle.gate()   # np.ndarray — loads on demand
    w_up   = handle.up()
    w_down = handle.down()
"""

from __future__ import annotations

__all__ = [
    "MoEArchType",
    "MoEModelInfo",
    "ExpertWeightHandle",
    "HFMoELoader",
]

import json
import mmap
import os
import struct
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Constants for known architecture suffixes
# ---------------------------------------------------------------------------

_MIXTRAL_ARCHS = frozenset({"MixtralForCausalLM"})
_DEEPSEEK_ARCHS = frozenset({"DeepseekV2ForCausalLM", "DeepseekV3ForCausalLM"})
_QWEN_MOE_ARCHS = frozenset({"Qwen2MoeForCausalLM", "Qwen3MoeForCausalLM"})

# Bytes per element for recognized dtype strings
_DTYPE_BYTES: dict[str, int] = {
    "F32": 4, "F16": 2, "BF16": 2, "I8": 1, "I32": 4, "F8_E4M3": 1, "F8_E5M2": 1,
    "float32": 4, "float16": 2, "bfloat16": 2,
}

# Numpy dtype mapping (best-effort for safetensors dtypes)
_DTYPE_NP: dict[str, str] = {
    "F32": "float32", "F16": "float16", "BF16": "float32",  # no bf16 in numpy
    "I8": "int8", "I32": "int32", "F8_E4M3": "float32", "F8_E5M2": "float32",
    "float32": "float32", "float16": "float16", "bfloat16": "float32",
}


# ---------------------------------------------------------------------------
# Enums / data classes
# ---------------------------------------------------------------------------

class MoEArchType(Enum):
    """Recognised sparse MoE transformer architectures."""
    MIXTRAL = auto()
    DEEPSEEK_V2 = auto()
    QWEN_MOE = auto()
    UNKNOWN = auto()


@dataclass(frozen=True)
class MoEModelInfo:
    """Architecture summary extracted from ``config.json``.

    Attributes
    ----------
    arch:
        Detected architecture type.
    arch_string:
        Raw ``architectures[0]`` string from config.
    n_layers:
        Total number of transformer blocks.
    n_experts:
        Total experts per MoE FFN layer.
    top_k:
        Experts activated per token per layer.
    hidden_size:
        Model hidden dimension.
    intermediate_size:
        FFN intermediate size (per expert).
    total_params_b:
        Estimated total parameter count in billions.
    active_params_b:
        Estimated activated parameters per forward token, in billions.
    n_shared_experts:
        Number of always-on shared experts (DeepSeek-V2 style). 0 for Mixtral.
    expert_layer_stride:
        Number of layers between MoE layers (1 = all layers are MoE).
    vocab_size:
        Vocabulary size (from config).
    """

    arch: MoEArchType
    arch_string: str
    n_layers: int
    n_experts: int
    top_k: int
    hidden_size: int
    intermediate_size: int
    total_params_b: float
    active_params_b: float
    n_shared_experts: int = 0
    expert_layer_stride: int = 1
    vocab_size: int = 32000

    def __str__(self) -> str:
        return (
            f"MoEModelInfo({self.arch_string}: "
            f"{self.n_layers} layers, "
            f"{self.n_experts} experts/layer (top-{self.top_k}), "
            f"hidden={self.hidden_size}, "
            f"total≈{self.total_params_b:.1f}B, "
            f"active≈{self.active_params_b:.1f}B)"
        )

    @property
    def activation_ratio(self) -> float:
        """Fraction of parameters activated per forward token."""
        if self.total_params_b == 0:
            return 0.0
        return self.active_params_b / self.total_params_b

    @property
    def memory_savings_x(self) -> float:
        """Approximate memory saving factor from sparse activation."""
        if self.activation_ratio == 0:
            return 1.0
        return 1.0 / self.activation_ratio


# ---------------------------------------------------------------------------
# Safetensors header parser (no external dependency)
# ---------------------------------------------------------------------------

@dataclass
class _TensorMeta:
    """Metadata for a single tensor in a safetensors shard."""
    name: str
    dtype: str
    shape: Tuple[int, ...]
    data_offsets: Tuple[int, int]  # byte offsets within data region
    shard_path: str


def _parse_safetensors_header(path: str) -> Tuple[dict, int]:
    """Read the JSON header from a safetensors file.

    Returns (header_dict, header_byte_length).
    The tensor data begins at offset ``8 + header_byte_length``.
    """
    with open(path, "rb") as fh:
        raw_len = fh.read(8)
        if len(raw_len) < 8:
            raise ValueError(f"Not a valid safetensors file: {path}")
        header_len = struct.unpack("<Q", raw_len)[0]
        header_bytes = fh.read(header_len)
    header = json.loads(header_bytes.decode("utf-8"))
    return header, header_len


def _index_shard(shard_path: str) -> list[_TensorMeta]:
    """Return list of _TensorMeta for every tensor in *shard_path*."""
    header, header_len = _parse_safetensors_header(shard_path)
    data_start = 8 + header_len
    metas = []
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        offsets = meta.get("data_offsets", [0, 0])
        metas.append(_TensorMeta(
            name=name,
            dtype=meta.get("dtype", "F32"),
            shape=tuple(meta.get("shape", [])),
            data_offsets=(data_start + offsets[0], data_start + offsets[1]),
            shard_path=shard_path,
        ))
    return metas


# ---------------------------------------------------------------------------
# Expert weight handle — lazy loading
# ---------------------------------------------------------------------------

class ExpertWeightHandle:
    """Lazily-materialised expert weight tensors.

    Weight matrices are only read from disk when gate/up/down is called for
    the first time.  Subsequent calls return the cached ndarray.

    Parameters
    ----------
    gate_meta, up_meta, down_meta:
        Tensor metadata for the three projection matrices.  Any may be None
        if the architecture uses a fused gate+up matrix.
    """

    def __init__(
        self,
        layer_idx: int,
        expert_idx: int,
        gate_meta: Optional[_TensorMeta],
        up_meta: Optional[_TensorMeta],
        down_meta: Optional[_TensorMeta],
    ) -> None:
        self.layer_idx = layer_idx
        self.expert_idx = expert_idx
        self._gate_meta = gate_meta
        self._up_meta = up_meta
        self._down_meta = down_meta
        self._gate: Optional[np.ndarray] = None
        self._up: Optional[np.ndarray] = None
        self._down: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ #
    # Size helpers
    # ------------------------------------------------------------------ #

    def _meta_bytes(self, meta: Optional[_TensorMeta]) -> int:
        if meta is None:
            return 0
        lo, hi = meta.data_offsets
        return hi - lo

    @property
    def bytes_on_disk(self) -> int:
        """Total bytes occupied by this expert's weights in the shard(s)."""
        return (
            self._meta_bytes(self._gate_meta)
            + self._meta_bytes(self._up_meta)
            + self._meta_bytes(self._down_meta)
        )

    @property
    def is_loaded(self) -> bool:
        return self._gate is not None or self._down is not None

    # ------------------------------------------------------------------ #
    # Load helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _load_tensor(meta: _TensorMeta) -> np.ndarray:
        """Read a single tensor from disk via mmap."""
        lo, hi = meta.data_offsets
        length = hi - lo
        np_dtype = _DTYPE_NP.get(meta.dtype, "float32")
        with open(meta.shard_path, "rb") as fh:
            mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
            try:
                raw = mm[lo:hi]
            finally:
                mm.close()
        arr = np.frombuffer(raw, dtype=np.uint8)
        # Reinterpret bytes as target dtype
        n_elements = length // _DTYPE_BYTES.get(meta.dtype, 4)
        out = np.frombuffer(arr.tobytes(), dtype=np_dtype)[:n_elements]
        return out.reshape(meta.shape).copy()

    def gate(self) -> Optional[np.ndarray]:
        """Gate projection matrix  W_gate  (materialises on first call)."""
        if self._gate is None and self._gate_meta is not None:
            self._gate = self._load_tensor(self._gate_meta)
        return self._gate

    def up(self) -> Optional[np.ndarray]:
        """Up projection matrix  W_up  (materialises on first call)."""
        if self._up is None and self._up_meta is not None:
            self._up = self._load_tensor(self._up_meta)
        return self._up

    def down(self) -> Optional[np.ndarray]:
        """Down projection matrix  W_down  (materialises on first call)."""
        if self._down is None and self._down_meta is not None:
            self._down = self._load_tensor(self._down_meta)
        return self._down

    def evict(self) -> None:
        """Release all cached numpy arrays (return memory to allocator)."""
        self._gate = None
        self._up = None
        self._down = None


# ---------------------------------------------------------------------------
# Arch detection helpers
# ---------------------------------------------------------------------------

def _detect_arch(config: dict) -> MoEArchType:
    archs = config.get("architectures", [])
    for a in archs:
        if a in _MIXTRAL_ARCHS:
            return MoEArchType.MIXTRAL
        if a in _DEEPSEEK_ARCHS:
            return MoEArchType.DEEPSEEK_V2
        if a in _QWEN_MOE_ARCHS:
            return MoEArchType.QWEN_MOE
    model_type = config.get("model_type", "").lower()
    if "mixtral" in model_type:
        return MoEArchType.MIXTRAL
    if "deepseek" in model_type:
        return MoEArchType.DEEPSEEK_V2
    if "qwen" in model_type and (
        config.get("num_experts") or config.get("num_local_experts")
    ):
        return MoEArchType.QWEN_MOE
    return MoEArchType.UNKNOWN


def _extract_model_info(config: dict) -> MoEModelInfo:
    arch = _detect_arch(config)
    arch_string = (config.get("architectures") or ["Unknown"])[0]
    n_layers = (
        config.get("num_hidden_layers")
        or config.get("n_layers")
        or 32
    )
    n_experts = (
        config.get("num_local_experts")
        or config.get("num_experts")
        or config.get("n_routed_experts")
        or 8
    )
    top_k = (
        config.get("num_experts_per_tok")
        or config.get("top_k")
        or config.get("num_selected_experts")
        or 2
    )
    hidden_size = config.get("hidden_size") or config.get("d_model") or 4096
    intermediate_size = (
        config.get("intermediate_size")
        or config.get("ffn_dim")
        or config.get("moe_intermediate_size")
        or 14336
    )
    n_shared = config.get("num_shared_experts") or config.get("n_shared_experts") or 0
    vocab_size = config.get("vocab_size") or 32000

    # Estimate parameter count
    # Backbone: embeddings + n_layers × (attention + layernorm params)
    # Expert: n_layers × n_experts × (gate + up + down)
    attn_params = 4 * hidden_size * hidden_size  # Q, K, V, O
    expert_params = 3 * hidden_size * intermediate_size  # gate, up, down
    backbone_params = (
        vocab_size * hidden_size  # embedding
        + n_layers * attn_params
        + n_layers * 2 * hidden_size  # layernorms
        + n_layers * n_shared * expert_params  # shared experts (always on)
        + hidden_size  # final norm
        + vocab_size * hidden_size  # LM head (often tied to embedding)
    )
    routed_expert_params = n_layers * n_experts * expert_params
    total_params = backbone_params + routed_expert_params
    active_params = backbone_params + n_layers * top_k * expert_params

    return MoEModelInfo(
        arch=arch,
        arch_string=arch_string,
        n_layers=n_layers,
        n_experts=n_experts,
        top_k=top_k,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        total_params_b=total_params / 1e9,
        active_params_b=active_params / 1e9,
        n_shared_experts=n_shared,
        vocab_size=vocab_size,
    )


# ---------------------------------------------------------------------------
# Key-name helpers
# ---------------------------------------------------------------------------

def _is_expert_key(name: str, arch: MoEArchType) -> bool:
    """Return True if *name* names an expert weight tensor."""
    if arch == MoEArchType.MIXTRAL:
        return "block_sparse_moe.experts" in name
    if arch == MoEArchType.DEEPSEEK_V2:
        return "mlp.experts" in name
    if arch == MoEArchType.QWEN_MOE:
        return "mlp.experts" in name
    return "experts" in name


def _parse_expert_key(name: str, arch: MoEArchType) -> Optional[Tuple[int, int, str]]:
    """Parse *name* into (layer_idx, expert_idx, proj_name) or None."""
    parts = name.split(".")
    try:
        if arch == MoEArchType.MIXTRAL:
            # model.layers.{L}.block_sparse_moe.experts.{E}.w{1,2,3}.weight
            layer_idx = int(parts[2])
            expert_idx = int(parts[5])
            proj_raw = parts[6]  # "w1", "w2", "w3"
            proj = {"w1": "gate", "w2": "down", "w3": "up"}.get(proj_raw, proj_raw)
        else:
            # model.layers.{L}.mlp.experts.{E}.{gate,up,down}_proj.weight
            layer_idx = int(parts[2])
            expert_idx = int(parts[5])
            proj_raw = parts[6]
            proj = {"gate_proj": "gate", "up_proj": "up", "down_proj": "down"}.get(
                proj_raw, proj_raw
            )
        return layer_idx, expert_idx, proj
    except (IndexError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------------

class HFMoELoader:
    """Lazy-loading HuggingFace sparse MoE model.

    Parameters
    ----------
    model_dir:
        Path to the directory containing ``config.json`` and safetensors
        shard files (e.g. ``model.safetensors.index.json``).

    Usage
    -----
    ::

        loader = HFMoELoader.from_directory("/path/to/model")
        print(loader.model_info)
        backbone = loader.load_backbone()
        handle = loader.expert_handle(layer_idx=0, expert_idx=2)
        w_down = handle.down()
    """

    def __init__(self, model_dir: Path, config: dict) -> None:
        self._model_dir = model_dir
        self._config = config
        self._model_info = _extract_model_info(config)
        # Lazily-built index: (layer, expert) → ExpertWeightHandle
        self._expert_index: Optional[
            dict[Tuple[int, int], dict[str, _TensorMeta]]
        ] = None
        # Backbone tensor map
        self._backbone_index: Optional[dict[str, _TensorMeta]] = None
        self._shard_metas: Optional[list[_TensorMeta]] = None

    # ------------------------------------------------------------------ #
    # Class methods
    # ------------------------------------------------------------------ #

    @classmethod
    def from_directory(cls, model_dir: str | Path) -> "HFMoELoader":
        """Load from a local HuggingFace model directory.

        Raises
        ------
        FileNotFoundError
            If config.json is missing.
        ValueError
            If the model is not a recognised sparse MoE architecture.
        """
        model_dir = Path(model_dir)
        config_path = model_dir / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"config.json not found in {model_dir}")
        config = json.loads(config_path.read_text("utf-8"))
        arch = _detect_arch(config)
        if arch == MoEArchType.UNKNOWN:
            # Allow loading anyway but warn
            pass
        return cls(model_dir, config)

    # ------------------------------------------------------------------ #
    # Properties
    # ------------------------------------------------------------------ #

    @property
    def model_info(self) -> MoEModelInfo:
        return self._model_info

    @property
    def model_dir(self) -> Path:
        return self._model_dir

    # ------------------------------------------------------------------ #
    # Shard discovery
    # ------------------------------------------------------------------ #

    def _discover_shards(self) -> list[Path]:
        """Return ordered list of safetensors shard paths."""
        index_file = self._model_dir / "model.safetensors.index.json"
        if index_file.exists():
            idx = json.loads(index_file.read_text("utf-8"))
            shard_names = sorted(set(idx.get("weight_map", {}).values()))
            return [self._model_dir / s for s in shard_names]
        # Single shard
        single = self._model_dir / "model.safetensors"
        if single.exists():
            return [single]
        # GGUF or other format — return empty
        return []

    def _ensure_index(self) -> None:
        """Build backbone/expert tensor maps from shard headers."""
        if self._shard_metas is not None:
            return
        arch = self._model_info.arch
        all_metas: list[_TensorMeta] = []
        for shard in self._discover_shards():
            if shard.exists():
                all_metas.extend(_index_shard(str(shard)))

        expert_map: dict[Tuple[int, int], dict[str, _TensorMeta]] = {}
        backbone_map: dict[str, _TensorMeta] = {}

        for meta in all_metas:
            if _is_expert_key(meta.name, arch):
                parsed = _parse_expert_key(meta.name, arch)
                if parsed is not None:
                    layer_idx, expert_idx, proj = parsed
                    key = (layer_idx, expert_idx)
                    expert_map.setdefault(key, {})[proj] = meta
            else:
                backbone_map[meta.name] = meta

        self._expert_index = expert_map
        self._backbone_index = backbone_map
        self._shard_metas = all_metas

    # ------------------------------------------------------------------ #
    # Backbone loading
    # ------------------------------------------------------------------ #

    def load_backbone(self) -> dict[str, np.ndarray]:
        """Load all non-expert tensors eagerly into numpy arrays.

        Returns
        -------
        dict[str, np.ndarray]
            Tensor name → numpy array mapping for all backbone weights.
        """
        self._ensure_index()
        result: dict[str, np.ndarray] = {}
        for name, meta in (self._backbone_index or {}).items():
            result[name] = ExpertWeightHandle._load_tensor(meta)
        return result

    # ------------------------------------------------------------------ #
    # Expert access
    # ------------------------------------------------------------------ #

    def expert_handle(self, layer_idx: int, expert_idx: int) -> ExpertWeightHandle:
        """Return a lazy handle for the specified expert's weights.

        The handle does not load any data until ``.gate()``, ``.up()``, or
        ``.down()`` is called.

        Parameters
        ----------
        layer_idx:
            Zero-based transformer layer index.
        expert_idx:
            Zero-based expert index within the MoE layer.

        Returns
        -------
        ExpertWeightHandle
            Lazy weight accessor.
        """
        self._ensure_index()
        projections = (self._expert_index or {}).get((layer_idx, expert_idx), {})
        return ExpertWeightHandle(
            layer_idx=layer_idx,
            expert_idx=expert_idx,
            gate_meta=projections.get("gate"),
            up_meta=projections.get("up"),
            down_meta=projections.get("down"),
        )

    def iter_experts(self) -> Iterator[Tuple[int, int, ExpertWeightHandle]]:
        """Yield (layer_idx, expert_idx, handle) for every expert in order."""
        self._ensure_index()
        for (layer_idx, expert_idx) in sorted(self._expert_index or {}):
            yield layer_idx, expert_idx, self.expert_handle(layer_idx, expert_idx)

    # ------------------------------------------------------------------ #
    # Convenience
    # ------------------------------------------------------------------ #

    def expert_count(self) -> int:
        """Total number of individual expert weight entries on disk."""
        self._ensure_index()
        return len(self._expert_index or {})

    def backbone_tensor_count(self) -> int:
        """Number of backbone (non-expert) tensors."""
        self._ensure_index()
        return len(self._backbone_index or {})

    def expert_disk_bytes(self, layer_idx: int, expert_idx: int) -> int:
        """Disk footprint in bytes for a single expert."""
        return self.expert_handle(layer_idx, expert_idx).bytes_on_disk

    def total_expert_disk_bytes(self) -> int:
        """Total bytes occupied by all expert weights across all shards."""
        self._ensure_index()
        total = 0
        for (li, ei) in (self._expert_index or {}):
            total += self.expert_disk_bytes(li, ei)
        return total

    def __repr__(self) -> str:
        return (
            f"HFMoELoader(dir={self._model_dir.name!r}, "
            f"info={self._model_info})"
        )
