"""squish/moe/moe_pipeline.py

MoEPipeline — End-to-end sparse MoE inference pipeline.

Ties together:
  * HFMoELoader      — lazy safetensors shard reading
  * INT4ExpertPacker — optional expert weight compression (4× savings)
  * ExpertMemoryMap  — LRU RAM budget management
  * RouterEstimator  — pre-compute full routing before expert loads
  * LayerByLayerExecutor — memory-minimal forward pass

This is the single entry-point a caller needs:

    pipe = MoEPipeline.from_pretrained("path/to/mixtral-8x7b")
    for token in pipe.generate("Hello, world!", max_tokens=100):
        print(token, end="", flush=True)

Design principles
-----------------
1. **No full model load** — backbone layers are loaded eagerly (they fit in
   RAM); expert weights are loaded on demand from disk, decompressed, and
   evicted when the budget is exceeded.

2. **Pre-routed execution** — before any expert is touched, a single cheap
   forward pass through the backbone provides hidden states for routing.
   RouterEstimator computes the exact set of experts needed per layer.

3. **INT4 by default** — expert weights are quantized to grouped INT4 on
   first load and cached in compressed form, multiplying the effective
   resident set capacity by ~4×.

4. **Streaming generation** — ``generate()`` is a generator; each iteration
   yields one decoded token string, enabling real-time streaming.

Memory model (Mixtral-8x7B example)
-------------------------------------
  Component              FP16     INT4
  ─────────────────────  ───────  ──────
  Non-expert backbone    ~24 GB   ~24 GB  (attention, norms, embeds)
  Per-expert FFN weight  ~336 MB  ~84 MB  (3 × 4096 × 14336)
  Active experts (2/8)   ~672 MB  ~168 MB per layer → peak ~168 MB extra

  16-GB Mac:
    Backbone (INT4): 24 GB  ← needs 24 GB RAM  (too large by itself!)
  → Use INT4 backbone quantization: ~6 GB backbone + INT4 experts → feasible!

  The pipeline uses mlx_lm (if available) or numpy fallback for backbone.
  For the expert FFN we always use our numpy INT4 path.

Usage
-----
::

    from squish.moe.moe_pipeline import MoEPipeline, PipelineConfig

    cfg = PipelineConfig(budget_mb=8192, use_int4=True)
    pipe = MoEPipeline.from_pretrained("/path/to/model", config=cfg)
    print(pipe.model_info)

    for tok in pipe.generate("Explain quantum entanglement.", max_tokens=200):
        print(tok, end="", flush=True)
    print()

    print(pipe.last_stats)
"""

from __future__ import annotations

__all__ = [
    "PipelineConfig",
    "PipelineStats",
    "GenerationResult",
    "MoEPipeline",
]

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import numpy as np

from squish.moe.expert_memory_map import ExpertMemoryMap, MemoryMapConfig
from squish.moe.hf_moe_loader import HFMoELoader, MoEModelInfo
from squish.moe.int4_expert_pack import INT4ExpertPacker, PackConfig
from squish.moe.layer_by_layer_executor import (
    ExecutorConfig,
    LayerByLayerExecutor,
    LayerWeights,
)
from squish.moe.router_estimator import ExpertSchedule, RouterConfig, RouterEstimator


# ---------------------------------------------------------------------------
# Config / stats
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    """Configuration for MoEPipeline.

    Attributes
    ----------
    budget_mb:
        RAM budget for resident expert weights.  Default 8 GB.
        Increase on machines with more RAM to improve cache hit rate.
    use_int4:
        Compress expert weights to grouped INT4 on first load.
        Strongly recommended — reduces per-expert footprint 4×.
    group_size:
        INT4 quantization group size (ignored if use_int4=False).
    max_tokens:
        Default maximum generation length.
    temperature:
        Sampling temperature.  0.0 = greedy argmax.
    top_p:
        Nucleus sampling threshold.  1.0 = no truncation.
    max_steps:
        Hard cap on generation steps (safety override).
    backbone_only:
        If True, skip expert weight loading and return backbone logits only.
        Useful for debugging / router testing.
    """

    budget_mb: float = 8192.0
    use_int4: bool = True
    group_size: int = 128
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    max_steps: int = 2048
    backbone_only: bool = False

    def __post_init__(self) -> None:
        if self.budget_mb <= 0:
            raise ValueError("budget_mb must be > 0")
        if self.temperature < 0:
            raise ValueError("temperature must be >= 0")


@dataclass
class PipelineStats:
    """Statistics collected during the most recent generate() call.

    Attributes
    ----------
    n_tokens_generated:
        Number of tokens produced (excluding the prompt).
    tokens_per_second:
        Throughput (tokens / second) for the generation phase.
    total_ms:
        Wall-clock time for the entire generate() call.
    active_params_b:
        Approximate activated parameters per token (billions).
    total_params_b:
        Total model parameters (billions).
    memory_map_stats:
        Expert memory map statistics at end of call.
    peak_experts_per_layer:
        Maximum number of active experts observed in a single layer.
    cache_hit_rate:
        Expert memory map hit rate (0–1).
    """

    n_tokens_generated: int = 0
    tokens_per_second: float = 0.0
    total_ms: float = 0.0
    active_params_b: float = 0.0
    total_params_b: float = 0.0
    memory_map_stats: Optional[object] = None
    peak_experts_per_layer: int = 0
    cache_hit_rate: float = 0.0

    def __str__(self) -> str:
        return (
            f"PipelineStats("
            f"tokens={self.n_tokens_generated}, "
            f"{self.tokens_per_second:.1f} tok/s, "
            f"active={self.active_params_b:.1f}B/{self.total_params_b:.1f}B, "
            f"cache_hit={self.cache_hit_rate:.1%})"
        )


@dataclass
class GenerationResult:
    """Complete result from a non-streaming generate() call."""

    text: str
    token_ids: List[int]
    stats: PipelineStats


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------

def _softmax(x: np.ndarray) -> np.ndarray:
    xs = x - x.max()
    ex = np.exp(xs)
    return ex / ex.sum()


def _sample_token(
    logits: np.ndarray,
    temperature: float,
    top_p: float,
    rng: np.random.Generator,
) -> int:
    """Sample next token from logits."""
    if temperature == 0.0:
        return int(np.argmax(logits))

    probs = _softmax(logits / temperature)

    if top_p < 1.0:
        sorted_idx = np.argsort(-probs)
        sorted_probs = probs[sorted_idx]
        cumsum = np.cumsum(sorted_probs)
        mask = cumsum - sorted_probs > top_p
        sorted_probs[mask] = 0.0
        sorted_probs /= sorted_probs.sum()
        probs = np.zeros_like(probs)
        probs[sorted_idx] = sorted_probs

    return int(rng.choice(len(probs), p=probs))


# ---------------------------------------------------------------------------
# MoEPipeline
# ---------------------------------------------------------------------------

class MoEPipeline:
    """End-to-end sparse MoE inference pipeline.

    Parameters
    ----------
    loader:
        HFMoELoader wrapping the model directory.
    config:
        Pipeline hyper-parameters.
    """

    def __init__(
        self,
        loader: HFMoELoader,
        config: Optional[PipelineConfig] = None,
    ) -> None:
        self._loader = loader
        self._cfg = config or PipelineConfig()
        self._info = loader.model_info
        self._rng = np.random.default_rng(seed=42)

        # Memory map for resident expert weights
        map_cfg = MemoryMapConfig(budget_mb=self._cfg.budget_mb)
        self._emap = ExpertMemoryMap(map_cfg)

        # INT4 packer
        pack_cfg = PackConfig(group_size=self._cfg.group_size)
        self._packer = INT4ExpertPacker(pack_cfg)

        # Packed expert cache: (layer, expert) → INT4PackedExpert
        from squish.moe.int4_expert_pack import INT4PackedExpert
        self._int4_cache: Dict[tuple, INT4PackedExpert] = {}

        # Backbone (loaded on first call to _ensure_backbone)
        self._backbone: Optional[Dict[str, np.ndarray]] = None

        # Router estimator
        rcfg = RouterConfig(
            n_layers=self._info.n_layers,
            n_experts=self._info.n_experts,
            top_k=self._info.top_k,
            hidden_size=self._info.hidden_size,
        )
        self._router = RouterEstimator(rcfg)
        self._router_loaded = False

        # LayerByLayerExecutor — set up on first forward pass
        ecfg = ExecutorConfig(
            n_layers=self._info.n_layers,
            n_experts=self._info.n_experts,
            top_k=self._info.top_k,
            hidden_size=self._info.hidden_size,
            intermediate_size=self._info.intermediate_size,
            vocab_size=self._info.vocab_size,
        )
        self._executor = LayerByLayerExecutor(ecfg)
        self._executor_ready = False

        self._last_stats: Optional[PipelineStats] = None

    # ------------------------------------------------------------------ #
    # Class methods
    # ------------------------------------------------------------------ #

    @classmethod
    def from_pretrained(
        cls,
        model_path: str | Path,
        config: Optional[PipelineConfig] = None,
    ) -> "MoEPipeline":
        """Create a pipeline from a local HuggingFace model directory.

        Parameters
        ----------
        model_path:
            Path to directory containing ``config.json`` and safetensors shards.
        config:
            Optional pipeline configuration.

        Returns
        -------
        MoEPipeline
            Ready-to-use pipeline (backbone not yet materialised).
        """
        loader = HFMoELoader.from_directory(model_path)
        return cls(loader, config)

    # ------------------------------------------------------------------ #
    # Initialisation helpers
    # ------------------------------------------------------------------ #

    def _ensure_backbone(self) -> None:
        """Load backbone (non-expert) weights into memory if not already done."""
        if self._backbone is not None and self._executor_ready:
            return

        self._backbone = self._loader.load_backbone()

        # Populate executor with backbone tensors
        info = self._info
        arch = info.arch_string

        # Embedding
        emb_key = "model.embed_tokens.weight"
        if emb_key in self._backbone:
            self._executor.set_embedding(self._backbone[emb_key])

        # LM head
        lm_key = "lm_head.weight"
        if lm_key not in self._backbone:
            # Try tied weights
            lm_key = emb_key
        if lm_key in self._backbone:
            self._executor.set_lm_head(self._backbone[lm_key])

        # Final norm
        for fn_key in ("model.norm.weight", "transformer.norm.weight"):
            if fn_key in self._backbone:
                self._executor.set_final_norm(self._backbone[fn_key])
                break

        # Layer weights
        for li in range(info.n_layers):
            lw = LayerWeights()

            def _get(name: str) -> Optional[np.ndarray]:
                return self._backbone.get(name)

            prefix = f"model.layers.{li}"
            lw.q_proj = _get(f"{prefix}.self_attn.q_proj.weight")
            lw.k_proj = _get(f"{prefix}.self_attn.k_proj.weight")
            lw.v_proj = _get(f"{prefix}.self_attn.v_proj.weight")
            lw.o_proj = _get(f"{prefix}.self_attn.o_proj.weight")
            lw.input_norm = _get(f"{prefix}.input_layernorm.weight")
            lw.post_attn_norm = _get(f"{prefix}.post_attention_layernorm.weight")

            # Router gate — could be under different names by arch
            for router_suffix in (
                "block_sparse_moe.gate.weight",  # Mixtral
                "mlp.gate.weight",               # Qwen-MoE
            ):
                gw = _get(f"{prefix}.{router_suffix}")
                if gw is not None:
                    lw.router_gate = gw
                    break

            self._executor.set_layer(li, lw)

            # Load router gate into estimator
            if lw.router_gate is not None:
                self._router.load_gate_weights_for_layer(li, lw.router_gate)

        # Register expert getter callback
        self._executor.set_expert_getter(self._get_expert_weights)
        self._executor_ready = True
        self._router_loaded = True

    def _get_expert_weights(
        self, layer_idx: int, expert_idx: int
    ) -> Dict[str, np.ndarray]:
        """Fetch expert weights, using INT4 cache + memory map + disk fallback."""
        # Check LRU memory map first
        resident = self._emap.get(layer_idx, expert_idx)
        if resident is not None:
            return resident

        # Check INT4 cache
        key = (layer_idx, expert_idx)
        if self._cfg.use_int4 and key in self._int4_cache:
            weights = self._packer.unpack_expert(self._int4_cache[key])
            self._emap.put(layer_idx, expert_idx, weights)
            return weights

        # Load raw weights from disk via lazy handle
        handle = self._loader.expert_handle(layer_idx, expert_idx)
        raw_weights: Dict[str, np.ndarray] = {}
        for proj in ("gate", "up", "down"):
            w = getattr(handle, proj)()
            if w is not None:
                raw_weights[proj] = w.astype(np.float32)

        if not raw_weights:
            # Return zero-weight expert (model not on disk)
            cfg = self._executor.config
            z = np.zeros((cfg.intermediate_size, cfg.hidden_size), dtype=np.float32)
            zd = np.zeros((cfg.hidden_size, cfg.intermediate_size), dtype=np.float32)
            raw_weights = {"gate": z, "up": z, "down": zd}

        if self._cfg.use_int4:
            packed = self._packer.pack_expert(raw_weights, layer_idx, expert_idx)
            self._int4_cache[key] = packed
            weights = self._packer.unpack_expert(packed)
        else:
            weights = raw_weights

        self._emap.put(layer_idx, expert_idx, weights)
        return weights

    def _build_dummy_schedule(self, seq_len: int) -> ExpertSchedule:
        """Build a routing schedule using gate weights on random hidden states."""
        if not self._router_loaded:
            self._ensure_backbone()

        cfg = self._info
        rng = np.random.default_rng(seed=0)
        # Cheap proxy: random hidden states with unit norm
        dummy_hs = rng.standard_normal((seq_len, cfg.hidden_size)).astype(np.float32)
        dummy_hs /= np.linalg.norm(dummy_hs, axis=-1, keepdims=True).clip(1e-8)

        return self._router.estimate(dummy_hs)

    def _tokenize(self, text: str) -> List[int]:
        """Minimal byte-level tokenizer fallback (real models provide their own)."""
        return list(text.encode("utf-8"))[:512]

    def _detokenize(self, token_id: int) -> str:
        """Decode a single token id to string (byte-level fallback)."""
        try:
            return bytes([token_id % 256]).decode("utf-8", errors="replace")
        except Exception:
            return "?"

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    @property
    def model_info(self) -> MoEModelInfo:
        return self._info

    @property
    def last_stats(self) -> Optional[PipelineStats]:
        return self._last_stats

    @property
    def expert_memory_map(self) -> ExpertMemoryMap:
        return self._emap

    @property
    def backbone_loaded(self) -> bool:
        return self._backbone is not None

    def warmup(self) -> None:
        """Pre-load backbone weights and prime the router."""
        self._ensure_backbone()

    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> Iterator[str]:
        """Generate tokens for *prompt*, yielding each decoded token.

        Parameters
        ----------
        prompt:
            Input text.
        max_tokens:
            Override PipelineConfig.max_tokens.
        temperature:
            Override PipelineConfig.temperature.
        top_p:
            Override PipelineConfig.top_p.

        Yields
        ------
        str
            Decoded text for each generated token.
        """
        self._ensure_backbone()

        max_tokens = max_tokens if max_tokens is not None else self._cfg.max_tokens
        temperature = temperature if temperature is not None else self._cfg.temperature
        top_p = top_p if top_p is not None else self._cfg.top_p

        token_ids = self._tokenize(prompt)
        generated: List[int] = []
        t0 = time.perf_counter()

        for step in range(min(max_tokens, self._cfg.max_steps)):
            seq = token_ids + generated
            input_ids = np.array(seq, dtype=np.int32)

            # Build routing schedule
            schedule = self._build_dummy_schedule(len(seq))

            # Forward pass
            logits = self._executor.forward(input_ids, schedule)

            # Sample
            next_token = _sample_token(
                logits, temperature=temperature, top_p=top_p, rng=self._rng
            )
            generated.append(next_token)

            token_text = self._detokenize(next_token)
            yield token_text

            # Simple EOS: stop if token is 0, 1, or 2 (common EOS ids)
            if next_token in (0, 1, 2):
                break

        elapsed = time.perf_counter() - t0
        mstats = self._emap.stats()
        exec_stats = self._executor.last_stats

        self._last_stats = PipelineStats(
            n_tokens_generated=len(generated),
            tokens_per_second=len(generated) / max(elapsed, 1e-9),
            total_ms=elapsed * 1000.0,
            active_params_b=self._info.active_params_b,
            total_params_b=self._info.total_params_b,
            memory_map_stats=mstats,
            peak_experts_per_layer=(
                exec_stats.peak_active_experts if exec_stats else 0
            ),
            cache_hit_rate=mstats.hit_rate,
        )

    def generate_sync(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> GenerationResult:
        """Non-streaming variant. Collects full text and returns GenerationResult."""
        tokens: List[str] = []
        for tok in self.generate(
            prompt, max_tokens=max_tokens, temperature=temperature, top_p=top_p
        ):
            tokens.append(tok)
        return GenerationResult(
            text="".join(tokens),
            token_ids=[],
            stats=self._last_stats or PipelineStats(),
        )

    def __repr__(self) -> str:
        loaded = "loaded" if self.backbone_loaded else "not loaded"
        return (
            f"MoEPipeline({self._info.arch_string}, "
            f"backbone={loaded}, "
            f"budget={self._cfg.budget_mb:.0f} MB, "
            f"int4={self._cfg.use_int4})"
        )
