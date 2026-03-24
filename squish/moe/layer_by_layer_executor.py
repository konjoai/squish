"""squish/moe/layer_by_layer_executor.py

LayerByLayerExecutor — Memory-minimal MoE forward pass.

Processes one transformer block at a time.  Between layers the expert
resident set is updated: experts no longer needed are evicted and the
experts required for the next layer are pre-loaded (if a prefetch callback
is provided).  Peak memory is therefore:

    backbone_memory + max_active_experts_per_layer × expert_size

rather than the naïve total model size.

Design
------
The executor is intentionally backend-agnostic: it operates on numpy arrays
throughout, making it runnable on CPU-only hardware (M-series, Linux CPU,
Windows) without MLX or PyTorch as a hard dependency.

Each transformer "block" is described by an :class:`ExecutorLayerPlan` which
bundles:
  * The attention weights (packed or float32).
  * References to the experts to activate, as pre-routed by RouterEstimator.
  * A simple attention implementation (dot-product, no hardware kernels).

Forward pass:

    h = embedding(input_ids)
    for layer in range(n_layers):
        h = attention(h, layer) + h          # standard residual
        h = moe_ffn(h, layer)  + h           # MoE residual
    logits = lm_head(h[:, -1, :])            # next-token distribution

References
----------
Fedus et al., "Switch Transformers: Scaling to Trillion Parameter Models
with Simple and Efficient Sparsity," JMLR 2022.
Eliseev & Panferov, "Fast Inference of Mixture-of-Experts Language Models
with Offloading," arXiv:2312.17238, 2023.

Usage
-----
::

    from squish.moe.layer_by_layer_executor import (
        ExecutorConfig, LayerWeights, LayerByLayerExecutor
    )

    cfg = ExecutorConfig(n_layers=32, n_experts=8, top_k=2, hidden_size=4096)
    executor = LayerByLayerExecutor(cfg)

    # Register layer weights
    for li in range(32):
        executor.set_layer(li, attention_weights=attn_w, norm_weights=norm_w)

    # Set expert weight getter callback
    executor.set_expert_getter(lambda li, ei: {"gate": Wg, "up": Wu, "down": Wd})

    logits = executor.forward(input_ids, routing_schedule)
"""

from __future__ import annotations

__all__ = [
    "ExecutorConfig",
    "LayerWeights",
    "ExecutorStats",
    "LayerByLayerExecutor",
]

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from squish.moe.router_estimator import ExpertSchedule, LayerRouting


# ---------------------------------------------------------------------------
# Config / data structures
# ---------------------------------------------------------------------------

@dataclass
class ExecutorConfig:
    """Configuration for layer-by-layer MoE forward pass.

    Attributes
    ----------
    n_layers:
        Number of transformer blocks.
    n_experts:
        Total experts per MoE layer.
    top_k:
        Experts activated per token per layer.
    hidden_size:
        Model hidden dimension.
    intermediate_size:
        Expert FFN intermediate dimension.
    vocab_size:
        Vocabulary size.
    max_seq_len:
        Maximum sequence length for positional embeddings.
    rms_norm_eps:
        Epsilon for RMSNorm layers.
    use_glu:
        If True, use SwiGLU activation (gate × up); if False, use plain SiLU(up).
    """

    n_layers: int = 32
    n_experts: int = 8
    top_k: int = 2
    hidden_size: int = 4096
    intermediate_size: int = 14336
    vocab_size: int = 32000
    max_seq_len: int = 4096
    rms_norm_eps: float = 1e-5
    use_glu: bool = True

    def __post_init__(self) -> None:
        if self.n_layers < 1:
            raise ValueError("n_layers must be >= 1")
        if self.n_experts < 1:
            raise ValueError("n_experts must be >= 1")
        if self.top_k < 1 or self.top_k > self.n_experts:
            raise ValueError("top_k must be in [1, n_experts]")
        if self.hidden_size < 1:
            raise ValueError("hidden_size must be >= 1")


@dataclass
class LayerWeights:
    """Weight tensors for a single transformer block.

    Attributes
    ----------
    q_proj, k_proj, v_proj, o_proj:
        Attention projection matrices of shape (hidden, hidden).
    input_norm, post_attn_norm:
        RMSNorm weight vectors of shape (hidden,).
    router_gate:
        Router weight of shape (n_experts, hidden).
    shared_gate, shared_up, shared_down:
        Optional shared-expert weights (DeepSeek-V2 style).
    """

    q_proj: Optional[np.ndarray] = None
    k_proj: Optional[np.ndarray] = None
    v_proj: Optional[np.ndarray] = None
    o_proj: Optional[np.ndarray] = None
    input_norm: Optional[np.ndarray] = None
    post_attn_norm: Optional[np.ndarray] = None
    router_gate: Optional[np.ndarray] = None
    # Optional shared experts (always-on)
    shared_gate: Optional[np.ndarray] = None
    shared_up: Optional[np.ndarray] = None
    shared_down: Optional[np.ndarray] = None


@dataclass
class ExecutorStats:
    """Execution statistics collected during a forward pass.

    Attributes
    ----------
    n_layers_executed:
        Number of layers processed.
    n_expert_loads:
        Total expert weight fetches from the expert getter.
    n_expert_activations:
        Total (token, expert) dispatch events.
    total_flops_approx:
        Approximate FLOPs for the entire forward pass.
    elapsed_ms:
        Wall-clock time in milliseconds.
    peak_active_experts:
        Maximum simultaneously active experts observed in a single layer.
    """

    n_layers_executed: int = 0
    n_expert_loads: int = 0
    n_expert_activations: int = 0
    total_flops_approx: int = 0
    elapsed_ms: float = 0.0
    peak_active_experts: int = 0


# ---------------------------------------------------------------------------
# Math primitives
# ---------------------------------------------------------------------------

def _rms_norm(x: np.ndarray, weight: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Root mean square normalisation: x / rms(x) * weight."""
    rms = np.sqrt((x ** 2).mean(axis=-1, keepdims=True) + eps)
    return (x / rms) * weight


def _silu(x: np.ndarray) -> np.ndarray:
    """SiLU / Swish activation: x * σ(x)."""
    return x * (1.0 / (1.0 + np.exp(-x.clip(-30, 30))))


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    xs = x - x.max(axis=axis, keepdims=True)
    ex = np.exp(xs)
    return ex / ex.sum(axis=axis, keepdims=True)


def _scaled_dot_product_attention(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
) -> np.ndarray:
    """Vanilla scaled dot-product attention (no masking for simplicity).

    Parameters
    ----------
    q, k, v: (seq_len, head_dim)

    Returns
    -------
    (seq_len, head_dim)
    """
    scale = q.shape[-1] ** -0.5
    attn = _softmax(q @ k.T * scale)
    return attn @ v


# ---------------------------------------------------------------------------
# Main executor
# ---------------------------------------------------------------------------

class LayerByLayerExecutor:
    """Memory-minimal layer-by-layer MoE forward pass.

    Parameters
    ----------
    config:
        Model and execution configuration.
    """

    def __init__(self, config: ExecutorConfig) -> None:
        self._config = config
        self._layers: Dict[int, LayerWeights] = {}
        self._embedding: Optional[np.ndarray] = None   # (vocab_size, hidden_size)
        self._lm_head: Optional[np.ndarray] = None     # (vocab_size, hidden_size)
        self._final_norm: Optional[np.ndarray] = None  # (hidden_size,)
        # Callback: (layer_idx, expert_idx) → {"gate": W, "up": W, "down": W}
        self._expert_getter: Optional[
            Callable[[int, int], Dict[str, np.ndarray]]
        ] = None
        # Optional pre-fetcher for next-layer experts
        self._expert_prefetcher: Optional[
            Callable[[int, List[int]], None]
        ] = None

    # ------------------------------------------------------------------ #
    # Setup
    # ------------------------------------------------------------------ #

    def set_embedding(self, embedding: np.ndarray) -> None:
        """Set the token embedding matrix of shape (vocab_size, hidden_size)."""
        self._embedding = embedding.astype(np.float32)

    def set_lm_head(self, lm_head: np.ndarray) -> None:
        """Set the LM head weight matrix of shape (vocab_size, hidden_size)."""
        self._lm_head = lm_head.astype(np.float32)

    def set_final_norm(self, norm: np.ndarray) -> None:
        """Set the final RMSNorm weight vector of shape (hidden_size,)."""
        self._final_norm = norm.astype(np.float32)

    def set_layer(self, layer_idx: int, weights: LayerWeights) -> None:
        """Register weights for transformer block *layer_idx*."""
        self._layers[layer_idx] = weights

    def set_expert_getter(
        self,
        getter: Callable[[int, int], Dict[str, np.ndarray]],
    ) -> None:
        """Register the callback that returns expert weights on demand.

        Signature: ``getter(layer_idx, expert_idx) → dict["gate"|"up"|"down", array]``
        """
        self._expert_getter = getter

    def set_expert_prefetcher(
        self,
        prefetcher: Callable[[int, List[int]], None],
    ) -> None:
        """Register an optional async prefetch callback.

        Called with ``(next_layer_idx, [expert_ids])`` before the current
        layer executes its expert FFN.  Intended for background IO.
        """
        self._expert_prefetcher = prefetcher

    # ------------------------------------------------------------------ #
    # Core primitives
    # ------------------------------------------------------------------ #

    def _attention_layer(
        self,
        h: np.ndarray,
        lw: LayerWeights,
    ) -> np.ndarray:
        """Single-head attention (simplified).  Returns residual output."""
        cfg = self._config
        if lw.q_proj is None:
            return h  # no-op if weights not registered

        norm_w = lw.input_norm if lw.input_norm is not None else np.ones(cfg.hidden_size, dtype=np.float32)
        h_norm = _rms_norm(h, norm_w, cfg.rms_norm_eps)

        q = h_norm @ lw.q_proj.T
        k = h_norm @ lw.k_proj.T
        v = h_norm @ lw.v_proj.T
        attn_out = _scaled_dot_product_attention(q, k, v)
        out = attn_out @ lw.o_proj.T
        return out

    def _expert_ffn(
        self,
        token_vec: np.ndarray,
        gate_w: np.ndarray,
        up_w: np.ndarray,
        down_w: np.ndarray,
    ) -> np.ndarray:
        """Compute SwiGLU or SiLU FFN for one token through one expert.

        Parameters
        ----------
        token_vec:
            Shape (hidden_size,).
        gate_w, up_w, down_w:
            Expert projection matrices.
        """
        gate_out = token_vec @ gate_w.T    # (intermediate_size,)
        up_out = token_vec @ up_w.T        # (intermediate_size,)
        if self._config.use_glu:
            hidden = _silu(gate_out) * up_out
        else:
            hidden = _silu(up_out)
        return hidden @ down_w.T           # (hidden_size,)

    def _moe_layer(
        self,
        h: np.ndarray,
        layer_idx: int,
        routing: LayerRouting,
        lw: LayerWeights,
        stats: ExecutorStats,
    ) -> np.ndarray:
        """Process the MoE FFN sub-layer for all tokens.

        Parameters
        ----------
        h:
            Hidden states of shape (seq_len, hidden_size).
        layer_idx:
            Current layer index (used for expert fetching).
        routing:
            Pre-computed routing decisions for this layer.
        lw:
            Layer weight container.
        stats:
            Mutable stats accumulator.
        """
        cfg = self._config
        seq_len = h.shape[0]

        # Optional post-attention RMS norm before MoE FFN
        norm_w = (
            lw.post_attn_norm
            if lw.post_attn_norm is not None
            else np.ones(cfg.hidden_size, dtype=np.float32)
        )
        h_norm = _rms_norm(h, norm_w, cfg.rms_norm_eps)

        output = np.zeros_like(h_norm)

        # Activate shared experts if present (always-on, weight combined additively)
        if lw.shared_gate is not None and lw.shared_up is not None and lw.shared_down is not None:
            for t in range(seq_len):
                shared_out = self._expert_ffn(
                    h_norm[t], lw.shared_gate, lw.shared_up, lw.shared_down
                )
                output[t] += shared_out

        # Routed experts
        expert_cache: Dict[int, Dict[str, np.ndarray]] = {}

        for t in range(seq_len):
            for slot in range(cfg.top_k):
                eid = int(routing.token_assignments[t, slot])
                weight = float(routing.token_weights[t, slot])

                # Fetch expert weights (cached within this layer call)
                if eid not in expert_cache:
                    if self._expert_getter is None:
                        raise RuntimeError(
                            "No expert_getter registered. "
                            "Call set_expert_getter() before forward()."
                        )
                    expert_cache[eid] = self._expert_getter(layer_idx, eid)
                    stats.n_expert_loads += 1

                ew = expert_cache[eid]
                gate_w = ew.get("gate", np.zeros((cfg.intermediate_size, cfg.hidden_size), dtype=np.float32))
                up_w = ew.get("up", np.zeros((cfg.intermediate_size, cfg.hidden_size), dtype=np.float32))
                down_w = ew.get("down", np.zeros((cfg.hidden_size, cfg.intermediate_size), dtype=np.float32))

                expert_out = self._expert_ffn(h_norm[t], gate_w, up_w, down_w)
                output[t] += weight * expert_out
                stats.n_expert_activations += 1

        stats.peak_active_experts = max(stats.peak_active_experts, len(expert_cache))
        return output

    # ------------------------------------------------------------------ #
    # Forward pass
    # ------------------------------------------------------------------ #

    def forward(
        self,
        input_ids: np.ndarray,
        schedule: ExpertSchedule,
    ) -> np.ndarray:
        """Compute next-token logits for *input_ids*.

        Parameters
        ----------
        input_ids:
            Integer token array of shape ``(seq_len,)`` or ``(1, seq_len)``.
        schedule:
            Expert routing schedule from RouterEstimator.

        Returns
        -------
        np.ndarray
            Float32 logits of shape ``(vocab_size,)`` for the last token.

        Raises
        ------
        RuntimeError
            If embedding or expert_getter is not configured.
        """
        if self._embedding is None:
            raise RuntimeError("Embedding not set. Call set_embedding() first.")

        start = time.perf_counter()
        stats = ExecutorStats()
        cfg = self._config

        input_ids = np.asarray(input_ids, dtype=np.int32).ravel()
        seq_len = len(input_ids)

        # Embed
        h = self._embedding[input_ids].astype(np.float32)  # (seq_len, hidden_size)

        # Layer loop
        for li in range(cfg.n_layers):
            lw = self._layers.get(li, LayerWeights())

            # Prefetch next layer's experts in background
            if self._expert_prefetcher is not None and li + 1 < cfg.n_layers:
                next_ids = schedule.experts_for_layer(li + 1).tolist()
                if next_ids:
                    try:
                        self._expert_prefetcher(li + 1, next_ids)
                    except Exception:  # noqa: BLE001
                        pass

            # Attention sub-layer
            attn_delta = self._attention_layer(h, lw)
            h = h + attn_delta

            # MoE FFN sub-layer
            routing = schedule.routings.get(li)
            if routing is not None:
                ffn_delta = self._moe_layer(h, li, routing, lw, stats)
                h = h + ffn_delta

            stats.n_layers_executed += 1

        # Final norm
        if self._final_norm is not None:
            h = _rms_norm(h, self._final_norm, cfg.rms_norm_eps)

        # LM head: take last token only
        last_token = h[-1, :]  # (hidden_size,)
        if self._lm_head is not None:
            logits = last_token @ self._lm_head.T  # (vocab_size,)
        else:
            # Fall back to dot with embedding (tied weights)
            logits = last_token @ self._embedding.T

        stats.elapsed_ms = (time.perf_counter() - start) * 1000.0
        self._last_stats = stats
        return logits.astype(np.float32)

    # ------------------------------------------------------------------ #
    # Post-forward stats
    # ------------------------------------------------------------------ #

    @property
    def last_stats(self) -> Optional[ExecutorStats]:
        """Return statistics from the most recent forward call, or None."""
        return getattr(self, "_last_stats", None)

    @property
    def config(self) -> ExecutorConfig:
        return self._config

    def __repr__(self) -> str:
        return (
            f"LayerByLayerExecutor("
            f"n_layers={self._config.n_layers}, "
            f"n_experts={self._config.n_experts}, "
            f"top_k={self._config.top_k})"
        )
