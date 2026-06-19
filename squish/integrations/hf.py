"""squish.integrations.hf — HuggingFace Transformers drop-in cache.

Install the optional extras for HF integration:

    pip install squish-ai[hf]   # installs transformers + tokenizers

Public surface
--------------
SquishCache      HF-compatible KV cache that uses squish's quantized storage.
                 Subclasses ``transformers.DynamicCache`` when transformers is
                 installed; otherwise exposes the same interface stand-alone so
                 tests run without the HF dependency.

squish_compress  Decorator that patches any ``AutoModelForCausalLM`` to use
                 ``SquishCache`` automatically on every forward pass.

Usage
-----
    from squish.integrations.hf import SquishCache, squish_compress

    # Option 1 — create the cache and pass it to generate()
    cache = SquishCache(quantization="int4", sink_token_count=4)
    outputs = model.generate(input_ids, past_key_values=cache, use_cache=True)

    # Option 2 — patch the model once, then use normally
    @squish_compress(quantization="int8", sink_token_count=4)
    def load_model():
        return AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

    model = load_model()
    # model now uses SquishCache automatically
"""
from __future__ import annotations

import logging
import numpy as np
from typing import Any

from squish.kv.kv_cache import (
    HadamardKVCache,
    QuantizedKVCache,
    CompressionResult,
    make_kv_cache,
)

_log = logging.getLogger(__name__)

# ── Try to import HF DynamicCache for real subclassing ────────────────────────
try:
    from transformers.cache_utils import DynamicCache as _HFDynamicCache  # type: ignore
    _HF_AVAILABLE = True  # pragma: no cover - transformers >=5.x dropped cache_utils.DynamicCache
except ImportError:
    _HFDynamicCache = object   # type: ignore[misc,assignment]
    _HF_AVAILABLE = False


# ---------------------------------------------------------------------------
# SquishCache
# ---------------------------------------------------------------------------


class SquishCache(_HFDynamicCache):
    """HuggingFace-compatible KV cache backed by ``squish.kv.kv_cache``.

    Implements the ``DynamicCache`` protocol (``update``, ``get_seq_length``,
    ``get_usable_length``) so it can be passed as ``past_key_values`` to any
    HF model's ``generate()`` or ``forward()`` without changes to model code.

    Squish compresses tokens that leave the recent FP16 window according to
    the chosen ``quantization`` tier; ``sink_token_count`` tokens at the very
    start of the sequence are kept permanently at FP16 (StreamingLLM).

    Parameters
    ----------
    quantization : "int8" | "int4" | "int2"  (default "int8")
        Storage precision for the compressed old-tier buffer.
    rotate : bool (default True)
        Use ``HadamardKVCache`` (recommended).  The QuaRot rotation spreads
        outlier energy before quantization, recovering 10-15 dB of SNR at INT2.
    window : int (default 64)
        FP16 recent-window size.  Tokens older than ``window`` are compressed.
    sink_token_count : int (default 4)
        Number of leading tokens kept permanently at FP16.  StreamingLLM shows
        4 tokens capture 45-55 % of total attention; preserving them prevents
        the perplexity spike that low-bit quantization otherwise causes.
    precision_map : dict | None (default None)
        Per-layer quantization override, e.g. ``{"0-3": "fp16", "4-28": "int4"}``.
    seed : int (default 42)
        Hadamard rotation seed (ignored when ``rotate=False``).

    Attributes
    ----------
    _squish_cache : QuantizedKVCache
        The underlying squish cache.  Access it for advanced operations.
    """

    def __init__(
        self,
        quantization: str = "int8",
        rotate: bool = True,
        window: int = 64,
        sink_token_count: int = 4,
        precision_map: "dict | None" = None,
        seed: int = 42,
    ) -> None:
        if _HF_AVAILABLE:  # pragma: no cover - only with older transformers exposing DynamicCache
            super().__init__()

        _VALID = {"int8", "int4", "int2"}
        if quantization not in _VALID:
            raise ValueError(
                f"quantization must be one of {sorted(_VALID)}, got {quantization!r}"
            )

        self._squish_quantization  = quantization
        self._squish_rotate        = rotate
        self._squish_window        = window
        self._squish_sink          = sink_token_count
        self._squish_precision_map = precision_map
        self._squish_seed          = seed

        # Lazily built on first update() call (n_layers not known until then).
        self._squish_cache: "QuantizedKVCache | None" = None
        self._squish_n_layers: int = 0

    # ── HF DynamicCache protocol ───────────────────────────────────────────

    def update(
        self,
        key_states: Any,
        value_states: Any,
        layer_idx: int,
        cache_kwargs: "dict | None" = None,
    ) -> "tuple[Any, Any]":
        """Append new K/V states and return the full accumulated cache.

        This is the primary entry-point for HF models.  ``key_states`` and
        ``value_states`` are either ``torch.Tensor`` or ``numpy.ndarray``
        with shape ``(batch, n_heads, seq_len, head_dim)``.

        Returns the same type and shape as the inputs, but with the full
        sequence (all past + current tokens) concatenated on the seq_len axis.
        """
        # Convert to numpy for squish (squish is numpy-native)
        k_np = _to_numpy(key_states)    # (batch, n_heads, T_new, head_dim)
        v_np = _to_numpy(value_states)

        batch, n_heads, T_new, head_dim = k_np.shape
        if batch != 1:
            raise ValueError(
                "SquishCache only supports batch_size=1 (single-request)"
            )
        # Flatten batch dimension: (n_heads, T_new, head_dim)
        k_np = k_np[0]
        v_np = v_np[0]

        # Build the squish cache lazily; grow its layer list on demand.
        # n_layers is unknown at construction time — the first update() call
        # for each new layer_idx extends the list.
        if self._squish_cache is None:
            cls = HadamardKVCache if self._squish_rotate else QuantizedKVCache
            kwargs: dict = dict(
                n_layers=layer_idx + 1,
                window=self._squish_window,
                mode=self._squish_quantization,
                sink_token_count=self._squish_sink,
                precision_map=self._squish_precision_map,
            )
            if self._squish_rotate:
                kwargs["seed"] = self._squish_seed
            self._squish_cache = cls(**kwargs)
            self._squish_n_layers = layer_idx + 1
            _log.debug(
                "SquishCache: built %s (layers=%d, mode=%s, sink=%d)",
                cls.__name__, self._squish_n_layers,
                self._squish_quantization, self._squish_sink,
            )
        elif layer_idx >= len(self._squish_cache._layers):
            # Model has more layers than first seen — extend the list in-place.
            from squish.kv.kv_cache import KVLayerCache  # avoid circular at module level
            lm = (self._squish_quantization
                  if self._squish_quantization in ("int4", "int2") else "int8")
            while len(self._squish_cache._layers) <= layer_idx:
                self._squish_cache._layers.append(
                    KVLayerCache(window=self._squish_window,
                                 kv_mode=lm,
                                 sink_count=self._squish_sink)
                )
            self._squish_cache.n_layers = len(self._squish_cache._layers)
            self._squish_n_layers = self._squish_cache.n_layers

        squish_layer = self._squish_cache._layers[layer_idx]

        # Push each new token into the squish layer
        for t in range(T_new):
            squish_layer.append(k_np[:, t, :], v_np[:, t, :])

        full_k, full_v = squish_layer.get_full_kv()
        if full_k is None:
            # Empty cache — return the new tokens unchanged
            return key_states, value_states

        # Add batch dim back: (1, n_heads, T_total, head_dim)
        result_k = full_k[np.newaxis, :]
        result_v = full_v[np.newaxis, :]

        return _restore_type(result_k, key_states), _restore_type(result_v, value_states)

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Total tokens in the cache (HF protocol)."""
        if self._squish_cache is None:
            return 0
        if layer_idx >= len(self._squish_cache._layers):
            return 0
        return self._squish_cache._layers[layer_idx].n_tokens

    def get_usable_length(
        self, new_seq_length: int, layer_idx: int = 0
    ) -> int:
        """Usable past-cache length (HF protocol)."""
        return self.get_seq_length(layer_idx)

    def reset(self) -> None:
        """Clear the cache (start a new conversation)."""
        if self._squish_cache is not None:
            self._squish_cache.reset()

    # ── Squish-specific extras ─────────────────────────────────────────────

    def metrics(self) -> "CompressionResult | None":
        """Return live compression statistics.  ``None`` before first token."""
        if self._squish_cache is None:
            return None
        return self._squish_cache.metrics()


# ---------------------------------------------------------------------------
# squish_compress decorator
# ---------------------------------------------------------------------------


def squish_compress(
    quantization: str = "int8",
    rotate: bool = True,
    window: int = 64,
    sink_token_count: int = 4,
    precision_map: "dict | None" = None,
    seed: int = 42,
) -> "Any":
    """Decorator that patches an ``AutoModelForCausalLM`` to use SquishCache.

    Apply to a *factory function* that returns the model — not to the model
    class itself — so the cache is swapped in after ``from_pretrained`` runs.

    Usage
    -----
        @squish_compress(quantization="int4", sink_token_count=4)
        def load():
            return AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B")

        model = load()
        # model.generate(...) now uses SquishCache automatically

    The decorator works by monkey-patching ``model._cache_class`` (when
    present) and by wrapping the ``forward`` method to inject a fresh
    ``SquishCache`` when ``use_cache=True`` and no ``past_key_values`` is
    provided.
    """
    def _decorator(fn: Any) -> Any:
        import functools

        @functools.wraps(fn)
        def _wrapper(*args: Any, **kwargs: Any) -> Any:
            model = fn(*args, **kwargs)
            return _patch_model(
                model,
                quantization=quantization,
                rotate=rotate,
                window=window,
                sink_token_count=sink_token_count,
                precision_map=precision_map,
                seed=seed,
            )
        return _wrapper
    return _decorator


def _patch_model(
    model: Any,
    *,
    quantization: str,
    rotate: bool,
    window: int,
    sink_token_count: int,
    precision_map: "dict | None",
    seed: int,
) -> Any:
    """Monkey-patch ``model`` to use ``SquishCache`` during generation."""
    # Register with HF's _cache_class attribute when available
    if hasattr(model, "_cache_class"):
        model._cache_class = SquishCache

    orig_forward = model.forward

    import functools

    @functools.wraps(orig_forward)
    def _patched_forward(*args: Any, **kwargs: Any) -> Any:
        # Inject SquishCache when no past_key_values and use_cache is on
        if kwargs.get("use_cache", True) and kwargs.get("past_key_values") is None:
            kwargs["past_key_values"] = SquishCache(
                quantization=quantization,
                rotate=rotate,
                window=window,
                sink_token_count=sink_token_count,
                precision_map=precision_map,
                seed=seed,
            )
        return orig_forward(*args, **kwargs)

    model.forward = _patched_forward
    _log.debug(
        "squish_compress: patched model %s with SquishCache(mode=%s, sink=%d)",
        type(model).__name__, quantization, sink_token_count,
    )
    return model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_numpy(x: Any) -> np.ndarray:
    """Convert a tensor-like to float16 numpy array."""
    if isinstance(x, np.ndarray):
        return x.astype(np.float16, copy=False)
    # torch.Tensor or mlx.array
    if hasattr(x, "numpy"):
        arr = x.numpy()
        return arr.astype(np.float16, copy=False)
    if hasattr(x, "__array__"):
        arr = np.asarray(x)
        return arr.astype(np.float16, copy=False)
    raise TypeError(
        f"Cannot convert {type(x).__name__} to numpy; "
        "expected torch.Tensor, mlx.core.array, or numpy.ndarray"
    )


def _restore_type(arr: np.ndarray, ref: Any) -> Any:
    """Restore the numpy result to the same type as ``ref``."""
    if isinstance(ref, np.ndarray):
        return arr.astype(ref.dtype, copy=False)
    # torch.Tensor
    if type(ref).__name__ == "Tensor" and hasattr(ref, "from_numpy"):
        import torch  # type: ignore[import]
        return torch.from_numpy(arr).to(ref.dtype).to(ref.device)
    # mlx.array
    if hasattr(ref, "__module__") and "mlx" in getattr(ref, "__module__", ""):
        try:
            import mlx.core as mx  # type: ignore[import]
            return mx.array(arr)
        except ImportError:
            pass
    return arr
