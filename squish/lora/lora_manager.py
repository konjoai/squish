#!/usr/bin/env python3
"""
squish/lora_manager.py

Runtime LoRA adapter loading and application for the Squish inference server.

Supports the HuggingFace PEFT safetensors format: a directory containing
``adapter_config.json`` (rank, alpha, target_modules) and one or more
``*.safetensors`` weight files.

Typical lifecycle
─────────────────
::

    mgr = LoRAManager()
    mgr.register("legal", "~/.squish/adapters/legal")
    mgr.register("code",  "~/.squish/adapters/code")

    mgr.apply(model, "legal")   # patch model for legal domain
    # … run inference …
    mgr.unapply(model)           # restore original weights

    info = mgr.adapter_info("legal")   # rank, alpha, total_params, size_mb

Thread safety
─────────────
``apply()`` and ``unapply()`` are protected by a ``threading.Lock``.  Concurrent
requests MUST NOT call either method simultaneously — the server should hold
its own request-routing guard around any adapter switch.
"""

from __future__ import annotations

import collections
import json
import threading
from pathlib import Path
from typing import Any

import numpy as np


class LoRAManager:
    """Load and apply HuggingFace PEFT LoRA adapters to a loaded model.

    Parameters
    ----------
    max_cache_size:
        Maximum number of adapters held in the in-memory LRU cache.
        Defaults to 4.  Oldest entry is evicted when the cache is full.
    """

    _MAX_CACHE: int = 4

    def __init__(self, max_cache_size: int = 4) -> None:
        self._MAX_CACHE = max_cache_size
        self._registry: dict[str, Path] = {}
        # OrderedDict used as LRU: move_to_end on access, popitem(last=False) on evict
        self._cache: collections.OrderedDict[str, dict[str, Any]] = (
            collections.OrderedDict()
        )
        # Snapshots for unapply
        self._original_weights: dict[str, Any] = {}
        self._patched_layers: list[str] = []
        # Per-domain config cache (avoid re-reading JSON)
        self._config_cache: dict[str, dict] = {}
        self._lock = threading.Lock()

    # ── Registry ─────────────────────────────────────────────────────────────

    def register(self, domain: str, path: str | Path) -> None:
        """Register an adapter directory path for *domain*."""
        self._registry[domain] = Path(path)

    def is_registered(self, domain: str) -> bool:
        """Return ``True`` if *domain* has a registered adapter path."""
        return domain in self._registry

    def registered_domains(self) -> list[str]:
        """Return a sorted list of all registered domain names."""
        return sorted(self._registry.keys())

    # ── Config ────────────────────────────────────────────────────────────────

    def _read_adapter_config(self, domain: str) -> dict:
        """Return the parsed ``adapter_config.json`` for *domain*.

        Raises
        ------
        KeyError
            If *domain* has not been registered.
        FileNotFoundError
            If ``adapter_config.json`` does not exist at the registered path.
        """
        if domain not in self._registry:
            raise KeyError(f"Domain not registered: {domain!r}")
        if domain in self._config_cache:
            return self._config_cache[domain]
        config_path = self._registry[domain] / "adapter_config.json"
        with open(config_path) as fh:
            cfg = json.load(fh)
        self._config_cache[domain] = cfg
        return cfg

    def invalidate_config_cache(self, domain: str) -> None:
        """Remove the cached config for *domain* so it is re-read on next access."""
        self._config_cache.pop(domain, None)

    # ── Loading ───────────────────────────────────────────────────────────────

    def load(self, domain: str) -> dict[str, Any]:
        """Return adapter weight dict, using LRU cache.

        Raises
        ------
        KeyError
            If *domain* is not registered.
        ImportError
            If the ``safetensors`` package is not installed.
        """
        if domain not in self._registry:
            raise KeyError(f"Domain not registered: {domain!r}")
        if domain in self._cache:
            self._cache.move_to_end(domain)
            return self._cache[domain]
        weights = self._load_safetensors(self._registry[domain])
        # Evict LRU entries to stay within capacity
        while len(self._cache) >= self._MAX_CACHE:
            self._cache.popitem(last=False)
        self._cache[domain] = weights
        return weights

    def _load_safetensors(self, path: Path) -> dict[str, Any]:
        """Load all ``*.safetensors`` files under *path* into a flat weight dict.

        Raises
        ------
        ImportError
            If the ``safetensors`` package is not available.
        """
        try:
            from safetensors import safe_open  # noqa: PLC0415
        except ImportError:  # pragma: no cover
            raise ImportError(
                "safetensors package required for LoRA adapter loading. "
                "Install with: pip install safetensors"
            )
        weights: dict[str, Any] = {}
        for st_file in sorted(path.glob("*.safetensors")):
            with safe_open(str(st_file), framework="numpy") as fh:
                for key in fh.keys():
                    weights[key] = fh.get_tensor(key)
        return weights

    def evict(self, domain: str) -> bool:
        """Remove *domain* from the weight cache.  Returns ``True`` if it was cached."""
        if domain in self._cache:
            del self._cache[domain]
            return True
        return False

    def cache_size(self) -> int:
        """Return the number of adapters currently held in cache."""
        return len(self._cache)

    # ── Application ───────────────────────────────────────────────────────────

    def _resolve_param(
        self, model: Any, layer_name: str
    ) -> tuple[Any, str]:
        """Walk the dotted *layer_name* on *model*.

        Returns ``(parent_object, attribute_name)`` if every intermediate
        attribute exists, otherwise ``(None, "")``.
        """
        parts = layer_name.split(".")
        obj = model
        for part in parts[:-1]:
            obj = getattr(obj, part, None)
            if obj is None:
                return None, ""
        return obj, parts[-1]

    def apply(self, model: Any, domain: str) -> None:
        """Apply LoRA delta weights to *model*.  Thread-safe.

        Snapshots the original parameter values so ``unapply()`` can restore
        them.  Calling ``apply()`` twice without ``unapply()`` in between will
        discard the first snapshot.

        Raises
        ------
        KeyError
            If *domain* is not registered.
        """
        with self._lock:
            cfg = self._read_adapter_config(domain)
            rank = int(cfg.get("r", 8))
            alpha = float(cfg.get("lora_alpha", rank))
            scale = alpha / rank
            weights = self.load(domain)
            self._patched_layers = []
            self._original_weights = {}
            for layer_name, delta in weights.items():
                obj, param_name = self._resolve_param(model, layer_name)
                if obj is None:
                    continue
                orig = getattr(obj, param_name, None)
                if orig is None:
                    continue
                self._original_weights[layer_name] = orig
                setattr(obj, param_name, orig + scale * delta)
                self._patched_layers.append(layer_name)

    def unapply(self, model: Any) -> None:
        """Restore model weights patched by the last ``apply()`` call.  Thread-safe."""
        with self._lock:
            for layer_name in reversed(self._patched_layers):
                orig = self._original_weights.get(layer_name)
                if orig is None:
                    continue
                obj, param_name = self._resolve_param(model, layer_name)
                if obj is None:
                    continue
                setattr(obj, param_name, orig)
            self._patched_layers = []
            self._original_weights = {}

    # ── Metadata ──────────────────────────────────────────────────────────────

    def adapter_info(self, domain: str) -> dict:
        """Return a metadata dict for the registered adapter.

        Keys: ``rank``, ``alpha``, ``target_modules``,
              ``total_params``, ``size_mb``.

        Raises
        ------
        KeyError
            If *domain* is not registered or has a missing config.
        """
        cfg = self._read_adapter_config(domain)
        rank = int(cfg.get("r", 8))
        alpha = float(cfg.get("lora_alpha", rank))
        target_modules = cfg.get("target_modules", [])

        adapter_path = self._registry.get(domain)
        size_bytes = 0
        if adapter_path is not None:
            for sf in adapter_path.glob("*.safetensors"):
                size_bytes += sf.stat().st_size

        total_params = 0
        if domain in self._cache:
            for arr in self._cache[domain].values():
                total_params += int(np.prod(arr.shape))

        return {
            "rank": rank,
            "alpha": alpha,
            "target_modules": target_modules,
            "total_params": total_params,
            "size_mb": round(size_bytes / 1024 / 1024, 2),
        }


# ── DARE-TIES: delta-sparse LoRA weight merging ───────────────────────────────
#
# Based on:
#   "DARE" — "Language Model Merging by Uncertainty-Based Gradient Matching"
#             (Yu et al., 2024;  arXiv:2402.11176)
#   "TIES" — "TIES-Merging: Resolving Interference When Merging Models"
#             (Yadav et al., NeurIPS 2023; arXiv:2306.01708)
#
# Key insight
# -----------
# Fine-tuned models diverge from the pre-trained base primarily through a
# sparse set of large weight *deltas*.  Merging multiple fine-tuned models
# by naively averaging deltas leads to destructive interference.
#
# DARE (Drop And REscale) prunes most of the delta noise at random and
# rescales the survivors to preserve expected magnitude.
#
# TIES (Trim, Elect Sign, Merge) then:
#   1. Trims low-magnitude updates.
#   2. Elects a dominant sign per parameter across models.
#   3. Averages only the models whose delta agrees with the elected sign.
#
# Together they produce a merged delta with far less cross-model interference.

from dataclasses import dataclass as _dtdc


@_dtdc
class DareTiesConfig:
    """Configuration for DARE-TIES weight-delta merging.

    Parameters
    ----------
    sparsity : float
        Fraction of each delta to *drop* randomly in the DARE step (0 < x < 1).
        Typical value: 0.9 (drop 90% of delta entries before merging).
    top_k_fraction : float | None
        If set (0 < x ≤ 1), TIES trim step keeps only the top-``top_k_fraction``
        magnitude entries per parameter across all deltas before sign election.
        ``None`` disables the trim step.
    scale : float
        Scalar applied to the merged delta before returning.  Use 1.0 for
        unweighted merge, or a smaller value to partial-scale the merge.
    seed : int
        RNG seed for the DARE random mask (for reproducibility in tests).
    """

    sparsity:        float         = 0.9
    top_k_fraction:  float | None = None
    scale:           float         = 1.0
    seed:            int           = 42

    def __post_init__(self) -> None:
        if not 0.0 < self.sparsity < 1.0:
            raise ValueError("sparsity must be in (0, 1)")
        if self.top_k_fraction is not None:
            if not 0.0 < self.top_k_fraction <= 1.0:
                raise ValueError("top_k_fraction must be in (0, 1]")
        if self.scale <= 0.0:
            raise ValueError("scale must be > 0")


class DareTiesMerger:
    """Apply DARE-TIES to merge a list of LoRA weight deltas.

    Parameters
    ----------
    config : DareTiesConfig
    """

    def __init__(self, config: DareTiesConfig) -> None:
        self._cfg = config

    # ── DARE step ───────────────────────────────────────────────────────────

    def sparsify_dare(
        self,
        delta: np.ndarray,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Apply DARE: randomly drop ``sparsity`` fraction of delta, then rescale.

        Parameters
        ----------
        delta : np.ndarray — weight delta (any shape).
        rng : numpy Generator for reproducible masks (uses ``self._cfg.seed`` if None).

        Returns
        -------
        Sparsified delta of the same shape.
        """
        if rng is None:
            rng = np.random.default_rng(self._cfg.seed)
        d     = np.asarray(delta, dtype=np.float32)
        mask  = rng.random(d.shape) >= self._cfg.sparsity   # True → keep
        scale = 1.0 / (1.0 - self._cfg.sparsity)            # rescale survivors
        return (d * mask * scale).astype(np.float32)

    # ── TIES steps ──────────────────────────────────────────────────────────

    def trim(self, deltas: list[np.ndarray]) -> list[np.ndarray]:
        """TIES trim: zero entries with magnitudes below the top-k threshold.

        If ``top_k_fraction`` is None, returns deltas unchanged.

        Parameters
        ----------
        deltas : list of same-shape float arrays.

        Returns
        -------
        List of trimmed arrays of the same shape.
        """
        frac = self._cfg.top_k_fraction
        if frac is None:
            return [np.asarray(d, dtype=np.float32) for d in deltas]
        trimmed = []
        for d in deltas:
            d_f  = np.asarray(d, dtype=np.float32)
            flat = np.abs(d_f.ravel())
            k    = max(1, int(len(flat) * frac))
            thresh = np.partition(flat, -k)[-k]
            t    = d_f.copy()
            t[np.abs(d_f) < thresh] = 0.0
            trimmed.append(t)
        return trimmed

    def elect_sign(self, deltas: list[np.ndarray]) -> np.ndarray:
        """TIES elect sign: return +1/-1 per element by majority sign vote.

        For each parameter, count how many deltas are positive vs negative;
        the majority sign (with ties broken by +1) is the elected sign.

        Parameters
        ----------
        deltas : list of same-shape float arrays.

        Returns
        -------
        Sign array (+1 or -1) of the same shape.
        """
        stacked  = np.stack([np.asarray(d, dtype=np.float32) for d in deltas])
        pos_vote = (stacked > 0).sum(axis=0)
        neg_vote = (stacked < 0).sum(axis=0)
        return np.where(pos_vote >= neg_vote, 1.0, -1.0).astype(np.float32)

    def ties_merge(self, deltas: list[np.ndarray]) -> np.ndarray:
        """Run the full TIES merge pipeline (trim → elect sign → average).

        Parameters
        ----------
        deltas : list of same-shape float arrays.

        Returns
        -------
        Merged delta array scaled by ``self._cfg.scale``.
        """
        if not deltas:
            raise ValueError("deltas must be non-empty")
        trimmed    = self.trim(deltas)
        sign_mask  = self.elect_sign(trimmed)
        # Keep only deltas that agree with the elected sign
        agreed = []
        for d in trimmed:
            d * np.sign(d)   # magnitude
            # Zero out entries where sign disagrees
            agrees   = (np.sign(d) == sign_mask) | (d == 0.0)
            agreed.append(d * agrees)
        merged = np.stack(agreed).mean(axis=0) * self._cfg.scale
        return merged.astype(np.float32)

    # ── Combined DARE → TIES ─────────────────────────────────────────────────

    def merge(self, deltas: list[np.ndarray]) -> np.ndarray:
        """Full pipeline: DARE sparsify each delta, then TIES merge.

        Parameters
        ----------
        deltas : list of same-shape float arrays (one per fine-tuned model).

        Returns
        -------
        Single merged delta array of the same shape.
        """
        if not deltas:
            raise ValueError("deltas must be non-empty")
        rng = np.random.default_rng(self._cfg.seed)
        sparsified = [self.sparsify_dare(d, rng) for d in deltas]
        return self.ties_merge(sparsified)
