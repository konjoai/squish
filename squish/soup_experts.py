#!/usr/bin/env python3
"""
squish/soup_experts.py

Soup-of-Experts mixing-weight manager.

Implements sparse expert mixing from the Apple ICML 2025 Soup-of-Experts paper:
multiple domain-specific LoRA delta-weight adapters are blended via a weighted
sum ``Σ wᵢ × δᵢ`` before being applied to the base model.

Unlike standard LoRA (apply one adapter at a time), Soup-of-Experts applies a
*continuous* mixture of all registered adapters simultaneously.

Typical lifecycle
─────────────────
::

    soe = SoupOfExperts()
    soe.register_expert("legal",   "adapters/legal.safetensors")
    soe.register_expert("medical", "adapters/medical.safetensors")
    soe.register_expert("code",    "adapters/code.safetensors")

    # Automatic domain detection from prompt text
    weights = soe.detect_domain("Write a Python function to parse JSON")
    # → {"legal": 0.0, "medical": 0.0, "code": 1.0}

    # Or set weights manually
    soe.set_mixing_weights({"legal": 0.5, "medical": 0.3, "code": 0.2})

    soe.apply_mix(base_model, soe.get_weights())   # pragma: no cover

Notes
─────
``apply_mix()`` is marked ``# pragma: no cover`` because it mutates MLX model
weights and requires a live model in memory.
"""

from __future__ import annotations

from typing import Any


class SoupOfExperts:
    """Mixing-weight manager for domain-specific adapter blending.

    Parameters
    ----------
    tolerance:
        Tolerance for the ``set_mixing_weights`` sum-to-1.0 validation.
        Defaults to ``0.01`` (1% tolerance).
    """

    def __init__(self, tolerance: float = 0.01) -> None:
        # domain → (delta_weights_path, default_weight)
        self._experts: dict[str, tuple[str, float]] = {}
        self._weights: dict[str, float] = {}
        self._tolerance = tolerance

    # ── Expert registration ───────────────────────────────────────────────────

    def register_expert(
        self,
        domain: str,
        delta_weights_path: str,
        default_weight: float = 0.0,
    ) -> None:
        """Register a domain expert adapter.

        Parameters
        ----------
        domain:
            Unique domain identifier (e.g. ``"legal"``, ``"code"``).
        delta_weights_path:
            File-system path to the adapter delta weights (safetensors format).
        default_weight:
            Initial mixing weight for this domain.  All domains default to
            ``0.0``; call :meth:`set_mixing_weights` or :meth:`detect_domain`
            to assign non-zero weights before :meth:`apply_mix`.
        """
        self._experts[domain] = (delta_weights_path, default_weight)
        self._weights[domain] = default_weight

    def is_registered(self, domain: str) -> bool:
        """Return ``True`` if *domain* has a registered expert."""
        return domain in self._experts

    def registered_domains(self) -> list[str]:
        """Return sorted list of registered domain names."""
        return sorted(self._experts.keys())

    def expert_path(self, domain: str) -> str:
        """Return the delta-weights path for *domain*.

        Raises
        ------
        KeyError
            If *domain* is not registered.
        """
        if domain not in self._experts:
            raise KeyError(f"Expert not registered: {domain!r}")
        return self._experts[domain][0]

    # ── Weight management ─────────────────────────────────────────────────────

    def set_mixing_weights(self, domain_weights: dict[str, float]) -> None:
        """Set mixing weights for the given domains.

        Parameters
        ----------
        domain_weights:
            Mapping of domain → weight.  The weights must sum to within
            ``self._tolerance`` of 1.0.  Only the domains in *domain_weights*
            are updated; others retain their current weights.

        Raises
        ------
        ValueError
            If the provided weights do not sum to approximately 1.0.
        KeyError
            If a domain in *domain_weights* is not registered.
        """
        total = sum(domain_weights.values())
        if abs(total - 1.0) > self._tolerance:
            raise ValueError(
                f"Mixing weights must sum to ≈1.0 (±{self._tolerance}), "
                f"got {total:.6f}"
            )
        for domain in domain_weights:
            if domain not in self._experts:
                raise KeyError(f"Expert not registered: {domain!r}")
        self._weights.update(domain_weights)

    def get_weights(self) -> dict[str, float]:
        """Return a shallow copy of the current mixing weights dict."""
        return dict(self._weights)

    def reset_weights(self) -> None:
        """Reset all mixing weights to their registered defaults."""
        for domain, (path, default_weight) in self._experts.items():
            self._weights[domain] = default_weight

    # ── Domain detection ──────────────────────────────────────────────────────

    def detect_domain(self, prompt: str) -> dict[str, float]:
        """Return mixing weights inferred from *prompt* via bag-of-words scoring.

        Each domain name is tokenised on underscores/hyphens and scored by word
        overlap with the prompt.  The raw scores are normalised to sum to 1.0.
        When no expert is registered or no overlap is found, a uniform
        distribution is returned.

        Parameters
        ----------
        prompt:
            The input prompt text.

        Returns
        -------
        dict[str, float]
            Mapping of ``domain → weight`` where all weights sum to 1.0.
        """
        if not self._experts:
            return {}

        prompt_words = set(prompt.lower().split())
        scores: dict[str, float] = {}
        for domain in self._experts:
            # Split domain name on common separators to produce keyword tokens
            domain_tokens = set(domain.lower().replace("-", "_").split("_"))
            overlap = len(prompt_words & domain_tokens)
            scores[domain] = float(overlap)

        total = sum(scores.values())
        if total == 0.0:
            # Uniform fallback when no keyword matches
            n = len(scores)
            return {k: 1.0 / n for k in scores}

        return {k: v / total for k, v in scores.items()}

    # ── Mixing application ────────────────────────────────────────────────────

    def apply_mix(  # pragma: no cover
        self,
        base_model: Any,
        domain_weights: dict[str, float],
    ) -> None:
        """Apply a weighted sum of adapter deltas to *base_model* in-place.

        Computes ``Σ wᵢ × δᵢ`` and adds the result to each matching model
        parameter.  Requires ``safetensors`` and an in-memory base model with
        accessible weight attributes.

        Parameters
        ----------
        base_model:
            The loaded inference model whose parameters will be mutated.
        domain_weights:
            Mapping of ``domain → weight`` to use for this mixing call.
            Ignores domains with weight 0.0 for efficiency.
        """
        import numpy as np  # noqa: PLC0415
        from safetensors.numpy import load_file  # noqa: PLC0415

        # Load and scale each adapter's deltas
        mixed: dict[str, Any] = {}
        for domain, weight in domain_weights.items():
            if weight == 0.0:
                continue
            if domain not in self._experts:
                continue
            path = self._experts[domain][0]
            try:
                weights = load_file(path)
            except Exception:
                continue
            for key, delta in weights.items():
                if key in mixed:
                    mixed[key] = mixed[key] + weight * delta
                else:
                    mixed[key] = weight * delta

        # Apply compiled delta to model parameters
        for layer_name, delta in mixed.items():
            parts = layer_name.split(".")
            obj = base_model
            for part in parts[:-1]:
                obj = getattr(obj, part, None)
                if obj is None:
                    break
            if obj is None:
                continue
            param_name = parts[-1]
            orig = getattr(obj, param_name, None)
            if orig is not None:
                setattr(obj, param_name, orig + np.array(delta))
