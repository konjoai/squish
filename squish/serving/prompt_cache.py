"""PromptCache: modular KV reuse for templated prompts (arXiv 2311.04934, EuroSys 2024).

Schema-defined prompt templates declare constant "anchor" spans and named
variable slots.  The constant spans are pre-materialised as KV cache entries
at server startup; at request time the cached shards are assembled with
freshly-computed variable-slot KV to form the full KV cache — yielding
effectively zero-prefill TTFT for matched schemas.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

__all__ = [
    "PromptCacheConfig",
    "PromptSchema",
    "PromptCacheResult",
    "PromptCacheKV",
]


@dataclass
class PromptCacheConfig:
    """Configuration for :class:`PromptCacheKV`.

    Attributes:
        max_schemas: Maximum number of schemas that can be registered.
        kv_dim: KV vector dimension (head dimension × heads, for simulation).
        n_heads: Number of attention heads.
        seed: RNG seed for synthetic KV generation.
    """

    max_schemas: int = 64
    kv_dim: int = 128
    n_heads: int = 8
    seed: int = 0

    def __post_init__(self) -> None:
        if self.max_schemas < 1:
            raise ValueError(f"max_schemas must be >= 1, got {self.max_schemas}")
        if self.kv_dim < 1:
            raise ValueError(f"kv_dim must be >= 1, got {self.kv_dim}")
        if self.n_heads < 1:
            raise ValueError(f"n_heads must be >= 1, got {self.n_heads}")


@dataclass
class PromptSchema:
    """A named prompt template with constant and variable spans.

    Attributes:
        name: Unique schema identifier.
        constant_spans: List of string spans that are invariant across all
            requests matching this schema (e.g. system prompt, few-shot
            examples).
        variable_slots: Names of the variable placeholders (e.g. ``["query",
            "context"]``).  At request time, callers supply the actual token
            counts for each slot.
    """

    name: str
    constant_spans: List[str]
    variable_slots: List[str]

    @property
    def n_constant_tokens(self) -> int:
        """Approximate token count from whitespace splitting concatenated spans."""
        return sum(len(s.split()) for s in self.constant_spans)


@dataclass
class PromptCacheResult:
    """Output of :class:`PromptCacheKV.lookup`.

    Attributes:
        hit: Whether a cached schema matched.
        schema_name: Name of the matched schema (or ``None``).
        cached_kv: Pre-materialised KV for constant spans — shape
            ``(n_cached_tokens, n_heads, kv_dim // n_heads)`` or ``None``.
        n_cached_tokens: Number of tokens served from cache.
        n_fresh_tokens: Number of tokens requiring fresh computation.
    """

    hit: bool
    schema_name: Optional[str]
    cached_kv: Optional[np.ndarray]
    n_cached_tokens: int
    n_fresh_tokens: int


class PromptCacheKV:
    """Schema-based modular KV cache for low-latency LLM serving.

    Workflow::

        cache = PromptCacheKV(PromptCacheConfig())
        schema = PromptSchema(
            name="rag_v1",
            constant_spans=["You are a helpful assistant.", "Context: ..."],
            variable_slots=["query"],
        )
        cache.register_schema(schema)
        cache.materialize("rag_v1")   # pre-computes KV for constant spans

        # At request time:
        result = cache.lookup("rag_v1")
        # result.cached_kv contains the pre-computed constant-span KV
    """

    def __init__(self, config: Optional[PromptCacheConfig] = None) -> None:
        self._config = config or PromptCacheConfig()
        self._schemas: Dict[str, PromptSchema] = {}
        self._kv_store: Dict[str, np.ndarray] = {}
        self._rng = np.random.default_rng(self._config.seed)

    @property
    def config(self) -> PromptCacheConfig:
        return self._config

    @property
    def n_schemas(self) -> int:
        """Number of registered schemas."""
        return len(self._schemas)

    @property
    def n_materialized(self) -> int:
        """Number of schemas with materialised KV."""
        return len(self._kv_store)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register_schema(self, schema: PromptSchema) -> None:
        """Register a prompt schema.

        Parameters
        ----------
        schema:
            The :class:`PromptSchema` to register.

        Raises
        ------
        ValueError
            If ``max_schemas`` would be exceeded or the name is already taken.
        """
        if schema.name in self._schemas:
            raise ValueError(f"Schema {schema.name!r} is already registered")
        if len(self._schemas) >= self._config.max_schemas:
            raise ValueError(
                f"Cannot register more than {self._config.max_schemas} schemas"
            )
        self._schemas[schema.name] = schema

    def materialize(
        self,
        schema_name: str,
        kv_data: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Pre-compute (or store) the KV cache for a schema's constant spans.

        Parameters
        ----------
        schema_name:
            Name of the registered schema to materialise.
        kv_data:
            Optional pre-computed KV array of shape
            ``(n_tokens, n_heads, head_dim)``.  When ``None``, a synthetic
            KV tensor is generated for the constant-span token count.

        Returns
        -------
        np.ndarray
            The stored KV tensor.
        """
        if schema_name not in self._schemas:
            raise KeyError(f"Unknown schema {schema_name!r}")
        schema = self._schemas[schema_name]
        cfg = self._config
        head_dim = cfg.kv_dim // cfg.n_heads

        if kv_data is not None:
            kv = np.asarray(kv_data, dtype=np.float32)
        else:
            n_tok = max(1, schema.n_constant_tokens)
            kv = self._rng.standard_normal(
                (n_tok, cfg.n_heads, head_dim)
            ).astype(np.float32)

        self._kv_store[schema_name] = kv
        return kv

    def lookup(
        self,
        schema_name: str,
        n_variable_tokens: int = 0,
    ) -> PromptCacheResult:
        """Retrieve the cached KV for a schema.

        Parameters
        ----------
        schema_name:
            Schema to look up.
        n_variable_tokens:
            Number of variable-slot tokens that will need fresh computation.

        Returns
        -------
        PromptCacheResult
        """
        if schema_name not in self._schemas:
            return PromptCacheResult(
                hit=False,
                schema_name=None,
                cached_kv=None,
                n_cached_tokens=0,
                n_fresh_tokens=n_variable_tokens,
            )

        if schema_name not in self._kv_store:
            return PromptCacheResult(
                hit=False,
                schema_name=schema_name,
                cached_kv=None,
                n_cached_tokens=0,
                n_fresh_tokens=n_variable_tokens,
            )

        kv = self._kv_store[schema_name]
        n_cached = kv.shape[0]
        return PromptCacheResult(
            hit=True,
            schema_name=schema_name,
            cached_kv=kv,
            n_cached_tokens=n_cached,
            n_fresh_tokens=n_variable_tokens,
        )

    def evict(self, schema_name: str) -> None:
        """Remove the materialised KV for a schema (keeps schema registered).

        Parameters
        ----------
        schema_name:
            Schema whose KV should be evicted from memory.
        """
        self._kv_store.pop(schema_name, None)

    def list_schemas(self) -> List[str]:
        """Return the names of all registered schemas."""
        return list(self._schemas.keys())
