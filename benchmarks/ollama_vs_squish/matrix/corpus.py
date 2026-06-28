"""Prompt corpus + reuse-prefix construction at exact token lengths.

The whole benchmark stands or falls on prompt construction. Two failure modes
are designed out here:

* **Repetition padding inflates prefix-cache hit rate.** A prompt grown by
  repeating one paragraph looks like a 32k prompt but compresses to a tiny
  unique prefix, so *any* prefix cache "hits" trivially. We instead synthesise
  genuinely varied technical prose — every sentence draws distinct entities,
  components, numbers and topics from large pools — so token-level variation is
  real. Real corpus files can also be dropped into ``corpus_dir`` and are used
  verbatim when present.

* **Reuse must be a controlled construction, not an accident.** A prompt at
  reuse level X is ``[fixed shared prefix sized to X% of target] + [unique tail
  sized to (100-X)%]``. The shared prefix is a fixed realistic context block
  (a system/document preamble); the tail varies every run. 0% has no shared
  prefix (a different full document each run); 100% resends an identical prompt.

Everything is deterministic from a base seed and the cell coordinates, and every
generated prompt is saved to disk with its seed and token count, so the corpus
is fully auditable. The tokenizer is injected (any object exposing ``encode`` /
``decode``) so this module is unit-testable without MLX.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol


class Tokenizer(Protocol):
    def encode(self, text: str) -> list[int]: ...
    def decode(self, ids: list[int]) -> str: ...


# ── lexical pools (large enough that sentences do not repeat) ──────────────────

_SUBSYSTEMS = [
    "auth service",
    "billing pipeline",
    "session cache",
    "search indexer",
    "notification queue",
    "feature-flag store",
    "rate limiter",
    "audit log",
    "payment gateway",
    "recommendation engine",
    "ingestion worker",
    "CDN edge",
    "schema migrator",
    "websocket gateway",
    "metrics collector",
    "blob store",
    "scheduler",
    "config server",
    "token rotator",
    "replication daemon",
]
_COMPONENTS = [
    "the retry loop",
    "the connection pool",
    "the LRU eviction path",
    "the TTL refresh",
    "the idempotency key",
    "the backpressure valve",
    "the circuit breaker",
    "the leader election",
    "the write-ahead log",
    "the bloom filter",
    "the consistent-hash ring",
    "the dead-letter handler",
    "the snapshot compactor",
    "the fan-out dispatcher",
    "the quota accountant",
]
_RISKS = [
    "a race under contention",
    "a memory leak on the slow path",
    "an unbounded queue",
    "a thundering herd on cache miss",
    "a partial write on crash",
    "a clock-skew dependency",
    "a silent fallback that hides errors",
    "an N+1 query",
    "a head-of-line block",
    "a reentrancy hazard",
    "a stale read after failover",
    "a lock held across an await",
    "an off-by-one in the ring buffer",
]
_ACTIONS = [
    "rejects",
    "retries",
    "coalesces",
    "throttles",
    "shards",
    "batches",
    "deduplicates",
    "invalidates",
    "rehydrates",
    "checkpoints",
    "fences",
    "backfills",
    "reconciles",
    "quarantines",
    "rebalances",
]
_METRICS = [
    "p99 latency",
    "tail amplification",
    "cache-hit ratio",
    "GC pause time",
    "queue depth",
    "error budget",
    "saturation",
    "write amplification",
    "replication lag",
    "connection churn",
    "allocation rate",
]
_TOPICS = [
    "Under sustained load the %(sub)s %(act)s requests while %(comp)s guards "
    "against %(risk)s; watch %(metric)s closely.",
    "When %(comp)s fails over, the %(sub)s must avoid %(risk)s — the review "
    "should confirm %(metric)s stays bounded.",
    "A correct %(sub)s %(act)s duplicate work in %(comp)s; the failure mode to "
    "fear is %(risk)s, surfaced first in %(metric)s.",
    "Consider the interaction between %(comp)s and the %(sub)s: if it %(act)s "
    "too aggressively you trade %(metric)s for %(risk)s.",
    "The %(sub)s relies on %(comp)s to bound %(metric)s; absent that you get "
    "%(risk)s the moment traffic %(act)s past the threshold.",
]


def _sentence(rng: random.Random) -> str:
    tmpl = rng.choice(_TOPICS)
    return tmpl % {
        "sub": rng.choice(_SUBSYSTEMS),
        "comp": rng.choice(_COMPONENTS),
        "risk": rng.choice(_RISKS),
        "act": rng.choice(_ACTIONS),
        "metric": rng.choice(_METRICS),
    }


def _document(rng: random.Random, n_sentences: int) -> str:
    """A block of varied technical prose — no sentence-level repetition."""
    return " ".join(_sentence(rng) for _ in range(n_sentences))


_SHARED_PREAMBLE_HEAD = (
    "You are a senior staff engineer performing a rigorous design and code "
    "review. The following is the standing engineering context for the service "
    "under review. Treat it as authoritative background that applies to every "
    "question that follows.\n\n"
)


# ── data types ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PromptSpec:
    """One fully-specified, auditable prompt for a single run."""

    reuse: float
    ctx_tokens: int
    run_index: int
    seed: int
    target_tokens: int
    shared_prefix_tokens: int
    tail_tokens: int
    measured_tokens: int
    sha256: str
    text: str

    def manifest_row(self) -> dict[str, object]:
        d = asdict(self)
        d.pop("text")  # text saved separately as a .txt; manifest stays compact
        return d


# ── corpus ────────────────────────────────────────────────────────────────────


class Corpus:
    """Deterministic source of varied documents at exact token lengths.

    A fixed ``base_seed`` plus the cell coordinates (reuse, ctx, run_index)
    determines every prompt, so the whole corpus regenerates byte-identically.
    """

    def __init__(
        self, tokenizer: Tokenizer, base_seed: int = 20260628, corpus_dir: Path | None = None
    ) -> None:
        self.tok = tokenizer
        self.base_seed = base_seed
        self._real_docs: list[str] = []
        if corpus_dir is not None and Path(corpus_dir).is_dir():
            for fp in sorted(Path(corpus_dir).glob("*.txt")):
                self._real_docs.append(fp.read_text(encoding="utf-8"))

    # -- low-level: produce text of an exact token length --------------------

    def _slice_to_tokens(self, text: str, n_tokens: int) -> str:
        ids = self.tok.encode(text)
        if len(ids) >= n_tokens:
            return self.tok.decode(ids[:n_tokens])
        return text  # caller grows the source first; never silently pads

    def _grow_document(self, rng: random.Random, n_tokens: int, seed_text: str = "") -> str:
        """Grow varied prose from *seed_text* to >= n_tokens, then slice exact."""
        text = seed_text
        if self._real_docs:
            while len(self.tok.encode(text)) < n_tokens and self._real_docs:
                text += "\n\n" + self._real_docs[rng.randrange(len(self._real_docs))]
        else:
            while len(self.tok.encode(text)) < n_tokens:
                text += _document(rng, 24) + "\n\n"
        return self._slice_to_tokens(text, n_tokens)

    # -- public: build one prompt for a cell ---------------------------------

    def build_prompt(self, reuse: float, ctx_tokens: int, run_index: int) -> PromptSpec:
        if not 0.0 <= reuse <= 1.0:
            raise ValueError(f"reuse must be in [0,1], got {reuse}")
        shared_n = round(reuse * ctx_tokens)
        tail_n = ctx_tokens - shared_n
        seed = self._seed_for(reuse, ctx_tokens, run_index)

        if reuse <= 0.0:
            # Unique full document per run; no shared prefix at all.
            rng = random.Random(seed)
            text = self._grow_document(rng, ctx_tokens)
        elif reuse >= 1.0:
            # Identical prompt every run — fixed seed independent of run_index.
            rng = random.Random(self._seed_for(1.0, ctx_tokens, 0))
            text = self._grow_document(rng, ctx_tokens, seed_text=_SHARED_PREAMBLE_HEAD)
        else:
            # Fixed shared prefix (grown to exactly shared_n) + per-run unique tail.
            prefix = self._build_prefix(reuse, ctx_tokens, shared_n)
            tail_rng = random.Random(seed)
            tail = self._grow_document(tail_rng, tail_n)
            text = self._slice_to_tokens(prefix + " " + tail, ctx_tokens)

        measured = len(self.tok.encode(text))
        return PromptSpec(
            reuse=reuse,
            ctx_tokens=ctx_tokens,
            run_index=run_index,
            seed=seed,
            target_tokens=ctx_tokens,
            shared_prefix_tokens=shared_n,
            tail_tokens=tail_n,
            measured_tokens=measured,
            sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
            text=text,
        )

    def _build_prefix(self, reuse: float, ctx_tokens: int, shared_n: int) -> str:
        """Fixed shared-prefix block grown to exactly shared_n tokens."""
        prefix_rng = random.Random(self._seed_for(reuse, ctx_tokens, -1))
        return self._grow_document(prefix_rng, shared_n, seed_text=_SHARED_PREAMBLE_HEAD)

    def shared_prefix_text(self, reuse: float, ctx_tokens: int) -> str:
        """The fixed shared-prefix block for a partial-reuse cell (for auditing)."""
        if not 0.0 < reuse < 1.0:
            return ""
        return self._build_prefix(reuse, ctx_tokens, round(reuse * ctx_tokens))

    def _seed_for(self, reuse: float, ctx_tokens: int, run_index: int) -> int:
        key = f"{self.base_seed}|{reuse:.4f}|{ctx_tokens}|{run_index}"
        return int(hashlib.sha256(key.encode()).hexdigest()[:12], 16)


# ── persistence ───────────────────────────────────────────────────────────────


def save_cell_prompts(out_dir: Path, cell_id: str, prompts: list[PromptSpec]) -> Path:
    """Persist every prompt of a cell + a compact manifest. Returns manifest path."""
    cell_dir = Path(out_dir) / "prompts" / cell_id
    cell_dir.mkdir(parents=True, exist_ok=True)
    for p in prompts:
        (cell_dir / f"run_{p.run_index:03d}.txt").write_text(p.text, encoding="utf-8")
    manifest = cell_dir / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "cell_id": cell_id,
                "n_prompts": len(prompts),
                "runs": [p.manifest_row() for p in prompts],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return manifest


def expected_hit_fraction(reuse: float) -> float:
    """The cache-hit fraction a correct cache should achieve at this reuse level.

    Equal to the reuse level by construction: the shared prefix is exactly
    ``reuse`` of the prompt, so a working prefix cache reuses ``reuse`` of it on
    every run after the first.
    """
    return max(0.0, min(1.0, reuse))
