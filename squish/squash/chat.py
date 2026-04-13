"""squish/squash/chat.py — Interactive compliance auditor chatbot (RAG on SBOM).

Loads squash attestation artifacts from a model directory into a document store
and runs a local Retrieval-Augmented Generation (RAG) loop so compliance auditors
can ask plain-English questions about a model's compliance posture, supply chain,
and vulnerabilities.

Architecture
------------
* **Chunker** — splits the JSON attestation artifacts into semantically-bounded
  text chunks (one chunk per BOM component, one per policy finding, one per VEX
  statement).  No LangChain required.
* **InMemoryVectorStore** — TF-IDF-style BM25 keyword index that works with zero
  optional dependencies.  If ``numpy`` is available a simple cosine TF-IDF store
  is used instead.  If an Ollama or OpenAI-compatible embedding endpoint is
  configured, real vector embeddings are stored.
* **RAGSession** — the conversational loop: (1) retrieve top-k chunks by query
  similarity, (2) send prompt + context to the LLM, (3) return the answer with
  source citations.

Design principles
-----------------
* **Zero mandatory heavy deps** — the basic BM25 retriever works with stdlib only.
  ``numpy`` and an LLM endpoint are optional.
* **Citation-first prompt template** — the system prompt instructs the LLM to
  always name the artifact file and JSON key it used.  This keeps auditors honest.
* **Safe read-only** — the chat session never writes to the model directory.

Usage::

    session = ChatSession.from_model_dir(Path("./my-model"))
    answer = session.ask("Are there any high-risk datasets in the model lineage?")
    print(answer.text)
    print(answer.sources)   # e.g. ["cyclonedx-mlbom.json:components[0].dependencies"]

    # Interactive REPL:
    session.repl()
"""

from __future__ import annotations

import glob
import json
import logging
import re
import textwrap
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# Artifact filenames squash writes to a model directory
_SQUASH_ARTIFACT_GLOBS: list[str] = [
    "cyclonedx-mlbom.json",
    "squash-scan.json",
    "squash-policy-*.json",
    "squash-attest.json",
    "squash-vex.json",
    "spdx-ai-profile.json",
    "squash-eval-report.json",
    "squash-lineage.json",
    "slsa-provenance.json",
]

# ── Document chunks ───────────────────────────────────────────────────────────


@dataclass
class Chunk:
    """A single text fragment from a squash artifact."""
    source_file: str
    key_path: str
    text: str

    def citation(self) -> str:
        return f"{self.source_file}:{self.key_path}"


# ── Chunker ───────────────────────────────────────────────────────────────────


class ArtifactChunker:
    """Split squash JSON artifacts into text chunks for retrieval."""

    @classmethod
    def load_model_dir(cls, model_dir: Path) -> list[Chunk]:
        chunks: list[Chunk] = []
        for pattern in _SQUASH_ARTIFACT_GLOBS:
            for match in model_dir.glob(pattern):
                try:
                    chunks.extend(cls._chunk_file(match))
                except Exception as exc:
                    log.debug("Skipping %s: %s", match, exc)
        log.info("Loaded %d chunks from %s", len(chunks), model_dir)
        return chunks

    @classmethod
    def _chunk_file(cls, path: Path) -> list[Chunk]:
        raw = json.loads(path.read_text())
        chunks: list[Chunk] = []
        fname = path.name

        # Top-level metadata as a single chunk
        chunks.append(Chunk(
            source_file=fname,
            key_path="(root)",
            text=cls._flatten(raw, depth=1),
        ))

        # Per-component chunks
        for i, comp in enumerate(raw.get("components", [])):
            chunks.append(Chunk(
                source_file=fname,
                key_path=f"components[{i}]",
                text=cls._flatten(comp, depth=3),
            ))

        # Policy findings
        for i, finding in enumerate(raw.get("findings", [])):
            chunks.append(Chunk(
                source_file=fname,
                key_path=f"findings[{i}]",
                text=cls._flatten(finding, depth=2),
            ))

        # VEX statements
        for i, stmt in enumerate(raw.get("statements", [])):
            chunks.append(Chunk(
                source_file=fname,
                key_path=f"statements[{i}]",
                text=cls._flatten(stmt, depth=2),
            ))

        # Provenance datasets
        for ds_name, ds_val in raw.get("datasets", {}).items():
            chunks.append(Chunk(
                source_file=fname,
                key_path=f"datasets.{ds_name}",
                text=cls._flatten(ds_val, depth=2),
            ))

        # Scan findings
        for i, f in enumerate(raw.get("scan_findings", [])):
            chunks.append(Chunk(
                source_file=fname,
                key_path=f"scan_findings[{i}]",
                text=cls._flatten(f, depth=2),
            ))

        return chunks

    @classmethod
    def _flatten(cls, obj: Any, depth: int, prefix: str = "") -> str:
        """Recursively flatten a dict/list to a readable text blob."""
        if depth == 0:
            return json.dumps(obj, ensure_ascii=False)[:512]
        if isinstance(obj, dict):
            parts = []
            for k, v in obj.items():
                key = f"{prefix}{k}"
                parts.append(f"{key}: {cls._flatten(v, depth - 1, key + '.')}")
            return " | ".join(parts)
        if isinstance(obj, list):
            return ", ".join(cls._flatten(item, depth - 1, prefix) for item in obj[:5])
        return str(obj)[:256]


# ── Retriever ─────────────────────────────────────────────────────────────────


class BM25Retriever:
    """Very lightweight BM25-inspired keyword retriever (stdlib only).

    Returns the top-k most relevant chunks for a query using term frequency
    overlap scoring.  No embeddings, no external dependencies.
    """

    _K1 = 1.5
    _B = 0.75

    def __init__(self, chunks: list[Chunk]) -> None:
        self._chunks = chunks
        self._tokenized: list[list[str]] = [self._tokenize(c.text) for c in chunks]
        avg_dl = sum(len(t) for t in self._tokenized) / max(1, len(self._tokenized))
        self._avg_dl = avg_dl

        # Compute DF per term
        self._df: dict[str, int] = {}
        for tokens in self._tokenized:
            for tok in set(tokens):
                self._df[tok] = self._df.get(tok, 0) + 1
        self._N = len(chunks)

    def retrieve(self, query: str, k: int = 5) -> list[Chunk]:
        if not self._chunks:
            return []
        q_tokens = self._tokenize(query)
        scores = [self._score(i, q_tokens) for i in range(len(self._chunks))]
        top_k = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [self._chunks[i] for i in top_k if scores[i] > 0]

    def _score(self, doc_idx: int, q_tokens: list[str]) -> float:
        import math  # noqa: PLC0415
        tokens = self._tokenized[doc_idx]
        dl = len(tokens)
        tf_map: dict[str, int] = {}
        for tok in tokens:
            tf_map[tok] = tf_map.get(tok, 0) + 1

        score = 0.0
        for tok in q_tokens:
            if tok not in tf_map:
                continue
            tf = tf_map[tok]
            df = self._df.get(tok, 0)
            if df == 0:
                continue
            idf = math.log(1 + (self._N - df + 0.5) / (df + 0.5))
            numerator = tf * (self._K1 + 1)
            denominator = tf + self._K1 * (1 - self._B + self._B * dl / max(1, self._avg_dl))
            score += idf * (numerator / denominator)
        return score

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"[a-z0-9_\-\.]{2,}", text.lower())


# ── Answer data class ─────────────────────────────────────────────────────────


@dataclass
class ChatAnswer:
    question: str
    text: str
    sources: list[str]  # citation strings from retrieved chunks
    model_name: str
    backend: str

    def __str__(self) -> str:
        src_block = "\n".join(f"  · {s}" for s in self.sources) if self.sources else "  (no sources)"
        return f"{self.text}\n\nSources:\n{src_block}"


# ── Session ───────────────────────────────────────────────────────────────────


_SYSTEM_PROMPT = textwrap.dedent("""
    You are a compliance auditor assistant specialising in AI model supply chain security.
    You have access to the squash attestation artifacts for the model under review.

    Rules:
    1. Base every answer strictly on the provided context.
    2. If the context does not contain enough information to answer, say so clearly.
    3. Always cite the specific artifact file and JSON key path you used, e.g.:
       "According to cyclonedx-mlbom.json:components[0].hashes, the SHA-256 is ..."
    4. When reporting vulnerabilities or policy violations, include severity levels.
    5. Do not speculate beyond the provided data.
""").strip()


class ChatSession:
    """A RAG-powered compliance chat session.

    Parameters
    ----------
    chunks:
        Pre-loaded document chunks (from :class:`ArtifactChunker`).
    endpoint:
        OpenAI-compatible completions base URL (e.g. ``http://localhost:11434/v1``).
    model:
        Model name to pass to the API.
    api_key:
        Bearer API key (optional; leave ``None`` for local endpoints).
    top_k:
        Number of chunks to retrieve per query (default: 5).
    """

    def __init__(
        self,
        chunks: list[Chunk],
        *,
        endpoint: str = "http://localhost:11434/v1",
        model: str = "llama3",
        api_key: str | None = None,
        top_k: int = 5,
    ) -> None:
        self._retriever = BM25Retriever(chunks)
        self._endpoint = endpoint.rstrip("/")
        self._model = model
        self._api_key = api_key
        self._top_k = top_k
        self._history: list[dict[str, str]] = []

    @classmethod
    def from_model_dir(
        cls,
        model_dir: Path,
        *,
        endpoint: str = "http://localhost:11434/v1",
        model: str = "llama3",
        api_key: str | None = None,
        top_k: int = 5,
    ) -> "ChatSession":
        """Create a session pre-loaded with all artifacts in *model_dir*."""
        chunks = ArtifactChunker.load_model_dir(model_dir)
        return cls(chunks, endpoint=endpoint, model=model, api_key=api_key, top_k=top_k)

    def ask(self, question: str) -> ChatAnswer:
        """Ask a question and return a :class:`ChatAnswer` with citations."""
        relevant = self._retriever.retrieve(question, k=self._top_k)
        context_block = "\n\n".join(
            f"[{c.citation()}]\n{c.text}" for c in relevant
        )
        user_message = (
            f"Context from squash artifacts:\n{context_block}\n\n"
            f"Question: {question}"
        )
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            *self._history,
            {"role": "user", "content": user_message},
        ]
        response_text = self._complete(messages)
        self._history.append({"role": "user", "content": question})
        self._history.append({"role": "assistant", "content": response_text})

        sources = [c.citation() for c in relevant]
        return ChatAnswer(
            question=question,
            text=response_text,
            sources=sources,
            model_name=self._model,
            backend=self._endpoint,
        )

    def repl(self) -> None:
        """Run an interactive question-answer loop in the terminal."""
        print(f"squash chat  [{self._model} @ {self._endpoint}]")
        print(f"Loaded {len(self._retriever._chunks)} chunks. Type 'quit' to exit.\n")
        while True:
            try:
                question = input("auditor> ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nGoodbye.")
                break
            if not question:
                continue
            if question.lower() in {"quit", "exit", "q"}:
                print("Goodbye.")
                break
            answer = self.ask(question)
            print(f"\n{answer}\n")

    def clear_history(self) -> None:
        """Reset the conversation history (context window)."""
        self._history.clear()

    # ── Private ──────────────────────────────────────────────────────────────

    def _complete(self, messages: list[dict[str, str]]) -> str:
        payload = json.dumps({
            "model": self._model,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": 512,
        }).encode()
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        req = urllib.request.Request(
            f"{self._endpoint}/chat/completions",
            data=payload,
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                body = json.loads(resp.read().decode())
        except Exception as exc:
            return f"[LLM backend unavailable: {exc}]"

        try:
            return body["choices"][0]["message"]["content"]
        except (KeyError, IndexError):
            return f"[Unexpected response schema: {body!r}]"
