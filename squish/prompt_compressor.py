#!/usr/bin/env python3
"""
squish/prompt_compressor.py

Token-budget prompt compression for long-context inference.

Two backends — both accessible via the same :func:`compress` entry point:

TF-IDF (default, zero extra dependencies)
    Splits the prompt into sentences, scores each sentence by the sum of its
    TF-IDF term weights, then greedily keeps the top-scoring sentences until
    the *ratio* budget is reached.  Fast (~1 ms for 1 K-word prompts).

LLMLingua (optional, ``pip install squish[llmlingua]``)
    Delegates to ``llmlingua.PromptCompressor`` for token-level selective
    compression.  Falls back to TF-IDF if llmlingua is not installed.

Usage
-----
    from squish.prompt_compressor import compress

    shorter = compress(long_prompt, ratio=0.5)          # keep 50 %
    shorter = compress(long_prompt, ratio=0.3,
                       question="What is the main theme?")
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# TF-IDF sentence scorer (pure Python + numpy)
# ---------------------------------------------------------------------------

def _sentence_split(text: str) -> list[str]:
    """Split *text* into a list of non-empty sentences."""
    # Split on sentence-terminal punctuation followed by whitespace or EOS
    raw = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in raw if s.strip()]


def _tfidf_compress(text: str, ratio: float, preserve_tokens: int = 0) -> str:
    """
    TF-IDF sentence-level compression.

    Keeps the top ``ceil(ratio * N)`` sentences ranked by the sum of their
    term TF-IDF scores, preserving original order.

    Parameters
    ----------
    text            : str   — input prompt (may contain multiple sentences)
    ratio           : float — fraction of sentences to keep (0 < ratio <= 1)
    preserve_tokens : int   — if > 0, the first ``preserve_tokens`` *words*
                              of ``text`` are never subject to compression
                              (useful to protect a system-prompt prefix so
                              that RadixAttention still hits on it).

    Returns
    -------
    Compressed str; returns *text* unchanged if fewer than 2 sentences.
    """
    import math

    import numpy as np

    # Split off an immutable prefix (system prompt) to protect it from pruning
    prefix = ""
    compressible = text
    if preserve_tokens > 0:
        words = text.split()
        if len(words) > preserve_tokens:
            prefix = " ".join(words[:preserve_tokens])
            compressible = " ".join(words[preserve_tokens:])
        else:
            return text  # entire text is within the preserved prefix

    sentences = _sentence_split(compressible)
    n = len(sentences)
    if n < 2:
        return text

    keep = max(1, math.ceil(ratio * n))
    if keep >= n:
        return text

    # Build term-frequency matrix
    # Tokenise: lowercase words, strip punctuation
    def _tokenise(s: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", s.lower())

    tokens_per_sent = [_tokenise(s) for s in sentences]
    vocab: dict[str, int] = {}
    for toks in tokens_per_sent:
        for t in toks:
            if t not in vocab:
                vocab[t] = len(vocab)
    V = len(vocab)
    if V == 0:
        return text

    # TF: raw count normalized by sentence length
    tf = np.zeros((n, V), dtype=np.float32)
    for i, toks in enumerate(tokens_per_sent):
        if not toks:
            continue
        for t in toks:
            tf[i, vocab[t]] += 1
        tf[i] /= len(toks)

    # IDF: log( (1 + N) / (1 + df) )  + 1  (smooth)
    df = np.count_nonzero(tf, axis=0).astype(np.float32)   # (V,)
    idf = np.log((1 + n) / (1 + df)) + 1.0

    # Sentence scores: sum of TF-IDF values
    scores = (tf * idf).sum(axis=1)  # (n,)

    # Select top-k indices, then restore order
    top_idx = np.argpartition(scores, -keep)[-keep:]
    top_idx_sorted = np.sort(top_idx)

    compressed_suffix = " ".join(sentences[i] for i in top_idx_sorted)
    if prefix:
        return prefix + " " + compressed_suffix
    return compressed_suffix


# ---------------------------------------------------------------------------
# Public compress() entry point
# ---------------------------------------------------------------------------

def compress(
    text: str,
    ratio: float = 0.5,
    question: str = "",
    min_tokens: int = 0,
    preserve_tokens: int = 0,
) -> str:
    """
    Compress *text* to approximately ``ratio`` of its original token count.

    Parameters
    ----------
    text            : str   — the prompt to compress
    ratio           : float — target fraction to keep  (0 < ratio ≤ 1)
    question        : str   — optional question/task hint  (used by LLMLingua)
    min_tokens      : int   — skip compression if word count < this threshold
    preserve_tokens : int   — protect the first N words of *text* from any
                              compression (keeps system-prompt prefix identical
                              across requests so RadixAttention cache hits).

    Returns
    -------
    Compressed str.  Falls back to the original *text* on any error.
    """
    if not text or ratio >= 1.0 or ratio <= 0.0:
        return text

    # Count words as a fast proxy for tokens
    word_count = len(text.split())
    if min_tokens > 0 and word_count < min_tokens:
        return text

    # Try LLMLingua first (higher quality)
    try:
        from llmlingua import PromptCompressor as _LLC  # type: ignore[import]
        _compressor = _LLC(
            model_name="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
            use_llmlingua2=True,
            device_map="cpu",
        )
        # Build forced_context to preserve the system-prompt prefix
        _kw: dict = {"ratio": ratio, "question": question or ""}
        if preserve_tokens > 0:
            words = text.split()
            if len(words) > preserve_tokens:
                _kw["context"] = " ".join(words[:preserve_tokens])
                text = " ".join(words[preserve_tokens:])
        result = _compressor.compress_prompt(text, **_kw)
        return result.get("compressed_prompt", text)
    except ImportError:
        pass  # llmlingua not installed — fall through to TF-IDF
    except Exception:
        pass  # llmlingua failed — fall through

    # TF-IDF fallback
    try:
        return _tfidf_compress(text, ratio=ratio, preserve_tokens=preserve_tokens)
    except Exception:
        return text  # never break the caller
