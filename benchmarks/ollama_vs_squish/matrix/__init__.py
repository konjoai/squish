"""Airtight Ollama-vs-Squish benchmark — reuse × context-length matrix.

This package rebuilds the head-to-head benchmark so every number is labelled for
exactly what it measures, across a matrix of prompt-reuse levels and context
lengths, with cache state *measured* per run for both systems, thermal control
preserved, fair like-for-like configs (Squish INT4 vs Ollama Q4_K_M), and full
statistical rigour (>=30 paired runs, paired Wilcoxon, Cliff's delta).

Execution requires Apple Silicon + MLX + a local Ollama + the on-disk models;
the orchestration modules (``systems``, ``cell``, ``thermal``, ``memory``) only
run there. The scientific core — corpus construction, statistics, cache-hit
classification, OOM/thermal math, reporting — is pure Python and unit-tested on
any platform.

Run order:
  1. ``python -m ...matrix.run_killtest``  — ONE cell (8k @ 50%), then STOP.
  2. (after human approval) ``python -m ...matrix.run_matrix --i-have-approved``.

See ``METHODOLOGY.md`` and ``ADVERSARIAL_REVIEW.md`` in this directory.
"""

from __future__ import annotations

__all__ = ["VERSION"]

VERSION = "1.0.0"
