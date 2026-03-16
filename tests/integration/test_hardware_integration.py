"""
tests/test_hardware_integration.py

Hardware integration tests for Squish on Apple Silicon.

These tests load a real quantised model and run actual inference — they are
intended to catch regressions that pure-unit mocks cannot detect.

Run conditions
──────────────
  • pytest --run-hardware                     # default model
  • pytest --run-hardware --model <hf-id>     # custom model

Skipped automatically in CI unless --run-hardware is passed.

Default model: mlx-community/Qwen2.5-1.5B-Instruct-4bit  (~900 MB, fast).
Choose a 4-bit quant so it fits in 8 GB unified memory with headroom.
"""
from __future__ import annotations

import sys
import time

import pytest

# ── Prerequisites ─────────────────────────────────────────────────────────────
_MLX_AVAILABLE = False
if sys.platform == "darwin":
    try:
        import mlx.core as _mx
        import mlx_lm as _mlx_lm  # noqa: F401

        _mx.array([0])          # force Metal initialisation
        _MLX_AVAILABLE = True
    except Exception:
        pass

_APPLE_SILICON = sys.platform == "darwin" and _MLX_AVAILABLE

_DEFAULT_MODEL = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"

# Latency guard-rails (generous — CI machines vary; real Apple Silicon is faster)
_MAX_LOAD_SECONDS    = 120.0   # cold load from disk (4-bit 1.5B)
_MAX_TTFT_SECONDS    =  10.0   # wall-clock time to first token
_MIN_TOKENS_EXPECTED =   5     # generation must produce at least this many tokens
_MAX_DECODE_SECONDS  =  30.0   # full 32-token decode budget

# Prompt known to produce a short, deterministic-ish answer
_PROBE_PROMPT  = "The capital of France is"
_PROBE_TOKENS  = 16   # ask for a short answer


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def hw_check():
    """Fail-fast if prerequisites are not met (skipped via conftest otherwise)."""
    if not _APPLE_SILICON:
        pytest.skip("requires Apple Silicon + MLX")


@pytest.fixture(scope="module")
def model_id(request):
    """Return the model ID from --model option or fall back to default."""
    opt = request.config.getoption("--model", default=None)
    return opt if opt else _DEFAULT_MODEL


@pytest.fixture(scope="module")
def loaded_model(hw_check, model_id):
    """Load the model once per module; yield (model, tokenizer, load_seconds)."""
    from squish.backend import BE

    t0 = time.perf_counter()
    model, tokenizer = BE.load_model(model_id)
    load_seconds = time.perf_counter() - t0
    yield model, tokenizer, load_seconds


# ── Tests ─────────────────────────────────────────────────────────────────────

@pytest.mark.hardware
class TestModelLoad:
    """Verify that model loading completes within an acceptable wall-clock budget."""

    def test_load_succeeds(self, loaded_model):
        model, tokenizer, _ = loaded_model
        assert model is not None, "BE.load_model returned a None model"
        assert tokenizer is not None, "BE.load_model returned a None tokenizer"

    def test_load_time_within_budget(self, loaded_model):
        _, _, load_seconds = loaded_model
        assert load_seconds < _MAX_LOAD_SECONDS, (
            f"Model load took {load_seconds:.1f}s — exceeds limit of {_MAX_LOAD_SECONDS}s. "
            f"Check for network fetching or very slow disk I/O."
        )

    def test_tokenizer_encodes_basic_text(self, loaded_model):
        _, tokenizer, _ = loaded_model
        # Tokenizer must be callable and produce at least 1 token
        tokens = tokenizer.encode("Hello")
        assert len(tokens) >= 1, "Tokenizer produced an empty token list for 'Hello'"


@pytest.mark.hardware
class TestGeneration:
    """Verify that stream_generate produces plausible output for a simple prompt."""

    def test_returns_non_empty_text(self, loaded_model):
        model, tokenizer, _ = loaded_model
        from squish.backend import BE

        chunks = list(BE.stream_generate(
            model, tokenizer, _PROBE_PROMPT,
            max_tokens=_PROBE_TOKENS, temperature=0.0,
        ))
        full_text = "".join(c for c, _ in chunks)
        assert len(full_text.strip()) > 0, (
            f"stream_generate produced empty output for prompt: {_PROBE_PROMPT!r}"
        )

    def test_minimum_token_count(self, loaded_model):
        model, tokenizer, _ = loaded_model
        from squish.backend import BE

        chunks = list(BE.stream_generate(
            model, tokenizer, _PROBE_PROMPT,
            max_tokens=_PROBE_TOKENS, temperature=0.0,
        ))
        assert len(chunks) >= _MIN_TOKENS_EXPECTED, (
            f"Expected at least {_MIN_TOKENS_EXPECTED} tokens; got {len(chunks)}"
        )

    def test_completion_within_time_budget(self, loaded_model):
        model, tokenizer, _ = loaded_model
        from squish.backend import BE

        t0 = time.perf_counter()
        for _ in BE.stream_generate(
            model, tokenizer, _PROBE_PROMPT,
            max_tokens=_PROBE_TOKENS, temperature=0.0,
        ):
            pass
        elapsed = time.perf_counter() - t0

        assert elapsed < _MAX_DECODE_SECONDS, (
            f"Generation of {_PROBE_TOKENS} tokens took {elapsed:.2f}s — "
            f"exceeds {_MAX_DECODE_SECONDS}s guard."
        )

    def test_no_degenerate_repetition(self, loaded_model):
        """Output must not be a single repeated character or token (model collapse)."""
        model, tokenizer, _ = loaded_model
        from squish.backend import BE

        chunks = list(BE.stream_generate(
            model, tokenizer, _PROBE_PROMPT,
            max_tokens=32, temperature=0.0,
        ))
        full_text = "".join(c for c, _ in chunks).strip()

        if len(full_text) < 4:
            pytest.skip("output too short to evaluate repetition")

        # Count unique 2-grams over characters
        bigrams = {full_text[i : i + 2] for i in range(len(full_text) - 1)}
        unique_ratio = len(bigrams) / max(len(full_text) - 1, 1)
        assert unique_ratio >= 0.10, (
            f"Output appears degenerate — bigram uniqueness ratio is only {unique_ratio:.2%}.\n"
            f"Full output: {full_text!r}"
        )

    def test_output_contains_word_characters(self, loaded_model):
        """Model must produce at least some alphanumeric characters."""
        model, tokenizer, _ = loaded_model
        import re

        from squish.backend import BE

        chunks = list(BE.stream_generate(
            model, tokenizer, _PROBE_PROMPT,
            max_tokens=_PROBE_TOKENS, temperature=0.0,
        ))
        full_text = "".join(c for c, _ in chunks)
        assert re.search(r"\w", full_text), (
            f"Output contains no word characters: {full_text!r}"
        )


@pytest.mark.hardware
class TestTTFT:
    """Measure time-to-first-token and verify it falls within an expected bound."""

    def test_ttft_within_budget(self, loaded_model):
        model, tokenizer, _ = loaded_model
        from squish.backend import BE

        gen = BE.stream_generate(
            model, tokenizer, _PROBE_PROMPT,
            max_tokens=_PROBE_TOKENS, temperature=0.0,
        )
        t0 = time.perf_counter()
        first_chunk = next(iter(gen), None)
        ttft = time.perf_counter() - t0

        assert first_chunk is not None, "stream_generate produced no tokens at all"
        assert ttft < _MAX_TTFT_SECONDS, (
            f"TTFT was {ttft:.3f}s — exceeds limit of {_MAX_TTFT_SECONDS}s. "
            f"First token: {first_chunk!r}"
        )

    def test_ttft_reported_in_stdout(self, loaded_model, capsys):
        """Smoke-test that we can measure and print TTFT without crashing."""
        model, tokenizer, _ = loaded_model
        from squish.backend import BE

        gen = BE.stream_generate(
            model, tokenizer, _PROBE_PROMPT,
            max_tokens=_PROBE_TOKENS, temperature=0.0,
        )
        t0 = time.perf_counter()
        first_chunk = next(iter(gen), None)
        ttft = time.perf_counter() - t0

        print(f"TTFT: {ttft * 1000:.1f} ms  first_token={first_chunk!r}")
        out = capsys.readouterr().out
        assert "TTFT:" in out
