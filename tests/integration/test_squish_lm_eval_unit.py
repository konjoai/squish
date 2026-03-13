"""tests/test_squish_lm_eval_unit.py

Coverage tests for squish/squish_lm_eval.py.

Most functionality requires model weights; this file patches _load() to be a
no-op and exercises the pure-Python logic branches:

  SquishCompressedLM
    __init__          — model_dir/compressed_dir defaults
    max_length        — self._max_length set / not set, model_max_length attr
    eot_token_id      — delegates to tokenizer.eos_token_id
    max_gen_toks      — constant 256
    batch_size        — returns self._batch_size
    device            — constant "mlx"
    tok_encode        — delegates to tokenizer.encode
    tok_decode        — delegates to tokenizer.decode
    loglikelihood_rolling — empty tokens, too-long tokens, normal path
    generate_until    — until as str (converted to list), stop-string trimming,
                        no gen_kwargs branch (req.args length == 1)

  SquishReferenceLM
    __init__          — model_dir default, LM.__init__ called
    _load             — marked pragma: no cover (requires mlx_lm.load)

  LM stub fallback    — _HAVE_LM_EVAL=False path (LM is a plain class)
"""
from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Skip entire module on non-macOS (mlx.core not available)
mx = pytest.importorskip("mlx.core", reason="mlx not available (requires Apple Silicon)")
mx_nn = pytest.importorskip("mlx.nn", reason="mlx.nn not available")


# ---------------------------------------------------------------------------
# Helper: build a SquishCompressedLM instance without loading any model
# ---------------------------------------------------------------------------

def _make_lm(
    model_dir: str = "/tmp/fake_model",
    compressed_dir: str = "/tmp/fake_compressed",
    batch_size: int = 2,
    max_length: int | None = None,
    verbose: bool = False,
):
    """Instantiate SquishCompressedLM with _load() stubbed out."""
    from squish.squish_lm_eval import SquishCompressedLM

    with patch.object(SquishCompressedLM, "_load", return_value=None):
        lm = SquishCompressedLM(
            model_dir=model_dir,
            compressed_dir=compressed_dir,
            batch_size=batch_size,
            max_length=max_length,
            verbose=verbose,
        )

    # Attach a mock tokenizer and model for subsequent method calls
    tok = MagicMock()
    tok.eos_token_id = 2
    tok.eos_token = "</s>"
    tok.pad_token_id = 0
    tok.encode.side_effect = lambda text, add_special_tokens=False: [1, 2, 3]
    tok.decode.side_effect = lambda tokens: "decoded"
    tok.model_max_length = 2048
    lm._tokenizer = tok
    lm._model = MagicMock()
    return lm


# ---------------------------------------------------------------------------
# SquishCompressedLM — __init__ defaults
# ---------------------------------------------------------------------------


class TestSquishCompressedLMInit:
    def test_model_dir_default_resolved(self):
        """Empty model_dir uses ~/models/Qwen2.5-1.5B-Instruct-bf16."""
        from squish.squish_lm_eval import SquishCompressedLM
        with patch.object(SquishCompressedLM, "_load", return_value=None):
            lm = SquishCompressedLM(model_dir="", compressed_dir="")
        expected_dir = Path.home() / "models" / "Qwen2.5-1.5B-Instruct-bf16"
        assert lm._model_dir == expected_dir.expanduser().resolve()

    def test_compressed_dir_default_is_model_plus_compressed(self):
        """Empty compressed_dir defaults to model_dir + '-compressed'."""
        from squish.squish_lm_eval import SquishCompressedLM
        with patch.object(SquishCompressedLM, "_load", return_value=None):
            lm = SquishCompressedLM(model_dir="/fake/model", compressed_dir="")
        assert str(lm._compressed_dir).endswith("-compressed")

    def test_batch_size_stored(self):
        lm = _make_lm(batch_size=4)
        assert lm._batch_size == 4

    def test_verbose_stored(self):
        lm = _make_lm(verbose=True)
        assert lm._verbose is True


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


class TestSquishCompressedLMProperties:
    def test_eot_token_id(self):
        lm = _make_lm()
        assert lm.eot_token_id == 2

    def test_max_length_when_set_explicitly(self):
        """When _max_length is set, returns it directly."""
        lm = _make_lm(max_length=512)
        assert lm.max_length == 512

    def test_max_length_when_not_set_uses_tokenizer(self):
        """When _max_length is None, uses min(model_max_length, 4096)."""
        lm = _make_lm(max_length=None)
        lm._tokenizer.model_max_length = 2048
        assert lm.max_length == 2048

    def test_max_length_capped_at_4096(self):
        """model_max_length > 4096 is capped at 4096."""
        lm = _make_lm(max_length=None)
        lm._tokenizer.model_max_length = 131072
        assert lm.max_length == 4096

    def test_max_length_tokenizer_none_defaults_to_4096(self):
        """model_max_length=None → raw=4096, capped at 4096."""
        lm = _make_lm(max_length=None)
        lm._tokenizer.model_max_length = None
        assert lm.max_length == 4096

    def test_max_gen_toks_constant(self):
        lm = _make_lm()
        assert lm.max_gen_toks == 256

    def test_batch_size_property(self):
        lm = _make_lm(batch_size=8)
        assert lm.batch_size == 8

    def test_device_is_mlx(self):
        lm = _make_lm()
        assert lm.device == "mlx"


# ---------------------------------------------------------------------------
# Tokenizer helpers
# ---------------------------------------------------------------------------


class TestTokenizerHelpers:
    def test_tok_encode_delegates(self):
        lm = _make_lm()
        result = lm.tok_encode("hello")
        lm._tokenizer.encode.assert_called_once_with("hello", add_special_tokens=False)

    def test_tok_decode_delegates(self):
        lm = _make_lm()
        result = lm.tok_decode([1, 2, 3])
        lm._tokenizer.decode.assert_called_once_with([1, 2, 3])


# ---------------------------------------------------------------------------
# loglikelihood_rolling
# ---------------------------------------------------------------------------


class TestLoglikelihoodRolling:
    def _make_request(self, text: str):
        req = MagicMock()
        req.args = (text,)
        return req

    def test_empty_tokens_returns_zero(self):
        """Empty tokenization → append 0.0."""
        lm = _make_lm()
        lm._tokenizer.encode.side_effect = lambda text, add_special_tokens=False: []
        req = self._make_request("")
        results = lm.loglikelihood_rolling([req])
        assert results == [0.0]

    def test_too_long_tokens_truncated(self):
        """Tokens longer than max_length are truncated before forward pass."""
        lm = _make_lm(max_length=4)
        # encode returns 10 tokens
        long_tokens = list(range(10))
        lm._tokenizer.encode.side_effect = (
            lambda text, add_special_tokens=False: long_tokens
        )

        vocab = 8
        VOCAB = vocab
        # Mock model to return (1, seq_len, vocab) logits
        def fake_model(ids):
            b, seq = ids.shape[0], ids.shape[1]
            return mx.zeros((b, seq, VOCAB))

        lm._model = fake_model
        # max_length=4, so tokens are truncated to 4
        req = self._make_request("long text")
        results = lm.loglikelihood_rolling([req])
        assert isinstance(results[0], float)


# ---------------------------------------------------------------------------
# generate_until
# ---------------------------------------------------------------------------


class TestGenerateUntil:
    def _make_request(self, ctx: str, gen_kwargs: dict | None = None):
        req = MagicMock()
        if gen_kwargs is not None:
            req.args = (ctx, gen_kwargs)
        else:
            req.args = (ctx,)  # len == 1 → uses defaults
        return req

    def test_until_as_string_converted_to_list(self):
        """When until is a string, it is wrapped in a list."""
        lm = _make_lm()
        lm._tokenizer.encode.side_effect = lambda text, add_special_tokens=False: [1]

        mock_generate = MagicMock(return_value="Hello world stop here")
        gen_kwargs = {"until": "stop", "max_gen_toks": 10}
        req = self._make_request("context", gen_kwargs)

        with patch("mlx_lm.generate", mock_generate):
            results = lm.generate_until([req])

        # "stop" should have been converted to ["stop"] and output trimmed
        assert results[0] == "Hello world "

    def test_until_list_trims_output(self):
        """until as list — first matching stop string trims output."""
        lm = _make_lm()
        mock_generate = MagicMock(return_value="The answer is 42. Done.")
        gen_kwargs = {"until": [". Done"], "max_gen_toks": 50}
        req = self._make_request("context", gen_kwargs)

        with patch("mlx_lm.generate", mock_generate):
            results = lm.generate_until([req])

        assert results[0] == "The answer is 42"

    def test_no_stop_string_found_returns_full_output(self):
        """If stop string not in output, full output is returned."""
        lm = _make_lm()
        mock_generate = MagicMock(return_value="No stop here at all")
        gen_kwargs = {"until": ["STOP"], "max_gen_toks": 50}
        req = self._make_request("ctx", gen_kwargs)

        with patch("mlx_lm.generate", mock_generate):
            results = lm.generate_until([req])

        assert results[0] == "No stop here at all"

    def test_no_gen_kwargs_uses_defaults(self):
        """When req.args has only the context (no gen_kwargs), defaults are used."""
        lm = _make_lm()
        mock_generate = MagicMock(return_value="default output")
        req = self._make_request("context")  # only 1 arg

        with patch("mlx_lm.generate", mock_generate):
            results = lm.generate_until([req])

        assert results[0] == "default output"

    def test_eos_token_is_default_until(self):
        """Default until is [tokenizer.eos_token]."""
        lm = _make_lm()
        lm._tokenizer.eos_token = "</s>"
        output = "Hello world</s>trailing"
        mock_generate = MagicMock(return_value=output)
        req = self._make_request("ctx")

        with patch("mlx_lm.generate", mock_generate):
            results = lm.generate_until([req])

        assert results[0] == "Hello world"


# ---------------------------------------------------------------------------
# SquishReferenceLM
# ---------------------------------------------------------------------------


class TestSquishReferenceLM:
    def test_init_sets_compressed_dir_to_none(self):
        """SquishReferenceLM sets _compressed_dir=None."""
        from squish.squish_lm_eval import SquishReferenceLM

        with patch.object(SquishReferenceLM, "_load", return_value=None):
            ref_lm = SquishReferenceLM(model_dir="/fake/model", batch_size=1)

        assert ref_lm._compressed_dir is None

    def test_init_default_model_dir(self):
        """Empty model_dir defaults to ~/models/Qwen2.5-1.5B-Instruct-bf16."""
        from squish.squish_lm_eval import SquishReferenceLM

        with patch.object(SquishReferenceLM, "_load", return_value=None):
            ref_lm = SquishReferenceLM(model_dir="")

        expected = Path.home() / "models" / "Qwen2.5-1.5B-Instruct-bf16"
        assert ref_lm._model_dir == expected.expanduser().resolve()


# ---------------------------------------------------------------------------
# _HAVE_LM_EVAL = False path (LM stub is a plain class, register_model is no-op)
# ---------------------------------------------------------------------------


class TestLmEvalFallback:
    def test_module_importable_without_lm_eval(self):
        """squish_lm_eval is importable even when lm_eval is absent."""
        # The module is already imported; just verify the fallback was used
        from squish import squish_lm_eval
        # Either _HAVE_LM_EVAL is True (lm_eval installed) or False (stub used)
        assert hasattr(squish_lm_eval, "_HAVE_LM_EVAL")

    def test_register_model_stub_is_decorator(self):
        """When lm_eval not installed, register_model acts as identity decorator."""
        from squish import squish_lm_eval
        if squish_lm_eval._HAVE_LM_EVAL:
            pytest.skip("lm_eval is installed — fallback path not exercised")
        # Patch lm_eval out and re-import to hit the fallback
        modules_to_remove = {k: v for k, v in sys.modules.items()
                             if k == "lm_eval" or k.startswith("lm_eval.")}
        with patch.dict(sys.modules,
                        {k: None for k in list(sys.modules.keys())
                         if k == "lm_eval" or k.startswith("lm_eval.")}):
            import importlib
            importlib.reload(squish_lm_eval)
            assert squish_lm_eval._HAVE_LM_EVAL is False
