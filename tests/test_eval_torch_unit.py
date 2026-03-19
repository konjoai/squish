"""tests/test_eval_torch_unit.py

Unit tests for squish/_eval_torch.py and squish/eval.py.

All tests mock _load() so no model weights or GPU are required.

Coverage:
  SquishCompressedLMTorch
    __init__         — defaults, explicit args, device auto-detect
    _load            — npy-dir path, BF16 fallback, model.eval() called
    properties       — eot_token_id, max_length (set/unset), max_gen_toks,
                       batch_size, device
    tok_encode/decode— delegates to tokenizer
    loglikelihood    — single, batched, truncation, flush remainder
    loglikelihood_rolling — empty tokens, over-length tokens, normal
    generate_until   — stop-string trimming, str until, no gen_kwargs

  SquishReferenceLMTorch
    __init__         — model_dir default, device auto, _load_reference called

  squish.eval router
    get_compressed_lm — returns correct class per platform
    get_reference_lm  — returns correct class per platform
    PLATFORM          — matches sys.platform
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_tokenizer(eos_id: int = 2, max_length: int = 4096) -> MagicMock:
    """Return a minimal mock tokenizer.

    encode.side_effect accepts arbitrary **kwargs so both
      tokenizer.encode(text, add_special_tokens=False)   (tok_encode path)
    and
      tokenizer.encode(text, return_tensors="pt")        (generate_until path)
    can be handled per-test by resetting side_effect.

    Default returns a 2-element list.
    """
    tok = MagicMock()
    tok.eos_token_id = eos_id
    tok.eos_token = "</s>"
    tok.pad_token_id = 0
    tok.model_max_length = max_length
    tok.encode.side_effect = lambda s, **kw: [1, 2]
    tok.decode.return_value = "decoded text"
    return tok


def _make_model(vocab_size: int = 20, batch: int = 2, seq: int = 6) -> MagicMock:
    """Return a mock causal-LM whose call returns (batch, seq, vocab) logits.

    Token 5 has slightly higher logit so argmax is deterministic.
    """
    model = MagicMock()
    logits = torch.zeros(batch, seq, vocab_size)
    logits[:, :, 5] = 1.0
    out = MagicMock()
    out.logits = logits
    model.return_value = out
    model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])
    return model


def _make_lm(
    model_dir: str = "/tmp/fake_model",
    compressed_dir: str = "/tmp/nonexistent_compressed",
    device: str = "cpu",
    batch_size: int = 2,
    max_length: int | None = None,
    vocab_size: int = 20,
    seq_len: int = 6,
):
    """Build a SquishCompressedLMTorch with _load() mocked out."""
    from squish._eval_torch import SquishCompressedLMTorch

    with patch.object(SquishCompressedLMTorch, "_load", return_value=None):
        lm = SquishCompressedLMTorch(
            model_dir=model_dir,
            compressed_dir=compressed_dir,
            device=device,
            batch_size=batch_size,
            max_length=max_length,
        )

    lm._tokenizer = _make_tokenizer()
    lm._model = _make_model(vocab_size=vocab_size, batch=batch_size, seq=seq_len)
    return lm


def _instance(ctx: str, cont: str) -> MagicMock:
    inst = MagicMock()
    inst.args = (ctx, cont)
    return inst


# ═══════════════════════════════════════════════════════════════════════════════
# TestSquishCompressedLMTorchInit
# ═══════════════════════════════════════════════════════════════════════════════

class TestSquishCompressedLMTorchInit:
    def test_default_model_dir(self):
        from squish._eval_torch import SquishCompressedLMTorch
        with patch.object(SquishCompressedLMTorch, "_load", return_value=None):
            lm = SquishCompressedLMTorch(device="cpu")
        assert "Qwen2.5-1.5B-Instruct-bf16" in str(lm._model_dir)

    def test_default_compressed_dir_derived_from_model_dir(self):
        from squish._eval_torch import SquishCompressedLMTorch
        with patch.object(SquishCompressedLMTorch, "_load", return_value=None):
            lm = SquishCompressedLMTorch(model_dir="/tmp/some_model", device="cpu")
        assert str(lm._compressed_dir).endswith("-compressed")

    def test_explicit_args_stored(self):
        lm = _make_lm(
            model_dir="/tmp/m",
            compressed_dir="/tmp/c",
            device="cpu",
            batch_size=8,
            max_length=2048,
        )
        assert lm._batch_size == 8
        assert lm._max_length == 2048
        assert lm._device == "cpu"

    def test_device_auto_selects_cpu_without_cuda(self):
        from squish._eval_torch import SquishCompressedLMTorch
        with patch.object(SquishCompressedLMTorch, "_load", return_value=None):
            with patch("torch.cuda.is_available", return_value=False):
                lm = SquishCompressedLMTorch(device="auto")
        assert lm._device == "cpu"

    def test_device_auto_selects_cuda_with_cuda(self):
        from squish._eval_torch import SquishCompressedLMTorch
        with patch.object(SquishCompressedLMTorch, "_load", return_value=None):
            with patch("torch.cuda.is_available", return_value=True):
                lm = SquishCompressedLMTorch(device="auto")
        assert lm._device == "cuda"

    def test_load_called_during_init(self):
        from squish._eval_torch import SquishCompressedLMTorch
        with patch.object(SquishCompressedLMTorch, "_load") as mock_load:
            SquishCompressedLMTorch(device="cpu")
        mock_load.assert_called_once()


# ═══════════════════════════════════════════════════════════════════════════════
# TestSquishCompressedLMTorchLoad
# ═══════════════════════════════════════════════════════════════════════════════

class TestSquishCompressedLMTorchLoad:
    def test_load_calls_bf16_fallback_when_no_npy_dir(self, tmp_path):
        """When compressed_dir doesn't exist, _load() falls back to transformers BF16."""
        from squish._eval_torch import SquishCompressedLMTorch

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        with patch.object(SquishCompressedLMTorch, "_load", return_value=None):
            lm = SquishCompressedLMTorch(
                model_dir=str(tmp_path / "model"),
                compressed_dir=str(tmp_path / "nonexistent"),
                device="cpu",
            )

        # Re-run _load() with patched transformers; patch via the package
        # because _load() imports them lazily with `from transformers import ...`.
        with (
            patch("transformers.AutoModelForCausalLM") as mock_hf,
            patch("transformers.AutoTokenizer") as mock_tok_cls,
        ):
            mock_hf.from_pretrained.return_value = mock_model
            mock_tok_cls.from_pretrained.return_value = mock_tokenizer
            lm._model_dir = tmp_path / "model"
            lm._compressed_dir = tmp_path / "nonexistent"
            lm._load()

        mock_hf.from_pretrained.assert_called_once()
        mock_model.eval.assert_called_once()

    def test_load_uses_compressed_loader_when_tensors_dir_exists(self, tmp_path):
        """When tensors/ subdir exists, _load() uses compressed_loader_torch."""
        from squish._eval_torch import SquishCompressedLMTorch

        npy_dir = tmp_path / "compressed"
        (npy_dir / "tensors").mkdir(parents=True)

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        with patch.object(SquishCompressedLMTorch, "_load", return_value=None):
            lm = SquishCompressedLMTorch(
                model_dir=str(tmp_path / "model"),
                compressed_dir=str(npy_dir),
                device="cpu",
            )

        with patch(
            "squish.compressed_loader_torch.load_compressed_model_torch",
            return_value=(mock_model, mock_tokenizer),
        ) as mock_load:
            lm._model_dir = tmp_path / "model"
            lm._compressed_dir = npy_dir
            lm._device = "cpu"
            lm._verbose = False
            lm._load()

        mock_load.assert_called_once()
        mock_model.eval.assert_called_once()


# ═══════════════════════════════════════════════════════════════════════════════
# TestSquishCompressedLMTorchProperties
# ═══════════════════════════════════════════════════════════════════════════════

class TestSquishCompressedLMTorchProperties:
    def test_eot_token_id(self):
        lm = _make_lm()
        lm._tokenizer.eos_token_id = 99
        assert lm.eot_token_id == 99

    def test_max_length_uses_override_when_set(self):
        lm = _make_lm(max_length=1024)
        assert lm.max_length == 1024

    def test_max_length_caps_at_4096(self):
        lm = _make_lm()
        lm._tokenizer.model_max_length = 128_000  # e.g. llama3
        assert lm.max_length == 4096

    def test_max_length_uses_model_max_when_smaller(self):
        lm = _make_lm()
        lm._tokenizer.model_max_length = 512
        assert lm.max_length == 512

    def test_max_length_default_when_none(self):
        lm = _make_lm()
        lm._tokenizer.model_max_length = None
        assert lm.max_length == 4096

    def test_max_gen_toks(self):
        assert _make_lm().max_gen_toks == 256

    def test_batch_size(self):
        assert _make_lm(batch_size=4).batch_size == 4

    def test_device_property(self):
        assert _make_lm(device="cpu").device == "cpu"

    def test_tok_encode_delegates(self):
        lm = _make_lm()
        lm._tokenizer.encode.side_effect = lambda s, **kw: [10, 20, 30]
        assert lm.tok_encode("hello") == [10, 20, 30]

    def test_tok_decode_delegates(self):
        lm = _make_lm()
        lm._tokenizer.decode.return_value = "world"
        assert lm.tok_decode([10, 20]) == "world"


# ═══════════════════════════════════════════════════════════════════════════════
# TestLoglikelihood
# ═══════════════════════════════════════════════════════════════════════════════

class TestLoglikelihood:
    """Tests for SquishCompressedLMTorch.loglikelihood().

    Logits shape convention: the mock always returns (batch_size, seq, vocab).
    For token cont=[5, 6] to be valid indices vocab must be > 6.
    We use vocab=20 throughout.
    """

    VOCAB = 20
    SEQ = 6  # must be >= max(n_ctx + n_cont) = 2 + 2 = 4

    def _lm(self, batch_size: int = 2) -> object:
        """LM with logits (batch_size, SEQ, VOCAB); encode always → [1, 2]."""
        lm = _make_lm(batch_size=batch_size, vocab_size=self.VOCAB, seq_len=self.SEQ)
        logits = torch.zeros(batch_size, self.SEQ, self.VOCAB)
        logits[:, :, 5] = 1.0  # token 5 is argmax everywhere
        out = MagicMock()
        out.logits = logits
        lm._model.return_value = out
        # encode: always returns [1, 2] (2 ctx or 2 cont tokens)
        lm._tokenizer.encode.side_effect = lambda s, **kw: [1, 2]
        return lm

    def test_single_request_returns_one_result(self):
        lm = self._lm()
        results = lm.loglikelihood([_instance("hello", "world")])
        assert len(results) == 1
        lp, is_greedy = results[0]
        assert isinstance(lp, float)
        assert isinstance(is_greedy, bool)

    def test_log_prob_is_non_positive(self):
        lm = self._lm()
        lp, _ = lm.loglikelihood([_instance("hello", "world")])[0]
        assert lp <= 0.0

    def test_is_greedy_true_when_cont_equals_argmax(self):
        """When the continuation IS the argmax token, is_greedy should be True."""
        lm = self._lm()
        # cont_toks = [5, 5]; argmax is always 5 → is_greedy True
        call_count = [0]
        def _enc(s, **kw):
            call_count[0] += 1
            return [1, 2] if call_count[0] % 2 != 0 else [5, 5]
        lm._tokenizer.encode.side_effect = _enc
        _, is_greedy = lm.loglikelihood([_instance("ctx", "cont")])[0]
        assert is_greedy is True

    def test_multiple_requests_batched(self):
        lm = self._lm(batch_size=2)
        reqs = [_instance(f"ctx{i}", f"cont{i}") for i in range(5)]
        results = lm.loglikelihood(reqs)
        assert len(results) == 5
        for lp, isg in results:
            assert isinstance(lp, float)
            assert isinstance(isg, bool)

    def test_truncation_skips_early_context_tokens(self):
        """When ctx+cont > max_length, context is left-trimmed; no exception raised."""
        # max_length=4, ctx=5 tokens, cont=2 → total 7 → ctx trimmed to 2
        lm = _make_lm(batch_size=1, vocab_size=self.VOCAB, seq_len=self.SEQ)
        lm._max_length = 4
        # Fresh logits consistent with batch_size=1
        out = MagicMock()
        out.logits = torch.zeros(1, self.SEQ, self.VOCAB)
        lm._model.return_value = out

        toggle = [False]
        def _enc_alternating(s, **kw):
            toggle[0] = not toggle[0]
            return [1, 2, 3, 4, 5] if toggle[0] else [6, 7]  # ctx 5t, cont 2t

        lm._tokenizer.encode.side_effect = _enc_alternating
        results = lm.loglikelihood([_instance("long context", "cont")])
        assert len(results) == 1


# ═══════════════════════════════════════════════════════════════════════════════
# TestLoglikelihoodRolling
# ═══════════════════════════════════════════════════════════════════════════════

class TestLoglikelihoodRolling:
    def test_empty_tokens_returns_zero(self):
        lm = _make_lm()
        lm._tokenizer.encode.side_effect = lambda s, **kw: []
        req = MagicMock()
        req.args = ("",)
        results = lm.loglikelihood_rolling([req])
        assert results == [0.0]

    def test_over_length_tokens_truncated_to_max(self):
        """Tokens beyond max_length are silently truncated before forward pass."""
        vocab, seq = 20, 4
        lm = _make_lm(vocab_size=vocab, seq_len=seq)
        lm._max_length = 4
        lm._tokenizer.encode.side_effect = lambda s, **kw: list(range(10))  # 10 > 4

        out = MagicMock()
        out.logits = torch.zeros(1, seq, vocab)
        lm._model.return_value = out

        req = MagicMock()
        req.args = ("some text",)
        lm.loglikelihood_rolling([req])

        # ids tensor passed to model must have width ≤ max_length
        passed_ids = lm._model.call_args[0][0]
        assert passed_ids.shape[1] <= 4

    def test_normal_path_returns_float(self):
        vocab, seq = 20, 4
        lm = _make_lm(vocab_size=vocab, seq_len=seq)
        lm._tokenizer.encode.side_effect = lambda s, **kw: [1, 2, 3, 4]

        out = MagicMock()
        out.logits = torch.zeros(1, seq, vocab)
        lm._model.return_value = out

        req = MagicMock()
        req.args = ("hello world",)
        results = lm.loglikelihood_rolling([req])
        assert len(results) == 1
        assert isinstance(results[0], float)

    def test_multiple_requests(self):
        vocab, seq = 20, 4
        lm = _make_lm(vocab_size=vocab, seq_len=seq)
        lm._tokenizer.encode.side_effect = lambda s, **kw: [1, 2, 3, 4]

        out = MagicMock()
        out.logits = torch.zeros(1, seq, vocab)
        lm._model.return_value = out

        reqs = [MagicMock(args=(f"text{i}",)) for i in range(3)]
        results = lm.loglikelihood_rolling(reqs)
        assert len(results) == 3
        assert all(isinstance(r, float) for r in results)


# ═══════════════════════════════════════════════════════════════════════════════
# TestGenerateUntil
# ═══════════════════════════════════════════════════════════════════════════════

class TestGenerateUntilTorch:
    """
    generate_until calls:
        ctx_ids = self._tokenizer.encode(ctx, return_tensors="pt").to(self._device)

    So tokenizer.encode must return a real tensor (has .to() and .shape).
    We set encode.side_effect=None and encode.return_value to a real tensor.
    """

    def _lm_for_generate(self, decoded: str = "output text"):
        lm = _make_lm()
        # Real tensor so .to() and .shape work
        ctx_tensor = torch.tensor([[1, 2, 3]])  # shape (1, 3)
        lm._tokenizer.encode.side_effect = None
        lm._tokenizer.encode.return_value = ctx_tensor
        # model.generate returns full ids including the 3 input tokens
        lm._model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5, 6]])
        lm._tokenizer.decode.return_value = decoded
        lm._tokenizer.eos_token = "</s>"
        lm._tokenizer.pad_token_id = 0
        return lm

    def test_stop_string_trimming(self):
        lm = self._lm_for_generate("hello STOP world")
        req = MagicMock()
        req.args = ("ctx", {"until": ["STOP"], "max_gen_toks": 32})
        results = lm.generate_until([req])
        assert "STOP" not in results[0]
        assert "hello" in results[0]

    def test_until_as_string_converted_to_list(self):
        """A plain string value for 'until' should be treated as a one-item list."""
        lm = self._lm_for_generate("result</s>extra")
        req = MagicMock()
        req.args = ("ctx", {"until": "</s>", "max_gen_toks": 32})
        results = lm.generate_until([req])
        assert "extra" not in results[0]

    def test_no_gen_kwargs_uses_defaults(self):
        """When req.args has only one element, defaults are used and no exception raised."""
        lm = self._lm_for_generate("output")
        req = MagicMock()
        req.args = ("ctx",)
        results = lm.generate_until([req])
        assert isinstance(results[0], str)

    def test_multiple_requests_returned_in_order(self):
        lm = _make_lm()
        ctx_tensor = torch.tensor([[1, 2]])
        lm._tokenizer.encode.side_effect = None
        lm._tokenizer.encode.return_value = ctx_tensor
        lm._model.generate.return_value = torch.tensor([[1, 2, 3]])
        lm._tokenizer.eos_token = "</s>"
        lm._tokenizer.pad_token_id = 0
        lm._tokenizer.decode.side_effect = ["first", "second", "third"]

        reqs = [MagicMock(args=(f"ctx{i}",)) for i in range(3)]
        results = lm.generate_until(reqs)
        assert results == ["first", "second", "third"]

    def test_generate_called_with_do_sample_false(self):
        """model.generate must be called with do_sample=False."""
        lm = self._lm_for_generate("out")
        req = MagicMock()
        req.args = ("ctx", {"until": [], "max_gen_toks": 64})
        lm.generate_until([req])
        call_kwargs = lm._model.generate.call_args[1]
        assert call_kwargs["do_sample"] is False

    def test_generate_called_with_max_new_tokens(self):
        lm = self._lm_for_generate("out")
        req = MagicMock()
        req.args = ("ctx", {"until": [], "max_gen_toks": 64})
        lm.generate_until([req])
        call_kwargs = lm._model.generate.call_args[1]
        assert call_kwargs["max_new_tokens"] == 64


# ═══════════════════════════════════════════════════════════════════════════════
# TestSquishReferenceLMTorch
# ═══════════════════════════════════════════════════════════════════════════════

class TestSquishReferenceLMTorch:
    def test_default_model_dir(self):
        from squish._eval_torch import SquishReferenceLMTorch
        with patch.object(SquishReferenceLMTorch, "_load_reference", return_value=None):
            lm = SquishReferenceLMTorch(device="cpu")
        assert "Qwen2.5-1.5B-Instruct-bf16" in str(lm._model_dir)

    def test_compressed_dir_is_devnull(self):
        from squish._eval_torch import SquishReferenceLMTorch
        with patch.object(SquishReferenceLMTorch, "_load_reference", return_value=None):
            lm = SquishReferenceLMTorch(device="cpu")
        assert str(lm._compressed_dir) == "/dev/null"

    def test_device_auto_cpu(self):
        from squish._eval_torch import SquishReferenceLMTorch
        with (
            patch.object(SquishReferenceLMTorch, "_load_reference", return_value=None),
            patch("torch.cuda.is_available", return_value=False),
        ):
            lm = SquishReferenceLMTorch(device="auto")
        assert lm._device == "cpu"

    def test_load_reference_called(self):
        from squish._eval_torch import SquishReferenceLMTorch
        with patch.object(SquishReferenceLMTorch, "_load_reference") as m:
            SquishReferenceLMTorch(device="cpu")
        m.assert_called_once()

    def test_inherits_loglikelihood(self):
        from squish._eval_torch import SquishReferenceLMTorch, SquishCompressedLMTorch
        assert SquishReferenceLMTorch.loglikelihood is SquishCompressedLMTorch.loglikelihood


# ═══════════════════════════════════════════════════════════════════════════════
# TestEvalRouter (squish/eval.py)
# ═══════════════════════════════════════════════════════════════════════════════

class TestEvalRouter:
    def test_platform_constant_is_valid(self):
        import squish.eval as ev
        assert ev.PLATFORM in ("darwin", "linux")

    def test_platform_matches_sys_platform(self):
        import squish.eval as ev
        expected = "darwin" if sys.platform == "darwin" else "linux"
        assert ev.PLATFORM == expected

    def test_get_compressed_lm_linux_calls_torch_class(self):
        """get_compressed_lm(platform='linux') instantiates SquishCompressedLMTorch."""
        import squish.eval as ev
        mock_cls = MagicMock(return_value=MagicMock())
        # Patch the attribute on the already-imported module so the lazy import
        # inside get_compressed_lm picks up the mock.
        with patch("squish._eval_torch.SquishCompressedLMTorch", mock_cls):
            ev.get_compressed_lm(platform="linux", model_dir="/tmp/m", device="cpu")
        mock_cls.assert_called_once_with(model_dir="/tmp/m", device="cpu")

    def test_get_reference_lm_linux_calls_torch_class(self):
        import squish.eval as ev
        mock_cls = MagicMock(return_value=MagicMock())
        with patch("squish._eval_torch.SquishReferenceLMTorch", mock_cls):
            ev.get_reference_lm(platform="linux", model_dir="/tmp/m", device="cpu")
        mock_cls.assert_called_once_with(model_dir="/tmp/m", device="cpu")

    def test_get_compressed_lm_auto_respects_platform_constant(self):
        """platform='auto' uses PLATFORM constant; forcing it to 'linux' → torch class."""
        import squish.eval as ev
        mock_cls = MagicMock(return_value=MagicMock())
        with patch.object(ev, "PLATFORM", "linux"):
            with patch("squish._eval_torch.SquishCompressedLMTorch", mock_cls):
                ev.get_compressed_lm(platform="auto", model_dir="/tmp/m", device="cpu")
        mock_cls.assert_called_once()

    def test_get_reference_lm_auto_respects_platform_constant(self):
        import squish.eval as ev
        mock_cls = MagicMock(return_value=MagicMock())
        with patch.object(ev, "PLATFORM", "linux"):
            with patch("squish._eval_torch.SquishReferenceLMTorch", mock_cls):
                ev.get_reference_lm(platform="auto", model_dir="/tmp/m", device="cpu")
        mock_cls.assert_called_once()


# ═══════════════════════════════════════════════════════════════════════════════
# TestLmEvalStub  — verify graceful fallback when lm_eval not installed
# ═══════════════════════════════════════════════════════════════════════════════

class TestLmEvalStub:
    def test_have_lm_eval_is_bool(self):
        from squish._eval_torch import _HAVE_LM_EVAL
        assert isinstance(_HAVE_LM_EVAL, bool)

    def test_lm_class_is_available(self):
        from squish._eval_torch import LM
        assert LM is not None

    def test_instance_class_is_available(self):
        from squish._eval_torch import Instance
        assert Instance is not None

    def test_squish_compressed_lm_torch_is_subclass_of_lm(self):
        from squish._eval_torch import SquishCompressedLMTorch, LM
        assert issubclass(SquishCompressedLMTorch, LM)
