"""squish/_eval_torch.py

Torch-based lm-evaluation-harness wrapper for Squish compressed models.

This is the Linux/CUDA counterpart to SquishCompressedLM in squish_lm_eval.py.
It loads a Squish npy-dir compressed model via compressed_loader_torch, runs
forward passes with PyTorch, and exposes the same loglikelihood / generate_until
interface required by lm-eval >= 0.4.x.

Registered as "squish-torch" in the lm-eval registry.

Usage:
    lm_eval \\
        --model squish-torch \\
        --model_args model_dir=~/models/Qwen2.5-7B-Instruct-bf16, \\
                     compressed_dir=~/models/Qwen2.5-7B-Instruct-compressed \\
        --tasks arc_easy \\
        --num_fewshot 0 \\
        --limit 200
"""

from __future__ import annotations

__all__ = ["SquishCompressedLMTorch", "SquishReferenceLMTorch"]

import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

# ── lm-eval base class (graceful stub if not installed) ──────────────────────
try:
    from lm_eval.api.instance import Instance
    from lm_eval.api.model import LM
    from lm_eval.api.registry import register_model
    _HAVE_LM_EVAL = True
except ImportError:
    class LM:                       # type: ignore[no-redef]
        pass
    class Instance:                 # type: ignore[no-redef]
        pass
    def register_model(*a, **kw):
        def _dec(cls):
            return cls
        return _dec
    _HAVE_LM_EVAL = False

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────


@register_model("squish-torch")
class SquishCompressedLMTorch(LM):
    """
    lm-evaluation-harness LM subclass that runs inference on a Squish compressed
    model on Linux (CUDA / CPU) using PyTorch and transformers.

    The model is loaded via compressed_loader_torch.load_compressed_model_torch
    which dequantizes the npy-dir INT4 weights into a HuggingFace CausalLM model.
    BF16 fallback is used when no compressed dir is supplied.

    Attributes accepted by lm-eval model_args:
        model_dir        : path to original BF16 safetensors model directory
        compressed_dir   : path to squish npy-dir output of `squish compress`
        device           : "cuda", "cpu", or "auto" (default: auto-detect)
        batch_size       : forward-pass batch dimension (default: 4)
        max_length       : override context-window cap (default: 4096)
    """

    def __init__(
        self,
        model_dir: str = "",
        compressed_dir: str = "",
        device: str = "auto",
        batch_size: int = 4,
        max_length: int | None = None,
        verbose: bool = False,
        trust_remote_code: bool = True,
    ):
        super().__init__()

        if not model_dir:
            model_dir = str(Path.home() / "models" / "Qwen2.5-1.5B-Instruct-bf16")
        if not compressed_dir:
            compressed_dir = model_dir + "-compressed"

        self._model_dir = Path(model_dir).expanduser().resolve()
        self._compressed_dir = Path(compressed_dir).expanduser().resolve()
        self._batch_size = int(batch_size)
        self._max_length = max_length
        self._verbose = verbose
        self._trust_remote_code = trust_remote_code

        # Resolve device
        if device == "auto":
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self._device = device

        self._model: Any = None
        self._tokenizer: Any = None

        logger.info("SquishCompressedLMTorch: loading model on %s …", self._device)
        t0 = time.perf_counter()
        self._load()
        logger.info(
            "SquishCompressedLMTorch: model ready in %.1fs on %s",
            time.perf_counter() - t0,
            self._device,
        )

    def _load(self) -> None:
        """Load the compressed model — npy-dir path when compressed_dir exists,
        otherwise falls back to loading the BF16 model via transformers."""
        npy_dir = self._compressed_dir

        _has_tensors = (npy_dir.exists() and (
            (npy_dir / "tensors").is_dir()
            or bool(list(npy_dir.glob("tensors/*.npy"))) if (npy_dir / "tensors").is_dir() else False
        ))
        if not _has_tensors:
            _has_tensors = (
                npy_dir.is_dir()
                and (
                    any(npy_dir.glob("*__q4a.npy"))
                    or (npy_dir / "tensors").is_dir()
                )
            )

        if _has_tensors:
            from squish.compressed_loader_torch import load_compressed_model_torch
            self._model, self._tokenizer = load_compressed_model_torch(
                npy_dir=str(self._compressed_dir),
                model_dir=str(self._model_dir),
                device=self._device,
                verbose=self._verbose,
            )
        else:
            # BF16 fallback — useful for reference model comparison
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                str(self._model_dir),
                trust_remote_code=self._trust_remote_code,
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                str(self._model_dir),
                torch_dtype=torch.bfloat16,
                device_map=self._device,
                trust_remote_code=self._trust_remote_code,
            )
        self._model.eval()

    # ── lm-eval required properties ──────────────────────────────────────────

    @property
    def eot_token_id(self) -> int:
        return self._tokenizer.eos_token_id

    @property
    def max_length(self) -> int:
        if self._max_length:
            return self._max_length
        raw = getattr(self._tokenizer, "model_max_length", 4096) or 4096
        return min(raw, 4096)

    @property
    def max_gen_toks(self) -> int:
        return 256

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def device(self) -> str:
        return self._device

    # ── tokeniser helpers ─────────────────────────────────────────────────────

    def tok_encode(self, string: str) -> list[int]:
        return self._tokenizer.encode(string, add_special_tokens=False)

    def tok_decode(self, tokens) -> str:
        return self._tokenizer.decode(tokens)

    # ── core forward pass ─────────────────────────────────────────────────────

    @torch.inference_mode()
    def _forward_logprobs(self, token_ids: list[int]) -> np.ndarray:
        """Run one forward pass; return (seq_len, vocab_size) log-probs as numpy."""
        ids = torch.tensor([token_ids], dtype=torch.long, device=self._device)
        logits = self._model(ids).logits[0]            # (seq_len, vocab)
        lp = F.log_softmax(logits.float(), dim=-1)
        return lp.cpu().numpy()

    @torch.inference_mode()
    def _forward_selected_logprobs(
        self,
        token_ids: list[int],
        cont_tokens: list[int],
    ) -> tuple[list[float], list[bool]]:
        """
        Efficient single forward pass — gather only continuation log-probs.

        Avoids materialising the full (seq_len × vocab) tensor to numpy;
        computes argmax on device and transfers only n_cont scalars.
        """
        ids = torch.tensor([token_ids], dtype=torch.long, device=self._device)  # (1, T)
        logits = self._model(ids).logits[0]   # (T, vocab)

        n_cont = len(cont_tokens)
        n_ctx = len(token_ids) - n_cont
        cont_logits = logits[n_ctx - 1 : n_ctx - 1 + n_cont]  # (n_cont, vocab)
        cont_lp = F.log_softmax(cont_logits.float(), dim=-1)   # (n_cont, vocab)

        target = torch.tensor(cont_tokens, dtype=torch.long, device=self._device)
        selected_lp = cont_lp[torch.arange(n_cont, device=self._device), target]
        argmax_ids = cont_logits.argmax(dim=-1)
        is_greedy = (argmax_ids == target)

        return selected_lp.cpu().tolist(), is_greedy.cpu().tolist()

    # ── loglikelihood ─────────────────────────────────────────────────────────

    def loglikelihood(
        self, requests: list[Instance]
    ) -> list[tuple[float, bool]]:
        """Compute P(continuation | context) for every (context, continuation) pair."""
        results: list[tuple[float, bool]] = []

        batch: list[tuple[list[int], list[int]]] = []
        _n_done = 0
        _n_total = len(requests)
        _t_start = time.perf_counter()

        def _flush_batch() -> None:
            nonlocal _n_done
            if not batch:
                return

            pad_id = getattr(self._tokenizer, "pad_token_id", 0) or 0
            max_len = max(len(c) + len(k) for c, k in batch)
            seqs: list[list[int]] = []
            meta: list[tuple[int, int]] = []

            for ctx_toks, cont_toks in batch:
                full = ctx_toks + cont_toks
                pad = [pad_id] * (max_len - len(full))
                seqs.append(full + pad)
                meta.append((len(ctx_toks), len(cont_toks)))

            ids_batch = torch.tensor(seqs, dtype=torch.long, device=self._device)

            with torch.inference_mode():
                logits_all = self._model(ids_batch).logits  # (B, max_len, vocab)

            for i, (n_ctx, n_cont) in enumerate(meta):
                cont_logits = logits_all[i, n_ctx - 1 : n_ctx - 1 + n_cont]  # (n_cont, vocab)
                cont_lp = F.log_softmax(cont_logits.float(), dim=-1)
                target = torch.tensor(batch[i][1], dtype=torch.long, device=self._device)
                rows = torch.arange(n_cont, device=self._device)
                lp_sum = float(cont_lp[rows, target].sum().item())
                is_g = bool((cont_logits.argmax(dim=-1) == target).all().item())
                results.append((lp_sum, is_g))

            _n_done += len(batch)
            batch.clear()

            if _n_done % 500 < self._batch_size:
                elapsed = time.perf_counter() - _t_start
                rate = _n_done / elapsed if elapsed > 0 else 0
                eta = (_n_total - _n_done) / rate if rate > 0 else float("inf")
                logger.warning(
                    "loglikelihood: %d/%d done  %.1f req/s  ETA %.0fm",
                    _n_done, _n_total, rate, eta / 60,
                )

        for req in requests:
            ctx, cont = req.args[0], req.args[1]
            ctx_tokens = self.tok_encode(ctx)
            cont_tokens = self.tok_encode(cont)
            total = len(ctx_tokens) + len(cont_tokens)
            if total > self.max_length:
                keep = self.max_length - len(cont_tokens)
                ctx_tokens = ctx_tokens[-keep:]
            batch.append((ctx_tokens, cont_tokens))
            if len(batch) >= self._batch_size:
                _flush_batch()

        _flush_batch()
        return results

    def loglikelihood_rolling(self, requests: list[Instance]) -> list[float]:
        """Compute unconditional log-likelihood of a string."""
        results = []
        for req in requests:
            text = req.args[0]
            tokens = self.tok_encode(text)
            if not tokens:
                results.append(0.0)
                continue
            if len(tokens) > self.max_length:
                tokens = tokens[: self.max_length]
            log_probs = self._forward_logprobs(tokens)
            lp_sum = float(
                sum(log_probs[i - 1, tokens[i]] for i in range(1, len(tokens)))
            )
            results.append(lp_sum)
        return results

    def generate_until(self, requests: list[Instance]) -> list[str]:
        """Generate open-ended completions using transformers model.generate()."""
        from transformers import TextIteratorStreamer  # noqa: F401 (verify avail)

        results = []
        for req in requests:
            ctx = req.args[0]
            gen_kwargs = req.args[1] if len(req.args) > 1 else {}
            max_new = int(gen_kwargs.get("max_gen_toks", self.max_gen_toks))
            until = gen_kwargs.get("until", [self._tokenizer.eos_token])
            if isinstance(until, str):
                until = [until]

            ctx_ids = self._tokenizer.encode(ctx, return_tensors="pt").to(self._device)
            with torch.inference_mode():
                out_ids = self._model.generate(
                    ctx_ids,
                    max_new_tokens=max_new,
                    do_sample=False,
                    pad_token_id=getattr(self._tokenizer, "pad_token_id", 0) or 0,
                )
            # Decode only the newly generated tokens
            new_ids = out_ids[0, ctx_ids.shape[1]:]
            output = self._tokenizer.decode(new_ids, skip_special_tokens=True)

            for stop in until:
                idx = output.find(stop)
                if idx != -1:
                    output = output[:idx]

            results.append(output)

        return results


@register_model("squish-torch-reference")
class SquishReferenceLMTorch(SquishCompressedLMTorch):
    """
    lm-eval wrapper that loads the uncompressed BF16 model via transformers.
    Used as the reference baseline when comparing compressed vs uncompressed.
    """

    def __init__(
        self,
        model_dir: str = "",
        device: str = "auto",
        batch_size: int = 1,
        max_length: int | None = None,
        verbose: bool = False,
        trust_remote_code: bool = True,
        **kwargs,
    ):
        # Bypass SquishCompressedLMTorch.__init__; call LM.__init__ + our own _load
        LM.__init__(self)
        if not model_dir:
            model_dir = str(Path.home() / "models" / "Qwen2.5-1.5B-Instruct-bf16")
        self._model_dir = Path(model_dir).expanduser().resolve()
        self._compressed_dir = Path("/dev/null")  # not used
        self._batch_size = int(batch_size)
        self._max_length = max_length
        self._verbose = verbose
        self._trust_remote_code = trust_remote_code

        if device == "auto":
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self._device = device

        self._model = None
        self._tokenizer = None

        t0 = time.perf_counter()
        self._load_reference()
        logger.info(
            "SquishReferenceLMTorch: model ready in %.1fs on %s",
            time.perf_counter() - t0,
            self._device,
        )

    def _load_reference(self) -> None:  # pragma: no cover
        """Load the BF16 model directly via transformers."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            str(self._model_dir),
            trust_remote_code=self._trust_remote_code,
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            str(self._model_dir),
            torch_dtype=torch.bfloat16,
            device_map=self._device,
            trust_remote_code=self._trust_remote_code,
        )
        self._model.eval()
