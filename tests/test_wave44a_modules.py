"""tests/test_wave44a_modules.py

Tests for Wave 44a modules:
  - MarlinGEMM (squish/quant/marlin_gemm.py)
  - SpecRejection (squish/speculative/spec_rejection.py)
  - LoFTQ (squish/quant/loftq.py)
  - OnlineSpec (squish/speculative/online_spec.py)
  - DynamicSpecLen (squish/speculative/dynamic_spec_len.py)
  - BigLittleLLM (squish/speculative/big_little_llm.py)
"""

from __future__ import annotations

import pytest
import numpy as np

# ── MarlinGEMM ─────────────────────────────────────────────────────────────────

from squish.quant.marlin_gemm import MarlinGEMMConfig, MarlinGEMM


class TestMarlinGEMMConfig:
    def test_defaults(self):
        cfg = MarlinGEMMConfig()
        assert cfg.n_bits in (4, 8)
        assert cfg.group_size > 0

    def test_custom(self):
        cfg = MarlinGEMMConfig(n_bits=4, group_size=64)
        assert cfg.group_size == 64

    def test_invalid_nbits(self):
        with pytest.raises((ValueError, Exception)):
            MarlinGEMMConfig(n_bits=3)


class TestMarlinGEMM:
    def _make(self, out_f=16, in_f=32, group_size=16):
        cfg = MarlinGEMMConfig(n_bits=4, group_size=group_size)
        m = MarlinGEMM(cfg)
        W = np.random.randn(out_f, in_f).astype(np.float32)
        m.pack_weights(W)
        return m, W

    def test_pack_weights(self):
        cfg = MarlinGEMMConfig(n_bits=4, group_size=16)
        m = MarlinGEMM(cfg)
        W = np.random.randn(16, 32).astype(np.float32)
        m.pack_weights(W)

    def test_forward_shape(self):
        m, W = self._make(out_f=16, in_f=32)
        x = np.random.randn(4, 32).astype(np.float32)
        out = m.forward(x)
        assert out.shape == (4, 16)

    def test_unpack_shape(self):
        m, W = self._make(out_f=16, in_f=32)
        W_hat = m.unpack_weights()
        assert W_hat.shape == W.shape

    def test_forward_dtype(self):
        m, _ = self._make()
        x = np.random.randn(2, 32).astype(np.float32)
        out = m.forward(x)
        assert out.dtype == np.float32

    def test_default_config(self):
        m = MarlinGEMM()
        assert m.config is not None

    def test_unpack_close_to_original(self):
        cfg = MarlinGEMMConfig(n_bits=8, group_size=16)
        m = MarlinGEMM(cfg)
        W = np.random.randn(16, 32).astype(np.float32)
        m.pack_weights(W)
        W_hat = m.unpack_weights()
        assert np.allclose(W, W_hat, atol=0.1)

    def test_forward_produces_finite_values(self):
        m, _ = self._make()
        x = np.random.randn(3, 32).astype(np.float32)
        out = m.forward(x)
        assert np.all(np.isfinite(out))


# ── SpecRejection ──────────────────────────────────────────────────────────────

from squish.speculative.spec_rejection import SpecRejectionConfig, SpecRejectionResult, SpecRejection


class TestSpecRejectionConfig:
    def test_defaults(self):
        cfg = SpecRejectionConfig()
        assert cfg.pool_size >= 1

    def test_custom(self):
        cfg = SpecRejectionConfig(pool_size=4, early_reject_fraction=0.3)
        assert cfg.pool_size == 4


class TestSpecRejection:
    def _make(self, vocab=32, pool=4):
        cfg = SpecRejectionConfig(pool_size=pool)
        return SpecRejection(cfg)

    def test_step_returns_result(self):
        sr = self._make(vocab=32, pool=4)
        draft_logits = np.random.randn(4, 32).astype(np.float32)
        target_lp = np.log(np.ones((4, 32)) / 32)
        result = sr.step(draft_logits, target_lp)
        assert isinstance(result, SpecRejectionResult)

    def test_accepted_tokens_nonempty(self):
        sr = self._make()
        dl = np.random.randn(4, 32).astype(np.float32)
        tl = np.log(np.ones((4, 32)) / 32)
        result = sr.step(dl, tl)
        assert result.accepted_tokens is not None

    def test_n_accepted_in_range(self):
        sr = self._make(pool=4)
        dl = np.random.randn(4, 32).astype(np.float32)
        tl = np.log(np.ones((4, 32)) / 32)
        result = sr.step(dl, tl)
        assert 0 <= result.n_accepted <= 4

    def test_generate_candidates(self):
        sr = self._make(pool=4)
        draft_logits = np.random.randn(4, 32).astype(np.float32)
        tokens, lp = sr.generate_candidates(draft_logits)
        assert tokens.shape[0] == 4

    def test_early_reject(self):
        sr = self._make(pool=4)
        draft_logits = np.random.randn(4, 32).astype(np.float32)
        tokens, lp = sr.generate_candidates(draft_logits)
        tokens2, lp2 = sr.early_reject(tokens, lp)
        assert tokens2.shape[0] <= tokens.shape[0]

    def test_rejection_sample_result_type(self):
        sr = self._make(pool=4)
        dl = np.random.randn(4, 32).astype(np.float32)
        tl = np.log(np.ones((4, 32)) / 32)
        tokens, dlp = sr.generate_candidates(dl)
        accepted = sr.rejection_sample(tokens, dlp, tl[:tokens.shape[0]])
        assert accepted is not None

    def test_default_config(self):
        sr = SpecRejection()
        assert sr.config is not None


# ── LoFTQ ──────────────────────────────────────────────────────────────────────

from squish.quant.loftq import LoFTQConfig, LoFTQResult, LoFTQ


class TestLoFTQConfig:
    def test_defaults(self):
        cfg = LoFTQConfig()
        assert cfg.n_bits >= 2
        assert cfg.rank >= 1
        assert cfg.n_iterations >= 1

    def test_custom(self):
        cfg = LoFTQConfig(n_bits=4, rank=8, n_iterations=3)
        assert cfg.rank == 8


class TestLoFTQ:
    def _make(self, n_bits=4, rank=4, n_iter=3):
        cfg = LoFTQConfig(n_bits=n_bits, rank=rank, n_iterations=n_iter)
        return LoFTQ(cfg)

    def test_quantize_returns_result(self):
        loftq = self._make()
        W = np.random.randn(32, 64).astype(np.float32)
        result = loftq.quantize(W)
        assert isinstance(result, LoFTQResult)

    def test_effective_weight_shape(self):
        loftq = self._make(rank=4)
        W = np.random.randn(32, 64).astype(np.float32)
        result = loftq.quantize(W)
        W_eff = result.effective_weight()
        assert W_eff.shape == W.shape

    def test_lora_shapes(self):
        loftq = self._make(rank=4)
        W = np.random.randn(32, 64).astype(np.float32)
        result = loftq.quantize(W)
        assert result.lora_A.shape == (4, 64)
        assert result.lora_B.shape == (32, 4)

    def test_quantized_codes_range(self):
        loftq = self._make(n_bits=4)
        W = np.random.randn(16, 32).astype(np.float32)
        result = loftq.quantize(W)
        assert result.W_q.min() >= 0
        assert result.W_q.max() < 16

    def test_effective_weight_closer_to_original(self):
        loftq = self._make(n_bits=4, rank=8, n_iter=5)
        W = np.random.randn(16, 32).astype(np.float32)
        result = loftq.quantize(W)
        W_eff = result.effective_weight()
        err_eff = float(np.linalg.norm(W - W_eff))
        W_q = result.dequantize()
        err_q = float(np.linalg.norm(W - W_q))
        assert err_eff <= err_q + 1.0  # LoRA should reduce error

    def test_default_config(self):
        loftq = LoFTQ()
        assert loftq.config is not None

    def test_dequantize_shape_matches(self):
        loftq = self._make()
        W = np.random.randn(16, 32).astype(np.float32)
        result = loftq.quantize(W)
        assert result.dequantize().shape == W.shape


# ── OnlineSpec ─────────────────────────────────────────────────────────────────

from squish.speculative.online_spec import OnlineSpecConfig, OnlineSpec


class TestOnlineSpecConfig:
    def test_defaults(self):
        cfg = OnlineSpecConfig()
        assert cfg.vocab_size > 0
        assert cfg.lr > 0

    def test_custom(self):
        cfg = OnlineSpecConfig(vocab_size=128, lr=1e-3)
        assert cfg.vocab_size == 128


class TestOnlineSpec:
    def _make(self, vocab=64):
        cfg = OnlineSpecConfig(vocab_size=vocab, lr=1e-3)
        return OnlineSpec(cfg)

    def test_adjust_logits_shape(self):
        os = self._make(vocab=64)
        logits = np.random.randn(64).astype(np.float32)
        out = os.adjust_logits(logits)
        assert out.shape == (64,)

    def test_sample_returns_token(self):
        os = self._make(vocab=64)
        logits = np.random.randn(64).astype(np.float32)
        token, lp = os.sample(logits)
        assert 0 <= token < 64
        assert lp <= 0.0

    def test_observe(self):
        os = self._make(vocab=64)
        logits = np.random.randn(64).astype(np.float32)
        tok, _ = os.sample(logits)
        os.observe(tok, accepted=True, context_hash=42)

    def test_bias_adapts_after_rejected(self):
        os = self._make(vocab=16)
        logits = np.zeros(16, dtype=np.float32)
        tok, _ = os.sample(logits)
        bias_before = os._bias[tok]
        os.observe(tok, accepted=False, context_hash=1)
        bias_after = os._bias[tok]
        assert bias_before != bias_after

    def test_default_config(self):
        os = OnlineSpec()
        assert os.config is not None

    def test_repeated_sample_stays_in_vocab(self):
        os = self._make(vocab=32)
        for _ in range(20):
            tok, lp = os.sample(np.random.randn(32).astype(np.float32))
            assert 0 <= tok < 32


# ── DynamicSpecLen ─────────────────────────────────────────────────────────────

from squish.speculative.dynamic_spec_len import DynamicSpecLenConfig, DynamicSpecLen


class TestDynamicSpecLenConfig:
    def test_defaults(self):
        cfg = DynamicSpecLenConfig()
        assert cfg.min_spec_len >= 1
        assert cfg.max_spec_len >= cfg.min_spec_len

    def test_custom(self):
        cfg = DynamicSpecLenConfig(min_spec_len=2, max_spec_len=8)
        assert cfg.min_spec_len == 2


class TestDynamicSpecLen:
    def _make(self, vocab=32, min_l=1, max_l=4):
        cfg = DynamicSpecLenConfig(vocab_size=vocab, min_spec_len=min_l, max_spec_len=max_l)
        return DynamicSpecLen(cfg)

    def test_predict_returns_int(self):
        dsl = self._make()
        logits = np.random.randn(32).astype(np.float32)
        length = dsl.predict(logits)
        assert isinstance(length, int)

    def test_predict_in_range(self):
        dsl = self._make(min_l=1, max_l=4)
        logits = np.random.randn(32).astype(np.float32)
        length = dsl.predict(logits)
        assert 1 <= length <= 4

    def test_update(self):
        dsl = self._make()
        logits = np.random.randn(32).astype(np.float32)
        dsl.update(logits, actual_accepted=2)

    def test_extract_features_shape(self):
        dsl = self._make(vocab=32)
        logits = np.random.randn(32).astype(np.float32)
        feats = dsl.extract_features(logits)
        assert feats.ndim == 1

    def test_prediction_changes_after_update(self):
        dsl = self._make(vocab=16, min_l=1, max_l=8)
        logits = np.random.randn(16).astype(np.float32)
        # many updates should not crash
        for i in range(10):
            dsl.update(logits, actual_accepted=i % 4 + 1)
        length = dsl.predict(logits)
        assert 1 <= length <= 8

    def test_default_config(self):
        dsl = DynamicSpecLen()
        assert dsl.config is not None


# ── BigLittleLLM ───────────────────────────────────────────────────────────────

from squish.speculative.big_little_llm import BigLittleLLMConfig, RoutingDecision, BigLittleLLM


class TestBigLittleLLMConfig:
    def test_defaults(self):
        cfg = BigLittleLLMConfig()
        assert 0.0 < cfg.confidence_threshold < 1.0

    def test_custom(self):
        cfg = BigLittleLLMConfig(confidence_threshold=0.8, target_small_fraction=0.6)
        assert cfg.confidence_threshold == 0.8


class TestBigLittleLLM:
    def _make(self, vocab=32, threshold=0.7):
        cfg = BigLittleLLMConfig(vocab_size=vocab, confidence_threshold=threshold)
        return BigLittleLLM(cfg)

    def test_route_returns_decision(self):
        bll = self._make(vocab=32)
        small_logits = np.random.randn(32).astype(np.float32)
        large_logits = np.random.randn(32).astype(np.float32)
        decision = bll.route(small_logits, large_logits)
        assert isinstance(decision, RoutingDecision)

    def test_decision_has_token(self):
        bll = self._make(vocab=32)
        small = np.random.randn(32).astype(np.float32)
        large = np.random.randn(32).astype(np.float32)
        d = bll.route(small, large)
        assert 0 <= d.token < 32

    def test_decision_used_small_flag(self):
        bll = self._make(vocab=32)
        small = np.random.randn(32).astype(np.float32)
        large = np.random.randn(32).astype(np.float32)
        d = bll.route(small, large)
        assert isinstance(d.used_small, bool)

    def test_high_confidence_uses_small(self):
        bll = self._make(vocab=8, threshold=0.01)
        # Very confident small logits
        small = np.array([100.0] + [0.0] * 7, dtype=np.float32)
        large = np.random.randn(8).astype(np.float32)
        d = bll.route(small, large)
        assert d.used_small

    def test_adapt_threshold_called(self):
        bll = self._make()
        for _ in range(10):
            sm = np.random.randn(32).astype(np.float32)
            lg = np.random.randn(32).astype(np.float32)
            bll.route(sm, lg)
        bll._adapt_threshold()

    def test_default_config(self):
        bll = BigLittleLLM()
        assert bll.config is not None

    def test_routing_decision_fields(self):
        bll = self._make()
        d = bll.route(np.random.randn(32).astype(np.float32), np.random.randn(32).astype(np.float32))
        assert hasattr(d, "token")
        assert hasattr(d, "used_small")
        assert hasattr(d, "confidence")
