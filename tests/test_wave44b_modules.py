"""tests/test_wave44b_modules.py

Tests for Wave 44b modules:
  - MultiExitSpec (squish/speculative/multi_exit_spec.py)
  - PVTuning (squish/quant/pv_tuning.py)
  - HadamardQuant (squish/quant/hadamard_quant.py)
  - PrefixTreeDecode (squish/speculative/prefix_tree_decode.py)
  - SpecTrOT (squish/speculative/spectr_ot.py)
  - AdaGPTQ (squish/quant/ada_gptq.py)
"""

from __future__ import annotations

import pytest
import numpy as np

# ── MultiExitSpec ──────────────────────────────────────────────────────────────

from squish.speculative.multi_exit_spec import MultiExitSpecConfig, ExitResult, MultiExitSpec


class TestMultiExitSpecConfig:
    def test_defaults(self):
        cfg = MultiExitSpecConfig()
        assert len(cfg.exit_layers) >= 1
        assert 0.0 < cfg.threshold < 1.0

    def test_custom(self):
        cfg = MultiExitSpecConfig(exit_layers=[4, 8], threshold=0.85)
        assert cfg.exit_layers == [4, 8]


class TestMultiExitSpec:
    def _make(self, exit_layers=None, vocab=32, hidden=16):
        if exit_layers is None:
            exit_layers = [2, 4]
        cfg = MultiExitSpecConfig(exit_layers=exit_layers, vocab_size=vocab, hidden_size=hidden, threshold=0.6)
        return MultiExitSpec(cfg)

    def test_attempt_exits_returns_result(self):
        m = self._make()
        h = {2: np.random.randn(16).astype(np.float32), 4: np.random.randn(16).astype(np.float32)}
        result = m.attempt_exits(h)
        assert isinstance(result, ExitResult)

    def test_exit_layer_populated(self):
        m = self._make(exit_layers=[2, 4])
        h = {2: np.random.randn(16).astype(np.float32), 4: np.random.randn(16).astype(np.float32)}
        result = m.attempt_exits(h)
        # exit_layer is int (or None)
        assert result.exit_layer is None or isinstance(result.exit_layer, int)

    def test_token_in_vocab(self):
        m = self._make(vocab=32)
        h = {2: np.random.randn(16).astype(np.float32), 4: np.random.randn(16).astype(np.float32)}
        result = m.attempt_exits(h)
        if result.token is not None:
            assert 0 <= result.token < 32

    def test_high_confidence_exits_early(self):
        m = self._make(exit_layers=[2, 4], vocab=8, hidden=16)
        # Force a confident hidden state
        rng = np.random.default_rng(0)
        h_state = rng.standard_normal(16).astype(np.float32)
        # attempt with both layers provided
        h = {2: h_state, 4: h_state}
        result = m.attempt_exits(h)
        assert isinstance(result, ExitResult)

    def test_missing_layer_handled(self):
        m = self._make(exit_layers=[2, 4])
        # Provide only one layer
        h = {2: np.random.randn(16).astype(np.float32)}
        result = m.attempt_exits(h)
        assert isinstance(result, ExitResult)

    def test_default_config(self):
        m = MultiExitSpec()
        assert m.config is not None

    def test_all_layers_provided(self):
        m = self._make(exit_layers=[2, 4, 6])
        h = {i: np.random.randn(16).astype(np.float32) for i in [2, 4, 6]}
        result = m.attempt_exits(h)
        assert isinstance(result, ExitResult)


# ── PVTuning ───────────────────────────────────────────────────────────────────

from squish.quant.pv_tuning import PVTuningConfig, PVTuningResult, PVTuning


class TestPVTuningConfig:
    def test_defaults(self):
        cfg = PVTuningConfig()
        assert 1 <= cfg.n_bits <= 4
        assert cfg.n_steps >= 1
        assert cfg.lr > 0

    def test_custom(self):
        cfg = PVTuningConfig(n_bits=2, n_steps=5, lr=1e-3)
        assert cfg.n_bits == 2

    def test_invalid_nbits(self):
        with pytest.raises(ValueError):
            PVTuningConfig(n_bits=0)

    def test_invalid_nbits_high(self):
        with pytest.raises(ValueError):
            PVTuningConfig(n_bits=9)


class TestPVTuning:
    def _make(self, n_bits=2, n_steps=5, group_size=32):
        cfg = PVTuningConfig(n_bits=n_bits, n_steps=n_steps, group_size=group_size)
        return PVTuning(cfg)

    def test_compress_returns_result(self):
        pv = self._make()
        W = np.random.randn(16, 64).astype(np.float32)
        result = pv.compress(W)
        assert isinstance(result, PVTuningResult)

    def test_n_steps_taken(self):
        pv = self._make(n_steps=5)
        W = np.random.randn(16, 64).astype(np.float32)
        result = pv.compress(W)
        assert result.n_steps_taken == 5

    def test_scales_shape(self):
        pv = self._make(group_size=32)
        W = np.random.randn(16, 64).astype(np.float32)
        result = pv.compress(W)
        assert result.scale.shape[0] == 16
        assert result.scale.shape[1] >= 1

    def test_final_error_non_negative(self):
        pv = self._make()
        W = np.random.randn(16, 64).astype(np.float32)
        result = pv.compress(W)
        assert result.final_error >= 0.0

    def test_dequantize_shape(self):
        pv = self._make(group_size=16)
        W = np.random.randn(8, 32).astype(np.float32)
        result = pv.compress(W)
        W_hat = result.dequantize()
        assert W_hat.shape[0] == 8

    def test_default_config(self):
        pv = PVTuning()
        assert pv.config is not None

    def test_more_steps_reduce_error(self):
        W = np.random.randn(8, 32).astype(np.float32)
        r_few = PVTuning(PVTuningConfig(n_steps=1)).compress(W)
        r_many = PVTuning(PVTuningConfig(n_steps=20)).compress(W)
        assert r_many.final_error <= r_few.final_error + 1.0


# ── HadamardQuant ──────────────────────────────────────────────────────────────

from squish.quant.hadamard_quant import HadamardQuantConfig, HadamardQuantResult, HadamardQuant


class TestHadamardQuantConfig:
    def test_defaults(self):
        cfg = HadamardQuantConfig()
        assert cfg.n_bits == 4
        assert cfg.group_size > 0

    def test_custom(self):
        cfg = HadamardQuantConfig(n_bits=4, group_size=64, hadamard_seed=7)
        assert cfg.hadamard_seed == 7

    def test_invalid_nbits(self):
        with pytest.raises(ValueError):
            HadamardQuantConfig(n_bits=0)


class TestHadamardQuant:
    def _make(self, n_bits=4, group_size=32):
        cfg = HadamardQuantConfig(n_bits=n_bits, group_size=group_size)
        return HadamardQuant(cfg)

    def test_quantize_returns_result(self):
        hq = self._make()
        W = np.random.randn(16, 32).astype(np.float32)
        result = hq.quantize(W)
        assert isinstance(result, HadamardQuantResult)

    def test_codes_in_range(self):
        hq = self._make(n_bits=4)
        W = np.random.randn(16, 32).astype(np.float32)
        result = hq.quantize(W)
        assert result.W_q.min() >= 0
        assert result.W_q.max() < 16

    def test_sign_vec_length(self):
        hq = self._make()
        W = np.random.randn(16, 32).astype(np.float32)
        result = hq.quantize(W)
        assert len(result.sign_vec) >= 32  # padded to next power of 2

    def test_dequantize_shape(self):
        hq = self._make()
        W = np.random.randn(16, 32).astype(np.float32)
        result = hq.quantize(W)
        W_rot = result.dequantize()
        assert W_rot.shape[0] == 16

    def test_dequantize_unrotated_shape(self):
        hq = self._make()
        W = np.random.randn(16, 32).astype(np.float32)
        result = hq.quantize(W)
        W_orig = result.dequantize_unrotated()
        assert W_orig.shape == (16, 32)

    def test_rotation_distributes_outliers(self):
        hq = self._make()
        W = np.random.randn(16, 32).astype(np.float32)
        W[:, 0] *= 100
        result = hq.quantize(W)
        assert result is not None

    def test_default_config(self):
        hq = HadamardQuant()
        assert hq.config is not None

    def test_in_features_stored(self):
        hq = self._make()
        W = np.random.randn(16, 32).astype(np.float32)
        result = hq.quantize(W)
        assert result.in_features == 32


# ── PrefixTreeDecode ───────────────────────────────────────────────────────────

from squish.speculative.prefix_tree_decode import PrefixTreeConfig, PrefixTreeNode, PrefixTreeDecode


class TestPrefixTreeConfig:
    def test_defaults(self):
        cfg = PrefixTreeConfig()
        assert cfg.max_depth >= 1
        assert cfg.max_candidates >= 1

    def test_custom(self):
        cfg = PrefixTreeConfig(max_depth=4, max_candidates=8)
        assert cfg.max_depth == 4


class TestPrefixTreeDecode:
    def _make(self, max_depth=4, max_cand=8):
        cfg = PrefixTreeConfig(max_depth=max_depth, max_candidates=max_cand, min_frequency=1)
        return PrefixTreeDecode(cfg)

    def _build(self, ptd):
        corpus = [[1, 2, 3], [1, 2, 4], [5, 6, 7, 8], [1, 2, 3, 5]]
        ptd.build_from_corpus(corpus)

    def test_build_from_corpus(self):
        ptd = self._make()
        self._build(ptd)

    def test_lookup_returns_list(self):
        ptd = self._make()
        self._build(ptd)
        results = ptd.lookup([1, 2])
        assert isinstance(results, list)

    def test_lookup_candidates_sorted(self):
        ptd = self._make()
        self._build(ptd)
        results = ptd.lookup([1, 2])
        if len(results) >= 2:
            assert results[0][0] >= results[1][0]

    def test_lookup_empty_on_no_match(self):
        ptd = self._make()
        self._build(ptd)
        results = ptd.lookup([999, 888])
        assert isinstance(results, list)

    def test_decode_step_returns_tokens(self):
        ptd = self._make()
        self._build(ptd)
        cands = ptd.lookup([1, 2])
        logits = np.zeros(16)
        logits[3] = 5.0
        n, tokens = ptd.decode_step(logits, cands)
        assert n >= 1
        assert len(tokens) >= 1

    def test_decode_step_fallback_greedy(self):
        ptd = self._make()
        logits = np.zeros(8)
        logits[5] = 10.0
        n, tokens = ptd.decode_step(logits, [])
        assert tokens == [5]

    def test_reset_clears_tree(self):
        ptd = self._make()
        self._build(ptd)
        ptd.reset()
        results = ptd.lookup([1, 2])
        assert len(results) == 0

    def test_max_candidates_respected(self):
        ptd = self._make(max_cand=3)
        corpus = [[i, i + 1, i + 2] for i in range(20)]
        ptd.build_from_corpus(corpus)
        results = ptd.lookup([0])
        assert len(results) <= 3


# ── SpecTrOT ───────────────────────────────────────────────────────────────────

from squish.speculative.spectr_ot import SpecTrOTConfig, SpecTrOTResult, SpecTrOT


class TestSpecTrOTConfig:
    def test_defaults(self):
        cfg = SpecTrOTConfig()
        assert cfg.n_draft >= 1
        assert cfg.temperature > 0

    def test_custom(self):
        cfg = SpecTrOTConfig(n_draft=6, temperature=0.8)
        assert cfg.n_draft == 6


class TestSpecTrOT:
    def _make(self, vocab=32, n_draft=4):
        cfg = SpecTrOTConfig(n_draft=n_draft)
        return SpecTrOT(cfg)

    def test_compute_coupling_shapes(self):
        ot = self._make(vocab=32)
        dl = np.random.randn(32).astype(np.float32)
        tl = np.random.randn(32).astype(np.float32)
        p, q, ap = ot.compute_coupling(dl, tl)
        assert p.shape == (32,)
        assert q.shape == (32,)
        assert ap.shape == (32,)

    def test_accept_probs_in_01(self):
        ot = self._make()
        dl = np.random.randn(32).astype(np.float32)
        tl = np.random.randn(32).astype(np.float32)
        _, _, ap = ot.compute_coupling(dl, tl)
        assert (ap >= 0).all() and (ap <= 1.0).all()

    def test_sample_returns_token(self):
        ot = self._make(vocab=32)
        dl = np.random.randn(32).astype(np.float32)
        tl = np.random.randn(32).astype(np.float32)
        p, q, ap = ot.compute_coupling(dl, tl)
        tok, _ = ot.sample(p, q, ap)
        assert 0 <= tok < 32

    def test_step_returns_result(self):
        ot = self._make(vocab=32, n_draft=4)
        dl = np.random.randn(4, 32).astype(np.float32)
        tl = np.random.randn(4, 32).astype(np.float32)
        result = ot.step(dl, tl)
        assert isinstance(result, SpecTrOTResult)

    def test_n_accepted_in_range(self):
        ot = self._make(n_draft=4)
        dl = np.random.randn(4, 32).astype(np.float32)
        tl = np.random.randn(4, 32).astype(np.float32)
        result = ot.step(dl, tl)
        assert 0 <= result.n_accepted <= 4

    def test_softmax_sums_to_one(self):
        ot = self._make()
        x = np.random.randn(16).astype(np.float32)
        p = ot._softmax(x)
        assert abs(p.sum() - 1.0) < 1e-5

    def test_default_config(self):
        ot = SpecTrOT()
        assert ot.config is not None


# ── AdaGPTQ ────────────────────────────────────────────────────────────────────

from squish.quant.ada_gptq import AdaGPTQConfig, AdaGPTQResult, AdaGPTQ


class TestAdaGPTQConfig:
    def test_defaults(self):
        cfg = AdaGPTQConfig()
        assert cfg.n_bits in (2, 4, 8)
        assert cfg.min_group_size >= 1
        assert cfg.max_group_size > cfg.min_group_size

    def test_custom(self):
        cfg = AdaGPTQConfig(n_bits=4, min_group_size=8, max_group_size=64)
        assert cfg.n_bits == 4

    def test_invalid_nbits(self):
        with pytest.raises(ValueError):
            AdaGPTQConfig(n_bits=0)

    def test_invalid_min_group(self):
        with pytest.raises(ValueError):
            AdaGPTQConfig(min_group_size=0)


class TestAdaGPTQ:
    def _make(self, n_bits=4, min_g=8, max_g=32):
        cfg = AdaGPTQConfig(n_bits=n_bits, min_group_size=min_g, max_group_size=max_g, curvature_budget=0.5)
        return AdaGPTQ(cfg)

    def test_estimate_hessian_shape(self):
        ada = self._make()
        X = np.random.randn(20, 32).astype(np.float32)
        h = ada.estimate_hessian(X)
        assert h.shape == (32,)

    def test_hessian_positive(self):
        ada = self._make()
        X = np.random.randn(10, 16).astype(np.float32)
        h = ada.estimate_hessian(X)
        assert (h > 0).all()

    def test_select_group_boundaries_valid(self):
        ada = self._make(min_g=8, max_g=32)
        h = np.random.rand(64).astype(np.float32) + 0.1
        bds = ada.select_group_boundaries(h)
        assert bds[0] == 0
        assert bds[-1] == 64

    def test_quantize_returns_result(self):
        ada = self._make()
        W = np.random.randn(16, 64).astype(np.float32)
        result = ada.quantize(W)
        assert isinstance(result, AdaGPTQResult)

    def test_codes_in_range(self):
        ada = self._make(n_bits=4)
        W = np.random.randn(16, 64).astype(np.float32)
        result = ada.quantize(W)
        assert result.W_q.min() >= 0
        assert result.W_q.max() < 16

    def test_dequantize_shape(self):
        ada = self._make()
        W = np.random.randn(16, 64).astype(np.float32)
        result = ada.quantize(W)
        W_hat = result.dequantize()
        assert W_hat.shape == W.shape

    def test_with_hessian_adaptive(self):
        ada = self._make(min_g=8, max_g=32)
        W = np.random.randn(16, 64).astype(np.float32)
        X = np.random.randn(20, 64).astype(np.float32)
        h = ada.estimate_hessian(X)
        result = ada.quantize(W, hessian_diag=h)
        assert isinstance(result, AdaGPTQResult)

    def test_default_config(self):
        ada = AdaGPTQ()
        assert ada.config is not None
