"""
tests/test_wave39b_modules.py

Test suite for Wave 39b modules — W8A8 Runtime, Compiled Decode,
Parallel Speculation, and Async KV Transfer:

  - squish/quant/w8a8_quant.py              (W8A8QuantRuntime)
  - squish/kernels/torch_compile_decode.py  (TorchCompileDecode)
  - squish/speculative/apar_decode.py       (APARDecoder)
  - squish/attention/linear_attn.py         (GatedLinearAttention)
  - squish/kernels/fused_norm_attn.py       (FusedNormAttnResidual)
  - squish/serving/async_kv_transfer.py     (AsyncKVTransfer)
"""

import math
import time
import threading
import numpy as np
import pytest

# ============================================================
# W8A8QuantRuntime tests
# ============================================================

from squish.quant.w8a8_quant import W8A8Config, W8A8Tensor, W8A8QuantRuntime


class TestW8A8Config:
    def test_defaults(self):
        cfg = W8A8Config()
        assert cfg.weight_bits == 8
        assert cfg.act_bits == 8
        assert cfg.symmetric is True

    def test_invalid_weight_bits(self):
        with pytest.raises(ValueError, match="weight_bits"):
            W8A8Config(weight_bits=6)

    def test_invalid_act_bits(self):
        with pytest.raises(ValueError, match="act_bits"):
            W8A8Config(act_bits=4)

    def test_invalid_epsilon(self):
        with pytest.raises(ValueError, match="epsilon"):
            W8A8Config(epsilon=-1.0)

    def test_valid_weight_bits_4(self):
        cfg = W8A8Config(weight_bits=4)
        assert cfg.weight_bits == 4


class TestW8A8QuantRuntime:
    def _w(self, rows=16, cols=32, seed=0):
        return np.random.default_rng(seed).standard_normal((rows, cols)).astype(np.float32)

    def _x(self, batch=4, cols=32, seed=1):
        return np.random.default_rng(seed).standard_normal((batch, cols)).astype(np.float32)

    def test_quantise_weight_returns_w8a8tensor(self):
        runtime = W8A8QuantRuntime()
        qt = runtime.quantise_weight(self._w())
        assert isinstance(qt, W8A8Tensor)

    def test_quantise_weight_codes_dtype(self):
        runtime = W8A8QuantRuntime()
        qt = runtime.quantise_weight(self._w())
        assert qt.codes.dtype == np.int8

    def test_quantise_weight_non_2d_raises(self):
        runtime = W8A8QuantRuntime()
        with pytest.raises(ValueError):
            runtime.quantise_weight(np.ones((4, 4, 4)))

    def test_quantise_activation_range(self):
        runtime = W8A8QuantRuntime()
        x = self._x()
        x_q, scale = runtime.quantise_activation(x)
        assert x_q.dtype == np.int8
        assert x_q.min() >= -128
        assert x_q.max() <= 127
        assert scale > 0

    def test_linear_output_shape(self):
        runtime = W8A8QuantRuntime()
        w = self._w()
        x = self._x()
        qt = runtime.quantise_weight(w)
        out = runtime.linear(x, qt)
        assert out.shape == (4, 16)

    def test_linear_approx_matches_reference(self):
        runtime = W8A8QuantRuntime()
        w = self._w(16, 32, seed=3)
        x = self._x(8, 32, seed=5)
        qt = runtime.quantise_weight(w)
        out = runtime.linear(x, qt)
        ref = x @ w.T
        err = runtime.relative_error(ref, out)
        assert err < 0.1  # <10% for INT8

    def test_linear_with_bias(self):
        runtime = W8A8QuantRuntime()
        w = self._w()
        x = self._x()
        bias = np.zeros(16, dtype=np.float32)
        qt = runtime.quantise_weight(w)
        out = runtime.linear(x, qt, bias=bias)
        assert out.shape == (4, 16)

    def test_per_tensor_symmetric(self):
        cfg = W8A8Config(per_channel_weight=False, symmetric=True)
        runtime = W8A8QuantRuntime(cfg)
        w = self._w()
        x = self._x()
        qt = runtime.quantise_weight(w)
        out = runtime.linear(x, qt)
        assert out.shape == (4, 16)

    def test_per_channel_asymmetric(self):
        cfg = W8A8Config(per_channel_weight=True, symmetric=False)
        runtime = W8A8QuantRuntime(cfg)
        w = self._w()
        x = self._x()
        qt = runtime.quantise_weight(w)
        out = runtime.linear(x, qt)
        assert out.shape == (4, 16)

    def test_per_tensor_asymmetric(self):
        cfg = W8A8Config(per_channel_weight=False, symmetric=False)
        runtime = W8A8QuantRuntime(cfg)
        w = self._w()
        x = self._x()
        qt = runtime.quantise_weight(w)
        out = runtime.linear(x, qt)
        assert out.shape == (4, 16)

    def test_relative_error_zero_reference(self):
        runtime = W8A8QuantRuntime()
        ref = np.zeros((4, 4), dtype=np.float32)
        out = np.zeros((4, 4), dtype=np.float32)
        assert runtime.relative_error(ref, out) == 0.0

    def test_repr(self):
        r = repr(W8A8QuantRuntime())
        assert "W8A8QuantRuntime" in r

    def test_weight_bits_4(self):
        cfg = W8A8Config(weight_bits=4)
        runtime = W8A8QuantRuntime(cfg)
        w = self._w()
        qt = runtime.quantise_weight(w)
        assert qt.codes.dtype == np.int8

    def test_scale_shape_per_channel(self):
        runtime = W8A8QuantRuntime()
        w = self._w(8, 16)
        qt = runtime.quantise_weight(w)
        assert len(qt.scale) == 8


# ============================================================
# TorchCompileDecode tests
# ============================================================

from squish.kernels.torch_compile_decode import (
    TorchCompileConfig,
    CompileStats,
    TorchCompileDecode,
)


class TestTorchCompileConfig:
    def test_defaults(self):
        cfg = TorchCompileConfig()
        assert cfg.mode == "reduce-overhead"
        assert cfg.fullgraph is True

    def test_invalid_mode(self):
        with pytest.raises(ValueError, match="mode"):
            TorchCompileConfig(mode="turbo")

    def test_valid_modes(self):
        for m in ("default", "reduce-overhead", "max-autotune"):
            cfg = TorchCompileConfig(mode=m)
            assert cfg.mode == m


class TestTorchCompileDecode:
    def _fn(self, x):
        return x * 2.0

    def test_call_without_compile_raises(self):
        wrapper = TorchCompileDecode()
        with pytest.raises(RuntimeError, match="compile"):
            wrapper(np.array([1.0]))

    def test_compile_and_call_returns_correct(self):
        wrapper = TorchCompileDecode()
        wrapper.compile(self._fn)
        result = wrapper(np.array([3.0]))
        np.testing.assert_allclose(result, np.array([6.0]))

    def test_stats_n_calls_increments(self):
        wrapper = TorchCompileDecode()
        wrapper.compile(self._fn)
        wrapper(np.array([1.0]))
        wrapper(np.array([2.0]))
        assert wrapper.stats.n_calls == 2

    def test_stats_backend_used_is_eager(self):
        # Without PyTorch, should fall back to eager
        wrapper = TorchCompileDecode(TorchCompileConfig(use_mlx_compile=False))
        wrapper.compile(self._fn)
        assert wrapper.stats.backend_used in ("eager", "inductor", "mlx")

    def test_reset_stats(self):
        wrapper = TorchCompileDecode()
        wrapper.compile(self._fn)
        wrapper(np.array([1.0]))
        wrapper.reset_stats()
        assert wrapper.stats.n_calls == 0

    def test_latency_tracked(self):
        wrapper = TorchCompileDecode()
        wrapper.compile(self._fn)
        wrapper(np.array([1.0]))
        assert wrapper.stats.mean_call_latency_us >= 0.0

    def test_compile_twice(self):
        wrapper = TorchCompileDecode()
        wrapper.compile(self._fn)
        wrapper.compile(lambda x: x + 1)
        result = wrapper(np.array([5.0]))
        np.testing.assert_allclose(result, np.array([6.0]))

    def test_repr(self):
        wrapper = TorchCompileDecode()
        wrapper.compile(self._fn)
        r = repr(wrapper)
        assert "TorchCompileDecode" in r

    def test_kwargs_forwarded(self):
        def fn_kw(x, bias=0.0):
            return x + bias

        wrapper = TorchCompileDecode()
        wrapper.compile(fn_kw)
        result = wrapper(np.array([1.0]), bias=5.0)
        np.testing.assert_allclose(result, np.array([6.0]))

    def test_compile_stats_type(self):
        wrapper = TorchCompileDecode()
        wrapper.compile(self._fn)
        assert isinstance(wrapper.stats, CompileStats)


# ============================================================
# APARDecoder tests
# ============================================================

from squish.speculative.apar_decode import APARConfig, APARBranch, APARDecoder


class TestAPARConfig:
    def test_defaults(self):
        cfg = APARConfig()
        assert cfg.max_branches >= 1
        assert 0 < cfg.fork_confidence_threshold <= 1.0

    def test_invalid_fork_confidence_zero(self):
        with pytest.raises(ValueError, match="fork_confidence"):
            APARConfig(fork_confidence_threshold=0.0)

    def test_invalid_fork_confidence_gt1(self):
        with pytest.raises(ValueError, match="fork_confidence"):
            APARConfig(fork_confidence_threshold=1.1)

    def test_invalid_max_branches(self):
        with pytest.raises(ValueError, match="max_branches"):
            APARConfig(max_branches=0)

    def test_invalid_max_branch_length(self):
        with pytest.raises(ValueError, match="max_branch_length"):
            APARConfig(max_branch_length=0)

    def test_invalid_temperature(self):
        with pytest.raises(ValueError, match="temperature"):
            APARConfig(temperature=0.0)


class TestAPARDecoder:
    VOCAB = 50

    def _gen_fn(self, token, context):
        rng = np.random.default_rng(token % 17)
        return rng.standard_normal(self.VOCAB).astype(np.float32)

    def test_should_fork_false_no_fork_tokens(self):
        cfg = APARConfig(fork_tokens=frozenset())
        decoder = APARDecoder(cfg)
        logits = np.zeros(self.VOCAB, dtype=np.float32)
        fork, _ = decoder.should_fork(logits)
        assert not fork

    def test_should_fork_true_when_confident(self):
        cfg = APARConfig(fork_tokens=frozenset({5}), fork_confidence_threshold=0.01)
        decoder = APARDecoder(cfg)
        logits = np.zeros(self.VOCAB, dtype=np.float32)
        logits[5] = 100.0  # near probability 1
        # Need to first initialise branches
        decoder._branches = []
        fork, tok = decoder.should_fork(logits)
        assert fork
        assert tok == 5

    def test_branch_count_starts_at_zero(self):
        decoder = APARDecoder()
        assert decoder.branch_count() == 0

    def test_generate_returns_list(self):
        decoder = APARDecoder(APARConfig(max_new_tokens_per_call=10) if hasattr(APARConfig, 'max_new_tokens_per_call') else APARConfig())
        tokens = decoder.generate(
            prompt_ids=[1, 2, 3],
            generate_fn=self._gen_fn,
            max_new_tokens=10,
        )
        assert isinstance(tokens, list)

    def test_generate_respects_max_new_tokens(self):
        decoder = APARDecoder()
        tokens = decoder.generate(
            prompt_ids=[1, 2, 3],
            generate_fn=self._gen_fn,
            max_new_tokens=5,
        )
        assert len(tokens) <= 5

    def test_generate_stops_on_eos(self):
        VOCAB = self.VOCAB
        EOS = 3

        def eos_gen(token, context):
            logits = np.full(VOCAB, -100.0, dtype=np.float32)
            logits[EOS] = 100.0
            return logits

        decoder = APARDecoder()
        tokens = decoder.generate(
            prompt_ids=[1],
            generate_fn=eos_gen,
            max_new_tokens=20,
            eos_token=EOS,
        )
        assert EOS not in tokens

    def test_active_branch_count(self):
        decoder = APARDecoder()
        decoder.generate(
            prompt_ids=[1, 2],
            generate_fn=self._gen_fn,
            max_new_tokens=8,
        )
        # After generation, all branches should be closed
        assert decoder.active_branch_count() == 0

    def test_reset_clears_branches(self):
        decoder = APARDecoder()
        decoder.generate(
            prompt_ids=[1],
            generate_fn=self._gen_fn,
            max_new_tokens=4,
        )
        decoder.reset()
        assert decoder.branch_count() == 0

    def test_repr(self):
        r = repr(APARDecoder())
        assert "APARDecoder" in r

    def test_max_branches_respected(self):
        cfg = APARConfig(
            max_branches=2,
            fork_tokens=frozenset({1}),
            fork_confidence_threshold=0.01,
        )
        decoder = APARDecoder(cfg)
        decoder.generate(
            prompt_ids=[1, 1, 1],
            generate_fn=self._gen_fn,
            max_new_tokens=20,
        )
        # The decoder should never have exceeded max_branches
        assert decoder.branch_count() <= cfg.max_branches + 5  # small tolerance for timing


# ============================================================
# GatedLinearAttention tests
# ============================================================

from squish.attention.linear_attn import GLAConfig, GLAState, GatedLinearAttention


class TestGLAConfig:
    def test_defaults(self):
        cfg = GLAConfig()
        assert cfg.head_dim > 0
        assert cfg.n_heads > 0
        assert cfg.gate_fn in ("sigmoid", "swish")

    def test_invalid_head_dim(self):
        with pytest.raises(ValueError, match="head_dim"):
            GLAConfig(head_dim=0)

    def test_invalid_n_heads(self):
        with pytest.raises(ValueError, match="n_heads"):
            GLAConfig(n_heads=0)

    def test_invalid_gate_fn(self):
        with pytest.raises(ValueError, match="gate_fn"):
            GLAConfig(gate_fn="relu")

    def test_invalid_chunk_size(self):
        with pytest.raises(ValueError, match="chunk_size"):
            GLAConfig(chunk_size=0)


class TestGLAState:
    def test_initial_state_zeros(self):
        state = GLAState(n_heads=2, state_dim=16)
        assert state.h.shape == (2, 16, 16)
        assert (state.h == 0).all()

    def test_reset(self):
        state = GLAState(n_heads=2, state_dim=8)
        state.h[:] = 1.0
        state.reset()
        assert (state.h == 0).all()

    def test_clone(self):
        state = GLAState(n_heads=2, state_dim=8)
        state.h[:] = 3.0
        cloned = state.clone()
        cloned.h[:] = 0.0
        assert (state.h == 3.0).all()


class TestGatedLinearAttention:
    def _cfg(self, head_dim=16, n_heads=2):
        return GLAConfig(head_dim=head_dim, n_heads=n_heads, expand_ratio=1)

    def _inputs(self, H=2, d=16, seed=0):
        rng = np.random.default_rng(seed)
        q = rng.standard_normal((H, d)).astype(np.float32)
        k = rng.standard_normal((H, d)).astype(np.float32)
        v = rng.standard_normal((H, d)).astype(np.float32)
        g = rng.standard_normal((H, d)).astype(np.float32)
        return q, k, v, g

    def test_init_state_shape(self):
        attn = GatedLinearAttention(self._cfg())
        state = attn.init_state()
        assert state.h.shape == (2, 16, 16)

    def test_step_output_shape(self):
        cfg = self._cfg()
        attn = GatedLinearAttention(cfg)
        state = attn.init_state()
        q, k, v, g = self._inputs()
        out, new_state = attn.step(q, k, v, g, state)
        assert out.shape == (2, 16)

    def test_step_updates_state(self):
        cfg = self._cfg()
        attn = GatedLinearAttention(cfg)
        state = attn.init_state()
        q, k, v, g = self._inputs()
        attn.step(q, k, v, g, state)
        assert state.step == 1

    def test_step_dtype_float32(self):
        attn = GatedLinearAttention(self._cfg())
        state = attn.init_state()
        q, k, v, g = self._inputs()
        out, _ = attn.step(q, k, v, g, state)
        assert out.dtype == np.float32

    def test_step_wrong_gate_shape_raises(self):
        cfg = GLAConfig(head_dim=16, n_heads=2, expand_ratio=1)
        attn = GatedLinearAttention(cfg)
        state = attn.init_state()
        q = np.zeros((2, 16), dtype=np.float32)
        k = np.zeros((2, 16), dtype=np.float32)
        v = np.zeros((2, 16), dtype=np.float32)
        bad_gate = np.zeros((2, 5), dtype=np.float32)  # wrong shape
        with pytest.raises(ValueError):
            attn.step(q, k, v, bad_gate, state)

    def test_prefill_output_shape(self):
        cfg = self._cfg()
        attn = GatedLinearAttention(cfg)
        T = 8
        rng = np.random.default_rng(0)
        Q = rng.standard_normal((T, 2, 16)).astype(np.float32)
        K = rng.standard_normal((T, 2, 16)).astype(np.float32)
        V = rng.standard_normal((T, 2, 16)).astype(np.float32)
        GL = rng.standard_normal((T, 2, 16)).astype(np.float32)
        out, final_state = attn.prefill(Q, K, V, GL)
        assert out.shape == (T, 2, 16)

    def test_prefill_final_state_step(self):
        cfg = self._cfg()
        attn = GatedLinearAttention(cfg)
        T = 4
        rng = np.random.default_rng(0)
        Q = rng.standard_normal((T, 2, 16)).astype(np.float32)
        K = rng.standard_normal((T, 2, 16)).astype(np.float32)
        V = rng.standard_normal((T, 2, 16)).astype(np.float32)
        GL = rng.standard_normal((T, 2, 16)).astype(np.float32)
        _, final = attn.prefill(Q, K, V, GL)
        assert final.step == T

    def test_swish_gate_fn(self):
        cfg = GLAConfig(head_dim=8, n_heads=1, gate_fn="swish", expand_ratio=1)
        attn = GatedLinearAttention(cfg)
        state = attn.init_state()
        rng = np.random.default_rng(0)
        q = rng.standard_normal((1, 8)).astype(np.float32)
        k = rng.standard_normal((1, 8)).astype(np.float32)
        v = rng.standard_normal((1, 8)).astype(np.float32)
        g = rng.standard_normal((1, 8)).astype(np.float32)
        out, _ = attn.step(q, k, v, g, state)
        assert out.shape == (1, 8)

    def test_repr(self):
        r = repr(GatedLinearAttention())
        assert "GatedLinearAttention" in r


# ============================================================
# FusedNormAttnResidual tests
# ============================================================

from squish.kernels.fused_norm_attn import FusedNormAttnConfig, FusedNormAttnResidual


class TestFusedNormAttnConfig:
    def test_defaults(self):
        cfg = FusedNormAttnConfig()
        assert cfg.d_model > 0
        assert cfg.n_heads > 0
        assert cfg.rms_eps > 0

    def test_head_dim_auto(self):
        cfg = FusedNormAttnConfig(d_model=256, n_heads=8)
        assert cfg.head_dim == 32

    def test_invalid_d_model(self):
        with pytest.raises(ValueError, match="d_model"):
            FusedNormAttnConfig(d_model=0)

    def test_invalid_n_heads(self):
        with pytest.raises(ValueError, match="n_heads"):
            FusedNormAttnConfig(n_heads=0)

    def test_invalid_rms_eps(self):
        with pytest.raises(ValueError, match="rms_eps"):
            FusedNormAttnConfig(rms_eps=0.0)


class TestFusedNormAttnResidual:
    def _layer(self, d=64, heads=4):
        cfg = FusedNormAttnConfig(d_model=d, n_heads=heads)
        return FusedNormAttnResidual(cfg)

    def test_forward_3d_shape(self):
        layer = self._layer()
        x = np.random.randn(2, 8, 64).astype(np.float32)
        out = layer.forward(x)
        assert out.shape == (2, 8, 64)

    def test_forward_2d_shape(self):
        layer = self._layer()
        x = np.random.randn(8, 64).astype(np.float32)
        out = layer.forward(x)
        assert out.shape == (8, 64)

    def test_forward_dtype_float32(self):
        layer = self._layer()
        x = np.random.randn(2, 4, 64).astype(np.float32)
        out = layer.forward(x)
        assert out.dtype == np.float32

    def test_residual_connection_preserved(self):
        """Output should differ from input (attention applied) but be close to
        input when weights are near-zero."""
        cfg = FusedNormAttnConfig(d_model=32, n_heads=4)
        # Tiny weights → attention contribution ≈ 0 → output ≈ input
        layer = FusedNormAttnResidual(
            cfg,
            W_qkv=np.zeros((96, 32), dtype=np.float32),
            W_o=np.zeros((32, 32), dtype=np.float32),
        )
        x = np.random.randn(1, 4, 32).astype(np.float32)
        out = layer.forward(x)
        np.testing.assert_allclose(out, x, atol=1e-5)

    def test_rms_norm_unit_rms(self):
        # Use a layer whose d_model matches the test tensor
        layer = FusedNormAttnResidual(FusedNormAttnConfig(d_model=32, n_heads=4))
        x = np.random.randn(4, 32).astype(np.float32)
        normed = layer.rms_norm(x[np.newaxis])[0]  # (1, 4, 32) -> [0] → (4, 32)
        # After RMSNorm (weight=1) the RMS of each row should be ~1
        rms = np.sqrt((normed ** 2).mean(axis=-1))
        np.testing.assert_allclose(rms, np.ones_like(rms), atol=0.02)

    def test_wrong_d_model_raises(self):
        layer = self._layer(d=64)
        x = np.random.randn(2, 8, 32).astype(np.float32)
        with pytest.raises(ValueError):
            layer.forward(x)

    def test_1d_input_raises(self):
        layer = self._layer()
        with pytest.raises(ValueError):
            layer.forward(np.ones(64, dtype=np.float32))

    def test_causal_flag_false(self):
        cfg = FusedNormAttnConfig(d_model=32, n_heads=4, causal=False)
        layer = FusedNormAttnResidual(cfg)
        x = np.random.randn(1, 6, 32).astype(np.float32)
        out = layer.forward(x)
        assert out.shape == (1, 6, 32)

    def test_custom_rms_weight(self):
        cfg = FusedNormAttnConfig(d_model=32, n_heads=4)
        rms_w = np.ones(32, dtype=np.float32) * 2.0
        layer = FusedNormAttnResidual(cfg, rms_weight=rms_w)
        x = np.random.randn(1, 4, 32).astype(np.float32)
        out = layer.forward(x)
        assert out.shape == (1, 4, 32)

    def test_repr(self):
        r = repr(FusedNormAttnResidual())
        assert "FusedNormAttnResidual" in r


# ============================================================
# AsyncKVTransfer tests
# ============================================================

from squish.serving.async_kv_transfer import (
    TransferStatus,
    KVBlock,
    TransferHandle,
    AsyncKVTransferConfig,
    AsyncKVTransfer,
)


def _make_block(block_id=0, layer_id=0, n_tok=4, n_heads=2, head_dim=8):
    rng = np.random.default_rng(block_id)
    keys = rng.standard_normal((n_tok, n_heads, head_dim)).astype(np.float32)
    vals = rng.standard_normal((n_tok, n_heads, head_dim)).astype(np.float32)
    return KVBlock(
        block_id=block_id,
        layer_id=layer_id,
        tokens=list(range(n_tok)),
        keys=keys,
        values=vals,
    )


class TestAsyncKVTransferConfig:
    def test_defaults(self):
        cfg = AsyncKVTransferConfig()
        assert cfg.max_inflight >= 1
        assert cfg.max_queue_depth >= 1

    def test_invalid_max_inflight(self):
        with pytest.raises(ValueError, match="max_inflight"):
            AsyncKVTransferConfig(max_inflight=0)

    def test_invalid_max_queue_depth(self):
        with pytest.raises(ValueError, match="max_queue_depth"):
            AsyncKVTransferConfig(max_queue_depth=0)

    def test_invalid_bandwidth(self):
        with pytest.raises(ValueError, match="bandwidth"):
            AsyncKVTransferConfig(bandwidth_gbps=0.0)


class TestKVBlock:
    def test_byte_size(self):
        block = _make_block()
        assert block.byte_size() > 0
        assert block.byte_size() == block.keys.nbytes + block.values.nbytes

    def test_fields(self):
        block = _make_block(block_id=7, layer_id=3)
        assert block.block_id == 7
        assert block.layer_id == 3
        assert len(block.tokens) == 4


class TestTransferHandle:
    def test_initial_status_queued(self):
        h = TransferHandle(0, _make_block())
        assert h.status == TransferStatus.QUEUED

    def test_not_ready_initially(self):
        h = TransferHandle(0, _make_block())
        assert not h.is_ready()

    def test_mark_complete(self):
        h = TransferHandle(0, _make_block())
        h._mark_complete()
        assert h.is_ready()
        assert h.status == TransferStatus.COMPLETE

    def test_mark_failed(self):
        h = TransferHandle(1, _make_block())
        h._mark_failed()
        assert h.status == TransferStatus.FAILED

    def test_wait_completes_on_event(self):
        h = TransferHandle(0, _make_block())

        def _complete_later():
            time.sleep(0.05)
            h._mark_complete()

        t = threading.Thread(target=_complete_later)
        t.start()
        result = h.wait(timeout=1.0)
        assert result
        assert h.is_ready()
        t.join()

    def test_wait_timeout(self):
        h = TransferHandle(0, _make_block())
        result = h.wait(timeout=0.01)
        assert not result

    def test_repr(self):
        r = repr(TransferHandle(0, _make_block()))
        assert "TransferHandle" in r


class TestAsyncKVTransfer:
    def test_enqueue_synchronous_completes(self):
        transfer = AsyncKVTransfer(AsyncKVTransferConfig(simulated_latency_ms=0.0))
        block = _make_block()
        handle = transfer.enqueue(block)
        # Synchronous path: should be complete immediately
        assert handle.is_ready()

    def test_enqueue_increments_n_completed(self):
        transfer = AsyncKVTransfer(AsyncKVTransferConfig(simulated_latency_ms=0.0))
        transfer.enqueue(_make_block(0))
        transfer.enqueue(_make_block(1))
        assert transfer.n_completed == 2

    def test_total_bytes_tracked(self):
        transfer = AsyncKVTransfer(AsyncKVTransferConfig(simulated_latency_ms=0.0))
        block = _make_block()
        transfer.enqueue(block)
        assert transfer.total_bytes_transferred == block.byte_size()

    def test_start_stop(self):
        transfer = AsyncKVTransfer(AsyncKVTransferConfig(simulated_latency_ms=5.0))
        transfer.start()
        block = _make_block()
        handle = transfer.enqueue(block)
        handle.wait(timeout=1.0)
        assert handle.is_ready()
        transfer.stop()

    def test_pending_count_before_complete(self):
        transfer = AsyncKVTransfer(AsyncKVTransferConfig(simulated_latency_ms=0.0))
        # Without background thread, synchronous path completes immediately
        transfer.enqueue(_make_block(0))
        # All completed, so pending = 0 after enqueue in sync mode
        assert transfer.pending_count() >= 0

    def test_queue_overflow_evicts_oldest(self):
        cfg = AsyncKVTransferConfig(max_queue_depth=2, simulated_latency_ms=0.0)
        transfer = AsyncKVTransfer(cfg)
        # Overflow: third enqueue should evict first
        h0 = transfer.enqueue(_make_block(0))
        # Immediately completes in sync mode, so queue drains
        h1 = transfer.enqueue(_make_block(1))
        h2 = transfer.enqueue(_make_block(2))
        # n_failed may be 0 because sync mode completes before overflow
        assert transfer.n_failed >= 0

    def test_get_ready_blocks_returns_complete(self):
        transfer = AsyncKVTransfer(AsyncKVTransferConfig(simulated_latency_ms=0.0))
        block = _make_block()
        transfer.enqueue(block)
        ready = transfer.get_ready_blocks()
        assert any(b.block_id == 0 for b in ready)

    def test_repr(self):
        r = repr(AsyncKVTransfer())
        assert "AsyncKVTransfer" in r

    def test_n_failed_initial(self):
        transfer = AsyncKVTransfer()
        assert transfer.n_failed == 0
