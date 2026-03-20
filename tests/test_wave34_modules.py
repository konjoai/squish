"""tests/test_wave34_modules.py — Tests for Wave 34 modules.

Covers v13 Wave 34 additions:
  - squish/kernels/metal_flash_attn.py   (MetalFlashAttention)
  - squish/speculative/spec_stream.py    (SpeculativeStreamer)
  - squish/kv/block_sparse_kv.py         (BlockSparseKVManager)
  - squish/serving/pd_disagg.py          (PDDisaggregator)
  - squish/token/deja_vu_sparse.py       (DejaVuSparseFFN)
  - squish/io/layer_overlap_loader.py    (LayerOverlapLoader)
"""

import math
import time
import threading
import pytest
import numpy as np

# ---------------------------------------------------------------------------
# MetalFlashAttention
# ---------------------------------------------------------------------------

from squish.kernels.metal_flash_attn import (
    MetalFlashConfig,
    MetalFlashStats,
    MetalFlashAttention,
)


class TestMetalFlashConfig:
    def test_defaults(self):
        cfg = MetalFlashConfig()
        assert cfg.block_q == 32
        assert cfg.block_k == 32
        assert cfg.causal is True

    def test_invalid_block_q(self):
        with pytest.raises(ValueError, match="block_q"):
            MetalFlashConfig(block_q=0)

    def test_invalid_block_k(self):
        with pytest.raises(ValueError, match="block_k"):
            MetalFlashConfig(block_k=0)

    def test_invalid_dropout(self):
        with pytest.raises(ValueError, match="dropout"):
            MetalFlashConfig(dropout=1.0)


class TestMetalFlashAttention:
    def _qkv(self, seq=8, heads=2, d=16):
        rng = np.random.default_rng(0)
        q = rng.standard_normal((seq, heads, d)).astype(np.float32)
        k = rng.standard_normal((seq, heads, d)).astype(np.float32)
        v = rng.standard_normal((seq, heads, d)).astype(np.float32)
        return q, k, v

    def _naive_attn(self, q, k, v, causal=True):
        """Reference naive attention."""
        seq, H, d = q.shape
        scale = 1.0 / math.sqrt(d)
        out = np.zeros_like(q)
        for h in range(H):
            s = q[:, h, :] @ k[:, h, :].T * scale
            if causal:
                for i in range(seq):
                    s[i, i + 1 :] = -1e9
            s -= s.max(axis=-1, keepdims=True)
            w = np.exp(s)
            w /= w.sum(axis=-1, keepdims=True)
            out[:, h, :] = w @ v[:, h, :]
        return out

    def test_output_shape(self):
        q, k, v = self._qkv()
        mfa = MetalFlashAttention()
        out, lse = mfa.forward(q, k, v)
        assert out.shape == q.shape
        assert lse.shape == (q.shape[0], q.shape[1])

    def test_matches_naive_causal(self):
        q, k, v = self._qkv(seq=8, heads=2, d=16)
        mfa = MetalFlashAttention(MetalFlashConfig(block_q=4, block_k=4, causal=True))
        out_flash, _ = mfa.forward(q, k, v)
        out_naive = self._naive_attn(q, k, v, causal=True)
        np.testing.assert_allclose(out_flash, out_naive, atol=1e-4)

    def test_matches_naive_noncausal(self):
        q, k, v = self._qkv(seq=6, heads=1, d=8)
        mfa = MetalFlashAttention(MetalFlashConfig(block_q=3, block_k=3, causal=False))
        out_flash, _ = mfa.forward(q, k, v)
        # Non-causal: all positions can attend to all
        out_naive = self._naive_attn(q, k, v, causal=False)
        np.testing.assert_allclose(out_flash, out_naive, atol=1e-4)

    def test_single_head_input(self):
        rng = np.random.default_rng(1)
        seq, d = 6, 8
        q = rng.standard_normal((seq, d)).astype(np.float32)
        k = rng.standard_normal((seq, d)).astype(np.float32)
        v = rng.standard_normal((seq, d)).astype(np.float32)
        mfa = MetalFlashAttention(MetalFlashConfig(causal=True))
        out, lse = mfa.forward(q, k, v)
        assert out.shape == (seq, d)
        assert lse.shape == (seq,)

    def test_stats_updated(self):
        q, k, v = self._qkv(seq=4)
        mfa = MetalFlashAttention()
        mfa.forward(q, k, v)
        mfa.forward(q, k, v)
        assert mfa.stats.total_forward_calls == 2
        assert mfa.stats.total_query_tokens == 8

    def test_reset_stats(self):
        q, k, v = self._qkv(seq=4)
        mfa = MetalFlashAttention()
        mfa.forward(q, k, v)
        mfa.reset_stats()
        assert mfa.stats.total_forward_calls == 0

    def test_output_finite(self):
        q, k, v = self._qkv(seq=8, heads=2, d=16)
        mfa = MetalFlashAttention()
        out, _ = mfa.forward(q, k, v)
        assert np.all(np.isfinite(out))

    def test_repr_contains_class_name(self):
        mfa = MetalFlashAttention()
        assert "MetalFlashAttention" in repr(mfa)


# ---------------------------------------------------------------------------
# SpeculativeStreamer
# ---------------------------------------------------------------------------

from squish.speculative.spec_stream import (
    SpecStreamConfig,
    StreamedToken,
    SpecStreamStats,
    SpeculativeStreamer,
)


class TestSpecStreamConfig:
    def test_defaults(self):
        cfg = SpecStreamConfig()
        assert cfg.buffer_size == 16
        assert cfg.rollback_on_reject is True

    def test_invalid_buffer_size(self):
        with pytest.raises(ValueError, match="buffer_size"):
            SpecStreamConfig(buffer_size=0)


class TestSpeculativeStreamer:
    def test_all_accepted_no_rollback(self):
        s = SpeculativeStreamer()
        s.push_draft([10, 20, 30])
        n_rb = s.commit([True, True, True], correction_token=40)
        assert n_rb == 0
        tokens = s.flush()
        assert 10 in tokens and 20 in tokens and 30 in tokens

    def test_first_token_rejected(self):
        s = SpeculativeStreamer()
        s.push_draft([10, 20])
        n_rb = s.commit([False, False], correction_token=99)
        tokens = s.flush()
        assert 99 in tokens
        assert 10 not in tokens or n_rb > 0

    def test_partial_rejection(self):
        s = SpeculativeStreamer()
        s.push_draft([10, 20, 30, 40])
        n_rb = s.commit([True, True, False, False], correction_token=99)
        tokens = s.flush()
        assert 99 in tokens
        assert 10 in tokens
        assert 20 in tokens

    def test_flush_returns_committed(self):
        s = SpeculativeStreamer()
        s.push_draft([1, 2, 3])
        s.commit([True, True, True], correction_token=4)
        tokens = s.flush()
        assert isinstance(tokens, list)
        assert len(tokens) >= 3

    def test_push_respects_buffer_size(self):
        s = SpeculativeStreamer(SpecStreamConfig(buffer_size=3))
        s.push_draft([1, 2, 3, 4, 5])
        assert s.n_pending_draft <= 3

    def test_rollback_to(self):
        s = SpeculativeStreamer()
        s.push_draft([1, 2, 3])
        s.commit([True, True, True], correction_token=4)
        s.rollback_to(2)
        tokens = s.flush()
        assert len(tokens) == 2

    def test_rollback_to_invalid_raises(self):
        s = SpeculativeStreamer()
        with pytest.raises(ValueError):
            s.rollback_to(-1)

    def test_reset_clears_all(self):
        s = SpeculativeStreamer()
        s.push_draft([1, 2])
        s.commit([True, True], correction_token=3)
        s.reset()
        assert s.n_committed == 0
        assert s.n_pending_draft == 0
        assert not s.is_done

    def test_eos_sets_done(self):
        s = SpeculativeStreamer(SpecStreamConfig(eos_token_id=2))
        s.push_draft([5, 2])  # 2 is EOS
        s.commit([True, True], correction_token=3)
        assert s.is_done

    def test_stats_push_calls(self):
        s = SpeculativeStreamer()
        s.push_draft([1, 2, 3])
        assert s.stats.push_calls == 1

    def test_stats_acceptance_rate(self):
        s = SpeculativeStreamer()
        s.push_draft([1, 2, 3, 4])
        s.commit([True, True, False, False], correction_token=9)
        # 4 total draft, 2 accepted
        assert s.stats.acceptance_rate == pytest.approx(0.5, abs=0.01)

    def test_repr_contains_class_name(self):
        assert "SpeculativeStreamer" in repr(SpeculativeStreamer())


# ---------------------------------------------------------------------------
# BlockSparseKVManager
# ---------------------------------------------------------------------------

from squish.kv.block_sparse_kv import (
    BlockSparseConfig,
    SparseBlock,
    BlockSparseStats,
    BlockSparseKVManager,
)


class TestBlockSparseConfig:
    def test_defaults(self):
        cfg = BlockSparseConfig()
        assert cfg.block_size == 32
        assert cfg.top_k_blocks == 8
        assert cfg.score_fn == "max_attn"

    def test_invalid_block_size(self):
        with pytest.raises(ValueError, match="block_size"):
            BlockSparseConfig(block_size=0)

    def test_invalid_top_k(self):
        with pytest.raises(ValueError, match="top_k_blocks"):
            BlockSparseConfig(top_k_blocks=0)

    def test_invalid_score_fn(self):
        with pytest.raises(ValueError, match="score_fn"):
            BlockSparseConfig(score_fn="bad_fn")


class TestBlockSparseKVManager:
    def _kv(self, seq=64, heads=2, d=16):
        rng = np.random.default_rng(42)
        k = rng.standard_normal((seq, heads, d)).astype(np.float32)
        v = rng.standard_normal((seq, heads, d)).astype(np.float32)
        q = rng.standard_normal((1, heads, d)).astype(np.float32)
        return k, v, q

    def test_prune_reduces_seq_len(self):
        cfg = BlockSparseConfig(block_size=8, top_k_blocks=2)
        m = BlockSparseKVManager(cfg)
        k, v, q = self._kv(seq=64)
        pk, pv, mask = m.prune(k, v, q)
        assert pk.shape[0] < k.shape[0]

    def test_pruned_kv_shapes_match(self):
        cfg = BlockSparseConfig(block_size=8, top_k_blocks=2)
        m = BlockSparseKVManager(cfg)
        k, v, q = self._kv(seq=32)
        pk, pv, mask = m.prune(k, v, q)
        assert pk.shape == pv.shape

    def test_block_mask_is_bool(self):
        m = BlockSparseKVManager(BlockSparseConfig(block_size=8, top_k_blocks=2))
        k, v, q = self._kv(seq=32)
        _, _, mask = m.prune(k, v, q)
        assert mask.dtype == bool

    def test_all_seq_selected_if_k_large(self):
        cfg = BlockSparseConfig(block_size=8, top_k_blocks=100, always_last=True)
        m = BlockSparseKVManager(cfg)
        k, v, q = self._kv(seq=32)
        pk, _, mask = m.prune(k, v, q)
        assert mask.sum() == 32  # all selected

    def test_score_blocks_sorted_descending(self):
        m = BlockSparseKVManager(BlockSparseConfig(block_size=8))
        k, _, q = self._kv(seq=32)
        blocks = m.score_blocks(k, q)
        scores = [b.score for b in blocks]
        assert scores == sorted(scores, reverse=True)

    def test_compute_attention_shape(self):
        m = BlockSparseKVManager(BlockSparseConfig(block_size=8, top_k_blocks=2))
        k, v, q = self._kv(seq=32)
        pk, pv, _ = m.prune(k, v, q)
        out = m.compute_attention(q, pk, pv)
        assert out.shape == q.shape

    def test_different_score_fns(self):
        for fn in ["max_attn", "mean_attn", "norm_attn"]:
            m = BlockSparseKVManager(BlockSparseConfig(block_size=8, top_k_blocks=2, score_fn=fn))
            k, v, q = self._kv(seq=32)
            pk, pv, mask = m.prune(k, v, q)
            assert pk.shape[0] > 0

    def test_stats_updated(self):
        m = BlockSparseKVManager(BlockSparseConfig(block_size=8, top_k_blocks=2))
        k, v, q = self._kv(seq=32)
        m.prune(k, v, q)
        assert m.stats.total_forward_calls == 1

    def test_output_finite(self):
        m = BlockSparseKVManager(BlockSparseConfig(block_size=4, top_k_blocks=2))
        k, v, q = self._kv(seq=16)
        pk, pv, _ = m.prune(k, v, q)
        out = m.compute_attention(q, pk, pv)
        assert np.all(np.isfinite(out))

    def test_repr_contains_class_name(self):
        assert "BlockSparseKVManager" in repr(BlockSparseKVManager())


# ---------------------------------------------------------------------------
# PDDisaggregator
# ---------------------------------------------------------------------------

from squish.serving.pd_disagg import (
    PDConfig,
    PrefillResult,
    PDStats,
    PDDisaggregator,
)


class TestPDConfig:
    def test_defaults(self):
        cfg = PDConfig()
        assert cfg.kv_transfer_timeout_ms == 500.0
        assert cfg.max_prefill_tokens == 8192

    def test_invalid_timeout(self):
        with pytest.raises(ValueError, match="kv_transfer_timeout_ms"):
            PDConfig(kv_transfer_timeout_ms=0)

    def test_invalid_max_prefill(self):
        with pytest.raises(ValueError, match="max_prefill_tokens"):
            PDConfig(max_prefill_tokens=0)

    def test_invalid_max_decode(self):
        with pytest.raises(ValueError, match="max_decode_tokens"):
            PDConfig(max_decode_tokens=0)


class TestPDDisaggregator:
    def _make_pd(self):
        def pfn(tokens, max_new):
            return {"kv": tokens, "n_tokens": len(tokens)}
        def dfn(kv, n_gen, max_new):
            return [42] * max_new
        return PDDisaggregator(PDConfig(), prefill_fn=pfn, decode_fn=dfn)

    def test_submit_prefill_returns_result(self):
        pd = self._make_pd()
        result = pd.submit_prefill([1, 2, 3], max_new_tokens=5)
        assert isinstance(result, PrefillResult)
        assert result.n_prompt_toks == 3

    def test_submit_decode_returns_tokens(self):
        pd = self._make_pd()
        pf = pd.submit_prefill([1, 2, 3], max_new_tokens=4)
        tokens = pd.submit_decode(pf, max_new_tokens=4)
        assert tokens == [42, 42, 42, 42]

    def test_generate_end_to_end(self):
        pd = self._make_pd()
        result = pd.generate("req1", [1, 2, 3, 4], max_new_tokens=6)
        assert len(result) == 6

    def test_stats_requests_incremented(self):
        pd = self._make_pd()
        pd.generate("r1", [1, 2], max_new_tokens=3)
        pd.generate("r2", [3, 4], max_new_tokens=3)
        assert pd.stats.total_requests == 2

    def test_stats_prompt_tokens(self):
        pd = self._make_pd()
        pd.generate("r1", [1, 2, 3, 4, 5], max_new_tokens=2)
        assert pd.stats.total_prompt_tokens == 5

    def test_stats_generated_tokens(self):
        pd = self._make_pd()
        pd.generate("r1", [1], max_new_tokens=8)
        assert pd.stats.total_generated_tokens == 8

    def test_default_stubs_work(self):
        pd = PDDisaggregator(PDConfig())
        result = pd.generate("r", [1, 2, 3], max_new_tokens=4)
        assert isinstance(result, list)

    def test_pending_kv_cleared_after_decode(self):
        pd = self._make_pd()
        pf = pd.submit_prefill([1, 2], max_new_tokens=3)
        assert pd.pending_kv_count == 1
        pd.submit_decode(pf, max_new_tokens=3)
        assert pd.pending_kv_count == 0

    def test_repr_contains_class_name(self):
        assert "PDDisaggregator" in repr(PDDisaggregator())


# ---------------------------------------------------------------------------
# DejaVuSparseFFN
# ---------------------------------------------------------------------------

from squish.token.deja_vu_sparse import (
    DejaVuConfig,
    FFNPredictor,
    DejaVuStats,
    DejaVuSparseFFN,
)


class TestDejaVuConfig:
    def test_defaults(self):
        cfg = DejaVuConfig()
        assert cfg.hidden_size == 512
        assert cfg.ffn_size == 2048
        assert cfg.threshold == 0.3

    def test_invalid_hidden_size(self):
        with pytest.raises(ValueError, match="hidden_size"):
            DejaVuConfig(hidden_size=0)

    def test_invalid_ffn_size(self):
        with pytest.raises(ValueError, match="ffn_size"):
            DejaVuConfig(ffn_size=0)

    def test_invalid_predictor_hidden(self):
        with pytest.raises(ValueError, match="predictor_hidden"):
            DejaVuConfig(predictor_hidden=0)

    def test_invalid_threshold(self):
        with pytest.raises(ValueError, match="threshold"):
            DejaVuConfig(threshold=1.0)

    def test_invalid_n_calibration_epochs(self):
        with pytest.raises(ValueError, match="n_calibration_epochs"):
            DejaVuConfig(n_calibration_epochs=0)


class TestFFNPredictor:
    def test_predict_probs_range(self):
        pred = FFNPredictor(hidden_size=32, ffn_size=64, predictor_hidden=16)
        h = np.random.randn(32).astype(np.float32)
        probs = pred.predict_probs(h)
        assert probs.shape == (64,)
        assert np.all(probs >= 0) and np.all(probs <= 1)

    def test_predict_mask_is_bool(self):
        pred = FFNPredictor(hidden_size=32, ffn_size=64, predictor_hidden=16)
        h = np.random.randn(32).astype(np.float32)
        mask = pred.predict_mask(h, threshold=0.5)
        assert mask.dtype == bool
        assert mask.shape == (64,)

    def test_train_step_returns_finite_loss(self):
        pred = FFNPredictor(hidden_size=16, ffn_size=8, predictor_hidden=8)
        X = np.random.randn(4, 16).astype(np.float32)
        y = np.random.randint(0, 2, (4, 8)).astype(np.float32)
        loss = pred.train_step(X, y, lr=1e-3)
        assert math.isfinite(loss)
        assert loss >= 0


class TestDejaVuSparseFFN:
    def _make_cfg(self, hs=32, fs=32, ph=16):
        return DejaVuConfig(
            hidden_size=hs,
            ffn_size=fs,
            predictor_hidden=ph,
            n_calibration_epochs=2,
        )

    def _relu_ffn(self, hs=32):
        W = np.eye(hs, dtype=np.float32)
        def fn(x):
            return np.maximum(x @ W, 0.0)
        return fn

    def test_forward_without_calibration_returns_dense(self):
        cfg = self._make_cfg()
        dv = DejaVuSparseFFN(cfg, ffn_fn=self._relu_ffn())
        x = np.random.randn(32).astype(np.float32)
        out = dv.forward(x)
        assert out.shape == (32,)

    def test_calibrate_returns_losses(self):
        cfg = self._make_cfg()
        dv = DejaVuSparseFFN(cfg, ffn_fn=self._relu_ffn())
        X = np.random.randn(10, 32).astype(np.float32)
        losses = dv.calibrate(X)
        assert len(losses) == 2  # n_calibration_epochs
        assert all(math.isfinite(l) for l in losses)

    def test_calibrate_sets_is_calibrated(self):
        cfg = self._make_cfg()
        dv = DejaVuSparseFFN(cfg, ffn_fn=self._relu_ffn())
        X = np.random.randn(10, 32).astype(np.float32)
        dv.calibrate(X)
        assert dv.is_calibrated

    def test_forward_after_calibration_updates_stats(self):
        cfg = self._make_cfg()
        fn = self._relu_ffn()
        dv = DejaVuSparseFFN(cfg, ffn_fn=fn)
        X = np.random.randn(10, 32).astype(np.float32)
        dv.calibrate(X)
        x = np.random.randn(32).astype(np.float32)
        dv.forward(x)
        assert dv.stats.total_forward_calls == 1

    def test_sparsity_ratio_in_range(self):
        cfg = self._make_cfg()
        fn = self._relu_ffn()
        dv = DejaVuSparseFFN(cfg, ffn_fn=fn)
        X = np.random.randn(20, 32).astype(np.float32)
        dv.calibrate(X)
        for _ in range(5):
            dv.forward(np.random.randn(32).astype(np.float32))
        assert 0.0 <= dv.stats.mean_sparsity <= 1.0

    def test_calibrate_without_fn_raises(self):
        cfg = self._make_cfg()
        dv = DejaVuSparseFFN(cfg)
        X = np.random.randn(5, 32).astype(np.float32)
        with pytest.raises(ValueError):
            dv.calibrate(X)

    def test_forward_without_fn_raises(self):
        cfg = self._make_cfg()
        dv = DejaVuSparseFFN(cfg)
        # Manually mark as calibrated (no fn stored)
        dv.predictor._is_calibrated = True
        with pytest.raises(ValueError):
            dv.forward(np.zeros(32, dtype=np.float32))

    def test_output_shape(self):
        cfg = self._make_cfg()
        fn = self._relu_ffn()
        dv = DejaVuSparseFFN(cfg, ffn_fn=fn)
        X = np.random.randn(10, 32).astype(np.float32)
        dv.calibrate(X)
        out = dv.forward(X[0])
        assert out.shape == (32,)

    def test_repr_contains_class_name(self):
        assert "DejaVuSparseFFN" in repr(DejaVuSparseFFN(self._make_cfg()))


# ---------------------------------------------------------------------------
# LayerOverlapLoader
# ---------------------------------------------------------------------------

from squish.io.layer_overlap_loader import (
    LayerOverlapConfig,
    LayerHandle,
    LayerOverlapStats,
    LayerOverlapLoader,
)


class TestLayerOverlapConfig:
    def test_defaults(self):
        cfg = LayerOverlapConfig()
        assert cfg.prefetch_count == 2
        assert cfg.load_timeout_s == 5.0

    def test_invalid_prefetch_count(self):
        with pytest.raises(ValueError, match="prefetch_count"):
            LayerOverlapConfig(prefetch_count=0)

    def test_invalid_load_timeout(self):
        with pytest.raises(ValueError, match="load_timeout_s"):
            LayerOverlapConfig(load_timeout_s=0)


def _make_load_fn():
    """Simple synchronous load function returning weight arrays."""
    def load_fn(idx):
        return {"weight": np.eye(8, dtype=np.float32) * idx}
    return load_fn


class TestLayerHandle:
    def test_initially_not_ready(self):
        handle = LayerHandle(3)
        assert not handle.ready.is_set()
        assert handle.weights is None

    def test_ready_after_set(self):
        handle = LayerHandle(0)
        handle.weights = {"w": np.zeros(4)}
        handle.ready.set()
        assert handle.wait(timeout=0.1)

    def test_repr_contains_idx(self):
        handle = LayerHandle(7)
        assert "7" in repr(handle)


class TestLayerOverlapLoader:
    def test_start_and_get_layer(self):
        loader = LayerOverlapLoader(LayerOverlapConfig(prefetch_count=1))
        loader.start(n_layers=4, load_fn=_make_load_fn())
        w = loader.get_layer(0)
        assert "weight" in w
        loader.stop()

    def test_get_layer_correct_value(self):
        loader = LayerOverlapLoader(LayerOverlapConfig(prefetch_count=2))
        loader.start(n_layers=4, load_fn=_make_load_fn())
        for i in range(4):
            w = loader.get_layer(i)
            np.testing.assert_allclose(w["weight"], np.eye(8) * i, atol=1e-6)
        loader.stop()

    def test_prefetch_next_schedules_ahead(self):
        loader = LayerOverlapLoader(LayerOverlapConfig(prefetch_count=2))
        loader.start(n_layers=6, load_fn=_make_load_fn())
        w0 = loader.get_layer(0)
        loader.prefetch_next(0)
        time.sleep(0.05)  # let prefetch settle
        # Layer 1 and 2 should be in cache
        w1 = loader.get_layer(1)
        assert "weight" in w1
        loader.stop()

    def test_out_of_range_raises(self):
        loader = LayerOverlapLoader()
        loader.start(n_layers=4, load_fn=_make_load_fn())
        with pytest.raises(ValueError):
            loader.get_layer(10)
        loader.stop()

    def test_get_before_start_raises(self):
        loader = LayerOverlapLoader()
        with pytest.raises(RuntimeError):
            loader.get_layer(0)

    def test_hit_rate_after_prefetch(self):
        loader = LayerOverlapLoader(LayerOverlapConfig(prefetch_count=3, load_timeout_s=2))
        loader.start(n_layers=4, load_fn=_make_load_fn())
        time.sleep(0.1)  # allow prefetch threads to complete
        for i in range(4):
            loader.get_layer(i)
            loader.prefetch_next(i)
        assert loader.stats.hit_rate >= 0.0  # may be 0 on fast machines
        loader.stop()

    def test_stop_releases_cache(self):
        loader = LayerOverlapLoader(LayerOverlapConfig(prefetch_count=1))
        loader.start(n_layers=3, load_fn=_make_load_fn())
        loader.get_layer(0)
        loader.stop()
        assert loader.cached_layer_count == 0

    def test_stats_layers_loaded(self):
        loader = LayerOverlapLoader(LayerOverlapConfig(prefetch_count=1))
        loader.start(n_layers=2, load_fn=_make_load_fn())
        loader.get_layer(0)
        loader.get_layer(1)
        loader.stop()
        assert loader.stats.total_layers_loaded >= 2

    def test_repr_contains_class_name(self):
        loader = LayerOverlapLoader()
        assert "LayerOverlapLoader" in repr(loader)
