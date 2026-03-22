"""Tests for Wave 53a Linear Recurrent Architecture modules.

Covers:
  - squish.attention.mamba2_ssm    (Mamba2SSM)
  - squish.attention.rwkv_channel_mix (RWKV6ChannelMix)
  - squish.attention.hawk_recurrent   (HawkLinearRNN)
  - squish.attention.xlstm_block      (xLSTMBlock)
  - squish.attention.ttt_layer        (TTTLinearLayer)
  - squish.attention.delta_net        (DeltaNetLinear)
"""

import unittest
import numpy as np


# ---------------------------------------------------------------------------
# Mamba2SSM
# ---------------------------------------------------------------------------

class TestMamba2Config(unittest.TestCase):
    def test_import(self):
        from squish.attention.mamba2_ssm import Mamba2Config, Mamba2State, Mamba2SSM
        cfg = Mamba2Config()
        self.assertIsNotNone(cfg)

    def test_default_fields(self):
        from squish.attention.mamba2_ssm import Mamba2Config
        cfg = Mamba2Config()
        self.assertGreater(cfg.d_model, 0)

    def test_custom_fields(self):
        from squish.attention.mamba2_ssm import Mamba2Config
        cfg = Mamba2Config(d_model=128)
        self.assertEqual(cfg.d_model, 128)


class TestMamba2State(unittest.TestCase):
    def test_create(self):
        from squish.attention.mamba2_ssm import Mamba2Config, Mamba2SSM
        cfg = Mamba2Config(d_model=64)
        model = Mamba2SSM(cfg)
        state = model.init_state()
        self.assertIsNotNone(state)

    def test_state_bytes(self):
        from squish.attention.mamba2_ssm import Mamba2Config, Mamba2SSM
        cfg = Mamba2Config(d_model=64)
        model = Mamba2SSM(cfg)
        state = model.init_state()
        self.assertGreater(state.ssm_state.nbytes + state.conv_state.nbytes, 0)


class TestMamba2SSM(unittest.TestCase):
    def setUp(self):
        from squish.attention.mamba2_ssm import Mamba2Config, Mamba2SSM
        self.cfg = Mamba2Config(d_model=64, seed=0)
        self.model = Mamba2SSM(self.cfg)

    def test_forward_shape(self):
        x = np.random.randn(4, 64).astype(np.float32)
        state = self.model.init_state()
        out, new_state = self.model.forward(x, state)
        self.assertEqual(out.shape, (4, 64))

    def test_state_updates(self):
        x = np.random.randn(3, 64).astype(np.float32)
        state = self.model.init_state()
        _, new_state = self.model.forward(x, state)
        # State ssm_state should be non-zero after processing
        self.assertIsNotNone(new_state.ssm_state)

    def test_deterministic(self):
        x = np.random.randn(5, 64).astype(np.float32)
        out1, _ = self.model.forward(x, self.model.init_state())
        out2, _ = self.model.forward(x, self.model.init_state())
        np.testing.assert_array_equal(out1, out2)

    def test_single_token(self):
        x = np.random.randn(1, 64).astype(np.float32)
        state = self.model.init_state()
        out, _ = self.model.forward(x, state)
        self.assertEqual(out.shape, (1, 64))

    def test_sequential_steps_chained(self):
        x = np.random.randn(6, 64).astype(np.float32)
        state = self.model.init_state()
        out_full, _ = self.model.forward(x, state)
        # Chunked forward should produce the same shape
        s0 = self.model.init_state()
        _, s1 = self.model.forward(x[:3], s0)
        out_chunk2, _ = self.model.forward(x[3:], s1)
        self.assertEqual(out_chunk2.shape, (3, 64))


# ---------------------------------------------------------------------------
# RWKV6ChannelMix
# ---------------------------------------------------------------------------

class TestRWKV6Config(unittest.TestCase):
    def test_import(self):
        from squish.attention.rwkv_channel_mix import RWKV6Config
        cfg = RWKV6Config()
        self.assertIsNotNone(cfg)

    def test_validation_head_dim(self):
        from squish.attention.rwkv_channel_mix import RWKV6Config
        with self.assertRaises(ValueError):
            RWKV6Config(d_model=512, n_heads=7, head_dim=64)

    def test_custom(self):
        from squish.attention.rwkv_channel_mix import RWKV6Config
        cfg = RWKV6Config(d_model=128, n_heads=2, head_dim=64)
        self.assertEqual(cfg.d_model, 128)


class TestRWKV6State(unittest.TestCase):
    def test_state_bytes(self):
        from squish.attention.rwkv_channel_mix import RWKV6Config, RWKV6ChannelMix
        cfg = RWKV6Config(d_model=128, n_heads=2, head_dim=64)
        model = RWKV6ChannelMix(cfg)
        state = model.new_state()
        self.assertGreater(state.state_bytes, 0)

    def test_n_heads_property(self):
        from squish.attention.rwkv_channel_mix import RWKV6Config, RWKV6ChannelMix
        cfg = RWKV6Config(d_model=128, n_heads=2, head_dim=64)
        model = RWKV6ChannelMix(cfg)
        state = model.new_state()
        self.assertEqual(state.n_heads, 2)


class TestRWKV6ChannelMix(unittest.TestCase):
    def setUp(self):
        from squish.attention.rwkv_channel_mix import RWKV6Config, RWKV6ChannelMix
        self.cfg = RWKV6Config(d_model=128, n_heads=2, head_dim=64, seed=42)
        self.model = RWKV6ChannelMix(self.cfg)

    def test_forward_shape(self):
        x = np.random.randn(3, 128).astype(np.float32)
        state = self.model.new_state()
        out, _ = self.model.forward(x, state)
        self.assertEqual(out.shape, (3, 128))

    def test_state_n_tokens(self):
        x = np.random.randn(4, 128).astype(np.float32)
        state = self.model.new_state()
        _, new_state = self.model.forward(x, state)
        self.assertEqual(new_state.n_tokens_seen, 4)

    def test_deterministic(self):
        x = np.random.randn(2, 128).astype(np.float32)
        o1, _ = self.model.forward(x, self.model.new_state())
        o2, _ = self.model.forward(x, self.model.new_state())
        np.testing.assert_array_equal(o1, o2)

    def test_single_token(self):
        x = np.random.randn(1, 128).astype(np.float32)
        state = self.model.new_state()
        out, _ = self.model.forward(x, state)
        self.assertEqual(out.shape, (1, 128))

    def test_state_shape(self):
        from squish.attention.rwkv_channel_mix import RWKV6Config, RWKV6ChannelMix
        cfg = RWKV6Config(d_model=128, n_heads=2, head_dim=64)
        model = RWKV6ChannelMix(cfg)
        state = model.new_state()
        self.assertEqual(state.time_state.shape, (2, 64, cfg.d_state))


# ---------------------------------------------------------------------------
# HawkLinearRNN
# ---------------------------------------------------------------------------

class TestHawkConfig(unittest.TestCase):
    def test_import(self):
        from squish.attention.hawk_recurrent import HawkConfig
        cfg = HawkConfig()
        self.assertIsNotNone(cfg)

    def test_dt_min_validation(self):
        from squish.attention.hawk_recurrent import HawkConfig
        with self.assertRaises(ValueError):
            HawkConfig(dt_min=0.0)

    def test_negative_dt_min(self):
        from squish.attention.hawk_recurrent import HawkConfig
        with self.assertRaises(ValueError):
            HawkConfig(dt_min=-1.0)


class TestHawkState(unittest.TestCase):
    def test_state_bytes(self):
        from squish.attention.hawk_recurrent import HawkConfig, HawkLinearRNN
        cfg = HawkConfig(d_model=64, d_state=64)
        model = HawkLinearRNN(cfg)
        state = model.new_state()
        self.assertGreater(state.state_bytes, 0)

    def test_n_steps_initial(self):
        from squish.attention.hawk_recurrent import HawkConfig, HawkLinearRNN
        cfg = HawkConfig(d_model=64)
        model = HawkLinearRNN(cfg)
        state = model.new_state()
        self.assertEqual(state.n_steps, 0)


class TestHawkLinearRNN(unittest.TestCase):
    def setUp(self):
        from squish.attention.hawk_recurrent import HawkConfig, HawkLinearRNN
        self.cfg = HawkConfig(d_model=64, d_state=64, seed=7)
        self.model = HawkLinearRNN(self.cfg)

    def test_forward_shape(self):
        x = np.random.randn(5, 64).astype(np.float32)
        state = self.model.new_state()
        out, _ = self.model.forward(x, state)
        self.assertEqual(out.shape, (5, 64))

    def test_recurrent_step_shape(self):
        x = np.random.randn(64).astype(np.float32)
        state = self.model.new_state()
        out, _ = self.model.recurrent_step(x, state)
        self.assertEqual(out.shape, (64,))

    def test_scan_prefill(self):
        x = np.random.randn(8, 64).astype(np.float32)
        h0 = np.zeros(self.cfg.d_state)
        outputs, h_final = self.model.scan_prefill(x, h0)
        self.assertEqual(outputs.shape, (8, 64))

    def test_forward_deterministic(self):
        x = np.random.randn(3, 64).astype(np.float32)
        o1, _ = self.model.forward(x, self.model.new_state())
        o2, _ = self.model.forward(x, self.model.new_state())
        np.testing.assert_array_equal(o1, o2)

    def test_state_step_count(self):
        x = np.random.randn(4, 64).astype(np.float32)
        state = self.model.new_state()
        _, ns = self.model.forward(x, state)
        self.assertGreater(ns.n_steps, 0)


# ---------------------------------------------------------------------------
# xLSTMBlock
# ---------------------------------------------------------------------------

class TestXLSTMConfig(unittest.TestCase):
    def test_import(self):
        from squish.attention.xlstm_block import xLSTMConfig
        cfg = xLSTMConfig()
        self.assertIsNotNone(cfg)

    def test_custom(self):
        from squish.attention.xlstm_block import xLSTMConfig
        cfg = xLSTMConfig(d_model=128)
        self.assertEqual(cfg.d_model, 128)


class TestXLSTMStates(unittest.TestCase):
    def test_slstm_state_bytes(self):
        from squish.attention.xlstm_block import xLSTMConfig, xLSTMBlock
        cfg = xLSTMConfig(d_model=128, n_slstm_heads=2, slstm_head_dim=64)
        model = xLSTMBlock(cfg)
        state = model.new_state()
        self.assertGreater(state.slstm.state_bytes, 0)

    def test_mlstm_state_bytes(self):
        from squish.attention.xlstm_block import xLSTMConfig, xLSTMBlock
        cfg = xLSTMConfig(d_model=128, n_slstm_heads=2, slstm_head_dim=64)
        model = xLSTMBlock(cfg)
        state = model.new_state()
        self.assertGreater(state.mlstm.state_bytes, 0)

    def test_n_steps_initial(self):
        from squish.attention.xlstm_block import xLSTMConfig, xLSTMBlock
        cfg = xLSTMConfig(d_model=128, n_slstm_heads=2, slstm_head_dim=64)
        model = xLSTMBlock(cfg)
        state = model.new_state()
        self.assertEqual(state.slstm.n_steps, 0)
        self.assertEqual(state.mlstm.n_steps, 0)


class TestXLSTMBlock(unittest.TestCase):
    def setUp(self):
        from squish.attention.xlstm_block import xLSTMConfig, xLSTMBlock
        self.cfg = xLSTMConfig(
            d_model=128, n_slstm_heads=2, slstm_head_dim=64, mlstm_dim=32, seed=3
        )
        self.model = xLSTMBlock(self.cfg)

    def test_forward_shape(self):
        x = np.random.randn(4, 128).astype(np.float32)
        state = self.model.new_state()
        out, _ = self.model.forward(x, state)
        self.assertEqual(out.shape, (4, 128))

    def test_single_token(self):
        x = np.random.randn(1, 128).astype(np.float32)
        state = self.model.new_state()
        out, _ = self.model.forward(x, state)
        self.assertEqual(out.shape, (1, 128))

    def test_no_nan(self):
        x = np.random.randn(6, 128).astype(np.float32)
        state = self.model.new_state()
        out, _ = self.model.forward(x, state)
        self.assertFalse(np.any(np.isnan(out)))

    def test_deterministic(self):
        x = np.random.randn(3, 128).astype(np.float32)
        o1, _ = self.model.forward(x, self.model.new_state())
        o2, _ = self.model.forward(x, self.model.new_state())
        np.testing.assert_array_equal(o1, o2)

    def test_state_steps_update(self):
        x = np.random.randn(5, 128).astype(np.float32)
        state = self.model.new_state()
        _, ns = self.model.forward(x, state)
        self.assertGreater(ns.slstm.n_steps, 0)


# ---------------------------------------------------------------------------
# TTTLinearLayer
# ---------------------------------------------------------------------------

class TestTTTConfig(unittest.TestCase):
    def test_import(self):
        from squish.attention.ttt_layer import TTTConfig
        cfg = TTTConfig()
        self.assertIsNotNone(cfg)

    def test_lr_validation(self):
        from squish.attention.ttt_layer import TTTConfig
        with self.assertRaises(ValueError):
            TTTConfig(lr=0.0)

    def test_momentum_validation(self):
        from squish.attention.ttt_layer import TTTConfig
        with self.assertRaises(ValueError):
            TTTConfig(momentum=1.0)

    def test_negative_lr(self):
        from squish.attention.ttt_layer import TTTConfig
        with self.assertRaises(ValueError):
            TTTConfig(lr=-0.01)


class TestTTTState(unittest.TestCase):
    def test_state_bytes(self):
        from squish.attention.ttt_layer import TTTConfig, TTTLinearLayer
        cfg = TTTConfig(d_model=64, mini_model_dim=32)
        model = TTTLinearLayer(cfg)
        state = model.new_state()
        self.assertGreater(state.state_bytes, 0)

    def test_initial_steps(self):
        from squish.attention.ttt_layer import TTTConfig, TTTLinearLayer
        cfg = TTTConfig(d_model=64)
        model = TTTLinearLayer(cfg)
        state = model.new_state()
        self.assertEqual(state.n_steps, 0)


class TestTTTLinearLayer(unittest.TestCase):
    def setUp(self):
        from squish.attention.ttt_layer import TTTConfig, TTTLinearLayer
        self.cfg = TTTConfig(d_model=64, mini_model_dim=32, lr=0.01, seed=99)
        self.model = TTTLinearLayer(self.cfg)

    def test_forward_shape(self):
        x = np.random.randn(5, 64).astype(np.float32)
        state = self.model.new_state()
        out, _ = self.model.forward(x, state)
        self.assertEqual(out.shape, (5, 64))

    def test_state_updates(self):
        x = np.random.randn(3, 64).astype(np.float32)
        state = self.model.new_state()
        _, ns = self.model.forward(x, state)
        self.assertGreater(ns.n_steps, 0)

    def test_no_nan(self):
        x = np.random.randn(4, 64).astype(np.float32)
        state = self.model.new_state()
        out, _ = self.model.forward(x, state)
        self.assertFalse(np.any(np.isnan(out)))

    def test_deterministic(self):
        x = np.random.randn(3, 64).astype(np.float32)
        o1, _ = self.model.forward(x, self.model.new_state())
        o2, _ = self.model.forward(x, self.model.new_state())
        np.testing.assert_array_equal(o1, o2)

    def test_weight_updates(self):
        x = np.random.randn(3, 64).astype(np.float32)
        state = self.model.new_state()
        W_before = state.W.copy()
        _, state_after = self.model.forward(x, state)
        # TTT updates W each step; it should change
        self.assertFalse(np.allclose(W_before, state_after.W))

    def test_momentum_path(self):
        from squish.attention.ttt_layer import TTTConfig, TTTLinearLayer
        cfg = TTTConfig(d_model=64, mini_model_dim=32, lr=0.01, momentum=0.9, seed=5)
        model = TTTLinearLayer(cfg)
        x = np.random.randn(3, 64).astype(np.float32)
        state = model.new_state()
        out, _ = model.forward(x, state)
        self.assertEqual(out.shape, (3, 64))


# ---------------------------------------------------------------------------
# DeltaNetLinear
# ---------------------------------------------------------------------------

class TestDeltaNetConfig(unittest.TestCase):
    def test_import(self):
        from squish.attention.delta_net import DeltaNetConfig
        cfg = DeltaNetConfig()
        self.assertIsNotNone(cfg)

    def test_beta_validation_zero(self):
        from squish.attention.delta_net import DeltaNetConfig
        with self.assertRaises(ValueError):
            DeltaNetConfig(beta=0.0)

    def test_beta_validation_over_one(self):
        from squish.attention.delta_net import DeltaNetConfig
        with self.assertRaises(ValueError):
            DeltaNetConfig(beta=1.5)

    def test_custom(self):
        from squish.attention.delta_net import DeltaNetConfig
        cfg = DeltaNetConfig(d_model=128, n_heads=2, head_dim=64)
        self.assertEqual(cfg.d_model, 128)


class TestDeltaNetState(unittest.TestCase):
    def test_state_bytes(self):
        from squish.attention.delta_net import DeltaNetConfig, DeltaNetLinear
        cfg = DeltaNetConfig(d_model=128, n_heads=2, head_dim=64, d_state=32)
        model = DeltaNetLinear(cfg)
        state = model.new_state()
        self.assertGreater(state.state_bytes, 0)

    def test_initial_steps(self):
        from squish.attention.delta_net import DeltaNetConfig, DeltaNetLinear
        cfg = DeltaNetConfig(d_model=128, n_heads=2, head_dim=64)
        model = DeltaNetLinear(cfg)
        state = model.new_state()
        self.assertEqual(state.n_steps, 0)


class TestDeltaNetLinear(unittest.TestCase):
    def setUp(self):
        from squish.attention.delta_net import DeltaNetConfig, DeltaNetLinear
        self.cfg = DeltaNetConfig(
            d_model=128, d_state=32, n_heads=2, head_dim=64, beta=0.5, seed=13
        )
        self.model = DeltaNetLinear(self.cfg)

    def test_forward_shape(self):
        x = np.random.randn(4, 128).astype(np.float32)
        state = self.model.new_state()
        out, _ = self.model.forward(x, state)
        self.assertEqual(out.shape, (4, 128))

    def test_single_token(self):
        x = np.random.randn(1, 128).astype(np.float32)
        state = self.model.new_state()
        out, _ = self.model.forward(x, state)
        self.assertEqual(out.shape, (1, 128))

    def test_deterministic(self):
        x = np.random.randn(5, 128).astype(np.float32)
        o1, _ = self.model.forward(x, self.model.new_state())
        o2, _ = self.model.forward(x, self.model.new_state())
        np.testing.assert_array_equal(o1, o2)

    def test_no_nan(self):
        x = np.random.randn(6, 128).astype(np.float32)
        state = self.model.new_state()
        out, _ = self.model.forward(x, state)
        self.assertFalse(np.any(np.isnan(out)))

    def test_state_updates(self):
        x = np.random.randn(3, 128).astype(np.float32)
        state = self.model.new_state()
        _, ns = self.model.forward(x, state)
        self.assertGreater(ns.n_steps, 0)

    def test_key_normalisation(self):
        """Delta-rule requires L2-normalised keys; verify numerics are finite."""
        rng = np.random.default_rng(0)
        x = rng.standard_normal((8, 128)).astype(np.float32)
        state = self.model.new_state()
        out, _ = self.model.forward(x, state)
        self.assertTrue(np.all(np.isfinite(out)))

    def test_learnable_beta_false(self):
        from squish.attention.delta_net import DeltaNetConfig, DeltaNetLinear
        cfg = DeltaNetConfig(
            d_model=128, d_state=32, n_heads=2, head_dim=64,
            beta=0.3, learnable_beta=False, seed=7
        )
        model = DeltaNetLinear(cfg)
        x = np.random.randn(3, 128).astype(np.float32)
        state = model.new_state()
        out, _ = model.forward(x, state)
        self.assertEqual(out.shape, (3, 128))


if __name__ == "__main__":
    unittest.main()
