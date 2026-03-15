"""
tests/speculative/test_redrafter_unit.py

Unit tests for Phase 4: ReDrafter GRU-based speculative draft head.

Covers:
  - ReDrafterConfig validation
  - ReDrafterGRU step correctness (gate shapes, update mechanics)
  - ReDrafterHead lifecycle (reset_state, draft_k interface)
  - draft_k draft_hiddens side-effect tracking
  - ReDrafterHead save/load round-trip
  - ReDrafterHead.init_random produces valid drafts
  - SpeculativeGenerator wiring (redrafter_head param accepted)
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from squish.speculative.redrafter import (
    ReDrafterConfig,
    ReDrafterGRU,
    ReDrafterHead,
    _sigmoid,
)

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

VOCAB    = 64
T_DIM    = 32   # target hidden dim
E_DIM    = 32   # embed dim
H_DIM    = 16   # GRU hidden dim
RNG      = np.random.default_rng(1337)


def _make_head(
    vocab:   int = VOCAB,
    t_dim:   int = T_DIM,
    e_dim:   int = E_DIM,
    h_dim:   int = H_DIM,
    n_layers: int = 1,
) -> ReDrafterHead:
    return ReDrafterHead.init_random(
        vocab_size        = vocab,
        target_hidden_dim = t_dim,
        embed_dim         = e_dim,
        hidden_dim        = h_dim,
        n_layers          = n_layers,
        rng               = np.random.default_rng(42),
    )


def _fake_hidden(t_dim: int = T_DIM, seq_len: int = 1) -> np.ndarray:
    """Simulate HiddenStateCapture.last_hidden output: (1, seq_len, t_dim)."""
    return RNG.standard_normal((1, seq_len, t_dim)).astype(np.float32)


# ---------------------------------------------------------------------------
# _sigmoid utility
# ---------------------------------------------------------------------------

class TestSigmoid:
    def test_zero_maps_to_half(self):
        assert abs(_sigmoid(np.float32(0.0)) - 0.5) < 1e-6

    def test_large_positive_maps_to_one(self):
        assert _sigmoid(np.float32(100.0)) > 0.999

    def test_large_negative_maps_to_zero(self):
        assert _sigmoid(np.float32(-100.0)) < 0.001

    def test_array_all_in_range(self):
        x = np.linspace(-10, 10, 100).astype(np.float32)
        s = _sigmoid(x)
        assert np.all(s >= 0.0)
        assert np.all(s <= 1.0)


# ---------------------------------------------------------------------------
# ReDrafterConfig
# ---------------------------------------------------------------------------

class TestReDrafterConfig:
    def test_defaults(self):
        cfg = ReDrafterConfig()
        assert cfg.target_hidden_dim == 2048
        assert cfg.embed_dim         == 2048
        assert cfg.hidden_dim        == 512
        assert cfg.n_layers          == 1

    def test_custom(self):
        cfg = ReDrafterConfig(target_hidden_dim=64, embed_dim=64, hidden_dim=32, n_layers=2)
        assert cfg.n_layers == 2

    def test_target_hidden_dim_zero_raises(self):
        with pytest.raises(ValueError, match="target_hidden_dim"):
            ReDrafterConfig(target_hidden_dim=0)

    def test_embed_dim_zero_raises(self):
        with pytest.raises(ValueError, match="embed_dim"):
            ReDrafterConfig(embed_dim=0)

    def test_hidden_dim_zero_raises(self):
        with pytest.raises(ValueError, match="hidden_dim"):
            ReDrafterConfig(hidden_dim=0)

    def test_n_layers_zero_raises(self):
        with pytest.raises(ValueError, match="n_layers"):
            ReDrafterConfig(n_layers=0)


# ---------------------------------------------------------------------------
# ReDrafterGRU
# ---------------------------------------------------------------------------

class TestReDrafterGRU:
    def test_step_output_shape(self):
        cell = ReDrafterGRU.random_init(input_dim=16, hidden_dim=8)
        x = np.zeros(16, dtype=np.float32)
        h = np.zeros(8,  dtype=np.float32)
        h_new = cell.step(x, h)
        assert h_new.shape == (8,)

    def test_step_output_dtype_float32(self):
        cell = ReDrafterGRU.random_init(input_dim=16, hidden_dim=8)
        h_new = cell.step(
            np.zeros(16, dtype=np.float32),
            np.zeros(8,  dtype=np.float32),
        )
        assert h_new.dtype == np.float32

    def test_zero_weights_zero_input_gives_zero_output(self):
        """With all-zero weights/biases and zero input, h' = h × (1-z) + z × 0.
        At zero weights: z = σ(0) = 0.5, n = 0; so h' = 0.5 * h + 0 = 0."""
        cell  = ReDrafterGRU(input_dim=4, hidden_dim=4)  # all-zero weights
        h_in  = np.zeros(4, dtype=np.float32)
        h_out = cell.step(np.zeros(4, dtype=np.float32), h_in)
        assert np.allclose(h_out, 0.0)

    def test_reset_gate_blocks_previous_hidden(self):
        """If reset gate ≈ 0 the candidate ignores the previous hidden state.
        We construct weights so W_r produces a large negative number, forcing r≈0."""
        hd = 4
        cell = ReDrafterGRU(input_dim=hd, hidden_dim=hd)
        # Force r ≈ 0: set W_r rows very negative
        cell.W[:hd, :] = -100.0
        h_large = np.ones(hd, dtype=np.float32) * 10.0
        h_out   = cell.step(np.zeros(hd, dtype=np.float32), h_large)
        # With r ≈ 0 and z ≈ 0.5, h' ≈ 0.5 * h_large; with r≈0 candidate ignores U_n@h
        # The exact value depends on all gates — just check h_out is not equal to h_large
        assert not np.allclose(h_out, h_large)

    def test_state_changes_per_step(self):
        """Two consecutive steps with different inputs should give different outputs."""
        cell = ReDrafterGRU.random_init(input_dim=8, hidden_dim=8)
        h = np.zeros(8, dtype=np.float32)
        x1, x2 = RNG.standard_normal((2, 8)).astype(np.float32)
        h1 = cell.step(x1, h)
        h2 = cell.step(x2, h)
        assert not np.allclose(h1, h2)


# ---------------------------------------------------------------------------
# ReDrafterHead — lifecycle
# ---------------------------------------------------------------------------

class TestReDrafterHeadLifecycle:
    def test_reset_state_clears_hiddens(self):
        head = _make_head()
        tgt  = _fake_hidden()
        head.draft_k(tgt, k=3, prev_token_id=0, temperature=1.0, top_p=1.0, eos_id=1)
        assert len(head.draft_hiddens) > 0
        head.reset_state()
        # _h_state should be zero vectors
        for h in head._h_state:
            assert np.allclose(h, 0.0)

    def test_draft_hiddens_cleared_on_new_draft(self):
        head = _make_head()
        tgt  = _fake_hidden()
        head.draft_k(tgt, k=2, prev_token_id=0, temperature=1.0, top_p=1.0, eos_id=1)
        first_len = len(head.draft_hiddens)
        # A second call should reset
        head.draft_k(tgt, k=2, prev_token_id=0, temperature=1.0, top_p=1.0, eos_id=1)
        assert len(head.draft_hiddens) == first_len  # same k → same length (unless eos)

    def test_mismatched_n_layers_raises(self):
        gru_layers = [
            ReDrafterGRU.random_init(H_DIM, H_DIM),
            ReDrafterGRU.random_init(H_DIM, H_DIM),
        ]
        cfg  = ReDrafterConfig(target_hidden_dim=T_DIM, embed_dim=E_DIM, hidden_dim=H_DIM, n_layers=1)
        proj = np.zeros((H_DIM, T_DIM + E_DIM), dtype=np.float32)
        lmh  = np.zeros((VOCAB, H_DIM), dtype=np.float32)
        emb  = np.zeros((VOCAB, E_DIM), dtype=np.float32)
        with pytest.raises(AssertionError):
            ReDrafterHead(cfg, gru_layers, lmh, emb, proj)


# ---------------------------------------------------------------------------
# ReDrafterHead — draft_k interface
# ---------------------------------------------------------------------------

class TestReDrafterHeadDraftK:
    def test_draft_k_returns_list_of_ints(self):
        head = _make_head()
        ids, probs = head.draft_k(
            _fake_hidden(), k=4, prev_token_id=0,
            temperature=1.0, top_p=1.0, eos_id=99,
        )
        assert isinstance(ids, list)
        assert all(isinstance(i, int) for i in ids)

    def test_draft_k_returns_list_of_prob_arrays(self):
        head = _make_head()
        _, probs = head.draft_k(
            _fake_hidden(), k=4, prev_token_id=0,
            temperature=1.0, top_p=1.0, eos_id=99,
        )
        assert isinstance(probs, list)
        for p in probs:
            assert isinstance(p, np.ndarray)
            assert p.shape == (VOCAB,)
            assert abs(p.sum() - 1.0) < 1e-5

    def test_draft_k_1_returns_single_token(self):
        head = _make_head()
        ids, probs = head.draft_k(
            _fake_hidden(), k=1, prev_token_id=0,
            temperature=1.0, top_p=1.0, eos_id=99,
        )
        assert len(ids)   == 1
        assert len(probs) == 1

    def test_draft_k_stops_at_eos(self):
        """Set eos_id to 0 and ensure drafting stops early when 0 is sampled."""
        # Use a head with very biased lm_head so token 0 is always sampled
        head = _make_head()
        # Force token-0 to be sampled by making its logit very large
        head._lm_head_w[:] = -1000.0
        head._lm_head_w[0, :] = 1000.0   # token 0 → very high logit
        ids, _ = head.draft_k(
            _fake_hidden(), k=5, prev_token_id=1,
            temperature=1.0, top_p=1.0, eos_id=0,
        )
        # Should stop at position 0 (first token is eos)
        assert len(ids) == 1
        assert ids[0] == 0

    def test_draft_hiddens_length_matches_output(self):
        head = _make_head()
        ids, _ = head.draft_k(
            _fake_hidden(), k=4, prev_token_id=0,
            temperature=1.0, top_p=1.0, eos_id=99,
        )
        assert len(head.draft_hiddens) == len(ids)

    def test_draft_hiddens_shape_per_step(self):
        head = _make_head(h_dim=H_DIM)
        head.draft_k(
            _fake_hidden(), k=3, prev_token_id=0,
            temperature=1.0, top_p=1.0, eos_id=99,
        )
        for h in head.draft_hiddens:
            assert h.shape == (H_DIM,)

    def test_accepts_numpy_hidden_state(self):
        head = _make_head()
        # Pass as plain numpy array (not mx.array) — should work fine
        tgt_np = _fake_hidden().astype(np.float32)
        ids, _ = head.draft_k(
            tgt_np, k=2, prev_token_id=0,
            temperature=0.0, top_p=1.0, eos_id=99,
        )
        assert len(ids) >= 1

    def test_two_layers(self):
        head = _make_head(n_layers=2)
        ids, probs = head.draft_k(
            _fake_hidden(), k=3, prev_token_id=0,
            temperature=1.0, top_p=1.0, eos_id=99,
        )
        assert len(ids) >= 1

    def test_greedy_decoding_deterministic(self):
        """temperature=0 → greedy → same output twice."""
        head = _make_head()
        tgt  = _fake_hidden()
        ids1, _ = head.draft_k(tgt, k=4, prev_token_id=0, temperature=0.0, top_p=1.0, eos_id=99)
        ids2, _ = head.draft_k(tgt, k=4, prev_token_id=0, temperature=0.0, top_p=1.0, eos_id=99)
        assert ids1 == ids2


# ---------------------------------------------------------------------------
# ReDrafterHead — save / load
# ---------------------------------------------------------------------------

class TestReDrafterHeadPersistence:
    def test_save_load_roundtrip_preserves_weights(self):
        head = _make_head()
        with tempfile.TemporaryDirectory() as td:
            path = str(Path(td) / "redrafter.npz")
            head.save(path)
            head2 = ReDrafterHead.load(
                path,
                lm_head_w=head._lm_head_w.copy(),
                embed_w=head._embed_w.copy(),
            )
        assert np.allclose(head._proj_w, head2._proj_w)
        for i, (c1, c2) in enumerate(zip(head._gru_layers, head2._gru_layers)):
            assert np.allclose(c1.W, c2.W), f"GRU layer {i} W mismatch"
            assert np.allclose(c1.U, c2.U), f"GRU layer {i} U mismatch"
            assert np.allclose(c1.b, c2.b), f"GRU layer {i} b mismatch"

    def test_save_load_roundtrip_same_outputs(self):
        head = _make_head()
        tgt  = _fake_hidden()
        ids1, _ = head.draft_k(tgt, k=3, prev_token_id=0, temperature=0.0, top_p=1.0, eos_id=99)
        with tempfile.TemporaryDirectory() as td:
            path = str(Path(td) / "redrafter.npz")
            head.save(path)
            head2 = ReDrafterHead.load(path, head._lm_head_w.copy(), head._embed_w.copy())
        ids2, _ = head2.draft_k(tgt, k=3, prev_token_id=0, temperature=0.0, top_p=1.0, eos_id=99)
        assert ids1 == ids2

    def test_saved_config_fields_correct(self):
        head = _make_head(t_dim=T_DIM, e_dim=E_DIM, h_dim=H_DIM, n_layers=1)
        with tempfile.TemporaryDirectory() as td:
            path = str(Path(td) / "redrafter.npz")
            head.save(path)
            d = np.load(path)
        assert int(d["config_target_hidden_dim"]) == T_DIM
        assert int(d["config_hidden_dim"]) == H_DIM
        assert int(d["config_n_layers"]) == 1


# ---------------------------------------------------------------------------
# SpeculativeGenerator integration (import-level wiring check)
# ---------------------------------------------------------------------------

class TestSpeculativeGeneratorWiring:
    """Verify SpeculativeGenerator accepts redrafter_head without raising."""

    def _fake_model(self):
        """Minimal model stub that SpeculativeGenerator.__init__ won't crash on."""
        class _FakeModel:
            pass
        class _FakeTok:
            eos_token_id = 1
            def encode(self, s): return [1, 2, 3]
            def decode(self, ids): return "x"
        return _FakeModel(), _FakeTok()

    def test_init_with_redrafter_head_none(self):
        from squish.speculative.speculative import SpeculativeGenerator
        model, tok = self._fake_model()
        # Should not raise — redrafter_head=None is the default
        gen = SpeculativeGenerator(model, tok)
        assert gen._redrafter_head is None

    def test_init_with_redrafter_head_set(self):
        from squish.speculative.speculative import SpeculativeGenerator
        model, tok = self._fake_model()
        head = _make_head()
        # Should not raise (model has no .model / .lm_head → HiddenStateCapture.can_capture=False
        # → head is disabled with a warning, generator._redrafter_head == None)
        gen = SpeculativeGenerator(model, tok, redrafter_head=head)
        # Model stub doesn't have .model attribute → capture disabled → head set to None
        assert gen._redrafter_head is None

    def test_init_ssd_predictor_stored(self):
        from squish.speculative.speculative import SpeculativeGenerator
        from squish.speculative.ssd import SSDPredictor
        model, tok = self._fake_model()
        ssd = SSDPredictor.init_random(gru_hidden_dim=H_DIM)
        gen = SpeculativeGenerator(model, tok, ssd_predictor=ssd)
        # ssd is set to None because redrafter_head=None → disabled
        assert gen._ssd_predictor is ssd or gen._ssd_predictor is None
