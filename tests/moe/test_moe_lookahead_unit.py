"""tests/moe/test_moe_lookahead_unit.py — 100 % coverage for squish/moe/moe_lookahead.py

All tests use the standard :mod:`unittest` library only; no pytest fixtures are
required (pytest can still be used as a test runner).
"""

from __future__ import annotations

import unittest

import numpy as np

from squish.moe.moe_lookahead import MoELookaheadConfig, MoELookaheadRouter

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

_N_EXPERTS  = 8
_TOP_K      = 2
_HIDDEN_DIM = 256
_BATCH      = 4

RNG = np.random.default_rng(42)


def _make_hidden(batch: int = _BATCH, hidden_dim: int = _HIDDEN_DIM) -> np.ndarray:
    """Return a float32 array of shape ``(batch, hidden_dim)``."""
    return RNG.standard_normal((batch, hidden_dim)).astype(np.float32)


def _default_router() -> MoELookaheadRouter:
    cfg = MoELookaheadConfig(
        n_experts=_N_EXPERTS,
        top_k=_TOP_K,
        hidden_dim=_HIDDEN_DIM,
    )
    return MoELookaheadRouter(cfg)


# ===========================================================================
# TestMoELookaheadConfig
# ===========================================================================

class TestMoELookaheadConfig(unittest.TestCase):

    def test_default_values(self):
        cfg = MoELookaheadConfig()
        self.assertEqual(cfg.n_experts, 8)
        self.assertEqual(cfg.top_k, 2)
        self.assertEqual(cfg.hidden_dim, 256)
        self.assertEqual(cfg.lookahead_steps, 3)
        self.assertAlmostEqual(cfg.ema_alpha, 0.1)
        self.assertAlmostEqual(cfg.load_balance_weight, 0.01)

    def test_valid_custom_config(self):
        cfg = MoELookaheadConfig(
            n_experts=16,
            top_k=4,
            hidden_dim=512,
            lookahead_steps=5,
            ema_alpha=0.3,
            load_balance_weight=0.001,
        )
        self.assertEqual(cfg.n_experts, 16)
        self.assertEqual(cfg.top_k, 4)
        self.assertEqual(cfg.hidden_dim, 512)
        self.assertEqual(cfg.lookahead_steps, 5)
        self.assertAlmostEqual(cfg.ema_alpha, 0.3)
        self.assertAlmostEqual(cfg.load_balance_weight, 0.001)

    def test_top_k_zero_raises(self):
        with self.assertRaises(ValueError):
            MoELookaheadConfig(top_k=0)

    def test_n_experts_less_than_top_k_raises(self):
        with self.assertRaises(ValueError):
            MoELookaheadConfig(n_experts=1, top_k=2)

    def test_hidden_dim_zero_raises(self):
        with self.assertRaises(ValueError):
            MoELookaheadConfig(hidden_dim=0)

    def test_lookahead_steps_zero_raises(self):
        with self.assertRaises(ValueError):
            MoELookaheadConfig(lookahead_steps=0)

    def test_ema_alpha_zero_raises(self):
        with self.assertRaises(ValueError):
            MoELookaheadConfig(ema_alpha=0.0)

    def test_ema_alpha_one_valid(self):
        # alpha = 1.0 is on the closed boundary — must NOT raise
        cfg = MoELookaheadConfig(ema_alpha=1.0)
        self.assertAlmostEqual(cfg.ema_alpha, 1.0)

    def test_ema_alpha_above_one_raises(self):
        with self.assertRaises(ValueError):
            MoELookaheadConfig(ema_alpha=1.0001)


# ===========================================================================
# TestMoELookaheadRouterInit
# ===========================================================================

class TestMoELookaheadRouterInit(unittest.TestCase):

    def test_router_created(self):
        router = _default_router()
        self.assertIsNotNone(router)
        # Underlying SparseMoERouter must exist
        self.assertIsNotNone(router._router)

    def test_ema_state_none_initially(self):
        router = _default_router()
        self.assertIsNone(router._prev_h)
        self.assertIsNone(router._ema_delta)
        self.assertIsNone(router._pending_prefetch)

    def test_metrics_zero_initially(self):
        router = _default_router()
        self.assertEqual(router._prefetch_hits, 0)
        self.assertEqual(router._prefetch_total, 0)
        self.assertEqual(router._prefetch_set_sum, 0)
        self.assertEqual(router._prefetch_calls, 0)


# ===========================================================================
# TestMoELookaheadRouterRoute
# ===========================================================================

class TestMoELookaheadRouterRoute(unittest.TestCase):

    def test_route_2d_input_shape(self):
        router = _default_router()
        h = _make_hidden(batch=4)
        indices, weights, aux_loss = router.route(h)
        self.assertEqual(indices.shape, (4, _TOP_K))
        self.assertEqual(weights.shape, (4, _TOP_K))
        self.assertIsInstance(float(aux_loss), float)

    def test_route_1d_input_shape(self):
        router = _default_router()
        h = _make_hidden(batch=1).squeeze(0)  # (hidden_dim,)
        self.assertEqual(h.ndim, 1)
        indices, weights, aux_loss = router.route(h)
        # The underlying router sees a (1, hidden_dim) input and returns
        # (1, top_k); those arrays are returned as-is for 1-D input.
        self.assertEqual(indices.shape, (1, _TOP_K))
        self.assertEqual(weights.shape, (1, _TOP_K))

    def test_route_updates_prev_h(self):
        router = _default_router()
        self.assertIsNone(router._prev_h)
        router.route(_make_hidden())
        self.assertIsNotNone(router._prev_h)
        self.assertEqual(router._prev_h.shape, (_HIDDEN_DIM,))

    def test_route_updates_ema_delta_on_second_call(self):
        router = _default_router()
        router.route(_make_hidden())
        prev_delta = router._ema_delta.copy()
        router.route(_make_hidden())  # second call, new delta direction
        # _ema_delta should have been updated (non-zero now unless by chance)
        self.assertIsNotNone(router._ema_delta)
        # Shape must be preserved
        self.assertEqual(router._ema_delta.shape, (_HIDDEN_DIM,))

    def test_ema_delta_zero_on_first_call(self):
        """After the very first route() call _ema_delta must be all zeros."""
        router = _default_router()
        router.route(_make_hidden())
        self.assertIsNotNone(router._ema_delta)
        np.testing.assert_array_equal(
            router._ema_delta,
            np.zeros(_HIDDEN_DIM, dtype=np.float32),
        )

    def test_route_evaluates_pending_prefetch(self):
        """If _pending_prefetch is set, route() must update hit counters."""
        router = _default_router()
        h = _make_hidden()
        # Manually set a prefetch that contains ALL possible experts so we
        # are guaranteed at least some hits.
        router._pending_prefetch = frozenset(range(_N_EXPERTS))
        router.route(h)
        # All actual experts are inside the all-expert set → hits == total
        self.assertGreater(router._prefetch_total, 0)
        self.assertEqual(router._prefetch_hits, router._prefetch_total)
        # Pending prefetch must be cleared after evaluation
        self.assertIsNone(router._pending_prefetch)

    def test_route_no_pending_prefetch_no_crash(self):
        """route() must not raise when _pending_prefetch is None."""
        router = _default_router()
        self.assertIsNone(router._pending_prefetch)
        # Should complete without error
        router.route(_make_hidden())
        self.assertEqual(router._prefetch_total, 0)


# ===========================================================================
# TestMoELookaheadRouterPredictLookahead
# ===========================================================================

class TestMoELookaheadRouterPredictLookahead(unittest.TestCase):

    def test_predict_returns_correct_shape(self):
        router = _default_router()
        h = _make_hidden(batch=4)
        result = router.predict_lookahead(h, steps=3)
        self.assertEqual(result.shape, (3, 4, _TOP_K))
        self.assertEqual(result.dtype, np.int64)

    def test_predict_default_steps(self):
        cfg = MoELookaheadConfig(lookahead_steps=5)
        router = MoELookaheadRouter(cfg)
        h = _make_hidden()
        result = router.predict_lookahead(h)  # steps=None → uses config default
        self.assertEqual(result.shape[0], 5)

    def test_predict_custom_steps(self):
        router = _default_router()
        h = _make_hidden(batch=2)
        result = router.predict_lookahead(h, steps=7)
        self.assertEqual(result.shape, (7, 2, _TOP_K))

    def test_predict_does_not_update_ema(self):
        """predict_lookahead() must leave _prev_h and _ema_delta unchanged."""
        router = _default_router()
        h = _make_hidden()
        # Establish a non-None EMA state via a real route() call
        router.route(h)
        prev_h_before    = router._prev_h.copy()
        ema_delta_before = router._ema_delta.copy()

        router.predict_lookahead(h, steps=3)

        np.testing.assert_array_equal(router._prev_h, prev_h_before)
        np.testing.assert_array_equal(router._ema_delta, ema_delta_before)

    def test_predict_does_not_update_pending_prefetch(self):
        """predict_lookahead() must not alter _pending_prefetch."""
        router = _default_router()
        h = _make_hidden()
        self.assertIsNone(router._pending_prefetch)
        router.predict_lookahead(h, steps=2)
        self.assertIsNone(router._pending_prefetch)

    def test_predict_no_ema_state(self):
        """When _ema_delta is None (fresh router), zeros delta is used."""
        router = _default_router()
        self.assertIsNone(router._ema_delta)
        h = _make_hidden(batch=2)
        result = router.predict_lookahead(h, steps=2)
        # Shape should still be correct
        self.assertEqual(result.shape, (2, 2, _TOP_K))

    def test_predict_1d_input(self):
        router = _default_router()
        h_1d = _make_hidden(batch=1).squeeze(0)
        result = router.predict_lookahead(h_1d, steps=2)
        # 1-D input is broadcast to batch=1 internally
        self.assertEqual(result.shape, (2, 1, _TOP_K))


# ===========================================================================
# TestMoELookaheadRouterPrefetchSet
# ===========================================================================

class TestMoELookaheadRouterPrefetchSet(unittest.TestCase):

    def test_returns_frozenset(self):
        router = _default_router()
        pset = router.prefetch_set(_make_hidden())
        self.assertIsInstance(pset, frozenset)

    def test_returns_expert_indices_in_range(self):
        router = _default_router()
        pset = router.prefetch_set(_make_hidden(), steps=4)
        for idx in pset:
            self.assertGreaterEqual(idx, 0)
            self.assertLess(idx, _N_EXPERTS)

    def test_updates_pending_prefetch(self):
        router = _default_router()
        pset = router.prefetch_set(_make_hidden())
        self.assertIsNotNone(router._pending_prefetch)
        self.assertEqual(router._pending_prefetch, pset)

    def test_increments_prefetch_calls(self):
        router = _default_router()
        self.assertEqual(router._prefetch_calls, 0)
        router.prefetch_set(_make_hidden())
        self.assertEqual(router._prefetch_calls, 1)
        router.prefetch_set(_make_hidden())
        self.assertEqual(router._prefetch_calls, 2)

    def test_increments_prefetch_set_sum(self):
        router = _default_router()
        pset = router.prefetch_set(_make_hidden())
        self.assertEqual(router._prefetch_set_sum, len(pset))
        pset2 = router.prefetch_set(_make_hidden())
        self.assertEqual(router._prefetch_set_sum, len(pset) + len(pset2))


# ===========================================================================
# TestMoELookaheadRouterReset
# ===========================================================================

class TestMoELookaheadRouterReset(unittest.TestCase):

    def _primed_router(self) -> MoELookaheadRouter:
        router = _default_router()
        h = _make_hidden()
        router.route(h)
        router.prefetch_set(h)
        return router

    def test_reset_clears_ema_state(self):
        router = self._primed_router()
        self.assertIsNotNone(router._prev_h)
        self.assertIsNotNone(router._ema_delta)
        router.reset()
        self.assertIsNone(router._prev_h)
        self.assertIsNone(router._ema_delta)

    def test_reset_clears_pending_prefetch(self):
        router = self._primed_router()
        self.assertIsNotNone(router._pending_prefetch)
        router.reset()
        self.assertIsNone(router._pending_prefetch)

    def test_reset_preserves_metrics(self):
        """Hit-rate and prefetch-size counters must survive a reset()."""
        router = _default_router()
        h = _make_hidden()
        router.prefetch_set(h)   # _prefetch_calls becomes 1
        router.route(h)          # evaluates hit rate

        hits    = router._prefetch_hits
        total   = router._prefetch_total
        set_sum = router._prefetch_set_sum
        calls   = router._prefetch_calls

        router.reset()

        self.assertEqual(router._prefetch_hits,    hits)
        self.assertEqual(router._prefetch_total,   total)
        self.assertEqual(router._prefetch_set_sum, set_sum)
        self.assertEqual(router._prefetch_calls,   calls)

    def test_reset_preserves_route_stats(self):
        """total_route_calls and total_tokens must not be touched by reset()."""
        router = _default_router()
        router.route(_make_hidden())
        router.route(_make_hidden())
        route_calls_before = router._router.stats.total_route_calls
        router.reset()
        self.assertEqual(router._router.stats.total_route_calls, route_calls_before)


# ===========================================================================
# TestMoELookaheadRouterStats
# ===========================================================================

class TestMoELookaheadRouterStats(unittest.TestCase):

    def test_stats_keys(self):
        router = _default_router()
        s = router.stats()
        expected_keys = {
            "hit_rate",
            "prefetch_hits",
            "prefetch_total",
            "avg_prefetch_set_size",
            "prefetch_calls",
            "total_route_calls",
            "total_tokens",
        }
        self.assertEqual(set(s.keys()), expected_keys)

    def test_stats_hit_rate_minus_one_when_no_data(self):
        router = _default_router()
        s = router.stats()
        self.assertAlmostEqual(s["hit_rate"], -1.0)

    def test_stats_hit_rate_computation(self):
        """Manually set counters and verify the computed ratio."""
        router = _default_router()
        router._prefetch_hits  = 3
        router._prefetch_total = 4
        s = router.stats()
        self.assertAlmostEqual(s["hit_rate"], 0.75)

    def test_stats_avg_prefetch_set_size_zero_when_no_calls(self):
        router = _default_router()
        s = router.stats()
        self.assertAlmostEqual(s["avg_prefetch_set_size"], 0.0)

    def test_stats_avg_prefetch_set_size_computation(self):
        router = _default_router()
        router._prefetch_set_sum = 12
        router._prefetch_calls   = 4
        s = router.stats()
        self.assertAlmostEqual(s["avg_prefetch_set_size"], 3.0)

    def test_stats_total_route_calls_from_router(self):
        router = _default_router()
        router.route(_make_hidden())
        router.route(_make_hidden())
        s = router.stats()
        # Each route() call delegates once to the underlying router
        self.assertEqual(s["total_route_calls"], 2)

    def test_stats_total_tokens(self):
        router = _default_router()
        router.route(_make_hidden(batch=3))
        router.route(_make_hidden(batch=5))
        s = router.stats()
        self.assertEqual(s["total_tokens"], 8)


# ===========================================================================
# TestMoELookaheadHitRateIntegration
# ===========================================================================

class TestMoELookaheadHitRateIntegration(unittest.TestCase):

    def test_hit_rate_after_round_trip(self):
        """prefetch_set() followed by route() must produce a non-negative hit rate."""
        router = _default_router()
        h = _make_hidden()

        # Build a prefetch set, then route the same input
        router.prefetch_set(h, steps=3)
        router.route(h)

        s = router.stats()
        # hit_rate must now be a real measurement (not the sentinel -1.0)
        self.assertGreaterEqual(s["hit_rate"], 0.0)
        self.assertLessEqual(s["hit_rate"], 1.0)

    def test_prefetch_set_is_subset_of_all_experts(self):
        """Every expert index in a prefetch set must be a valid expert id."""
        router = _default_router()
        pset = router.prefetch_set(_make_hidden(), steps=5)
        for idx in pset:
            self.assertGreaterEqual(idx, 0)
            self.assertLess(idx, _N_EXPERTS)

    def test_multiple_round_trips_accumulate_stats(self):
        """Several prefetch+route cycles must accumulate totals correctly."""
        router = _default_router()
        n_rounds = 5
        for _ in range(n_rounds):
            h = _make_hidden()
            router.prefetch_set(h)
            router.route(h)

        s = router.stats()
        self.assertEqual(s["prefetch_calls"], n_rounds)
        self.assertGreaterEqual(s["prefetch_total"], n_rounds)

    def test_hit_rate_all_experts_prefetched(self):
        """Prefetching all experts yields a perfect hit rate of 1.0."""
        router = _default_router()
        h = _make_hidden()
        # Override the pending prefetch with the universe of all expert ids
        router._pending_prefetch = frozenset(range(_N_EXPERTS))
        router.route(h)
        s = router.stats()
        self.assertAlmostEqual(s["hit_rate"], 1.0)

    def test_ema_evolves_across_multiple_route_calls(self):
        """_ema_delta should be non-zero after two route() calls on different inputs."""
        router = _default_router()
        router.route(_make_hidden())
        router.route(_make_hidden())
        # After two calls with different hidden states the EMA delta should
        # have absorbed a non-trivial direction (practically never all-zeros).
        self.assertIsNotNone(router._ema_delta)
        self.assertFalse(np.all(router._ema_delta == 0.0))


if __name__ == "__main__":
    unittest.main()
