"""
tests/kernels/test_dispatch.py

Unit tests for squish/kernels/dispatch.py

Phase 6 — Mojo Structured Kernel Registry

Coverage:
  • KernelBackend enum and AUTO order
  • KernelRegistry.register — normal, AUTO raises, overwrite
  • KernelRegistry.dispatch — explicit backend, AUTO priority, missing kernel
  • KernelRegistry.available_backends, list_kernels, is_registered
  • Module-level register_kernel + dispatch wrappers
  • Built-in NUMPY kernels (softmax, top_k, top_p, rep_penalty, int8_quantize, int8_dequantize)
  • INT8 round-trip accuracy
  • MLX fallback (not available in CI — registers dummy and verifies AUTO selection)
"""

from __future__ import annotations

import numpy as np
import pytest

from squish.kernels.dispatch import (
    _AUTO_ORDER,
    KernelBackend,
    KernelRegistry,
    dispatch,
    kernel_registry,
    register_kernel,
)

# ---------------------------------------------------------------------------
# KernelBackend enum
# ---------------------------------------------------------------------------

class TestKernelBackend:
    def test_values(self):
        assert KernelBackend.NUMPY.value == "numpy"
        assert KernelBackend.MLX.value   == "mlx"
        assert KernelBackend.MOJO.value  == "mojo"
        assert KernelBackend.AUTO.value  == "auto"

    def test_auto_order_priority(self):
        # MOJO > MLX > NUMPY
        assert _AUTO_ORDER[0] is KernelBackend.MOJO
        assert _AUTO_ORDER[1] is KernelBackend.MLX
        assert _AUTO_ORDER[2] is KernelBackend.NUMPY

    def test_auto_order_length(self):
        # AUTO itself is not in the order list
        assert KernelBackend.AUTO not in _AUTO_ORDER
        assert len(_AUTO_ORDER) == 3


# ---------------------------------------------------------------------------
# KernelRegistry — registration
# ---------------------------------------------------------------------------

class TestKernelRegistryRegister:
    def _fresh(self) -> KernelRegistry:
        return KernelRegistry()

    def test_register_and_dispatch_numpy(self):
        reg = self._fresh()

        @reg.register("add_one", KernelBackend.NUMPY)
        def _impl(x):
            return x + 1

        assert reg.dispatch("add_one", 5) == 6

    def test_register_auto_raises(self):
        reg = self._fresh()
        with pytest.raises(ValueError, match="AUTO"):

            @reg.register("bad", KernelBackend.AUTO)
            def _impl():  # pragma: no cover
                pass

    def test_register_returns_original_function(self):
        reg = self._fresh()

        def _impl(x):
            return x * 2

        result = reg.register("double", KernelBackend.NUMPY)(_impl)
        assert result is _impl  # transparent decorator

    def test_overwrite_registration(self):
        reg = self._fresh()

        @reg.register("fn", KernelBackend.NUMPY)
        def _v1():
            return "v1"

        @reg.register("fn", KernelBackend.NUMPY)
        def _v2():
            return "v2"

        assert reg.dispatch("fn") == "v2"

    def test_multiple_backends_registered_same_kernel(self):
        reg = self._fresh()

        @reg.register("kernel", KernelBackend.NUMPY)
        def _np():
            return "numpy"

        @reg.register("kernel", KernelBackend.MLX)
        def _mlx():
            return "mlx"

        # AUTO should prefer MLX over NUMPY
        assert reg.dispatch("kernel") == "mlx"


# ---------------------------------------------------------------------------
# KernelRegistry — dispatch
# ---------------------------------------------------------------------------

class TestKernelRegistryDispatch:
    def _fresh_with_numpy_fn(self, name: str, fn):
        reg = KernelRegistry()
        reg.register(name, KernelBackend.NUMPY)(fn)
        return reg

    def test_explicit_backend_calls_correct_impl(self):
        reg = KernelRegistry()
        reg.register("f", KernelBackend.NUMPY)(lambda: "numpy")
        reg.register("f", KernelBackend.MLX)(lambda: "mlx")
        assert reg.dispatch("f", backend=KernelBackend.NUMPY) == "numpy"
        assert reg.dispatch("f", backend=KernelBackend.MLX)   == "mlx"

    def test_explicit_missing_backend_raises_keyerror(self):
        reg = self._fresh_with_numpy_fn("g", lambda: None)
        with pytest.raises(KeyError, match="mojo"):
            reg.dispatch("g", backend=KernelBackend.MOJO)

    def test_unknown_kernel_raises_keyerror(self):
        reg = KernelRegistry()
        with pytest.raises(KeyError, match="does_not_exist"):
            reg.dispatch("does_not_exist")

    def test_auto_selects_highest_priority_available(self):
        reg = KernelRegistry()
        reg.register("h", KernelBackend.NUMPY)(lambda: "numpy")
        reg.register("h", KernelBackend.MLX)(lambda: "mlx")
        # MLX > NUMPY in AUTO order
        assert reg.dispatch("h", backend=KernelBackend.AUTO) == "mlx"

    def test_auto_falls_back_when_preferred_absent(self):
        reg = KernelRegistry()
        reg.register("h2", KernelBackend.NUMPY)(lambda: "numpy")
        # Only NUMPY registered; MOJO/MLX absent
        assert reg.dispatch("h2") == "numpy"

    def test_auto_no_backends_raises_keyerror(self):
        reg = KernelRegistry()
        # Register under a backend not in _AUTO_ORDER — impossible with current enum,
        # so we test the "no impls" path by patching internal state directly.
        reg._reg["ghost"] = {}
        with pytest.raises(KeyError):
            reg.dispatch("ghost")

    def test_kwargs_forwarded(self):
        reg = KernelRegistry()
        reg.register("kwf", KernelBackend.NUMPY)(lambda x, scale=1.0: x * scale)
        assert reg.dispatch("kwf", 3.0, scale=2.0) == 6.0

    def test_positional_args_forwarded(self):
        reg = KernelRegistry()
        reg.register("sum2", KernelBackend.NUMPY)(lambda a, b: a + b)
        assert reg.dispatch("sum2", 10, 7) == 17


# ---------------------------------------------------------------------------
# KernelRegistry — introspection
# ---------------------------------------------------------------------------

class TestKernelRegistryIntrospection:
    def test_available_backends_empty(self):
        reg = KernelRegistry()
        assert reg.available_backends("missing") == []

    def test_available_backends_sorted_by_priority(self):
        reg = KernelRegistry()
        reg.register("k", KernelBackend.NUMPY)(lambda: None)
        reg.register("k", KernelBackend.MLX)(lambda: None)
        backends = reg.available_backends("k")
        # MLX > NUMPY in priority
        assert backends == [KernelBackend.MLX, KernelBackend.NUMPY]

    def test_list_kernels_empty(self):
        reg = KernelRegistry()
        assert reg.list_kernels() == {}

    def test_list_kernels_returns_names_and_backends(self):
        reg = KernelRegistry()
        reg.register("alpha", KernelBackend.NUMPY)(lambda: None)
        reg.register("beta",  KernelBackend.NUMPY)(lambda: None)
        result = reg.list_kernels()
        assert "alpha" in result
        assert "beta"  in result
        assert "numpy" in result["alpha"]

    def test_list_kernels_sorted_alphabetically(self):
        reg = KernelRegistry()
        for name in ["zeta", "alpha", "mu"]:
            reg.register(name, KernelBackend.NUMPY)(lambda: None)
        keys = list(reg.list_kernels().keys())
        assert keys == sorted(keys)

    def test_is_registered_true(self):
        reg = KernelRegistry()
        reg.register("x", KernelBackend.NUMPY)(lambda: None)
        assert reg.is_registered("x") is True

    def test_is_registered_false_unknown(self):
        reg = KernelRegistry()
        assert reg.is_registered("unknown") is False

    def test_is_registered_specific_backend_true(self):
        reg = KernelRegistry()
        reg.register("x", KernelBackend.NUMPY)(lambda: None)
        assert reg.is_registered("x", KernelBackend.NUMPY) is True

    def test_is_registered_specific_backend_false(self):
        reg = KernelRegistry()
        reg.register("x", KernelBackend.NUMPY)(lambda: None)
        assert reg.is_registered("x", KernelBackend.MLX) is False

    def test_is_registered_auto_requires_at_least_one(self):
        reg = KernelRegistry()
        reg.register("y", KernelBackend.NUMPY)(lambda: None)
        assert reg.is_registered("y", KernelBackend.AUTO) is True


# ---------------------------------------------------------------------------
# Module-level wrappers
# ---------------------------------------------------------------------------

class TestModuleLevelWrappers:
    def test_register_kernel_is_usable_as_decorator(self):
        # Verify the module-level register_kernel/dispatch work on the global registry
        unique_name = "_test_register_kernel_wrapper_unique"
        # Clean up in case of re-run
        kernel_registry._reg.pop(unique_name, None)

        @register_kernel(unique_name, KernelBackend.NUMPY)
        def _impl(x):
            return x * 3

        assert dispatch(unique_name, 4) == 12
        # Cleanup
        del kernel_registry._reg[unique_name]

    def test_dispatch_auto_default(self):
        unique_name = "_test_dispatch_auto_default_unique"
        kernel_registry._reg.pop(unique_name, None)

        @register_kernel(unique_name, KernelBackend.NUMPY)
        def _impl():
            return "ok"

        assert dispatch(unique_name) == "ok"
        del kernel_registry._reg[unique_name]

    def test_dispatch_explicit_backend(self):
        unique_name = "_test_dispatch_explicit_unique"
        kernel_registry._reg.pop(unique_name, None)

        @register_kernel(unique_name, KernelBackend.NUMPY)
        def _impl():
            return "numpy"

        assert dispatch(unique_name, backend=KernelBackend.NUMPY) == "numpy"
        del kernel_registry._reg[unique_name]


# ---------------------------------------------------------------------------
# Built-in NUMPY kernel registrations
# ---------------------------------------------------------------------------

class TestBuiltinNumpyKernels:
    """Smoke-tests for the pre-registered NUMPY kernels."""

    def test_softmax_registered(self):
        assert kernel_registry.is_registered("softmax", KernelBackend.NUMPY)

    def test_top_k_registered(self):
        assert kernel_registry.is_registered("top_k", KernelBackend.NUMPY)

    def test_top_p_registered(self):
        assert kernel_registry.is_registered("top_p", KernelBackend.NUMPY)

    def test_rep_penalty_registered(self):
        assert kernel_registry.is_registered("rep_penalty", KernelBackend.NUMPY)

    def test_int8_quantize_registered(self):
        assert kernel_registry.is_registered("int8_quantize", KernelBackend.NUMPY)

    def test_int8_dequantize_registered(self):
        assert kernel_registry.is_registered("int8_dequantize", KernelBackend.NUMPY)

    # ── softmax ──────────────────────────────────────────────────────────────

    def test_softmax_sums_to_one(self):
        logits = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        probs = dispatch("softmax", logits)
        assert abs(probs.sum() - 1.0) < 1e-6

    def test_softmax_temperature_scaling(self):
        logits = np.array([1.0, 2.0], dtype=np.float32)
        p_cold = dispatch("softmax", logits, temperature=0.1)
        p_warm = dispatch("softmax", logits, temperature=2.0)
        # Cold temperature → sharper peak
        assert p_cold[1] > p_warm[1]

    def test_softmax_identity_temperature(self):
        logits = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        p1 = dispatch("softmax", logits, temperature=1.0)
        p2 = dispatch("softmax", logits)
        np.testing.assert_allclose(p1, p2, atol=1e-6)

    def test_softmax_dtype_float32(self):
        out = dispatch("softmax", np.array([0.0, 1.0]))
        assert out.dtype == np.float32

    # ── top_k ────────────────────────────────────────────────────────────────

    def test_top_k_masks_non_top(self):
        logits = np.array([1.0, 5.0, 2.0, 3.0], dtype=np.float32)
        out = dispatch("top_k", logits, 2)
        # Top-2: indices 1 (5.0) and 3 (3.0)
        assert out[1] == pytest.approx(5.0)
        assert out[3] == pytest.approx(3.0)
        assert out[0] == -np.inf
        assert out[2] == -np.inf

    # ── top_p ────────────────────────────────────────────────────────────────

    def test_top_p_sums_to_one(self):
        probs = np.array([0.6, 0.2, 0.1, 0.1], dtype=np.float32)
        out = dispatch("top_p", probs, 0.8)
        assert abs(out.sum() - 1.0) < 1e-6

    # ── rep_penalty ───────────────────────────────────────────────────────────

    def test_rep_penalty_divides_positive(self):
        logits = np.array([4.0, 2.0], dtype=np.float32)
        out = dispatch("rep_penalty", logits, [0], 2.0)
        assert out[0] == pytest.approx(2.0)
        assert out[1] == pytest.approx(2.0)  # unchanged

    # ── int8_quantize / int8_dequantize round-trip ────────────────────────────

    def test_int8_quantize_returns_two_arrays(self):
        x = np.random.default_rng(0).standard_normal(16).astype(np.float32)
        q, s = dispatch("int8_quantize", x)
        assert q.dtype == np.int8
        assert s.dtype == np.float32

    def test_int8_quantize_values_in_range(self):
        x = np.random.default_rng(1).standard_normal(32).astype(np.float32)
        q, _ = dispatch("int8_quantize", x)
        assert q.min() >= -127
        assert q.max() <= 127

    def test_int8_roundtrip_small_error(self):
        rng = np.random.default_rng(42)
        x = rng.standard_normal((4, 16)).astype(np.float32) * 2.0
        q, s = dispatch("int8_quantize", x)
        x_rec = dispatch("int8_dequantize", q, s)
        # Relative MSE should be small (< 2%)
        rel_mse = float(np.mean((x - x_rec) ** 2) / (np.mean(x ** 2) + 1e-8))
        assert rel_mse < 0.02

    def test_int8_quantize_scale_positive(self):
        x = np.array([[1.0, -2.0, 3.0]], dtype=np.float32)
        _, s = dispatch("int8_quantize", x)
        assert (s > 0).all()

    def test_int8_dequantize_dtype_float32(self):
        x = np.array([[1.0, -2.0]], dtype=np.float32)
        q, s = dispatch("int8_quantize", x)
        out = dispatch("int8_dequantize", q, s)
        assert out.dtype == np.float32

    def test_int8_dequantize_sign_preserved(self):
        x = np.array([[3.0, -3.0, 0.5]], dtype=np.float32)
        q, s = dispatch("int8_quantize", x)
        rec = dispatch("int8_dequantize", q, s)
        # Signs should match
        assert rec[0, 0] > 0
        assert rec[0, 1] < 0


# ---------------------------------------------------------------------------
# MLX dummy backend — verifies AUTO priority overrides NUMPY when registered
# ---------------------------------------------------------------------------

class TestMLXDummyPriority:
    def test_mlx_dummy_preferred_over_numpy(self):
        """Register a dummy MLX backend and verify AUTO selects it over NUMPY."""
        reg = KernelRegistry()
        reg.register("prio_test", KernelBackend.NUMPY)(lambda: "numpy")
        reg.register("prio_test", KernelBackend.MLX)(lambda: "mlx")
        assert reg.dispatch("prio_test") == "mlx"

    def test_mojo_dummy_preferred_over_mlx_and_numpy(self):
        reg = KernelRegistry()
        reg.register("prio2", KernelBackend.NUMPY)(lambda: "numpy")
        reg.register("prio2", KernelBackend.MLX)(lambda: "mlx")
        reg.register("prio2", KernelBackend.MOJO)(lambda: "mojo")
        assert reg.dispatch("prio2") == "mojo"


# ---------------------------------------------------------------------------
# Phase 3 — svd_score kernel
# ---------------------------------------------------------------------------

class TestSvdScoreKernel:
    def test_registered(self):
        assert kernel_registry.is_registered("svd_score", KernelBackend.NUMPY)

    def test_output_shape(self):
        rng = np.random.default_rng(0)
        n_heads, n_tokens, rank, head_dim, n_recent = 4, 16, 8, 32, 4

        kproj  = rng.standard_normal((n_heads, n_tokens, rank)).astype(np.float32)
        recent = rng.standard_normal((n_heads, n_recent, head_dim)).astype(np.float32)
        basis  = rng.standard_normal((n_heads, rank, head_dim)).astype(np.float32)

        scores = dispatch("svd_score", kproj, recent, basis)
        assert scores.shape == (n_tokens,)

    def test_output_dtype_float32(self):
        rng = np.random.default_rng(1)
        kproj  = rng.standard_normal((2, 8, 4)).astype(np.float32)
        recent = rng.standard_normal((2, 2, 16)).astype(np.float32)
        basis  = rng.standard_normal((2, 4, 16)).astype(np.float32)

        scores = dispatch("svd_score", kproj, recent, basis)
        assert scores.dtype == np.float32

    def test_scores_in_minus1_to_1(self):
        """Cosine similarities are bounded in [-1, 1]."""
        rng = np.random.default_rng(2)
        kproj  = rng.standard_normal((4, 32, 8)).astype(np.float32)
        recent = rng.standard_normal((4, 4, 16)).astype(np.float32)
        basis  = rng.standard_normal((4, 8, 16)).astype(np.float32)
        # Normalise basis rows so projections are unit-bounded
        b_norm = basis / (np.linalg.norm(basis, axis=-1, keepdims=True) + 1e-8)
        scores = dispatch("svd_score", kproj, recent, b_norm)
        assert scores.min() >= -1.1   # allow small float error
        assert scores.max() <= 1.1

    def test_identical_tokens_equal_scores(self):
        """All identical cached key projections should receive the same score."""
        rng = np.random.default_rng(3)
        n_heads, rank = 2, 4
        single = rng.standard_normal((n_heads, rank)).astype(np.float32)
        kproj  = np.stack([single] * 10, axis=1)             # (n_heads, 10, rank)
        recent = rng.standard_normal((n_heads, 3, 16)).astype(np.float32)
        basis  = rng.standard_normal((n_heads, rank, 16)).astype(np.float32)

        scores = dispatch("svd_score", kproj, recent, basis)
        assert scores.shape == (10,)
        np.testing.assert_allclose(scores, scores[0], atol=1e-5)

    def test_accepts_float16_inputs(self):
        rng = np.random.default_rng(4)
        kproj  = rng.standard_normal((2, 6, 4)).astype(np.float16)
        recent = rng.standard_normal((2, 2, 8)).astype(np.float16)
        basis  = rng.standard_normal((2, 4, 8)).astype(np.float16)
        scores = dispatch("svd_score", kproj, recent, basis)
        assert scores.dtype == np.float32


# ---------------------------------------------------------------------------
# Phase 4 — gru_step kernel
# ---------------------------------------------------------------------------

class TestGruStepKernel:
    def test_registered(self):
        assert kernel_registry.is_registered("gru_step", KernelBackend.NUMPY)

    def _make_gru_params(self, input_dim: int, hidden_dim: int, rng):
        W = rng.standard_normal((3 * hidden_dim, input_dim)).astype(np.float32) * 0.1
        U = rng.standard_normal((3 * hidden_dim, hidden_dim)).astype(np.float32) * 0.1
        b = np.zeros(3 * hidden_dim, dtype=np.float32)
        return W, U, b

    def test_output_shape(self):
        rng = np.random.default_rng(10)
        hd, id_ = 32, 16
        x, h = rng.standard_normal(id_).astype(np.float32), np.zeros(hd, dtype=np.float32)
        W, U, b = self._make_gru_params(id_, hd, rng)
        h_new = dispatch("gru_step", x, h, W, U, b)
        assert h_new.shape == (hd,)

    def test_output_dtype_float32(self):
        rng = np.random.default_rng(11)
        hd, id_ = 16, 8
        x, h = rng.standard_normal(id_).astype(np.float32), np.zeros(hd, dtype=np.float32)
        W, U, b = self._make_gru_params(id_, hd, rng)
        h_new = dispatch("gru_step", x, h, W, U, b)
        assert h_new.dtype == np.float32

    def test_zero_weights_output_zero(self):
        """With zero weights and bias, output should be 0 (h*(1-z) + z*tanh(0) = 0)."""
        hd, id_ = 8, 4
        x = np.zeros(id_, dtype=np.float32)
        h = np.zeros(hd, dtype=np.float32)
        W = np.zeros((3 * hd, id_), dtype=np.float32)
        U = np.zeros((3 * hd, hd), dtype=np.float32)
        b = np.zeros(3 * hd, dtype=np.float32)
        h_new = dispatch("gru_step", x, h, W, U, b)
        np.testing.assert_allclose(h_new, np.zeros(hd, dtype=np.float32), atol=1e-6)

    def test_output_bounded(self):
        """GRU hidden state should remain bounded due to tanh/sigmoid gates."""
        rng = np.random.default_rng(12)
        hd, id_ = 32, 16
        x = rng.standard_normal(id_).astype(np.float32) * 10.0
        h = rng.standard_normal(hd).astype(np.float32) * 10.0
        W, U, b = self._make_gru_params(id_, hd, rng)
        # Large scaling to ensure gates are tested at extremes
        W *= 10.0; U *= 10.0
        h_max = float(np.abs(h).max())
        h_new = dispatch("gru_step", x, h, W, U, b)
        assert np.isfinite(h_new).all()
        # h_new[i] = (1-z)*h[i] + z*n[i], z∈[0,1], n∈[-1,1]
        # so |h_new[i]| ≤ max(|h[i]|, 1) — bounded by prior hidden state
        assert float(np.abs(h_new).max()) <= max(h_max, 1.0) + 1e-4

    def test_deterministic(self):
        rng = np.random.default_rng(13)
        hd, id_ = 16, 8
        x, h = rng.standard_normal(id_).astype(np.float32), rng.standard_normal(hd).astype(np.float32)
        W, U, b = self._make_gru_params(id_, hd, rng)
        h1 = dispatch("gru_step", x, h, W, U, b)
        h2 = dispatch("gru_step", x, h, W, U, b)
        np.testing.assert_array_equal(h1, h2)


# ---------------------------------------------------------------------------
# Phase 5 — outer_product_update and fast_weight_query kernels
# ---------------------------------------------------------------------------

class TestOuterProductUpdateKernel:
    def test_registered(self):
        assert kernel_registry.is_registered("outer_product_update", KernelBackend.NUMPY)

    def test_output_shape(self):
        rng = np.random.default_rng(20)
        n_heads, hd, n_tok = 4, 16, 8
        W_f    = np.zeros((n_heads, hd, hd), dtype=np.float32)
        keys   = rng.standard_normal((n_heads, n_tok, hd)).astype(np.float32)
        values = rng.standard_normal((n_heads, n_tok, hd)).astype(np.float32)
        W_new  = dispatch("outer_product_update", W_f, keys, values, lr=0.01, decay=0.999)
        assert W_new.shape == (n_heads, hd, hd)

    def test_output_dtype_float32(self):
        rng = np.random.default_rng(21)
        W_f  = np.zeros((2, 8, 8), dtype=np.float32)
        k    = rng.standard_normal((2, 4, 8)).astype(np.float32)
        v    = rng.standard_normal((2, 4, 8)).astype(np.float32)
        W_new = dispatch("outer_product_update", W_f, k, v, lr=0.01, decay=1.0)
        assert W_new.dtype == np.float32

    def test_zero_lr_output_is_decayed(self):
        """lr=0 → W_new = decay * W_f (nothing added)."""
        rng    = np.random.default_rng(22)
        W_f    = rng.standard_normal((2, 4, 4)).astype(np.float32)
        k      = rng.standard_normal((2, 3, 4)).astype(np.float32)
        v      = rng.standard_normal((2, 3, 4)).astype(np.float32)
        W_new  = dispatch("outer_product_update", W_f, k, v, lr=0.0, decay=0.5)
        np.testing.assert_allclose(W_new, 0.5 * W_f, atol=1e-6)

    def test_decay_1_no_forgetting(self):
        """decay=1 → W_new contains W_f contribution unchanged."""
        W_f  = np.ones((1, 4, 4), dtype=np.float32)
        k    = np.zeros((1, 1, 4), dtype=np.float32)
        v    = np.zeros((1, 1, 4), dtype=np.float32)
        W_new = dispatch("outer_product_update", W_f, k, v, lr=0.01, decay=1.0)
        np.testing.assert_allclose(W_new, W_f, atol=1e-6)

    def test_positive_absorption_increases_norm(self):
        rng   = np.random.default_rng(23)
        W_f   = np.zeros((2, 8, 8), dtype=np.float32)
        k     = rng.standard_normal((2, 4, 8)).astype(np.float32)
        v     = rng.standard_normal((2, 4, 8)).astype(np.float32)
        W_new = dispatch("outer_product_update", W_f, k, v, lr=0.1, decay=1.0)
        assert np.linalg.norm(W_new) > 0.0

    def test_does_not_mutate_input(self):
        """The function must return a new array, not mutate W_f in place."""
        rng   = np.random.default_rng(24)
        W_f   = rng.standard_normal((2, 4, 4)).astype(np.float32)
        W_ref = W_f.copy()
        k     = rng.standard_normal((2, 2, 4)).astype(np.float32)
        v     = rng.standard_normal((2, 2, 4)).astype(np.float32)
        _     = dispatch("outer_product_update", W_f, k, v, lr=0.5, decay=0.9)
        np.testing.assert_array_equal(W_f, W_ref)


class TestFastWeightQueryKernel:
    def test_registered(self):
        assert kernel_registry.is_registered("fast_weight_query", KernelBackend.NUMPY)

    def test_output_shape(self):
        rng     = np.random.default_rng(30)
        n_heads, hd = 4, 16
        W_f     = rng.standard_normal((n_heads, hd, hd)).astype(np.float32)
        queries = rng.standard_normal((n_heads, hd)).astype(np.float32)
        out     = dispatch("fast_weight_query", W_f, queries)
        assert out.shape == (n_heads, hd)

    def test_output_dtype_float32(self):
        rng     = np.random.default_rng(31)
        W_f     = rng.standard_normal((2, 8, 8)).astype(np.float32)
        queries = rng.standard_normal((2, 8)).astype(np.float32)
        out     = dispatch("fast_weight_query", W_f, queries)
        assert out.dtype == np.float32

    def test_zero_wf_returns_zero(self):
        queries = np.ones((2, 4), dtype=np.float32)
        W_f     = np.zeros((2, 4, 4), dtype=np.float32)
        out     = dispatch("fast_weight_query", W_f, queries)
        np.testing.assert_allclose(out, np.zeros((2, 4), dtype=np.float32), atol=1e-8)

    def test_identity_wf_returns_query(self):
        """W_f = I → output should equal query."""
        n_heads, hd = 3, 8
        W_f     = np.stack([np.eye(hd, dtype=np.float32)] * n_heads, axis=0)
        queries = np.random.default_rng(32).standard_normal((n_heads, hd)).astype(np.float32)
        out     = dispatch("fast_weight_query", W_f, queries)
        np.testing.assert_allclose(out, queries, atol=1e-5)

    def test_linearity_in_queries(self):
        """fast_weight_query should be linear: W_f @ (α·q) = α · (W_f @ q)."""
        rng     = np.random.default_rng(33)
        W_f     = rng.standard_normal((2, 6, 6)).astype(np.float32) * 0.1
        q       = rng.standard_normal((2, 6)).astype(np.float32)
        alpha   = np.float32(3.14)
        out1    = dispatch("fast_weight_query", W_f, q) * alpha
        out2    = dispatch("fast_weight_query", W_f, (q * alpha))
        np.testing.assert_allclose(out1, out2, atol=1e-4)
