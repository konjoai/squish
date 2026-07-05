"""tests/test_wave145_calibration_accuracy_gate.py

Wave 145 — accuracy validation gate: compare
collect_activation_scales_streaming's AWQ scale vectors directly against
the existing full-load collect_activation_scales on a real model small
enough to run both ways, before trusting the streaming path on any model
that can't be full-loaded (where there's no ground truth to compare
against).

This is deliberately NOT a synthetic-weights test like Waves 142-144's —
those pin the calibration *mechanism* (construction, shard bookkeeping,
deletion timing) and pass with float32 test fixtures regardless of
whether the reconstructed forward pass is numerically faithful to the
real model. Only a real bf16 checkpoint run through mlx_lm.load() can
catch a real fidelity bug — which is exactly what happened while building
this test: streaming calibration was passing `mask=None` unconditionally
to each standalone block, meaning no causal masking at all (every token
attending to every position, including future ones). Correlation against
the full-load ground truth was as low as 0.75 on some layers with that
bug present. The fix (`create_attention_mask(h)`, matching exactly what
the full-load model applies per layer) brought every one of 112 compared
layers to >=0.9999 correlation. Synthetic-weights tests alone would never
have caught this, since they have no independent ground truth to diverge
from — this is why the brief calls for validating against the real
full-load path specifically, not just testing the streaming path in
isolation.

Downloads a small real model (~2.5 GB) and is comparatively slow (~30-60s
per calibration pass), so it's skipped by default — run explicitly with:

    SQUISH_RUN_ACCURACY_GATE=1 pytest tests/test_wave145_calibration_accuracy_gate.py -v

Tolerance: mean correlation >= 0.99 and max relative error <= 0.05 across
all compared layers — chosen well above what floating-point/RNG
sample-selection noise alone would produce (observed post-fix: mean
correlation 1.0000, max relative error 0.0045), so a regression that
reintroduces meaningful divergence (e.g. dropping the causal mask again,
or a family with a subtly different forward-pass signature) fails loudly
rather than silently.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

pytest.importorskip(
    "mlx_lm.models.llama",
    reason="mlx_lm blocked in sandbox — set CI=1 to enable",
    exc_type=ImportError,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("SQUISH_RUN_ACCURACY_GATE") != "1",
    reason="Downloads a real ~2.5GB model — run explicitly with SQUISH_RUN_ACCURACY_GATE=1",
)

MEAN_CORRELATION_MIN = 0.99
MAX_RELATIVE_ERROR_MAX = 0.05
MODEL_REPO = "mlx-community/Llama-3.2-1B-Instruct-bf16"


@pytest.fixture(scope="module")
def real_model_dir(tmp_path_factory):
    from huggingface_hub import snapshot_download

    d = tmp_path_factory.mktemp("wave145_real_model")
    snapshot_download(MODEL_REPO, local_dir=str(d))
    return d


class TestStreamingMatchesFullLoadOnRealModel:
    def test_scale_vectors_correlate_with_full_load_ground_truth(self, real_model_dir):
        import numpy as np
        from mlx_lm import load

        from squish.quant.awq import collect_activation_scales
        from squish.quant.awq_streaming import collect_activation_scales_streaming
        from squish.quant.shard_index import load_shard_index

        model, tokenizer = load(str(real_model_dir))
        full_scales = collect_activation_scales(
            model, tokenizer, n_samples=16, seq_len=64, verbose=False
        )

        idx = load_shard_index(real_model_dir)
        stream_scales = collect_activation_scales_streaming(
            real_model_dir, tokenizer, idx, n_samples=16, seq_len=64, verbose=False
        )

        assert stream_scales is not None
        assert set(full_scales.keys()) == set(stream_scales.keys())

        correlations = []
        relative_errors = []
        for key in full_scales:
            a, b = full_scales[key], stream_scales[key]
            assert a.shape == b.shape, f"{key}: shape mismatch {a.shape} vs {b.shape}"
            correlations.append(np.corrcoef(a, b)[0, 1])
            relative_errors.append(np.abs(a - b).mean() / (np.abs(a).mean() + 1e-8))

        correlations = np.array(correlations)
        relative_errors = np.array(relative_errors)

        assert correlations.mean() >= MEAN_CORRELATION_MIN, (
            f"mean correlation {correlations.mean():.4f} below gate "
            f"{MEAN_CORRELATION_MIN} — streaming calibration has diverged "
            f"from the full-load ground truth (worst layer: "
            f"{correlations.min():.4f})"
        )
        assert relative_errors.max() <= MAX_RELATIVE_ERROR_MAX, (
            f"max relative error {relative_errors.max():.4f} exceeds gate {MAX_RELATIVE_ERROR_MAX}"
        )


class TestStreamingProducesEquivalentQuantizedWeights:
    """Downstream check: do the two calibration paths' scales actually
    produce equivalent quantized models, not just similar-looking scale
    vectors in isolation? Compares real INT4-quantized tensor values
    directly — a more direct, less noisy signal than an lm_eval accuracy
    proxy would be, since accuracy benchmarks add their own sampling
    noise on top of whatever the quantization itself introduces."""

    INT_DIFF_FRACTION_MAX = 0.10  # per-tensor fraction of INT4 values allowed to differ
    FLOAT_RELATIVE_ERROR_MAX = 0.01

    def test_int4_quantized_weights_nearly_identical(self, real_model_dir, tmp_path):
        import numpy as np
        from mlx_lm import load

        from squish.convert import process_weights_streaming
        from squish.quant.awq import (
            collect_activation_scales,
            load_awq_scales,
            save_awq_scales,
        )
        from squish.quant.awq_streaming import collect_activation_scales_streaming
        from squish.quant.shard_index import load_shard_index

        model, tokenizer = load(str(real_model_dir))
        full_scales = collect_activation_scales(
            model, tokenizer, n_samples=16, seq_len=64, verbose=False
        )
        idx = load_shard_index(real_model_dir)
        stream_scales = collect_activation_scales_streaming(
            real_model_dir, tokenizer, idx, n_samples=16, seq_len=64, verbose=False
        )
        del model  # free before quantizing — matches the real workflow's intent

        full_scales_dir = tmp_path / "full_awq_scales"
        stream_scales_dir = tmp_path / "stream_awq_scales"
        save_awq_scales(full_scales, full_scales_dir, verbose=False)
        save_awq_scales(stream_scales, stream_scales_dir, verbose=False)

        out_full = tmp_path / "out_full"
        out_stream = tmp_path / "out_stream"
        process_weights_streaming(
            real_model_dir,
            out_full,
            [],
            20.0,
            False,
            awq_scales=load_awq_scales(full_scales_dir),
            use_int4=True,
            min_free_gb=0.0,
        )
        process_weights_streaming(
            real_model_dir,
            out_stream,
            [],
            20.0,
            False,
            awq_scales=load_awq_scales(stream_scales_dir),
            use_int4=True,
            min_free_gb=0.0,
        )

        full_files = sorted(p.name for p in (out_full / "tensors").glob("*.npy"))
        stream_files = sorted(p.name for p in (out_stream / "tensors").glob("*.npy"))
        assert full_files == stream_files

        worst_int_diff_fraction = 0.0
        worst_float_rel_error = 0.0
        for name in full_files:
            a = np.load(out_full / "tensors" / name)
            b = np.load(out_stream / "tensors" / name)
            assert a.shape == b.shape, f"{name}: shape mismatch"
            if a.dtype.kind in "iu":
                frac = np.sum(a != b) / a.size
                worst_int_diff_fraction = max(worst_int_diff_fraction, frac)
            else:
                rel = np.abs(a.astype(np.float64) - b.astype(np.float64)).mean() / (
                    np.abs(a).mean() + 1e-8
                )
                worst_float_rel_error = max(worst_float_rel_error, rel)

        assert worst_int_diff_fraction <= self.INT_DIFF_FRACTION_MAX, (
            f"worst INT4 tensor has {worst_int_diff_fraction:.3%} differing values "
            f"between the two calibration paths — exceeds {self.INT_DIFF_FRACTION_MAX:.0%}"
        )
        assert worst_float_rel_error <= self.FLOAT_RELATIVE_ERROR_MAX, (
            f"worst float/passthrough tensor relative error {worst_float_rel_error:.4f} "
            f"exceeds {self.FLOAT_RELATIVE_ERROR_MAX}"
        )
