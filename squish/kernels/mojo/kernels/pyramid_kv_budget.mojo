# squish/kernels/mojo/kernels/pyramid_kv_budget.mojo
# PyramidKV: parallel per-layer KV-cache budget computation.
#
# Computes a linearly-decaying token budget per transformer layer:
#
#   budget[l] = max(min_budget, round(base * (1 − alpha * l / (L − 1))))
#
# Parallelised over layers with Rayon-style ``parallelize``.
#
# Reference: Cai et al., "PyramidKV: Dynamic KV Cache Compression based on
# Pyramidal Information Funneling," arXiv 2406.02069, 2024.

from algorithm import parallelize, vectorize


fn pyramid_kv_budget_kernel(
    out_ptr:    UnsafePointer[Int32],  # (n_layers,) output budgets
    base:       Float32,
    alpha:      Float32,
    n_layers:   Int,
    min_budget: Int,
):
    """Compute per-layer linearly-decaying KV-cache budgets.

    Each layer independently computes:
        frac = l / max(1, n_layers - 1)
        val  = base * (1.0 - alpha * frac)
        budget[l] = max(min_budget, round(val))

    Parallelises over ``n_layers`` with ``parallelize``.
    """

    @parameter
    fn compute_layer(l: Int):
        var frac = Float32(l) / Float32(n_layers - 1) if n_layers > 1 else Float32(0.0)
        var val  = base * (Float32(1.0) - alpha * frac)
        var rounded = Int(val + Float32(0.5))  # round half-up
        out_ptr[l] = Int32(rounded) if rounded >= min_budget else Int32(min_budget)

    parallelize[compute_layer](n_layers)
