# squish/kernels/mojo/kernels/aqlm_encode.mojo
# AQLM greedy multi-codebook nearest-lookup + residual subtract kernel.
#
# For each output feature (row), iterates over n_groups groups and finds the
# nearest codebook entry by L2 distance.  Records the index and subtracts the
# selected codeword from the residual (sequential codebook peeling).
# Parallelised over out_feat rows with ``parallelize``.
#
# Reference: Egiazarian et al., "Extreme Compression of Large Language Models
# via Additive Quantization," arXiv 2401.06118, 2024.

from algorithm import parallelize, vectorize
from math import sqrt


fn aqlm_encode_kernel(
    residuals_ptr:  UnsafePointer[Float32],  # (out_feat * n_groups * gs) flat
    codebook_ptr:   UnsafePointer[Float32],  # (cb_size * gs) flat
    indices_ptr:    UnsafePointer[UInt16],   # (out_feat * n_groups) output
    updated_res:    UnsafePointer[Float32],  # (out_feat * n_groups * gs) output
    out_feat:       Int,
    n_groups:       Int,
    gs:             Int,
    cb_size:        Int,
):
    """Greedy nearest-codebook-entry lookup and residual subtraction.

    For each of the ``out_feat`` rows, iterates over ``n_groups`` groups of
    size ``gs`` and computes the argmin L2 distance to all ``cb_size`` codebook
    entries.  Writes the winning index to ``indices_ptr`` and subtracts the
    selected codeword from ``updated_res``.

    Parallelises over the ``out_feat`` outer loop.
    """
    # Copy residuals into output buffer
    var total = out_feat * n_groups * gs
    for i in range(total):
        updated_res[i] = residuals_ptr[i]

    @parameter
    fn process_row(row: Int):
        for g in range(n_groups):
            var base = row * n_groups * gs + g * gs
            var best_dist = Float32(1e38)
            var best_ci = 0
            for ci in range(cb_size):
                var dist = Float32(0.0)
                for k in range(gs):
                    var diff = updated_res[base + k] - codebook_ptr[ci * gs + k]
                    dist += diff * diff
                if dist < best_dist:
                    best_dist = dist
                    best_ci = ci
            indices_ptr[row * n_groups + g] = UInt16(best_ci)
            for k in range(gs):
                updated_res[base + k] -= codebook_ptr[best_ci * gs + k]

    parallelize[process_row](out_feat)


fn aqlm_kmeans_kernel(
    vecs_ptr:       UnsafePointer[Float32],  # (n * gs) flat
    centroids_ptr:  UnsafePointer[Float32],  # (k * gs) output
    n:              Int,
    gs:             Int,
    k:              Int,
    n_iter:         Int,
    seed:           Int,
):
    """K-means++ init + Lloyd clustering for AQLM codebook construction.

    Initialises ``k`` centroids from evenly-spaced sample of ``vecs``, then
    runs ``n_iter`` Lloyd iterations with parallel E-step assignment and
    sequential M-step centroid update.
    """
    # Initialise centroids from evenly-spaced samples
    for ci in range(k):
        var idx = (ci * n // k) if k <= n else ci % n
        for d in range(gs):
            centroids_ptr[ci * gs + d] = vecs_ptr[idx * gs + d]

    var assignments = UnsafePointer[Int32].alloc(n)

    for _iter in range(n_iter):
        # E-step: assign each vector to nearest centroid
        @parameter
        fn assign_vec(i: Int):
            var best_dist = Float32(1e38)
            var best_ci = 0
            for ci in range(k):
                var dist = Float32(0.0)
                for d in range(gs):
                    var diff = vecs_ptr[i * gs + d] - centroids_ptr[ci * gs + d]
                    dist += diff * diff
                if dist < best_dist:
                    best_dist = dist
                    best_ci = ci
            assignments[i] = Int32(best_ci)

        parallelize[assign_vec](n)

        # M-step: update centroids (sequential scatter-reduce)
        for ci in range(k):
            var sum_buf = UnsafePointer[Float64].alloc(gs)
            for d in range(gs):
                sum_buf[d] = Float64(0.0)
            var cnt = Int64(0)
            for i in range(n):
                if Int(assignments[i]) == ci:
                    for d in range(gs):
                        sum_buf[d] += Float64(vecs_ptr[i * gs + d])
                    cnt += 1
            if cnt > 0:
                for d in range(gs):
                    centroids_ptr[ci * gs + d] = Float32(sum_buf[d] / Float64(cnt))
            sum_buf.free()

    assignments.free()
