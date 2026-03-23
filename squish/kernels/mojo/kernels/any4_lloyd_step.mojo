# squish/kernels/mojo/kernels/any4_lloyd_step.mojo
# Any4 Lloyd k-means: parallel E-step + sequential M-step.
#
# Runs ``n_iter`` EM iterations:
#   E-step: assign each value to its nearest centroid (parallel, SIMD)
#   M-step: recompute centroid means (sequential, centroid count is tiny ≤16)
#
# Reference: Tseng et al., "QuIP#: Even Better LLM Quantization with
# Hadamard Incoherence and Lattice Codebooks," ICML 2024.

from algorithm import parallelize, vectorize


fn any4_lloyd_step_kernel(
    values_ptr:    UnsafePointer[Float32],  # (N,) float32 weight elements
    centroids_ptr: UnsafePointer[Float32],  # (k,) float32 initial centroids (modified in-place)
    out_ptr:       UnsafePointer[Float32],  # (k,) float32 final centroids
    N: Int,
    k: Int,
    n_iter: Int,
):
    """Lloyd k-means centroid refinement.

    E-step parallelised over ``N`` values; M-step sequential over ≤16
    centroids (centroid count is too small to benefit from parallelism).
    """
    alias SIMD_W = 8

    # Copy initial centroids to output buffer (we update out_ptr in-place).
    for i in range(k):
        out_ptr[i] = centroids_ptr[i]

    # Allocate assignment and accumulator arrays on the heap.
    var assign  = UnsafePointer[Int32].alloc(N)
    var sums    = UnsafePointer[Float32].alloc(k)
    var counts  = UnsafePointer[Int32].alloc(k)

    for _iter in range(n_iter):
        # --- E-step: assign each value to nearest centroid ---
        @parameter
        fn assign_value(i: Int):
            var v = values_ptr[i]
            var best_c = 0
            var best_d = (v - out_ptr[0]) * (v - out_ptr[0])
            for c in range(1, k):
                var d = (v - out_ptr[c]) * (v - out_ptr[c])
                if d < best_d:
                    best_d = d
                    best_c = c
            assign[i] = Int32(best_c)

        parallelize[assign_value](N)

        # --- M-step: compute new centroids ---
        for c in range(k):
            sums[c]   = Float32(0.0)
            counts[c] = Int32(0)
        for i in range(N):
            var c = Int(assign[i])
            sums[c]   = sums[c] + values_ptr[i]
            counts[c] = counts[c] + Int32(1)
        for c in range(k):
            if counts[c] > 0:
                out_ptr[c] = sums[c] / Float32(Int(counts[c]))
            # else: leave centroid unchanged (dead centroid re-init not done here)

    assign.free()
    sums.free()
    counts.free()
