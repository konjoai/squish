# squish/kernels/mojo/kernels/ternary_gemv.mojo
# Ternary weight GEMV: INT8 {-1, 0, +1} weight × FP32 activation.
#
# For each output row, accumulates activations via SIMD signed-integer
# comparison — no FP multiply, only FP add/subtract for non-zero weights.
# Parallelises over output rows.
#
# Reference: Ma et al., "The Era of 1-bit LLMs." arXiv 2402.17764.

from algorithm import parallelize, vectorize


fn ternary_gemv_kernel(
    w_ptr: UnsafePointer[Int8],       # (out_features * in_features,) INT8 {-1,0,+1}
    a_ptr: UnsafePointer[Float32],    # (in_features,) FP32 activations
    out_ptr: UnsafePointer[Float32],  # (out_features,) FP32 output
    out_features: Int,
    in_features: Int,
    scale: Float32,
):
    alias SIMD_W = 8  # process 8 elements per FP32 SIMD iteration

    @parameter
    fn compute_row(row: Int):
        var acc = Float32(0.0)
        var row_off = row * in_features

        for i in range(in_features):
            var w_val = w_ptr[row_off + i]
            if w_val == 1:
                acc += a_ptr[i]
            elif w_val == -1:
                acc -= a_ptr[i]
            # skip zero weights

        out_ptr[row] = acc * scale

    parallelize[compute_row](out_features)
