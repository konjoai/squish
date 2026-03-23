# squish/kernels/mojo/kernels/mamba2_scan.mojo
# Mamba-2 SSD chunked parallel scan kernel.
#
# Implements: h_t = exp(a_t) * h_{t-1} + b_t * x_t,  y_t = dot(c_t, h_t)
# Parallelises over d_state channels within each time step.
#
# Reference: Dao & Gu, "Transformers are SSMs." ICML 2024.

from sys.info import simdwidthof
from algorithm import parallelize, vectorize
from memory import DTypePointer

alias FP32 = DType.float32
alias SIMD_WIDTH = simdwidthof[FP32]()


fn mamba2_scan_kernel(
    a_ptr: DTypePointer[FP32],   # (T,) log-A values
    b_ptr: DTypePointer[FP32],   # (T * d_state,) B matrices
    c_ptr: DTypePointer[FP32],   # (T * d_state,) C matrices
    x_ptr: DTypePointer[FP32],   # (T,) input scalars
    h_ptr: DTypePointer[FP32],   # (d_state,) state — updated in place
    out_ptr: DTypePointer[FP32], # (T,) output
    t_len: Int,
    d_state: Int,
):
    # Sequential over time; parallel over d_state for the inner updates
    for t in range(t_len):
        let a_t = a_ptr[t].exp()
        let x_t = x_ptr[t]
        let b_off = t * d_state
        let c_off = t * d_state

        @parameter
        fn update_state[simd_w: Int](i: Int):
            let hi = a_t * h_ptr[i] + b_ptr[b_off + i] * x_t
            h_ptr[i] = hi

        vectorize[update_state, SIMD_WIDTH](d_state)

        # y_t = dot(c[t], h) — accumulate with vectorize
        var y_acc = SIMD[FP32, SIMD_WIDTH](0)

        @parameter
        fn dot_accum[simd_w: Int](i: Int):
            y_acc += c_ptr[c_off + i].load[width=simd_w]() * h_ptr[i].load[width=simd_w]()

        vectorize[dot_accum, SIMD_WIDTH](d_state)
        out_ptr[t] = y_acc.reduce_add()
