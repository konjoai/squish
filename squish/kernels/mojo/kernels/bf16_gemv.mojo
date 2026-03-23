# bf16_gemv.mojo — Native BF16 weight × FP32 activation GEMV Mojo kernel.
#
# Wave 59b — MojoBF16GEMV
#
# Uses SIMD[DType.bfloat16, 8] weight loads + SIMD.cast to FP32 accumulator
# + FMA dot-product with parallelize over output rows; eliminates the
# bf16→fp32 upcast allocation in compressed_loader.py decode path.
#
# Specialization:
#     @parameter on hidden_dim ∈ {2048, 4096}
#     parallelize over out_features
#
# Reference:
#     ARM Architecture Reference Manual — ARMv8.6-A BF16 SIMD (FBFMMLA).

from algorithm import parallelize, vectorize
from sys.info import simdwidthof

alias float32 = DType.float32
alias bfloat16 = DType.bfloat16
alias simd_width = simdwidthof[float32]()


fn bf16_gemv[hidden_dim: Int](
    weight: DTypePointer[bfloat16],      # (out_features, hidden_dim)
    activation: DTypePointer[float32],   # (hidden_dim,)
    output: DTypePointer[float32],       # (out_features,)
    out_features: Int,
):
    """
    Compute output[o] = sum_i(weight[o, i] * activation[i])
    where weight is stored as BF16 and activation is FP32.

    Loads BF16 weight rows as SIMD[bfloat16, 8], casts to FP32 accumulator,
    and computes FMA dot-product — no intermediate FP32 weight copy.
    """
    @parameter
    fn compute_row(o: Int):
        var w_ptr = weight + o * hidden_dim
        var acc = SIMD[float32, simd_width](0)

        @parameter
        fn fma_step[width: Int](i: Int):
            var w_bf16 = SIMD[bfloat16, width].load(w_ptr + i)
            var w_f32 = w_bf16.cast[float32]()
            var a_f32 = SIMD[float32, width].load(activation + i)
            acc += w_f32 * a_f32
        vectorize[fma_step, simd_width](hidden_dim)

        output[o] = acc.reduce_add()

    parallelize[compute_row](out_features)
