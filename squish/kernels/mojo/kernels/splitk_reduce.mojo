# splitk_reduce.mojo — Flash-Decode split-K LSE merge Mojo kernel.
#
# Wave 59b — MojoSplitKReduce
#
# Merges P Flash-Decode split results via log-sum-exp renormalization:
#     global_max[h] = max_s(lse[s, h]) for s in range(n_splits)
#     weights[s, h] = exp(lse[s, h] - global_max[h])
#     normalize weights → sum = 1
#     output[h] = sum_s(weights[s, h] * output_split[s, h, :])
#
# Specialization:
#     @parameter on n_splits ∈ {4, 8, 16, 32}
#     @parameter on head_dim ∈ {64, 128}
#     parallelize over n_heads
#
# Reference:
#     Dao et al. — Flash-Decoding (MLSys 2024).

from algorithm import parallelize, vectorize
from math import exp, log
from sys.info import simdwidthof

alias float32 = DType.float32
alias simd_width = simdwidthof[float32]()


fn splitk_reduce[n_splits: Int, head_dim: Int](
    split_outputs: DTypePointer[float32],  # (n_splits, n_heads, head_dim)
    split_lses: DTypePointer[float32],     # (n_splits, n_heads)
    output: DTypePointer[float32],         # (n_heads, head_dim)
    n_heads: Int,
):
    """
    Merge n_splits Flash-Decode partial outputs via log-sum-exp.

    For each head h:
    1. Find global_max = max over splits of lse[s, h]
    2. Compute weights[s] = exp(lse[s, h] - global_max)
    3. output[h] = sum_s(weights[s] * split_output[s, h, :]) / sum(weights)
    """
    @parameter
    fn merge_head(h: Int):
        # Find global maximum LSE across splits
        var global_max: Float32 = -3.4028235e38
        for s in range(n_splits):
            var lse = split_lses[s * n_heads + h]
            if lse > global_max:
                global_max = lse

        # Compute weights and total
        var total: Float32 = 0.0
        var weights = SIMD[float32, n_splits](0)
        for s in range(n_splits):
            var w = exp(split_lses[s * n_heads + h] - global_max)
            weights[s] = w
            total += w

        # Normalize weights
        var inv_total = 1.0 / total

        # Weighted output accumulate
        var out_ptr = output + h * head_dim
        for i in range(head_dim):
            out_ptr[i] = 0.0

        for s in range(n_splits):
            var w = weights[s] * inv_total
            var src_ptr = split_outputs + (s * n_heads + h) * head_dim
            @parameter
            fn axpy[width: Int](i: Int):
                var o = SIMD[float32, width].load(out_ptr + i)
                var v = SIMD[float32, width].load(src_ptr + i)
                SIMD[float32, width].store(out_ptr + i, o + w * v)
            vectorize[axpy, simd_width](head_dim)

    parallelize[merge_head](n_heads)
