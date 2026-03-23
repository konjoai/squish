# flash_decode_split.mojo — Flash-Decode per-split attention Mojo kernel.
#
# Wave 59b — MojoFlashDecodeKernel
#
# Parallelizes over n_heads; per-head:
#     scores[t] = dot(K_split[kv_h, t, :], q[h, :]) * scale
#     online softmax: running max + exp accumulate (no score materialization)
#     output_h = sum_t(w[t] * V_split[kv_h, t, :])
#
# Specialization:
#     @parameter on head_dim ∈ {64, 128}
#     @parameter on split_len (power-of-two)
#     parallelize over n_heads
#
# Reference:
#     Dao et al. — Flash-Decoding for Long-Context Inference (MLSys 2024).

from algorithm import parallelize, vectorize
from math import log, exp, sqrt
from sys.info import simdwidthof

alias float32 = DType.float32
alias simd_width = simdwidthof[float32]()


fn flash_decode_split[head_dim: Int, split_len: Int](
    q: DTypePointer[float32],            # (n_heads, head_dim)
    k_split: DTypePointer[float32],      # (n_kv_heads * split_len, head_dim)
    v_split: DTypePointer[float32],      # (n_kv_heads * split_len, head_dim)
    output: DTypePointer[float32],       # (n_heads, head_dim) — write output
    lse_out: DTypePointer[float32],      # (n_heads,) — log-sum-exp per head
    max_out: DTypePointer[float32],      # (n_heads,) — max score per head
    n_heads: Int,
    n_kv_heads: Int,
    gqa_group: Int,
    scale: Float32,
):
    """
    Compute Flash-Decode split attention for all heads in parallel.

    Each head:
    1. GEMV: scores[t] = dot(K_split[kv_h, t], q[h]) * scale
    2. Online softmax: running max + exp accumulate
    3. Output: sum_t(w_t * V_split[kv_h, t])
    """
    @parameter
    fn compute_head(h: Int):
        var kv_h = h // gqa_group
        if kv_h >= n_kv_heads:
            kv_h = n_kv_heads - 1

        var q_ptr = q + h * head_dim
        var max_score: Float32 = -3.4028235e38
        var exp_sum: Float32 = 0.0

        # First pass: compute scores and running max (online softmax pass 1)
        var scores_max: Float32 = -3.4028235e38
        for t in range(split_len):
            var k_ptr = k_split + (kv_h * split_len + t) * head_dim
            var dot: Float32 = 0.0
            @parameter
            fn dot_step[width: Int](i: Int):
                var q_vec = SIMD[float32, width].load(q_ptr + i)
                var k_vec = SIMD[float32, width].load(k_ptr + i)
                dot += (q_vec * k_vec).reduce_add()
            vectorize[dot_step, simd_width](head_dim)
            var s = dot * scale
            if s > scores_max:
                scores_max = s

        # Second pass: accumulate exp-weighted output
        var out_ptr = output + h * head_dim
        for i in range(head_dim):
            out_ptr[i] = 0.0

        var total_weight: Float32 = 0.0
        for t in range(split_len):
            var k_ptr = k_split + (kv_h * split_len + t) * head_dim
            var dot: Float32 = 0.0
            @parameter
            fn dot2[width: Int](i: Int):
                var q_vec = SIMD[float32, width].load(q_ptr + i)
                var k_vec = SIMD[float32, width].load(k_ptr + i)
                dot += (q_vec * k_vec).reduce_add()
            vectorize[dot2, simd_width](head_dim)
            var w = exp(dot * scale - scores_max)
            total_weight += w
            var v_ptr = v_split + (kv_h * split_len + t) * head_dim
            @parameter
            fn axpy[width: Int](i: Int):
                var out_vec = SIMD[float32, width].load(out_ptr + i)
                var v_vec = SIMD[float32, width].load(v_ptr + i)
                SIMD[float32, width].store(out_ptr + i, out_vec + w * v_vec)
            vectorize[axpy, simd_width](head_dim)

        # Normalize output
        var inv_w = 1.0 / total_weight
        for i in range(head_dim):
            out_ptr[i] *= inv_w

        lse_out[h] = scores_max + log(total_weight)
        max_out[h] = scores_max

    parallelize[compute_head](n_heads)
