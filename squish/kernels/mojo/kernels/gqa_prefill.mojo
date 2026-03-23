# gqa_prefill.mojo — Grouped-Query Attention prefill Mojo kernel.
#
# Wave 59b — MojoGQAPrefill
#
# Eliminates np.repeat KV expansion:
#     kv_h = q_h // group_size  (computed at index time)
#     parallelize(n_q_heads * T_out) tasks
#     tiled score accumulate @parameter on head_dim and tile_T (32, 64)
#     softmax over row
#     vectorize output sum(attn_t * V[kv_h, t, :])
#
# Specialization:
#     @parameter on head_dim ∈ {64, 128}
#     @parameter on tile_T ∈ {32, 64}
#     @parameter on group_size ∈ {4, 8}
#
# Reference:
#     Ainslie et al. (EMNLP 2023, arXiv 2305.13245) — GQA.

from algorithm import parallelize, vectorize
from math import exp
from sys.info import simdwidthof

alias float32 = DType.float32
alias simd_width = simdwidthof[float32]()


fn gqa_prefill[head_dim: Int, group_size: Int](
    q: DTypePointer[float32],      # (n_q_heads, T_q, head_dim)
    k: DTypePointer[float32],      # (n_kv_heads, T_k, head_dim)
    v: DTypePointer[float32],      # (n_kv_heads, T_k, head_dim)
    output: DTypePointer[float32], # (n_q_heads, T_q, head_dim)
    n_q_heads: Int,
    n_kv_heads: Int,
    T_q: Int,
    T_k: Int,
    scale: Float32,
):
    """
    GQA prefill: for each (q_head, t_out), compute attention over all T_k
    positions of the corresponding kv_head = q_head // group_size.

    Each task:
    1. Compute scores[t] = dot(Q[h, t_out], K[kv_h, t]) * scale
    2. Softmax with causal mask (t <= t_out when T_q == T_k)
    3. Output += attn[t] * V[kv_h, t]
    """
    @parameter
    fn compute_token(task_id: Int):
        var h = task_id // T_q
        var t_out = task_id % T_q
        var kv_h = h // group_size
        if kv_h >= n_kv_heads:
            kv_h = n_kv_heads - 1

        var q_ptr = q + (h * T_q + t_out) * head_dim
        var out_ptr = output + (h * T_q + t_out) * head_dim

        # Compute scores and running softmax
        var max_s: Float32 = -3.4028235e38
        var exp_sum: Float32 = 0.0

        for t in range(T_k):
            # causal: if T_q == T_k skip future tokens
            if T_q == T_k and t > t_out:
                continue
            var k_ptr = k + (kv_h * T_k + t) * head_dim
            var dot: Float32 = 0.0
            @parameter
            fn score_step[width: Int](i: Int):
                var q_vec = SIMD[float32, width].load(q_ptr + i)
                var k_vec = SIMD[float32, width].load(k_ptr + i)
                dot += (q_vec * k_vec).reduce_add()
            vectorize[score_step, simd_width](head_dim)
            var s = dot * scale
            if s > max_s:
                max_s = s

        for i in range(head_dim):
            out_ptr[i] = 0.0

        for t in range(T_k):
            if T_q == T_k and t > t_out:
                continue
            var k_ptr = k + (kv_h * T_k + t) * head_dim
            var dot: Float32 = 0.0
            @parameter
            fn score2[width: Int](i: Int):
                var q_vec = SIMD[float32, width].load(q_ptr + i)
                var k_vec = SIMD[float32, width].load(k_ptr + i)
                dot += (q_vec * k_vec).reduce_add()
            vectorize[score2, simd_width](head_dim)
            var w = exp(dot * scale - max_s)
            exp_sum += w
            var v_ptr = v + (kv_h * T_k + t) * head_dim
            @parameter
            fn axpy[width: Int](i: Int):
                var o = SIMD[float32, width].load(out_ptr + i)
                var v_vec = SIMD[float32, width].load(v_ptr + i)
                SIMD[float32, width].store(out_ptr + i, o + w * v_vec)
            vectorize[axpy, simd_width](head_dim)

        var inv = 1.0 / exp_sum
        for i in range(head_dim):
            out_ptr[i] *= inv

    parallelize[compute_token](n_q_heads * T_q)
