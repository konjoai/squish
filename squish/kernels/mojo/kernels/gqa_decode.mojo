"""
gqa_decode.mojo — GQA decode SDPA Mojo kernel.

Wave 57b — MojoGQADecodeKernel

Computes decode-mode Grouped Query Attention:
    scores[h, c] = dot(q[h], k_cache[c, h//group_size]) * scale
    weights[h]   = softmax(scores[h])
    out[h]       = sum_c(weights[h, c] * v_cache[c, h//group_size])

Specialization:
    @parameter on n_kv_heads (8 for Llama-3), head_dim (128)
    SIMD[DType.float32, 8] inner dot product with 16-lane unrolling
    parallelize over n_heads

Reference:
    Ainslie et al. (EMNLP 2023) — GQA: Training Generalized Multi-Query
    Transformer Models from Multi-Head Checkpoints (arXiv:2305.13245).
"""

from math import exp
from algorithm import parallelize, vectorize
from sys.info import simdwidthof

alias float32 = DType.float32
alias SIMD_WIDTH = simdwidthof[float32]()


fn gqa_decode_sdpa[n_kv_heads: Int, head_dim: Int](
    q: DTypePointer[float32],
    k_cache: DTypePointer[float32],
    v_cache: DTypePointer[float32],
    out: DTypePointer[float32],
    n_heads: Int,
    cache_len: Int,
    scale: Float32,
) raises:
    """GQA decode SDPA: Q (n_heads, hd) × K/V cache (cache_len, n_kv_heads, hd)."""
    let group_size = n_heads // n_kv_heads

    @parameter
    fn process_head(h: Int):
        let kv_head = h // group_size
        let q_base = h * head_dim
        var max_score: Float32 = -1e9
        let scores_ptr = DTypePointer[float32].alloc(cache_len)

        for c in range(cache_len):
            let k_base = (c * n_kv_heads + kv_head) * head_dim
            var dot: Float32 = 0.0

            @parameter
            fn dot_product[simd_width: Int](i: Int):
                dot += (q.load[width=simd_width](q_base + i) *
                        k_cache.load[width=simd_width](k_base + i)).reduce_add()

            vectorize[dot_product, SIMD_WIDTH](head_dim)
            let s = dot * scale
            scores_ptr.store(c, s)
            if s > max_score:
                max_score = s

        var sum_exp: Float32 = 0.0
        for c in range(cache_len):
            let e = exp(scores_ptr.load(c) - max_score)
            scores_ptr.store(c, e)
            sum_exp += e

        let inv_sum = 1.0 / sum_exp
        let out_base = h * head_dim
        for d in range(head_dim):
            out.store(out_base + d, Float32(0.0))

        for c in range(cache_len):
            let w = scores_ptr.load(c) * inv_sum
            let v_base = (c * n_kv_heads + kv_head) * head_dim

            @parameter
            fn accumulate[simd_width: Int](i: Int):
                let o = out.load[width=simd_width](out_base + i)
                out.store[width=simd_width](
                    out_base + i,
                    o + SIMD[float32, simd_width](w) * v_cache.load[width=simd_width](v_base + i)
                )

            vectorize[accumulate, SIMD_WIDTH](head_dim)
        scores_ptr.free()

    parallelize[process_head](n_heads)
