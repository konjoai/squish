"""
dual_chunk_attn.mojo — Dual-Chunk Attention intra-chunk SDPA Mojo kernel.

Wave 58b — MojoDualChunkAttn

Tiled causal SDPA per 512-token chunk with online softmax max tracking
(no explicit causal mask materialization):
    scores[q,k] = dot(Q[h,q], K[h,k]) * scale   (k <= q within chunk)
    attn        = online_softmax(scores)
    out[h,q]    = sum_k(attn[q,k] * V[h,k])

Specialization:
    @parameter on chunk_size=512, head_dim=128
    parallelize over n_heads
    Vectorized QK dot + inlined online-max softmax accumulation

Reference:
    An et al. (arXiv:2406.17419, 2024) — Dual Chunk Attention with Online
    Cross-chunk Interaction.
"""

from algorithm import parallelize, vectorize
from math import exp
from sys.info import simdwidthof

alias float32 = DType.float32
alias SIMD_WIDTH = simdwidthof[float32]()


fn dual_chunk_attn_forward[
    chunk_size: Int,
    head_dim: Int,
](
    Q: DTypePointer[float32],  # (n_heads, seq_len, head_dim)
    K: DTypePointer[float32],
    V: DTypePointer[float32],
    out: DTypePointer[float32],
    n_heads: Int,
    seq_len: Int,
    scale: Float32,
) -> None:
    """Compute dual-chunk causal SDPA over all heads in parallel.

    Each head processes n_chunks blocks of chunk_size tokens.  Within each
    chunk, scores are computed with an online row-max for numerical stability.
    """
    alias simd_w = simdwidthof[float32]()
    let n_chunks = (seq_len + chunk_size - 1) // chunk_size

    @parameter
    fn process_head(h: Int):
        for c in range(n_chunks):
            let lo = c * chunk_size
            let hi = min(lo + chunk_size, seq_len)
            let cs_actual = hi - lo
            let q_off = h * seq_len * head_dim
            let o_off = h * seq_len * head_dim

            var scores = InlineArray[Float32, chunk_size * chunk_size](0.0)

            # Build QK^T with causal constraint
            for qi in range(cs_actual):
                let qr = q_off + (lo + qi) * head_dim
                var row_max: Float32 = -1.0e38
                for ki in range(qi + 1):
                    let kr = q_off + (lo + ki) * head_dim
                    var dot: Float32 = 0.0

                    @parameter
                    fn dot_v[w: Int](j: Int):
                        dot += (
                            SIMD[float32, w].load(Q, qr + j)
                            * SIMD[float32, w].load(K, kr + j)
                        ).reduce_add()

                    vectorize[dot_v, simd_w](head_dim)
                    let sv = dot * scale
                    scores[qi * chunk_size + ki] = sv
                    if sv > row_max:
                        row_max = sv

                # Online softmax normalisation
                var row_sum: Float32 = 0.0
                for ki in range(qi + 1):
                    scores[qi * chunk_size + ki] = exp(
                        scores[qi * chunk_size + ki] - row_max
                    )
                    row_sum += scores[qi * chunk_size + ki]
                let inv = 1.0 / (row_sum + 1.0e-9)
                for ki in range(qi + 1):
                    scores[qi * chunk_size + ki] *= inv

            # Accumulate attn-weighted values
            for qi in range(cs_actual):
                let or_ = o_off + (lo + qi) * head_dim
                for ki in range(qi + 1):
                    let a = scores[qi * chunk_size + ki]
                    let vr = o_off + (lo + ki) * head_dim

                    @parameter
                    fn acc_v[w: Int](j: Int):
                        SIMD[float32, w].store(
                            out, or_ + j,
                            SIMD[float32, w].load(out, or_ + j)
                            + a * SIMD[float32, w].load(V, vr + j),
                        )

                    vectorize[acc_v, simd_w](head_dim)

    parallelize[process_head](n_heads)

