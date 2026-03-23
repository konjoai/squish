"""
infini_attn.mojo — Infini-Attention compressive memory Mojo kernel.

Wave 58b — MojoInfiniAttnMemory

Fused outer-product-accumulate memory update + matrix-vector retrieval:
    K_gate = ELU(K) + 1              (optional gating, per token)
    M += outer(K_gate[h,t], V[h,t])  (rank-1 accumulate over T tokens)
    Z += K_gate[h,t]                  (normalizer accumulate)
    A[h,t] = M[h] @ sigma(Q[h,t]) / (Z[h] · sigma(Q[h,t]) + eps)

Specialization:
    @parameter on head_dim ∈ {64, 128}
    parallelize over n_heads
    SIMD outer-product (d×d unrolled) + matvec retrieval
    Matches MojoRetentionState outer-product pattern from Wave 57b

Reference:
    Munkhdalai et al. (arXiv:2404.07143, 2024) — Infini-attention.
"""

from algorithm import parallelize, vectorize
from math import exp, tanh
from sys.info import simdwidthof

alias float32 = DType.float32
alias SIMD_WIDTH = simdwidthof[float32]()


fn infini_attn_update[head_dim: Int](
    K: DTypePointer[float32],  # (n_heads, T, head_dim)
    V: DTypePointer[float32],  # (n_heads, T, head_dim)
    M: DTypePointer[float32],  # (n_heads, head_dim, head_dim) — in/out
    Z: DTypePointer[float32],  # (n_heads, head_dim) — in/out
    n_heads: Int,
    T: Int,
    use_elu: Int,
) -> None:
    """Update compressive memory M += sum_t outer(gate(K[h,t]), V[h,t]).

    Modifies M and Z in-place.
    """
    alias simd_w = simdwidthof[float32]()

    @parameter
    fn update_head(h: Int):
        let k_off = h * T * head_dim
        let v_off = h * T * head_dim
        let m_off = h * head_dim * head_dim
        let z_off = h * head_dim

        for t in range(T):
            let kt = k_off + t * head_dim
            let vt = v_off + t * head_dim

            # ELU+1 gating: inline per element
            for i in range(head_dim):
                let ki = K[kt + i]
                let ki_gate: Float32
                if use_elu == 1:
                    ki_gate = ki if ki > 0 else exp(ki) - 1.0 + 1.0
                else:
                    ki_gate = ki

                # Outer product row: M[h, i, :] += ki_gate * V[h, t, :]
                @parameter
                fn outer_v[w: Int](j: Int):
                    SIMD[float32, w].store(
                        M, m_off + i * head_dim + j,
                        SIMD[float32, w].load(M, m_off + i * head_dim + j)
                        + ki_gate * SIMD[float32, w].load(V, vt + j),
                    )

                vectorize[outer_v, simd_w](head_dim)
                Z[z_off + i] += ki_gate

    parallelize[update_head](n_heads)


fn infini_attn_retrieve[head_dim: Int](
    Q: DTypePointer[float32],    # (n_heads, T, head_dim)
    M: DTypePointer[float32],    # (n_heads, head_dim, head_dim)
    Z: DTypePointer[float32],    # (n_heads, head_dim)
    out: DTypePointer[float32],  # (n_heads, T, head_dim)
    n_heads: Int,
    T: Int,
    eps: Float32,
) -> None:
    """Retrieve from compressive memory: A = M @ sigma(Q) / (Z·sigma(Q) + eps)."""
    alias simd_w = simdwidthof[float32]()

    @parameter
    fn retrieve_head(h: Int):
        let q_off = h * T * head_dim
        let m_off = h * head_dim * head_dim
        let z_off = h * head_dim
        let o_off = h * T * head_dim

        for t in range(T):
            let qt = q_off + t * head_dim

            # Compute sigma(Q) = ELU(Q)+1 inline and M @ sigma(Q)
            var z_dot: Float32 = 0.0
            var a_row = InlineArray[Float32, head_dim](0.0)

            for i in range(head_dim):
                let qi = Q[qt + i]
                let qi_gate = qi if qi > 0 else exp(qi) - 1.0 + 1.0
                z_dot += Z[z_off + i] * qi_gate

                # M[h, :, i] * qi_gate accumulated into a_row
                for d in range(head_dim):
                    a_row[d] += M[m_off + d * head_dim + i] * qi_gate

            let inv_norm = 1.0 / (z_dot + eps)
            let ot = o_off + t * head_dim

            @parameter
            fn write_v[w: Int](j: Int):
                SIMD[float32, w].store(out, ot + j,
                    SIMD[float32, w](a_row[j]) * inv_norm)

            vectorize[write_v, simd_w](head_dim)

    parallelize[retrieve_head](n_heads)
