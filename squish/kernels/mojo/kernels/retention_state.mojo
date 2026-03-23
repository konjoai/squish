"""
retention_state.mojo — Retention recurrent state Mojo kernel.

Wave 57b — MojoRetentionState

Computes RetNet recurrent state update and retrieval:
    S_new[h]  = gamma * S[h] + outer(k[h], v[h])   (rank-1 update)
    o[h]      = S_new[h] @ q[h]                      (matrix-vector retrieval)

Specialization:
    @parameter on head_dim ∈ {64, 128}
    parallelize over n_heads
    SIMD outer-product accumulate + matvec for state matrix (head_dim, head_dim)
    Extensible to RWKV-6 channel-mix and SSM recurrent state (Wave 53)

Reference:
    Sun et al. (arXiv:2307.08621, 2023) — RetNet: Retaining Training
    Transformers' Performance for Inference.
"""

from algorithm import parallelize, vectorize
from sys.info import simdwidthof

alias float32 = DType.float32
alias SIMD_WIDTH = simdwidthof[float32]()


fn retention_step[head_dim: Int](
    q: DTypePointer[float32],
    k: DTypePointer[float32],
    v: DTypePointer[float32],
    state: DTypePointer[float32],
    out: DTypePointer[float32],
    new_state: DTypePointer[float32],
    n_heads: Int,
    gamma: Float32,
) raises:
    """Retention recurrent step for all heads in parallel."""
    let state_size = head_dim * head_dim

    @parameter
    fn process_head(h: Int):
        let q_off = h * head_dim
        let k_off = h * head_dim
        let v_off = h * head_dim
        let s_off = h * state_size
        let o_off = h * head_dim

        # Decay existing state: new_state[h] = gamma * state[h]
        @parameter
        fn decay[simd_width: Int](i: Int):
            let sv = state.load[width=simd_width](s_off + i)
            new_state.store[width=simd_width](s_off + i, SIMD[float32, simd_width](gamma) * sv)

        vectorize[decay, SIMD_WIDTH](state_size)

        # Rank-1 outer product update: new_state[h] += outer(k[h], v[h])
        for ki in range(head_dim):
            let k_val = k.load(k_off + ki)

            @parameter
            fn outer_update[simd_width: Int](vi: Int):
                let vv = v.load[width=simd_width](v_off + vi)
                let ns = new_state.load[width=simd_width](s_off + ki * head_dim + vi)
                new_state.store[width=simd_width](
                    s_off + ki * head_dim + vi,
                    ns + SIMD[float32, simd_width](k_val) * vv
                )

            vectorize[outer_update, SIMD_WIDTH](head_dim)

        # Matrix-vector retrieval: o[h] = new_state[h] @ q[h]
        for oi in range(head_dim):
            var acc: Float32 = 0.0
            let row_off = s_off + oi * head_dim

            @parameter
            fn matvec[simd_width: Int](qi: Int):
                acc += (new_state.load[width=simd_width](row_off + qi) *
                        q.load[width=simd_width](q_off + qi)).reduce_add()

            vectorize[matvec, SIMD_WIDTH](head_dim)
            out.store(o_off + oi, acc)

    parallelize[process_head](n_heads)
