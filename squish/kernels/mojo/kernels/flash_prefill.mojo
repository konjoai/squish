"""
flash_prefill.mojo — Mojo block-tiled flash attention prefill kernel.

Reference implementation stub.
"""

from math import exp

alias F32 = DType.float32


fn flash_prefill_f32(
    Q: DTypePointer[F32],
    K: DTypePointer[F32],
    V: DTypePointer[F32],
    out: DTypePointer[F32],
    n_heads: Int,
    seq_len: Int,
    head_dim: Int,
    block_size: Int,
    causal: Bool,
) -> None:
    """Block-tiled SDPA with per-block online log-sum-exp (Flash Attention 2 style).

    Implements: out = softmax(Q K^T / sqrt(head_dim)) V

    Each block of Q is processed against all K/V blocks, maintaining running
    (max, sum-of-exp) statistics for numerical stability without materialising
    the full N×N attention matrix.
    """
    let scale = Float32(1.0) / Float32(head_dim).sqrt()
    let n_blocks = (seq_len + block_size - 1) // block_size

    for h in range(n_heads):
        for q_block in range(n_blocks):
            let q_start = q_block * block_size
            let q_end   = min(q_start + block_size, seq_len)

            # Per-query-row running statistics: m (max), l (normaliser)
            for qi in range(q_start, q_end):
                var m_i: Float32 = -1e30
                var l_i: Float32 = 0.0

                # Accumulate across K/V blocks
                for k_block in range(n_blocks):
                    let k_start = k_block * block_size
                    let k_end   = min(k_start + block_size, seq_len)

                    # Inner product: score[kj] = Q[h,qi] · K[h,kj] * scale
                    for kj in range(k_start, k_end):
                        if causal and kj > qi:
                            continue
                        var dot: Float32 = 0.0
                        for d in range(head_dim):
                            let qv = Q[(h * seq_len + qi) * head_dim + d]
                            let kv = K[(h * seq_len + kj) * head_dim + d]
                            dot += qv * kv
                        dot *= scale

                        # Online softmax update
                        let m_new = max(m_i, dot)
                        let exp_dot = exp(dot - m_new)
                        l_i = l_i * exp(m_i - m_new) + exp_dot
                        m_i = m_new

                        # Accumulate into output
                        let exp_norm = exp_dot / l_i
                        for d in range(head_dim):
                            let out_idx = (h * seq_len + qi) * head_dim + d
                            out[out_idx] = out[out_idx] * (1.0 - exp_norm) + V[(h * seq_len + kj) * head_dim + d] * exp_norm
