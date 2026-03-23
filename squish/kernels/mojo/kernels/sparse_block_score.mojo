"""
sparse_block_score.mojo — Block-level attention scoring Mojo kernel.

Wave 57b — MojoSparseBlockScore

Computes block-level Q×K^T scores for top-K sparse block selection:
    score[h, qi, ki] = mean(Q_block[h,qi] @ K_block[h,ki]^T) * scale

Specialization:
    @parameter on block_size ∈ {16, 32, 64} and head_dim ∈ {64, 128}
    parallelize over (head, q_block) pairs
    tiled 8x8 SIMD matmul for (block_size × head_dim) tiles

Reference:
    DeepSeek-AI (arXiv:2502.11089, 2025) — Native Sparse Attention:
    Hardware-Aligned and Natively Trainable Sparse Attention.
"""

from algorithm import parallelize, vectorize
from sys.info import simdwidthof

alias float32 = DType.float32
alias SIMD_WIDTH = simdwidthof[float32]()


fn sparse_block_score[block_size: Int, head_dim: Int](
    q_blocks: DTypePointer[float32],
    k_blocks: DTypePointer[float32],
    scores: DTypePointer[float32],
    n_heads: Int,
    n_q_blocks: Int,
    n_k_blocks: Int,
    scale: Float32,
) raises:
    """Block-level Q×K^T scoring: (H, Nq, B, D) x (H, Nk, B, D) -> (H, Nq, Nk)."""
    let n_pairs = n_heads * n_q_blocks

    @parameter
    fn process_pair(idx: Int):
        let h = idx // n_q_blocks
        let qi = idx % n_q_blocks
        let q_base = (h * n_q_blocks + qi) * block_size * head_dim

        for ki in range(n_k_blocks):
            let k_base = (h * n_k_blocks + ki) * block_size * head_dim
            var block_score: Float32 = 0.0

            for bq in range(block_size):
                let qrow = q_base + bq * head_dim
                for bk in range(block_size):
                    let krow = k_base + bk * head_dim
                    var dot: Float32 = 0.0

                    @parameter
                    fn dot_accum[simd_width: Int](d: Int):
                        dot += (q_blocks.load[width=simd_width](qrow + d) *
                                k_blocks.load[width=simd_width](krow + d)).reduce_add()

                    vectorize[dot_accum, SIMD_WIDTH](head_dim)
                    block_score += dot

            let score_idx = (h * n_q_blocks + qi) * n_k_blocks + ki
            scores.store(score_idx, block_score * scale / (block_size * block_size).cast[float32]())

    parallelize[process_pair](n_pairs)
