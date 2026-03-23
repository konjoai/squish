"""
token_cos_sim.mojo — All-pairs token cosine similarity Mojo kernel.

Wave 57b — MojoTokenCosSim

Computes (T_a, T_b) cosine similarity matrix:
    sim[i, j] = dot(a[i], b[j]) / (||a[i]|| * ||b[j]||)

Specialization:
    @parameter on D ∈ {128, 256, 512, 1024}
    parallelize over T_a rows
    vectorize[SIMD_WIDTH] for partial norm and dot product
    SIMD rsqrt for inverse norm

Reference:
    Bolya et al. (ICLR 2023) — Token Merging: Your ViT but Faster
    (arXiv:2210.09461).
"""

from math import sqrt, rsqrt
from algorithm import parallelize, vectorize
from sys.info import simdwidthof

alias float32 = DType.float32
alias SIMD_WIDTH = simdwidthof[float32]()


fn token_cos_sim[dim: Int](
    a: DTypePointer[float32],
    b: DTypePointer[float32],
    out: DTypePointer[float32],
    ta: Int,
    tb: Int,
    eps: Float32,
) raises:
    """All-pairs cosine similarity: a (ta, dim), b (tb, dim) -> out (ta, tb)."""

    # Precompute b norms
    let b_norms_ptr = DTypePointer[float32].alloc(tb)
    for j in range(tb):
        let jbase = j * dim
        var sq: Float32 = 0.0

        @parameter
        fn b_sum_sq[simd_width: Int](i: Int):
            let bv = b.load[width=simd_width](jbase + i)
            sq += (bv * bv).reduce_add()

        vectorize[b_sum_sq, SIMD_WIDTH](dim)
        b_norms_ptr.store(j, rsqrt(sq + eps))

    @parameter
    fn process_row(i: Int):
        let ibase = i * dim
        var a_sq: Float32 = 0.0

        @parameter
        fn a_sum_sq[simd_width: Int](k: Int):
            let av = a.load[width=simd_width](ibase + k)
            a_sq += (av * av).reduce_add()

        vectorize[a_sum_sq, SIMD_WIDTH](dim)
        let a_inv_norm = rsqrt(a_sq + eps)

        for j in range(tb):
            let jbase = j * dim
            var dot: Float32 = 0.0

            @parameter
            fn dot_product[simd_width: Int](k: Int):
                dot += (a.load[width=simd_width](ibase + k) *
                        b.load[width=simd_width](jbase + k)).reduce_add()

            vectorize[dot_product, SIMD_WIDTH](dim)
            out.store(i * tb + j, dot * a_inv_norm * b_norms_ptr.load(j))

    parallelize[process_row](ta)
    b_norms_ptr.free()
