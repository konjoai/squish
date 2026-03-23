"""
swiglu.mojo — Fused SwiGLU parallel Mojo kernel.

Wave 57b — MojoSwiGLUParallel

Computes:
    out[i] = gate[i] / (1 + exp(-gate[i])) * up[i]   (SiLU fused with multiply)

Specialization:
    @parameter on ffn_dim in {14336, 16384}
    parallelize over sequence rows
    vectorize[SIMD_WIDTH] over ffn_dim

Reference:
    Shazeer (2020) — GLU Variants Improve Transformer (arXiv:2002.05202).
"""

from math import exp
from sys.info import simdwidthof
from algorithm import parallelize, vectorize

alias float32 = DType.float32
alias SIMD_WIDTH = simdwidthof[float32]()


fn swiglu_parallel[ffn_dim: Int](
    gate: DTypePointer[float32],
    up: DTypePointer[float32],
    out: DTypePointer[float32],
    seq_len: Int,
) raises:
    """Fused SwiGLU over (seq_len, ffn_dim) gate and up projections."""

    @parameter
    fn process_row(s: Int):
        let base = s * ffn_dim

        @parameter
        fn compute[simd_width: Int](i: Int):
            let g = gate.load[width=simd_width](base + i)
            let u = up.load[width=simd_width](base + i)
            let silu_g = g / (SIMD[float32, simd_width](1.0) + exp(-g))
            out.store[width=simd_width](base + i, silu_g * u)

        vectorize[compute, SIMD_WIDTH](ffn_dim)

    parallelize[process_row](seq_len)
