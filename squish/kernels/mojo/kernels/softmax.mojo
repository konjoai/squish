"""
softmax.mojo — SIMD-accelerated online softmax kernel.

Reference implementation stub.  Compile with:
  magic run mojo build --emit shared softmax.mojo -o libsquish_kernels.dylib

This file serves as the source-of-truth for the Mojo SIMD softmax;
the Python wrapper (softmax_mojo.py) falls back to Rust/NumPy when
this file has not been compiled into a shared library.
"""

from math import exp
from algorithm import vectorize
from sys.info import simdwidthof

alias F32 = DType.float32
alias SIMD_WIDTH = simdwidthof[F32]()


fn softmax_f32(
    logits: DTypePointer[F32],
    out: DTypePointer[F32],
    n: Int,
) -> None:
    """Numerically stable SIMD softmax.

    Pass 1: find max via SIMD horizontal reduce.
    Pass 2: subtract max, exponentiate, accumulate sum.
    Pass 3: normalise in-place.
    """
    # Pass 1 — find max
    var vmax = SIMD[F32, SIMD_WIDTH].splat(-1e30)

    @parameter
    fn reduce_max[width: Int](i: Int):
        vmax = vmax.reduce_max(logits.load[width=width](i))

    vectorize[reduce_max, SIMD_WIDTH](n)
    let max_val = vmax.reduce_max()

    # Pass 2 — shifted exp + partial sum
    var vsum = SIMD[F32, SIMD_WIDTH].splat(0.0)

    @parameter
    fn exp_and_sum[width: Int](i: Int):
        let v = logits.load[width=width](i) - max_val
        let e = v.exp()
        out.store[width=width](i, e)
        vsum += e

    vectorize[exp_and_sum, SIMD_WIDTH](n)
    let total = vsum.reduce_add()
    let inv_total = 1.0 / total

    # Pass 3 — normalise
    @parameter
    fn normalise[width: Int](i: Int):
        out.store[width=width](i, out.load[width=width](i) * inv_total)

    vectorize[normalise, SIMD_WIDTH](n)
