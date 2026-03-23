"""
rmsnorm.mojo — Fused residual-add + RMSNorm Mojo kernel.

Wave 57b — MojoRMSNormFused

Computes:
    x_sum    = x + residual                            (SIMD add)
    ms       = mean(x_sum ** 2)                        (vectorize horizontal-sum)
    x_norm   = x_sum / sqrt(ms + eps)                  (vectorize rsqrt + multiply)
    out      = x_norm * weight                         (vectorize element-scale)
    new_residual = x_sum                               (passthrough for next layer)

Specialization:
    @parameter on hidden_dim ∈ {4096, 7168, 8192}
    Single memory read of (x, residual), single write of (out, new_residual).
    Applies 64× per 32-layer decode step.

Reference:
    Zhang & Sennrich (NeurIPS 2019) — Root Mean Square Layer Normalization.
"""

from math import sqrt
from sys import simdwidthof

alias float32 = DType.float32
alias SIMD_WIDTH = simdwidthof[float32]()


fn rmsnorm_fused[hidden_dim: Int](
    x: DTypePointer[float32],
    residual: DTypePointer[float32],
    weight: DTypePointer[float32],
    out: DTypePointer[float32],
    new_residual: DTypePointer[float32],
    seq_len: Int,
    eps: Float32,
) raises:
    """Fused residual-add + RMSNorm for seq_len rows of hidden_dim elements."""
    for s in range(seq_len):
        let base = s * hidden_dim
        var sum_sq: Float32 = 0.0

        @parameter
        fn accumulate[simd_width: Int](i: Int):
            let xi = x.load[width=simd_width](base + i)
            let ri = residual.load[width=simd_width](base + i)
            let xs = xi + ri
            new_residual.store[width=simd_width](base + i, xs)
            sum_sq += (xs * xs).reduce_add()

        vectorize[accumulate, SIMD_WIDTH](hidden_dim)
        let rms_inv = rsqrt(sum_sq / hidden_dim + eps)

        @parameter
        fn normalize[simd_width: Int](i: Int):
            let xs = new_residual.load[width=simd_width](base + i)
            let w = weight.load[width=simd_width](i)
            out.store[width=simd_width](base + i, xs * rms_inv * w)

        vectorize[normalize, SIMD_WIDTH](hidden_dim)
