"""
int4_gemm.mojo — Mojo fused INT4 dequant + GEMM kernel.

Reference implementation stub.
"""

alias F32 = DType.float32
alias U8 = DType.uint8


fn int4_gemm_f32(
    x: DTypePointer[F32],
    W_packed: DTypePointer[U8],
    scales: DTypePointer[F32],
    offsets: DTypePointer[F32],
    out: DTypePointer[F32],
    m: Int,
    k: Int,
    n: Int,
    group_size: Int,
) -> None:
    """Fused INT4 dequantization + matrix multiply.

    No intermediate float32 weight matrix is materialised.  For each output
    element out[i, j], the kernel streams through W_packed row j, dequantises
    each nibble on-the-fly, and accumulates the dot product with x row i.

    Parameters
    ----------
    x:         float32 activation (m × k).
    W_packed:  uint8 packed weights (n × k/2).
    scales:    float32 per-group scales (n × k/group_size).
    offsets:   float32 per-group asymmetric zero-points (n × k/group_size).
    out:       float32 result (m × n).
    """
    let half_k = k // 2

    for i in range(m):
        for j in range(n):
            var acc: Float32 = 0.0
            for packed_idx in range(half_k):
                let byte  = W_packed[j * half_k + packed_idx]
                let q0    = Int(byte & 0x0F)
                let q1    = Int((byte >> 4) & 0x0F)
                let ki0   = packed_idx * 2
                let ki1   = ki0 + 1
                let g0    = ki0 // group_size
                let g1    = ki1 // group_size
                let n_groups = k // group_size
                let w0    = Float32(q0) * scales[j * n_groups + g0] + offsets[j * n_groups + g0]
                let w1    = Float32(q1) * scales[j * n_groups + g1] + offsets[j * n_groups + g1]
                acc += x[i * k + ki0] * w0 + x[i * k + ki1] * w1
            out[i * n + j] = acc
