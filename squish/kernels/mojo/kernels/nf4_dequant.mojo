"""
nf4_dequant.mojo — Mojo NF4 dequantization kernel.

Reference implementation stub.
"""

alias F32 = DType.float32
alias U8 = DType.uint8

# NF4 LUT: 16 standard-normal quantile levels (QLoRA, arXiv 2305.14314)
alias NF4_LUT = StaticTuple[Float32, 16](
    -1.0, -0.6961928, -0.5250730, -0.3949174,
    -0.2844413, -0.1847734, -0.0910500,  0.0,
     0.0795802,  0.1609302,  0.2461123,  0.3379152,
     0.4407098,  0.5626170,  0.7229568,  1.0
)


fn dequantize_nf4_f32(
    packed: DTypePointer[U8],
    scales: DTypePointer[F32],
    out: DTypePointer[F32],
    n_rows: Int,
    n_packed: Int,
    group_size: Int,
) -> None:
    """Dequantize NF4 nibble-packed weights to float32.

    Two nibbles per byte; each nibble indexes the 16-entry NF4_LUT, then
    multiplied by the per-group scale.
    """
    let n_cols = n_packed * 2

    for row in range(n_rows):
        let row_packed = packed + row * n_packed
        let row_scales = scales + row * (n_cols // group_size)
        let row_out    = out + row * n_cols

        for i in range(n_packed):
            let byte  = row_packed[i]
            let idx0  = Int(byte & 0x0F)
            let idx1  = Int((byte >> 4) & 0x0F)
            let j0    = i * 2
            let j1    = j0 + 1
            let g0    = j0 // group_size
            let g1    = j1 // group_size
            row_out[j0] = NF4_LUT[idx0] * row_scales[g0]
            row_out[j1] = NF4_LUT[idx1] * row_scales[g1]
