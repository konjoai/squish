# squish/kernels/mojo/kernels/flute_lut.mojo
# FLUTE per-group LUT quantization: encode and decode kernels.
#
# encode_kernel: maps each weight value to the nearest codebook entry
#                (argmin L1 distance) per column group.
#                Parallelised over column groups.
# decode_kernel: gathers codebook values by code indices.
#                Parallelised over column groups.
#
# Reference: Guo et al., "FLUTE: Flexible Lookup Table Engine for
# LUT-based Network Inference." CVPR 2024.

from algorithm import parallelize, vectorize


fn flute_lut_encode_kernel(
    w_ptr: UnsafePointer[Float32],    # (rows * cols,)          FP32 weights
    cb_ptr: UnsafePointer[Float32],   # (n_groups * cb_size,)   FP32 codebook
    out_ptr: UnsafePointer[UInt8],    # (rows * cols,)          UInt8 code indices
    rows: Int,
    cols: Int,
    n_groups: Int,
    cb_size: Int,
    group_size: Int,
):
    """Encode each weight to nearest codebook entry index (argmin L1)."""

    @parameter
    fn process_group(g: Int):
        var col_start = g * group_size
        var col_end = col_start + group_size
        if col_end > cols:
            col_end = cols
        var cb_off = g * cb_size

        for row in range(rows):
            for c in range(col_start, col_end):
                var v = w_ptr[row * cols + c]
                var best_idx = 0
                var best_dist = Float32(1.0e30)
                for k in range(cb_size):
                    var d = v - cb_ptr[cb_off + k]
                    if d < 0.0:
                        d = -d
                    if d < best_dist:
                        best_dist = d
                        best_idx = k
                out_ptr[row * cols + c] = UInt8(best_idx)

    parallelize[process_group](n_groups)


fn flute_lut_decode_kernel(
    codes_ptr: UnsafePointer[UInt8],   # (rows * cols,)          UInt8 code indices
    cb_ptr: UnsafePointer[Float32],    # (n_groups * cb_size,)   FP32 codebook
    out_ptr: UnsafePointer[Float32],   # (rows * cols,)          FP32 decoded values
    rows: Int,
    cols: Int,
    n_groups: Int,
    cb_size: Int,
    group_size: Int,
):
    """Decode code indices to float32 by gathering from the codebook."""

    @parameter
    fn process_group(g: Int):
        var col_start = g * group_size
        var col_end = col_start + group_size
        if col_end > cols:
            col_end = cols
        var cb_off = g * cb_size

        for row in range(rows):
            for c in range(col_start, col_end):
                var idx = Int(codes_ptr[row * cols + c])
                if idx >= cb_size:
                    idx = cb_size - 1
                out_ptr[row * cols + c] = cb_ptr[cb_off + idx]

    parallelize[process_group](n_groups)
