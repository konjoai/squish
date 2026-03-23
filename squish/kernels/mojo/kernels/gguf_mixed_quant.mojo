# squish/kernels/mojo/kernels/gguf_mixed_quant.mojo
# GGUF-style mixed block quantisation with super-block meta-scaling.
#
# Parallelises over rows.  Within each row, iterates over blocks of
# ``group_size`` elements computing per-block min, scale, and zero.
# Every 8 consecutive blocks share a super-block whose meta-scale is the
# mean of the individual block scales.
#
# Reference: GGML/GGUF quantisation format as used in llama.cpp, 2023–.

from algorithm import parallelize


fn gguf_mixed_quant_kernel(
    w_ptr:          UnsafePointer[Float32],  # (rows * cols) flat
    q_ptr:          UnsafePointer[Int8],     # (rows * cols) output
    scales_ptr:     UnsafePointer[Float32],  # (n_blocks) output
    mins_ptr:       UnsafePointer[Float32],  # (n_blocks) output
    super_scales:   UnsafePointer[Float32],  # (n_super) output
    rows:           Int,
    cols:           Int,
    bits:           Int,
    group_size:     Int,
):
    """GGUF-style block-quantisation with two-tier super-block scaling.

    For each row, computes per-block min/max scale and a super-block meta-scale
    (mean of 8 consecutive block scales).  Parallelises over rows.

    Args:
        w_ptr:        Flat float32 weight matrix (rows × cols).
        q_ptr:        Output int8 quantised weights.
        scales_ptr:   Output per-block scales.
        mins_ptr:     Output per-block mins.
        super_scales: Output per-super-block meta-scales.
        rows:         Number of rows.
        cols:         Number of columns.
        bits:         Quantisation bits.
        group_size:   Block size (elements per block).
    """
    var gs = group_size if group_size > 0 else 1
    var groups_per_row = (cols + gs - 1) // gs
    var levels = Float32((1 << bits) - 1)

    @parameter
    fn process_row(r: Int):
        var row_off = r * cols
        for g in range(groups_per_row):
            var g_start = g * gs
            var g_end = g_start + gs if g_start + gs <= cols else cols
            var mn = Float32(1e38)
            var mx = Float32(-1e38)
            for j in range(g_start, g_end):
                var v = w_ptr[row_off + j]
                if v < mn:
                    mn = v
                if v > mx:
                    mx = v
            var rng = mx - mn
            if rng < Float32(1e-8):
                rng = Float32(1e-8)
            var scale = rng / levels
            var blk_idx = r * groups_per_row + g
            scales_ptr[blk_idx] = scale
            mins_ptr[blk_idx] = mn
            for j in range(g_start, g_end):
                var v = w_ptr[row_off + j]
                var qv = Int((v - mn) / rng * levels + Float32(0.5))
                if qv < 0:
                    qv = 0
                if qv > Int(levels):
                    qv = Int(levels)
                q_ptr[row_off + j] = Int8(qv)

    parallelize[process_row](rows)

    # Compute super-block meta-scales (sequential — lightweight)
    var n_blocks = rows * groups_per_row
    var super_blk = 8
    var n_super = (n_blocks + super_blk - 1) // super_blk
    for si in range(n_super):
        var start = si * super_blk
        var end = start + super_blk if start + super_blk <= n_blocks else n_blocks
        var sum = Float32(0.0)
        var cnt = end - start
        for bi in range(start, end):
            sum += scales_ptr[bi]
        var mean = sum / Float32(cnt)
        super_scales[si] = mean if mean > Float32(1e-8) else Float32(1e-8)
