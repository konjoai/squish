# squish/kernels/mojo/kernels/bit_distiller.mojo
# BitDistiller per-group asymmetric weight quantisation kernel.
#
# Parallelises over rows; within each row iterates over chunks of ``group_size``
# elements computing per-group min, max, scale, and zero-point.
#
# Reference: Du et al., "BitDistiller: Unleashing the Potential of Sub-4-Bit
# LLMs via Self-Distillation," arXiv 2402.10631, 2024.

from algorithm import parallelize


fn bit_distiller_quant_kernel(
    w_ptr:      UnsafePointer[Float32],  # (rows * cols) flat
    q_ptr:      UnsafePointer[Int8],     # (rows * cols) output
    scales_ptr: UnsafePointer[Float32],  # (rows * groups_per_row) output
    zeros_ptr:  UnsafePointer[Float32],  # (rows * groups_per_row) output
    rows:       Int,
    cols:       Int,
    bits:       Int,
    group_size: Int,
):
    """Per-group asymmetric quantisation of a 2-D weight matrix.

    For each row, iterates over groups of ``group_size`` columns, computes
    the per-group min/max range, derives scale and zero-point, then clips
    and rounds each element.

    Parallelises over the ``rows`` outer loop.

    Args:
        w_ptr:      Flat float32 weight matrix (rows × cols).
        q_ptr:      Output int8 quantised weights.
        scales_ptr: Output per-group scales.
        zeros_ptr:  Output per-group zero-points.
        rows:       Number of rows.
        cols:       Number of columns.
        bits:       Quantisation bits.
        group_size: Elements per quantisation group.
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
            # Find group min/max
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
            var zero = -mn / scale
            var blk_idx = r * groups_per_row + g
            scales_ptr[blk_idx] = scale
            zeros_ptr[blk_idx] = zero
            # Quantise
            for j in range(g_start, g_end):
                var v = w_ptr[row_off + j]
                var qv = (v - mn) / rng * levels
                # Round and clamp
                var rounded = Int(qv + Float32(0.5))
                if rounded < 0:
                    rounded = 0
                if rounded > Int(levels):
                    rounded = Int(levels)
                q_ptr[row_off + j] = Int8(rounded)
                q_ptr[row_off + j] = rounded

    parallelize[process_row](rows)
