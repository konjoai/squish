# squish/kernels/mojo/kernels/milo_int3_pack.mojo
# MILO INT3 bit-packing and group-wise symmetric quantisation kernels.
#
# milo_int3_pack_kernel: packs N signed INT3 values (i8, range -4..3) into
#   bytes.  8 INT3 values occupy exactly 24 bits (3 bytes).  Full groups of 8
#   are processed in parallel via ``parallelize``; any remainder is handled
#   sequentially.
#
# milo_quant_kernel: group-wise symmetric abs-max quantisation to INT3.
#   Parallel over rows; per-row chunks(group_size) with abs-max scale.

from algorithm import parallelize


fn milo_int3_pack_kernel(
    values_ptr: UnsafePointer[Int8],   # (n,) INT3 values
    out_ptr:    UnsafePointer[UInt8],  # (ceil(n*3/8),) output
    n:          Int,
):
    """Pack N signed INT3 values (i8) into compact bytes.

    8 INT3 values → 24 bits → 3 bytes.  Full groups of 8 are packed in
    parallel; any remainder is handled sequentially.

    Args:
        values_ptr: Flat int8 array of N INT3 values in range -4..3.
        out_ptr:    Output byte array of ceil(N * 3 / 8) bytes.
        n:          Number of input INT3 values.
    """
    var n_full = n // 8

    @parameter
    fn pack_group(gi: Int):
        var base = gi * 8
        var bits = UInt32(0)
        for k in range(8):
            var v3 = UInt32(Int(values_ptr[base + k]) & 0x7)
            bits = bits | (v3 << UInt32(k * 3))
        var dst = gi * 3
        out_ptr[dst]     = UInt8(bits & UInt32(0xFF))
        out_ptr[dst + 1] = UInt8((bits >> UInt32(8)) & UInt32(0xFF))
        out_ptr[dst + 2] = UInt8((bits >> UInt32(16)) & UInt32(0xFF))

    parallelize[pack_group](n_full)

    # Handle remainder (< 8 values) sequentially
    var rem_start = n_full * 8
    if rem_start < n:
        var bits = UInt32(0)
        for k in range(n - rem_start):
            var v3 = UInt32(Int(values_ptr[rem_start + k]) & 0x7)
            bits = bits | (v3 << UInt32(k * 3))
        var byte_start = n_full * 3
        var rem_bits = (n - rem_start) * 3
        var rem_bytes = (rem_bits + 7) // 8
        for b in range(rem_bytes):
            out_ptr[byte_start + b] = UInt8((bits >> UInt32(b * 8)) & UInt32(0xFF))


fn milo_quant_kernel(
    w_ptr:      UnsafePointer[Float32],  # (rows * cols) flat
    q_ptr:      UnsafePointer[Int8],     # (rows * cols) output
    scales_ptr: UnsafePointer[Float32],  # (rows * groups_per_row) output
    rows:       Int,
    cols:       Int,
    group_size: Int,
):
    """Group-wise symmetric INT3 quantisation.

    For each row, iterates over groups of ``group_size`` columns, finds the
    abs-max value, computes scale = abs_max / 3.0, then clips and rounds each
    element to the range -3..3.  Parallelises over rows.

    Args:
        w_ptr:      Flat float32 weight matrix (rows × cols).
        q_ptr:      Output int8 quantised weights.
        scales_ptr: Output per-group scales.
        rows:       Number of rows.
        cols:       Number of columns.
        group_size: Elements per quantisation group.
    """
    var gs = group_size if group_size > 0 else 1
    var groups_per_row = (cols + gs - 1) // gs
    var max_int3 = Float32(3.0)

    @parameter
    fn process_row(r: Int):
        var row_off = r * cols
        for g in range(groups_per_row):
            var g_start = g * gs
            var g_end = g_start + gs if g_start + gs <= cols else cols
            var abs_max = Float32(0.0)
            for j in range(g_start, g_end):
                var v = w_ptr[row_off + j]
                var av = v if v >= Float32(0.0) else -v
                if av > abs_max:
                    abs_max = av
            var scale = abs_max / max_int3
            if scale < Float32(1e-8):
                scale = Float32(1e-8)
            scales_ptr[r * groups_per_row + g] = scale
            for j in range(g_start, g_end):
                var v = w_ptr[row_off + j]
                var qv = Int(v / scale + Float32(0.5) if v >= Float32(0.0) else v / scale - Float32(0.5))
                if qv < -3:
                    qv = -3
                if qv > 3:
                    qv = 3
                q_ptr[row_off + j] = Int8(qv)

    parallelize[process_row](rows)
